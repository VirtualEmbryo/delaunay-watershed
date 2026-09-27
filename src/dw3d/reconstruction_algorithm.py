"""Module defining MeshReconstructionAlgorithm, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ckdtree

from dw3d.io import save_rec, save_vtk
from dw3d.mesh_surgery import post_process_mesh_surgery
from dw3d.mesh_utilities import (
    WELD,
    center_around_origin,
    exclude_offsets_from_surface,
    filter_unused_points,
    labeled_mesh_from_labeled_graph,
    set_pixel_size,
    set_points_min_max,
)
from dw3d.segmentation import extract_seed_coords_and_indices
from dw3d.tesselation_graph import TesselationGraph
from dw3d.watershed import seeded_watershed_map

# segmentation mask -> EDT image
EdtCreationFunction = Callable[[NDArray[np.uint]], NDArray[np.float64]]
# optional segmentation mask + EDT image -> array of pixel coordinates.
# The return is `(points, indices_of_sorted_maxes)`, or `(points, indices_of_sorted_maxes,
# weights)` for a scheme that wants a weighted (regular) triangulation — see the
# junction-protected `peak_local_points_junction_protected` — or `(points,
# indices_of_sorted_maxes, weights, point_metadata)` for a scheme that declares which of its
# points are *tesselation scaffolding* rather than surface geometry (the offset-exclusion
# work). The 2-tuple form is unchanged and remains what every earlier point-placing
# function returns.
#
# `point_metadata` is a dict with `(n_points,)` arrays:
#   `family`               — the point-family code that placed each point (`dw3d.points_on_edt`);
#   `surface_merge_target` — the point the *extracted surface* should use in place of each
#                            point; identity except on the boundary-layer offsets, where it
#                            is the interface sample the offset displaces;
#   `guard_duplicate_faces`  — scalar, forwarded to `exclude_offsets_from_surface`.
# The switch lives here, in the point-placing scheme's own return value, rather than in
# `MeshReconstructionAlgorithm.__init__` — whose public signature is kept frozen — and
# that is also the right home for it: the scheme that emits scaffolding points is the thing
# that knows they are scaffolding.
PointPlacingFunction = Callable[
    [NDArray[np.uint] | None, NDArray[np.float64]],
    tuple[NDArray[np.uint], NDArray[np.uint]]
    | tuple[NDArray[np.uint], NDArray[np.uint], NDArray[np.float64]]
    | tuple[NDArray[np.uint], NDArray[np.uint], NDArray[np.float64] | None, dict],
]
# array of pixel coordinates (+ optional per-point weights) -> tesselation: array of points
# coordinates + array of tetrahedrons as points indices. The weight channel was added for
# the junction-protection work so junction samples can carry protecting-ball radii into a
# regular triangulation;
# `simple_delaunay_tesselation` ignores it, so the default output is unchanged. This widens
# the *callable's* contract, not `MeshReconstructionAlgorithm.__init__`'s signature (the
# frozen public one, whose parameter list is untouched).
TesselationCreationFunction = Callable[
    [NDArray[np.uint], NDArray[np.float64] | None],
    tuple[NDArray[np.float64], NDArray[np.int64]],
]
# EDT image, array of points + triangle faces (from tesselation) -> Array of scores
ScoreComputationFunction = Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]], NDArray[np.float64]]
# Watershed : one version only
# Mesh recreation : one version only
# Mesh surgery : one bool to perform surgery ?


class MeshReconstructionAlgorithm:
    """Algoithm to build a mesh from a segmented image."""

    def __init__(
        self,
        print_info: bool,
        edt_creation_function: EdtCreationFunction,
        point_placing_function: PointPlacingFunction,
        tesselation_creation_function: TesselationCreationFunction,
        score_computation_function: ScoreComputationFunction,
        perform_mesh_postprocess_surgery: bool,
        relocate_junctions: bool = True,
    ) -> None:
        """Set the methods used by this algorithm to reconstruct a mesh from a segmentation mask.

        Args:
            print_info (bool): Whether to show details about the computation during the algorithm's execution.
            edt_creation_function (EdtCreationFunction): Function used to compute an EDT from the segmentation mask.
            point_placing_function (PointPlacingFunction): Function used to place points on the EDT for a tesselation.
            tesselation_creation_function (TesselationCreationFunction): Function used to tesselate the points.
            score_computation_function (ScoreComputationFunction): Function used to compute the scores for Watershed.
            perform_mesh_postprocess_surgery (bool): Whether to try to detect and fix problems on the output mesh.
            relocate_junctions (bool): Whether to move the finished mesh's trijunction vertices onto
                the trijunction curves the mask itself gives, as a post-process. **Defaults to
                `True` since 0.5.0.** `False` reproduces the 0.4 meshes exactly. The post-process
                is `dw3d.relocate_junction_vertices`; it changes vertex positions only, never
                connectivity or labels, under a guard that cannot create a self-intersection.
                Cost, in seconds: about +0.75 s on the benchmark meshes (whose reconstruction takes
                about 5 s, +15 %), and 0.1-1.5 s per timepoint on real embryo volumes (median
                0.34 s and 0.48 s on two time series, whose reconstruction takes about 0.6 s);
                within one real series it grows faster than linearly with mesh size. See
                `dw3d.junction_relocation` for what it was measured to buy, `relocation_report` for
                what it did on the last mesh, and `relocation_provenance` for whether it ran and
                whether `CoarseSpacingRelocationWarning` fired. At a point-placement `min_distance`
                of 5 or more it emits that warning once per reconstruction.
        """
        self.print_info = print_info
        self.edt_creation_function = edt_creation_function
        self.point_placing_function = point_placing_function
        self.tesselation_creation_function = tesselation_creation_function
        self.score_computation_function = score_computation_function
        self.perform_mesh_postprocess_surgery = perform_mesh_postprocess_surgery
        self.relocate_junctions = relocate_junctions
        self._relocation_report = None
        self._coarse_spacing_warning_emitted = False
        self._relocation_ran = False

        self._first_computation_done = False

    def construct_mesh_from_segmentation_mask(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Build a 3D mesh from a segmentation mask.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
               - mesh points (geometry)
               - mesh triangles (topology)
               - labels (materials) on each side of the triangles. 0 is exterior.
        """
        self._segmented_image = segmented_image

        self._edt_image = self.edt_creation_function(self._segmented_image)

        placed = self.point_placing_function(self._segmented_image, self._edt_image)
        # A point-placing scheme may return a third element, the per-point weights for a
        # regular (weighted) triangulation (the junction-protection work), and a fourth, the
        # point metadata that keeps scaffolding points out of the extracted surface (the
        # offset-exclusion work). Every earlier scheme returns a 2-tuple and gets
        # `weights = None` and no metadata, i.e. exactly the previous behaviour.
        self._point_weights = None
        self._point_metadata = None
        if len(placed) == 4:
            points_for_tesselation, indices_of_sorted_maxes, self._point_weights, self._point_metadata = placed
        elif len(placed) == 3:
            points_for_tesselation, indices_of_sorted_maxes, self._point_weights = placed
        else:
            points_for_tesselation, indices_of_sorted_maxes = placed

        t_init_tesselation = perf_counter()
        tesselation_points, tesselation_tetrahedrons = self.tesselation_creation_function(
            points_for_tesselation,
            self._point_weights,
        )
        if self.print_info:
            print(f"Delaunay Tesselation built in {perf_counter() - t_init_tesselation:.2} seconds")

        self._tesselation_graph = TesselationGraph(
            tesselation_points,
            tesselation_tetrahedrons,
            indices_of_sorted_maxes,
            self.score_computation_function,
            self._edt_image,
            print_info=self.print_info,
        )

        # Here we use again the segmented image to get seeds, but another input to create a mesh could be
        # the EDT directly + those seeds obtained with another method ?
        # (this is now the segmented image's last use: the "zero nodes" lookup that followed the watershed
        #  was dead code and was removed -- see seeded_watershed_map's docstring)
        self._seeds_coords, self._seeds_indices = extract_seed_coords_and_indices(
            self._segmented_image,
            self._edt_image,
        )
        self._watershed_seeded()

        self._points, self._triangles, self._labels = labeled_mesh_from_labeled_graph(
            self._tesselation_graph,
            self._map_label_to_nodes_ids,
        )

        self._mesh_surgery()
        self._relocate_junctions()

        self._first_computation_done = True
        return self.last_constructed_mesh

    def _relocate_junctions(self) -> None:
        """Move the finished mesh's trijunction vertices onto the mask's own trijunction curves.

        On by default (`relocate_junctions=True`). The post-process runs last, on the surgeried
        mesh, because it takes the mesh's connectivity as given and freezes it -- a later surgery
        pass would invalidate the guarantee that no self-intersection was created. It reads the
        same segmentation mask the reconstruction read and no other input.

        When it runs and the point placement's `min_distance` is at least
        `dw3d.junction_relocation.COARSE_SPACING_MIN_DISTANCE` (5), it emits
        `CoarseSpacingRelocationWarning` exactly once for this reconstruction. A placement function
        that carries no `min_distance` never triggers the warning (see `placement_min_distance`).
        """
        self._coarse_spacing_warning_emitted = False
        self._relocation_ran = False
        if not self.relocate_junctions:
            self._relocation_report = None
            return
        from dw3d.junction_relocation import (
            COARSE_SPACING_MIN_DISTANCE,
            COARSE_SPACING_RELOCATION_MESSAGE,
            CoarseSpacingRelocationWarning,
            relocate_junction_vertices,
        )

        min_distance = self.placement_min_distance
        if min_distance is not None and min_distance >= COARSE_SPACING_MIN_DISTANCE:
            warnings.warn(COARSE_SPACING_RELOCATION_MESSAGE, CoarseSpacingRelocationWarning, stacklevel=3)
            self._coarse_spacing_warning_emitted = True

        t_init = perf_counter()
        self._points, self._relocation_report = relocate_junction_vertices(
            self._segmented_image,
            self._points,
            self._triangles,
            self._labels,
        )
        if self.print_info:
            report = self._relocation_report
            print(
                f"Junction vertices relocated in {perf_counter() - t_init:.2} seconds: "
                f"{report.n_accepted} of {report.n_targets} moves accepted, "
                f"{report.n_rejected_self_intersection} refused by the self-intersection guard",
            )
        self._relocation_ran = True

    @property
    def placement_min_distance(self) -> int | None:
        """The point placement's `min_distance`, or `None` when it cannot be read.

        The factory's placements are `functools.partial` objects carrying `min_distance` as a
        keyword; a custom placement callable carries none, and then this is `None` and
        `CoarseSpacingRelocationWarning` is never emitted.

        Returns:
            int | None: the minimum distance between placed points, in voxels.
        """
        value = getattr(self.point_placing_function, "keywords", {}).get("min_distance")
        return None if value is None else int(value)

    @property
    def relocation_provenance(self) -> dict:
        """Whether junction relocation ran on the last constructed mesh, with which guard, and what it warned.

        Returns:
            dict: `relocate_junctions` (the setting), `ran` (whether the post-process ran on the
            last mesh), `guard` (the acceptance test it used), `min_distance` (the placement's, or
            `None`), `coarse_spacing_warning_emitted` (whether `CoarseSpacingRelocationWarning`
            fired for the last mesh) and `report` (`JunctionRelocationReport.as_dict()`, or `None`
            when it did not run). JSON-serialisable.
        """
        return {
            "relocate_junctions": bool(self.relocate_junctions),
            "ran": bool(self._relocation_ran),
            "guard": (
                "local orientation/area guard plus exact nonlocal self-intersection guard, "
                "whole-mesh certificate before and after (dw3d.relocate_junction_vertices, verify=True)"
            ),
            "min_distance": self.placement_min_distance,
            "coarse_spacing_warning_emitted": bool(self._coarse_spacing_warning_emitted),
            "report": None if self._relocation_report is None else self._relocation_report.as_dict(),
        }

    @property
    def relocation_report(self):  # noqa: ANN201 -- JunctionRelocationReport | None, imported lazily
        """What the junction relocation did on the last constructed mesh, or `None` if it was off.

        Returns:
            dw3d.junction_relocation.JunctionRelocationReport | None: the accepted and refused move
            counts, the displacement distribution, and the whole-mesh self-intersection count before
            and after.
        """
        return self._relocation_report

    def _watershed_seeded(self) -> None:
        """Perform watershed algorithm to label tetrahedrons of the tesselation networkX graph."""
        t1 = perf_counter()
        seeds_nodes = _compute_seeds_idx_from_voxel_coords(
            self._tesselation_graph.compute_nodes_centroids(),
            self._seeds_coords,
        )

        nx_graph = self._tesselation_graph.to_networkx_graph()
        self._map_label_to_nodes_ids, self._map_node_id_to_label = seeded_watershed_map(
            nx_graph,
            seeds_nodes,
            self._seeds_indices,
        )

        if self.print_info:
            print(f"Watershed done in {perf_counter() - t1:.3} seconds.")

    def _mesh_surgery(self) -> None:
        """Try to detect and fix mesh problems while we have all the Watershed data."""
        # return
        # Optional part
        if self.perform_mesh_postprocess_surgery:
            post_process_mesh_surgery(self)

        self._exclude_scaffolding_from_surface()

        # Always do this part
        # filter unused points
        # Recorded before the filter renumbers them: `_surface_point_ids[i]` is the
        # *tesselation* index of mesh vertex `i`. The offset-exclusion work needs it to
        # attribute a per-mesh-vertex measurement (its distance to the true surface) to the
        # point family that placed the vertex, which was previously done ad hoc outside the
        # library.
        self._surface_point_ids = np.unique(self._triangles)
        self._points, self._triangles = filter_unused_points(self._points, self._triangles)

    def _exclude_scaffolding_from_surface(self) -> None:
        """Keep the point-placing scheme's scaffolding points out of the surface.

        A no-op unless the point-placing function supplied `surface_merge_target` in its
        `point_metadata`, so every configuration without offset exclusion is bit-identical.

        Placed **after** mesh surgery and **before** `filter_unused_points`, deliberately:

        * surgery re-extracts the mesh from the tesselation graph on every iteration that
          changes a label, so anything done before it would be thrown away — and surgery
          reads only connectivity and labels (`_find_abnormal_non_manifold_edges`), never
          coordinates, so it is unaffected by running second;
        * the filter is what drops the tesselation points the surface does not reference, and
          the merge is precisely a statement about which points those are.
        """
        metadata = getattr(self, "_point_metadata", None)
        self._surface_exclusion_info = None
        if not metadata or "surface_merge_target" not in metadata:
            return
        self._triangles, self._labels, self._surface_exclusion_info = exclude_offsets_from_surface(
            self._points,
            self._triangles,
            self._labels,
            metadata["surface_merge_target"],
            guard_duplicate_faces=metadata.get("guard_duplicate_faces", True),
            collapse_rule=metadata.get("collapse_rule", WELD),
        )
        if self.print_info:
            info = self._surface_exclusion_info
            print(
                f"Offsets excluded from the surface: {info['n_offset_vertices_removed']} vertices, "
                f"{info['n_triangles_before']} -> {info['n_triangles_after']} triangles",
            )

    @property
    def last_constructed_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Get points, triangles and labels (materials) describing the mesh obtained from segmented image."""
        return self._points, self._triangles, self._labels

    def compress_segmentation_mask(self, segmented_image: NDArray[np.uint]) -> dict[str]:
        """Compress a segmentation mask using a 3D mesh reconstruction.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            dict[str]:
               - dictionary of the compressed segmentation that can be reconstructed with this package.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return self.last_compressed_segmentation

    @property
    def last_compressed_segmentation(self) -> dict[str]:
        """Export mesh, seeds coordinates and image shape in a dictionary that can be saved with numpy.save()."""
        return {
            "points": self._points,
            "triangles": self._triangles,
            "seeds": self._seeds_coords,
            "image_shape": self._segmented_image.shape,
        }

    def both_construct_mesh_and_compressed_segmentation(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Construct and return a 3D mesh from segmentation and obtain the dictionary of the compressed segmentation.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
               - mesh points (geometry)
               - mesh triangles (topology)
               - labels (materials) on each side of the triangles. 0 is exterior.
               - dictionary of the compressed segmentation that can be reconstructed with this package.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return *self.last_constructed_mesh, self.last_compressed_segmentation

    def center_around_zero(self) -> None:
        """Center the mesh around 0."""
        self._points = center_around_origin(self._points)

    def set_pixel_size(self, xy_pixel_size: float, z_pixel_size: float) -> None:
        """Scales the mesh from pixel coordinates to real coordinates, given microscope's xy and z pixel size."""
        self._points = set_pixel_size(self._points, xy_pixel_size, z_pixel_size)

    def set_global_min_max(self, global_min: float, global_max: float) -> None:
        """Scales homogeneously the mesh such that its min & max values are global_min and global_max."""
        self._points = set_points_min_max(self._points, global_min, global_max)

    def save_to_rec_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk in the rec format."""
        save_rec(filename, self._points, self._triangles, self._labels, binary_mode)

    def save_to_vtk_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk in the vtk format."""
        save_vtk(filename, self._points, self._triangles, self._labels, binary_mode)


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str]) -> None:
    """Save a compressed segmentation on disk with numpy.save."""
    np.save(filename, compressed_segmentation)


def _compute_seeds_idx_from_voxel_coords(
    centroids: NDArray[np.float64],
    seed_pixel_coords: NDArray[np.uint],
) -> NDArray[np.uint]:
    """Find, for each seed voxel, the index of the nearest tetrahedron centroid.

    Used to express the watershed's seeds as tesselation nodes.

    This used to flatten `seed_pixel_coords` to linear indices and immediately unflatten
    them again with `_linear_to_3d_index`, which is the exact inverse of that flattening,
    so the round trip returned its own input (verified
    bit-identical, same dtype, on the seeds of all 4 in-repo images). It was a leftover
    from indexing into a materialised array of every voxel coordinate — see the deleted
    `_pixels_coords` helper in the git history of this file. Removing it also removed the
    function's `edt` parameter, which existed only to supply that array's shape.
    """
    tree = ckdtree.cKDTree(centroids)
    _, idx_seeds = tree.query(seed_pixel_coords)
    return idx_seeds  # "seed" nodes ids
