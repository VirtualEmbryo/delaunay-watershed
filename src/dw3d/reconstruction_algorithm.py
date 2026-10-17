"""The mesh reconstruction algorithm: from a 3-D segmentation mask to a multimaterial surface mesh.

`MeshReconstructionAlgorithm` runs the pipeline (distance transform, point placement, tesselation,
seeded watershed, mesh extraction and surgery, crop-face marking, junction relocation, cell-volume
check) and keeps the last mesh it built. Get one from `dw3d.MeshReconstructionAlgorithmFactory` or
`dw3d.get_default_mesh_reconstruction_algorithm`.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from dw3d.crop_faces import CropFaceReport, classify_crop_faces, normalise_physical_image_faces
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
# points are *tesselation scaffolding* rather than surface geometry (the offset exclusion).
# The 2-tuple form is unchanged and remains what every earlier point-placing
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
# junction protection so junction samples can carry protecting-ball radii into a
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
    """Algorithm that builds a multimaterial surface mesh from a 3-D segmentation mask.

    Build one with `dw3d.MeshReconstructionAlgorithmFactory` (or one of its `get_*_algorithm`
    methods) rather than by calling the constructor. The same object can reconstruct several
    masks; each reconstruction replaces the last mesh and every report about it.

    The mesh is `(points, triangles, labels)`: `points` `(n, 3)` float64, in the mask's index order
    (`(z, y, x)` for a `(z, y, x)` stack) and in voxel units unless built by
    `construct_mesh_in_physical_units`; `triangles` `(m, 3)` vertex indices; `labels` `(m, 2)`, the
    two regions each triangle separates, smaller label first, label 0 being the medium.

    Attributes:
        print_info (bool): whether to print the timing of each step.
        relocate_junctions (bool): whether the next reconstruction runs junction relocation.
        physical_image_faces (tuple[str, ...]): the image faces declared physical; see `__init__`.
    """

    def __init__(
        self,
        print_info: bool,
        edt_creation_function: EdtCreationFunction,
        point_placing_function: PointPlacingFunction,
        tesselation_creation_function: TesselationCreationFunction,
        score_computation_function: ScoreComputationFunction,
        perform_mesh_postprocess_surgery: bool,
        relocate_junctions: bool = True,
        physical_image_faces: tuple[str, ...] = (),
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
            physical_image_faces (tuple[str, ...]): the faces of the image box, among
                `dw3d.crop_faces.IMAGE_FACES` (`"axis0_low"`, `"axis0_high"`, ... `"axis2_high"`),
                where the tissue really ends, for example on a coverslip. A cell surface lying on
                such a face is contact with the medium. **Defaults to none**: every image face is
                a crop, the triangles a cell has on it are *crop faces* (`crop_faces`) where the
                mask's cell reaches that face, and the cells they touch are *cut cells*
                (`cut_cells`). Can also be set as an attribute before a reconstruction. Changes no
                point, triangle or label of the mesh; see `dw3d.crop_faces`.
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
        self._cell_volume_report = None
        self.physical_image_faces = normalise_physical_image_faces(physical_image_faces)
        self._crop_face_report: CropFaceReport | None = None
        # Set by `construct_mesh_in_physical_units`, cleared by every reconstruction: the `.rec` appendix fields
        # saying the mesh is in physical units and which voxels it came from.
        self._physical_frame: dict | None = None

        self._first_computation_done = False

    def construct_mesh_from_segmentation_mask(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Build a 3D mesh from a segmentation mask.

        The mask is meshed on its own voxel grid, whatever its voxel size: for a mask whose voxels
        are longer along one axis, use `construct_mesh_in_physical_units`. The point placement's
        `min_distance` counts voxels of this grid.

        After the call, `last_constructed_mesh`, `crop_faces`, `cut_cells`, `crop_face_report`,
        `cell_volume_report`, `relocation_report` and `relocation_provenance` describe this mesh,
        and `save_to_rec_mesh` writes no physical frame.

        Args:
            segmented_image (NDArray[np.uint]): the 3-D integer label mask; label 0 is the medium,
                every other label a cell.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]: `(points, triangles,
                labels)`. `points` is `(n, 3)` float64 in voxel units of the mask, in its index
                order: column `i` is the position along the mask's axis `i`, so `(z, y, x)` for a
                `(z, y, x)` stack, and the centre of voxel `(i, j, k)` is the point `(i, j, k)`.
                `triangles` is `(m, 3)` vertex indices. `labels` is `(m, 2)`: the two regions each
                triangle separates, smaller label first; the triangle's normal points out of the
                first into the second. These are the algorithm's own arrays, not copies.

        Warns:
            CoarseSpacingRelocationWarning: once, when junction relocation is on and the point
                placement's `min_distance` is 5 or more.
            CellVolumeDisagreementWarning: once, when some cells' mesh volume disagrees with their
                voxel count (see `cell_volume_report`).
        """
        self._segmented_image = segmented_image
        self._physical_frame = None

        self._edt_image = self.edt_creation_function(self._segmented_image)

        placed = self.point_placing_function(self._segmented_image, self._edt_image)
        # A point-placing scheme may return a third element, the per-point weights for a
        # regular (weighted) triangulation (junction protection), and a fourth, the
        # point metadata that keeps scaffolding points out of the extracted surface (the
        # offset exclusion). Every earlier scheme returns a 2-tuple and gets
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
        self._classify_crop_faces()
        self._relocate_junctions()
        self._compare_cell_volumes_with_mask()

        self._first_computation_done = True
        return self.last_constructed_mesh

    def construct_mesh_in_physical_units(
        self,
        segmented_image: NDArray[np.uint],
        spacing: tuple[float, float, float],
        *,
        resampling: str = "shape",
        spacing_unit: str | None = None,
        axis_order: str = "zyx",
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Build a mesh, in physical units, from a mask whose voxels may be longer along one axis.

        The mask is first resampled to cubic voxels of the shortest spacing by
        `dw3d.mask_preprocessing.resample_to_isotropic` (shape-based interpolation by default), then meshed by
        `construct_mesh_from_segmentation_mask` with this algorithm's settings unchanged (`min_distance` counts voxels
        of the resampled cubic grid), and the mesh is scaled to physical units: a point at voxel index `i` of the
        cubic grid is at `i * h`, `h = min(spacing)`, so the centre of the mask's first voxel is the origin. The
        columns stay in the mask's index order, `(z, y, x)` for a `(z, y, x)` stack.

        Why resample rather than mesh the native grid and scale it: measured on 12 simulated tissues of 90 and 165
        cells at a real cell size, down-sampled along z, against the same tissues meshed at full resolution,
        meshing the native grid costs +1.38, +1.97 and +4.10 degrees of contact-angle error at `k = 5/3`, 2 and 3
        (`k` = long spacing / short spacing), and resampling first +0.83, +0.71 and +1.17; see
        `dw3d.mask_preprocessing`. The cost is that
        of a mask acquired at the short spacing: `k` times the voxels.

        The reports (`crop_faces`, `cut_cells`, `crop_face_report`, `cell_volume_report`, `relocation_report`)
        describe the mesh on the resampled cubic grid, in its voxel units, before the scaling.

        After this call `save_to_rec_mesh` writes, in the file's metadata appendix, `coordinate_frame = physical`,
        `axis_order`, `spacing` (the mask's own voxel size, which the mesh coordinates are already scaled by),
        `spacing_unit` when given, `source_shape`, `resampling` and `resampled_spacing`. Any later reconstruction
        clears them. Do not call `set_pixel_size` on the result: it is already in physical units.

        `last_compressed_segmentation` then holds the physical points with the seeds and the shape of the resampled
        cubic grid: divide its points by `resampled_spacing` before `dw3d.reconstruct_mask_from_dict` (see that
        property).

        Args:
            segmented_image (NDArray[np.uint]): the 3-D integer label mask; label 0 is the medium.
            spacing (tuple[float, float, float]): voxel size along the mask's axes 0, 1, 2, so `(dz, dy, dx)` for a
                `(z, y, x)` stack, in any one unit. One axis may be coarser than the other two, which must be equal
                (to a relative tolerance of 1e-6).
            resampling (str, optional): `"shape"` (default) or `"nearest"`; see `dw3d.mask_preprocessing`.
            spacing_unit (str | None, optional): the unit of `spacing` (e.g. `"um"`), recorded, not used.
            axis_order (str, optional): what the mask's axes are, recorded, not used. Defaults to `"zyx"`.

        Returns:
            tuple: `(points, triangles, labels)` as `construct_mesh_from_segmentation_mask` returns them, the points in
                the unit of `spacing`. For cubic voxels of size 1 they are bitwise those of
                `construct_mesh_from_segmentation_mask(segmented_image)`.

        Raises:
            ValueError: for a spacing that is not three positive finite numbers or has two different fine axes, an
                unknown `resampling` method, or a mask that is not 3-D.

        Warns:
            UnmeasuredAnisotropyWarning: when the voxels are more than `MEASURED_ANISOTROPY_RATIO_MAX` (3) times longer
                along one axis, a range in which the accuracy was not measured. The mesh is built anyway.
            CoarseSpacingRelocationWarning: as in `construct_mesh_from_segmentation_mask`.
            CellVolumeDisagreementWarning: as in `construct_mesh_from_segmentation_mask`.
        """
        from dw3d.mask_preprocessing import (
            MEASURED_ANISOTROPY_RATIO_MAX,
            SPACING_RELATIVE_TOLERANCE,
            UnmeasuredAnisotropyWarning,
            coarse_axis_and_factor,
            isotropic_spacing,
            resample_to_isotropic,
        )

        found = coarse_axis_and_factor(spacing)
        if found is not None and found[1] > MEASURED_ANISOTROPY_RATIO_MAX * (1 + SPACING_RELATIVE_TOLERANCE):
            warnings.warn(
                f"voxels {found[1]:.3g} times longer along axis {found[0]} than along the other two: the accuracy of "
                f"meshing after resampling was measured up to a ratio of {MEASURED_ANISOTROPY_RATIO_MAX:g} only "
                "(on larger cells the contact-angle error kept growing above it). The mesh is built anyway.",
                UnmeasuredAnisotropyWarning,
                stacklevel=2,
            )
        cubic = resample_to_isotropic(segmented_image, spacing, method=resampling)
        h = isotropic_spacing(spacing)
        self.construct_mesh_from_segmentation_mask(cubic)
        self._points = self._points * h
        frame = {
            "coordinate_frame": "physical",
            "axis_order": axis_order,
            "spacing": tuple(float(v) for v in spacing),
            "source_shape": tuple(int(n) for n in np.shape(segmented_image)),
            "resampling": resampling,
            "resampled_spacing": h,
        }
        if spacing_unit is not None:
            frame["spacing_unit"] = spacing_unit
        self._physical_frame = frame
        return self.last_constructed_mesh

    def _classify_crop_faces(self) -> None:
        """Mark the border triangles that are crop faces, and the cells they cut.

        Runs after surgery, when the triangles are final, and **before** junction relocation,
        because the image face a border triangle lies on is read from its vertices' voxel
        coordinates, which relocation may move. Changes nothing in the mesh. See
        `dw3d.crop_faces`.
        """
        self.physical_image_faces = normalise_physical_image_faces(self.physical_image_faces)
        self._crop_face_report = classify_crop_faces(
            self._points,
            self._triangles,
            self._labels,
            self._surface_point_ids,
            self._tesselation_graph.lone_faces,
            self._segmented_image,
            self.physical_image_faces,
        )
        if self.print_info and self._crop_face_report.n_border_triangles:
            report = self._crop_face_report
            print(
                f"Crop faces: {report.n_crop_faces} of {report.n_border_triangles} border triangles; "
                f"{len(report.cut_cells)} cut cells",
            )

    @property
    def crop_faces(self) -> NDArray[np.bool_] | None:
        """Which triangles of the last constructed mesh are crop faces.

        A crop face is a triangle on a face of the image box that was not declared physical
        (`physical_image_faces`) and that the mask's cell reaches: the cell it closes goes on
        beyond the image, so the triangle is not contact with the medium, and no tension or angle
        should be read on it. A triangle the reconstruction put on an image face the mask's cell
        does not reach stays contact with the medium. foambryo
        reads this mask (`DcelData(..., crop_faces=...)`) and keeps crop faces, and every
        junction edge and vertex that touches them, out of its equations.

        Returns:
            NDArray[np.bool_] | None: `(n_triangles,)`, aligned with the triangles
            `construct_mesh_from_segmentation_mask` returned; all `False` when no cell of the
            mask reaches a crop image face. `None` before the first reconstruction.
        """
        return None if self._crop_face_report is None else self._crop_face_report.crop_faces

    @property
    def cut_cells(self) -> tuple[int, ...] | None:
        """The cells of the last constructed mesh that the crop cuts.

        A cell is cut when it has a crop face or a vertex on one. Its volume and part of its
        surface were not observed, so its pressure is not a measurement.

        Returns:
            tuple[int, ...] | None: sorted labels; empty when nothing is cut; `None` before the
            first reconstruction.
        """
        return None if self._crop_face_report is None else self._crop_face_report.cut_cells

    @property
    def crop_face_report(self) -> CropFaceReport | None:
        """The crop faces, the image face of each border triangle and the cut cells; `None` before a reconstruction."""
        return self._crop_face_report

    def _rec_crop_metadata(self) -> dict | None:
        """The `.rec` appendix fields for the crop, or `None` when the mesh meets no image face its cells reach."""
        report = self._crop_face_report
        if report is None or not report.meets_the_image_box:
            return None
        return {"physical_image_faces": " ".join(report.physical_image_faces) or "none"}

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

    def _compare_cell_volumes_with_mask(self) -> None:
        """List the cells whose final mesh volume disagrees with their voxel count, and warn once.

        Runs last, on the finished mesh in mask voxel units, and changes nothing in it. The list is
        `cell_volume_report`; when it is not empty, `CellVolumeDisagreementWarning` is emitted
        exactly once for this reconstruction. See `dw3d.cell_volume_agreement` for the rule and for
        what the listed cells were measured to be.
        """
        from dw3d.cell_volume_agreement import (
            CellVolumeDisagreementWarning,
            compare_cell_volumes_with_mask,
            disagreement_message,
        )

        self._cell_volume_report = compare_cell_volumes_with_mask(
            self._segmented_image,
            self._points,
            self._triangles,
            self._labels,
        )
        if self._cell_volume_report.cells:
            warnings.warn(disagreement_message(self._cell_volume_report), CellVolumeDisagreementWarning, stacklevel=3)

    @property
    def cell_volume_report(self):  # noqa: ANN201 -- CellVolumeReport | None, imported lazily
        """The cells of the last constructed mesh whose volume disagrees with their voxel count.

        A cell is listed when its mesh volume over its voxel count is outside `[0.5, 1.5]`, when its
        mesh volume is not positive, or when no triangle carries it. Computed on the mesh as
        `construct_mesh_from_segmentation_mask` returned it, in voxel units of the grid it was meshed
        on, before any later pixel-size scaling.

        Returns:
            dw3d.cell_volume_agreement.CellVolumeReport | None: the listed cells with their voxel
                count, mesh volume, ratio and reason, or `None` before the first reconstruction.
        """
        return self._cell_volume_report

    @property
    def placement_min_distance(self) -> int | None:
        """The point placement's `min_distance`, or `None` when it cannot be read.

        The factory's placements are `functools.partial` objects carrying `min_distance` as a
        keyword; a custom placement callable carries none, and then this is `None` and
        `CoarseSpacingRelocationWarning` is never emitted.

        Returns:
            int | None: the minimum distance between placed points, in voxels of the grid the mask is
                meshed on.
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
                "local orientation/area guard, a clause that turns no cell inside out, plus exact nonlocal "
                "self-intersection guard, "
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
                counts, the displacement distribution, and the whole-mesh self-intersection count
                before and after.
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
        # *tesselation* index of mesh vertex `i`. The offset-exclusion measurements need it to
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
        """The last mesh built, `(points, triangles, labels)`, after any later scaling or centring.

        These are the algorithm's own arrays, not copies: modifying them modifies the mesh that
        `save_to_rec_mesh` and `save_to_vtk_mesh` write.
        """
        return self._points, self._triangles, self._labels

    def compress_segmentation_mask(self, segmented_image: NDArray[np.uint]) -> dict[str]:
        """Compress a segmentation mask using a 3D mesh reconstruction.

        Args:
            segmented_image (NDArray[np.uint]): the 3-D integer label mask; label 0 is the medium.

        Returns:
            dict[str]: the compressed segmentation (see `last_compressed_segmentation`), which
                `dw3d.reconstruct_mask_from_dict` turns back into an approximate mask.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return self.last_compressed_segmentation

    @property
    def last_compressed_segmentation(self) -> dict[str]:
        """The last mesh's points and triangles, the watershed seeds and the mask's shape, as a dictionary.

        Keys: `points`, `triangles`, `seeds` (one voxel index per label of the mask, label 0 included when present,
        in ascending label order) and `image_shape`. Save it with `dw3d.save_compressed_segmentation`
        (`numpy.save`). `dw3d.reconstruct_mask_from_dict` expects the points in voxel units of the
        mask, so read this before any scaling (`set_pixel_size`, `set_global_min_max`).

        The frame of each entry:

        * `points` are the mesh's current points, `last_constructed_mesh[0]` itself: voxel indices of the meshed
          grid after `construct_mesh_from_segmentation_mask`, but in physical units (the unit of `spacing`) after
          `construct_mesh_in_physical_units`, and as scaled by any later `set_pixel_size`, `set_global_min_max` or
          `center_around_zero`.
        * `seeds` and `image_shape` are always voxel indices and the shape of the grid that was meshed. After
          `construct_mesh_in_physical_units` that is the mask resampled to cubic voxels of `resampled_spacing`
          (the shortest spacing), not the mask that was passed in.

        So after `construct_mesh_in_physical_units` the points are not in the frame of the seeds: divide them by
        `resampled_spacing` before `dw3d.reconstruct_mask_from_dict`, which then rebuilds the resampled cubic-grid
        mask. Passed as they are, the mesh is rasterised at the wrong scale (measured on a two-cell mask with
        spacing `(0.6, 0.3, 0.3)`: 7 % of voxels agree with the resampled mask, against 96 % after the division).
        """
        return {
            "points": self._points,
            "triangles": self._triangles,
            "seeds": self._seeds_coords,
            "image_shape": self._segmented_image.shape,
        }

    def both_construct_mesh_and_compressed_segmentation(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint], dict[str, Any]]:
        """Construct and return a 3D mesh from segmentation and obtain the dictionary of the compressed segmentation.

        Args:
            segmented_image (NDArray[np.uint]): the 3-D integer label mask; label 0 is the medium.

        Returns:
            tuple: four items, `(points, triangles, labels, compressed_segmentation)`: the mesh as
                `construct_mesh_from_segmentation_mask` returns it, then the dictionary
                `last_compressed_segmentation` returns.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return *self.last_constructed_mesh, self.last_compressed_segmentation

    def center_around_zero(self) -> None:
        """Translate the mesh, in place, so that the centre of its bounding box is the origin."""
        self._points = center_around_origin(self._points)

    def set_pixel_size(self, xy_pixel_size: float, z_pixel_size: float) -> None:
        """Scale the mesh, in place, from voxel coordinates to physical coordinates, given the xy and z pixel sizes.

        The mesh points are in the mask's `(z, y, x)` index order: column 0 is scaled by `z_pixel_size`, columns 1
        and 2 by `xy_pixel_size` (see `dw3d.mesh_utilities.set_pixel_size`). Changed in 0.7.0: before, z was
        multiplied by the xy size and x by the z size. For a mask whose voxels are longer along z, prefer
        `construct_mesh_in_physical_units`, which is more accurate; do not call this method on its result, which is
        already in physical units. The reports (`cell_volume_report`, `crop_face_report`, ...) are not rescaled.

        Args:
            xy_pixel_size (float): the pixel size in the image plane, which multiplies columns 1 and 2.
            z_pixel_size (float): the z step, which multiplies column 0.
        """
        self._points = set_pixel_size(self._points, xy_pixel_size, z_pixel_size)

    def set_global_min_max(self, global_min: float, global_max: float) -> None:
        """Scale and shift the mesh in place, one factor on all axes, so it spans `[global_min, global_max]`.

        The minimum and maximum are taken over all coordinates together, not per axis, so the shape is
        kept.

        Args:
            global_min (float): the new smallest coordinate.
            global_max (float): the new largest coordinate.
        """
        self._points = set_points_min_max(self._points, global_min, global_max)

    def save_to_rec_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the last mesh on disk in the rec format, with its current points.

        A mesh made by `construct_mesh_in_physical_units` gets its physical frame recorded in the file's metadata
        appendix (see that method); a mesh made by `construct_mesh_from_segmentation_mask` gets none.

        When the mesh lies on an image face that one of its cells reaches in the mask, the crop
        faces are written as the **last block of triangles** and the file's metadata appendix
        records `crop_faces = <first row> <count>` and `physical_image_faces`, so the triangle rows
        of the file differ in order from `last_constructed_mesh`. Read them back with
        `dw3d.io.load_rec_with_crop_faces`. Otherwise (in particular for every mask none of whose
        cells reaches the image box) the file is byte for byte what it was before crop faces were
        recorded.

        Args:
            filename (str | Path): where to write the file (`.rec`).
            binary_mode (bool, optional): write the binary variant of the format. Defaults to False.
        """
        report = self._crop_face_report
        crop = report is not None and report.meets_the_image_box
        if not crop and self._physical_frame is None:
            save_rec(filename, self._points, self._triangles, self._labels, binary_mode)
            return
        metadata = {**(self._physical_frame or {}), **(self._rec_crop_metadata() or {})}
        save_rec(
            filename,
            self._points,
            self._triangles,
            self._labels,
            binary_mode,
            metadata=metadata,
            crop_faces=report.crop_faces if crop else None,
        )

    def save_to_vtk_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the last mesh on disk in the vtk format, with its current points.

        When the mesh lies on an image face that one of its cells reaches in the mask, a
        `crop_face` cell array (1 on crop faces) is written beside `label1` and `label2`; otherwise
        the file is what it was before. No physical frame is recorded.

        Args:
            filename (str | Path): where to write the file; meshio picks the format from its extension.
            binary_mode (bool, optional): write a binary file. Defaults to False.
        """
        report = self._crop_face_report
        crop = None if report is None or not report.meets_the_image_box else report.crop_faces
        save_vtk(filename, self._points, self._triangles, self._labels, binary_mode, crop_faces=crop)


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str]) -> None:
    """Save a compressed segmentation on disk with `numpy.save` (same as `dw3d.save_compressed_segmentation`).

    Args:
        filename (str | Path): where to write the file; numpy adds `.npy` when it has no extension.
        compressed_segmentation (dict[str]): the dictionary `last_compressed_segmentation` returns.
    """
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
    tree = cKDTree(centroids)
    _, idx_seeds = tree.query(seed_pixel_coords)
    return idx_seeds  # "seed" nodes ids
