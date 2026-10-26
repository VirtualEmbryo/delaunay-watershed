"""Build mesh reconstruction algorithms: the named configurations and a factory for custom ones.

`MeshReconstructionAlgorithmFactory`'s static `get_*_algorithm` methods return the named
configurations (`get_default_algorithm` is the one to use for tension inference); an instance of
the factory combines the steps one by one and `make_algorithm` builds the result.

Matthieu Perez 2024
"""

from functools import partial
from typing import Self

import numpy as np

from dw3d.crop_faces import normalise_physical_image_faces
from dw3d.edt import compute_edt_boundary_bias, compute_edt_classical, compute_edt_float32, compute_edt_tiled
from dw3d.mesh_utilities import LINK_CONDITION, WELD
from dw3d.points_on_edt import (
    peak_local_points,
    peak_local_points_bias_boundaries,
    peak_local_points_boundary_layer,
    peak_local_points_dithered,
    peak_local_points_junction_protected,
)
from dw3d.reconstruction_algorithm import (
    EdtCreationFunction,
    MeshReconstructionAlgorithm,
    PointPlacingFunction,
    ScoreComputationFunction,
    TesselationCreationFunction,
)
from dw3d.score_computation import (
    compute_scores_by_max_value,
    compute_scores_by_max_value_cubic,
    compute_scores_by_mean_value,
    compute_scores_by_mean_value_cubic,
)
from dw3d.tesselation import simple_delaunay_tesselation, weighted_delaunay_tesselation


class MeshReconstructionAlgorithmFactory:
    """Factory to build custom algorithms for mesh reconstruction from segmentation masks.

    Each `set_*` method chooses one step and returns the factory, so calls can be chained;
    `make_algorithm` builds the algorithm.
    """

    def __init__(self, print_info: bool = False, perform_mesh_postprocess_surgery: bool = True) -> None:
        """Initialize the factory with the steps of the deterministic configuration.

        A new factory's `make_algorithm` builds what `get_deterministic_algorithm` returns: the
        classical EDT, deterministic EDT extrema with `min_distance=3`, plain Delaunay
        tesselation, max-value face scores, mesh surgery, junction relocation on, the canonical
        tetrahedron order and no physical image face. This is **not** `get_default_algorithm`,
        which places its points with the dithered rule (`set_dithered_peak_local_points_placement_method`).

        Args:
            print_info (bool, optional): print the timing of each step. Defaults to False.
            perform_mesh_postprocess_surgery (bool, optional): try to detect and fix problems on the
                output mesh. Defaults to True.
        """
        self.print_info = print_info
        self._edt_creation_function: EdtCreationFunction = _edt_creation_function_classical(print_info=print_info)
        self._point_placing_function: PointPlacingFunction = _point_placing_function_peak_local(print_info=print_info)
        self._tesselation_creation_function: TesselationCreationFunction = simple_delaunay_tesselation
        self._score_computation_function: ScoreComputationFunction = compute_scores_by_max_value
        self.perform_mesh_postprocess_surgery = perform_mesh_postprocess_surgery
        self.relocate_junctions = True
        self.canonical_tetrahedron_order = True
        self.physical_image_faces: tuple[str, ...] = ()

    def set_physical_image_faces(self, *faces: str) -> Self:
        """Declare faces of the image box where the tissue really ends, for example on a coverslip.

        By default every face of the image box is a crop: a cell surface lying on it is a crop
        face, not contact with the medium, and the cell is a cut cell (see `dw3d.crop_faces` and
        the algorithm's `crop_faces` and `cut_cells`). A face declared here is physical: the
        cell surfaces on it are contact with the medium, as they were before crop faces were
        recorded. The mesh itself is the same either way.

        Args:
            *faces (str): names among `dw3d.crop_faces.IMAGE_FACES`, `"axis0_low"` (index 0
                along the mask's first axis) to `"axis2_high"` (the last index along its third).
                No argument declares none, the default.

        Returns:
            Self: this factory, for chaining.

        Raises:
            ValueError: if a name is not an image face.
        """
        self.physical_image_faces = normalise_physical_image_faces(faces)
        return self

    def set_junction_relocation(self, enabled: bool = True) -> Self:
        """Choose whether the finished mesh's trijunction vertices move onto the mask's own trijunction curves.

        **On by default since 0.5.0**, for this factory and for every `get_*_algorithm` static
        method. `set_junction_relocation(False)` switches it off and reproduces the 0.4 meshes
        exactly. The post-process is :func:`dw3d.relocate_junction_vertices`; it changes vertex
        positions only, never connectivity or labels, under a guard that provably cannot create a
        self-intersection anywhere in the mesh.

        Cost, in seconds: about +0.75 s on the benchmark meshes (whose reconstruction takes about
        5 s, +15 %); 0.1-1.5 s per timepoint on real embryo volumes (median 0.34 s and 0.48 s on two
        time series, whose reconstruction takes about 0.6 s). Within one real series it grows
        faster than linearly with mesh size (fitted exponent 2.68 over a 2.2-fold range of
        triangle counts in a fixed volume -- a description of that series, not a tissue-scale law).

        Measured over 40 equilibrium benchmark cases, four reconstruction spacings and four point
        samplers -- 16 independent adjudications, all passing: line position error, local tangent
        error and discrete curvature all improve with their intervals clear of zero, no pre-existing
        validity predicate changes, and the self-intersection count never rises. On the seed-free
        samplers it also improves contact angles by 2.3-3.5 degrees and gauge-matched tension error
        by 20-40 %; on the seeded default sampler the line improves and the angle does not move
        significantly. At a point-placement `min_distance` of 5 or more the reconstruction emits
        :class:`dw3d.CoarseSpacingRelocationWarning` once.

        Args:
            enabled (bool, optional): whether the post-process runs. Defaults to True, so that
                `set_junction_relocation()` reads as switching it on.

        Returns:
            Self: this factory, for chaining.
        """
        self.relocate_junctions = enabled
        return self

    def set_canonical_tetrahedron_order(self, enabled: bool = True) -> Self:
        """Choose whether the tesselation's tetrahedra are put in canonical order before anything reads them.

        **On by default since 0.7.1**, for this factory and for every `get_*_algorithm` static
        method. The canonical order sorts the vertex indices within each tetrahedron and the
        tetrahedra lexicographically (`dw3d.tesselation.canonical_tetrahedron_order`), so the mesh
        no longer depends on the row order in which Qhull returns the same tetrahedra.
        `set_canonical_tetrahedron_order(False)` keeps Qhull's row order, which reproduces the
        meshes of dw3d 0.7 and earlier on the machine they were made on. See
        `MeshReconstructionAlgorithm.__init__` for what the row order changes.

        Args:
            enabled (bool, optional): whether the tetrahedra are put in canonical order. Defaults
                to True, so that `set_canonical_tetrahedron_order()` reads as switching it on.

        Returns:
            Self: this factory, for chaining.
        """
        self.canonical_tetrahedron_order = enabled
        return self

    def set_classical_edt_method(self) -> Self:
        """Use the classical method to create the Euclidean Distance Transform from a segmentation mask (the default).

        Returns:
            Self: this factory, for chaining.
        """
        self._edt_creation_function = _edt_creation_function_classical(self.print_info)
        return self

    def set_edt_boundary_bias_method(self) -> Self:
        """Use the biased method to create the Euclidean Distance Transform from a segmentation mask.

        Returns:
            Self: this factory, for chaining.
        """
        self._edt_creation_function = _edt_creation_function_boundary_bias(self.print_info)
        return self

    def set_float32_edt_method(self, parallel: int = 1) -> Self:
        """Use the float32 EDT: the same field, 2 full-volume arrays instead of ~7.

        **Bit-identical** to `set_classical_edt_method`, not an approximation — see
        `dw3d.edt`'s module docstring for why (`edt.edt` already returns float32; the
        classical path's float64 is an `int32` promotion artifact). Safe as a drop-in.

        Args:
            parallel (int, optional): Threads for `edt.edt`. `1` matches the classical path
                and keeps the timing comparison honest; the result does not depend on it.

        Returns:
            Self: this factory, for chaining.
        """
        self._edt_creation_function = partial(
            compute_edt_float32,
            print_info=self.print_info,
            parallel=parallel,
        )
        return self

    def set_tiled_edt_method(
        self,
        tile_size: int = 256,
        halo: int = 16,
        parallel: int = 1,
        dtype: type = np.float32,
    ) -> Self:
        """Use the tiled EDT: 1 full-volume array plus `O(tile_size**3)` scratch.

        Bit-identical in *value* to the classical path, by the certified escalating-halo argument
        in `dw3d.edt`'s module docstring; it raises rather than returning an upper bound if a core
        ever fails to certify.

        **`dtype` is not cosmetic — it decides whether this composes with the current default.**
        At the default (`np.float32`) the *field* matches the classical one bit for bit but its
        dtype does not, and the junction-protected configuration reads the EDT's last bits (the
        boundary-layer scheme's Hessian eigenvector normals, the 27-sample face score
        interpolation), so the finished reconstruction moves on 8 of 8 in-repo configurations —
        see `tests/test_edt_dtype_composition.py`. Passing
        `dtype=np.float64` gives a field that reproduces the default bit for bit
        (`tests/test_edt_float64_tiling.py`) at 8 bytes/voxel instead of 4.

        Args:
            tile_size (int, optional): Core side length. Keep it well above the cells'
                inradius: the per-tile work factor is `((tile_size + 2*halo)/tile_size)**3`.
                Defaults to 256.
            halo (int, optional): Initial halo; tiles escalate once if they need more.
            parallel (int, optional): Threads for `edt.edt`. Defaults to 1.
            dtype (type, optional): `np.float32` (this EDT's own contract) or `np.float64`
                (composes with the current default). Defaults to `np.float32`, so this
                method's existing behaviour and every fixture pinned to it are unchanged.

        Returns:
            Self: this factory, for chaining.
        """
        self._edt_creation_function = partial(
            compute_edt_tiled,
            tile_size=tile_size,
            halo=halo,
            print_info=self.print_info,
            parallel=parallel,
            dtype=dtype,
        )
        return self

    def set_peak_local_points_placement_method(self, min_distance: int = 3) -> Self:
        """Place points on the EDT image using local extrema of the EDT (and corners), deterministically.

        This is the placement of `get_deterministic_algorithm` and of a new factory.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels of the grid
                the mask is meshed on. Defaults to 3.

        Returns:
            Self: this factory, for chaining.
        """
        self._point_placing_function = _point_placing_function_peak_local(min_distance, self.print_info)
        return self

    def set_dithered_peak_local_points_placement_method(self, min_distance: int = 3, seed: int = 42) -> Self:
        """Place points using the randomly dithered EDT (dw3d <= 0.3.6 behaviour).

        This is the placement of `get_dithered_algorithm`, which `get_default_algorithm` returns;
        a new factory uses the deterministic placement instead
        (`set_peak_local_points_placement_method`). See `dw3d.points_on_edt` for the difference.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels of the grid
                the mask is meshed on. Defaults to 3.
            seed (int, optional): Seed of the `U(0, 1e-5)` dither. Defaults to 42, the
                value that was hard-wired before point placement was made deterministic.

        Returns:
            Self: this factory, for chaining.
        """
        self._point_placing_function = _point_placing_function_peak_local_dithered(min_distance, seed, self.print_info)
        return self

    def set_boundary_layer_points_placement_method(self, min_distance: int = 3, delta: float | None = None) -> Self:
        """Place points using the boundary-layer scheme.

        The deterministic extrema plus a structured layer of offset points a distance
        `delta` into each material adjacent to a genuine interface; see `dw3d.points_on_edt`
        for the construction. Opt-in: the default algorithm does not use it.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.
            delta (float | None, optional): Offset distance into each material. Defaults to
                `min_distance` (the interface spacing `h_S`) when None.

        Returns:
            Self: this factory, for chaining.
        """
        self._point_placing_function = _point_placing_function_boundary_layer(min_distance, delta, self.print_info)
        return self

    def set_junction_protected_points_placement_method(
        self,
        min_distance: int = 3,
        delta: float | None = None,
        junction_spacing: float | None = None,
        shell_coarsening: int = 1,
    ) -> Self:
        """Place points using the junction-protected scheme.

        The boundary layer plus 0-/1-junction detection, protecting balls and a junction
        boundary layer; see `dw3d.points_on_edt` for the construction. This scheme returns
        per-point weights, so pair it with `set_weighted_delaunay_tesselation_method` --
        `get_junction_protected_algorithm` does that for you. Opt-in: the default algorithm does
        not use it.

        Its defaults are those of the junction-protection scheme as first designed, not those of
        `get_junction_protected_algorithm`: the junction boundary layer is on and the shell
        coarsening is off (`get_junction_protected_algorithm` defaults to
        `junction_boundary_layer=False` and `shell_coarsening=3`).

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.
            delta (float | None, optional): Boundary-layer offset distance. Defaults to
                `min_distance`.
            junction_spacing (float | None, optional): Arclength `h_J` between 1-junction
                samples. Defaults to `2*min_distance + 1`, the effective interface spacing.
            shell_coarsening (int, optional): Re-pack the bounding-box / background shell
                minima at this multiple of `2*min_distance + 1`. `1` is off. Defaults to 1.

        Returns:
            Self: this factory, for chaining.
        """
        self._point_placing_function = _point_placing_function_junction_protected(
            min_distance=min_distance,
            delta=delta,
            junction_spacing=junction_spacing,
            shell_coarsening=shell_coarsening,
            print_info=self.print_info,
        )
        return self

    def set_weighted_delaunay_tesselation_method(self) -> Self:
        """Use a regular (weighted Delaunay) triangulation, driven by the point weights.

        Falls back to `simple_delaunay_tesselation` bit-for-bit when the point-placing
        function supplies no weights, or supplies all-zero ones (see `dw3d.tesselation`).

        Returns:
            Self: this factory, for chaining.
        """
        self._tesselation_creation_function = weighted_delaunay_tesselation
        return self

    def set_peak_local_points_bias_boundary_placement_method(self, min_distance: int = 3) -> Self:
        """Place points on the EDT image using local extrema of the EDT (and corners) and boundaries.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.

        Returns:
            Self: this factory, for chaining.
        """
        self._point_placing_function = _point_placing_function_peak_local_bias_boundaries(min_distance, self.print_info)
        return self

    def set_delaunay_tesselation_method(self) -> Self:
        """Use Delaunay algorithm to create the tesselation from a list of points (the default).

        Returns:
            Self: this factory, for chaining.
        """
        self._tesselation_creation_function = simple_delaunay_tesselation
        return self

    def set_score_computation_by_mean_value(self, spline_order: int = 1) -> Self:
        """Use the mean value of the EDT image on triangle faces of the tesselation to compute scores for Watershed.

        Args:
            spline_order (int, optional): 1 for trilinear interpolation of the EDT (the
                historical behaviour), 3 for the cubic B-spline interpolant. Defaults to 1.

        Returns:
            Self: this factory, for chaining.
        """
        self._score_computation_function = (
            compute_scores_by_mean_value_cubic if spline_order == 3 else compute_scores_by_mean_value
        )
        return self

    def set_score_computation_by_max_value(self, spline_order: int = 1) -> Self:
        """Use the max value of the EDT image on triangle faces of the tesselation to compute scores for Watershed.

        Args:
            spline_order (int, optional): 1 for trilinear interpolation of the EDT (the
                historical behaviour), 3 for the cubic B-spline interpolant. Defaults to 1.
                The cubic interpolant is `C2` where trilinear is only `C0`, which both
                smooths the score field and is a prerequisite for sub-voxel extremum
                refinement (not implemented here). It changes the labelling, so it is opt-in.

        Returns:
            Self: this factory, for chaining.
        """
        self._score_computation_function = (
            compute_scores_by_max_value_cubic if spline_order == 3 else compute_scores_by_max_value
        )
        return self

    def make_algorithm(self) -> MeshReconstructionAlgorithm:
        """Build a new mesh reconstruction algorithm using the previously set methods.

        Returns:
            MeshReconstructionAlgorithm: Mesh Reconstruction Algorithm ready to be executed.
        """
        return MeshReconstructionAlgorithm(
            print_info=self.print_info,
            edt_creation_function=self._edt_creation_function,
            point_placing_function=self._point_placing_function,
            tesselation_creation_function=self._tesselation_creation_function,
            score_computation_function=self._score_computation_function,
            perform_mesh_postprocess_surgery=self.perform_mesh_postprocess_surgery,
            relocate_junctions=self.relocate_junctions,
            physical_image_faces=self.physical_image_faces,
            canonical_tetrahedron_order=self.canonical_tetrahedron_order,
        )

    @staticmethod
    def get_default_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the default mesh reconstruction algorithm with sensible default values.

        **This is `get_dithered_algorithm` at its own defaults**: the published dw3d 0.3.6
        configuration. The default was for a time the junction-protected, offset-excluded
        configuration (`get_link_checked_algorithm`); it went back to the dithered one
        because the end-to-end tension-inference accuracy that the meshing pipeline exists to
        serve is *worse* with the revised meshing than with the dithered configuration, on
        every method measured, despite the revised mesh's better geometric quality in
        isolation:

        | method | dithered `E_total` (gauge-matched median) | revised (`link_checked`) `E_total` |
        |---|---:|---:|
        | `YoungDupre` | 0.0763 | 0.1003 |
        | `YoungDupreLocal` | 0.0911 | 0.1266 |
        | `ForceBalanceKKT` | 0.1138 | 0.3262 |
        | `BayesianMAP` | 0.1186 | 0.3149 |

        (42 equilibrium cases of `foambryo`'s benchmarking dataset, `min_distance` 3, dw3d 0.7.1,
        paired per case; see `BENCHMARKS.md` at the repository root.) The revised pipeline is real, deterministic,
        junction-protected and topologically cleaner than the dithered default by every
        meshing-only metric (interface-area bias, sliver fraction, score-gap degeneracy,
        non-manifold edges) -- it is kept, fully reachable, as
        an opt-in variant: **`get_link_checked_algorithm`**, a name that describes its
        mechanism (a link-condition-checked, topology-preserving offset collapse). Use it when
        meshing quality (junction placement, interface smoothness, topology) matters more than
        today's downstream tension-inference accuracy; use the default when tension inference
        is the goal, which is the common case.

        Every configuration reachable from here keeps its own reference meshes and 51-case
        benchmark records:

        * `get_link_checked_algorithm` -- boundary layer + junction protection + link-checked
          offset exclusion; the opt-in "revised" variant, see above.
        * `get_junction_protected_algorithm` at its defaults -- boundary layer + junction
          protection alone, with the boundary-layer offsets left in the extracted surface.
          **Carries a +15.7 % interface-area regression** against ground truth that the
          link-checked exclusion removes; kept so it can be benchmarked against. (Its synonym
          `get_offset_included_algorithm`, deprecated since 0.5.0, was removed in 0.7.0.)
        * `get_deterministic_algorithm` -- deterministic EDT extrema only, plain Delaunay, no
          boundary layer or junction protection: the baseline the boundary-layer and
          junction-protection placements are measured against.

        Every configuration, this one included, relocates junction vertices by default since
        0.5.0; set `relocate_junctions = False` on the returned algorithm to reproduce 0.4. Every
        configuration puts the tetrahedra in canonical order by default since 0.7.1; set
        `canonical_tetrahedron_order = False` on the returned algorithm to keep Qhull's row order,
        as dw3d 0.7 did.

        The EDT is `compute_edt_classical`. The tiled EDT (`set_tiled_edt_method`, reachable
        from a factory instance, not used here) produces a field whose *values* are bit-identical
        at a third of the peak RSS, and it is the variant to prefer at scale. Composition with a
        given reconstruction algorithm depends on whether that algorithm reads the EDT field's
        last bits (`get_dithered_algorithm` and `get_deterministic_algorithm` do not;
        `get_link_checked_algorithm` and its boundary-layer-derived relatives do, through the
        Hessian-eigenvector normals and the 27-sample face score interpolation -- see
        `tests/test_edt_dtype_composition.py`, and
        `tests/test_edt_float64_tiling.py` for the `dtype=np.float64` variant that does compose
        with them at 8 bytes/voxel instead of 4).

        Args:
            min_distance (int, optional): Minimum distance (in voxels of the grid the mask is meshed
                on) between extrema when placing tesselation points. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing. Defaults to False.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        return MeshReconstructionAlgorithmFactory.get_dithered_algorithm(
            min_distance=min_distance,
            print_info=print_info,
        )

    @staticmethod
    def get_deterministic_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the deterministic algorithm: EDT extrema only, no dither, plain Delaunay.

        This was `get_default_algorithm` for a time, between the dithered configuration and
        the junction-protected one. It is kept as a named variant, with its own reference
        meshes (`tests/golden/deterministic/`) and 51-case benchmark records
        (`dw3d_benchmarks/baseline/deterministic/`), because it is the baseline the
        boundary-layer and junction-protection placements are measured against -- dropping it
        would make those measurements unreproducible. A new `MeshReconstructionAlgorithmFactory`
        builds the same configuration.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        return MeshReconstructionAlgorithm(
            print_info=print_info,
            edt_creation_function=_edt_creation_function_classical(print_info=print_info),
            point_placing_function=_point_placing_function_peak_local(min_distance=min_distance, print_info=print_info),
            tesselation_creation_function=simple_delaunay_tesselation,
            score_computation_function=compute_scores_by_max_value,
            perform_mesh_postprocess_surgery=True,
        )

    @staticmethod
    def get_dithered_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        seed: int = 42,
    ) -> MeshReconstructionAlgorithm:
        """Return the reconstruction algorithm as it was in dw3d 0.3.6, before point placement was made deterministic.

        **This is `get_default_algorithm` again** (it was the default originally, then
        superseded for a time by the junction-protected and link-checked configurations,
        then restored): the revised meshing pipeline (`get_link_checked_algorithm`) produces
        better mesh geometry in isolation but *worse* end-to-end tension-inference accuracy
        on every method measured, so the dithered configuration ships as the default and the
        revised pipeline is the opt-in variant; see `get_default_algorithm`'s docstring for
        the numbers.

        The *point placement* is the historical randomly dithered one, so this reproduces
        the 0.3.6 point set exactly. The watershed's edge ordering is **not** reverted: the
        historical ordering was a sort with no tie-break over a score field that is ~30 %
        exactly tied, so it has no single well-defined answer to reproduce — it depended on
        numpy's sorting kernel. `tests/golden/dithered/` records what this factory produces
        on the current stack, and is therefore a point-budget reference rather than a
        bit-for-bit replica of published output.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            seed (int, optional): Seed of the dither. Defaults to 42, the historical value.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        return MeshReconstructionAlgorithm(
            print_info=print_info,
            edt_creation_function=_edt_creation_function_classical(print_info=print_info),
            point_placing_function=_point_placing_function_peak_local_dithered(
                min_distance=min_distance,
                seed=seed,
                print_info=print_info,
            ),
            tesselation_creation_function=simple_delaunay_tesselation,
            score_computation_function=compute_scores_by_max_value,
            perform_mesh_postprocess_surgery=True,
        )

    @staticmethod
    def get_boundary_layer_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        delta: float | None = None,
    ) -> MeshReconstructionAlgorithm:
        """Return the boundary-layer reconstruction algorithm (opt-in, not the default).

        Same as `get_deterministic_algorithm` but with the boundary-layer point placement
        (`set_boundary_layer_points_placement_method`): the deterministic extrema plus offset
        points. Kept as a named variant so its reference meshes and benchmark records live
        alongside the others'; it did not become the default.

        Args:
            min_distance (int, optional): Minimum distance between extrema, in voxels. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            delta (float | None, optional): Offset distance into each material. Defaults to
                `min_distance` when None.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        return MeshReconstructionAlgorithm(
            print_info=print_info,
            edt_creation_function=_edt_creation_function_classical(print_info=print_info),
            point_placing_function=_point_placing_function_boundary_layer(
                min_distance=min_distance,
                delta=delta,
                print_info=print_info,
            ),
            tesselation_creation_function=simple_delaunay_tesselation,
            score_computation_function=compute_scores_by_max_value,
            perform_mesh_postprocess_surgery=True,
        )

    @staticmethod
    def get_junction_protected_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        delta: float | None = None,
        junction_spacing: float | None = None,
        shell_coarsening: int = 3,
        protect_radius: float | None = None,
        junction_delta: float | None = None,
        protect_junctions: bool = True,
        junction_boundary_layer: bool = False,
        use_weights: bool = True,
        spline_order: int = 1,
        exclude_offsets_from_surface: bool = False,
        guard_duplicate_faces: bool = True,
        collapse_rule: str = WELD,
    ) -> MeshReconstructionAlgorithm:
        """Return the junction-protected reconstruction algorithm.

        **This is the configuration `get_default_algorithm` returned for a time before the
        link-checked offset exclusion became the default** (its former synonym
        `get_offset_included_algorithm` was removed in 0.7.0). It is the only way to reach the
        ablation switches below, and the corresponding benchmark records are keyed on it. This
        configuration **plus** the link-checked offset exclusion (`get_link_checked_algorithm`)
        is reachable from here as
        `exclude_offsets_from_surface=True, collapse_rule=LINK_CONDITION`.

        The boundary layer **plus** 0-/1-junction detection, protecting balls and shell
        coarsening, tesselated by the regular (weighted) triangulation that consumes the
        protecting-ball radii. The boundary layer and junction protection are assessed as a
        pair, so this getter is that combined configuration;
        `get_boundary_layer_algorithm` remains the boundary layer alone and
        `get_deterministic_algorithm` is the configuration that predates both.

        **Two defaults differ from the junction-protection scheme as first designed, both on
        measured grounds:**

        * `junction_boundary_layer=False`. The first design asks for the boundary layer
          around junction samples too, on the grounds that it prevents the "more points
          locally -> more small tets" failure. Measured over 11 cases it *causes* that
          failure: it takes valence->=4 edges 33 -> 119 and abnormal non-manifold edges
          2 -> 91, essentially all of the added valence->=4 edges being the 3-material
          (pinched-triple-line) kind. The mechanism is structural, not a bad constant: the
          junction offsets refill the protecting balls that the same construction has just
          emptied, and a protecting ball only protects while it stays empty. The layer is
          implemented and reachable with `junction_boundary_layer=True` so the originally
          specified configuration stays reproducible.
        * `shell_coarsening=3`. Off (`1`) is a valid configuration and meets the topology
          criteria too; 3 is chosen for the point budget and wall time, not to make any
          acceptance criterion pass, on the strength of a sweep over the coarsening factor.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            delta (float | None, optional): Boundary-layer offset distance. Defaults to
                `min_distance` when None.
            junction_spacing (float | None, optional): `h_J`; defaults to
                `2*min_distance + 1` when None.
            shell_coarsening (int, optional): Shell re-packing factor; 1 is off.
                Defaults to 3.
            protect_radius (float | None, optional): Protecting-ball radius. Defaults to
                `junction_spacing / 2`.
            junction_delta (float | None, optional): Junction boundary-layer offset
                distance. Defaults to `protect_radius`.
            protect_junctions (bool, optional): Ablation switch; see
                `dw3d.points_on_edt.junction_protected_families`. Defaults to True.
            junction_boundary_layer (bool, optional): Ablation switch; `True` restores the
                junction-protection scheme as first designed. Defaults to False — see above.
            use_weights (bool, optional): Ablation switch — `False` tesselates the same
                point set with plain Delaunay, isolating the regular triangulation's own
                contribution. Defaults to True.
            spline_order (int, optional): EDT interpolation order for the watershed scores;
                `3` selects the cubic B-spline. Defaults to 1, trilinear, which is what the
                default algorithm uses. See
                `dw3d.score_computation`.
            exclude_offsets_from_surface (bool, optional): Keep the boundary-layer offsets in
                the tesselation and out of the extracted surface. The tesselation, the watershed and
                the labelling are bit-identical either way. Defaults to False.
            guard_duplicate_faces (bool, optional): Offset-exclusion ablation switch —
                refuse the merges that would duplicate a face. `False` reproduces the
                unguarded construction, which trades 4 boundary edges for 38 fewer offsets
                on `3.tif`. Defaults to True. Ignored under the
                `"link_condition"` collapse rule, which forbids the same thing.
            collapse_rule (str, optional): `"weld"` merges every offset it can (the
                unconditional-merge rule, which regressed valence->=4 edges 115 -> 184 and
                abnormal non-manifold edges 7 -> 76), `"link_condition"` accepts only the
                merges that cannot change the surface topology. See
                `dw3d.mesh_utilities.exclude_offsets_from_surface`. Defaults to `"weld"` so
                that the unconditional-merge benchmark records stay reproducible;
                `get_link_checked_algorithm` is the link-condition configuration.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        return MeshReconstructionAlgorithm(
            print_info=print_info,
            edt_creation_function=_edt_creation_function_classical(print_info=print_info),
            point_placing_function=_point_placing_function_junction_protected(
                min_distance=min_distance,
                delta=delta,
                junction_spacing=junction_spacing,
                shell_coarsening=shell_coarsening,
                protect_radius=protect_radius,
                junction_delta=junction_delta,
                protect_junctions=protect_junctions,
                junction_boundary_layer=junction_boundary_layer,
                exclude_offsets_from_surface=exclude_offsets_from_surface,
                guard_duplicate_faces=guard_duplicate_faces,
                collapse_rule=collapse_rule,
                print_info=print_info,
            ),
            tesselation_creation_function=(
                weighted_delaunay_tesselation if use_weights else simple_delaunay_tesselation
            ),
            score_computation_function=(
                compute_scores_by_max_value_cubic if spline_order == 3 else compute_scores_by_max_value
            ),
            perform_mesh_postprocess_surgery=True,
        )

    @staticmethod
    def get_link_checked_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        spline_order: int = 1,
    ) -> MeshReconstructionAlgorithm:
        """Return the "revised meshing" variant: junction-protected with a topology-preserving offset collapse.

        **This was `get_default_algorithm` for a time**, until the default went back to the
        dithered configuration because end-to-end tension-inference accuracy is worse with
        this pipeline than with the dithered one on every method measured, despite its
        better mesh quality in isolation (interface-area bias, sliver fraction, junction
        placement, topology -- see `get_default_algorithm`'s docstring for the numbers). It
        is deterministic, junction-protected and topologically corrected relative to the
        dithered configuration; "link-checked" names its mechanism (a link-condition-checked,
        topology-preserving offset collapse). Use it when mesh geometry and topology matter
        more than today's downstream tension-inference accuracy. Its benchmark records are
        keyed on the variant name `offset_excluded_linkcheck`.

        Identical to `get_junction_protected_algorithm(exclude_offsets_from_surface=True)` except
        that each offset is removed from the surface by a **link-condition-checked edge contraction** rather than an
        unconditional weld. Same EDT, same point set, same regular triangulation, same
        watershed, same mesh surgery, same set of *candidate* merges — only the test that
        decides which candidates are taken differs.

        **Why this exists.** The unconditional weld bought a large geometric improvement
        (the offset-included default over-estimates every interface area by +15.7 %, the
        weld-based exclusion by +1.2 %) at the cost of a real topological regression:
        welding two vertices makes formerly distinct edges the same edge, so their triangle
        incidences add, and over the 51-case benchmark that took valence->=4 edges
        115 -> 184 and abnormal non-manifold edges 7 -> 76. The
        link-condition-checked collapse refuses exactly the merges that would do that; see
        `dw3d.mesh_utilities._refuse_by_link_condition` for the four tests, and for the
        non-manifold counterexample showing why the classical link condition alone is not
        sufficient here.

        It refuses **fewer** merges than the duplicate-face guard on the unconditional weld,
        not more, because retrying a refused candidate once its neighbours have collapsed
        recovers most of them.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            spline_order (int, optional): EDT interpolation order for the watershed scores;
                `3` composes this collapse rule with the cubic B-spline. Defaults to 1.

        Returns:
            MeshReconstructionAlgorithm: a new algorithm, with junction relocation on.
        """
        # The offset exclusion with the link-condition collapse rule (guard_duplicate_faces keeps
        # its default).
        return MeshReconstructionAlgorithmFactory.get_junction_protected_algorithm(
            min_distance=min_distance,
            print_info=print_info,
            spline_order=spline_order,
            exclude_offsets_from_surface=True,
            collapse_rule=LINK_CONDITION,
        )


def _edt_creation_function_classical(
    print_info: bool = False,
) -> EdtCreationFunction:
    """Get a function that compute an EDT from a segmentation mask.

    Args:
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        EdtCreationFunction:
            - the actual function which takes a segmentation mask and return an Euclidean Distance Transform image.
    """
    return partial(compute_edt_classical, print_info=print_info)


def _edt_creation_function_boundary_bias(
    print_info: bool = False,
) -> EdtCreationFunction:
    """Get a function that compute an EDT from a segmentation mask.

    Args:
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        EdtCreationFunction:
            - the actual function which takes a segmentation mask and return an Euclidean Distance Transform image.
    """
    return partial(compute_edt_boundary_bias, print_info=print_info)


def _point_placing_function_peak_local(
    min_distance: int = 3,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get a function that peaks local min and max points (+ corner points) from an EDT image.

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.uint]]:
            - the actual function which takes only an EDT image and return an array of 3D pixel coordinates
              of local min & max of the EDT (+ corners)
    """
    return partial(peak_local_points, min_distance=min_distance, print_info=print_info)


def _point_placing_function_peak_local_dithered(
    min_distance: int = 3,
    seed: int = 42,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get the historical dithered point-placing function (dw3d <= 0.3.6).

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        seed (int, optional): Seed of the `U(0, 1e-5)` dither. Defaults to 42.
        print_info (bool, optional): Print details about the algorithm. Defaults to False.

    Returns:
        PointPlacingFunction:
            - the actual function which takes a segmentation mask and an EDT image and
              returns an array of 3D pixel coordinates of dithered local min & max of the
              EDT (+ corners), plus the indices of the maxima sorted by EDT value.
    """
    return partial(peak_local_points_dithered, min_distance=min_distance, seed=seed, print_info=print_info)


def _point_placing_function_boundary_layer(
    min_distance: int = 3,
    delta: float | None = None,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get the boundary-layer point-placing function.

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        delta (float | None, optional): Offset distance into each material. Defaults to
            `min_distance` when None.
        print_info (bool, optional): Print details about the algorithm. Defaults to False.

    Returns:
        PointPlacingFunction:
            - a function taking a segmentation mask and an EDT image and returning the
              stacked boundary-layer point set plus the indices of the non-interface
              interior points sorted by EDT value.
    """
    return partial(peak_local_points_boundary_layer, min_distance=min_distance, delta=delta, print_info=print_info)


def _point_placing_function_junction_protected(
    min_distance: int = 3,
    delta: float | None = None,
    junction_spacing: float | None = None,
    shell_coarsening: int = 1,
    protect_radius: float | None = None,
    junction_delta: float | None = None,
    protect_junctions: bool = True,
    junction_boundary_layer: bool = True,
    exclude_offsets_from_surface: bool = False,
    guard_duplicate_faces: bool = True,
    collapse_rule: str = WELD,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get the junction-protected point-placing function.

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        delta (float | None, optional): Boundary-layer offset distance. Defaults to
            `min_distance` when None.
        junction_spacing (float | None, optional): `h_J`. Defaults to `2*min_distance + 1`.
        shell_coarsening (int, optional): Shell re-packing factor; 1 is off. Defaults to 1.
        protect_radius (float | None, optional): Protecting-ball radius. Defaults to
            `junction_spacing / 2`.
        junction_delta (float | None, optional): Junction boundary-layer offset distance.
            Defaults to `protect_radius`.
        protect_junctions (bool, optional): Ablation switch. Defaults to True.
        junction_boundary_layer (bool, optional): Ablation switch. Defaults to True.
        exclude_offsets_from_surface (bool, optional): Return the point metadata that keeps
            the boundary-layer offsets out of the extracted surface. Defaults to False.
        guard_duplicate_faces (bool, optional): Offset-exclusion ablation switch, forwarded
            into the metadata. Defaults to True.
        collapse_rule (str, optional): Offset-exclusion collapse rule, forwarded into the
            metadata. Defaults to `"weld"`.
        print_info (bool, optional): Print details about the algorithm. Defaults to False.

    Returns:
        PointPlacingFunction:
            - a function taking a segmentation mask and an EDT image and returning the
              stacked junction-protected point set, the indices of the non-interface
              interior points sorted by EDT value, and the per-point protecting-ball
              weights (the 3-tuple form of the contract).
    """
    return partial(
        peak_local_points_junction_protected,
        min_distance=min_distance,
        delta=delta,
        junction_spacing=junction_spacing,
        shell_coarsening=shell_coarsening,
        protect_radius=protect_radius,
        junction_delta=junction_delta,
        protect_junctions=protect_junctions,
        junction_boundary_layer=junction_boundary_layer,
        exclude_offsets_from_surface=exclude_offsets_from_surface,
        guard_duplicate_faces=guard_duplicate_faces,
        collapse_rule=collapse_rule,
        print_info=print_info,
    )


def _point_placing_function_peak_local_bias_boundaries(
    min_distance: int = 3,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get a function that peaks local min and max points (+ corner points) from an EDT image.

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.uint]]:
            - the actual function which takes only an EDT image and return an array of 3D pixel coordinates
              of local min & max of the EDT (+ corners)
    """
    return partial(peak_local_points_bias_boundaries, min_distance=min_distance, print_info=print_info)
