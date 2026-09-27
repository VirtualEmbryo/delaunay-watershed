"""Module for factory class MeshReconstructionAlgorithmFactory.

This class allows to build custom algorithms for mesh reconstruction from segmentation masks.

Matthieu Perez 2024
"""

import warnings
from functools import partial
from typing import Self

import numpy as np

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
    """Factory to build custom algorithms for mesh reconstruction from segmentation masks."""

    def __init__(self, print_info: bool = False, perform_mesh_postprocess_surgery: bool = True) -> None:
        """Initialize the factory."""
        self.print_info = print_info
        self._edt_creation_function: EdtCreationFunction = _edt_creation_function_classical(print_info=print_info)
        self._point_placing_function: PointPlacingFunction = _point_placing_function_peak_local(print_info=print_info)
        self._tesselation_creation_function: TesselationCreationFunction = simple_delaunay_tesselation
        self._score_computation_function: ScoreComputationFunction = compute_scores_by_max_value
        self.perform_mesh_postprocess_surgery = perform_mesh_postprocess_surgery
        self.relocate_junctions = True

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

    def set_classical_edt_method(self) -> Self:
        """Use the classical method to create the Euclidean Distance Transform from a segmentation mask."""
        self._edt_creation_function = _edt_creation_function_classical(self.print_info)
        return self

    def set_edt_boundary_bias_method(self) -> Self:
        """Use the biased method to create the Euclidean Distance Transform from a segmentation mask."""
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
        (`tests/test_edt_float64_tiling.py`) at 8 bytes/voxel instead of 4, which is the trade
        measured when the tiled EDT's memory/precision trade-off was characterized.

        Args:
            tile_size (int, optional): Core side length. Keep it well above the cells'
                inradius: the per-tile work factor is `((tile_size + 2*halo)/tile_size)**3`.
                Defaults to 256.
            halo (int, optional): Initial halo; tiles escalate once if they need more.
            parallel (int, optional): Threads for `edt.edt`. Defaults to 1.
            dtype (type, optional): `np.float32` (this EDT's own contract) or `np.float64`
                (composes with the current default). Defaults to `np.float32`, so this
                method's existing behaviour and every fixture pinned to it are unchanged.
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
        """Place points on the EDT image using local extrema of the EDT (and corners).

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        """
        self._point_placing_function = _point_placing_function_peak_local(min_distance, self.print_info)
        return self

    def set_dithered_peak_local_points_placement_method(self, min_distance: int = 3, seed: int = 42) -> Self:
        """Place points using the randomly dithered EDT (dw3d <= 0.3.6 behaviour).

        Kept reachable so the seed-to-seed determinism sweep and the dithered-configuration
        golden masters can still be run; see `dw3d.points_on_edt` for why it is no longer
        the default point-placement rule for the deterministic configuration.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            seed (int, optional): Seed of the `U(0, 1e-5)` dither. Defaults to 42, the
                value that was hard-wired before point placement was made deterministic.
        """
        self._point_placing_function = _point_placing_function_peak_local_dithered(min_distance, seed, self.print_info)
        return self

    def set_boundary_layer_points_placement_method(self, min_distance: int = 3, delta: float | None = None) -> Self:
        """Place points using the boundary-layer scheme.

        The deterministic extrema plus a structured layer of offset points a distance
        `delta` into each material adjacent to a genuine interface; see `dw3d.points_on_edt`
        for the construction. Opt-in: the default remained the extrema-only rule until the
        benchmark numbers justified making the junction-protected configuration the default.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            delta (float | None, optional): Offset distance into each material. Defaults to
                `min_distance` (the interface spacing `h_S`) when None.
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
        `get_junction_protected_algorithm` does that for you. Opt-in: the default is
        unchanged by this alone.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            delta (float | None, optional): Boundary-layer offset distance. Defaults to
                `min_distance`.
            junction_spacing (float | None, optional): Arclength `h_J` between 1-junction
                samples. Defaults to `2*min_distance + 1`, the effective interface spacing.
            shell_coarsening (int, optional): Re-pack the bounding-box / background shell
                minima at this multiple of `2*min_distance + 1`. `1` is off. Defaults to 1.
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
        """
        self._tesselation_creation_function = weighted_delaunay_tesselation
        return self

    def set_peak_local_points_bias_boundary_placement_method(self, min_distance: int = 3) -> Self:
        """Place points on the EDT image using local extrema of the EDT (and corners) and boundaries.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
        """
        self._point_placing_function = _point_placing_function_peak_local_bias_boundaries(min_distance, self.print_info)
        return self

    def set_delaunay_tesselation_method(self) -> Self:
        """Use Delaunay algorithm to create the tesselation from a list of points."""
        self._tesselation_creation_function = simple_delaunay_tesselation
        return self

    def set_score_computation_by_mean_value(self, spline_order: int = 1) -> Self:
        """Use the mean value of the EDT image on triangle faces of the tesselation to compute scores for Watershed.

        Args:
            spline_order (int, optional): 1 for trilinear interpolation of the EDT (the
                historical behaviour), 3 for the cubic B-spline interpolant. Defaults to 1.
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
        )

    @staticmethod
    def get_default_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the default mesh reconstruction algorithm with sensible default values.

        **This is `get_dithered_algorithm` at its own defaults**: the published dw3d 0.3.6
        configuration. The default was for a time the junction-protected, offset-excluded
        configuration (`get_link_checked_algorithm`); that was reverted here, on
        H. Turlier's decision, because the end-to-end tension-inference accuracy that the
        meshing pipeline exists to serve is *worse* with the revised meshing than with the
        dithered configuration, on every method measured, despite the revised mesh's better
        geometric quality in isolation:

        | method | dithered `E_total` (gauge-matched median) | revised (`link_checked`) `E_total` |
        |---|---:|---:|
        | `YoungDupre` | 0.0882 | 0.1049 |
        | `YoungDupreLocal` | 0.0949 | 0.1245 |
        | `ForceBalanceKKT` | 0.1299 | 0.3436 |
        | `BayesianMAP` | 0.1799 | 0.1962 |

        (42 equilibrium cases of `foambryo`'s benchmarking dataset, paired per case; see
        `BENCHMARKS.md` at the repository root.) The revised pipeline is real, deterministic,
        junction-protected and topologically cleaner than the dithered default by every
        meshing-only metric (interface-area bias, sliver fraction, score-gap degeneracy,
        non-manifold edges) -- it is kept, fully reachable, as
        an opt-in variant: **`get_link_checked_algorithm`**, the name it already had, chosen
        deliberately over introducing another synonym, because it already names the
        mechanism (a link-condition-checked, topology-preserving offset collapse) rather than
        the phase that produced it. Use it when meshing quality (junction placement, interface
        smoothness, topology) matters more than today's downstream tension-inference accuracy;
        use the default when tension inference is the goal, which is the common case.

        Every configuration reachable from here keeps its own golden masters, fingerprints
        and 51-case benchmark records:

        * `get_link_checked_algorithm` -- boundary layer + junction protection + link-checked
          offset exclusion; the opt-in "revised" variant, see above.
        * `get_junction_protected_algorithm` at its defaults -- boundary layer + junction
          protection alone, with the boundary-layer offsets left in the extracted surface.
          **Carries a +15.7 % interface-area regression** against ground truth that the
          link-checked exclusion removes; kept so it can be benchmarked against. (Its synonym
          `get_offset_included_algorithm` is deprecated since 0.5.0 and removed in 0.6.0.)

        Every configuration, this one included, relocates junction vertices by default since
        0.5.0; set `relocate_junctions = False` on the returned algorithm to reproduce 0.4.
        * `get_deterministic_algorithm` -- deterministic EDT extrema only, plain Delaunay, no
          boundary layer or junction protection. Every acceptance criterion for the
          boundary-layer and junction-protection work is stated against it.

        The EDT is `compute_edt_classical`. The tiled EDT (`set_tiled_edt_method`, merged in
        `512fc05`, reachable here, not wired in) produces a field whose *values* are bit-identical
        at a third of the peak RSS, and it is the variant to prefer at scale. Composition with a
        given reconstruction algorithm depends on whether that algorithm reads the EDT field's
        last bits (`get_dithered_algorithm` and `get_deterministic_algorithm` do not;
        `get_link_checked_algorithm` and its boundary-layer-derived relatives do, through the
        Hessian-eigenvector normals and the 27-sample face score interpolation -- see
        `tests/test_edt_dtype_composition.py`, and
        `tests/test_edt_float64_tiling.py` for the `dtype=np.float64` variant that does compose
        with them at 8 bytes/voxel instead of 4).

        Args:
            min_distance (int, optional): Minimum distance (in pixels) between extrema when placing
                tesselation points. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
        """
        return MeshReconstructionAlgorithmFactory.get_dithered_algorithm(
            min_distance=min_distance,
            print_info=print_info,
        )

    @staticmethod
    def get_offset_included_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the boundary-layer + junction-protection configuration, offsets left in the surface.

        **This was `get_default_algorithm` for a time (`e598909`), before the link-checked
        offset exclusion became the default.** It is the same code path as
        `get_junction_protected_algorithm` at its own defaults -- `offset_included` is the
        *configuration's* name, `junction_protected` the *mechanism's*, and
        `tests/test_golden_master.py` pins that the two agree -- but it is registered
        separately because it is now a **documented regression** with a commit to point at,
        not merely a superseded default, and downstream work needs to benchmark with and
        without it by name.

        **What it costs, measured over the 47 ground-truth cases:** signed interface-area error
        **+15.73 %**, with
        **94.1 %** of all 677 interfaces over-estimated and every interface over-estimated on
        25 of 47 whole cases; total reconstructed interface area **1.1616x** ground truth.
        The mechanism is the boundary layer's offsets entering the surface as flap vertices;
        it is a bias *plus* an 11-point spread (pooled p90 25.9 %), so it is **not**
        absorbable downstream as a gauge factor.

        **What it gains, and what the link-checked default keeps bit-identically:**
        all-surface tetrahedra 39.9 % -> 0.81 %, slivers 5.97 % -> 5.70 %, score gaps < 1e-5
        0.484 -> 0.202, junction length 1.2548x -> 1.0085x ground truth, valence->=4 edges
        133 -> 115. None of that is traded away by excluding the offsets from the surface --
        that step changes only which tesselation vertices are allowed to become *surface*
        vertices.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.

        Deprecated:
            Since 0.5.0, and removed in 0.6.0. This is an exact synonym of
            `get_junction_protected_algorithm(min_distance=..., print_info=...)`: same code path,
            same mesh, bit for bit. Call that instead.
        """
        warnings.warn(
            "get_offset_included_algorithm is deprecated since 0.5.0 and will be removed in 0.6.0; it is an "
            "exact synonym of get_junction_protected_algorithm(min_distance=..., print_info=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        return MeshReconstructionAlgorithmFactory.get_junction_protected_algorithm(
            min_distance=min_distance,
            print_info=print_info,
        )

    @staticmethod
    def get_deterministic_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the deterministic algorithm: EDT extrema only, no dither, plain Delaunay.

        This was `get_default_algorithm` before the boundary-layer and junction-protection
        work made the junction-protected configuration the default. It is kept as a named
        variant, with its own golden masters (`tests/golden/deterministic/`), fingerprints
        and 51-case benchmark records (`benchmarks/baseline/deterministic/`), because it is
        the baseline every acceptance criterion for the boundary-layer and
        junction-protection work is stated against -- dropping it would make those
        measurements unreproducible.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
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
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            seed (int, optional): Seed of the dither. Defaults to 42, the historical value.
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

        Same as `get_default_algorithm` but with the boundary-layer point placement
        (`set_boundary_layer_points_placement_method`). Kept as a named variant so its
        golden masters, fingerprints and benchmark records live alongside the default's;
        the recorded numbers are what decided whether this became the shipped default.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            delta (float | None, optional): Offset distance into each material. Defaults to
                `min_distance` when None.
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
        link-checked offset exclusion became the default**; `get_offset_included_algorithm`
        is the name that says so, and this one stays because it is the only way to reach the
        ablation switches below and because the corresponding benchmark records are keyed on
        it. Since the link-checked default, this configuration **plus** the offset exclusion
        (`get_link_checked_algorithm`) is reachable from here as
        `exclude_offsets_from_surface=True, collapse_rule=LINK_CONDITION`.

        The boundary layer **plus** 0-/1-junction detection, protecting balls and shell
        coarsening, tesselated by the regular (weighted) triangulation that consumes the
        protecting-ball radii. The boundary layer and junction protection are assessed as a
        pair, so this getter is that combined configuration;
        `get_boundary_layer_algorithm` remains the boundary layer alone and
        `get_deterministic_algorithm` is the configuration that predates both.

        **Two defaults differ from the original junction-protection specification, both on
        measured grounds:**

        * `junction_boundary_layer=False`. The original specification asks for the boundary layer
          around junction samples too, on the grounds that it prevents Perez's "more points
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
                original junction-protection specification. Defaults to False — see above.
            use_weights (bool, optional): Ablation switch — `False` tesselates the same
                point set with plain Delaunay, isolating the regular triangulation's own
                contribution. Defaults to True.
            spline_order (int, optional): EDT interpolation order for the watershed scores;
                `3` selects the cubic B-spline (`get_cubic_score_algorithm`). Defaults to 1,
                trilinear, which is what the default algorithm uses. See
                `dw3d.score_computation`.
            exclude_offsets_from_surface (bool, optional): Keep the boundary-layer offsets in
                the tesselation and out of the extracted surface. `True` is what
                `get_offset_excluded_algorithm` returns. The tesselation, the watershed and
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
    def get_offset_excluded_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        spline_order: int = 1,
        guard_duplicate_faces: bool = True,
        collapse_rule: str = WELD,
    ) -> MeshReconstructionAlgorithm:
        """Return the offset-excluded variant: junction-protected with the boundary layer kept off the surface.

        Identical to `get_offset_included_algorithm` — same EDT, same point set, same
        regular triangulation, same watershed, same mesh surgery — except that the
        *extracted surface* is not allowed to use the boundary layer's offsets as vertices.
        Each offset is merged onto the interface sample it displaces and the triangles that
        collapse are dropped; see `dw3d.mesh_utilities.exclude_offsets_from_surface`.

        **Not the default.** Its default `collapse_rule="weld"` is the unconditional merge
        rule, which regressed valence->=4 edges 115 -> 184 and abnormal non-manifold edges
        7 -> 76 over the 51-case benchmark. `get_link_checked_algorithm` is the same
        exclusion with the collapse rule that cannot do that, and **it** is the opt-in
        "revised meshing" variant (the default for a time, no longer the default -- see
        `get_default_algorithm`'s docstring for why). This getter is kept so the
        unconditional-merge measurement stays reproducible as the ablation that motivated
        the link-condition-checked collapse.

        **Why this exists.** The extraction is primal, so every mesh vertex is a seed point,
        and the boundary layer's offsets sit `delta` voxels off the surface by construction.
        Measurement found them at 30-35 % of mesh vertices carrying **90 % of `epsilon^2`**,
        the vertex-to-surface error that drives the junction-angle error through
        `error = C * epsilon / ell` with `C = 33.2 +- 1.2` deg.
        They are scaffolding for the tetrahedra, not geometry.

        Opt-in, with its own golden masters and 51-case records: it changes the surface on
        every case.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.
            spline_order (int, optional): EDT interpolation order for the watershed scores;
                `3` composes this exclusion with the cubic B-spline. Defaults to 1, matching
                `get_link_checked_algorithm`.
            guard_duplicate_faces (bool, optional): Ablation switch; `False` gives the
                unguarded merge. Defaults to True. See
                `dw3d.mesh_utilities.exclude_offsets_from_surface`.
            collapse_rule (str, optional): `"link_condition"` is what
                `get_link_checked_algorithm` returns. Defaults to `"weld"`, the
                unconditional-merge rule.

        Deprecated:
            Since 0.5.0, and removed in 0.6.0. This is an exact synonym of
            `get_junction_protected_algorithm(min_distance=..., print_info=...,
            exclude_offsets_from_surface=True, spline_order=..., guard_duplicate_faces=...,
            collapse_rule=...)`: same code path, same mesh, bit for bit. Call that instead.
        """
        warnings.warn(
            "get_offset_excluded_algorithm is deprecated since 0.5.0 and will be removed in 0.6.0; it is an "
            "exact synonym of get_junction_protected_algorithm(min_distance=..., print_info=..., "
            "exclude_offsets_from_surface=True, spline_order=..., guard_duplicate_faces=..., collapse_rule=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        return MeshReconstructionAlgorithmFactory.get_junction_protected_algorithm(
            min_distance=min_distance,
            print_info=print_info,
            spline_order=spline_order,
            exclude_offsets_from_surface=True,
            guard_duplicate_faces=guard_duplicate_faces,
            collapse_rule=collapse_rule,
        )

    @staticmethod
    def get_link_checked_algorithm(
        min_distance: int = 3,
        print_info: bool = False,
        spline_order: int = 1,
    ) -> MeshReconstructionAlgorithm:
        """Return the "revised meshing" variant: junction-protected with a topology-preserving offset collapse.

        **This was `get_default_algorithm` for a time**, when the default reverted to the
        dithered configuration because end-to-end tension-inference accuracy is worse with
        this pipeline than with the dithered one on every method measured, despite its
        better mesh quality in isolation (interface-area bias, sliver fraction, junction
        placement, topology -- see `get_default_algorithm`'s docstring for the numbers). It
        is deterministic, junction-protected and topologically corrected relative to the
        dithered configuration, and stays fully reachable under this name -- chosen
        deliberately over a new alias, since "link-checked" already names the mechanism (a
        link-condition-checked, topology-preserving offset collapse) rather than the phase
        that produced it. Use it when mesh geometry and topology matter more than today's
        downstream tension-inference accuracy. It stays a named getter so that the
        corresponding benchmark records, keyed on the variant name
        `offset_excluded_linkcheck`, remain readable.

        Identical to `get_offset_excluded_algorithm` except that each offset is removed from
        the surface by a **link-condition-checked edge contraction** rather than an
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
        """
        # Called directly rather than through the deprecated get_offset_excluded_algorithm, with
        # exactly the arguments that synonym would pass (guard_duplicate_faces keeps its default).
        return MeshReconstructionAlgorithmFactory.get_junction_protected_algorithm(
            min_distance=min_distance,
            print_info=print_info,
            spline_order=spline_order,
            exclude_offsets_from_surface=True,
            collapse_rule=LINK_CONDITION,
        )

    @staticmethod
    def get_cubic_score_algorithm(min_distance: int = 3, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the cubic-interpolation variant: junction-protected, with cubic B-spline EDT interpolation.

        Identical to `get_offset_included_algorithm` except that the watershed's face scores
        are read from a `C2` cubic B-spline of the EDT instead of a `C0` trilinear one
        (`dw3d.score_computation`). Opt-in: it changes the scores, hence the labelling,
        hence the mesh, so it carries its own golden masters until a benchmark justifies
        flipping the default again.

        **Note it composes with `offset_included`, not with the current default.** It was
        written when `offset_included` *was* the default and its golden masters and 51-case
        records were measured in that configuration; re-pointing it at the current default
        would silently invalidate them. Composing it with the link-checked exclusion is
        reachable (`get_link_checked_algorithm(spline_order=3)`) and has **never been
        benchmarked**; whether that composition helps is an open question.

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 3.
            print_info (bool, optional): Print algorithm details while executing.

        Deprecated:
            Since 0.5.0, and removed in 0.6.0. This is an exact synonym of
            `get_junction_protected_algorithm(min_distance=...,
            print_info=..., spline_order=3)`: same code path, same mesh, bit for bit. Call that instead.
        """
        warnings.warn(
            "get_cubic_score_algorithm is deprecated since 0.5.0 and will be removed in 0.6.0; it is an "
            "exact synonym of get_junction_protected_algorithm(min_distance=..., print_info=..., spline_order=3).",
            DeprecationWarning,
            stacklevel=2,
        )
        return MeshReconstructionAlgorithmFactory.get_junction_protected_algorithm(
            min_distance=min_distance,
            print_info=print_info,
            spline_order=3,
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
