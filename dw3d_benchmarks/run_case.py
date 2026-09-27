#!/usr/bin/env python
r"""CLI: emit one JSON row of dw3d reconstruction metrics for a segmentation mask.

Takes a mask and, optionally, a ground-truth `.rec`.

No dw3d algorithm behaviour is exercised beyond the factory getters in `VARIANT_GETTERS`
— this only *measures* them.

Usage:
    uv run python benchmarks/run_case.py --mask data/Images/3.tif --min-distance 3
    uv run python benchmarks/run_case.py --mask data/Images/3.tif --min-distance 3 \\
        --with-cost --with-determinism --n-seeds 20 --output benchmarks/baseline/3_md3.json
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks import metrics as m
from dw3d import (
    get_boundary_layer_mesh_reconstruction_algorithm,
    get_default_mesh_reconstruction_algorithm,
    get_deterministic_mesh_reconstruction_algorithm,
    get_dithered_mesh_reconstruction_algorithm,
    get_junction_protected_mesh_reconstruction_algorithm,
    get_link_checked_mesh_reconstruction_algorithm,
)
from dw3d.io import load_rec

# Map a variant name to its factory getter, shared by run_case / profiling / run_baseline.
# `LEGACY_VARIANT_ALIASES` below keeps the old, phase-numbered variant strings working for
# anyone with them in a shell script or notebook.
VARIANT_GETTERS = {
    # The default is `dithered` (get_default_algorithm returns get_dithered_algorithm).
    # `offset_excluded_linkcheck` (boundary layer + junction protection + link-checked offset
    # exclusion) was the default for a time and was reverted; all three names are kept
    # because the 51-case records are keyed on them, and the offset-exclusion comparisons
    # read them by name. `benchmarks/baseline/` holds the default's records, and
    # `benchmarks/baseline/offset_excluded_linkcheck/` the pre-flip run they must
    # reproduce -- which is the check that the flip is a re-pointing and not a re-tuning.
    "default": get_default_mesh_reconstruction_algorithm,
    "dithered": get_dithered_mesh_reconstruction_algorithm,
    # The deterministic default, the baseline the boundary-layer, regular-triangulation and
    # junction-protection work's acceptance criteria are stated against.
    "deterministic": get_deterministic_mesh_reconstruction_algorithm,
    "boundary_layer": get_boundary_layer_mesh_reconstruction_algorithm,
    # Boundary layer + junction protection: a former default, and the configuration that
    # carries the +15.7 % interface-area regression. Same code path as `junction_protected`;
    # the separate name exists so downstream work can benchmark the regression by name. Its
    # 51-case records are `benchmarks/baseline/junction_protected/` -- they are the same
    # configuration, so they are not duplicated.
    "offset_included": get_junction_protected_mesh_reconstruction_algorithm,
    # The junction-protection recommended configuration: junction samples with protecting
    # balls, a regular triangulation, shell coarsening at 3x, and the junction boundary
    # layer off.
    "junction_protected": get_junction_protected_mesh_reconstruction_algorithm,
    # Junction protection with the regular triangulation replaced by plain Delaunay on the
    # *same* point set: the ablation that isolates what the weights themselves contribute.
    "junction_protected_noweights": partial(
        get_junction_protected_mesh_reconstruction_algorithm,
        use_weights=False,
    ),
    # Boundary layer + junction protection with a cubic B-spline EDT interpolant for the
    # watershed scores. Composes with `offset_included`, not with the current default -- it
    # predates the link-checked flip and its records were measured in that configuration.
    # Composing it with link-checked offset exclusion is unbenchmarked.
    "cubic_score": partial(get_junction_protected_mesh_reconstruction_algorithm, spline_order=3),
    # Boundary layer + junction protection, with the boundary-layer offsets kept in the
    # tesselation and out of the extracted surface, by an unconditional weld. Same points,
    # same tesselation, same labelling -- the surface extraction is the only difference,
    # which is what makes the comparison attributable. Kept as the ablation whose topology
    # regression motivated the link-condition-checked collapse.
    "offset_excluded": partial(get_junction_protected_mesh_reconstruction_algorithm, exclude_offsets_from_surface=True),
    # The same exclusion, decided by a link-condition-checked edge collapse instead of an
    # unconditional weld, so it cannot create the non-manifold topology the weld regressed
    # on. Same candidate merges as `offset_excluded`; only the acceptance test differs.
    "offset_excluded_linkcheck": get_link_checked_mesh_reconstruction_algorithm,
    # Offset exclusion composed with the cubic score field, the two independent levers
    # together.
    "offset_excluded_cubic": partial(
        get_junction_protected_mesh_reconstruction_algorithm,
        exclude_offsets_from_surface=True,
        spline_order=3,
    ),
    # Offset exclusion with the duplicate-face guard off: excludes ~6 % more offsets, at the
    # cost of the boundary edges the guard exists to prevent. The ablation that puts that
    # trade on record.
    "offset_excluded_unguarded": partial(
        get_junction_protected_mesh_reconstruction_algorithm,
        exclude_offsets_from_surface=True,
        guard_duplicate_faces=False,
    ),
    # The plan's literal junction-protection step 5: junction boundary layer on, shell
    # coarsening off. Kept benchmarked so the falsified variant is on the record with
    # numbers, not just prose.
    "junction_protected_plan": partial(
        get_junction_protected_mesh_reconstruction_algorithm,
        shell_coarsening=1,
        junction_boundary_layer=True,
    ),
}

# Old variant strings, kept working (undocumented) for anyone with them in a saved command.
LEGACY_VARIANT_ALIASES = {
    "v0_3": "dithered",
    "a1b": "deterministic",
    "a3_a5": "offset_included",
}
for _legacy_name, _canonical_name in LEGACY_VARIANT_ALIASES.items():
    VARIANT_GETTERS[_legacy_name] = VARIANT_GETTERS[_canonical_name]


def _stringify_keys(d: dict) -> dict:
    """JSON object keys must be strings; our metrics use int/tuple keys (label ids/tuples)."""
    return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): v for k, v in d.items()}


def _junction_records(
    mask: np.ndarray,
    algo: object,
    min_distance: int,
    points: np.ndarray,
    triangles: np.ndarray,
    _variant: str,
) -> dict:
    """The junction-preservation block, for the variants that sample junctions.

    Recomputes the point-placement families (deterministic, no RNG, so this cannot perturb
    the finished reconstruction) to recover the sampled junction network, then measures how
    much of it survived. Empty for the variants that place no junction samples.
    """
    # Keyed on the *algorithm*, not on the variant name: the default has at times been the
    # junction-protected scheme, so a name test would silently drop this whole metric block
    # from the default's records and make them incomparable with the junction-protection
    # work's.
    from dw3d.points_on_edt import junction_protected_families, peak_local_points_junction_protected

    if getattr(algo.point_placing_function, "func", None) is not peak_local_points_junction_protected:
        return {}

    keywords = getattr(algo.point_placing_function, "keywords", {})
    families = junction_protected_families(
        mask,
        algo._edt_image,
        min_distance,
        delta=keywords.get("delta"),
        junction_spacing=keywords.get("junction_spacing"),
        shell_coarsening=keywords.get("shell_coarsening", 1),
        protect_radius=keywords.get("protect_radius"),
        junction_delta=keywords.get("junction_delta"),
        protect_junctions=keywords.get("protect_junctions", True),
        junction_boundary_layer=keywords.get("junction_boundary_layer", True),
    )
    record = m.junction_preservation_stats(families, algo._tesselation_graph, points, triangles)
    record["point_families"] = {
        "n_corners": len(families["corners"]),
        "n_maxima": len(families["maxima"]),
        "n_minima": len(families["minima"]),
        "n_interface_offsets": len(families["offsets"]),
        "n_junction_samples": len(families["junction_points"]),
        "n_junction_offsets": len(families["junction_offsets"]),
    }
    record["junction_detection"] = families["junction"]
    record["shell_coarsening"] = families["shell"]
    return record


def _ground_truth_records(
    mask: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    measured_lengths: dict,
    ground_truth_rec: Path,
    tensions_path: Path | None,
) -> dict:
    """Junction-length and junction-angle error against the ground truth.

    Angles are compared **directly**: a similarity preserves them, so the `.rec`'s different
    frame does not matter. Lengths are compared after registering the ground truth into the
    mask's voxel frame (`m.similarity_to_mask_frame`) — a fit that depends only on the mask
    and the reference, never on the reconstruction being scored.

    When the case's ground-truth tensions are available, the angles are *also* compared
    against Neumann's law, which is analytic and needs neither a reference mesh nor a
    registration. The two references answer different questions: the `.rec` says "does this
    reconstruct the simulated mesh", Neumann says "is this a mechanically admissible foam".
    """
    gt_points, gt_triangles, gt_labels = load_rec(ground_truth_rec)
    measured_angles = m.attributed_triple_line_angles(points, triangles, labels)
    reference_angles = m.attributed_triple_line_angles(gt_points, gt_triangles, gt_labels)

    record: dict = {
        "n_points": len(gt_points),
        "n_triangles": len(gt_triangles),
        "angle_error_vs_reference_mesh": m.compare_angle_dicts(measured_angles, reference_angles),
    }

    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    record["registration"] = {k: v for k, v in registration.items() if k not in ("rotation", "translation")}
    if registration.get("scale") is not None:
        scaled = registration["scale"] * gt_points
        reference_lengths = m.edge_topology_stats(scaled, gt_triangles, gt_labels)["triple_line_lengths"]
        record["length_error_vs_reference_mesh"] = m.junction_length_error(measured_lengths, reference_lengths)

    if tensions_path is not None and tensions_path.exists():
        tensions = np.load(tensions_path, allow_pickle=True).item()
        neumann = m.neumann_angles_from_tensions(tensions)
        record["angle_error_vs_neumann"] = m.compare_angle_dicts(measured_angles, neumann)
        record["reference_mesh_angle_error_vs_neumann"] = m.compare_angle_dicts(reference_angles, neumann)
    return record


def run_case(
    mask: np.ndarray,
    min_distance: int,
    ground_truth_rec: Path | None = None,
    variant: str = "default",
    tensions_path: Path | None = None,
) -> dict:
    """Reconstruct once and compute every single-reconstruction harness metric.

    That is geometry, topology, mesh quality and watershed health. Cost and determinism
    are separate, opt-in, more expensive measurements that need repeated reconstructions
    (see `--with-cost` / `--with-determinism`).

    `variant` selects the algorithm: `"default"` (boundary layer + junction protection +
    link-checked offset exclusion, the same configuration as
    `"offset_excluded_linkcheck"`), `"offset_included"` (a former default: boundary layer +
    junction protection), `"deterministic"` (an earlier default: no dither, no boundary
    layer), `"dithered"` (the original dithered placement), `"boundary_layer"` (the
    boundary layer alone) or one of the `junction_protected*` / `offset_excluded*`
    ablations, so all of them can be measured on identical metrics.
    """
    getter = VARIANT_GETTERS[variant]
    algo = getter(min_distance=min_distance, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    tesselation_graph = algo._tesselation_graph

    point_placement = m.point_placement_counts(
        algo._edt_image,
        min_distance,
        point_placing_function=algo.point_placing_function,
        segmented_image=mask,
    )
    point_classes = m.classify_tesselation_points(len(tesselation_graph.vertices), point_placement["n_interior_points"])
    surface_stats = m.tetrahedron_surface_stats(tesselation_graph, point_classes)
    quality = m.tetrahedron_quality(tesselation_graph)
    sliver = m.sliver_stats(quality, surface_stats.pop("_all_surface_mask"))
    score_gaps = m.score_gap_stats(tesselation_graph)

    edge_stats = m.edge_topology_stats(points, triangles, labels)
    dihedral = m.dihedral_angle_stats_per_triple_line(points, triangles, labels)
    interface_areas = m.interface_areas(points, triangles, labels)
    cell_volumes = m.cell_volumes_from_tetrahedra(tesselation_graph, algo._map_label_to_nodes_ids)
    triangle_quality = m.triangle_quality_stats(points, triangles)
    components_euler = m.connected_components_and_euler_per_cell(triangles, labels)
    n_abnormal_edges = m.abnormal_non_manifold_edge_count(points, triangles, labels)
    valence_split = m.valence_geq4_edges_by_material_count(triangles, labels)
    junction = _junction_records(mask, algo, min_distance, points, triangles, variant)

    record = {
        "n_points": len(points),
        "n_triangles": len(triangles),
        # The offset-exclusion work's own report, empty for every variant that keeps its
        # offsets in the surface. Recorded per case so the guard's cost
        # (`n_merges_refused`) and the flap area removed can be aggregated over the 51
        # rather than quoted from one image.
        "surface_exclusion": getattr(algo, "_surface_exclusion_info", None) or {},
        "point_placement": point_placement,
        "tetrahedron_surface_stats": surface_stats,
        "tet_quality": sliver,
        "watershed_score_gaps": score_gaps,
        "interface_areas": _stringify_keys(interface_areas),
        "cell_volumes": _stringify_keys(cell_volumes),
        "triple_line_lengths": _stringify_keys(edge_stats["triple_line_lengths"]),
        "dihedral_angle_stats": dihedral,
        "valence_histogram": {str(k): v for k, v in edge_stats["valence_histogram"].items()},
        "n_holes": edge_stats["n_holes"],
        "n_valence_geq4_edges": edge_stats["n_valence_geq4_edges"],
        "length_valence_geq4_edges": edge_stats["length_valence_geq4_edges"],
        "triangle_quality": triangle_quality,
        "connected_components_and_euler_per_cell": {str(k): v for k, v in components_euler.items()},
        "n_abnormal_non_manifold_edges": n_abnormal_edges,
        "valence_geq4_split": valence_split,
    }
    if junction:
        record["junction_preservation"] = junction

    if ground_truth_rec is not None:
        record["ground_truth"] = _ground_truth_records(
            mask,
            points,
            triangles,
            labels,
            edge_stats["triple_line_lengths"],
            ground_truth_rec,
            tensions_path,
        )

    return record


def main() -> None:
    """Run one case from the command line and print its JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="Path to a segmentation mask (.tif)")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional ground-truth .rec mesh")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--with-cost", action="store_true", help="Also measure wall time / peak RSS")
    parser.add_argument("--with-determinism", action="store_true", help="Also sweep dither seeds (expensive)")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--tensions", type=Path, default=None, help="Optional ground-truth dict_tensions.npy")
    parser.add_argument("--variant", choices=tuple(VARIANT_GETTERS), default="default")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    mask = io.imread(args.mask)
    record = run_case(mask, args.min_distance, args.ground_truth, variant=args.variant, tensions_path=args.tensions)
    record["case"] = args.mask.stem
    record["min_distance"] = args.min_distance
    record["variant"] = args.variant

    if args.with_cost:
        from dw3d_benchmarks.profiling import cost_profile

        record["cost"] = cost_profile(mask, args.min_distance, variant=args.variant)

    if args.with_determinism:
        from dw3d_benchmarks.profiling import determinism_profile

        record["determinism"] = determinism_profile(mask, args.min_distance, n_seeds=args.n_seeds)

    output_text = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
