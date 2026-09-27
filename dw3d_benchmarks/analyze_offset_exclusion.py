#!/usr/bin/env python
"""Aggregate the offset-exclusion comparison over the 51 committed baseline records.

The offset-exclusion work keeps the boundary layer's offsets in the tesselation and out of
the extracted surface. Because it changes **only** the extraction — same EDT, same points,
same regular triangulation, same watershed labelling — every difference reported here is
attributable to that one change, and the tesselation-side metrics (all-surface tets,
slivers, score gaps, cell volumes) must come out *identical*. This script checks that they
do rather than assuming it, and prints the rest as a before/after table.

Counts are totals over the case set; fractions and errors are medians over it, the same
convention as the junction-protection comparison tables, so the two can be read side by side.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.analyze_offset_exclusion \
        --variants offset_included offset_excluded offset_excluded_linkcheck \
        --out benchmarks/baseline/summary_default_offset_excluded_link_checked.json

**Read the variant names, not the word "default".** The default resolves to
`benchmarks/baseline/`, which now holds the boundary layer + junction protection +
**link-checked offset exclusion**. The "before" column of every table of the original
offset-exclusion comparison is boundary layer + junction protection, which is `offset_included` (aliased
to the `junction_protected` records). Passing `default` as the first variant today compares
link-checked offset exclusion against itself and reports zeros everywhere -- true, and
useless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "benchmarks" / "baseline"


# `offset_included` and `junction_protected` are one configuration, so they share one
# record directory. The alias exists because the default no longer means boundary layer +
# junction protection alone -- it means that plus link-checked offset exclusion -- and every
# table of the original offset-exclusion comparison was written against boundary layer +
# junction protection as the "before". Reproducing those tables therefore needs
# `--variants offset_included ...` where they said `default`.
_VARIANT_DIR_ALIASES = {"offset_included": "junction_protected"}


def _variant_dir(variant: str) -> Path:
    if variant == "default":
        return BASELINE_DIR
    return BASELINE_DIR / _VARIANT_DIR_ALIASES.get(variant, variant)


def load_records(variant: str) -> dict[str, dict]:
    """Every committed record for a variant, keyed by case stem."""
    directory = _variant_dir(variant)
    records = {}
    for path in sorted(directory.glob("*_md3.json")):
        record = json.loads(path.read_text())
        records[record["case"]] = record
    return records


def _median(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.median(present)) if present else None


def _get(record: dict, path: str, default: object = None) -> object:
    node = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def summarise(records: dict[str, dict]) -> dict:
    """The metric block for one variant, over whatever cases it has."""
    values = list(records.values())
    with_gt = [r for r in values if _get(r, "ground_truth.angle_error_vs_neumann") is not None]
    return {
        "n_cases": len(values),
        "n_cases_with_ground_truth": len(with_gt),
        "total_mesh_points": sum(r["n_points"] for r in values),
        "total_triangles": sum(r["n_triangles"] for r in values),
        "total_valence_geq4_edges": sum(r["n_valence_geq4_edges"] for r in values),
        "total_valence_geq4_with_3_materials": sum(
            _get(r, "valence_geq4_split.n_valence_geq4_with_3_materials", 0) for r in values
        ),
        "total_valence_geq4_with_4plus_materials": sum(
            _get(r, "valence_geq4_split.n_valence_geq4_with_4plus_materials", 0) for r in values
        ),
        "total_holes": sum(r["n_holes"] for r in values),
        "total_abnormal_non_manifold_edges": sum(r["n_abnormal_non_manifold_edges"] for r in values),
        "median_all_surface_tet_fraction": _median(
            [_get(r, "tetrahedron_surface_stats.fraction_all_surface_tets") for r in values],
        ),
        "median_sliver_fraction": _median([_get(r, "tet_quality.sliver_fraction") for r in values]),
        "median_score_gap_fraction_lt_1e5": _median(
            [_get(r, "watershed_score_gaps.fraction_score_gaps_below_1e-5") for r in values],
        ),
        "median_angle_error_vs_neumann_deg": _median(
            [_get(r, "ground_truth.angle_error_vs_neumann.angle_error_mean_deg") for r in with_gt],
        ),
        "median_angle_error_vs_reference_deg": _median(
            [_get(r, "ground_truth.angle_error_vs_reference_mesh.angle_error_mean_deg") for r in with_gt],
        ),
        "median_reference_angle_error_vs_neumann_deg": _median(
            [_get(r, "ground_truth.reference_mesh_angle_error_vs_neumann.angle_error_mean_deg") for r in with_gt],
        ),
        "median_junction_length_error": _median(
            [_get(r, "ground_truth.length_error_vs_reference_mesh.length_error_median") for r in with_gt],
        ),
        "median_total_junction_length_ratio": _median(
            [_get(r, "ground_truth.length_error_vs_reference_mesh.total_length_ratio") for r in with_gt],
        ),
        "median_mesh_edge_junction_preservation": _median(
            [_get(r, "junction_preservation.mesh_edge_preservation") for r in values],
        ),
        "median_triangle_min_angle_deg": _median([_get(r, "triangle_quality.min_angle_median_deg") for r in values]),
        "median_triangle_min_angle_p10_deg": _median([_get(r, "triangle_quality.min_angle_p10_deg") for r in values]),
        "total_wall_time_s": sum(_get(r, "cost.total_wall_time_s", 0.0) for r in values),
        # The offset-exclusion work's own report
        "total_offset_vertices_removed": sum(_get(r, "surface_exclusion.n_offset_vertices_removed", 0) for r in values),
        "total_merges_refused": sum(_get(r, "surface_exclusion.n_merges_refused", 0) for r in values),
        "max_guard_rounds": max([_get(r, "surface_exclusion.n_guard_rounds", 0) for r in values], default=0),
        "total_duplicate_conflicts": sum(_get(r, "surface_exclusion.n_duplicate_conflicts", 0) for r in values),
        "total_triangles_collapsed": sum(_get(r, "surface_exclusion.n_triangles_collapsed", 0) for r in values),
        # The link-condition-checked collapse: which of the four collapse tests refused the merges that were refused,
        # and how often each fired at all (a test that only ever fires on candidates a later
        # round accepts is doing no work, and should be visible as such rather than inferred).
        **{
            f"total_refused_{reason}": sum(_get(r, f"surface_exclusion.n_refused_{reason}", 0) for r in values)
            for reason in REFUSAL_REASONS
        },
        **{
            f"total_fired_{reason}": sum(_get(r, f"surface_exclusion.n_fired_{reason}", 0) for r in values)
            for reason in REFUSAL_REASONS
        },
    }


REFUSAL_REASONS = ("no_edge", "link_vertices", "link_edges", "valence")


def topology_bound_checks(baseline: dict[str, dict], other: dict[str, dict]) -> dict:
    """Per-case check that no topology count rose — the link-condition-checked collapse's
    acceptance criterion.

    The summary table compares *totals*, which can hide a case that got worse behind one that
    got better. Its contract is stronger and per-case: no edge valence can increase
    anywhere, so on **every** case the hole count, the valence->=4 count and the abnormal
    non-manifold edge count must each be at or below the baseline's.

    Each count is reported with `n_cases_where_<metric>_was_present`, for the reason
    `unchanged_checks` gives: a metric-path typo makes both sides `None`, `None <= None`
    raises but `None != None` is False, and a comparison that never ran looks like a
    comparison that passed.
    """
    shared = sorted(set(baseline) & set(other))
    metrics = (
        ("holes", "n_holes"),
        ("valence_geq4_edges", "n_valence_geq4_edges"),
        ("abnormal_non_manifold_edges", "n_abnormal_non_manifold_edges"),
    )
    result: dict[str, object] = {"n_cases_compared": len(shared)}
    for name, _key in metrics:
        result[f"n_cases_where_{name}_rose"] = 0
        result[f"n_cases_where_{name}_fell"] = 0
        result[f"n_cases_where_{name}_was_present"] = 0
        result[f"worst_increase_in_{name}"] = 0
    for case in shared:
        for name, key in metrics:
            left, right = baseline[case].get(key), other[case].get(key)
            if left is None or right is None:
                continue
            result[f"n_cases_where_{name}_was_present"] += 1
            if right > left:
                result[f"n_cases_where_{name}_rose"] += 1
                result[f"worst_increase_in_{name}"] = max(result[f"worst_increase_in_{name}"], right - left)
            elif right < left:
                result[f"n_cases_where_{name}_fell"] += 1
    return result


def unchanged_checks(baseline: dict[str, dict], other: dict[str, dict]) -> dict:
    """The metrics offset exclusion must leave *bit-identical*, since it does not touch the tesselation.

    Reported as a check rather than assumed: if any of these moved, the comparison would not
    be attributable to the extraction and every number in the comparison would be suspect.

    **The check counts its own non-vacuity**, and it does so because it was caught being
    vacuous: a metric path typo makes `_get` return `None` on both sides, `None == None`, and
    the comparison passes while measuring nothing. `n_cases_where_<metric>_was_present` must
    equal `n_cases_compared` for the corresponding zero to mean anything.
    """
    shared = sorted(set(baseline) & set(other))
    worst = {
        "n_cases_compared": len(shared),
        "max_cell_volume_relative_difference": 0.0,
        "n_cases_with_differing_all_surface_tets": 0,
        "n_cases_with_differing_sliver_fraction": 0,
        "n_cases_with_differing_score_gaps": 0,
        "n_cases_with_differing_tesselation_point_count": 0,
    }
    checked = (
        ("all_surface_tets", "tetrahedron_surface_stats.fraction_all_surface_tets"),
        ("sliver_fraction", "tet_quality.sliver_fraction"),
        ("score_gaps", "watershed_score_gaps.fraction_score_gaps_below_1e-5"),
        ("tesselation_point_count", "point_placement.n_interface_points"),
    )
    for name, _path in checked:
        worst[f"n_cases_where_{name}_was_present"] = 0
    for case in shared:
        left, right = baseline[case], other[case]
        for name, path in checked:
            left_value, right_value = _get(left, path), _get(right, path)
            if left_value is not None and right_value is not None:
                worst[f"n_cases_where_{name}_was_present"] += 1
            if left_value != right_value:
                worst[f"n_cases_with_differing_{name}"] += 1
        left_volumes, right_volumes = left.get("cell_volumes", {}), right.get("cell_volumes", {})
        for label in set(left_volumes) & set(right_volumes):
            denominator = abs(left_volumes[label]) or 1.0
            difference = abs(left_volumes[label] - right_volumes[label]) / denominator
            worst["max_cell_volume_relative_difference"] = max(
                worst["max_cell_volume_relative_difference"],
                difference,
            )
    return worst


def interface_area_error(records: dict[str, dict], reference: dict[str, dict]) -> float | None:
    """Median relative interface-area difference against the *default*'s areas.

    Not an accuracy metric on its own — the ground-truth `.rec` is a different discretisation,
    so `benchmarks/metrics.py` deliberately keys everything by label tuples and does not claim
    an absolute area error. This reports how far offset exclusion moves the areas, which is the quantity the
    flap removal is expected to change and the one a reader will ask about.
    """
    ratios = []
    for case, record in records.items():
        if case not in reference:
            continue
        left, right = reference[case].get("interface_areas", {}), record.get("interface_areas", {})
        ratios.extend(right[key] / left[key] for key in set(left) & set(right) if left[key] > 0)
    return float(np.median(ratios)) if ratios else None


ROWS: tuple[tuple[str, str], ...] = (
    ("mesh points (total)", "total_mesh_points"),
    ("triangles (total)", "total_triangles"),
    ("valence->=4 edges (total)", "total_valence_geq4_edges"),
    ("... with >=4 materials", "total_valence_geq4_with_4plus_materials"),
    ("... with exactly 3 materials", "total_valence_geq4_with_3_materials"),
    ("holes / valence-1 edges (total)", "total_holes"),
    ("abnormal non-manifold edges (total)", "total_abnormal_non_manifold_edges"),
    ("all-surface tets (median)", "median_all_surface_tet_fraction"),
    ("slivers (median)", "median_sliver_fraction"),
    ("score gaps < 1e-5 (median)", "median_score_gap_fraction_lt_1e5"),
    ("angle error vs Neumann (median, deg)", "median_angle_error_vs_neumann_deg"),
    ("angle error vs GT mesh (median, deg)", "median_angle_error_vs_reference_deg"),
    ("GT mesh vs Neumann (control, deg)", "median_reference_angle_error_vs_neumann_deg"),
    ("junction length error (median)", "median_junction_length_error"),
    ("total junction length / GT (median)", "median_total_junction_length_ratio"),
    ("junction preservation, mesh edge", "median_mesh_edge_junction_preservation"),
    ("triangle min angle (median, deg)", "median_triangle_min_angle_deg"),
    ("triangle min angle p10 (median, deg)", "median_triangle_min_angle_p10_deg"),
    ("offsets removed from surface (total)", "total_offset_vertices_removed"),
    ("merges refused (total)", "total_merges_refused"),
    ("... refused: not an edge", "total_refused_no_edge"),
    ("... refused: link, vertex part", "total_refused_link_vertices"),
    ("... refused: link, edge part", "total_refused_link_edges"),
    ("... refused: would raise a valence", "total_refused_valence"),
    ("guard / collapse rounds (max)", "max_guard_rounds"),
    ("duplicate label conflicts (total)", "total_duplicate_conflicts"),
    ("triangles collapsed (total)", "total_triangles_collapsed"),
    ("wall time, single pass (total, s)", "total_wall_time_s"),
)


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def main() -> None:
    """Print the offset-exclusion comparison table and optionally write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["a3_a5", "offset_excluded", "offset_excluded_linkcheck"],
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    loaded = {variant: load_records(variant) for variant in args.variants}
    for variant, records in loaded.items():
        if not records:
            print(f"WARNING: no records found for variant {variant} under {_variant_dir(variant)}")
    # Compare on the cases every variant has, so the totals are like for like.
    common = set.intersection(*[set(r) for r in loaded.values()]) if loaded else set()
    summaries = {v: summarise({c: r for c, r in loaded[v].items() if c in common}) for v in args.variants}

    width = max(len(label) for label, _ in ROWS) + 2
    header = "".join(f"{v:>26s}" for v in args.variants)
    print(f"\n{len(common)} cases common to all variants\n")
    print(" " * width + header)
    for label, key in ROWS:
        cells = "".join(f"{_format(summaries[v][key]):>26s}" for v in args.variants)
        print(f"{label:<{width}s}{cells}")

    print("\nMetrics offset exclusion must leave identical (it does not touch the tesselation):")
    checks = {}
    baseline = {c: r for c, r in loaded[args.variants[0]].items() if c in common}
    for variant in args.variants[1:]:
        other = {c: r for c, r in loaded[variant].items() if c in common}
        checks[variant] = unchanged_checks(baseline, other)
        checks[variant]["median_interface_area_ratio_vs_default"] = interface_area_error(other, baseline)
        print(f"  {variant}:")
        for key, value in checks[variant].items():
            print(f"    {key} = {_format(value)}")

    print("\nPer-case topology bounds (link-condition-checked collapse: no count may rise on any case):")
    bounds = {}
    for variant in args.variants[1:]:
        other = {c: r for c, r in loaded[variant].items() if c in common}
        bounds[variant] = topology_bound_checks(baseline, other)
        print(f"  {variant}:")
        for key, value in bounds[variant].items():
            print(f"    {key} = {_format(value)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "n_common_cases": len(common),
                    "summaries": summaries,
                    "unchanged_checks": checks,
                    "topology_bound_checks": bounds,
                },
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
