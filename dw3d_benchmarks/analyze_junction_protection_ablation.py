#!/usr/bin/env python
"""Compare the deterministic / boundary-layer / junction-protected benchmark record sets over the 51 cases.

Reads `benchmarks/baseline/<variant>/` (or `benchmarks/baseline/` for the current default,
which is now the link-checked configuration, **not** the boundary-layer + junction-protected
configuration these tables were written against -- use `--variants ... offset_included ...`
for that) and prints the table junction protection is judged on, plus the ablation and
shell-coarsening context. It computes nothing new — every number here comes out of a
committed record — so the comparison tables can be regenerated from the repository at any
time.

The acceptance bar the boundary layer and junction protection are judged against **as a
pair** (the boundary layer alone took valence >= 4 edges from 133 to 520 and abnormal
non-manifold edges from 5 to 327, so it is not judged on its own):

1. valence-4 edges and abnormal non-manifold edges at or below the deterministic level (133 / 5);
2. all-surface tets < 15 %, slivers not regressed;
3. junction-length and junction-angle error improved >= 20 % vs deterministic;
4. combined boundary-layer + junction-protection wall time within 20 % of deterministic.

Note on the degeneracy metric: **`score gaps < 1e-5` is the primary figure.** The `< 1e-3`
one is face-count-confounded (a variant that doubles the face count shrinks consecutive gaps
mechanically), so it is reported here only in the density-normalised form
`fraction_below_1e-3 / n_faces` relative to the deterministic configuration, alongside the
raw value for continuity.

Usage::

    python -m benchmarks.analyze_junction_protection_ablation
    python -m benchmarks.analyze_junction_protection_ablation --variants deterministic boundary_layer junction_protected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "benchmarks" / "baseline"
DEFAULT_VARIANTS = ("deterministic", "boundary_layer", "junction_protected", "junction_protected_plan")
LABELS = {
    "default": "the current default (link-checked)",
    "deterministic": "deterministic (pre-boundary-layer default)",
    "offset_included": "boundary layer + junction protection (a former default)",
    "boundary_layer": "boundary layer",
    "junction_protected": "boundary layer + junction protection",
    "junction_protected_noweights": "boundary layer + junction protection, no weights",
    "junction_protected_plan": "boundary layer + junction protection, plan-literal",
    "v0_3": "v0.3",
}


# See `benchmarks/analyze_offset_exclusion.py` for why this alias exists: `offset_included`
# and `junction_protected` are one configuration sharing one record directory, and since the
# link-checked configuration became the default, `default` is no longer either of them.
_VARIANT_DIR_ALIASES = {"offset_included": "junction_protected"}


def load_variant(variant: str) -> dict[str, dict]:
    """Every committed record of one variant, keyed by case stem."""
    directory = BASELINE_DIR if variant == "default" else BASELINE_DIR / _VARIANT_DIR_ALIASES.get(variant, variant)
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*_md*.json"))}


def _median(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else None


def summarise(records: dict[str, dict], reference: dict[str, dict] | None = None) -> dict:
    """Aggregate one variant's records; wall time is a ratio when a reference is supplied."""
    cases = sorted(records)
    summary: dict = {"n_cases": len(cases)}

    summary["valence_geq4_total"] = sum(r["n_valence_geq4_edges"] for r in records.values())
    summary["valence_geq4_4mat_total"] = sum(
        r.get("valence_geq4_split", {}).get("n_valence_geq4_with_4plus_materials", 0) for r in records.values()
    )
    summary["valence_geq4_3mat_total"] = sum(
        r.get("valence_geq4_split", {}).get("n_valence_geq4_with_3_materials", 0) for r in records.values()
    )
    summary["abnormal_total"] = sum(r["n_abnormal_non_manifold_edges"] for r in records.values())
    summary["holes_total"] = sum(r["n_holes"] for r in records.values())

    all_surface = [r["tetrahedron_surface_stats"]["fraction_all_surface_tets"] for r in records.values()]
    slivers = [r["tet_quality"]["sliver_fraction"] for r in records.values()]
    summary["all_surface"] = (min(all_surface), float(np.median(all_surface)), max(all_surface))
    summary["slivers"] = (min(slivers), float(np.median(slivers)), max(slivers))
    summary["all_surface_over_15pct"] = sum(v >= 0.15 for v in all_surface)

    summary["gaps_1e-5"] = _median(
        [r["watershed_score_gaps"]["fraction_score_gaps_below_1e-5"] for r in records.values()],
    )
    summary["gaps_1e-3"] = _median(
        [r["watershed_score_gaps"]["fraction_score_gaps_below_1e-3"] for r in records.values()],
    )
    summary["n_faces"] = _median([r["watershed_score_gaps"]["n_faces"] for r in records.values()])

    summary["wall_time_total"] = sum(r["cost"]["total_wall_time_s"] for r in records.values())
    if reference is not None:
        ratios = [
            records[case]["cost"]["total_wall_time_s"] / reference[case]["cost"]["total_wall_time_s"]
            for case in cases
            if case in reference and reference[case]["cost"]["total_wall_time_s"] > 0
        ]
        summary["wall_time_ratio"] = (min(ratios), float(np.median(ratios)), max(ratios)) if ratios else None

    summary["tesselation_points"] = sum(
        r["point_placement"]["n_interface_points"] + r["point_placement"]["n_interior_points"] for r in records.values()
    )

    ground_truth = [r["ground_truth"] for r in records.values() if "ground_truth" in r]
    summary["n_ground_truth_cases"] = len(ground_truth)
    summary["angle_error_vs_mesh"] = _median(
        [g["angle_error_vs_reference_mesh"]["angle_error_median_deg"] for g in ground_truth],
    )
    summary["angle_error_vs_neumann"] = _median(
        [g["angle_error_vs_neumann"]["angle_error_median_deg"] for g in ground_truth if "angle_error_vs_neumann" in g],
    )
    summary["length_error"] = _median(
        [
            g["length_error_vs_reference_mesh"]["length_error_median"]
            for g in ground_truth
            if "length_error_vs_reference_mesh" in g
        ],
    )
    summary["total_length_ratio"] = _median(
        [
            g["length_error_vs_reference_mesh"]["total_length_ratio"]
            for g in ground_truth
            if "length_error_vs_reference_mesh" in g
        ],
    )
    summary["missing_triple_lines"] = sum(
        g["length_error_vs_reference_mesh"]["n_missing_triple_lines"]
        for g in ground_truth
        if "length_error_vs_reference_mesh" in g
    )

    preservation = [r["junction_preservation"] for r in records.values() if "junction_preservation" in r]
    if preservation:
        summary["mesh_edge_preservation"] = _median([p.get("mesh_edge_preservation") for p in preservation])
        summary["tesselation_edge_preservation"] = _median(
            [p.get("tesselation_edge_preservation") for p in preservation],
        )
        summary["vertex_preservation"] = _median([p.get("vertex_preservation") for p in preservation])
        summary["junction_samples_total"] = sum(p["n_junction_samples"] for p in preservation)
        shells = [p["shell_coarsening"] for p in preservation if "shell_coarsening" in p]
        if shells:
            summary["shell_minima_total"] = sum(s["n_shell_minima"] for s in shells)
            summary["shell_minima_kept_total"] = sum(s["n_shell_minima_kept"] for s in shells)
            summary["interface_minima_total"] = sum(s["n_interface_minima"] for s in shells)
            summary["minima_before_total"] = sum(s["n_minima_before"] for s in shells)
    return summary


def _format(value: float | None, spec: str) -> str:
    return "     -" if value is None else format(value, spec)


def main() -> None:
    """Print the junction-protection comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()

    loaded = {}
    for variant in args.variants:
        records = load_variant(variant)
        if records:
            loaded[variant] = records
        else:
            print(f"note: no records found for variant {variant!r}")
    reference = loaded.get("deterministic")
    summaries = {variant: summarise(records, reference) for variant, records in loaded.items()}

    header = f"{'metric':38s}" + "".join(f"{LABELS.get(v, v):>22s}" for v in summaries)
    print(header)
    print("-" * len(header))

    def row(name: str, fn) -> None:  # noqa: ANN001 - local formatting helper
        print(f"{name:38s}" + "".join(f"{fn(s):>22s}" for s in summaries.values()))

    row("cases", lambda s: str(s["n_cases"]))
    row("valence>=4 edges (total)", lambda s: str(s["valence_geq4_total"]))
    row("  ... with >=4 materials", lambda s: str(s["valence_geq4_4mat_total"]))
    row("  ... with 3 materials (pinched)", lambda s: str(s["valence_geq4_3mat_total"]))
    row("abnormal non-manifold edges (total)", lambda s: str(s["abnormal_total"]))
    row("holes (total)", lambda s: str(s["holes_total"]))
    row("all-surface tets min/med/max %", lambda s: "/".join(f"{v * 100:.1f}" for v in s["all_surface"]))
    row("  cases >= 15 %", lambda s: str(s["all_surface_over_15pct"]))
    row("slivers min/med/max %", lambda s: "/".join(f"{v * 100:.1f}" for v in s["slivers"]))
    row("score gaps < 1e-5 (median) [PRIMARY]", lambda s: _format(s["gaps_1e-5"], ".3f"))
    row("score gaps < 1e-3 (median, confounded)", lambda s: _format(s["gaps_1e-3"], ".3f"))
    row("tesselation faces (median)", lambda s: _format(s["n_faces"], ".0f"))
    row("tesselation points (total)", lambda s: str(s["tesselation_points"]))
    row(
        "wall time ratio min/med/max",
        lambda s: "/".join(f"{v:.2f}" for v in s["wall_time_ratio"]) if s.get("wall_time_ratio") else "-",
    )
    print()
    row("GT cases", lambda s: str(s["n_ground_truth_cases"]))
    row("junction ANGLE err vs GT mesh (deg)", lambda s: _format(s["angle_error_vs_mesh"], ".2f"))
    row("junction ANGLE err vs Neumann (deg)", lambda s: _format(s["angle_error_vs_neumann"], ".2f"))
    row("junction LENGTH err vs GT (median)", lambda s: _format(s["length_error"], ".4f"))
    row("total junction length / GT", lambda s: _format(s["total_length_ratio"], ".4f"))
    row("triple lines missed (total)", lambda s: str(s["missing_triple_lines"]))
    print()
    row("junction samples (total)", lambda s: str(s.get("junction_samples_total", "-")))
    row("preservation: vertex (median)", lambda s: _format(s.get("vertex_preservation"), ".4f"))
    row("preservation: tesselation edge", lambda s: _format(s.get("tesselation_edge_preservation"), ".4f"))
    row("preservation: MESH edge (median)", lambda s: _format(s.get("mesh_edge_preservation"), ".4f"))
    row("minima: total before shell coarsening", lambda s: str(s.get("minima_before_total", "-")))
    row("  ... on a real interface (|L|>=2)", lambda s: str(s.get("interface_minima_total", "-")))
    row(
        "  ... shell (|L|==1), kept",
        lambda s: f"{s.get('shell_minima_kept_total', '-')} / {s.get('shell_minima_total', '-')}",
    )

    if reference is not None and "deterministic" in summaries:
        base = summaries["deterministic"]
        print("\nRelative to the deterministic configuration (improvement is positive):")
        for variant, summary in summaries.items():
            if variant == "deterministic":
                continue
            parts = []
            for name, key in (
                ("angle vs mesh", "angle_error_vs_mesh"),
                ("angle vs Neumann", "angle_error_vs_neumann"),
                ("length", "length_error"),
            ):
                if base[key] and summary[key] is not None:
                    parts.append(f"{name} {100 * (base[key] - summary[key]) / base[key]:+6.1f} %")
            print(f"  {LABELS.get(variant, variant):22s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
