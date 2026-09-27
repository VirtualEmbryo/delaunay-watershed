#!/usr/bin/env python
"""Build the four mesh-quality-comparison figures from a results JSON. Never re-runs anything.

Figures (PNG for a talk, PDF for a paper), self-explanatory filenames:

1. `scorecard` -- every headline metric x every mesh set, best value per row marked.
2. `paired_per_case` -- one small multiple per metric, mesh sets on x, one faint line per case.
3. `flattening` -- signed line-mean angle error vs (true angle - 120), one panel per mesh set.
4. `geometry_fidelity` -- areas, volumes, curvature: reconstruction vs ground truth, y=x.

All four use only the **core** rows (`min_distance == 3`, the four named configurations);
the extension sweep's rows stay in the same JSON but do not add a fifth figure -- the
comparison is designed around four, no more.

Requires matplotlib: `pip install -e ".[benchmarks]"` (the `[viewing]` extra also installs
Polyscope, which the figures do not need).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import _style

CORE_MIN_DISTANCE = 3


def _version_line(repo_label: str, version: dict) -> str:
    """One repo's line for the figure footer: commit, date, branch, and drift from its remote."""
    if not version.get("available"):
        return f"{repo_label}: version unavailable"
    line = f"{repo_label} {version['short_commit']} ({version['date']}, {version['branch']})"
    pushed = version.get("last_pushed")
    if not pushed:
        return line
    ahead, behind, ref = pushed["commits_ahead_of_remote"], pushed["commits_behind_remote"], pushed["ref"]
    if ahead == 0 and behind == 0:
        return f"{line} = {ref}"
    if pushed["is_ancestor_of_head"]:
        return f"{line}, {ahead} commit(s) ahead of {ref} ({pushed['short_commit']}, unreleased)"
    if ahead == 0:
        return f"{line}, {behind} commit(s) behind {ref} ({pushed['short_commit']}, last pushed)"
    return f"{line}, diverged from {ref} ({pushed['short_commit']}: {ahead} ahead / {behind} behind)"


def version_footer(provenance: dict, mesh_set_sources: dict[str, str] | None = None) -> str:
    """Figure footer naming the code version(s) behind the numbers.

    Two cases, and conflating them would be a lie on the figure:

    * **Configuration comparison** -- every mesh set was built here, by one dw3d commit, so
      that commit *is* the provenance of all columns.
    * **Version comparison** -- the columns are meshes built by *different* dw3d versions and
      then measured by this one. Saying only "dw3d f0b68a7" would imply both arms came from
      f0b68a7. So the footer says "measured by" and lists each arm's own version, which
      `mesh_set_sources` carries from the results JSON.
    """
    dw3d_line = _version_line("dw3d", provenance.get("dw3d_version", {}))
    foambryo_line = _version_line("foambryo (analysis only)", provenance.get("foambryo_version", {}))
    if mesh_set_sources:
        arms = "; ".join(f"{name}: {label}" for name, label in sorted(mesh_set_sources.items()))
        return f"meshes: {arms}  |  measured by {dw3d_line}  |  {foambryo_line}"
    return f"{dw3d_line}  |  {foambryo_line}"


def _mesh_set_sources(records: list[dict]) -> dict[str, str]:
    """Per mesh set, a human label for the dw3d version that *built* its meshes.

    Only meaningful when meshes were loaded from a directory (a version comparison). The
    version itself comes from the `export_manifest.json` that `compare_versions.py --export-to`
    writes beside the per-arm mesh directories, so the figure stays derivable from files on
    disk rather than from anything re-run.
    """
    sources: dict[str, str] = {}
    for record in records:
        source = record.get("mesh_source")
        if not source or source == "built here" or record["mesh_set"] in sources:
            continue
        manifest_path = Path(source).parent / "export_manifest.json"
        label = Path(source).name
        if manifest_path.exists():
            try:
                arm = json.loads(manifest_path.read_text())["arms"].get(record["mesh_set"], {})
                version = arm.get("version") or {}
                label = version.get("describe") or version.get("installed_version") or label
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        sources[record["mesh_set"]] = label
    return sources


def load_core_records(results_path: Path) -> tuple[list[dict], list[str], str]:
    """Load the core (`min_distance == 3`) records, mesh-set order, and version footer text.

    The mesh-set order comes from the manifest, so colour assignment is stable across
    figures and across re-runs.
    """
    data = json.loads(Path(results_path).read_text())
    mesh_sets_in_order = data["manifest"]["mesh_sets"]
    records = [r for r in data["records"] if r["min_distance"] == CORE_MIN_DISTANCE]
    footer = version_footer(data["manifest"].get("provenance", {}), _mesh_set_sources(records))
    return records, mesh_sets_in_order, footer


def _equilibrium(records: list[dict]) -> list[dict]:
    return [r for r in records if r["group"] == "equilibrium"]


def _by_mesh_set(records: list[dict], mesh_sets_in_order: list[str]) -> dict[str, list[dict]]:
    return {ms: [r for r in records if r["mesh_set"] == ms] for ms in mesh_sets_in_order}


# ---------------------------------------------------------------------------
# 1. Scorecard
# ---------------------------------------------------------------------------

# (label, extractor, lower_is_better) -- extractor returns a per-case value or None.
#
# Count-type rows are aggregated by SUM, not median, via `_SUM_METRICS` below. A median over
# cases hides exactly the differences these rows exist to show: measured, dw3d 0.3.6 carries 3
# abnormal non-manifold edges over the cohort (cases 011, 013, 036, one each) and current dw3d
# carries 0 -- but 3 affected cases out of 40 gives a median of 0 on *both* arms, so a
# median-aggregated row reads "0 vs 0, no change" for a defect that was in fact fixed.
#
# Two M1 rows, not one, and each says which statistic it is. The first row used to be labelled
# `M1 |angle err| median`, which reads as the median absolute wedge error but computes
# `|median(signed error)|` -- the magnitude of the **net bias**, which alternating errors of
# opposite sign cancel out of. The accuracy statistic is the second row. They rank
# configurations differently, so the label matters. `median_abs_error_deg` is absent from
# results JSON written before that accuracy row was added, so its extractor returns `None` and the
# cell is left blank rather than crashing on an old file.
_SCORECARD_METRICS: list[tuple[str, Callable, bool]] = [
    (
        "M1 net bias |median signed err| (deg, vs Neumann)",
        lambda r: (
            abs(r["m1_contact_angles"].get("vs_neumann", {}).get("signed_error_median_deg"))
            if r.get("m1_contact_angles", {}).get("vs_neumann")
            else None
        ),
        True,
    ),
    (
        "M1 accuracy median |wedge err| (deg, vs Neumann)",
        lambda r: r.get("m1_contact_angles", {}).get("vs_neumann", {}).get("median_abs_error_deg"),
        True,
    ),
    (
        "M1 flattening slope, differential bias (length-wtd, vs Neumann)",
        lambda r: (r["m1_contact_angles"].get("vs_neumann", {}).get("flattening_slope_length_weighted") or {}).get(
            "slope",
        ),
        True,
    ),
    (
        "M2 |area err| median (%)",
        lambda r: (
            100 * abs(r["m2_interface_areas"].get("signed_relative_error_median"))
            if "signed_relative_error_median" in r.get("m2_interface_areas", {})
            else None
        ),
        True,
    ),
    (
        "M3 |volume err| median, cells (%)",
        lambda r: (
            100 * abs(r["m3_cell_volumes"]["cells"].get("signed_relative_error_median"))
            if r.get("m3_cell_volumes", {}).get("cells", {}).get("signed_relative_error_median") is not None
            else None
        ),
        True,
    ),
    (
        "M4 curvature |rel err| median, above crossover (%)",
        lambda r: (
            100 * abs(median)
            if (median := (r.get("m4_curvature", {}).get("above_crossover_signed_relative_error") or {}).get("median"))
            is not None
            else None
        ),
        True,
    ),
    (
        "M5 tangent error, mean (deg) [provisional]",
        lambda r: r.get("m5_junction_geometry", {}).get("tangent_error_mean_deg"),
        True,
    ),
    (
        "M6 abnormal non-manifold edges (total)",
        lambda r: r.get("m6_validity", {}).get("n_abnormal_non_manifold_edges"),
        True,
    ),
    (
        "M6 malformed triple lines (total)",
        lambda r: r.get("m6_validity", {}).get("n_malformed_triple_line_edges"),
        True,
    ),
    ("M6 sliver fraction", lambda r: r.get("m6_validity", {}).get("sliver_fraction"), True),
    ("M7 wall-clock, median (s)", lambda r: r.get("m7_budget", {}).get("wall_time_s"), True),
    ("M7 triangles per cell, mean", lambda r: r.get("m7_budget", {}).get("triangles_per_cell_mean"), False),
]


#: Rows whose meaningful cohort aggregate is a sum over cases, not a median (see the note on
#: `_SCORECARD_METRICS`). Matched on the row label.
_SUM_METRICS = ("(total)",)


def _aggregate(label: str, values: list[float]) -> float:
    return float(np.sum(values)) if any(tag in label for tag in _SUM_METRICS) else float(np.median(values))


def _scorecard_table(equilibrium: list[dict], mesh_sets_in_order: list[str]) -> np.ndarray:
    """Per-metric, per-mesh-set aggregated value: `_SCORECARD_METRICS` rows x `mesh_sets_in_order` columns."""
    grouped = _by_mesh_set(equilibrium, mesh_sets_in_order)
    table = np.full((len(_SCORECARD_METRICS), len(mesh_sets_in_order)), np.nan)
    for i, (label, extractor, _lib) in enumerate(_SCORECARD_METRICS):
        for j, ms in enumerate(mesh_sets_in_order):
            values = [v for r in grouped[ms] if (v := extractor(r)) is not None and np.isfinite(v)]
            if values:
                table[i, j] = _aggregate(label, values)
    return table


def _scorecard_cell_text_and_colors(table: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
    """Per-cell display text and color, with the best value per row marked in green."""
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []
    for i, (_label, _extractor, lower_is_better) in enumerate(_SCORECARD_METRICS):
        row = table[i]
        finite = row[np.isfinite(row)]
        best = (np.nanmin(row) if lower_is_better else np.nanmax(row)) if finite.size else None
        row_text, row_color = [], []
        for value in row:
            if not np.isfinite(value):
                row_text.append("n/a")
                row_color.append(_style.TEXT_MUTED)
            else:
                row_text.append(f"{value:.3g}")
                is_best = best is not None and np.isclose(value, best)
                row_color.append(_style.GREEN if is_best else _style.TEXT_PRIMARY)
        cell_text.append(row_text)
        cell_colors.append(row_color)
    return cell_text, cell_colors


def make_scorecard(records: list[dict], mesh_sets_in_order: list[str], footer: str) -> None:
    """Save `scorecard.{png,pdf}`: every headline metric x every mesh set, best value per row marked."""
    equilibrium = _equilibrium(records)
    n_cases = len(_by_mesh_set(equilibrium, mesh_sets_in_order)[mesh_sets_in_order[0]]) if mesh_sets_in_order else 0
    table = _scorecard_table(equilibrium, mesh_sets_in_order)

    _style.apply_style()
    n_rows, n_cols = len(_SCORECARD_METRICS), len(mesh_sets_in_order)
    fig, ax = plt.subplots(figsize=(3.0 + 1.7 * n_cols, 0.42 * n_rows + 1.0))
    ax.axis("off")

    row_labels = [label for label, _e, _l in _SCORECARD_METRICS]
    cell_text, cell_colors = _scorecard_cell_text_and_colors(table)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=mesh_sets_in_order,
        loc="center",
        cellLoc="center",
        rowLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.7)
    tbl.auto_set_column_width(col=list(range(-1, n_cols)))

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(_style.GRID)
        if row == 0 and col >= 0:
            cell.set_text_props(
                fontweight="bold", color=_style.mesh_set_color(mesh_sets_in_order[col], mesh_sets_in_order),
            )
        elif col == -1:
            cell.set_text_props(ha="left", fontsize=7.5)
        elif row >= 1:
            cell.set_text_props(
                color=cell_colors[row - 1][col],
                fontweight="bold" if cell_colors[row - 1][col] == _style.GREEN else "normal",
            )

    ax.set_title(
        f"Mesh-quality scorecard — {n_cases} equilibrium cases, min_distance=3", loc="left", fontsize=10, pad=14,
    )
    fig.text(
        0.01,
        0.028,
        "Median over the equilibrium cohort; best value per row in green. Bootstrap CIs and n are in the results JSON.",
        fontsize=6.5,
        color=_style.TEXT_MUTED,
    )
    fig.text(0.01, 0.005, footer, fontsize=6.5, color=_style.TEXT_MUTED, fontstyle="italic")
    _style.save_fig(fig, "scorecard")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Paired per-case comparison
# ---------------------------------------------------------------------------

#: Selected by scorecard **label**, not by list position: inserting an M1 row once made
#: every positional index in this list silently point one row up, which is how a
#: panel labelled "M2 area" would have drawn the M1 slope. Named lookup cannot drift that way.
_SCORECARD_BY_LABEL: dict[str, Callable] = {label: extractor for label, extractor, _ in _SCORECARD_METRICS}

_PAIRED_METRICS: list[tuple[str, Callable]] = [
    ("M1 net bias |median signed err| (deg)", _SCORECARD_BY_LABEL["M1 net bias |median signed err| (deg, vs Neumann)"]),
    ("M1 accuracy median |wedge err| (deg)", _SCORECARD_BY_LABEL["M1 accuracy median |wedge err| (deg, vs Neumann)"]),
    ("M2 |area err| median (%)", _SCORECARD_BY_LABEL["M2 |area err| median (%)"]),
    ("M3 |volume err| median, cells (%)", _SCORECARD_BY_LABEL["M3 |volume err| median, cells (%)"]),
    ("M6 abnormal edges", _SCORECARD_BY_LABEL["M6 abnormal non-manifold edges (total)"]),
    ("M7 wall-clock (s)", _SCORECARD_BY_LABEL["M7 wall-clock, median (s)"]),
    ("M7 n triangles", lambda r: r.get("m7_budget", {}).get("n_triangles")),
]


def make_paired_per_case(records: list[dict], mesh_sets_in_order: list[str], footer: str) -> None:
    """Save `paired_per_case.{png,pdf}`: one small multiple per metric, one faint line per case."""
    equilibrium = _equilibrium(records)
    cases = sorted({r["case"] for r in equilibrium})
    by_case_and_set: dict[tuple[str, str], dict] = {(r["case"], r["mesh_set"]): r for r in equilibrium}

    _style.apply_style()
    n = len(_PAIRED_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    x = np.arange(len(mesh_sets_in_order))
    for ax, (label, extractor) in zip(axes, _PAIRED_METRICS, strict=True):
        per_case_values = []
        for case in cases:
            values = [by_case_and_set.get((case, ms)) for ms in mesh_sets_in_order]
            y = [extractor(v) if v is not None else None for v in values]
            if any(v is None or not np.isfinite(v) for v in y):
                continue
            ax.plot(x, y, color=_style.TEXT_MUTED, alpha=0.25, linewidth=0.8, zorder=1)
            per_case_values.append(y)
        if per_case_values:
            medians = np.median(np.array(per_case_values), axis=0)
            for j, ms in enumerate(mesh_sets_in_order):
                ax.scatter([x[j]], [medians[j]], color=_style.mesh_set_color(ms, mesh_sets_in_order), zorder=3, s=28)
            ax.plot(x, medians, color=_style.TEXT_PRIMARY, linewidth=1.4, zorder=2)
        else:
            # No case had a finite value for this metric -- e.g. wall-clock when the meshes were
            # loaded from disk rather than built here. An empty panel keeps its default
            # 0-centred axis, which reads as "the value is 0"; say so instead.
            ax.text(
                0.5,
                0.5,
                "not available\nfor this mesh set",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color=_style.TEXT_MUTED,
            )
            ax.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels(mesh_sets_in_order, rotation=30, ha="right")
        ax.set_title(label, fontsize=8)
        _style.strip_spines(ax)
    fig.suptitle(
        f"Paired per-case comparison — {len(cases)} equilibrium cases, min_distance=3 "
        "(faint line = one case; bold = median)",
        fontsize=9,
    )
    fig.text(0.01, 0.005, footer, fontsize=6.5, color=_style.TEXT_MUTED, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _style.save_fig(fig, "paired_per_case")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Flattening
# ---------------------------------------------------------------------------


def make_flattening(records: list[dict], mesh_sets_in_order: list[str], footer: str) -> None:
    """Save `flattening.{png,pdf}`: signed line-mean angle error vs (true angle - 120), one panel per mesh set."""
    equilibrium = _equilibrium(records)
    grouped = _by_mesh_set(equilibrium, mesh_sets_in_order)

    _style.apply_style()
    n = len(mesh_sets_in_order)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ms in zip(axes, mesh_sets_in_order, strict=True):
        xs, ys, ws, floors = [], [], [], []
        for r in grouped[ms]:
            block = r.get("m1_contact_angles", {}).get("vs_neumann")
            if not block or "raw_true_minus_120_deg" not in block:
                continue
            xs.extend(block["raw_true_minus_120_deg"])
            ys.extend(block["raw_signed_error_deg"])
            ws.extend(block["raw_length_weight"])
            floor = block.get("matched_spacing_floor_deg")
            if floor is not None:
                floors.append(floor)
        color = _style.mesh_set_color(ms, mesh_sets_in_order)
        ax.scatter(xs, ys, s=4, alpha=0.15, color=color, linewidths=0)
        if xs:
            x_arr, y_arr, w_arr = np.array(xs), np.array(ys), np.array(ws)
            w_arr = np.where(w_arr > 0, w_arr, 1e-9)
            x_bar = np.average(x_arr, weights=w_arr)
            y_bar = np.average(y_arr, weights=w_arr)
            sxx = np.sum(w_arr * (x_arr - x_bar) ** 2)
            slope = np.sum(w_arr * (x_arr - x_bar) * (y_arr - y_bar)) / sxx if sxx > 0 else 0.0
            intercept = y_bar - slope * x_bar
            line_x = np.linspace(x_arr.min(), x_arr.max(), 2)
            ax.plot(line_x, intercept + slope * line_x, color=_style.TEXT_PRIMARY, linewidth=1.3)
        if floors:
            floor_med = float(np.median(floors))
            ax.axhspan(-floor_med, floor_med, color=_style.GRID, alpha=0.6, zorder=0)
        ax.axhline(0, color=_style.TEXT_MUTED, linewidth=0.6)
        ax.set_title(ms, color=color, fontsize=9)
        ax.set_xlabel("true angle - 120° (deg)")
        _style.strip_spines(ax)
    axes[0].set_ylabel("signed line-mean angle error (deg)")
    fig.suptitle(
        "Flattening — signed per-wedge angle error vs true angle, pooled equilibrium cases, "
        "min_distance=3. Line = length-weighted fit; its slope is the differential bias and its "
        "intercept the net bias. Band = the analytic-wedge floor, an ABSOLUTE median|angle-120| "
        "at an asserted 0.5-voxel vertex noise and this mesh's own spacing — it is comparable "
        "with median|err|, not with the scatter shown here. Reference: true Neumann angles.",
        fontsize=8.0,
    )
    fig.text(0.01, 0.005, footer, fontsize=6.5, color=_style.TEXT_MUTED, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    _style.save_fig(fig, "flattening")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Geometry fidelity
# ---------------------------------------------------------------------------

_FIDELITY_PANELS = [
    ("interface area (voxel^2)", "m2_interface_areas", "raw_measured_area", "raw_reference_area"),
    ("cell volume (voxel^3)", "m3_cell_volumes", None, None),  # special-cased: nested under "cells"
    ("interface mean curvature (1/voxel)", "m4_curvature", "raw_measured_curvature", "raw_reference_curvature"),
]


def make_geometry_fidelity(records: list[dict], mesh_sets_in_order: list[str], footer: str) -> None:
    """Save `geometry_fidelity.{png,pdf}`: areas, volumes, curvature -- reconstruction vs ground truth, y=x."""
    equilibrium = _equilibrium(records)

    _style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, (label, block_key, measured_key, reference_key) in zip(axes, _FIDELITY_PANELS, strict=True):
        all_measured, all_reference = [], []
        for ms in mesh_sets_in_order:
            color = _style.mesh_set_color(ms, mesh_sets_in_order)
            measured_vals, reference_vals = [], []
            for r in equilibrium:
                if r["mesh_set"] != ms:
                    continue
                block = r.get(block_key, {})
                if block_key == "m3_cell_volumes":
                    block = block.get("cells", {})
                    m, ref = block.get("raw_measured_volume"), block.get("raw_reference_volume")
                else:
                    m, ref = block.get(measured_key), block.get(reference_key)
                if m and ref:
                    measured_vals.extend(m)
                    reference_vals.extend(ref)
            # `compute_curvature_interfaces` can return nan for an interface with no
            # non-trijunction vertex to read (its own documented limitation); nan pairs are
            # invisible to scatter but must not enter a min/max (nan comparisons are always
            # False, so a plain min()/max() over a nan-containing list is order-dependent).
            paired = [
                (r, m) for r, m in zip(reference_vals, measured_vals, strict=True) if np.isfinite(r) and np.isfinite(m)
            ]
            finite_reference = [r for r, _m in paired]
            finite_measured = [m for _r, m in paired]
            ax.scatter(reference_vals, measured_vals, s=6, alpha=0.35, color=color, label=ms, linewidths=0)
            all_measured.extend(finite_measured)
            all_reference.extend(finite_reference)
        if all_reference:
            # Robust (1st-99th percentile) range: a single outlier must not compress the rest
            # of a small-multiple's axis into illegibility.
            combined = np.array(all_reference + all_measured)
            lo, hi = np.percentile(combined, [1, 99])
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color=_style.TEXT_MUTED, linewidth=1.0, linestyle="--", zorder=0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ground truth {label}")
        ax.set_ylabel(f"reconstruction {label}")
        ax.set_title(label, fontsize=8.5)
        _style.strip_spines(ax)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(
        "Geometry fidelity — reconstruction vs ground truth, equilibrium cases, min_distance=3 (dashed: y = x)",
        fontsize=9,
    )
    fig.text(0.01, 0.005, footer, fontsize=6.5, color=_style.TEXT_MUTED, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _style.save_fig(fig, "geometry_fidelity")
    plt.close(fig)


def main() -> None:
    """CLI entry point: load the core records and save all four figures under `--out-dir`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    # `save_fig` reads `_style.FIGURES_OUT` at call time, so honour `--out-dir` by rebinding it.
    # Without this the flag was accepted and silently ignored, and a second figure set (e.g. the
    # version arms) would overwrite the first (the configuration arms) in the package directory.
    _style.FIGURES_OUT = args.out_dir.resolve()
    _style.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    records, mesh_sets_in_order, footer = load_core_records(args.results)
    if len(records) == 0:
        message = f"no min_distance={CORE_MIN_DISTANCE} records found in {args.results}"
        raise SystemExit(message)

    make_scorecard(records, mesh_sets_in_order, footer)
    make_paired_per_case(records, mesh_sets_in_order, footer)
    make_flattening(records, mesh_sets_in_order, footer)
    make_geometry_fidelity(records, mesh_sets_in_order, footer)


if __name__ == "__main__":
    main()
