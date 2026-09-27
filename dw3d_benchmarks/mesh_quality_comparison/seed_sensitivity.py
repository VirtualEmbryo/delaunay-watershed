#!/usr/bin/env python
r"""Is the `dithered` configuration's advantage real, or an artefact of seed 42?

`dithered` -- dw3d's shipped default -- places points with a **randomly dithered** rule whose
draw is fixed by a seed (`get_dithered_algorithm(seed=42)`). Every published comparison of it
uses that one seed. That makes two different claims easy to confuse:

* "this mesh is reproducible" -- true, the seed is fixed and the RNG is private; and
* "this configuration is better" -- which, measured at one seed, could be a lucky draw.

This script separates them. It re-runs `dithered` over `--seeds` on the whole cohort and reports,
per metric:

* `within_case_spread` -- for each case, the spread of the metric across seeds, then the median
  of those spreads. How much the seed moves a *single* case's number.
* `cohort_median_per_seed` -- the cohort median recomputed for each seed, and the spread of
  *that*. This is the decisive one: it is the quantity a scorecard reports, so if it moves by
  more than the gap between two configurations, the ranking between them is not established.

Interpretation is deliberately left to the caller and the report: this script measures the
seed's influence, it does not decide whether the default should change.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.seed_sensitivity \
        --dataset-dir /path/to/benchmarking-dataset --seeds 42 0 1 2 3 4 5 6 \
        --workers 6 --out results/seed_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d import get_dithered_mesh_reconstruction_algorithm

from . import bootstrap, case_metadata, case_metrics

#: Metric name -> (extractor over a per-(case, seed) block, lower_is_better).
#:
#: `m1_net_bias_deg` was called `m1_abs_angle_error_deg` until the statistics were renamed. The value
#: is unchanged -- `abs(median(signed error))` -- but the old name said "absolute angle error"
#: for what is the magnitude of the **net bias**, a statistic that alternating errors of
#: opposite sign cancel out of. `m1_median_abs_angle_error_deg` is the accuracy statistic the
#: old name promised. Both are reported; historical JSON keeps its own key names on disk.
METRICS: dict[str, tuple] = {
    "m1_net_bias_deg": (lambda b: b["m1_net_bias_deg"], True),
    "m1_median_abs_angle_error_deg": (lambda b: b["m1_median_abs_angle_error_deg"], True),
    "m1_flattening_slope": (lambda b: b["m1_flattening_slope"], True),
    "m2_abs_area_error_pct": (lambda b: b["m2_abs_area_error_pct"], True),
    "m6_abnormal_edges": (lambda b: b["m6_abnormal_edges"], True),
    "n_triangles": (lambda b: b["n_triangles"], False),
}


def measure_one(dataset_dir: str, case_id: str, seed: int, min_distance: int) -> dict:
    """Reconstruct `case_id` with `dithered` at `seed` and return the headline metrics."""
    dataset_path = Path(dataset_dir)
    mask = io.imread(dataset_path / f"{case_id}_labels_filled.tif")
    algo = get_dithered_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False, seed=seed)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

    gt = case_metrics.load_ground_truth(dataset_path, case_id)
    block: dict = {"case": case_id, "seed": seed, "n_triangles": len(triangles)}
    block["m6_abnormal_edges"] = case_metrics.compute_m6_validity(points, triangles, labels, gt)[
        "n_abnormal_non_manifold_edges"
    ]

    from foambryo.dcel import DcelData

    mesh = DcelData(points, triangles, labels)
    gt_mesh = DcelData(gt["points"], gt["triangles"], gt["labels"])
    neumann = case_metrics.neumann_wedge_angles_deg(gt["tensions"]) if gt["tensions"] is not None else None
    m1 = case_metrics.compute_m1_contact_angles(mesh, gt_mesh, neumann, None)
    reference = m1.get("vs_neumann") or m1.get("vs_ground_truth_mesh") or {}
    # Two different quantities, each under a name that says which it is. The first is the
    # magnitude of the net bias; the second is how wrong a typical wedge is. They rank
    # configurations differently, so reading one under the other's name is not a detail.
    block["m1_net_bias_deg"] = abs(reference.get("signed_error_median_deg", float("nan")))
    block["m1_median_abs_angle_error_deg"] = reference.get("median_abs_error_deg", float("nan"))
    block["m1_flattening_slope"] = (reference.get("flattening_slope_length_weighted") or {}).get("slope", float("nan"))

    m2 = case_metrics.compute_m2_interface_areas(points, triangles, labels, mask, gt)
    block["m2_abs_area_error_pct"] = 100 * abs(m2.get("signed_relative_error_median", float("nan")))
    return block


def summarise(blocks: list[dict], seeds: list[int], case_ids: list[str]) -> dict:
    """Per metric: how much the seed moves one case, and how much it moves the cohort median."""
    by_key = {(b["case"], b["seed"]): b for b in blocks}
    summary: dict = {}
    for name, (extract, lower_is_better) in METRICS.items():
        within: list[float] = []
        for case in case_ids:
            values = [
                v
                for s in seeds
                if (b := by_key.get((case, s))) is not None and np.isfinite(v := float(extract(b)))
            ]
            if len(values) > 1:
                within.append(max(values) - min(values))

        per_seed: dict[str, float] = {}
        for s in seeds:
            values = [
                v
                for case in case_ids
                if (b := by_key.get((case, s))) is not None and np.isfinite(v := float(extract(b)))
            ]
            if values:
                per_seed[str(s)] = float(np.median(values))

        medians = list(per_seed.values())
        summary[name] = {
            "lower_is_better": lower_is_better,
            "within_case_spread_median": float(np.median(within)) if within else None,
            "within_case_spread_max": float(np.max(within)) if within else None,
            "cohort_median_per_seed": per_seed,
            "cohort_median_min": float(np.min(medians)) if medians else None,
            "cohort_median_max": float(np.max(medians)) if medians else None,
            "cohort_median_range": float(np.max(medians) - np.min(medians)) if medians else None,
            "cohort_median_at_seed_42": per_seed.get("42"),
            # Is seed 42 a lucky draw? Where does it sit among the seeds tried?
            "seed_42_rank_among_seeds": (
                sorted(medians).index(per_seed["42"]) + 1 if "42" in per_seed and per_seed["42"] in medians else None
            ),
            "n_seeds": len(medians),
        }
        # A case-level bootstrap CI at seed 42, so the seed range can be read against the
        # sampling uncertainty the scorecard already carries.
        values_42 = [
            v for case in case_ids
            if (b := by_key.get((case, 42))) is not None and np.isfinite(v := float(extract(b)))
        ]
        summary[name]["bootstrap_ci_at_seed_42"] = bootstrap.bootstrap_median_ci(values_42)
    return summary


def main() -> None:
    """CLI entry point: sweep seeds x cases in a process pool and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--cases", nargs="*", default=None, help="default: the 40 equilibrium cases")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    case_ids = args.cases or case_metadata.accuracy_cohort_case_ids(args.dataset_dir)
    jobs = [(c, s) for c in case_ids for s in args.seeds]
    print(f"[seeds] {len(jobs)} jobs ({len(case_ids)} cases x {len(args.seeds)} seeds), {args.workers} workers")

    blocks: list[dict] = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(measure_one, args.dataset_dir, case, seed, args.min_distance): (case, seed)
            for case, seed in jobs
        }
        for future in as_completed(futures):
            try:
                blocks.append(future.result())
            except Exception as exc:
                n_failed += 1
                print(f"[seeds] FAILED {futures[future]}: {exc!r}")
            if len(blocks) % 40 == 0:
                print(f"[seeds] {len(blocks)}/{len(jobs)} done")

    report = {
        "configuration": "dithered",
        "min_distance": args.min_distance,
        "seeds": args.seeds,
        "cases": case_ids,
        "n_failed": n_failed,
        "summary": summarise(blocks, args.seeds, case_ids),
        "blocks": sorted(blocks, key=lambda b: (b["case"], b["seed"])),
    }
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        print(f"[seeds] wrote {args.out}")


if __name__ == "__main__":
    main()
