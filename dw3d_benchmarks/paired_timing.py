#!/usr/bin/env python
"""Paired wall-time comparison of two variants, measured back to back in one process.

**Why this exists.** `benchmarks/baseline/<variant>/*.json` each carry a `cost` block, but
they were written in different sessions on a machine with other work on it, so dividing one
committed total by another measures the sessions as much as the algorithms. The
junction-protection work avoided this by regenerating both variants' records in one run;
later runs must not regenerate the deterministic records, because they are the frozen
pre-flip baseline. This script gives the like-for-like ratio without touching them: for
every case it runs variant A then variant B in the same process, `--repeats` times, and
keeps the **minimum** per (case, variant) — the standard choice for wall-time
benchmarking, since scheduler noise is one-sided.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.paired_timing \
        --variants deterministic default --repeats 2 \
        --out benchmarks/baseline/timing_deterministic_vs_default.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks.run_case import VARIANT_GETTERS

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
IN_REPO_IMAGES = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
MIN_DISTANCE = 3


def _cases() -> list[tuple[str, Path]]:
    cases = [(p.stem, p) for p in IN_REPO_IMAGES if p.exists()]
    cases += [(p.stem, p) for p in sorted(DATASET_DIR.glob("*_labels_filled.tif"))]
    return cases


def main() -> None:
    """Time every case under each variant and report the ratio distribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["deterministic", "default"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records: dict[str, dict[str, float]] = {}
    for case, path in _cases():
        mask = io.imread(path)
        best: dict[str, float] = {}
        for _ in range(args.repeats):
            for variant in args.variants:
                algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
                start = time.perf_counter()
                algo.construct_mesh_from_segmentation_mask(mask)
                elapsed = time.perf_counter() - start
                best[variant] = min(best.get(variant, np.inf), elapsed)
        records[case] = best
        first, last = args.variants[0], args.variants[-1]
        print(f"  {case}: " + "  ".join(f"{v}={best[v]:.2f}s" for v in args.variants)
              + f"  ratio={best[last] / best[first]:.3f}")

    first, last = args.variants[0], args.variants[-1]
    ratios = np.array([r[last] / r[first] for r in records.values()])
    totals = {v: float(sum(r[v] for r in records.values())) for v in args.variants}
    summary = {
        "variants": args.variants,
        "repeats": args.repeats,
        "n_cases": len(records),
        "totals_s": totals,
        "ratio_of_totals": totals[last] / totals[first],
        "ratio_median": float(np.median(ratios)),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "n_cases_above_1_20": int((ratios > 1.20).sum()),
    }
    print(json.dumps(summary, indent=2))

    if args.out is not None:
        args.out.write_text(json.dumps({"summary": summary, "per_case": records}, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
