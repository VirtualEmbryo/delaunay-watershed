#!/usr/bin/env python
r"""Summarise `interface_area_error` records into a signed-bias comparison across variants.

**Why this exists (the first step of the offset-exclusion change).** The offset-exclusion
work measured the then-default (boundary layer + junction protection) at **+15.0 %** signed
median interface-area error on 4 cases (000, 007, 023, 040; `min_distance = 3`) and read it as
"a previously unmeasured defect of the current `dw3d` default". That reading is only
complete if the configurations the default *replaced* are unbiased — otherwise the number
is a property of the pipeline rather than of the junction-protection default flip. This
module answers that
by comparing variants on the same records, and it reports enough to tell a **bias** from a
**spread**:

* the signed median (bias) beside the absolute median (total error) — for a pure one-sided
  bias the two coincide;
* `fraction_interfaces_over_estimated`, pooled over every interface of every case, which is
  the direct measurement of one-sidedness (0.5 is unbiased, 1.0 is "every interface too
  big");
* `n_cases_one_sided`, how many cases have *every* interface over- (or under-) estimated.

**Guarded against the vacuity failure mode** the offset-exclusion work ran into — a
metric-path typo made both sides of a comparison `None`, so `None == None` reported "0 cases
differ" while measuring nothing. Every quantity here is reported with the number of cases and
the number of interfaces it was actually computed from, `--require-cases` fails the run when a
variant is missing cases, and a variant whose per-interface payload is empty is an error
rather than a silently skipped row.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.analyze_interface_area_bias \
        --records benchmarks/baseline/interface_area_error_step0.json --require-cases 47
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def summarise(rows: list[dict], variant: str) -> dict:
    """Pool one variant's records into a bias summary, or raise if the payload is empty."""
    selected = [r for r in rows if r["variant"] == variant]
    if not selected:
        message = f"no records for variant {variant!r}"
        raise ValueError(message)

    per_case_signed = np.array([r["signed_relative_error_median"] for r in selected])
    per_case_absolute = np.array([r["absolute_relative_error_median"] for r in selected])
    per_case_total = np.array([r["total_area_ratio"] for r in selected])

    pooled: list[float] = []
    n_one_sided_positive = 0
    n_one_sided_negative = 0
    for record in selected:
        values = np.array(list(record["per_interface_signed_relative_error"].values()), dtype=float)
        if values.size == 0:
            message = f"variant {variant!r} case {record['case']!r} has an empty per-interface payload"
            raise ValueError(message)
        # Cross-check the stored summary against the payload it claims to summarise: this is
        # the check that would have caught the offset-exclusion work's vacuity bug at its source.
        if not np.isclose(float(np.median(values)), record["signed_relative_error_median"]):
            message = f"variant {variant!r} case {record['case']!r}: stored median disagrees with the payload"
            raise ValueError(message)
        pooled.extend(values.tolist())
        if np.all(values > 0):
            n_one_sided_positive += 1
        elif np.all(values < 0):
            n_one_sided_negative += 1

    pooled_array = np.array(pooled)
    return {
        "variant": variant,
        "n_cases": len(selected),
        "n_interfaces": int(pooled_array.size),
        "signed_median_of_case_medians": float(np.median(per_case_signed)),
        "absolute_median_of_case_medians": float(np.median(per_case_absolute)),
        "total_area_ratio_median": float(np.median(per_case_total)),
        "total_area_ratio_min": float(per_case_total.min()),
        "total_area_ratio_max": float(per_case_total.max()),
        "n_cases_with_positive_median": int((per_case_signed > 0).sum()),
        "n_cases_one_sided_positive": n_one_sided_positive,
        "n_cases_one_sided_negative": n_one_sided_negative,
        "pooled_signed_median": float(np.median(pooled_array)),
        "pooled_absolute_median": float(np.median(np.abs(pooled_array))),
        "pooled_absolute_p90": float(np.percentile(np.abs(pooled_array), 90)),
        "fraction_interfaces_over_estimated": float((pooled_array > 0).mean()),
    }


def main() -> None:
    """Print the per-variant bias table and, optionally, write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--require-cases",
        type=int,
        default=None,
        help="fail unless every variant has exactly this many cases (guards a silently short run)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.records.read_text())
    variants = args.variants or list(dict.fromkeys(r["variant"] for r in rows))

    summaries = [summarise(rows, variant) for variant in variants]

    if args.require_cases is not None:
        short = [s for s in summaries if s["n_cases"] != args.require_cases]
        if short:
            names = ", ".join(f"{s['variant']} ({s['n_cases']})" for s in short)
            message = f"expected {args.require_cases} cases per variant; got {names}"
            raise SystemExit(message)

    header = (
        f"{'variant':26s} {'cases':>5s} {'ifaces':>7s} {'signed':>9s} {'|rel|':>8s} "
        f"{'total':>8s} {'>0 frac':>8s} {'1-sided+':>9s}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['variant']:26s} {s['n_cases']:5d} {s['n_interfaces']:7d} "
            f"{100 * s['signed_median_of_case_medians']:+8.2f}% "
            f"{100 * s['absolute_median_of_case_medians']:7.2f}% "
            f"{s['total_area_ratio_median']:7.4f}x "
            f"{s['fraction_interfaces_over_estimated']:8.3f} "
            f"{s['n_cases_one_sided_positive']:4d}/{s['n_cases']:<4d}",
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
