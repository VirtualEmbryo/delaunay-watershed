#!/usr/bin/env python
"""Generate the baseline JSON: 4 in-repo images + 47 benchmarking-dataset masks.

`benchmarks/baseline/` holds the **current default** algorithm's records — that is
boundary layer + junction protection + link-checked offset exclusion. The records of every
default it replaces are kept beside it so that every metric each flip moved can be diffed
rather than described: `benchmarks/baseline/junction_protected/` (boundary layer + junction
protection, a former default, also reachable as variant `offset_included`),
`benchmarks/baseline/deterministic/` (an earlier default) and
`benchmarks/baseline/dithered/` (the benchmark harness's first records, of the original
dithered algorithm).
`benchmarks/baseline/offset_excluded_linkcheck/` is the pre-flip run, which the top-level
records must reproduce byte for byte on every metric block — that is the check that the
flip to link-checked offset exclusion is a re-pointing and not a re-tuning. Regenerate any
of them with `--variant`. Note that the `dithered/` records keep their **original wall
times**, which are the reference the reconstruction work's speed budget is measured against;
re-running with `--variant dithered` will overwrite them with times from the current
machine and session.

Scope decision: the N_seeds=20 determinism sweep only runs on the 4 in-repo reference
images, matching the original determinism analysis of the dither (done on `3.tif` alone)
— running it on all 51 cases would take
on the order of an hour and is deferred, not silently dropped. All 51 cases get the full
geometry/topology/quality/watershed-health metrics plus a single-run cost profile (cheap).

The sweep measures the *dithered v0.3* algorithm whichever variant is selected: the
default has no seed left, so sweeping it would return zeros. See
`benchmarks/profiling.determinism_profile`, and `deterministic_vs_dither_sweep` for the
comparison that replaces it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import skimage.io as io

from dw3d_benchmarks.profiling import cost_profile, determinism_profile
from dw3d_benchmarks.run_case import LEGACY_VARIANT_ALIASES, run_case

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
BASELINE_DIR = REPO_ROOT / "benchmarks" / "baseline"
MIN_DISTANCE = 3
N_SEEDS_DETERMINISM = 20

IN_REPO_IMAGES = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"
DATASET_MASKS = sorted(DATASET_DIR.glob("*_labels_filled.tif"))


def _ground_truth_paths(mask_path: Path) -> tuple[Path | None, Path | None]:
    """The `.rec` mesh and tension dict beside a `benchmarking-dataset` mask, if they exist.

    The junction-protection work added the junction-length and junction-angle error metrics, which need them.
    The 4 in-repo images have no ground truth, so those cases simply report the intrinsic
    metrics — recorded rather than silently skipped.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec = mask_path.parent / f"{case}_mesh.rec"
    tensions = mask_path.parent / f"{case}_dict_tensions.npy"
    return (rec if rec.exists() else None, tensions if tensions.exists() else None)


def _run_one(mask_path: Path, with_determinism: bool, variant: str = "default") -> dict:
    mask = io.imread(mask_path)
    ground_truth_rec, tensions_path = _ground_truth_paths(mask_path)
    t0 = time.perf_counter()
    record = run_case(mask, MIN_DISTANCE, ground_truth_rec, variant=variant, tensions_path=tensions_path)
    record["case"] = mask_path.stem
    record["min_distance"] = MIN_DISTANCE
    record["variant"] = variant
    record["cost"] = cost_profile(mask, MIN_DISTANCE, variant=variant)
    if with_determinism:
        record["determinism"] = determinism_profile(mask, MIN_DISTANCE, n_seeds=N_SEEDS_DETERMINISM)
    print(f"  {mask_path.name}: {time.perf_counter() - t0:.1f}s")
    return record


def main() -> None:
    """Run the baseline case set and write one JSON record per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "default",
            "dithered",
            "deterministic",
            "cubic_score",
            "boundary_layer",
            "offset_included",
            "junction_protected",
            "junction_protected_noweights",
            "junction_protected_plan",
            "offset_excluded",
            "offset_excluded_cubic",
            "offset_excluded_unguarded",
            "offset_excluded_linkcheck",
            # Deprecated, still accepted:
            "v0_3",
            "a1b",
            "a3_a5",
        ),
        default="default",
    )
    parser.add_argument("--only", default=None, help="substring filter on the case stem (for partial re-runs)")
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="omit the 20-seed dither sweep on the 4 in-repo images (it is the slow part)",
    )
    args = parser.parse_args()
    if args.variant in LEGACY_VARIANT_ALIASES:
        canonical = LEGACY_VARIANT_ALIASES[args.variant]
        print(f"note: --variant {args.variant!r} is deprecated; use --variant {canonical!r} instead.")
        args.variant = canonical

    baseline_dir = BASELINE_DIR if args.variant == "default" else BASELINE_DIR / args.variant
    baseline_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # The 20-seed sweep is a *dither* sweep (v0.3 point placement); it is meaningless for
    # the deterministic default and the deterministic boundary-layer variant, so only the
    # dithered baseline carries it.
    determinism = (not args.skip_determinism) and args.variant == "dithered"
    print(f"In-repo images (variant={args.variant}, determinism={determinism}, n_seeds={N_SEEDS_DETERMINISM}):")
    for image_path in IN_REPO_IMAGES:
        if args.only and args.only not in image_path.stem:
            continue
        if not image_path.exists():
            print(f"  SKIP {image_path} (not found)")
            continue
        record = _run_one(image_path, with_determinism=determinism, variant=args.variant)
        out_path = baseline_dir / f"in_repo_{record['case']}_md{MIN_DISTANCE}.json"
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        summary.append(record["case"])

    print(f"benchmarking-dataset masks ({len(DATASET_MASKS)} found at {DATASET_DIR}):")
    if not DATASET_MASKS:
        print(f"  WARNING: no masks found under {DATASET_DIR}")
    for mask_path in DATASET_MASKS:
        if args.only and args.only not in mask_path.stem:
            continue
        record = _run_one(mask_path, with_determinism=False, variant=args.variant)
        out_path = baseline_dir / f"dataset_{record['case']}_md{MIN_DISTANCE}.json"
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        summary.append(record["case"])

    print(f"Wrote {len(summary)} baseline JSON files to {baseline_dir}")


if __name__ == "__main__":
    main()
