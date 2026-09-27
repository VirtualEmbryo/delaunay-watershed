#!/usr/bin/env python
"""Two safeguards, proven rather than merely asserted.

1. **No ground truth in any estimator.** A reconstruction is run once from the dataset
   directory (ground truth sitting right next to the mask) and once from a directory
   containing *only* the mask, copied to an isolated temporary location. If the two
   reconstructions hash identically, ground truth was not merely unused but physically
   unreachable from the reconstruction call.
2. **Null control.** The same configuration, run twice from two independent worktrees at the
   same commit, must agree bitwise. Divergence would mean the pipeline is nondeterministic
   and every mesh-set comparison in this package is noise.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.safeguards \
        --dataset-dir benchmarking-dataset --case 000 --configuration dithered \
        --other-worktree /path/to/a/second/dw3d/worktree/at/the/same/commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import skimage.io as io


def _hash_mesh(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> str:
    """SHA-256 over the raw bytes of a mesh's three arrays, in a fixed dtype/order.

    Bytes, not `repr`/JSON text, so no textual rounding could mask a real difference.
    """
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(triangles, dtype=np.uint64).tobytes())
    hasher.update(np.ascontiguousarray(labels, dtype=np.uint64).tobytes())
    return hasher.hexdigest()


def check_ground_truth_unreachable(dataset_dir: Path, case_id: str, configuration: str, min_distance: int = 3) -> dict:
    """Reconstruct once with ground truth present, once with it physically absent; hash both."""
    from dw3d_benchmarks.mesh_quality_comparison import mesh_sets

    mask_path = dataset_dir / f"{case_id}_labels_filled.tif"
    mask = io.imread(mask_path)

    algo_with_gt = mesh_sets.get_algorithm(configuration, min_distance)
    points_a, triangles_a, labels_a = algo_with_gt.construct_mesh_from_segmentation_mask(mask)
    hash_with_gt_present = _hash_mesh(points_a, triangles_a, labels_a)

    with tempfile.TemporaryDirectory() as tmp:
        isolated_mask_path = Path(tmp) / mask_path.name
        shutil.copyfile(mask_path, isolated_mask_path)
        other_files = [p.name for p in dataset_dir.iterdir() if p.name.startswith(f"{case_id}_")]
        isolated_mask = io.imread(isolated_mask_path)
        algo_isolated = mesh_sets.get_algorithm(configuration, min_distance)
        points_b, triangles_b, labels_b = algo_isolated.construct_mesh_from_segmentation_mask(isolated_mask)
        hash_with_gt_absent = _hash_mesh(points_b, triangles_b, labels_b)

    return {
        "case": case_id,
        "configuration": configuration,
        "min_distance": min_distance,
        "files_present_in_dataset_dir_for_this_case": sorted(other_files),
        "hash_with_ground_truth_present": hash_with_gt_present,
        "hash_with_ground_truth_physically_absent": hash_with_gt_absent,
        "identical": hash_with_gt_present == hash_with_gt_absent,
    }


def check_null_control(
    dataset_dir: Path,
    case_id: str,
    configuration: str,
    other_worktree: Path,
    min_distance: int = 3,
) -> dict:
    """Reconstruct the same case+configuration in this worktree and in `other_worktree`; hash both.

    The other worktree only needs a plain `dw3d` (+ `dw3d_benchmarks`, already part of the
    checked-in repository) install -- it does not need this uncommitted package, so the
    generated script uses only `dw3d_benchmarks.run_case.VARIANT_GETTERS` directly.
    """
    from dw3d_benchmarks.mesh_quality_comparison import mesh_sets

    mask = io.imread(dataset_dir / f"{case_id}_labels_filled.tif")
    algo = mesh_sets.get_algorithm(configuration, min_distance)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    hash_here = _hash_mesh(points, triangles, labels)

    variant_key = mesh_sets.NAMED_CONFIGURATIONS[configuration]
    script = (
        "import hashlib, skimage.io as io, numpy as np\n"
        "from dw3d_benchmarks.run_case import VARIANT_GETTERS\n"
        f"mask = io.imread(r'{dataset_dir / f'{case_id}_labels_filled.tif'}')\n"
        f"algo = VARIANT_GETTERS['{variant_key}'](min_distance={min_distance}, print_info=False)\n"
        "points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)\n"
        "h = hashlib.sha256()\n"
        "h.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())\n"
        "h.update(np.ascontiguousarray(triangles, dtype=np.uint64).tobytes())\n"
        "h.update(np.ascontiguousarray(labels, dtype=np.uint64).tobytes())\n"
        "print(h.hexdigest())\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, python interpreter is an absolute path
        [str(other_worktree / ".venv" / "bin" / "python"), "-c", script],
        cwd=other_worktree,
        env={"PYTHONPATH": f"{other_worktree / 'src'}:{other_worktree}", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    hash_other = result.stdout.strip().splitlines()[-1]

    return {
        "case": case_id,
        "configuration": configuration,
        "min_distance": min_distance,
        "worktree_a": str(Path(__file__).resolve().parents[2]),
        "worktree_b": str(other_worktree),
        "hash_a": hash_here,
        "hash_b": hash_other,
        "identical": hash_here == hash_other,
    }


def main() -> None:
    """CLI entry point: run the requested safeguard check and print its result."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--case", default="000")
    parser.add_argument("--configuration", default="dithered")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--other-worktree", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "ground_truth_unreachable": check_ground_truth_unreachable(
            args.dataset_dir,
            args.case,
            args.configuration,
            args.min_distance,
        ),
    }
    if args.other_worktree is not None:
        report["null_control"] = check_null_control(
            args.dataset_dir,
            args.case,
            args.configuration,
            args.other_worktree,
            args.min_distance,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    failed = not report["ground_truth_unreachable"]["identical"] or (
        "null_control" in report and not report["null_control"]["identical"]
    )
    if failed:
        print("SAFEGUARD FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
