#!/usr/bin/env python
r"""Per-interface **area** error against the registered ground-truth mesh.

**Why this exists.** The offset-exclusion work removes the boundary layer's offsets from
the extracted surface, which flattens the "flaps" those offsets held open and takes the
total surface area to **0.888x** the default's. The first measurement could say only that this *ought*
to be an improvement — a flap reaching `delta` voxels off the surface and back adds area no
interface has — because nothing in the harness measured area against the ground truth. That
left the sign of the offset-exclusion work's effect on the one geometric quantity `foambryo`
actually consumes (interface area, through `compute_area_derivatives`) unknown. This module
measures it.

It is a **length-scale** comparison, so unlike the angle metrics it needs the registration:
`.rec` meshes live in a normalised frame whose axes are in the opposite order to the `.tif`
(see `metrics.similarity_to_mask_frame`). Areas are invariant under the rotation and the
reflection but not under the scale, so ground-truth areas are converted with `scale**2`. The
registration is fitted **mask-to-ground-truth only** and never sees the reconstruction, so it
is common to every variant and no variant can improve its score by moving the fit.

Keyed by label pair throughout, per the harness's standing rule (`AUDIT.md` Part III.3): the
two meshes are different discretisations and have no vertex correspondence.

The registration residual (~1 % of the object, `similarity_to_mask_frame`) enters area as
~2 %, so differences below a few per cent are not resolvable here. That is stated rather than
hidden, and it is well below the ~11 % effect being judged.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.interface_area_error \
        --cases 000 007 023 040 --variants default offset_excluded \
        --out benchmarks/baseline/interface_area_error_default_pilot8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks import metrics as m
from dw3d_benchmarks.run_case import VARIANT_GETTERS
from dw3d.io import load_rec

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
MIN_DISTANCE = 3


def interface_area_error(mask_path: Path, variant: str = "default") -> dict | None:
    """Relative per-interface area error of one reconstruction against its ground truth.

    Returns None when the case has no ground truth or the registration cannot be fitted.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec_path = mask_path.parent / f"{case}_mesh.rec"
    if not rec_path.exists():
        return None

    mask = io.imread(mask_path)
    algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    measured = m.interface_areas(points, triangles, labels)

    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    if registration.get("scale") is None:
        return None
    # A similarity scales every area by scale**2; the rotation and the reflection do not.
    scale_squared = float(registration["scale"]) ** 2
    gt_areas = m.interface_areas(gt_points, gt_triangles, gt_labels)
    reference = {key: value * scale_squared for key, value in gt_areas.items()}

    common = sorted(set(measured) & set(reference))
    if not common:
        return None
    relative = np.array([(measured[key] - reference[key]) / reference[key] for key in common])
    return {
        "case": case,
        "variant": variant,
        "registration_residual_mean_voxels": registration["residual_mean_voxels"],
        "n_common_interfaces": len(common),
        "n_measured_interfaces": len(measured),
        "n_reference_interfaces": len(reference),
        # Signed, because the sign is the whole question: the default is expected to be
        # positive (the flaps add area) and offset exclusion to be closer to zero.
        "signed_relative_error_median": float(np.median(relative)),
        "signed_relative_error_mean": float(np.mean(relative)),
        "absolute_relative_error_median": float(np.median(np.abs(relative))),
        "absolute_relative_error_p90": float(np.percentile(np.abs(relative), 90)),
        "total_area_ratio": float(sum(measured[k] for k in common) / sum(reference[k] for k in common)),
        "per_interface_signed_relative_error": {
            f"{key[0]},{key[1]}": float((measured[key] - reference[key]) / reference[key]) for key in common
        },
    }


def main() -> None:
    """Measure every (case, variant) and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["000", "007", "023", "040"])
    parser.add_argument("--variants", nargs="+", default=["default", "offset_excluded"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for case in args.cases:
        mask_path = DATASET_DIR / f"{case}_labels_filled.tif"
        if not mask_path.exists():
            print(f"  SKIP {case} (mask not found)")
            continue
        for variant in args.variants:
            record = interface_area_error(mask_path, variant=variant)
            if record is None:
                print(f"  SKIP {case} {variant} (no ground truth, or registration failed)")
                continue
            rows.append(record)
            print(
                f"  {case} {variant:24s} signed median {100 * record['signed_relative_error_median']:+7.2f} %"
                f"  |rel| median {100 * record['absolute_relative_error_median']:6.2f} %"
                f"  p90 {100 * record['absolute_relative_error_p90']:6.2f} %"
                f"  total {record['total_area_ratio']:.4f}x",
            )

    if rows:
        print("\nMedian over cases, per variant:")
        for variant in args.variants:
            selected = [r for r in rows if r["variant"] == variant]
            if not selected:
                continue
            signed = 100 * np.median([r["signed_relative_error_median"] for r in selected])
            absolute = 100 * np.median([r["absolute_relative_error_median"] for r in selected])
            total = np.median([r["total_area_ratio"] for r in selected])
            print(f"  {variant:24s} signed {signed:+7.2f} %  |rel| {absolute:6.2f} %  total {total:.4f}x")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True, default=float) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
