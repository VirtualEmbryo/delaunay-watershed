#!/usr/bin/env python
r"""Measure the clamp-saturation rate of a sub-voxel extremum refinement (never wired into
the pipeline; this is its acceptance gate).

The acceptance rule is explicit: report the fraction of clamp saturations. The original
audit of the pipeline measured a median shift of 0.590 voxel with frequent saturation on the
raw EDT; if saturation is still frequent after the cubic-interpolation work, sub-voxel
refinement is not yet trustworthy and must be gated on upsampling the EDT. This script
produces that number, for the two candidate refinements and for both interpolation orders,
**before** any of it is wired into the pipeline.

Two refinements, because the specified one has a structural problem:

* `separable` -- the specified construction: an independent 3-point parabola fit per axis,
  `delta_a = (v[-1] - v[+1]) / (2 * (v[-1] - 2 v[0] + v[+1]))`, clamped to `+-1/2` voxel.
  Its denominator is the second difference along that axis. **At an interface minimum the
  EDT is a valley, not a bowl**: it curves across the sheet and is flat along it, so two of
  the three second differences are near zero and their `delta` blows up and clamps. That is
  the audit's "frequent saturation", and it is geometry, not numerics -- so cubic interpolation's smoother
  field cannot fix it, because the flatness is real.
* `normal` -- refine along the across-sheet direction only, using the same EDT-Hessian
  dominant eigenvector the boundary-layer placer already computes
  (`points_on_edt._edt_hessian_normals`). One well-conditioned 1-D fit instead of three, two
  of which are ill-posed by construction.

Both are measured on the raw EDT and on the cubic B-spline field (evaluated through
`score_computation._interpolate_image`), so the "does cubic interpolation fix the saturation" question is
answered rather than assumed. Note in advance that a 3-point fit reads the field **at the
integer nodes only**, where an interpolating spline reproduces the data exactly -- so the
`separable`/`raw` and `separable`/`cubic` numbers must agree, and any difference would be a
bug. The cubic field only changes the answer for a scheme that samples off-node, which the
`normal` refinement does.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.subvoxel_saturation \
        --cases 3.tif 000 007 --out benchmarks/baseline/a6_subvoxel_saturation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import skimage.io as io
from numpy.typing import NDArray

from dw3d.edt import compute_edt_classical
from dw3d.points_on_edt import _edt_hessian_normals, plateau_packing_extrema
from dw3d.score_computation import _interpolate_image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
MIN_DISTANCE = 3
CLAMP = 0.5


def separable_shift(field: NDArray[np.float64], coords: NDArray[np.int64]) -> tuple[NDArray, NDArray]:
    """Per-axis 3-point parabola vertex offset, and which components had to be clamped.

    Returns `(shift, saturated)` with `shift` of shape `(n, 3)` already clamped to `+-1/2`
    and `saturated` a boolean array of the same shape. A zero (or numerically tiny) second
    difference is reported as saturated *and* given a zero shift: an unbounded step is not a
    refinement, and silently letting the clamp turn it into exactly `+-1/2` would hide the
    failure inside a plausible-looking number.
    """
    shape = np.asarray(field.shape, dtype=np.int64)
    centre = np.clip(coords.astype(np.int64), 1, shape - 2)
    shift = np.zeros((len(centre), 3))
    saturated = np.zeros((len(centre), 3), dtype=bool)
    for axis in range(3):
        step = np.zeros(3, dtype=np.int64)
        step[axis] = 1
        minus = field[tuple((centre - step).T)]
        here = field[tuple(centre.T)]
        plus = field[tuple((centre + step).T)]
        second = minus - 2 * here + plus
        degenerate = np.abs(second) < 1e-12
        raw = np.where(degenerate, 0.0, 0.5 * (minus - plus) / np.where(degenerate, 1.0, second))
        saturated[:, axis] = degenerate | (np.abs(raw) > CLAMP)
        shift[:, axis] = np.where(degenerate, 0.0, np.clip(raw, -CLAMP, CLAMP))
    return shift, saturated


def normal_shift(
    edt_image: NDArray[np.float64],
    coords: NDArray[np.int64],
    evaluate,  # noqa: ANN001 - the interpolant closure from `_interpolate_image`
    probe: float = 0.5,
) -> tuple[NDArray, NDArray]:
    """1-D parabola vertex offset along the across-sheet normal, and its saturation flags.

    The normal is the EDT Hessian's dominant eigenvector at the sample (the direction of
    strongest curvature, i.e. across the interface sheet -- see `points_on_edt`). The field
    is sampled at `-probe, 0, +probe` **along that direction**, so this genuinely exercises
    the interpolant off the integer lattice, and the parabola vertex is clamped to `+-1/2`.
    """
    normals = _edt_hessian_normals(edt_image, coords)
    upper = np.asarray(edt_image.shape, dtype=np.float64) - 1.0
    base = np.clip(coords.astype(np.float64), 0.0, upper)

    def at(offset: float) -> NDArray[np.float64]:
        # The trilinear interpolator *raises* outside [0, n-1]; the probe can leave the
        # domain at a sample on the padded shell, so clip rather than let it throw.
        return evaluate(np.clip(base + offset * normals, 0.0, upper))

    minus, here, plus = at(-probe), at(0.0), at(probe)
    second = minus - 2 * here + plus
    degenerate = np.abs(second) < 1e-12
    raw = np.where(degenerate, 0.0, 0.5 * probe * (minus - plus) / np.where(degenerate, 1.0, second))
    saturated = degenerate | (np.abs(raw) > CLAMP)
    magnitude = np.where(degenerate, 0.0, np.clip(raw, -CLAMP, CLAMP))
    return magnitude[:, None] * normals, saturated


def _summarise(shift: NDArray, saturated: NDArray) -> dict:
    magnitude = np.linalg.norm(shift, axis=1) if shift.ndim == 2 else np.abs(shift)
    per_point = saturated.any(axis=1) if saturated.ndim == 2 else saturated
    return {
        "n": len(magnitude),
        "median_shift_voxels": float(np.median(magnitude)),
        "p90_shift_voxels": float(np.percentile(magnitude, 90)),
        "fraction_saturated_per_point": float(np.mean(per_point)),
        "fraction_saturated_per_component": float(np.mean(saturated)),
    }


def measure(mask_path: Path, min_distance: int = MIN_DISTANCE) -> dict:
    """Both refinements x both interpolation orders, on the minima and on the maxima."""
    mask = io.imread(mask_path)
    edt = compute_edt_classical(mask)
    minima = plateau_packing_extrema(edt, min_distance, maximise=False)
    maxima = plateau_packing_extrema(edt, min_distance, maximise=True)

    record: dict = {"case": mask_path.stem, "n_minima": len(minima), "n_maxima": len(maxima)}
    for order in (1, 3):
        evaluate = _interpolate_image(edt, order)
        for family, coords in (("minima", minima), ("maxima", maxima)):
            if len(coords) == 0:
                continue
            record[f"separable_order{order}_{family}"] = _summarise(*separable_shift(edt, coords))
            record[f"normal_order{order}_{family}"] = _summarise(
                *normal_shift(edt, coords, evaluate),
            )
    return record


def main() -> None:
    """Measure the saturation rate on the requested cases and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["3.tif", "000", "007", "023"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for case in args.cases:
        path = REPO_ROOT / "data" / "Images" / case if case.endswith(".tif") else (
            DATASET_DIR / f"{case}_labels_filled.tif"
        )
        if not path.exists():
            print(f"  SKIP {case} ({path} not found)")
            continue
        record = measure(path)
        records.append(record)
        print(f"{record['case']}: {record['n_minima']} minima, {record['n_maxima']} maxima")
        for key, value in record.items():
            if not isinstance(value, dict):
                continue
            print(
                f"    {key:28s} median shift {value['median_shift_voxels']:.3f} vox  "
                f"saturated {100 * value['fraction_saturated_per_point']:5.1f} % of points, "
                f"{100 * value['fraction_saturated_per_component']:5.1f} % of components",
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
