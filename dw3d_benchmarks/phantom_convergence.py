#!/usr/bin/env python
r"""Fit the junction-angle convergence exponent on the analytic tetrahedral-vertex phantom.

The junction-protection work's acceptance criterion: *"on the analytic phantoms,
angle-error convergence order improves from `O(h)` toward `O(h^2)` (fit the exponent,
report it with a confidence interval)"*. This is the harness for that fit, and it is run
on the **current default** first, so the exponent that work has to improve on is measured
rather than assumed to be 1.

**Two sweeps, because `min_distance` is in voxels and the answer depends on which one you
run.** Upsampling the EDT has the same trap (`min_distance` must be rescaled with it, or the
benchmark measures resolution rather than method); it applies just as much to the
convergence study itself.

* `fixed` -- `min_distance = 3` at every resolution. Refining the image also refines the
  *mesh*, since the seed spacing is `2*min_distance+1` **voxels**. Under the error model
  `angle_error ~ C * epsilon / ell` with both `epsilon` and `ell` in voxels and both held
  fixed, this sweep should converge at **`O(h^0)`** -- i.e. not at all. If it does, that is
  a property of the algorithm's parameterisation, not a failure of the reconstruction.
* `scaled` -- `min_distance` grows in proportion to the resolution, holding the mesh's
  *physical* edge length fixed. Now `epsilon` is `O(h)` in physical units while `ell` is
  constant, so the model predicts **`O(h)`**, which is the exponent lattice quantisation
  predicts for the current default and the one sub-voxel refinement is required to improve
  toward `O(h^2)`. A sub-voxel scheme that makes
  `epsilon = o(h)` would push this exponent above 1.

The phantom is `benchmarks.phantoms.tetrahedral_vertex_label_at`: four cells meeting at a
point, with an exact **120 degrees** along each of the four interior triple lines (the six
triples that involve the exterior label 0 are the outer-boundary truncation artefact and are
excluded, as in `generate_phantom_masks.py`).

The exponent is an ordinary least-squares fit of `log(error)` on `log(h)`, with a Student-t
95 % interval on the slope. With four or five resolutions the interval is wide by
construction; it is reported rather than hidden, because a point estimate of "1.3" from five
points is not evidence of anything on its own.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.phantom_convergence \
        --variants default deterministic \
        --out benchmarks/baseline/phantom_convergence_deterministic_vs_default.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from dw3d_benchmarks import metrics as m
from dw3d_benchmarks import phantoms as ph
from dw3d_benchmarks.run_case import VARIANT_GETTERS

HALF_EXTENT = 1.2
BASE_RESOLUTION = 40
BASE_MIN_DISTANCE = 3
DEFAULT_RESOLUTIONS = (40, 56, 80, 112, 160)


def voxelize_tetrahedral_vertex(resolution: int) -> NDArray[np.int32]:
    """Voxelise the tetrahedral-vertex phantom, matching `generate_phantom_masks._voxelize`."""
    step = 2 * HALF_EXTENT / resolution
    coords = -HALF_EXTENT + step * (np.arange(resolution) + 0.5)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    volume = np.zeros((resolution, resolution, resolution), dtype=np.int32)
    for iz, z in enumerate(coords):
        points = np.column_stack([xy_flat, np.full(len(xy_flat), z)])
        volume[:, :, iz] = ph.tetrahedral_vertex_label_at(points, scale=1.0).reshape(resolution, resolution)
    return volume


def interior_angle_error(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> dict:
    """Median and spread of `|angle - 120|` over the phantom's four interior triple lines."""
    measured = m.attributed_triple_line_angles(points, triangles, labels)
    errors = [
        abs(value - ph.TETRAHEDRAL_LINE_ANGLE_DEG)
        for key, line in measured.items()
        if 0 not in key
        for value in line.values()
    ]
    return {
        "n_interior_angles": len(errors),
        "median_deg": float(np.median(errors)) if errors else None,
        "mean_deg": float(np.mean(errors)) if errors else None,
        "p90_deg": float(np.percentile(errors, 90)) if errors else None,
    }


def fit_exponent(step_sizes: list[float], errors: list[float]) -> dict:
    """OLS slope of `log(error)` on `log(h)`, with a Student-t 95 % interval."""
    usable = [(h, e) for h, e in zip(step_sizes, errors, strict=True) if e is not None and e > 0]
    if len(usable) < 3:
        return {"exponent": None, "n_points": len(usable)}
    x = np.log(np.array([h for h, _ in usable]))
    y = np.log(np.array([e for _, e in usable]))
    fit = stats.linregress(x, y)
    critical = stats.t.ppf(0.975, df=len(x) - 2)
    return {
        "exponent": float(fit.slope),
        "stderr": float(fit.stderr),
        "ci95": [float(fit.slope - critical * fit.stderr), float(fit.slope + critical * fit.stderr)],
        "r_squared": float(fit.rvalue**2),
        "n_points": len(usable),
    }


def run_sweep(variant: str, resolutions: tuple[int, ...], mode: str) -> dict:
    """Reconstruct the phantom at each resolution and fit the exponent."""
    rows = []
    for resolution in resolutions:
        volume = voxelize_tetrahedral_vertex(resolution)
        if mode == "scaled":
            min_distance = max(1, round(BASE_MIN_DISTANCE * resolution / BASE_RESOLUTION))
        else:
            min_distance = BASE_MIN_DISTANCE
        algo = VARIANT_GETTERS[variant](min_distance=min_distance, print_info=False)
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(volume)
        error = interior_angle_error(points, triangles, labels)
        rows.append(
            {
                "resolution": resolution,
                "min_distance": min_distance,
                "h": 2 * HALF_EXTENT / resolution,
                "n_mesh_points": len(points),
                "n_triangles": len(triangles),
                **error,
            },
        )
        print(
            f"  {variant:8s} {mode:6s} res={resolution:4d} md={min_distance:2d} "
            f"h={rows[-1]['h']:.4f} n_angles={error['n_interior_angles']:3d} "
            f"err={error['median_deg'] if error['median_deg'] is None else round(error['median_deg'], 2)} deg",
        )
    return {
        "mode": mode,
        "rows": rows,
        "fit": fit_exponent([r["h"] for r in rows], [r["median_deg"] for r in rows]),
    }


def main() -> None:
    """Run both sweeps for each requested variant and write the fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["default"])
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--modes", nargs="+", default=["fixed", "scaled"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {}
    for variant in args.variants:
        report[variant] = {}
        for mode in args.modes:
            sweep = run_sweep(variant, tuple(args.resolutions), mode)
            report[variant][mode] = sweep
            fit = sweep["fit"]
            if fit.get("exponent") is None:
                print(f"  -> {variant} {mode}: too few usable resolutions to fit")
            else:
                print(
                    f"  -> {variant} {mode}: exponent q = {fit['exponent']:.2f} "
                    f"95% CI [{fit['ci95'][0]:.2f}, {fit['ci95'][1]:.2f}]  R^2 = {fit['r_squared']:.3f}",
                )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
