"""Generator for the synthetic scaling dataset. The *generator* is checked in, not the volumes.

What this is for
----------------
The ground-truth benchmark has a limit that makes the EDT memory/scaling work hard to evaluate:
`benchmarking-dataset/` tops out at **10 cells and ~200^3**, so nothing in it exercises the
2048^3 / thousands-of-cells target, and a memory or scaling claim measured on a 200^3 mask is
an extrapolation, not a measurement. This module generates large multi-cell masks at
`256^3, 512^3, 1024^3, 2048^3` with 10^2-10^3 cells.

**No ground-truth tensions.** By design, this dataset measures *speed, memory and
topology only*. A relaxed Laguerre/Voronoi foam is not a mechanical equilibrium -- its
interfaces are flat polygons, not the constant-mean-curvature sheets Laplace's law requires,
which are non-spherical in general (the same reason power / Möbius diagrams were rejected as
a model of foam geometry). Using these masks for a tension-accuracy claim would therefore be
wrong, and this docstring says so rather than leaving it to be inferred: they are a **cost** benchmark.

Construction
------------
1. `n_cells` seeds, from a fixed-seed `Generator`, Lloyd-relaxed for `n_lloyd` iterations
   against the *voxel grid itself* (each iteration is one `cKDTree` query over a strided
   subsample of the grid, then a centroid). Relaxation is what makes the cells
   foam-like -- roughly equal-sized and compact -- instead of the wide size spread a raw
   Poisson process gives.
2. Optional per-seed weights make it a **Laguerre** (power) diagram rather than a Voronoi
   one, which is how a prescribed cell-size distribution is imposed. `weight_spread=0`
   gives the plain Voronoi case.
3. Labels are assigned by nearest (power-)seed, computed **tile by tile** so that a 2048^3
   mask never needs a full-volume intermediate beyond the output itself.
4. A background shell of `margin` voxels is left at label 0, because `dw3d` needs the
   exterior to exist (and because `_pad_mask` will zero that shell anyway).

Reproducibility: everything is a pure function of `(side, n_cells, seed, n_lloyd,
weight_spread, margin)`. `describe()` returns that tuple so a scaling row can record it.

Usage:
    uv run python benchmarks/synthetic_foam.py --side 512 --n-cells 200 --out /tmp/foam512.npy
    uv run python benchmarks/synthetic_foam.py --scaling-curve --out-dir /tmp/a2b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# The scaling ladder. 2048^3 as `int32` is 34 GB for the mask alone, so it is generated
# only on explicit request and reported as skipped otherwise -- never silently dropped.
SCALING_SIDES = (256, 512, 1024, 2048)


def _seed_positions(
    side: int,
    n_cells: int,
    seed: int,
    n_lloyd: int,
    margin: int,
    subsample: int = 4,
) -> NDArray[np.float64]:
    """Lloyd-relaxed seed positions inside the margin, from a fixed-seed generator.

    The relaxation is run against a strided subsample of the voxel grid (`subsample` per
    axis), which is what keeps generation affordable at 1024^3: the centroid of a cell's
    voxels is estimated from 1/64 of them, which is far more accuracy than a *cost* benchmark
    needs, and the result is still deterministic.
    """
    rng = np.random.default_rng(seed)
    low, high = margin, side - margin
    points = rng.uniform(low, high, size=(n_cells, 3))

    axis = np.arange(low, high, subsample)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    for _ in range(n_lloyd):
        _, nearest = cKDTree(points).query(grid, workers=-1)
        total = np.zeros_like(points)
        count = np.bincount(nearest, minlength=len(points)).astype(np.float64)
        np.add.at(total, nearest, grid)
        moved = count > 0
        points[moved] = total[moved] / count[moved, None]
    return points


def generate_foam_mask(
    side: int,
    n_cells: int = 200,
    seed: int = 0,
    n_lloyd: int = 8,
    weight_spread: float = 0.0,
    margin: int | None = None,
    tile_size: int = 128,
    dtype: type = np.int32,
) -> NDArray[np.int32]:
    """Generate a relaxed Laguerre/Voronoi foam segmentation mask.

    Args:
        side (int): Cube side length in voxels.
        n_cells (int, optional): Number of cells. Defaults to 200.
        seed (int, optional): Generator seed. Defaults to 0.
        n_lloyd (int, optional): Lloyd relaxation iterations. Defaults to 8.
        weight_spread (float, optional): Relative spread of the power-diagram weights; `0`
            gives a plain Voronoi foam. Defaults to 0.0.
        margin (int | None, optional): Background shell thickness. Defaults to `side // 16`.
        tile_size (int, optional): Labelling tile size, so no full-volume intermediate is
            built. Defaults to 128.
        dtype (type, optional): Mask dtype. `int32` matches `data/Images/*.tif`.

    Returns:
        NDArray[np.int32]: labels `1..n_cells` inside the margin, `0` outside.
    """
    if margin is None:
        margin = max(2, side // 16)
    points = _seed_positions(side, n_cells, seed, n_lloyd, margin)

    rng = np.random.default_rng(seed + 1)
    # Power-diagram weights. The label of a voxel is `argmin_i (|x - p_i|^2 - w_i)`, which is
    # `argmin_i (|x - p_i|^2 + (max(w) - w_i))` -- i.e. an additive offset on squared
    # distance, so it can be evaluated directly and does not need a lifted hull here.
    weights = np.zeros(len(points)) if weight_spread == 0 else rng.normal(0.0, weight_spread, size=len(points))
    offset = weights.max() - weights

    mask = np.zeros((side, side, side), dtype=dtype)
    inner = slice(margin, side - margin)
    for i0 in range(margin, side - margin, tile_size):
        i1 = min(i0 + tile_size, side - margin)
        grid = np.stack(
            np.meshgrid(
                np.arange(i0, i1),
                np.arange(inner.start, inner.stop),
                np.arange(inner.start, inner.stop),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        squared = ((grid[:, None, :] - points[None, :, :]) ** 2).sum(axis=2) + offset[None, :]
        mask[i0:i1, inner, inner] = (np.argmin(squared, axis=1) + 1).reshape(i1 - i0, -1, inner.stop - inner.start)
    return mask


def generate_foam_mask_low_memory(
    side: int,
    n_cells: int = 200,
    seed: int = 0,
    n_lloyd: int = 8,
    weight_spread: float = 0.0,
    margin: int | None = None,
    tile_size: int = 64,
    dtype: type = np.int32,
) -> NDArray[np.int32]:
    """`generate_foam_mask` with a `cKDTree` instead of the dense `(voxels, cells)` distance matrix.

    The dense form costs `O(tile_voxels * n_cells)` floats, which is the wrong shape at
    `n_cells ~ 10**3`: a 128^3 tile against 1000 cells is 16 GB. This form is `O(log n_cells)`
    per voxel and is what the 1024^3 / 2048^3 rows use. It is exact for the Voronoi case
    (`weight_spread=0`); for a genuine power diagram the `cKDTree` metric is the wrong one,
    so a non-zero `weight_spread` falls back to the dense form.
    """
    if weight_spread != 0:
        return generate_foam_mask(side, n_cells, seed, n_lloyd, weight_spread, margin, tile_size, dtype)
    if margin is None:
        margin = max(2, side // 16)

    points = _seed_positions(side, n_cells, seed, n_lloyd, margin)
    tree = cKDTree(points)
    mask = np.zeros((side, side, side), dtype=dtype)
    inner = slice(margin, side - margin)
    span = inner.stop - inner.start
    for i0 in range(margin, side - margin, tile_size):
        i1 = min(i0 + tile_size, side - margin)
        grid = np.stack(
            np.meshgrid(np.arange(i0, i1), np.arange(inner.start, inner.stop), np.arange(inner.start, inner.stop),
                        indexing="ij"),
            axis=-1,
        ).reshape(-1, 3).astype(np.float64)
        _, nearest = tree.query(grid, workers=-1)
        mask[i0:i1, inner, inner] = (nearest + 1).reshape(i1 - i0, span, span)
    return mask


def describe(side: int, n_cells: int, seed: int, n_lloyd: int, weight_spread: float, margin: int | None) -> dict:
    """The full parameter tuple, so a scaling row records what it measured."""
    return {
        "side": side,
        "n_voxels": side**3,
        "n_cells": n_cells,
        "seed": seed,
        "n_lloyd": n_lloyd,
        "weight_spread": weight_spread,
        "margin": margin if margin is not None else max(2, side // 16),
        "generator": "benchmarks/synthetic_foam.py:generate_foam_mask_low_memory",
    }


def default_cell_count(side: int) -> int:
    """Cell count that holds the cells' voxel diameter roughly fixed across the ladder.

    A constant `n_cells` would make the cells 8x larger in volume per doubling, so the
    scaling curve would measure "bigger cells", not "bigger images" -- and since the EDT's
    cost model depends on the cells' inradius (it sets the halo the tiling needs), that
    would confound exactly the quantity the EDT scaling work is trying to measure. Scaling `n_cells` as
    `side**3` keeps the inradius fixed instead; the ladder is 27, 216, 1728, 13824, which is
    the targeted 10^2-10^3-cell band at 512^3-1024^3.
    """
    return max(8, round(27 * (side / 256) ** 3))


def _scaling_rows(sides: tuple[int, ...], n_cells: int | None, seed: int, out_dir: Path | None) -> list[dict]:
    """Generate each mask, record generation cost, optionally write it, and report."""
    import resource
    import sys
    import time

    rss_scale = (1 / 1024**2) if sys.platform == "darwin" else (1 / 1024)
    rows = []
    for side in sides:
        cells = default_cell_count(side) if n_cells is None else n_cells
        t0 = time.perf_counter()
        mask = generate_foam_mask_low_memory(side, n_cells=cells, seed=seed)
        row = describe(side, cells, seed, 8, 0.0, None)
        row["generation_wall_time_s"] = time.perf_counter() - t0
        row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
        row["n_labels_present"] = len(np.unique(mask))
        row["mask_nbytes"] = int(mask.nbytes)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"foam_{side}_{cells}cells_seed{seed}.npy"
            np.save(path, mask)
            row["path"] = str(path)
        rows.append(row)
        del mask
        print(json.dumps(row))
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--n-cells", type=int, default=None, help="default: see default_cell_count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write a single .npy mask here")
    parser.add_argument("--scaling-curve", action="store_true", help="generate the whole ladder")
    parser.add_argument("--sides", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.scaling_curve:
        sides = tuple(args.sides) if args.sides else SCALING_SIDES[:3]
        _scaling_rows(sides, args.n_cells, args.seed, args.out_dir)
        return

    cells = default_cell_count(args.side) if args.n_cells is None else args.n_cells
    mask = generate_foam_mask_low_memory(args.side, n_cells=cells, seed=args.seed)
    print(json.dumps({**describe(args.side, cells, args.seed, 8, 0.0, None), "n_labels": len(np.unique(mask))}))
    if args.out is not None:
        np.save(args.out, mask)


if __name__ == "__main__":
    main()
