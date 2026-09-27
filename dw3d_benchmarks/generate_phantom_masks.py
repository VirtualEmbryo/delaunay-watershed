#!/usr/bin/env python
r"""CLI: voxelise the analytic phantoms and smoke-test dw3d against them.

Writes segmentation masks and compares `dw3d`'s default reconstruction to the phantom's
known ground-truth angle.

Mirrors `foambryo/benchmarks/generate_phantoms.py`'s voxelisation (see
`benchmarks/phantoms.py`'s module docstring for why the label-classifier logic is
duplicated rather than imported across repos). This script additionally runs the
default mesh reconstruction algorithm on each mask and reports the measured dihedral
angle at the phantom's known trijunction(s) via `benchmarks.metrics.
dihedral_angle_stats_per_triple_line` — a smoke test that the two harnesses agree on
what "correct" means, not a full convergence study (that belongs to the sub-voxel
refinement work, once there is a sub-voxel scheme to study; on the current lattice-quantised
default every mesh vertex sits on the integer voxel lattice, so the angle error is expected
to be `O(h)`).

HARD BOUNDARY: this reports the measured angle at a handful of resolutions as a sanity
check. It does not fit a convergence exponent or draw a conclusion about sub-voxel
refinement — that is out of scope for this script.

Usage:
    uv run python benchmarks/generate_phantom_masks.py --output-dir /tmp/phantom_masks \\
        --resolutions 32 64 128 --min-distance 2
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from pathlib import Path

import numpy as np
import skimage.io as io
from numpy.typing import NDArray

from dw3d_benchmarks import phantoms as ph
from dw3d_benchmarks.metrics import dihedral_angle_stats_per_triple_line
from dw3d import get_default_mesh_reconstruction_algorithm

DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)

# A phantom's point classifier: (n, 3) query points -> (n,) integer labels.
LabelFunction = Callable[[NDArray[np.float64]], NDArray[np.int64]]


def _voxelize(label_fn: LabelFunction, half_extent: float, resolution: int) -> np.ndarray:
    step = 2 * half_extent / resolution
    coords = -half_extent + step * (np.arange(resolution) + 0.5)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    volume = np.zeros((resolution, resolution, resolution), dtype=np.int32)
    for iz, z in enumerate(coords):
        pts = np.column_stack([xy_flat, np.full(len(xy_flat), z)])
        volume[:, :, iz] = label_fn(pts).reshape(resolution, resolution)
    return volume


def smoke_test_tetrahedral_vertex(output_dir: Path, resolutions: list[int], min_distance: int) -> None:
    """Voxelise the tetrahedral-vertex phantom and reconstruct it with dw3d's default.

    Reports the measured angle at the 4 interior (all-cell) trijunctions against the
    known 120-degree prediction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"tetrahedral_vertex (predicted 120.0 deg along each of 4 interior rays, "
          f"{ph.TETRAHEDRAL_POINT_ANGLE_DEG:.4f} deg between rays at the point):")
    for resolution in resolutions:
        volume = _voxelize(lambda pts: ph.tetrahedral_vertex_label_at(pts, scale=1.0), 1.2, resolution)
        io.imsave(output_dir / f"tetrahedral_vertex_res{resolution}.tif", volume, check_contrast=False)

        algo = get_default_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False)
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(volume)
        stats = dihedral_angle_stats_per_triple_line(points, triangles, labels)

        # keys are str(tuple), e.g. "(1, 2, 3)"; interior (all-cell) trijunctions
        # exclude label 0 (exterior) — see tetrahedral_vertex_mesh's docstring in
        # foambryo/benchmarks/phantoms.py for why only those 4 carry the 120-degree
        # prediction (the other 6 are the outer-boundary truncation artifact).
        interior_angles = [v["mean_deg"] for k, v in stats.items() if 0 not in ast.literal_eval(k)]
        if interior_angles:
            print(
                f"  res={resolution}: n_interior_trijunctions={len(interior_angles)} "
                f"mean_angle={np.mean(interior_angles):.2f} deg "
                f"(error={np.mean(interior_angles) - ph.TETRAHEDRAL_LINE_ANGLE_DEG:+.2f} deg)",
            )
        else:
            print(f"  res={resolution}: no interior trijunction reconstructed (resolution too coarse)")


def main() -> None:
    """Generate the phantom masks and run the smoke test from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--min-distance", type=int, default=2)
    args = parser.parse_args()
    smoke_test_tetrahedral_vertex(args.output_dir, args.resolutions, args.min_distance)


if __name__ == "__main__":
    main()
