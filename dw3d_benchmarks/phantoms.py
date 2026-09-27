"""Analytic-phantom point classifiers, mirrored from `foambryo/benchmarks/phantoms.py`.

The phantoms must serve both the `dw3d` and the `foambryo` benchmark harnesses. The
derivation, limit checks (120 / 109.47 degrees for the tetrahedral vertex; the doublet
contact-angle formula and its `ratio=1 -> 120 degrees` check) and the exact analytic
*mesh* construction live in `foambryo/benchmarks/phantoms.py` — read that module's
docstring first. This file duplicates only the label-classifier functions (the part
`delaunay-watershed-3d` needs to voxelise a mask), by derivation rather than by import,
since the two repositories are kept independent (no cross-repo runtime
dependency for benchmarking infrastructure). Keep the two files' classifier logic in
sync by hand; a mismatch here would silently make the two harnesses test different
geometries.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

TETRAHEDRAL_LINE_ANGLE_DEG = 120.0
TETRAHEDRAL_POINT_ANGLE_DEG = float(np.degrees(np.arccos(-1.0 / 3.0)))  # 109.47122...

_TETRA_DIRECTIONS = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=float,
) / np.sqrt(3)


def doublet_predicted_angles(ratio: float) -> dict[str, float]:
    """Predicted rim wedge angles for the symmetric doublet, in degrees.

    See `foambryo/benchmarks/phantoms.py` for the derivation and its limit check.
    """
    if not (0 < ratio < 2):
        message = f"ratio={ratio} outside the physical domain (0, 2)"
        raise ValueError(message)
    theta = np.degrees(np.arccos(ratio / 2.0))
    return {
        "theta_deg": theta,
        "angle_through_medium_deg": 2 * theta,
        "angle_through_cell_deg": 180 - theta,
    }


def doublet_label_at(points: NDArray[np.float64], ratio: float, r0: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for the doublet: 0 = medium, 1 = cell A, 2 = cell B."""
    theta = np.arccos(ratio / 2.0)
    radius = r0 / np.sin(theta)
    zc = r0 / np.tan(theta)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    labels = np.zeros(len(points), dtype=np.int64)
    in_a = (z > 0) & (x**2 + y**2 + (z - zc) ** 2 <= radius**2)
    in_b = (z < 0) & (x**2 + y**2 + (z + zc) ** 2 <= radius**2)
    labels[in_a] = 1
    labels[in_b] = 2
    return labels


def tetrahedral_vertex_label_at(points: NDArray[np.float64], scale: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for the tetrahedral vertex: 0 = exterior, 1..4 = cells.

    See `foambryo.benchmarks.phantoms.tetrahedral_vertex_label_at` for the derivation of
    the `argmin` rule (cell `i`'s pyramid is over the face *opposite* vertex `i`).
    """
    vertices = _TETRA_DIRECTIONS * scale
    inside = np.ones(len(points), dtype=bool)
    for i in range(4):
        others = [j for j in range(4) if j != i]
        face_centroid = vertices[others].mean(axis=0)
        outward = -vertices[i] / np.linalg.norm(vertices[i])
        inside &= np.dot(points - face_centroid, outward) <= 1e-9
    labels = np.zeros(len(points), dtype=np.int64)
    dots = points @ _TETRA_DIRECTIONS.T
    nearest_pyramid = np.argmin(dots, axis=1) + 1
    labels[inside] = nearest_pyramid[inside]
    return labels


def _kelvin_lattice_points(cell_size: float, padding_shells: int = 2) -> NDArray[np.float64]:
    axis = np.arange(-padding_shells, padding_shells + 1)
    integer_pts = np.array(list(itertools.product(axis, axis, axis)), dtype=float)
    half_pts = integer_pts + 0.5
    return np.vstack([integer_pts, half_pts]) * cell_size


def _kelvin_kept_mask(centers: NDArray[np.float64], cell_size: float) -> NDArray[np.bool_]:
    return np.linalg.norm(centers, axis=1) <= cell_size + 1e-9


def kelvin_label_at(points: NDArray[np.float64], cell_size: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for the periodic Kelvin (BCC Voronoi) foam.

    NOT a mechanical equilibrium (flat faces) — a topology/performance stress test only.
    See `foambryo.benchmarks.phantoms.kelvin_label_at`'s docstring.
    """
    centers = _kelvin_lattice_points(cell_size)
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    kept = _kelvin_kept_mask(centers, cell_size)
    label_of_center = np.zeros(len(centers), dtype=np.int64)
    label_of_center[kept] = np.arange(1, kept.sum() + 1)
    return label_of_center[nearest]
