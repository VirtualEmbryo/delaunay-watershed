"""Nearest-neighbour lookups whose exact ties go to the lowest index, whatever the SciPy build.

`cKDTree.query` breaks an exact tie by its tree traversal, and the tree is partitioned by the C++
standard library, so two builds of SciPy (macOS and Linux) can return different neighbours for the
same input. On dw3d's data exact ties are common: points, seeds and tetrahedron centroids sit on
the voxel lattice or on its quarters. Here the k-d tree only proposes candidates, and the choice
among them is made in NumPy, by squared distance, then by index.

Herve Turlier's group, 2026.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

#: Absolute margin, in the points' units, by which the last candidate must be farther than the
#: first for the candidates to hold every exact minimum. Far above the rounding error of a
#: distance between points of magnitude below 1e6, far below any distance between lattice points.
CANDIDATE_MARGIN = 1e-9


def candidates_holding_every_nearest(tree: cKDTree, queries: NDArray[np.float64]) -> NDArray[np.int64]:
    """The `k` nearest points of each query, `k` doubled until no exact minimum can be left out.

    Args:
        tree (cKDTree): the tree over the points.
        queries (NDArray[np.float64]): `(n, d)` query positions.

    Returns:
        NDArray[np.int64]: `(n, k)` point indices; every point at the minimum distance of a query
            is in its row (`k` stops growing once the `k`-th is farther than the first by more
            than `CANDIDATE_MARGIN`, or holds every point).
    """
    n_points = tree.n
    k = min(8, n_points)
    while True:
        distance, candidate = tree.query(queries, k=k)
        distance, candidate = distance.reshape(len(queries), k), candidate.reshape(len(queries), k)
        if k == n_points or np.all(distance[:, -1] > distance[:, 0] + CANDIDATE_MARGIN):
            return candidate.astype(np.int64)
        k = min(2 * k, n_points)


def nearest_with_ties_to_the_lowest_index(
    points: ArrayLike,
    queries: ArrayLike,
    tree: cKDTree | None = None,
) -> NDArray[np.int64]:
    """For each query, the index of the nearest point; on an exact tie, the lowest index.

    The squared distances of the candidates are recomputed with separate NumPy operations, summed
    axis by axis, which round alike on every platform. They are exact when the coordinates are
    integers or quarter-integers below `2**23`, as dw3d's lattice points and tetrahedron centroids
    are, so a tie there is a true tie.

    Args:
        points (ArrayLike): `(m, d)` points.
        queries (ArrayLike): `(n, d)` query positions.
        tree (cKDTree | None): a tree already built over `points`.

    Returns:
        NDArray[np.int64]: `(n,)` indices into `points`.
    """
    points = np.asarray(points, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if len(queries) == 0:
        return np.zeros(0, dtype=np.int64)
    candidate = candidates_holding_every_nearest(cKDTree(points) if tree is None else tree, queries)
    offsets = points[candidate] - queries[:, None, :]
    squared_distance = offsets[:, :, 0] * offsets[:, :, 0]
    for axis in range(1, offsets.shape[2]):  # summed in a fixed order, axis by axis
        squared_distance = squared_distance + offsets[:, :, axis] * offsets[:, :, axis]
    is_minimum = squared_distance == squared_distance.min(axis=1, keepdims=True)
    return np.where(is_minimum, candidate, np.iinfo(np.int64).max).min(axis=1)
