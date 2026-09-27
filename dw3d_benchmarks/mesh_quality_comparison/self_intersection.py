r"""The nonlocal self-intersection certificate, as the M6 validity block emits it.

The predicate itself lives in :mod:`dw3d.triangle_intersection`, in the library, because
:mod:`dw3d.junction_relocation` uses it as its own acceptance test and the library must not depend
on this benchmark package. This module is what the evaluator needs on top of it: the six M6 keys,
and the independent oracle the predicate is checked against.

Why the predicate exists, in the evaluator's terms
--------------------------------------------------
Every other validity number in :mod:`.case_metrics` is **local or combinatorial**: watertightness,
abnormal non-manifold edges, degenerate and duplicate faces, quadjunction edges, reflex wedges and
interface identity are all read from a vertex's own neighbourhood or from the label structure. A
mesh whose two *distant* sheets pass through each other satisfies every one of them. Reviewing the
evaluator confirmed the gap; it was demonstrated the first time a post-process moved junction vertices
under a guard that checked only the triangles incident to the vertex it moved:

    all twelve local predicates stayed bit-identical on 40 of 40 cases while 23 nonlocal
    self-intersections existed on 5 of them.

**A local guard cannot certify a nonlocal property.**

The count is a **count, never a score**, and no threshold is proposed. It is reported beside the
other M6 counts and is compared before and after a change, never against a remembered number.

A self-intersection is **not** affine-invariant in general, so a mesh and its registered copy are
two different meshes for this predicate and must be reported as such.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dw3d.triangle_intersection import (
    CONTACT_TOLERANCE,
    DEGENERATE_TWICE_AREA,
    drop_adjacent_pairs,
    find_self_intersections,
    pair_intersection_flags,
    triangle_centroids_and_circumradii,
    twice_areas,
)

__all__ = (
    "CONTACT_TOLERANCE",
    "DEGENERATE_TWICE_AREA",
    "VALIDITY_KEYS",
    "compute_self_intersections",
    "differential_test",
    "drop_adjacent_pairs",
    "find_self_intersections",
    "oracle_pair_intersects",
    "pair_intersection_flags",
    "triangle_centroids_and_circumradii",
    "twice_areas",
)

_TRIANGLE_VERTICES = 3
_MOLLER_TRUMBORE_PARALLEL = 1e-14
#: The keys :func:`compute_self_intersections` emits into the M6 validity block. Fixed, and
#: additive: no pre-existing M6 key is renamed, removed or reordered by this module.
VALIDITY_KEYS: tuple[str, ...] = (
    "n_self_intersecting_triangle_pairs",
    "n_touching_triangle_pairs",
    "n_self_intersection_candidate_pairs",
    "n_self_intersection_tested_pairs",
    "n_self_intersection_degenerate_triangles",
    "self_intersection_search_radius",
)


def compute_self_intersections(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
) -> dict:
    """The nonlocal self-intersection counts, as M6 emits them.

    A thin, JSON-serialisable projection of :func:`find_self_intersections` carrying exactly
    :data:`VALIDITY_KEYS` and no array, so that adding it to the validity block adds six plain
    numbers and nothing else.

    Args:
        points (NDArray[np.float64]): vertex coordinates, the mesh's own units.
        triangles (NDArray[np.integer]): triangle vertex indices.

    Returns:
        dict: the six keys of :data:`VALIDITY_KEYS`. A count, never a score; no threshold is
        proposed here or anywhere else in this package.
    """
    found = find_self_intersections(points, triangles)
    return {key: found[key] for key in VALIDITY_KEYS}


def _segment_crosses_triangle(
    origin: NDArray[np.float64],
    end: NDArray[np.float64],
    triangle: NDArray[np.float64],
) -> bool:
    """Moller-Trumbore segment-triangle test -- the independent oracle's only primitive.

    Deliberately written the long way, one pair at a time and with a different parametrisation from
    :func:`pair_intersection_flags`, so that a shared mistake cannot make
    :func:`differential_test` agree for the wrong reason.
    """
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    direction = end - origin
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) < _MOLLER_TRUMBORE_PARALLEL:
        return False
    inverse = 1.0 / determinant
    tvec = origin - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < 0.0 or u > 1.0:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < 0.0 or u + v > 1.0:
        return False
    t = float(np.dot(edge2, qvec)) * inverse
    return 0.0 <= t <= 1.0


def oracle_pair_intersects(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Two non-coplanar triangles meet iff an edge of one crosses the other.

    Args:
        a (NDArray[np.float64]): one triangle's three corners, `(3, 3)`.
        b (NDArray[np.float64]): the other's.

    Returns:
        bool: whether they meet, decided by six Moller-Trumbore segment tests and nothing this
        module's own predicate uses.
    """
    for i in range(_TRIANGLE_VERTICES):
        if _segment_crosses_triangle(a[i], a[(i + 1) % 3], b):
            return True
        if _segment_crosses_triangle(b[i], b[(i + 1) % 3], a):
            return True
    return False


def differential_test(n_pairs: int = 4000, seed: int = 20260913) -> dict:
    """Agree with the independent oracle on random triangle pairs, or say where they differ.

    Coplanar pairs are excluded from the comparison because the oracle's primitive is undefined
    there and this module reports them as *touching* rather than as intersections; random pairs are
    coplanar with probability zero, and the constructed cases in the test suite cover that branch.

    Args:
        n_pairs (int): how many random pairs to compare.
        seed (int): fixed; the same seed gives the same pairs.

    Returns:
        dict: `n_pairs`, `n_intersecting` found by both, and `n_disagreements`.

    Raises:
        AssertionError: on any disagreement.
    """
    rng = np.random.default_rng(seed)
    n_disagreements, n_intersecting = 0, 0
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    for _ in range(n_pairs):
        # overlapping boxes, so a useful fraction of the pairs genuinely cross
        a = rng.uniform(-1.0, 1.0, size=(3, 3))
        b = rng.uniform(-1.0, 1.0, size=(3, 3))
        mine = find_self_intersections(np.vstack([a, b]), triangles)
        theirs = oracle_pair_intersects(a, b)
        n_intersecting += int(theirs)
        if bool(mine["n_self_intersecting_triangle_pairs"] > 0) != theirs:
            n_disagreements += 1
    assert n_disagreements == 0, f"{n_disagreements} of {n_pairs} pairs disagreed with the oracle"
    return {"n_pairs": n_pairs, "n_intersecting": n_intersecting, "n_disagreements": 0}
