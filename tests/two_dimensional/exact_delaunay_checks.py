"""Exact checks of a Delaunay triangulation of integer points, for the golden-master tests.

dw2d triangulates integer pixel points. Four or more of them are often cocircular; the Delaunay
triangulation is then not unique, and Qhull picks a triangulation of each cocircular polygon by
floating-point rounding, which differs between machines (measured when dw2d 0.8.0 was prepared: on
macOS arm64 it differs from x86-64 in 13 of the 18 golden cases, in 1 to 7 cocircular squares each). These
helpers decide, in exact integer arithmetic, whether two triangulations of the same points differ
only inside cocircular polygons.
"""
import numpy as np


def canonical(triangles):
    """Triangles with sorted vertices, rows in lexicographic order: the triangulation as a set."""
    s = np.sort(np.asarray(triangles), axis=1)
    return s[np.lexsort(s.T[::-1])]


def integer_points(points):
    p = np.asarray(points)
    q = np.rint(p).astype(np.int64)
    assert np.array_equal(q.astype(p.dtype), p), "the triangulated points are not integer"
    assert np.abs(q).max() < 10**4, "coordinates too large for an exact int64 in-circle test"
    return q


def _orientation(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _incircle(P, abc, d):
    """Exact in-circle determinant of counter-clockwise triangles ``abc`` and points ``d``: > 0 inside."""
    rows = []
    for k in range(3):
        u = P[abc[:, k]] - P[d]
        rows.append((u[:, 0], u[:, 1], u[:, 0] ** 2 + u[:, 1] ** 2))
    (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = rows
    return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)


def delaunay_edge_test(P, T):
    """(number of flat triangles, number of non-Delaunay interior edges, list of cocircular polygons).

    An interior edge is cocircular when the vertex opposite it in one triangle lies exactly on the
    circumcircle of the other; a cocircular polygon is a maximal set of triangles joined by such
    edges, given as the set of its vertex indices.
    """
    T = np.asarray(T)
    o = _orientation(P, T)
    ccw = T.copy()
    ccw[o < 0] = ccw[o < 0][:, [0, 2, 1]]
    owner, shared = {}, []
    for t, tri in enumerate(ccw):
        for k in range(3):
            e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            if e in owner:
                shared.append((owner.pop(e), t, next(int(v) for v in tri if v not in e)))
            else:
                owner[e] = t
    E = np.array(shared, dtype=np.int64).reshape(-1, 3)
    det = _incircle(P, ccw[E[:, 0]], E[:, 2])
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t1, t2, _ in E[det == 0]:
        parent[find(int(t1))] = find(int(t2))
    groups = {}
    for t1, t2, _ in E[det == 0]:
        groups.setdefault(find(int(t1)), set()).update((int(t1), int(t2)))
    polygons = [frozenset(int(v) for t in g for v in T[t]) for g in groups.values()]
    return int(np.sum(o == 0)), int(np.sum(det > 0)), polygons


def differences_inside_cocircular_polygons(points, triangles, reference):
    """Compare two triangulations of the same integer points.

    Returns the number of cocircular polygons triangulated differently. Raises ``AssertionError``
    if either triangulation has a flat triangle or a non-Delaunay edge, or if they differ anywhere
    outside the cocircular polygons of ``triangles``.
    """
    P = integer_points(points)
    a, b = canonical(triangles), canonical(reference)
    assert a.shape == b.shape, f"{len(a)} triangles against {len(b)}"
    flat, bad, polygons = delaunay_edge_test(P, a)
    assert flat == 0 and bad == 0, f"this triangulation has {flat} flat triangles and {bad} non-Delaunay edges"
    flat, bad, _ = delaunay_edge_test(P, b)
    assert flat == 0 and bad == 0, f"the reference has {flat} flat triangles and {bad} non-Delaunay edges"
    differing = {tuple(t) for t in a} ^ {tuple(t) for t in b}
    touched = set()
    for t in differing:
        home = [k for k, poly in enumerate(polygons) if set(t) <= poly]
        assert home, f"triangle {t} differs outside every cocircular polygon"
        touched.add(home[0])
    return len(touched)
