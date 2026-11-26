"""A Delaunay triangulation of dw2d's integer points that is the same on every machine.

dw2d triangulates integer pixel points. Where four or more of them lie on one empty circle, the
Delaunay subdivision has a convex cell with k >= 4 vertices, every triangulation of that cell is a
valid Delaunay triangulation, and Qhull picks one by floating-point rounding: on macOS arm64 it
picks differently from Linux x86-64, which changes the mesh. Qhull's row
order of the triangles is arbitrary too, and the watershed meets equal scores in that order.

``deterministic_triangles`` keeps Qhull's triangles outside the cocircular cells, re-triangulates
every cell by a fixed rule, and lists the triangles in canonical order (each row sorted, rows in
lexicographic order). The cells are found with exact integer in-circle tests, so the result
depends only on the points and their order (scikit-image's ``peak_local_max`` sorts with a stable
argsort, so that order is the same on every machine), never on which valid triangulation Qhull
returned or on its row order.

The rule (chosen on the development split of the equilibrium-foam
benchmark among three pre-registered rules): the fan from the cell vertex ``ring[h mod k]``, where
``ring`` lists the k vertices counter-clockwise from the lexicographically smallest (row, col),
and ``h`` is the first 8 bytes (little-endian) of SHA-256 of the cell's vertex coordinates, sorted
lexicographically, as little-endian int64. For a quadrilateral each diagonal is taken for half of
the apexes, so no lattice direction is preferred: on 4 491 quadrilaterals whose two diagonals lie
on either side of 90 degrees, 50.4 % took the one in [0, 90) degrees (Qhull: 49.6 %; a fan from
the lexicographically smallest vertex, the usual symbolic perturbation: 90.6 %).
"""
from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

_INT64_SAFE_SPAN = 2**14  # in-circle terms are bounded by 12 * span**4 < 2**63


class TriangulationOfPoints:
    """The two attributes of ``scipy.spatial.Delaunay`` that dw2d uses: ``points`` and ``simplices``."""

    def __init__(self, points: NDArray, simplices: NDArray):
        self.points = points
        self.simplices = simplices


def _integer_points(points: NDArray) -> NDArray:
    p = np.asarray(points)
    q = np.rint(p).astype(np.int64)
    if not np.array_equal(q.astype(p.dtype), p):
        raise ValueError("the triangulated points are not integer")
    if len(q) and int(q.max()) - int(q.min()) >= _INT64_SAFE_SPAN:
        return q.astype(object)  # exact Python integers
    return q


def _orientation(P: NDArray, T: NDArray) -> NDArray:
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _incircle(P: NDArray, abc: NDArray, d: NDArray) -> NDArray:
    """Exact in-circle determinant of counter-clockwise triangles ``abc`` and points ``d``: > 0 inside, 0 on."""
    rows = []
    for k in range(3):
        u = P[abc[:, k]] - P[d]
        rows.append((u[:, 0], u[:, 1], u[:, 0] ** 2 + u[:, 1] ** 2))
    (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = rows
    return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)


def cocircular_cells(points: NDArray, simplices: NDArray) -> list[dict]:
    """The cells of the Delaunay subdivision with four or more vertices, from any Delaunay triangulation.

    Returns a list of dict(ring, triangles): ``ring`` the vertex indices counter-clockwise from the
    lexicographically smallest (row, col), ``triangles`` the rows of ``simplices`` covering the
    cell. Raises ValueError if ``simplices`` has a flat triangle or is not Delaunay.
    """
    P = _integer_points(points)
    T = np.asarray(simplices, dtype=np.int64).copy()
    o = _orientation(P, T)
    if np.any(o == 0):
        raise ValueError("flat triangle in the triangulation")
    T[o < 0] = T[o < 0][:, [0, 2, 1]]
    owner, shared = {}, []
    for t, tri in enumerate(T):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            e = (min(a, b), max(a, b))
            if e in owner:
                shared.append((owner.pop(e), t, int(next(v for v in tri if v not in e))))
            else:
                owner[e] = t
    E = np.array(shared, dtype=np.int64).reshape(-1, 3)
    det = _incircle(P, T[E[:, 0]], E[:, 2]) if len(E) else np.zeros(0, np.int64)
    if np.any(det > 0):
        raise ValueError("the triangulation is not Delaunay")
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cocircular = E[det == 0]
    for t1, t2, _ in cocircular:
        parent[find(int(t1))] = find(int(t2))
    groups: dict[int, set] = {}
    for t1, t2, _ in cocircular:
        groups.setdefault(find(int(t1)), set()).update((int(t1), int(t2)))
    cells = []
    for g in groups.values():
        tris = sorted(g)
        directed = {(int(u), int(v)) for t in tris for u, v in ((T[t, 0], T[t, 1]), (T[t, 1], T[t, 2]), (T[t, 2], T[t, 0]))}
        nxt = {u: v for (u, v) in directed if (v, u) not in directed}
        vertices = {int(v) for t in tris for v in T[t]}
        if len(nxt) != len(vertices) or len(tris) != len(vertices) - 2:
            raise ValueError("a cocircular cell is not a triangulated convex polygon")
        start = min(vertices, key=lambda v: (int(P[v, 0]), int(P[v, 1])))
        ring, v = [start], nxt[start]
        while v != start:
            ring.append(v)
            v = nxt[v]
        cells.append(dict(ring=ring, triangles=tris))
    cells.sort(key=lambda c: c["ring"])
    return cells


def fan_apex(points: NDArray, ring: list[int]) -> int:
    """Position in ``ring`` of the fan apex: SHA-256 of the sorted integer vertex coordinates, modulo k."""
    coords = np.array(sorted((int(round(points[v][0])), int(round(points[v][1]))) for v in ring), dtype="<i8")
    h = int.from_bytes(hashlib.sha256(coords.tobytes()).digest()[:8], "little")
    return h % len(ring)


def canonical_order(triangles: NDArray) -> NDArray:
    """Each row sorted ascending, rows in lexicographic order."""
    s = np.sort(np.asarray(triangles, dtype=np.int64), axis=1)
    return s[np.lexsort(s.T[::-1])]


def deterministic_triangles(points: NDArray, simplices: NDArray) -> NDArray:
    """Qhull's triangles with every cocircular cell re-triangulated by the rule, in canonical order."""
    cells = cocircular_cells(points, simplices)
    T = np.asarray(simplices, dtype=np.int64)
    keep = np.ones(len(T), bool)
    new = []
    for c in cells:
        keep[c["triangles"]] = False
        k = fan_apex(points, c["ring"])
        r = c["ring"][k:] + c["ring"][:k]
        new += [(r[0], r[i], r[i + 1]) for i in range(1, len(r) - 1)]
    return canonical_order(np.vstack([T[keep], np.array(new, dtype=np.int64).reshape(-1, 3)]))
