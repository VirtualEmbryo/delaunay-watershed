"""Exact integer checks that a set of tetrahedra is a Delaunay tetrahedralisation of lattice points.

dw3d tesselates points on the integer voxel lattice, where many sets of five or more points are
cospherical. There the Delaunay tetrahedralisation is not unique, and Qhull picks one by
floating-point rounding, which may differ between machines. Two Delaunay tetrahedralisations of
the same points differ only inside such cospherical sets. These helpers check, in exact int64
arithmetic (coordinates up to a few thousand voxels), that a set of tetrahedra is one of them:

* every interior face is locally Delaunay: the fourth vertex of the neighbour across the face is
  not strictly inside the circumsphere of the tetrahedron, for every pair of non-flat neighbours;
* the tetrahedra cover the same volume as a reference tetrahedralisation, without overlap
  (the sum of their absolute volumes equals the reference's, and every face is shared by at most
  two tetrahedra);
* two such tetrahedralisations differ only inside cospherical sets: every differing tetrahedron has
  a fifth point exactly on its circumsphere (`difference_lies_inside_cospherical_sets`).

The configurations with junction protection tesselate weighted points (a regular, or power,
tetrahedralisation: the lower convex hull of the points lifted to `|x|^2 - w`). The same checks
hold there with the lift `|x|^2 - w` in place of `|x|^2`, in Python integers, the weights (floats,
hence dyadic rationals) scaled by a power of two that makes every one an integer
(`count_non_regular_faces`, `difference_lies_inside_copower_sets`).
"""

import numpy as np
from numpy.typing import NDArray

# Local faces of a tetrahedron (a, b, c) and its opposite vertex d.
_LOCAL_FACES = ((1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3))


def integer_points(points: NDArray[np.floating]) -> NDArray[np.int64]:
    """The points as int64, checking that they lie exactly on the integer lattice."""
    integer = np.rint(points).astype(np.int64)
    if not np.array_equal(integer.astype(np.float64), np.asarray(points, dtype=np.float64)):
        message = "the points are not on the integer lattice; the exact checks do not apply"
        raise ValueError(message)
    return integer


def _det3(u: NDArray[np.int64], v: NDArray[np.int64], w: NDArray[np.int64]) -> NDArray[np.int64]:
    return (
        u[:, 0] * (v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1])
        - u[:, 1] * (v[:, 0] * w[:, 2] - v[:, 2] * w[:, 0])
        + u[:, 2] * (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0])
    )


def orientation(points: NDArray[np.int64], tetrahedra: NDArray[np.int64]) -> NDArray[np.int64]:
    """Six times the signed volume of each tetrahedron, exactly."""
    p = points[tetrahedra]
    return _det3(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])


def _insphere(points: NDArray[np.int64], tetrahedra: NDArray[np.int64], query: NDArray[np.int64]) -> NDArray[np.int64]:
    """The lifted 4x4 determinant of (p_i - q, |p_i - q|^2); its sign times the orientation's says inside or outside."""
    p = points[tetrahedra]
    q = points[query]
    r = [p[:, i] - q for i in range(4)]
    lift = [(x * x).sum(axis=1) for x in r]
    return (
        -lift[0] * _det3(r[1], r[2], r[3])
        + lift[1] * _det3(r[0], r[2], r[3])
        - lift[2] * _det3(r[0], r[1], r[3])
        + lift[3] * _det3(r[0], r[1], r[2])
    )


def _inside_sign() -> int:
    """The sign of insphere * orientation for a point strictly inside the circumsphere (calibrated, not assumed)."""
    points = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [1, 1, 1]], dtype=np.int64)
    tetrahedron = np.array([[0, 1, 2, 3]])
    return int(np.sign(_insphere(points, tetrahedron, np.array([4]))[0] * orientation(points, tetrahedron)[0]))


INSIDE_SIGN = _inside_sign()


def face_neighbours(tetrahedra: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """For every face shared by two tetrahedra: the two tetrahedra and the vertex of the second opposite the face.

    Raises:
        ValueError: if a face is shared by more than two tetrahedra (an overlap).
    """
    rows = np.sort(np.asarray(tetrahedra, dtype=np.int64), axis=1)
    n = len(rows)
    faces = np.concatenate([rows[:, [a, b, c]] for a, b, c, _ in _LOCAL_FACES])
    opposite = np.concatenate([rows[:, d] for *_, d in _LOCAL_FACES])
    owner = np.tile(np.arange(n, dtype=np.int64), 4)
    order = np.lexsort(faces.T[::-1])
    faces, opposite, owner = faces[order], opposite[order], owner[order]
    same_as_next = np.all(faces[1:] == faces[:-1], axis=1)
    if np.any(same_as_next[1:] & same_as_next[:-1]):
        message = "a face is shared by more than two tetrahedra"
        raise ValueError(message)
    first = np.flatnonzero(same_as_next)
    return owner[first], owner[first + 1], opposite[first + 1]


def count_non_delaunay_faces(points: NDArray[np.int64], tetrahedra: NDArray[np.int64]) -> int:
    """Number of interior faces whose neighbouring vertex lies strictly inside a non-flat tetrahedron's circumsphere."""
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    first, second, opposite = face_neighbours(tetrahedra)
    orient = orientation(points, tetrahedra)
    keep = (orient[first] != 0) & (orient[second] != 0)
    first, opposite = first[keep], opposite[keep]
    sign = np.sign(_insphere(points, tetrahedra[first], opposite)) * np.sign(orient[first])
    return int(np.count_nonzero(sign == INSIDE_SIGN))


def covered_volume_times_six(points: NDArray[np.int64], tetrahedra: NDArray[np.int64]) -> int:
    """Six times the total unsigned volume of the tetrahedra, exactly (Python integer)."""
    return int(np.abs(orientation(points, np.asarray(tetrahedra, dtype=np.int64))).astype(object).sum())


def differing_tetrahedra(first: NDArray[np.int64], second: NDArray[np.int64]) -> int:
    """How many tetrahedra (as vertex sets) are in one set and not the other, counted in both directions."""
    a = {tuple(row) for row in np.sort(np.asarray(first, dtype=np.int64), axis=1).tolist()}
    b = {tuple(row) for row in np.sort(np.asarray(second, dtype=np.int64), axis=1).tolist()}
    return len(a ^ b)


def _on_circumsphere(
    points: NDArray[np.int64],
    tetrahedra: NDArray[np.int64],
    candidates: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """For each non-flat tetrahedron, whether a candidate other than its own vertices lies exactly on its sphere."""
    found = np.zeros(len(tetrahedra), dtype=bool)
    for query in candidates:
        own = np.any(tetrahedra == query, axis=1)
        on = _insphere(points, tetrahedra, np.full(len(tetrahedra), query)) == 0
        found |= on & ~own
    return found


def _cocircular(points: NDArray[np.int64], tetrahedron: NDArray[np.int64]) -> bool:
    """Whether the four points of a flat tetrahedron lie on one circle (or one line), in Python integers.

    The sphere through three of them and a point off their plane meets the plane in their circle,
    so the fourth lies on that circle exactly when it lies on that sphere.
    """
    p = [[int(c) for c in points[v]] for v in tetrahedron]
    u = [p[1][k] - p[0][k] for k in range(3)]
    w = [p[2][k] - p[0][k] for k in range(3)]
    # The cross product in Python integers (numpy 2.0's np.cross refuses object arrays).
    normal = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
    if not any(normal):
        return True
    lifted = np.array([*p[:3], [p[0][k] + normal[k] for k in range(3)], p[3]], dtype=object)
    return int(_insphere(lifted, np.array([[0, 1, 2, 3]]), np.array([4]))[0]) == 0


def difference_lies_inside_cospherical_sets(
    points: NDArray[np.int64],
    first: NDArray[np.int64],
    second: NDArray[np.int64],
) -> bool:
    """Whether two tetrahedralisations of the same lattice points differ only inside cospherical sets, exactly.

    True when both are Delaunay (no strictly non-Delaunay interior face), they cover the same volume,
    and every tetrahedron present in one and not the other has, among the vertices of the differing
    tetrahedra, a fifth point exactly on its circumsphere (a flat one: its four points on one circle).
    The Delaunay subdivision of a point set is unique and two Delaunay tetrahedralisations differ only
    in how they split its cells of five or more cospherical points, so the certificate is complete:
    any difference outside such cells makes it False.
    """
    first = np.sort(np.asarray(first, dtype=np.int64), axis=1)
    second = np.sort(np.asarray(second, dtype=np.int64), axis=1)
    if count_non_delaunay_faces(points, first) or count_non_delaunay_faces(points, second):
        return False
    if covered_volume_times_six(points, first) != covered_volume_times_six(points, second):
        return False
    a = {tuple(row) for row in first.tolist()}
    b = {tuple(row) for row in second.tolist()}
    differing = np.array(sorted(a ^ b), dtype=np.int64).reshape(-1, 4)
    if len(differing) == 0:
        return True
    candidates = np.unique(differing)
    flat = orientation(points, differing) == 0
    if not np.all(_on_circumsphere(points, differing[~flat], candidates)):
        return False
    return all(_cocircular(points, tetrahedron) for tetrahedron in differing[flat])


def exact_weights(weights: NDArray[np.floating]) -> tuple[list[int], int]:
    """The weights as Python integers `W_i = S * w_i`, `S` the smallest power of two that makes them all integers."""
    ratios = [float(w).as_integer_ratio() for w in np.asarray(weights, dtype=np.float64)]
    scale = max((denominator for _, denominator in ratios), default=1)
    return [numerator * (scale // denominator) for numerator, denominator in ratios], scale


def _power_insphere(
    points: NDArray[np.int64],
    scaled_weights: list[int],
    scale: int,
    tetrahedra: NDArray[np.int64],
    query: NDArray[np.int64],
) -> NDArray[np.object_]:
    """`_insphere` with the lift `S |p_i - q|^2 - (W_i - W_q)`: the sign says whether `q` is inside the power sphere.

    With all weights zero it is `S` times `_insphere`, so `INSIDE_SIGN` applies unchanged. Python integers.
    """
    p = points[tetrahedra].astype(object)
    q = points[query].astype(object)
    weights = np.array(scaled_weights, dtype=object)
    r = [p[:, i] - q for i in range(4)]
    lift = [scale * (x * x).sum(axis=1) - (weights[tetrahedra[:, i]] - weights[query]) for i, x in enumerate(r)]
    return (
        -lift[0] * _det3(r[1], r[2], r[3])
        + lift[1] * _det3(r[0], r[2], r[3])
        - lift[2] * _det3(r[0], r[1], r[3])
        + lift[3] * _det3(r[0], r[1], r[2])
    )


def count_non_regular_faces(
    points: NDArray[np.int64],
    weights: NDArray[np.floating],
    tetrahedra: NDArray[np.int64],
) -> int:
    """Number of interior faces whose neighbouring vertex lies strictly inside a non-flat tetrahedron's power sphere."""
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    scaled, scale = exact_weights(weights)
    first, second, opposite = face_neighbours(tetrahedra)
    orient = orientation(points, tetrahedra)
    keep = (orient[first] != 0) & (orient[second] != 0)
    first, opposite = first[keep], opposite[keep]
    power = _power_insphere(points, scaled, scale, tetrahedra[first], opposite)
    sign = np.array([(v > 0) - (v < 0) for v in power], dtype=np.int64) * np.sign(orient[first])
    return int(np.count_nonzero(sign == INSIDE_SIGN))


def _lifted_points_on_one_plane(points: NDArray[np.int64], scaled: list[int], scale: int, tetrahedron) -> bool:
    """Whether the four lifted points `(p, S |p|^2 - W)` of a flat tetrahedron lie on one 2-plane (Python integers).

    The power analogue of four coplanar points on one circle: the three difference vectors in 4D have
    rank at most 2, i.e. every 3 x 3 minor of their 3 x 4 matrix vanishes.
    """
    lifted = [
        [int(c) for c in points[v]] + [scale * sum(int(c) * int(c) for c in points[v]) - scaled[v]] for v in tetrahedron
    ]
    rows = [[lifted[i][k] - lifted[0][k] for k in range(4)] for i in (1, 2, 3)]
    for skip in range(4):
        m = [[row[k] for k in range(4) if k != skip] for row in rows]
        det = (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
        if det:
            return False
    return True


def difference_lies_inside_copower_sets(
    points: NDArray[np.int64],
    weights: NDArray[np.floating],
    first: NDArray[np.int64],
    second: NDArray[np.int64],
) -> bool:
    """Whether two regular tetrahedralisations of the same weighted lattice points differ only inside degenerate sets.

    The weighted counterpart of `difference_lies_inside_cospherical_sets`: True when both use the same
    vertices (a regular tetrahedralisation may leave a point out), neither has a strictly non-regular
    interior face, they cover the same volume, and every tetrahedron present in one and not the other has
    a fifth point exactly on its power sphere (a flat one: its four lifted points on one 2-plane).
    """
    first = np.sort(np.asarray(first, dtype=np.int64), axis=1)
    second = np.sort(np.asarray(second, dtype=np.int64), axis=1)
    if not np.array_equal(np.unique(first), np.unique(second)):
        return False
    if count_non_regular_faces(points, weights, first) or count_non_regular_faces(points, weights, second):
        return False
    if covered_volume_times_six(points, first) != covered_volume_times_six(points, second):
        return False
    a = {tuple(row) for row in first.tolist()}
    b = {tuple(row) for row in second.tolist()}
    differing = np.array(sorted(a ^ b), dtype=np.int64).reshape(-1, 4)
    if len(differing) == 0:
        return True
    scaled, scale = exact_weights(weights)
    candidates = np.unique(differing)
    flat = orientation(points, differing) == 0
    found = np.zeros(int(np.count_nonzero(~flat)), dtype=bool)
    solid = differing[~flat]
    for query in candidates:
        own = np.any(solid == query, axis=1)
        power = _power_insphere(points, scaled, scale, solid, np.full(len(solid), query))
        found |= np.array([v == 0 for v in power], dtype=bool) & ~own
    if not np.all(found):
        return False
    return all(_lifted_points_on_one_plane(points, scaled, scale, tetrahedron) for tetrahedron in differing[flat])
