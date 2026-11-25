"""Split the voxel centres of an image into the regions a triangle mesh encloses, in exact arithmetic.

Two neighbouring voxel centres are in the same region when the segment between them crosses no triangle of
the mesh; the regions are the connected components of that neighbourhood graph (6-connectivity). The answer
is a property of the mesh's own triangles: no library subdivides or samples the mesh, and every geometric
decision is the sign of a determinant, certified in floating point or recomputed with exact rationals, so
the result does not depend on the NumPy or SciPy version, nor on the machine.

A voxel centre lying exactly on a triangle, which is common because most mesh vertices sit on the voxel
lattice, is decided by moving it infinitesimally (simulation of simplicity): by `delta_0` along axis 0, then
`delta_1 << delta_0` along axis 1, then `delta_2 << delta_1` along axis 2, each towards increasing index,
except on the last voxel of an axis, which is moved towards decreasing index (into the image). The centre then
belongs to the region it is moved into. A centre on the image's outer faces is thereby moved into the image,
so a cell cut by the image box keeps its outermost voxels.

Herve Turlier's group, 2026.
"""

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Below this magnitude, integer-valued coordinates give exact float64 determinants: differences stay below
#: 2**15, the cross product's components below 2**31 and the 3-D determinant below 2**48 < 2**53.
_EXACT_INTEGER_LIMIT = 2.0**14

#: Relative error bounds under which a float64 determinant's sign is certified (Shewchuk's bounds are
#: about 3.3e-16 and 7.8e-16 times the sum of the absolute products; these are a thousand times wider).
_ORIENT2D_RELATIVE_BOUND = 1e-12
_ORIENT3D_RELATIVE_BOUND = 1e-12


def _is_small_integer(*arrays: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether every given coordinate is integer-valued and of magnitude below `_EXACT_INTEGER_LIMIT`."""
    result = np.ones(np.shape(arrays[0]), dtype=bool)
    for array in arrays:
        result &= (array == np.round(array)) & (np.abs(array) < _EXACT_INTEGER_LIMIT)
    return result


def _orient2d_fraction(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> int:
    """Exact sign of `(bx - ax)(cy - ay) - (by - ay)(cx - ax)` with rationals."""
    ax, ay, bx, by, cx, cy = (Fraction(v) for v in (ax, ay, bx, by, cx, cy))
    det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (det > 0) - (det < 0)


def sign_orient2d(
    ax: NDArray[np.float64],
    ay: NDArray[np.float64],
    bx: NDArray[np.float64],
    by: NDArray[np.float64],
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
) -> NDArray[np.int8]:
    """Exact sign of `(bx - ax)(cy - ay) - (by - ay)(cx - ax)`, elementwise.

    The float64 value decides where its magnitude exceeds the error bound, or where every input is a small
    integer (then the float64 value is exact); the remaining entries are recomputed with rationals.
    """
    left = (bx - ax) * (cy - ay)
    right = (by - ay) * (cx - ax)
    det = left - right
    sign = np.sign(det).astype(np.int8)
    certain = np.abs(det) > _ORIENT2D_RELATIVE_BOUND * (np.abs(left) + np.abs(right))
    certain |= _is_small_integer(ax, ay, bx, by, cx, cy)
    for index in np.flatnonzero(~certain):
        sign.flat[index] = _orient2d_fraction(
            ax.flat[index],
            ay.flat[index],
            bx.flat[index],
            by.flat[index],
            cx.flat[index],
            cy.flat[index],
        )
    return sign


def _orient3d_fraction(p0: NDArray, p1: NDArray, p2: NDArray, x: NDArray) -> int:
    """Exact sign of `((p1 - p0) x (p2 - p0)) . (x - p0)` with rationals."""
    a, b, c, d = ([Fraction(float(v)) for v in point] for point in (p0, p1, p2, x))
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    w = [d[i] - a[i] for i in range(3)]
    det = (u[1] * v[2] - u[2] * v[1]) * w[0] + (u[2] * v[0] - u[0] * v[2]) * w[1] + (u[0] * v[1] - u[1] * v[0]) * w[2]
    return (det > 0) - (det < 0)


def sign_orient3d(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.int8]:
    """Exact sign of `((p1 - p0) x (p2 - p0)) . (x - p0)` for `(m, 3)` arrays, row by row."""
    u, v, w = p1 - p0, p2 - p0, x - p0
    terms = np.stack(
        [
            u[:, 1] * v[:, 2] * w[:, 0],
            -u[:, 2] * v[:, 1] * w[:, 0],
            u[:, 2] * v[:, 0] * w[:, 1],
            -u[:, 0] * v[:, 2] * w[:, 1],
            u[:, 0] * v[:, 1] * w[:, 2],
            -u[:, 1] * v[:, 0] * w[:, 2],
        ],
    )
    det = (
        (u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]) * w[:, 0]
        + (u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]) * w[:, 1]
        + (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * w[:, 2]
    )
    sign = np.sign(det).astype(np.int8)
    certain = np.abs(det) > _ORIENT3D_RELATIVE_BOUND * np.abs(terms).sum(axis=0)
    certain &= np.isfinite(det)
    certain |= _is_small_integer(p0, p1, p2, x).all(axis=1)
    for index in np.flatnonzero(~certain):
        sign[index] = _orient3d_fraction(p0[index], p1[index], p2[index], x[index])
    return sign


def _normal_signs(p0: NDArray[np.float64], p1: NDArray[np.float64], p2: NDArray[np.float64]) -> NDArray[np.int8]:
    """Exact signs of the three components of `(p1 - p0) x (p2 - p0)`, as a `(m, 3)` array."""
    signs = np.empty((len(p0), 3), dtype=np.int8)
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        signs[:, axis] = sign_orient2d(p0[:, b], p0[:, c], p1[:, b], p1[:, c], p2[:, b], p2[:, c])
    return signs


def _perturbed_edge_sign(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    query: NDArray[np.float64],
    direction: NDArray[np.int8],
) -> NDArray[np.int8]:
    """Sign of `orient2d(start, end, query + (d_b delta_b, d_c delta_c))`, `delta_c << delta_b`, never zero.

    `start`, `end` and `query` are `(m, 2)` arrays in the coordinates `(b, c)`, `b < c`; `direction` holds the
    signs `(d_b, d_c)` of the infinitesimal move. The determinant gains `-(end_c - start_c) d_b delta_b` and
    `(end_b - start_b) d_c delta_c`, which decide in that order where the unperturbed value is zero. All three
    are zero only when `start` and `end` coincide, which the caller excludes (a zero-area projection).
    """
    sign = sign_orient2d(start[:, 0], start[:, 1], end[:, 0], end[:, 1], query[:, 0], query[:, 1])
    first = (-np.sign(end[:, 1] - start[:, 1]) * direction[:, 0]).astype(np.int8)
    second = (np.sign(end[:, 0] - start[:, 0]) * direction[:, 1]).astype(np.int8)
    sign = np.where(sign != 0, sign, first)
    return np.where(sign != 0, sign, second).astype(np.int8)


def _move_direction(index: NDArray[np.int64], size: int) -> NDArray[np.int8]:
    """The sign of a centre's infinitesimal move along one axis: +1, or -1 on the last voxel of the axis."""
    return np.where(index == size - 1, -1, 1).astype(np.int8)


def _side_of_plane(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    normal_signs: NDArray[np.int8],
    centre: NDArray[np.int64],
    shape: tuple[int, int, int],
) -> NDArray[np.int8]:
    """Side of the triangle's plane on which each moved voxel centre lies (+1 along the normal, -1 against)."""
    sign = sign_orient3d(p0, p1, p2, centre.astype(np.float64))
    for axis in range(3):
        moved = normal_signs[:, axis] * _move_direction(centre[:, axis], shape[axis])
        sign = np.where(sign != 0, sign, moved)
    return sign.astype(np.int8)


def _cut_neighbour_links(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    shape: tuple[int, int, int],
    axis: int,
) -> NDArray[np.bool_]:
    """Which links between neighbouring voxel centres along `axis` cross at least one triangle.

    Returns:
        NDArray[np.bool_]: the image's shape with `shape[axis] - 1` along `axis`; entry `x` is the link from
            centre `x` to centre `x + e_axis`.
    """
    b, c = (other for other in range(3) if other != axis)
    cut = np.zeros(tuple(n - 1 if i == axis else n for i, n in enumerate(shape)), dtype=bool)
    if len(triangles) == 0 or shape[axis] < 2:
        return cut
    p0, p1, p2 = (points[triangles[:, k]] for k in range(3))
    normal_signs = _normal_signs(p0, p1, p2)
    # The projection along `axis` in the coordinates (b, c): its area's sign. A zero area means the plane
    # contains the axis direction, and no moved line crosses the triangle.
    area = sign_orient2d(p0[:, b], p0[:, c], p1[:, b], p1[:, c], p2[:, b], p2[:, c])
    keep = np.flatnonzero(area != 0)
    if len(keep) == 0:
        return cut

    # Every lattice line (q_b, q_c) inside each triangle's closed bounding box in (b, c).
    corners = np.stack([p0[keep], p1[keep], p2[keep]])
    low_b = np.clip(np.ceil(corners[:, :, b].min(axis=0)), 0, shape[b] - 1).astype(np.int64)
    high_b = np.clip(np.floor(corners[:, :, b].max(axis=0)), -1, shape[b] - 1).astype(np.int64)
    low_c = np.clip(np.ceil(corners[:, :, c].min(axis=0)), 0, shape[c] - 1).astype(np.int64)
    high_c = np.clip(np.floor(corners[:, :, c].max(axis=0)), -1, shape[c] - 1).astype(np.int64)
    count_b = np.maximum(high_b - low_b + 1, 0)
    count_c = np.maximum(high_c - low_c + 1, 0)
    count = count_b * count_c
    owner = np.repeat(np.arange(len(keep)), count)
    if len(owner) == 0:
        return cut
    offset = np.arange(len(owner)) - np.repeat(np.cumsum(count) - count, count)
    q_b = low_b[owner] + offset // count_c[owner]
    q_c = low_c[owner] + offset % count_c[owner]

    triangle = keep[owner]
    query = np.stack([q_b, q_c], axis=1).astype(np.float64)
    direction = np.stack([_move_direction(q_b, shape[b]), _move_direction(q_c, shape[c])], axis=1)
    projected = [np.stack([p[triangle, b], p[triangle, c]], axis=1) for p in (p0, p1, p2)]
    expected = area[triangle]
    inside = np.ones(len(triangle), dtype=bool)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        inside &= _perturbed_edge_sign(projected[start], projected[end], query, direction) == expected
    triangle, q_b, q_c = triangle[inside], q_b[inside], q_c[inside]
    if len(triangle) == 0:
        return cut

    # Where the line meets the plane, in float64: only to choose the few centres whose side is tested exactly.
    t0, t1, t2 = p0[triangle], p1[triangle], p2[triangle]
    normal = np.cross(t1 - t0, t2 - t0)
    height = t0[:, axis] - (normal[:, b] * (q_b - t0[:, b]) + normal[:, c] * (q_c - t0[:, c])) / normal[:, axis]
    base = np.floor(height).astype(np.int64)
    n_axis = shape[axis]
    sides = []
    candidates = []
    for shift in (-1, 0, 1, 2):
        index = np.clip(base + shift, 0, n_axis - 1)
        centre = np.empty((len(triangle), 3), dtype=np.int64)
        centre[:, axis], centre[:, b], centre[:, c] = index, q_b, q_c
        sides.append(_side_of_plane(t0, t1, t2, normal_signs[triangle], centre, shape))
        candidates.append(index)
    # The link from centre i to i + 1 is cut where the moved centres lie on opposite sides of the plane.
    for k in range(3):
        lower, upper = candidates[k], candidates[k + 1]
        flips = (upper == lower + 1) & (sides[k] != sides[k + 1])
        index = [None, None, None]
        index[axis], index[b], index[c] = lower[flips], q_b[flips], q_c[flips]
        cut[tuple(index)] = True
    return cut


def enclosed_regions_of_voxel_centres(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    shape: tuple[int, int, int],
) -> tuple[NDArray[np.int64], int]:
    """Label every voxel centre of an image by the region of space the mesh's triangles enclose it in.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points in voxel units of the image (the centre of voxel
            `(i, j, k)` is the point `(i, j, k)`).
        triangles (NDArray[np.integer]): `(m, 3)` point indices. Orientation and labels are not used.
        shape (tuple[int, int, int]): the image's shape.

    Returns:
        tuple: `(regions, n_regions)`, `regions` of the image's shape with one integer per connected region
            (6-connected centres whose links cross no triangle). The numbering of the regions is arbitrary;
            only the partition is meaningful.
    """
    shape = tuple(int(n) for n in shape)
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    size = int(np.prod(shape))
    flat = np.arange(size, dtype=np.int64).reshape(shape)
    rows, columns = [], []
    for axis in range(3):
        cut = _cut_neighbour_links(points, triangles, shape, axis)
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis], upper[axis] = slice(0, -1), slice(1, None)
        rows.append(flat[tuple(lower)][~cut])
        columns.append(flat[tuple(upper)][~cut])
    rows = np.concatenate(rows)
    columns = np.concatenate(columns)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(size, size)).tocsr()
    n_regions, regions = connected_components(graph, directed=False)
    return regions.reshape(shape).astype(np.int64), int(n_regions)
