r"""Exact nonlocal triangle-triangle intersection for a surface mesh.

Two triangles that share a vertex index are **adjacent** and are never tested: adjacent triangles
touch by construction, and in a correct multimaterial mesh three triangles meet along every
trijunction edge, so counting those would report the intended non-manifold structure as a defect.
Every other pair is nonlocal and is tested exactly.

Why the library carries this and not only the benchmarks
--------------------------------------------------------
Every other validity property a mesh consumer can check cheaply is local or combinatorial:
watertightness, non-manifold edges, degenerate and duplicate faces, interface identity. A mesh whose
two *distant* sheets pass through each other satisfies all of them. Measured on this package's own
benchmark cohort: all twelve local predicates stayed bit-identical on 40 of 40 cases while 23
nonlocal crossings existed on 5 of them, introduced by a post-process whose guard checked only the
triangles incident to the vertex it moved.

**A local guard cannot certify a nonlocal property.** :mod:`dw3d.junction_relocation` therefore
uses this predicate as its own acceptance test, which is why the predicate belongs to the library
rather than to a benchmark package the library must not depend on.

The test
--------
For each candidate pair, the exact Moller triangle-triangle predicate:

1. Signed distances of triangle ``B``'s vertices to triangle ``A``'s plane. If all three carry the
   same strict sign, that plane separates them and they cannot meet.
2. The same with the roles exchanged.
3. If either triangle is **coplanar** with the other's plane, fall back to a 2-D test in that
   plane: any edge-edge crossing, or either triangle's vertex inside the other.
4. Otherwise both triangles cross the line where the two planes meet. Each cuts an interval on that
   line, and the triangles meet iff the two intervals overlap.

Steps 1, 2 and 4 are the standard construction and are exact up to floating point. The predicate is
checked against an independent Moller-Trumbore segment-triangle oracle that shares no arithmetic
with it, over random pairs; that differential test lives beside the benchmark evaluator.

Complexity
----------
Candidate pairs come from a ``cKDTree`` on triangle centroids queried at twice the largest
circumradius about the centroid. That bound is sound: two triangles whose centroids are further
apart than the sum of their circumradii cannot meet, and twice the largest circumradius is at least
that sum. The exact predicate then runs vectorised over the candidate set.

Units, tolerances and failure modes
-----------------------------------
Coordinates are the mesh's own units -- mask voxels for a reconstruction read in the mask frame --
and all arithmetic is float64. :data:`CONTACT_TOLERANCE` is an absolute distance in those units: a
pair whose contact is within it is reported as *touching* rather than counted as an intersection,
and both numbers are emitted so a borderline pair cannot hide inside a single total. A degenerate
triangle has no well-defined plane; such triangles are excluded from the test and **counted
separately**, because silently skipping one would let a sliver hide a crossing. A mesh with fewer
than two triangles returns zero counts.

A self-intersection is **not** affine-invariant in general, so a mesh and its registered copy are
two different meshes for this predicate and must be reported as such.

The count is a **count, never a score**, and no threshold is proposed here or anywhere else.

No new dependency: numpy and scipy only, both already required.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

#: Absolute distance, in the mesh's own units, within which a contact is reported as *touching*
#: rather than counted as an intersection. Tight on purpose, and both counts are emitted.
CONTACT_TOLERANCE = 1e-9
#: A triangle whose twice-area falls below this has no usable plane; it is excluded and counted.
DEGENERATE_TWICE_AREA = 1e-12

_TRIANGLE_VERTICES = 3

#: The keys :func:`find_self_intersections` emits as plain numbers, in a fixed order.
SELF_INTERSECTION_KEYS: tuple[str, ...] = (
    "n_self_intersecting_triangle_pairs",
    "n_touching_triangle_pairs",
    "n_self_intersection_candidate_pairs",
    "n_self_intersection_tested_pairs",
    "n_self_intersection_degenerate_triangles",
    "self_intersection_search_radius",
)


def count_self_intersections(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
) -> dict:
    """The nonlocal self-intersection counts, without the pair arrays.

    A thin, JSON-serialisable projection of :func:`find_self_intersections` carrying exactly
    :data:`SELF_INTERSECTION_KEYS` and no array.

    Args:
        points (NDArray[np.float64]): vertex coordinates, the mesh's own units.
        triangles (NDArray[np.integer]): triangle vertex indices.

    Returns:
        dict: the six keys of :data:`SELF_INTERSECTION_KEYS`. A count, never a score.
    """
    found = find_self_intersections(points, triangles)
    return {key: found[key] for key in SELF_INTERSECTION_KEYS}


def triangle_centroids_and_circumradii(
    vertices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each triangle's centroid and the radius of the smallest ball about it containing it.

    Args:
        vertices (NDArray[np.float64]): triangle corner coordinates, `(m, 3, 3)`.

    Returns:
        tuple: `(centroids, radii)`, shapes `(m, 3)` and `(m,)`. Two triangles whose centroids are
        further apart than the sum of their radii cannot intersect, which is what makes a
        centroid-radius neighbour search a sound filter rather than a heuristic.
    """
    centroids = vertices.mean(axis=1)
    radii = np.linalg.norm(vertices - centroids[:, None, :], axis=2).max(axis=1)
    return centroids, radii


def twice_areas(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Twice each triangle's area, from the cross product of two of its edges.

    Args:
        vertices (NDArray[np.float64]): triangle corner coordinates, `(m, 3, 3)`.

    Returns:
        NDArray[np.float64]: one value per triangle; below :data:`DEGENERATE_TWICE_AREA` the
        triangle has no usable plane.
    """
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    return np.linalg.norm(normals, axis=1)


def drop_adjacent_pairs(
    pairs: NDArray[np.int64],
    triangles: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Remove the pairs that share at least one vertex index.

    Args:
        pairs (NDArray[np.int64]): candidate index pairs, `(k, 2)`, into `triangles`.
        triangles (NDArray[np.int64]): triangle vertex indices.

    Returns:
        NDArray[np.int64]: the nonlocal subset, `(k', 2)`.
    """
    if len(pairs) == 0:
        return pairs
    a = triangles[pairs[:, 0]]
    b = triangles[pairs[:, 1]]
    shares = (a[:, :, None] == b[:, None, :]).any(axis=(1, 2))
    return pairs[~shares]


def _plane_of(vertices: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit normal and offset `d` with `n . x + d = 0`, for each triangle."""
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths > 0, lengths, 1.0)
    unit = normals / lengths
    offset = -np.einsum("ij,ij->i", unit, vertices[:, 0])
    return unit, offset


def _interval_on_line(
    distances: NDArray[np.float64],
    projections: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The interval a triangle cuts on the intersection line of the two planes.

    `distances` are its three vertices' signed distances to the *other* plane and `projections`
    their coordinates along the intersection line's direction.

    Every edge whose endpoints' signed distances have a non-positive product straddles that plane
    and contributes its crossing point; the interval is the extent of those crossings. This is used
    in preference to locating the odd-one-out apex because a vertex lying exactly **on** the other
    plane makes the apex ill-defined while leaving the interval perfectly well-defined. An edge
    lying wholly in the plane contributes its own endpoints, since both its neighbouring edges then
    straddle too.
    """
    n = len(distances)
    lows = np.full(n, np.inf)
    highs = np.full(n, -np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = distances[:, i], distances[:, j]
        pi, pj = projections[:, i], projections[:, j]
        straddles = (di * dj) <= 0
        denominator = di - dj
        t = np.where(np.abs(denominator) > 0, di / np.where(denominator != 0, denominator, 1.0), 0.0)
        crossing = pi + t * (pj - pi)
        lows = np.where(straddles, np.minimum(lows, crossing), lows)
        highs = np.where(straddles, np.maximum(highs, crossing), highs)
    return lows, highs


def _coplanar_intersects(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """2-D overlap test for coplanar triangle pairs, dropped into the plane's dominant axes."""
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    drop = np.argmax(np.abs(normal), axis=1)
    keep = np.stack([(drop + 1) % 3, (drop + 2) % 3], axis=1)
    rows = np.arange(len(a))[:, None, None]
    axis = keep[:, None, :]
    a2 = a[rows, np.arange(_TRIANGLE_VERTICES)[None, :, None], axis]
    b2 = b[rows, np.arange(_TRIANGLE_VERTICES)[None, :, None], axis]

    def cross(o, p, q):  # noqa: ANN001, ANN202
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (
            q[..., 0] - o[..., 0]
        )

    hit = np.zeros(len(a), dtype=bool)
    for i in range(_TRIANGLE_VERTICES):
        for j in range(_TRIANGLE_VERTICES):
            p1, p2 = a2[:, i], a2[:, (i + 1) % 3]
            q1, q2 = b2[:, j], b2[:, (j + 1) % 3]
            d1, d2 = cross(p1, p2, q1), cross(p1, p2, q2)
            d3, d4 = cross(q1, q2, p1), cross(q1, q2, p2)
            hit |= ((d1 * d2) < 0) & ((d3 * d4) < 0)

    def contains(tri, point):  # noqa: ANN001, ANN202
        s1 = cross(tri[:, 0], tri[:, 1], point)
        s2 = cross(tri[:, 1], tri[:, 2], point)
        s3 = cross(tri[:, 2], tri[:, 0], point)
        return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))

    for i in range(_TRIANGLE_VERTICES):
        hit |= contains(a2, b2[:, i])
        hit |= contains(b2, a2[:, i])
    return hit


def pair_intersection_flags(
    vertices: NDArray[np.float64],
    pairs: NDArray[np.int64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Run the exact predicate on an explicit list of triangle pairs.

    This is the kernel both the whole-mesh certificate and any incremental guard must share, so
    that a guard and the certificate judging it cannot disagree about what an intersection *is*.
    Adjacency is **not** filtered here: the caller decides which pairs are nonlocal.

    Args:
        vertices (NDArray[np.float64]): triangle corner coordinates, `(m, 3, 3)`, float64.
        pairs (NDArray[np.int64]): index pairs into `vertices`, `(k, 2)`.

    Returns:
        tuple: `(intersects, touching)`, boolean arrays of length `k`. A pair is *touching* when
        its contact lies within :data:`CONTACT_TOLERANCE`; it is then not counted as an
        intersection, and both flags are false for a separated pair.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    intersects = np.zeros(len(pairs), dtype=bool)
    touching = np.zeros(len(pairs), dtype=bool)
    if len(pairs) == 0:
        return intersects, touching

    a = vertices[pairs[:, 0]]
    b = vertices[pairs[:, 1]]
    normal_a, offset_a = _plane_of(a)
    normal_b, offset_b = _plane_of(b)

    dist_b = np.einsum("ij,ikj->ik", normal_a, b) + offset_a[:, None]
    dist_a = np.einsum("ij,ikj->ik", normal_b, a) + offset_b[:, None]

    # steps 1 and 2: a strictly one-sided triangle is separated by the other's plane
    sep_b = np.all(dist_b > CONTACT_TOLERANCE, axis=1) | np.all(dist_b < -CONTACT_TOLERANCE, axis=1)
    sep_a = np.all(dist_a > CONTACT_TOLERANCE, axis=1) | np.all(dist_a < -CONTACT_TOLERANCE, axis=1)
    alive = ~(sep_a | sep_b)

    coplanar = alive & (
        np.all(np.abs(dist_b) <= CONTACT_TOLERANCE, axis=1)
        | np.all(np.abs(dist_a) <= CONTACT_TOLERANCE, axis=1)
    )
    where = np.flatnonzero(coplanar)
    if len(where):
        touching[where] = _coplanar_intersects(a[where], b[where], normal_a[where])

    crossing = alive & ~coplanar
    where = np.flatnonzero(crossing)
    if len(where):
        direction = np.cross(normal_a[where], normal_b[where])
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        length = np.where(length > 0, length, 1.0)
        direction = direction / length
        proj_a = np.einsum("ij,ikj->ik", direction, a[where])
        proj_b = np.einsum("ij,ikj->ik", direction, b[where])
        lo_a, hi_a = _interval_on_line(dist_a[where], proj_a)
        lo_b, hi_b = _interval_on_line(dist_b[where], proj_b)
        overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
        finite = np.isfinite(overlap)
        intersects[where] = finite & (overlap > CONTACT_TOLERANCE)
        touching[where] = finite & (overlap > -CONTACT_TOLERANCE) & (overlap <= CONTACT_TOLERANCE)

    return intersects, touching


def find_self_intersections(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
) -> dict:
    """Every nonlocal triangle-triangle intersection of one surface mesh, and the pairs themselves.

    Args:
        points (NDArray[np.float64]): vertex coordinates, the mesh's own units.
        triangles (NDArray[np.integer]): triangle vertex indices, `(m, 3)`.

    Returns:
        dict: the six counts of :data:`VALIDITY_KEYS` plus `intersecting_pairs` and
        `touching_pairs`, each an `(k, 2)` array of indices **into the input `triangles`**, so a
        caller can draw or diagnose a specific crossing. Degenerate triangles are excluded from the
        test and counted.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    empty_pairs = np.zeros((0, 2), dtype=np.int64)
    empty = {
        "n_self_intersecting_triangle_pairs": 0,
        "n_touching_triangle_pairs": 0,
        "n_self_intersection_candidate_pairs": 0,
        "n_self_intersection_tested_pairs": 0,
        "n_self_intersection_degenerate_triangles": 0,
        "self_intersection_search_radius": 0.0,
        "intersecting_pairs": empty_pairs,
        "touching_pairs": empty_pairs,
    }
    if len(triangles) < 2:
        return empty

    vertices = points[triangles]
    healthy = twice_areas(vertices) > DEGENERATE_TWICE_AREA
    n_degenerate = int((~healthy).sum())
    live = np.flatnonzero(healthy)
    if len(live) < 2:
        return {**empty, "n_self_intersection_degenerate_triangles": n_degenerate}

    vertices = vertices[live]
    centroids, radii = triangle_centroids_and_circumradii(vertices)
    radius = float(2.0 * radii.max())

    tree = cKDTree(centroids)
    pairs = np.asarray(
        tree.query_pairs(r=radius, output_type="ndarray"), dtype=np.int64,
    ).reshape(-1, 2)
    n_candidates = len(pairs)
    pairs = drop_adjacent_pairs(pairs, triangles[live])
    if len(pairs) == 0:
        return {
            **empty,
            "n_self_intersection_candidate_pairs": n_candidates,
            "n_self_intersection_degenerate_triangles": n_degenerate,
            "self_intersection_search_radius": radius,
        }

    intersects, touching = pair_intersection_flags(vertices, pairs)
    return {
        "n_self_intersecting_triangle_pairs": int(intersects.sum()),
        "n_touching_triangle_pairs": int(touching.sum()),
        "n_self_intersection_candidate_pairs": n_candidates,
        "n_self_intersection_tested_pairs": len(pairs),
        "n_self_intersection_degenerate_triangles": n_degenerate,
        "self_intersection_search_radius": radius,
        "intersecting_pairs": live[pairs[intersects]],
        "touching_pairs": live[pairs[touching]],
    }
