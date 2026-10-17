r"""Move a finished mesh's trijunction vertices onto the trijunction curves the mask itself gives.

What it does, and what it deliberately does not
-----------------------------------------------
:func:`relocate_junction_vertices` changes ``points`` and **never** ``triangles`` or ``labels``. The
connectivity is frozen, which is the whole argument for this construction over earlier attempts to
improve junction geometry in this package: those moved *tessellation* points and let the Delaunay
re-bracket the junction line, so the line could move the wrong way when the bracket changed. Here
there is no bracket to change, and the treated mesh can be hashed against its own base.

The target is read from the **mask**, through :func:`dw3d.junction_curves.extract_junction_curves`,
and only its ``matched`` curves are used -- the ones with exactly one counterpart among the mesh's
own trijunctions of the same material triple. ``invented`` curves are never used as targets;
``fragmented`` curves are left untouched. No ground truth enters at any point: nothing here reads a
reference mesh, a tension, a pressure or a Neumann angle, and the whole path has been re-run from a
directory holding only the label image with a bit-identical result.

Two guards, and why both are needed
-----------------------------------
**Local.** Moves are applied one vertex at a time in ascending vertex index, against the
configuration *as it stands* rather than against the original, so two neighbouring junction vertices
cannot both be accepted on the strength of a geometry neither will end up in. A move is accepted
only if every triangle incident to that vertex keeps its normal orientation and keeps its area above
:data:`MIN_TRIANGLE_AREA`, and if it leaves no cell on either side of those triangles with a signed
volume of zero or less (a cell that already had one before the pass is not constrained). A rejected
move is **dropped, never repaired**: the vertex stays where it was.

The cell clause exists because the triangle clause is blind to it: on a real embryo timepoint a cell
made of a single tetrahedron had one vertex moved 6.7 voxels across the opposite face, every incident
triangle kept a normal within 49 degrees of its previous one, and the cell's volume went from +35 to
-5.8 cubic voxels. Its surface stayed closed, oriented, manifold and free of crossings, so no other
check in the package saw it. The clause tests only the sign, deliberately: a floor at a fraction of the
cell's volume before relocation was tried and refused more, because for the small cells that need it
that volume is itself wrong, and a refused move freezes a vertex shared by several cells.

**Nonlocal.** A local guard cannot certify a nonlocal property. A move can drive one sheet of the
mesh through a distant sheet while every triangle incident to the moved vertex stays perfectly
oriented; measured on this package's benchmark cohort, all twelve local validity predicates stayed
bit-identical on 40 of 40 cases while 23 such crossings existed on 5 of them. The second guard
closes that: moving one vertex ``v`` changes the geometry of exactly the triangles incident to
``v``, so every non-adjacent triangle pair with neither member incident to ``v`` has the same status
before and after. The change in the whole-mesh crossing count therefore **equals** the change
restricted to the pairs with at least one member incident to ``v`` -- an exact identity, not a
neighbourhood approximation, because those partners range over the entire mesh. Accepting a move
only when that restricted change is non-positive makes the crossing count **non-increasing along
the accepted sequence by construction**.

That argument is checked rather than trusted: with ``verify=True`` the whole-mesh certificate of
:mod:`dw3d.triangle_intersection` runs before and after, both counts are reported, and a rise raises
rather than being absorbed.

What this costs, and what it buys
---------------------------------
Measured over 40 equilibrium benchmark cases at four reconstruction spacings and on four point
samplers, 16 independent adjudications. What the pass costs is best stated in seconds, because the
reconstruction it follows varies by an order of magnitude between volumes: on the benchmark meshes,
whose reconstruction takes about 5 s, it adds about 0.75 s (+15 %); on real embryo volumes, whose
reconstruction takes about 0.6 s, it adds 0.1 to 1.5 s per timepoint (median 0.34 s and 0.48 s on the
two measured time series, +27 % to +208 %). Within one real series it grows faster than linearly with
the number of triangles -- a fitted exponent of 2.68 over a 2.2-fold range of mesh sizes in a fixed
imaging volume, which describes that series and is not a tissue-scale law. About half of it is the
nonlocal guard's candidate-pair search and a third the mask trijunction-curve extraction.
The nonlocal guard refuses between 0.04 % and 0.21 % of the moves the local guard accepts, and that
is enough to take the crossing count to zero everywhere. Line position error, local tangent error
and discrete curvature all improve with their confidence intervals clear of zero at every spacing
and on every sampler, and no pre-existing validity predicate changes on any case. On the seed-free
samplers it also improves contact angles by 2.3 to 3.5 degrees and gauge-matched tension error by
20-40 %; on the seeded default sampler the line improves and the angle does not move significantly.

Quadruple points
----------------
A junction vertex lying on two or more matched trijunction lines has targets that disagree. The rule
is to **leave it where it is**, and report how many there are. Nothing here invents a consensus
position for such a vertex.

Units and conventions
---------------------
All coordinates are mask voxel units -- the frame a reconstructed ``dw3d`` mesh's ``points`` live in
and the frame :func:`~dw3d.junction_curves.extract_junction_curves` emits its polylines in. That
extractor refuses a mesh whose coordinates do not lie inside the mask's extent, which catches most
permuted axis orders; a mesh scaled down into the extent, or permuted on a mask whose sides are
nearly equal, passes that check. Areas are square voxels. Vertex order, array shape and dtype are
preserved exactly.

Failure modes
-------------
A mesh with no matched curve is returned unchanged with ``n_targets = 0``; that is reported, not
raised. A matched curve of fewer than two vertices carries no segment and is skipped and counted. A
vertex with no incident triangle is skipped. The thresholds here are fixed and are not tuning
parameters; they must not be adjusted to improve a reported metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from dw3d.junction_curves import Estimator
from dw3d.triangle_intersection import (
    DEGENERATE_TWICE_AREA,
    drop_adjacent_pairs,
    find_self_intersections,
    pair_intersection_flags,
    triangle_centroids_and_circumradii,
    twice_areas,
)

#: A move that would take an incident triangle's area below this is refused as degenerating it.
#: Square voxels. Fixed, not a tuning parameter.
MIN_TRIANGLE_AREA = 1e-9

#: How far a triangle's own displacement may inflate the neighbour search that builds the candidate
#: pair sets. A centroid moves by at most the largest displacement ``d`` of its three vertices and
#: the radius about it grows by at most ``2 d``, so ``3 d`` covers both. Not a tuning parameter: a
#: larger value only enlarges a superset, and testing a pair that cannot change costs time and
#: changes no answer.
_DISPLACEMENT_INFLATION = 3.0

#: The neighbour search of :func:`build_vertex_pair_sets` groups triangles by their reach into bins
#: whose upper bound grows by this factor, and searches each bin with that bin's largest reach, so a
#: query is never inflated by the single largest triangle of the mesh. Fixed, not a tuning
#: parameter: the pair sets are filtered by the exact criterion afterwards and do not depend on it.
_REACH_BIN_RATIO = 2.0

#: Relative and absolute padding of each neighbour-search radius, so that the k-d tree's own
#: rounding can never drop a pair the exact criterion keeps. The search only proposes candidates;
#: a larger radius changes no answer.
_SEARCH_RADIUS_RELATIVE_PAD = 1e-9
_SEARCH_RADIUS_ABSOLUTE_PAD = 1e-12

#: A junction vertex lying on this many or more distinct material triples is a quadruple point.
_QUADRUPLE_TRIPLE_COUNT = 2
#: A polyline needs at least this many vertices to carry a segment.
_MIN_POLYLINE_VERTICES = 2

#: The smallest point-placement ``min_distance`` at which a reconstruction that relocates junctions
#: emits :class:`CoarseSpacingRelocationWarning`. Voxels. Fixed by the measurement the warning
#: reports, not a tuning parameter.
COARSE_SPACING_MIN_DISTANCE = 5

#: The text of :class:`CoarseSpacingRelocationWarning`, stated once so documentation and tests can
#: quote it exactly.
COARSE_SPACING_RELOCATION_MESSAGE = (
    "At min_distance \u2265 5, the check on surface-tangent contact angles showed a small drift that "
    "lies within the check's own noise, and larger spacings have not been measured; facet-based "
    "contact angles improve. If tangent-based angles matter for your analysis, compare with "
    "`relocate_junctions=False` (or `set_junction_relocation(False)`)."
)


class CoarseSpacingRelocationWarning(UserWarning):
    """Junction relocation ran at a point spacing where its effect on tangent-based angles is unverified.

    Emitted by :meth:`dw3d.reconstruction_algorithm.MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`
    **once per reconstruction**, when junction relocation is on (the default) and the point-placement
    function's ``min_distance`` is at least :data:`COARSE_SPACING_MIN_DISTANCE`. A placement function
    that carries no ``min_distance`` (a custom callable rather than one of the factory's
    ``functools.partial`` placements) never triggers it, because the spacing cannot be read.

    What it means: at ``min_distance`` 5 the surface-tangent description of the contact angles moved
    by a small amount lying within the noise of the check that reads it, and no larger spacing has
    been measured. Facet-based contact angles improve at every spacing measured. The warning is a
    subclass of :class:`UserWarning`, so it can be silenced or promoted to an error on its own with
    :func:`warnings.filterwarnings`, and whether it fired is recorded in the algorithm's
    ``relocation_provenance``. To reproduce the 0.4 behaviour, construct the algorithm with
    ``relocate_junctions=False`` or call ``set_junction_relocation(False)`` on the factory.
    """


@dataclass
class JunctionRelocationReport:
    """Everything one relocation did.

    Attributes:
        n_matched_curves: matched mask curves the extractor returned.
        n_invented_curves: curves the mask has and the mesh has no trijunction for; never used as
            targets, counted so their number is visible.
        n_fragmented_curves: split, merged or ambiguous curves; left untouched and counted.
        n_targets: mesh junction vertices that received a target.
        n_quadruple_points_skipped: junction vertices on two or more matched triples, left where
            they are by this module's rule.
        n_accepted: moves applied.
        n_rejected_inversion: moves dropped because an incident triangle would have flipped.
        n_rejected_degenerate: moves dropped because an incident triangle would have collapsed.
        n_rejected_self_intersection: moves the local guard accepted and the nonlocal guard refused.
        n_rejected_cell_inversion: moves the triangle clauses accepted and the cell clause refused,
            because a cell they touch would have been left with a signed volume of zero or less.
        cells_kept_from_inverting: the labels for which at least one move was refused, ascending.
        n_cells_not_positive_before_relocation: non-zero labels whose signed volume was already zero
            or less before the pass, to which the cell clause therefore does not apply.
        proposed_displacement_voxels: the distance every target asked for, one per target.
        applied_displacement_voxels: the distance every accepted move actually travelled.
        n_self_intersections_before: whole-mesh crossings before the pass, or `None` when
            `verify=False`.
        n_self_intersections_after: whole-mesh crossings after the pass, or `None` when
            `verify=False`. Never greater than `n_self_intersections_before`.
        n_pairs_tested: triangle-pair evaluations the nonlocal guard ran, both configurations.
        n_degenerate_encountered: triangles found degenerate during the pass. A non-zero value means
            the incremental count and the whole-mesh certificate counted different populations, and
            the comparison between them is void.
        guard_seconds: wall time inside the nonlocal guard.
        index_seconds: wall time building the candidate pair sets.
        curve_counts: the extractor's own identity counts, carried through.
        match_rate: the extractor's matched-over-mesh-trijunction rate.
        spacing_voxels: the spacing the extractor resampled its polylines at.
    """

    n_matched_curves: int = 0
    n_invented_curves: int = 0
    n_fragmented_curves: int = 0
    n_targets: int = 0
    n_quadruple_points_skipped: int = 0
    n_accepted: int = 0
    n_rejected_inversion: int = 0
    n_rejected_degenerate: int = 0
    n_rejected_self_intersection: int = 0
    n_rejected_cell_inversion: int = 0
    cells_kept_from_inverting: list[int] = field(default_factory=list)
    n_cells_not_positive_before_relocation: int = 0
    proposed_displacement_voxels: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    applied_displacement_voxels: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    n_self_intersections_before: int | None = None
    n_self_intersections_after: int | None = None
    n_pairs_tested: int = 0
    n_degenerate_encountered: int = 0
    guard_seconds: float = 0.0
    index_seconds: float = 0.0
    curve_counts: dict = field(default_factory=dict)
    match_rate: float | None = None
    spacing_voxels: float | None = None

    def as_dict(self) -> dict:
        """A JSON-serialisable summary, the displacement distributions reduced to quantiles.

        Returns:
            dict: every count, plus the median, 90th percentile and maximum of both displacement
            distributions, or `None` for each where the distribution is empty.
        """

        def quantiles(values: NDArray[np.float64], name: str) -> dict:
            if values.size == 0:
                return {
                    f"{name}_median_voxels": None,
                    f"{name}_p90_voxels": None,
                    f"{name}_max_voxels": None,
                }
            return {
                f"{name}_median_voxels": float(np.median(values)),
                f"{name}_p90_voxels": float(np.percentile(values, 90)),
                f"{name}_max_voxels": float(np.max(values)),
            }

        return {
            "n_matched_curves": self.n_matched_curves,
            "n_invented_curves": self.n_invented_curves,
            "n_fragmented_curves": self.n_fragmented_curves,
            "n_targets": self.n_targets,
            "n_quadruple_points_skipped": self.n_quadruple_points_skipped,
            "n_accepted": self.n_accepted,
            "n_rejected_inversion": self.n_rejected_inversion,
            "n_rejected_degenerate": self.n_rejected_degenerate,
            "n_rejected_self_intersection": self.n_rejected_self_intersection,
            "n_rejected_cell_inversion": self.n_rejected_cell_inversion,
            "cells_kept_from_inverting": list(self.cells_kept_from_inverting),
            "n_cells_not_positive_before_relocation": self.n_cells_not_positive_before_relocation,
            "accepted_fraction": (self.n_accepted / self.n_targets) if self.n_targets else None,
            **quantiles(self.proposed_displacement_voxels, "proposed_displacement"),
            **quantiles(self.applied_displacement_voxels, "applied_displacement"),
            "n_self_intersections_before": self.n_self_intersections_before,
            "n_self_intersections_after": self.n_self_intersections_after,
            "n_pairs_tested": self.n_pairs_tested,
            "n_degenerate_encountered": self.n_degenerate_encountered,
            "guard_seconds": self.guard_seconds,
            "index_seconds": self.index_seconds,
            "curve_counts": self.curve_counts,
            "match_rate": self.match_rate,
            "spacing_voxels": self.spacing_voxels,
        }


def project_to_segments(
    queries: NDArray[np.float64],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distance from each query to the nearest of a set of segments, and the foot of it.

    Args:
        queries (NDArray[np.float64]): query positions, `(n, 3)`, voxel units.
        starts (NDArray[np.float64]): segment start points, `(m, 3)`.
        ends (NDArray[np.float64]): segment end points, `(m, 3)`.

    Returns:
        tuple: `(distances, feet)`, shapes `(n,)` and `(n, 3)`. With no segment or no query the
        distances are `inf` and the feet are the queries themselves.
    """
    queries = np.asarray(queries, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if len(starts) == 0 or len(queries) == 0:
        return np.full(len(queries), np.inf), np.array(queries, copy=True)
    direction = ends - starts
    denominator = (direction * direction).sum(axis=1)
    denominator[denominator == 0] = 1.0
    delta = queries[:, None, :] - starts[None, :, :]
    t = np.clip((delta * direction[None, :, :]).sum(axis=2) / denominator[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * direction[None, :, :]
    distances = np.linalg.norm(queries[:, None, :] - closest, axis=2)
    nearest = np.argmin(distances, axis=1)
    rows = np.arange(len(queries))
    return distances[rows, nearest], closest[rows, nearest]


def triangle_normals(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Unnormalised triangle normals; their length is twice the triangle's area.

    Args:
        points (NDArray[np.float64]): vertex coordinates.
        triangles (NDArray[np.integer]): triangle vertex indices.

    Returns:
        NDArray[np.float64]: one normal per triangle, `(m, 3)`.
    """
    corners = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def vertex_incidence(triangles: NDArray[np.integer], n_points: int) -> list[list[int]]:
    """Triangle indices incident to each vertex.

    Args:
        triangles (NDArray[np.integer]): triangle vertex indices.
        n_points (int): number of vertices, so an isolated vertex gets an empty list.

    Returns:
        list[list[int]]: per vertex, the indices of its incident triangles.
    """
    incident: list[list[int]] = [[] for _ in range(n_points)]
    for index, triangle in enumerate(np.asarray(triangles, dtype=np.int64)):
        for vertex in triangle:
            incident[int(vertex)].append(index)
    return incident


def matched_curve_targets(
    curves,  # noqa: ANN001 -- dw3d.junction_curves.JunctionCurves
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
) -> tuple[dict[int, NDArray[np.float64]], dict[int, float], set[int], dict]:
    """Per mesh junction vertex, its projection onto its own triple's matched mask curve.

    A mesh vertex belonging to more than one matched triple is a quadruple point: its targets
    disagree, so it is excluded here and counted.

    Args:
        curves (JunctionCurves): the extractor's output for this `(mask, mesh)`.
        points (NDArray[np.float64]): mesh vertex coordinates, voxel units.
        triangles (NDArray[np.integer]): triangle vertex indices.
        labels (NDArray[np.integer]): the two materials each triangle separates.

    Returns:
        tuple: `(targets, proposed_distance, quadruple_vertices, diagnostics)` where `targets` maps
        a vertex index to its projected position, `proposed_distance` maps the same keys to how far
        that move would travel, and `quadruple_vertices` are the excluded ones.
    """
    from dw3d.junction_curves import mesh_trijunction_components

    components = mesh_trijunction_components(
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )

    by_vertex: dict[int, list[tuple[float, NDArray[np.float64]]]] = {}
    triples_of_vertex: dict[int, set[tuple[int, int, int]]] = {}
    n_curves_without_segment = 0

    for curve in curves.matched:
        polyline = np.asarray(curve.points, dtype=np.float64)
        blocks = components.get(tuple(curve.triple), [])
        if len(polyline) < _MIN_POLYLINE_VERTICES or len(blocks) != 1:
            # `matched` means exactly one mesh component, so the second case cannot normally fire;
            # if it does the curve is skipped rather than guessed at.
            n_curves_without_segment += 1
            continue
        vertex_ids = np.asarray(blocks[0]["vertices"], dtype=np.int64)
        distances, feet = project_to_segments(
            np.asarray(points, dtype=np.float64)[vertex_ids], polyline[:-1], polyline[1:],
        )
        for vertex, foot, distance in zip(vertex_ids, feet, distances, strict=True):
            key = int(vertex)
            by_vertex.setdefault(key, []).append(
                (float(distance), np.asarray(foot, dtype=np.float64)),
            )
            triples_of_vertex.setdefault(key, set()).add(tuple(curve.triple))

    quadruple = {
        vertex for vertex, triples in triples_of_vertex.items()
        if len(triples) >= _QUADRUPLE_TRIPLE_COUNT
    }
    targets: dict[int, NDArray[np.float64]] = {}
    proposed: dict[int, float] = {}
    for vertex, offers in by_vertex.items():
        if vertex in quadruple:
            continue
        distance, foot = min(offers, key=lambda pair: pair[0])
        targets[vertex] = foot
        proposed[vertex] = distance

    return targets, proposed, quadruple, {
        "n_curves_without_segment": n_curves_without_segment,
        "n_vertices_offered": len(by_vertex),
    }


def build_vertex_pair_sets(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    targets: dict[int, NDArray[np.float64]],
) -> tuple[dict[int, NDArray[np.int64]], float]:
    """For each vertex with a target, a superset of the pairs its move can change the status of.

    The superset is built from the *base* centroids with an inflation that covers every displacement
    the pass can ever propose, so one static index serves the whole pass: a triangle's centroid
    moves by at most the largest displacement `d` of its own three vertices, its radius about that
    centroid grows by at most `2 d`, and two triangles can meet only when their centroids lie within
    the sum of their radii. A pair that can ever meet therefore satisfies
    `|c_f - c_t| <= r_f + r_t + 3 (d_f + d_t)` at base. Writing `reach = r + 3 d`, the pairs of a
    vertex are every `(f, t)` with `f` incident to the vertex, `t != f`,
    `|c_f - c_t| <= reach_f + reach_t` (evaluated as `np.linalg.norm` of the centroid difference),
    and `f` and `t` sharing no vertex.

    The neighbour search is one batched k-d tree search per pair of reach bins
    (:func:`_neighbours_within_reach`) rather than one ball query per triangle inflated by the
    mesh's largest reach; the exact criterion above then decides, so the pair sets are the same
    whatever the search proposes.

    Args:
        points (NDArray[np.float64]): the base vertex coordinates, voxel units.
        triangles (NDArray[np.integer]): triangle vertex indices, never modified.
        targets (dict[int, NDArray[np.float64]]): vertex index to proposed position; only these
            vertices move, and their displacements set the inflation.

    Returns:
        tuple: `(pairs_by_vertex, seconds)`. Each value is a `(k, 2)` `int64` array of triangle
        index pairs, each row `(smaller, larger)`, rows in ascending lexicographic order, with
        adjacency already removed, ready for the exact predicate. A vertex whose move can change
        nothing maps to an empty `(0, 2)` array.
    """
    clock = time.perf_counter()
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    centroids, radii = triangle_centroids_and_circumradii(points[triangles])

    displacement = np.zeros(len(points), dtype=np.float64)
    for vertex, target in targets.items():
        displacement[vertex] = float(np.linalg.norm(np.asarray(target) - points[vertex]))
    reach = radii + _DISPLACEMENT_INFLATION * displacement[triangles].max(axis=1)

    incident = vertex_incidence(triangles, len(points))
    vertices = sorted(targets)
    # `np.unique`: a triangle with a repeated vertex is listed twice in that vertex's incidence.
    faces_of = [np.unique(np.asarray(incident[vertex], dtype=np.int64)) for vertex in vertices]
    queried = np.unique(np.concatenate(faces_of)) if faces_of else np.zeros(0, dtype=np.int64)
    face_rows, partners = _neighbours_within_reach(centroids, reach, queried)

    # Per queried face, its partners (CSR layout over `queried`).
    order = np.argsort(face_rows, kind="stable")
    face_rows, partners = face_rows[order], partners[order]
    bounds = np.searchsorted(face_rows, np.arange(len(queried) + 1))

    pairs_by_vertex: dict[int, NDArray[np.int64]] = {}
    for vertex, faces in zip(vertices, faces_of, strict=True):
        if not len(faces):
            pairs_by_vertex[vertex] = np.zeros((0, 2), dtype=np.int64)
            continue
        rows = np.searchsorted(queried, faces)
        counts = bounds[rows + 1] - bounds[rows]
        if not counts.sum():
            pairs_by_vertex[vertex] = np.zeros((0, 2), dtype=np.int64)
            continue
        own = np.repeat(faces, counts)
        other = np.concatenate([partners[bounds[r]:bounds[r + 1]] for r in rows])
        pairs = np.stack([np.minimum(own, other), np.maximum(own, other)], axis=1)
        # A pair is listed twice only when both triangles are incident to this vertex, and those
        # share it, so dropping adjacency also removes every repeat; what remains is sorted.
        pairs = drop_adjacent_pairs(pairs, triangles)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        pairs_by_vertex[vertex] = pairs
    return pairs_by_vertex, time.perf_counter() - clock


def _neighbours_within_reach(
    centroids: NDArray[np.float64],
    reach: NDArray[np.float64],
    queried: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Every `(f, t)` with `f` in `queried`, `t != f` and `|c_f - c_t| <= reach_f + reach_t`.

    Triangles are grouped into bins of reach whose upper bounds grow by `_REACH_BIN_RATIO`; for
    each pair of a queried bin and a partner bin, one k-d tree search at the sum of the two bins'
    largest reaches (padded by `_SEARCH_RADIUS_*_PAD`) proposes candidates, and the exact criterion,
    `np.linalg.norm(c_t - c_f) <= reach_f + reach_t`, keeps them.

    Args:
        centroids (NDArray[np.float64]): `(m, 3)` triangle centroids.
        reach (NDArray[np.float64]): `(m,)` non-negative reach per triangle.
        queried (NDArray[np.int64]): sorted distinct triangle indices to find partners for.

    Returns:
        tuple: `(rows, partners)`, two `int64` arrays of equal length: `queried[rows[i]]` and
        `partners[i]` form one pair. Each pair appears once; their order is unspecified.
    """
    empty = np.zeros(0, dtype=np.int64)
    if not len(queried) or not len(centroids):
        return empty, empty
    positive = reach[reach > 0]
    floor = float(positive.min()) if len(positive) else 1.0
    bins = np.floor(np.log(np.maximum(reach, floor) / floor) / np.log(_REACH_BIN_RATIO)).astype(np.int64)
    bins = np.maximum(bins, 0)
    rows_out: list[NDArray[np.int64]] = []
    partners_out: list[NDArray[np.int64]] = []
    query_bins = bins[queried]
    query_groups = []
    for query_bin in np.unique(query_bins):
        local_rows = np.flatnonzero(query_bins == query_bin)
        faces = queried[local_rows]
        query_groups.append((local_rows, float(reach[faces].max()), cKDTree(centroids[faces])))
    for partner_bin in np.unique(bins):
        members = np.flatnonzero(bins == partner_bin)
        partner_tree = cKDTree(centroids[members])
        partner_reach = float(reach[members].max())
        for local_rows, query_reach, query_tree in query_groups:
            radius = (query_reach + partner_reach) * (1.0 + _SEARCH_RADIUS_RELATIVE_PAD)
            radius += _SEARCH_RADIUS_ABSOLUTE_PAD
            candidates = query_tree.sparse_distance_matrix(partner_tree, radius, output_type="ndarray")
            if not len(candidates):
                continue
            row = local_rows[candidates["i"].astype(np.int64)]
            partner = members[candidates["j"].astype(np.int64)]
            face = queried[row]
            separation = np.linalg.norm(centroids[partner] - centroids[face], axis=1)
            keep = (separation <= reach[face] + reach[partner]) & (partner != face)
            rows_out.append(row[keep])
            partners_out.append(partner[keep])
    if not rows_out:
        return empty, empty
    return np.concatenate(rows_out), np.concatenate(partners_out)


def _crossing_count(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    pairs: NDArray[np.int64],
) -> tuple[int, int]:
    """Intersections among an explicit pair list, and how many of its triangles are degenerate.

    Only the triangles the pairs involve are gathered: the predicate reads each pair's own six
    corners and nothing else, so the result is the same as on the whole mesh's corner array.
    """
    involved = np.unique(pairs)
    corners = points[triangles[involved]]
    n_degenerate = int((twice_areas(corners) <= DEGENERATE_TWICE_AREA).sum())
    intersects, _touching = pair_intersection_flags(corners, np.searchsorted(involved, pairs))
    return int(intersects.sum()), n_degenerate


def _apply_moves(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    targets: dict[int, NDArray[np.float64]],
    pairs_by_vertex: dict[int, NDArray[np.int64]],
    report: JunctionRelocationReport,
    labels: NDArray[np.integer] | None = None,
) -> NDArray[np.float64]:
    """Apply each move only if it keeps its triangles valid, turns no cell inside out and creates no crossing.

    The clauses run cheapest first: the triangle clauses (area, then orientation), the cell clause,
    then the nonlocal one, which is the expensive one; a move an earlier clause refuses need not be
    certified by a later one.

    **The cell clause.** Each label's signed volume is taken at the start of the pass by the
    divergence theorem with the package's winding (`+det(p0, p1, p2) / 6` to a triangle's first label,
    `-det / 6` to its second; cubic voxels, the mesh in mask voxel units), and kept current as moves
    are accepted. A move changes only the determinants of the triangles incident to the moved vertex,
    so the volume after it is the current volume plus their change. The move is refused when any
    non-zero label on either side of those triangles, whose volume before the pass was positive, would
    have a volume `<= 0` after it (zero counts as inverted). The medium (label 0) and labels whose
    volume before the pass is already `<= 0` are not constrained, and counted. Where the clause never
    refuses a move, the result is bitwise the result without it: it reads the geometry and changes
    nothing. Its cost is one pass over the triangles to start, then a handful of operations per
    candidate move, on the triangles incident to the moved vertex.

    Failure modes: a refused move leaves its vertex where it was, so that junction vertex is not
    relocated, and the cell keeps the volume it had; the clause does not judge whether that volume is
    right (a cell whose mesh is far from its mask before relocation stays so). The test is on the
    floating-point sum, so a cell whose true volume is within rounding of zero is decided by rounding.

    Args:
        points (NDArray[np.float64]): vertex coordinates; not modified in place.
        triangles (NDArray[np.integer]): triangle vertex indices; never modified.
        targets (dict[int, NDArray[np.float64]]): vertex index to proposed position.
        pairs_by_vertex (dict[int, NDArray[np.int64]]): :func:`build_vertex_pair_sets`' output.
        report (JunctionRelocationReport): filled in place with every count this pass produces.
        labels (NDArray[np.integer] | None): the two materials each triangle separates, `(m, 2)`.
            `None` switches the cell clause off.

    Returns:
        NDArray[np.float64]: the moved vertex coordinates, same shape, dtype and order as `points`.
    """
    moved = np.array(points, dtype=np.float64, copy=True)
    triangles = np.asarray(triangles, dtype=np.int64)
    incident = vertex_incidence(triangles, len(moved))
    baseline_normals = triangle_normals(moved, triangles)
    applied: list[float] = []
    volumes = _CellVolumes(moved, triangles, labels)
    report.n_cells_not_positive_before_relocation = volumes.n_unconstrained
    protected: set[int] = set()

    for vertex in sorted(targets):
        faces = incident[vertex]
        if not faces:
            continue
        original = moved[vertex].copy()
        moved[vertex] = targets[vertex]
        candidate_normals = triangle_normals(moved, triangles[faces])
        areas = 0.5 * np.linalg.norm(candidate_normals, axis=1)
        reference = baseline_normals[faces]
        reference_norm = np.linalg.norm(reference, axis=1)
        candidate_norm = np.linalg.norm(candidate_normals, axis=1)
        safe = (reference_norm > 0) & (candidate_norm > 0)
        cosines = np.ones(len(faces))
        cosines[safe] = np.einsum("ij,ij->i", reference[safe], candidate_normals[safe]) / (
            reference_norm[safe] * candidate_norm[safe]
        )
        if np.any(areas < MIN_TRIANGLE_AREA):
            moved[vertex] = original
            report.n_rejected_degenerate += 1
            continue
        if np.any(cosines <= 0):
            moved[vertex] = original
            report.n_rejected_inversion += 1
            continue
        inverted = volumes.cells_inverted_by(moved, faces)
        if inverted:
            moved[vertex] = original
            report.n_rejected_cell_inversion += 1
            protected.update(inverted)
            continue

        pairs = pairs_by_vertex.get(vertex)
        if pairs is not None and len(pairs):
            clock = time.perf_counter()
            after, degenerate_after = _crossing_count(moved, triangles, pairs)
            moved[vertex] = original
            before, degenerate_before = _crossing_count(moved, triangles, pairs)
            report.n_pairs_tested += 2 * len(pairs)
            report.n_degenerate_encountered += degenerate_after + degenerate_before
            report.guard_seconds += time.perf_counter() - clock
            if after > before:
                report.n_rejected_self_intersection += 1
                continue
            moved[vertex] = targets[vertex]

        report.n_accepted += 1
        applied.append(float(np.linalg.norm(targets[vertex] - original)))
        baseline_normals[faces] = candidate_normals
        volumes.accept(moved, faces)

    report.applied_displacement_voxels = np.asarray(applied, dtype=np.float64)
    report.cells_kept_from_inverting = sorted(protected)
    return moved


class _CellVolumes:
    """Each label's signed volume, kept current through a sequence of single-vertex moves.

    Volumes follow the package's winding (`+det(p0, p1, p2) / 6` to a triangle's first label,
    `-det / 6` to its second), in cubic voxels. Only the triangles incident to a moved vertex change
    their determinant, so a candidate volume is the current one plus those triangles' change. Built
    with `labels=None`, it constrains nothing and records nothing.
    """

    def __init__(
        self,
        points: NDArray[np.float64],
        triangles: NDArray[np.int64],
        labels: NDArray[np.integer] | None,
    ) -> None:
        self.enabled = labels is not None
        if not self.enabled:
            labels = np.zeros((0, 2), dtype=np.int64)
            triangles = triangles[:0]
        labels = np.asarray(labels, dtype=np.int64)
        self.triangles = triangles
        self.labels = labels
        self.determinants = self._determinants(points, triangles)
        present, inverse = np.unique(labels, return_inverse=True)
        self.index = {int(label): i for i, label in enumerate(present.tolist())}
        inverse = inverse.reshape(labels.shape)
        volume = np.bincount(inverse[:, 0], weights=self.determinants, minlength=len(present))
        volume -= np.bincount(inverse[:, 1], weights=self.determinants, minlength=len(present))
        self.current = volume / 6.0
        self.constrained = (present != 0) & (self.current > 0)
        self.n_unconstrained = int(((present != 0) & ~(self.current > 0)).sum())

    @staticmethod
    def _determinants(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.float64]:
        corners = points[triangles]
        return np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))

    def _change(self, points: NDArray[np.float64], faces: list[int]) -> tuple[NDArray[np.float64], dict[int, float]]:
        # `np.unique`: a triangle with a repeated vertex is listed twice in that vertex's incidence.
        faces_array = np.unique(np.asarray(faces, dtype=np.int64))
        new = self._determinants(points, self.triangles[faces_array])
        delta = new - self.determinants[faces_array]
        change: dict[int, float] = {}
        for (first, second), d in zip(self.labels[faces_array].tolist(), delta.tolist(), strict=True):
            change[first] = change.get(first, 0.0) + d / 6.0
            change[second] = change.get(second, 0.0) - d / 6.0
        return new, change

    def cells_inverted_by(self, points: NDArray[np.float64], faces: list[int]) -> list[int]:
        """The constrained labels the candidate configuration leaves with a volume of zero or less."""
        if not self.enabled:
            return []
        _, change = self._change(points, faces)
        inverted = []
        for label, d in change.items():
            i = self.index[label]
            if self.constrained[i] and self.current[i] + d <= 0:
                inverted.append(label)
        return inverted

    def accept(self, points: NDArray[np.float64], faces: list[int]) -> None:
        """Make the candidate configuration of `faces` the current one."""
        if not self.enabled:
            return
        new, change = self._change(points, faces)
        self.determinants[np.unique(np.asarray(faces, dtype=np.int64))] = new
        for label, d in change.items():
            self.current[self.index[label]] += d


def relocate_junction_vertices(
    segmented_mask: NDArray[np.uint],
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    *,
    estimator: Estimator = "plain",
    curves=None,  # noqa: ANN001 -- a precomputed JunctionCurves, to avoid extracting twice
    verify: bool = True,
) -> tuple[NDArray[np.float64], JunctionRelocationReport]:
    """Move a mesh's trijunction vertices onto the mask's own trijunction curves.

    Connectivity is frozen: `triangles` and `labels` are not read for anything but the junction
    structure and are never modified, so only `points` changes and the result can be hashed against
    its input. See the module docstring for the two guards and for what the construction was
    measured to cost and to buy.

    Args:
        segmented_mask (NDArray[np.uint]): the label image the mesh was reconstructed from.
        points (NDArray[np.float64]): mesh vertex coordinates in mask voxel units, `(n, 3)`, in the
            mask's index order, as `construct_mesh_from_segmentation_mask` returns them.
        triangles (NDArray[np.integer]): triangle vertex indices, `(m, 3)`.
        labels (NDArray[np.integer]): the two materials each triangle separates, `(m, 2)`.
        estimator (Estimator): which sub-voxel estimator
            :func:`~dw3d.junction_curves.extract_junction_curves` runs. `"plain"` is the default
            this construction was specified and measured against; `"anchor"` costs roughly fifteen
            times as much for 9-11 % of the line-position gain.
        curves (JunctionCurves | None): a precomputed :class:`~dw3d.junction_curves.JunctionCurves`
            for this `(mask, mesh)`, to avoid extracting twice when several treatments share one base
            mesh. Defaults to None, which extracts them.
        verify (bool): run the whole-mesh self-intersection certificate before and after and record
            both counts. Adds about 0.08 s per benchmark mesh. Leave it on unless the certificate is
            already being run by the caller.

    Returns:
        tuple: `(moved_points, report)`. `moved_points` is a new float64 array with the same shape
            and vertex order as `points`; `points` is not modified. `report` is a
            `JunctionRelocationReport`.

    Raises:
        ValueError: propagated from the extractor if the mesh does not lie in the mask's frame.
        RuntimeError: if `verify` is on and the whole-mesh crossing count rose. That is impossible
            if the nonlocal guard is correct, so it is a defect in this module and is raised rather
            than absorbed.
    """
    from dw3d.junction_curves import extract_junction_curves

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    if curves is None:
        curves = extract_junction_curves(
            segmented_mask, points, triangles, labels, estimator=estimator,
        )

    targets, proposed, quadruple, _diagnostics = matched_curve_targets(
        curves, points, triangles, labels,
    )
    report = JunctionRelocationReport(
        n_matched_curves=len(curves.matched),
        n_invented_curves=len(curves.invented),
        n_fragmented_curves=len(curves.fragmented),
        n_targets=len(targets),
        n_quadruple_points_skipped=len(quadruple),
        proposed_displacement_voxels=np.asarray(
            [proposed[vertex] for vertex in sorted(targets) if vertex in proposed],
            dtype=np.float64,
        ),
        curve_counts=dict(curves.counts),
        match_rate=curves.match_rate,
        spacing_voxels=curves.spacing_voxels,
    )
    if verify:
        report.n_self_intersections_before = find_self_intersections(points, triangles)[
            "n_self_intersecting_triangle_pairs"
        ]

    pairs_by_vertex, report.index_seconds = build_vertex_pair_sets(points, triangles, targets)
    moved = _apply_moves(points, triangles, targets, pairs_by_vertex, report, labels)

    if verify:
        report.n_self_intersections_after = find_self_intersections(moved, triangles)[
            "n_self_intersecting_triangle_pairs"
        ]
        if report.n_self_intersections_after > report.n_self_intersections_before:
            message = (
                "the junction relocation raised the self-intersection count from "
                f"{report.n_self_intersections_before} to {report.n_self_intersections_after}. "
                "The nonlocal guard makes this impossible, so it is a defect in "
                "dw3d.junction_relocation and not a property of this mesh."
            )
            raise RuntimeError(message)
    return moved, report
