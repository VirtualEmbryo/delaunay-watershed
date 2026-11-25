"""Relabelling of near-tie membrane slivers between the seeded watershed and the mesh extraction.

Why this exists
---------------
The Delaunay triangulation of dw2d's seed points puts many flat triangles along a membrane: their
three vertices are minima of the distance transform (points on the membrane) and their largest angle
is close to 180 degrees. Where two such slivers share an edge and the seeded watershed gives them
different labels, the interface between the two labels runs through the sliver pair and turns back
on itself twice within a few pixels: a Z-shaped fold of the interface polyline. On synthetic
equilibrium foams with cells of 40 px radius (53 images, ``min_dist=3``) the default labelling left
453 such folds (400 interface chains). They are the folds the locus projection would otherwise
flatten into an edge contact, which its contact guard then refuses.

At such a pair the watershed's choice is often a near tie: the sliver's best edge score towards its
own label and its best edge score towards its partner's label differ by a small fraction of a pixel.
The rule below gives the sliver its partner's label in that case, but only when doing so strictly
removes interface reversals at the sliver's three vertices.

The rule
--------
Definitions (``points`` in pixel units, ``(row, column)``):

* a **membrane point** is one of the first ``n_membrane_points`` points of the triangulation, the
  minima of the distance transform (``build_triangulation`` stacks them first);
* a **valley sliver** is a triangle whose three vertices are membrane points and whose largest
  interior angle exceeds 150 degrees;
* an **alternating sliver pair** is a dual-graph edge joining two valley slivers of different labels;
* the **margin** of a triangle ``t`` towards a label ``m`` is the largest watershed score of a dual
  edge from ``t`` to a neighbour labelled like ``t``, minus the largest score of a dual edge from ``t``
  to a neighbour labelled ``m`` (scores: the mean distance-transform value along the Delaunay edge,
  in pixels; ``Delaunay_Graph.compute_scores``); it is undefined when either set is empty;
* at a vertex with exactly two interface edges (dual edges whose two triangles carry different
  labels), a **reversal** is an interior angle below 30 degrees between the two edges, i.e. the
  interface turns by more than 150 degrees; a vertex is **irregular** when its number of interface
  edges is neither 0 nor 2.

Each pass lists the alternating sliver pairs under the labels at the start of the pass, in the order
of (smaller triangle index, larger triangle index), and visits each triangle of a pair (first the one
on the first side of the dual edge, then the other). A visited triangle ``t`` with partner ``p`` is
relabelled to ``p``'s label when all hold:

1. ``t`` and ``p`` still carry different labels;
2. ``t`` is neither a seed triangle nor a zero-node triangle (``protected``: the overrides that follow
   the watershed would undo the change there);
3. its margin towards ``p``'s label is defined and below ``margin`` (0.1 px by default);
4. the relabelling strictly lowers the number of reversals at ``t``'s three vertices and does not
   raise the number of irregular vertices among them.

Otherwise the label is kept. Passes repeat until one relabels nothing. A relabelling changes interface
edges only at the edges of ``t``, so only the reversals at ``t``'s vertices can change: every
accepted relabelling lowers the total number of reversals by at least one, and the passes terminate.
``max_passes`` is a safety bound; ``report.converged`` says whether it was reached.

Only labels change: no point moves and the triangulation is unchanged.

Measured effect (equilibrium-foam benchmark, held-out split of 159 images, ``min_dist=3``)
----------------------------------------------------------------------------------------
With cells of 40 px radius the rule relabels 453 triangles in 48 of 53 images and removes every
Z-fold (453 to 0); with 20 px cells it relabels 4 triangles in 4 images; with 10 px cells none. It
converges on every image. Junction angles read by chords are unchanged on every image; the median
paired change of the per-image pressure error over the 52 changed images is -0.0026 of the RMS true
pressure (95 % bootstrap interval [-0.0053, -0.0013]) on the plain mesh, and -0.00011 ([-0.0030,
+0.0015], not distinguishable from 0) on the mesh after junction relocation and locus projection. No
per-image error (angles, tensions, pressures) became worse in a way the bootstrap could distinguish
from zero. On the golden cases and on the shipped real masks the rule relabels nothing.

Failure modes
-------------
The rule never raises on a valid triangulation. It does nothing when no triangle is a valley sliver
or when no alternating pair has a near tie. It does not repair folds whose watershed choice is not a
near tie (margin at or above ``margin``), nor folds made of triangles that are not valley slivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEAR_TIE_SCORE_MARGIN = 0.1
"""Default margin (pixels of distance-transform score) below which a sliver's label is a near tie."""

SLIVER_MAX_ANGLE_DEG = 150.0
"""A triangle of membrane points is a valley sliver when its largest interior angle exceeds this."""

REVERSAL_INTERIOR_ANGLE_DEG = 30.0
"""An interface vertex is a reversal when its two interface edges make an interior angle below this."""


@dataclass
class NearTieSliverReport:
    """What ``relabel_near_tie_slivers`` did.

    ``relabelled_triangles``: sorted indices of the triangles whose final label differs from the input
    label. ``triangles_changed``: their number. ``flips``: accepted relabellings (a triangle relabelled
    twice counts twice). ``candidates_examined``: visits of a triangle of an alternating pair whose
    labels still differed. ``skipped_protected``: of those, seed or zero-node triangles.
    ``below_margin``: visits whose margin was below the threshold. ``refused_no_gain``: of those, the
    relabellings undone because they did not strictly lower the reversals (or raised the irregular
    vertices). ``passes``: passes run, the last one relabelling nothing when ``converged``.
    ``valley_slivers``: number of valley-sliver triangles. ``alternating_pairs_before``: alternating
    sliver pairs in the input labels.
    """

    margin: float
    valley_slivers: int = 0
    alternating_pairs_before: int = 0
    passes: int = 0
    flips: int = 0
    candidates_examined: int = 0
    below_margin: int = 0
    refused_no_gain: int = 0
    skipped_protected: int = 0
    converged: bool = False
    triangles_changed: int = 0
    relabelled_triangles: list = field(default_factory=list)


def largest_interior_angles_deg(points, triangles):
    """Largest interior angle of every triangle, in degrees (``points`` (n, 2), ``triangles`` (t, 3))."""
    P = np.asarray(points, float)[np.asarray(triangles)]
    out = np.zeros(len(P))
    for i in range(3):
        u = P[:, (i + 1) % 3] - P[:, i]
        v = P[:, (i + 2) % 3] - P[:, i]
        c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        out = np.maximum(out, np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def valley_sliver_triangles(points, triangles, n_membrane_points):
    """Boolean mask of the valley slivers: three membrane points and a largest angle above 150 degrees."""
    triangles = np.asarray(triangles)
    on_membrane = np.all(triangles < n_membrane_points, axis=1)
    return on_membrane & (largest_interior_angles_deg(points, triangles) > SLIVER_MAX_ANGLE_DEG)


class _DualGraph:
    """The dual graph's adjacency in the order the rule reads it.

    Dual edge ``k`` joins triangles ``t1[k]`` and ``t2[k]`` across the Delaunay edge ``(u[k], w[k])``
    with watershed score ``score[k]``. ``edges_of_triangle[t]`` and ``edges_at_vertex[v]`` list dual
    edges by increasing ``k``; the order at a vertex fixes which interface edge is "first" in the
    reversal angle, so it is part of the rule's bitwise definition.
    """

    def __init__(self, delaunay_graph):
        linked = np.asarray(delaunay_graph.Nodes_linked_by_Edges, np.int64).reshape(-1, 2)
        edges = np.asarray(delaunay_graph.Edges, np.int64).reshape(-1, 2)[:, :2]
        self.t1, self.t2 = linked[:, 0], linked[:, 1]
        self.u, self.w = edges[:, 0], edges[:, 1]
        self.score = np.asarray(delaunay_graph.Scores, float)
        self.triangles = np.asarray(delaunay_graph.Tris, np.int64)
        n_edges = len(linked)
        ks = np.arange(n_edges)
        self.edges_of_triangle = self._group(np.concatenate([self.t1, self.t2]), np.concatenate([ks, ks]),
                                             len(self.triangles))
        n_vertices = int(max(self.triangles.max(initial=-1), edges.max(initial=-1))) + 1
        self.edges_at_vertex = self._group(np.concatenate([self.u, self.w]), np.concatenate([ks, ks]), n_vertices)

    @staticmethod
    def _group(keys, ks, n):
        order = np.lexsort((ks, keys))
        keys, ks = keys[order], ks[order]
        bounds = np.searchsorted(keys, np.arange(n + 1))
        return [ks[bounds[i]:bounds[i + 1]].tolist() for i in range(n)]

    def other(self, k, t):
        return int(self.t2[k]) if self.t1[k] == t else int(self.t1[k])


def _vertex_state(dual, points, labels, v):
    """(number of interface edges at ``v``, whether ``v`` is a reversal) under ``labels``."""
    ks = [k for k in dual.edges_at_vertex[v] if labels[dual.t1[k]] != labels[dual.t2[k]]]
    if len(ks) != 2:
        return len(ks), False
    ends = [int(dual.w[k]) if dual.u[k] == v else int(dual.u[k]) for k in ks]
    p = points[v]
    a, b = points[ends[0]] - p, points[ends[1]] - p
    c = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    return 2, bool(np.degrees(np.arccos(np.clip(c, -1, 1))) < REVERSAL_INTERIOR_ANGLE_DEG)


def _local_cost(dual, points, labels, vertices):
    """(reversals, irregular vertices) among ``vertices``."""
    reversals = irregular = 0
    for v in vertices:
        degree, reversal = _vertex_state(dual, points, labels, v)
        reversals += int(reversal)
        irregular += int(degree not in (0, 2))
    return reversals, irregular


def _margin(dual, labels, t, other_label):
    own = [dual.score[k] for k in dual.edges_of_triangle[t] if labels[dual.other(k, t)] == labels[t]]
    oth = [dual.score[k] for k in dual.edges_of_triangle[t] if labels[dual.other(k, t)] == other_label]
    if not own or not oth:
        return None
    return float(max(own) - max(oth))


def relabel_near_tie_slivers(delaunay_graph, labels, n_membrane_points, protected_triangles,
                             margin=NEAR_TIE_SCORE_MARGIN, max_passes=100):
    """Relabel the near-tie triangles of alternating membrane-sliver pairs (see the module docstring).

    Parameters
    ----------
    delaunay_graph : dw2d.Graph_functions.Delaunay_Graph
        Its ``Vertices`` (pixel ``(row, column)``), ``Tris``, ``Edges``, ``Nodes_linked_by_Edges`` and
        ``Scores`` are read; nothing is modified.
    labels : (n_triangles,) integer array
        The label of every triangle after the seeded watershed (``seeded_watershed_aggregation``),
        before the zero-node and seed overrides.
    n_membrane_points : int
        Number of distance-transform minima at the start of the triangulation's points.
    protected_triangles : array of triangle indices
        Seed and zero-node triangles; never relabelled.
    margin : float
        Score margin in pixels below which a sliver's label is a near tie (default 0.1).
    max_passes : int
        Safety bound on the passes (the rule terminates by construction; see the module docstring).

    Returns
    -------
    (new_labels, report) : a new integer array (the input is not modified) and a
    ``NearTieSliverReport``. Where the rule relabels nothing, ``new_labels`` equals ``labels``.
    """
    L = np.array(labels, copy=True)
    report = NearTieSliverReport(margin=float(margin))
    points = np.asarray(delaunay_graph.Vertices, float)
    sliver = valley_sliver_triangles(points, delaunay_graph.Tris, n_membrane_points)
    report.valley_slivers = int(sliver.sum())
    dual = _DualGraph(delaunay_graph)
    protected = np.zeros(len(dual.triangles), bool)
    protected[np.asarray(protected_triangles, np.int64)] = True
    pair_edges = np.flatnonzero(sliver[dual.t1] & sliver[dual.t2])
    lo, hi = np.minimum(dual.t1[pair_edges], dual.t2[pair_edges]), np.maximum(dual.t1[pair_edges], dual.t2[pair_edges])
    pair_edges = pair_edges[np.lexsort((hi, lo))]

    def alternating():
        return [int(k) for k in pair_edges if L[dual.t1[k]] != L[dual.t2[k]]]

    report.alternating_pairs_before = len(alternating())
    for _ in range(max_passes):
        report.passes += 1
        flips = 0
        for k in alternating():
            for t, partner in ((int(dual.t1[k]), int(dual.t2[k])), (int(dual.t2[k]), int(dual.t1[k]))):
                if L[t] == L[partner]:
                    continue
                report.candidates_examined += 1
                if protected[t]:
                    report.skipped_protected += 1
                    continue
                m = _margin(dual, L, t, L[partner])
                if m is None or m >= margin:
                    continue
                report.below_margin += 1
                vertices = [int(x) for x in dual.triangles[t]]
                reversals_before, irregular_before = _local_cost(dual, points, L, vertices)
                old = L[t]
                L[t] = L[partner]
                reversals_after, irregular_after = _local_cost(dual, points, L, vertices)
                if reversals_after < reversals_before and irregular_after <= irregular_before:
                    flips += 1
                else:
                    L[t] = old
                    report.refused_no_gain += 1
        report.flips += flips
        if flips == 0:
            report.converged = True
            break
    changed = np.flatnonzero(L != np.asarray(labels))
    report.triangles_changed = int(len(changed))
    report.relabelled_triangles = changed.tolist()
    return L, report
