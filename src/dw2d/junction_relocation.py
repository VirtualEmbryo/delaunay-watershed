r"""Move a finished 2D mesh's junction vertices onto the junctions the mask itself gives (opt-in).

Opt-in. It is the 2D port of dw3d's ``junction_relocation`` + ``junction_curves``: the connectivity is frozen
(the edge array, hence every interface and junction identity, is never changed); only the coordinates
of junction vertices move, and only when four guards accept the move.

Why
---
dw2d's mesh vertices are Delaunay points chosen on the pixel lattice (integer row, column), so every
junction vertex sits on a pixel centre, while a junction of the label image lies on the *dual* lattice:
the corner shared by four pixels. On the equilibrium-foam benchmark (exact rasterisations of
certified circular-arc foams with known tensions; held-out split, 159 images, ``min_dist`` 3) moving
the junctions onto the mask's own junctions lowered the case angle error (per case the median over
junctions of the mean absolute error of the three material-attributed angles, then the median over
cases) by 1.76 deg [1.51, 1.94] from 9.15 deg, the junction position error by 0.21 px, and the
gauge-matched tension error by 0.025 [0.011, 0.029] (per-junction Young-Dupre rows), on every scale
tier (cell radius 10, 20, 40 px). With smooth boundary noise of 0.5 px RMS it still helps
(-1.49 deg, -0.019); at 1 px it helps the angles only (-0.66 deg); at 2 px it is neutral on angles.
These are measurements on synthetic masks, not a guarantee on real masks.

What it does
------------
* **Mask junctions.** A dual-lattice corner at continuous (row, col) = (i + 0.5, j + 0.5) is shared by
  pixels (i, j), (i, j+1), (i+1, j), (i+1, j+1). It is a junction corner of the triple (a, b, c) when
  its four pixels hold exactly those three labels; a corner whose four pixels hold four labels counts
  for each of the four triples it contains (dw3d's rule, which carries a junction into a four-fold
  point instead of stopping it short). Per triple, 8-connected components of junction corners are one
  mask junction each, located at the component's centroid.
* **Identity.** Mesh junctions (vertices whose incident edges involve exactly three materials) are
  paired with mask junctions of the same triple by the Hungarian assignment on Euclidean distance,
  pairs farther than ``gate_px`` (3 px) rejected, and only for triples whose two counts agree
  ("matched"). A vertex of four or more materials is never moved.
* **Guards.** Moves are applied one vertex at a time in ascending vertex index against the
  configuration as it stands. A move is accepted only if (1) no incident edge becomes shorter than
  ``MIN_EDGE_PX``; (2) the number of proper crossings between non-adjacent edges does not increase —
  moving v changes only the edges incident to v, so the change of the whole-mesh count equals the
  change restricted to pairs involving those edges (an exact identity, evaluated against a candidate
  set that provably contains every edge that can meet them); (3) no edge incident to the moved
  vertex comes into a contact it was not in before the move (see "Edge contacts" below); (4) no
  material's signed area (dw2d ``Dcel.compute_areas`` over the oriented edges) changes sign or reaches
  zero. A refused move is dropped, never repaired: the vertex stays where it was. Whole-mesh recounts
  of crossings and of contacts before and after certify (2) and (3) globally; a rise raises.

Edge contacts
-------------
Guard (2) counts proper crossings in floating point, so it lets through a move that makes two edges
*touch*: on the equilibrium-foam benchmark at a cell radius of 40 px, the projection of
:mod:`dw2d.locus_projection` flattened 141 Z-shaped turns of plain interface chains so that a vertex
landed exactly on a non-adjacent edge (an endpoint inside another edge) and the edge before it folded
back by exactly 180 degrees. Guard (3) refuses every new contact of either kind, decided exactly:

* two edges sharing **no vertex** are in contact when the closed segments meet: a proper crossing,
  a T-touch (an endpoint of one lies on the other), coincident vertices (two distinct vertices with
  identical coordinates, endpoints of both), or a collinear overlap (four collinear endpoints sharing
  one point or a sub-segment);
* two edges sharing **one vertex** s are in contact when their other endpoints lie on the same ray
  from s: one edge runs back along the other (an adjacent overlap, a turn of exactly 180 degrees).

A pair counts as new when it is not in contact before the move, or in contact of another class.
Every coordinate is a binary64 float, hence an exact rational. Orientation signs use Shewchuk's static
filter for ``orient2d`` and, when the filter cannot certify the sign, the determinant in
``fractions.Fraction``; comparisons of coordinates are float comparisons, which are exact. On that
benchmark the guard refuses 143 projection moves in 38 of the 53 meshes at 40 px and leaves 0
contacts; at 10 and 20 px, where the projection creates none, it refuses nothing and the vertices are
bitwise unchanged. It keeps the Z turns of the plain mesh (two turns of opposite signs) rather than
removing them.

Units and conventions: coordinates in pixels, (row, col), the frame of ``geometry_reconstruction_2d``'s
vertices; the label image is the one the mesh was built from (after ``expansion_labels``). Nothing
reads ground truth. Deterministic.

Failure modes: a mesh with no matched triple is returned unchanged (``planned = 0``); a triple whose
mesh and mask junction counts differ is left untouched and counted (``unmatched_triples``); the
thresholds are fixed and are not tuning parameters.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

if TYPE_CHECKING:  # imported lazily at run time
    from collections.abc import Iterable

    from dw2d.reports import JunctionRelocationReport

#: Shortest edge a move may leave, in pixels. Fixed; not a tuning parameter.
MIN_EDGE_PX = 1e-6
#: Largest distance between a mesh junction and the mask junction it is paired with, in pixels (fixed).
GATE_PX = 3.0
#: Shewchuk's static error bound of ``orient2d`` in binary64 (``ccwerrboundA``): a float determinant
#: larger than this many times the sum of the magnitudes of its two products has the exact sign.
_CCW_ERRBOUND_A = (3.0 + 16.0 * 2.0 ** -53) * 2.0 ** -53


def _orient(a, b, c):
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _proper_crossings(P1, P2, Q1, Q2):
    """Boolean array: segment P1P2 properly crosses Q1Q2 (strict: touching or collinear is not counted)."""
    d1, d2 = _orient(Q1, Q2, P1), _orient(Q1, Q2, P2)
    d3, d4 = _orient(P1, P2, Q1), _orient(P1, P2, Q2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _orient_sign(a, b, c):
    """Exact sign (+1, 0, -1) of the orientation of the points a, b, c, each (2,) float.

    +1 when (a, b, c) turn counter-clockwise in the (column 0, column 1) frame. Shewchuk's static
    filter decides when it can; otherwise the determinant is evaluated exactly in rationals.
    """
    ax, ay, bx, by, cx, cy = float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(c[0]), float(c[1])
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    bound = _CCW_ERRBOUND_A * (abs(detleft) + abs(detright))
    if det > bound:
        return 1
    if -det > bound:
        return -1
    ax, ay, bx, by, cx, cy = (Fraction(x) for x in (ax, ay, bx, by, cx, cy))
    d = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (d > 0) - (d < 0)


def _contact_class(p1, p2, q1, q2):
    """Contact class of segments p1p2 and q1q2 that share no vertex index, or None if they are disjoint.

    In this order: ``"collinear_overlap"`` (four collinear endpoints, a common sub-segment of positive
    length), ``"collinear_point"`` (collinear, one common point), ``"proper_crossing"``,
    ``"coincident_vertices"`` (they meet at a point that is an endpoint of both), ``"t_touch"`` (an
    endpoint of one lies in the interior of the other). Exact (see the module docstring).
    """
    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or max(q1[0], q2[0]) < min(p1[0], p2[0])
            or max(p1[1], p2[1]) < min(q1[1], q2[1]) or max(q1[1], q2[1]) < min(p1[1], p2[1])):
        return None
    o1, o2 = _orient_sign(p1, p2, q1), _orient_sign(p1, p2, q2)
    o3, o4 = _orient_sign(q1, q2, p1), _orient_sign(q1, q2, p2)
    if o1 == 0 and o2 == 0:
        a, b = sorted(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
        c, d = sorted(((float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))))
        lo, hi = max(a, c), min(b, d)  # lexicographic order is the order along the common line
        if lo < hi:
            return "collinear_overlap"
        if lo == hi:
            return "collinear_point"
        return None
    if o1 * o2 < 0 and o3 * o4 < 0:
        return "proper_crossing"
    if o1 * o2 <= 0 and o3 * o4 <= 0:
        ends_p = {(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))}
        ends_q = {(float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))}
        return "coincident_vertices" if ends_p & ends_q else "t_touch"
    return None


def _adjacent_contact_class(s, x, y):
    """``"adjacent_overlap"`` when edges (s, x) and (s, y), sharing vertex s, lie on one ray from s; else None.

    Exact: the orientation sign is exact, and the sign of a float difference is the sign of the exact one.
    """
    if _orient_sign(s, x, y) != 0:
        return None
    if all(np.sign(x[k] - s[k]) == np.sign(y[k] - s[k]) for k in (0, 1)):
        return "adjacent_overlap"
    return None


def _pair_contact(verts, E, i, j):
    """Contact class of edges i and j of the edge array E (first two columns), or None."""
    a, b = int(E[i, 0]), int(E[i, 1])
    c, d = int(E[j, 0]), int(E[j, 1])
    shared = {a, b} & {c, d}
    if len(shared) == 2:
        return None
    if len(shared) == 1:
        (s,) = shared
        x = b if a == s else a
        y = d if c == s else c
        if (verts[s] == verts[x]).all() or (verts[s] == verts[y]).all():
            return None
        return _adjacent_contact_class(verts[s], verts[x], verts[y])
    return _contact_class(verts[a], verts[b], verts[c], verts[d])


def _boxes_meet(P1, P2, Q1, Q2):
    """Boolean array: the bounding boxes of segments P1P2 and Q1Q2 (arrays (n, 2)) intersect (closed, exact)."""
    return ((np.maximum(P1[:, 0], P2[:, 0]) >= np.minimum(Q1[:, 0], Q2[:, 0]))
            & (np.maximum(Q1[:, 0], Q2[:, 0]) >= np.minimum(P1[:, 0], P2[:, 0]))
            & (np.maximum(P1[:, 1], P2[:, 1]) >= np.minimum(Q1[:, 1], Q2[:, 1]))
            & (np.maximum(Q1[:, 1], Q2[:, 1]) >= np.minimum(P1[:, 1], P2[:, 1])))


def whole_mesh_contacts(verts, edges):
    """Number of pairs of edges of a 2D mesh in contact, of any class (see "Edge contacts" above).

    ``verts`` (V, 2) float, ``edges`` (E, >= 2) int (first two columns = vertex indices). Pairs of
    edges sharing no vertex count when the closed segments meet (proper crossing, T-touch, coincident
    vertices, collinear overlap); pairs sharing one vertex count when one runs back along the other.
    Zero-length edges are skipped. Candidate pairs come from a KD-tree on edge midpoints with a radius
    covering the two longest half-edges, then an exact bounding-box test; the classes are exact.
    """
    verts = np.asarray(verts, float)
    E = np.asarray(edges)[:, :2]
    if len(E) < 2:
        return 0
    P = verts[E[:, 0]]; Q = verts[E[:, 1]]
    zero = (P == Q).all(1)
    mid = (P + Q) / 2; half = np.linalg.norm(Q - P, axis=1) / 2
    pairs = cKDTree(mid).query_pairs(2 * half.max() + 1e-6, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = ~zero[i] & ~zero[j] & _boxes_meet(P[i], Q[i], P[j], Q[j])
    return sum(_pair_contact(verts, E, int(a), int(b)) is not None for a, b in zip(i[keep], j[keep]))


def whole_mesh_crossings(verts, edges):
    """Number of proper crossings between non-adjacent edges of a 2D mesh.

    ``verts`` (V, 2) float, ``edges`` (E, >= 2) int (first two columns = vertex indices). Candidate
    pairs come from a KD-tree on edge midpoints with a radius covering the two longest half-edges, so
    no crossing can be missed.
    """
    verts = np.asarray(verts, float)
    E = np.asarray(edges)[:, :2]
    if len(E) < 2:
        return 0
    P = verts[E[:, 0]]; Q = verts[E[:, 1]]
    mid = (P + Q) / 2; half = np.linalg.norm(Q - P, axis=1) / 2
    pairs = cKDTree(mid).query_pairs(2 * half.max() + 1e-9, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    share = (E[i, 0] == E[j, 0]) | (E[i, 0] == E[j, 1]) | (E[i, 1] == E[j, 0]) | (E[i, 1] == E[j, 1])
    i, j = i[~share], j[~share]
    return int(_proper_crossings(P[i], Q[i], P[j], Q[j]).sum())


def signed_areas(verts, edges):
    """{material: signed area} from dw2d's oriented edge array (the formula of ``Dcel.compute_areas``).

    On dw2d meshes every cell's value has the same sign (about its pixel count, positive since 0.8.0, when each
    edge became the positively oriented boundary of its first material; negative before); the area guard relies
    only on that consistency, which ``tests/test_junction_relocation.py`` pins.
    """
    A = defaultdict(float)
    for i1, i2, m1, m2 in np.asarray(edges):
        v1, v2 = verts[i1], verts[i2]
        dx, dy = v2[0] - v1[0], v2[1] - v1[1]
        w = 0.5 * (-v2[1] * dx + v2[0] * dy)
        A[int(m1)] += w; A[int(m2)] -= w
    return dict(A)


class GuardedVertexMover:
    """Vertex positions over a frozen edge array, moved one vertex at a time under the four guards.

    Inputs: ``verts`` (V, 2) float (copied), ``edges`` (E, 4) int [v1, v2, m1, m2]. ``try_move(v, p)``
    applies the move and returns ``""`` or the name of the refusing guard, checked in this order:
    ``"short_edge"``, ``"crossing"``, ``"contact"``, ``"area_sign"``. The candidate search for the
    crossing and contact guards is a KD-tree on the *initial* edge midpoints; its radius adds twice
    the largest displacement applied so far, which bounds both how far any midpoint has moved and how
    much any half-length has grown, so the candidate set always contains every edge the moved edges
    can meet.
    """

    def __init__(self, verts, edges):
        self.v = np.array(verts, float)
        self.e = np.asarray(edges)
        E = self.e[:, :2]
        self.incident = defaultdict(list)
        for k, (a, b) in enumerate(E):
            self.incident[int(a)].append(k); self.incident[int(b)].append(k)
        L = np.linalg.norm(self.v[E[:, 0]] - self.v[E[:, 1]], axis=1)
        self._max_half = float(L.max() / 2) if len(L) else 0.0
        self._tree = cKDTree((self.v[E[:, 0]] + self.v[E[:, 1]]) / 2) if len(E) else None
        self._largest_move = 0.0
        self.area = signed_areas(self.v, self.e)
        self._materials = [(int(m1), int(m2)) for m1, m2 in self.e[:, 2:4]]

    def _candidates(self, vid, pos):
        """Edges that can meet an edge incident to ``vid`` when ``vid`` sits at ``pos`` (sorted array)."""
        E = self.e[:, :2]
        ks = self.incident[vid]
        old = self.v[vid].copy()
        P = self.v[E[ks, 0]].copy(); Q = self.v[E[ks, 1]].copy()
        P[E[ks, 0] == vid] = pos; Q[E[ks, 1] == vid] = pos
        reach = (float(np.max(np.linalg.norm(Q - P, axis=1))) / 2 + self._max_half
                 + float(np.linalg.norm(pos - old)) + 2 * self._largest_move + 1e-9)
        cand = set()
        for c in (P + Q) / 2:
            cand.update(self._tree.query_ball_point(c, reach))
        return np.array(sorted(cand), int)

    def _local_contacts(self, vid, pos):
        """Set of (class, edge, edge) contacts between an edge incident to ``vid`` and any edge, with ``vid`` at ``pos``.

        Edge pairs are listed as (smaller index, larger index); pairs of two edges incident to ``vid``
        are included. Classes as in ``_contact_class`` and ``_adjacent_contact_class``.
        """
        E = self.e[:, :2]
        ks = self.incident[vid]
        cand = self._candidates(vid, pos)
        old = self.v[vid].copy()
        self.v[vid] = pos
        try:
            out = set()
            for k in ks:
                cc = cand[cand != k]
                if not len(cc):
                    continue
                a, b = E[k]
                n = len(cc)
                meet = _boxes_meet(np.broadcast_to(self.v[a], (n, 2)), np.broadcast_to(self.v[b], (n, 2)),
                                   self.v[E[cc, 0]], self.v[E[cc, 1]])
                for j in cc[meet]:
                    cls = _pair_contact(self.v, E, k, int(j))
                    if cls:
                        out.add((cls, min(k, int(j)), max(k, int(j))))
            return out
        finally:
            self.v[vid] = old

    def _local_crossings(self, vid, pos):
        E = self.e[:, :2]
        ks = self.incident[vid]
        old = self.v[vid].copy()
        self.v[vid] = pos
        try:
            P = self.v[E[ks, 0]]; Q = self.v[E[ks, 1]]
            reach = (float(np.max(np.linalg.norm(Q - P, axis=1))) / 2 + self._max_half
                     + float(np.linalg.norm(pos - old)) + 2 * self._largest_move + 1e-9)
            cand = set()
            for c in (P + Q) / 2:
                cand.update(self._tree.query_ball_point(c, reach))
            cand = np.array(sorted(cand - set(ks)), int)
            n = 0
            for k in ks:
                a, b = E[k]
                if not len(cand):
                    break
                adjacent = (E[cand, 0] == a) | (E[cand, 1] == a) | (E[cand, 0] == b) | (E[cand, 1] == b)
                cc = cand[~adjacent]
                if len(cc):
                    n += int(_proper_crossings(np.broadcast_to(self.v[a], (len(cc), 2)), np.broadcast_to(self.v[b], (len(cc), 2)),
                                               self.v[E[cc, 0]], self.v[E[cc, 1]]).sum())
            return n
        finally:
            self.v[vid] = old

    def try_move(self, vid, pos):
        pos = np.asarray(pos, float)
        E = self.e[:, :2]
        ks = self.incident[vid]
        for k in ks:
            other = E[k, 1] if E[k, 0] == vid else E[k, 0]
            if np.linalg.norm(self.v[other] - pos) < MIN_EDGE_PX:
                return "short_edge"
        if self._local_crossings(vid, pos) > self._local_crossings(vid, self.v[vid].copy()):
            return "crossing"
        after = self._local_contacts(vid, pos)
        if after and not after <= self._local_contacts(vid, self.v[vid].copy()):
            return "contact"
        old = self.v[vid].copy()
        delta = defaultdict(float)
        for k in ks:
            i1, i2 = E[k]; m1, m2 = self._materials[k]
            for sign, P in ((-1, old), (+1, pos)):
                v1 = P if i1 == vid else self.v[i1]
                v2 = P if i2 == vid else self.v[i2]
                dx, dy = v2[0] - v1[0], v2[1] - v1[1]
                w = 0.5 * (-v2[1] * dx + v2[0] * dy)
                delta[m1] += sign * w; delta[m2] -= sign * w
        for m, d in delta.items():
            a0 = self.area[m]; a1 = a0 + d
            if a0 * a1 <= 0 or abs(a1) < 1e-12:
                return "area_sign"
        for m, d in delta.items():
            self.area[m] += d
        self._largest_move = max(self._largest_move, float(np.linalg.norm(pos - old)))
        self.v[vid] = pos
        return ""


def mask_junctions(labels):
    """Junctions of a label image on the dual lattice.

    Returns a list of dict(materials: sorted triple, position: [row, col] (the component centroid, on
    half-integer coordinates for a single corner), n_corners).
    """
    L = np.asarray(labels)
    if L.ndim != 2 or min(L.shape) < 2:
        return []
    S = np.sort(np.stack([L[:-1, :-1], L[:-1, 1:], L[1:, :-1], L[1:, 1:]], -1), -1)
    n_unique = 1 + (np.diff(S, axis=-1) != 0).sum(-1)
    corners = defaultdict(list)
    for i, j in np.argwhere(n_unique >= 3):
        for tri in combinations(sorted(set(S[i, j].tolist())), 3):
            corners[tri].append((int(i), int(j)))
    out = []
    for tri, cs in corners.items():
        members, seen = set(cs), set()
        for c in cs:
            if c in seen:
                continue
            comp, stack = [], [c]; seen.add(c)
            while stack:
                p = stack.pop(); comp.append(p)
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        q = (p[0] + di, p[1] + dj)
                        if q in members and q not in seen:
                            seen.add(q); stack.append(q)
            out.append(dict(materials=tri, position=(np.array(comp, float) + 0.5).mean(0).tolist(), n_corners=len(comp)))
    return out


def mesh_junction_vertices(edges):
    """{vertex index: sorted material triple} for vertices whose incident edges involve exactly three materials."""
    mats = defaultdict(set)
    for v1, v2, m1, m2 in np.asarray(edges):
        mats[int(v1)] |= {int(m1), int(m2)}; mats[int(v2)] |= {int(m1), int(m2)}
    return {v: tuple(sorted(m)) for v, m in mats.items() if len(m) == 3}


def _relocate_on_edges(verts, edges, labels, gate_px=GATE_PX, fixed_vertices=()):
    """Move the matched junction vertices of a 2D mesh onto the mask's own junctions.

    Inputs: ``verts`` (V, 2) float (row, col) px, ``edges`` (E, 4) int [v1, v2, m1, m2] as returned by
    ``geometry_reconstruction_2d.return_mesh()``, ``labels`` the label image the mesh was built from.
    ``fixed_vertices``: vertices never moved (``geometry_reconstruction_2d`` passes the image border vertices,
    ``dw2d.Dcel.image_border_vertices``). The matching is made with them, as without; a fixed vertex's move is
    then dropped, so a junction on the image border stays on it.
    Returns ``(new_verts, report)``: a moved copy of the vertices (same shape and dtype float64) and a
    dict (mask_junctions, mesh_junctions, planned, fixed (planned moves dropped because the vertex is fixed),
    moved, refused {guard: count}, unmatched_triples, crossings_before, crossings_after, contacts_before,
    contacts_after, median_displacement_px).
    Raises ``RuntimeError`` if the whole-mesh crossing count or contact count rose, which the guards
    make impossible by construction.
    """
    verts = np.asarray(verts, float); edges = np.asarray(edges)
    mover = GuardedVertexMover(verts, edges)
    targets = mask_junctions(labels)
    junctions = mesh_junction_vertices(edges)
    by_triple_t, by_triple_m = defaultdict(list), defaultdict(list)
    for t in targets:
        by_triple_t[t["materials"]].append(t)
    for v, tri in junctions.items():
        by_triple_m[tri].append(v)
    plan, unmatched = {}, 0
    for tri, vs in by_triple_m.items():
        ts = by_triple_t.get(tri, [])
        if len(ts) != len(vs):
            unmatched += 1
            continue
        C = np.array([[np.linalg.norm(mover.v[v] - np.array(t["position"])) for t in ts] for v in vs])
        for a, b in zip(*linear_sum_assignment(C)):
            if C[a, b] <= gate_px:
                plan[vs[a]] = np.array(ts[b]["position"])
    fixed = {int(v) for v in fixed_vertices}
    n_fixed = sum(1 for v in plan if v in fixed)
    plan = {v: target for v, target in plan.items() if v not in fixed}
    before = whole_mesh_crossings(mover.v, edges)
    contacts_before = whole_mesh_contacts(mover.v, edges)
    refused, moved, displacement = defaultdict(int), 0, []
    for v in sorted(plan):
        why = mover.try_move(v, plan[v])
        if why:
            refused[why] += 1
        else:
            moved += 1; displacement.append(float(np.linalg.norm(plan[v] - verts[v])))
    after = whole_mesh_crossings(mover.v, edges)
    if after > before:
        raise RuntimeError(f"edge crossings rose from {before} to {after}: the crossing guard is wrong")
    contacts_after = whole_mesh_contacts(mover.v, edges)
    if contacts_after > contacts_before:
        raise RuntimeError(f"edge contacts rose from {contacts_before} to {contacts_after}: the contact guard is wrong")
    return mover.v, dict(mask_junctions=len(targets), mesh_junctions=len(junctions), planned=len(plan) + n_fixed, fixed=n_fixed, moved=moved,
                         refused=dict(refused), unmatched_triples=unmatched, crossings_before=before, crossings_after=after,
                         contacts_before=contacts_before, contacts_after=contacts_after,
                         median_displacement_px=float(np.median(displacement)) if displacement else None)


def relocate_junction_vertices(
    mask: np.ndarray,
    points: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
    *,
    gate: float = GATE_PX,
    fixed_vertices: Iterable[int] = (),
) -> tuple[np.ndarray, JunctionRelocationReport]:
    """Move the matched junction vertices of a 2D mesh onto the mask's own junctions (connectivity frozen).

    The 2D counterpart of `dw3d.relocate_junction_vertices`, with the same argument order: the mask first, then the
    mesh. This module's docstring says what it does, under which guards, and what it was measured to buy.
    `dw2d.MeshReconstructionAlgorithm` runs it by default (`set_junction_relocation`).

    Args:
        mask: the 2-D label image the mesh was built from (after any label expansion).
        points: `(n, 2)` float vertex positions in pixels, `(row, column)`.
        segments: `(m, 2)` int vertex indices of each edge.
        labels: `(m, 2)` int, the two materials each edge separates (smaller first; the orientation contract of
            `dw2d.MeshReconstructionAlgorithm`).
        gate: largest distance, in pixels, between a mesh junction and the mask junction it is paired with.
        fixed_vertices: vertices never moved (the algorithm passes its `image_border_vertices`).

    Returns:
        tuple: `(points, report)`: a moved copy of `points` (float64) and a `dw2d.JunctionRelocationReport`.

    Raises:
        RuntimeError: if the whole-mesh crossing or contact count rose, which the guards make impossible.
    """
    from dw2d.reports import JunctionRelocationReport

    edges = np.hstack((np.asarray(segments), np.asarray(labels)))
    moved, report = _relocate_on_edges(np.asarray(points, float), edges, np.asarray(mask), gate, fixed_vertices)
    return moved.copy(), JunctionRelocationReport(**report)
