"""Repair of the labelled Delaunay triangulation before the 2D mesh is extracted.

Why this exists
---------------
``DCEL_Data`` needs every material's boundary to be **one simple closed loop**: every vertex of
a material's mesh edges must have degree 2 and the edges must form a single cycle
(``Dcel.create_halfedges_from_edges`` asserts it). On multicellular masks the labelled
triangulation produced by the seeded watershed violates this in two ways (measured on the
masks shipped with the package):

* **fragments** — a label whose triangles form several edge-connected components. The seeded
  watershed itself never produces one; every fragment is created afterwards by the
  "zero-node" override (``Networkx_functions.seeded_watershed_map``), which forces to the
  medium (0) every triangle whose centroid falls in an all-zero 2x2 pixel block, and so cuts
  cells (and the medium) into pieces;
* **pinch vertices** — a vertex around which some label occupies two or more separate angular
  sectors, so that the label's boundary visits the vertex twice. This is the 2D analogue of
  dw3d's *abnormal non-manifold edge* (an edge around which a material occupies two or more
  separate blocks of the tetrahedron cycle), and the repair below is the 2D analogue of
  ``dw3d.mesh_surgery.post_process_mesh_surgery``: relabel a few elements of the ring around
  the defect so every material occupies one block.

What it does
------------
``repair_labelled_triangulation`` works on triangle labels only; it never moves a point and
never changes the triangulation. Two passes, repeated until nothing changes (at most
``max_passes`` times):

1. **Fragment absorption (cells only).** For every cell label, the edge-connected component
   holding its seed triangle is kept; every other component is relabelled, as a whole, to the
   neighbouring label it shares the largest total watershed edge score with. The medium (label 0)
   is never absorbed: each of its components lies on zero pixels of the mask by construction of
   the zero-node override, and absorbing them was measured to invent cell-cell contacts and lose
   cell-medium contacts on an under-resolved mask (``mask_Cells_2`` without expansion); a medium in several pieces is represented by several boundary loops instead
   (``DCEL_Data(..., allow_several_loops=True)``).
2. **Pinch repair.** At every pinch vertex, the triangles of the fan (in angular order; a
   convex-hull vertex gets one fixed virtual "outside" slot of label 0) are relabelled so
   every label occupies one circular run. Each changed triangle takes a label already present
   in the fan. Candidates are compared lexicographically: (a) the pinch is gone; (b) when the
   mask is given, the fewest new contacts between two labels that do not touch in the mask
   (most pinches are of the medium, between two cells that meet at one vertex: joining them
   with one relabelled triangle invents an interface the mask may not have, where separating
   them needs two or three; added after the first measurements); (c) the
   fewest triangles changed; (d) the fewest new pinch vertices created at the changed
   triangles' other vertices; (e) the **largest** score agreement — the sum, over the Delaunay
   edges of the changed triangles, of the watershed edge score when the two triangles across
   the edge now carry the same label. The watershed merges across high-score edges first, so
   a higher agreement is a labelling the watershed itself would prefer: it is maximised
   (``argmax``). dw3d's selector was found minimising a "higher is better" score and was
   corrected to ``argmax``; do not invert this one.

Guards (a candidate violating one is never applied): a seed triangle is never relabelled; a
label never loses its last triangle; no label touched by the move ends with more edge-connected
components than it had (the medium's components joined through the outside count as one).

Inputs and outputs
------------------
``repair_labelled_triangulation(delaunay_graph, map_end, seed_nodes, max_passes=10,
max_changes=3)`` takes dw2d's ``Delaunay_Graph`` (``Nodes`` = triangles, ``Edges`` /
``Nodes_linked_by_Edges`` / ``Scores`` = shared edges, their two triangles and their mean
EDT score, ``Lone_edges`` / ``Nodes_linked_by_lone_edges`` = convex-hull edges), the
watershed's ``Map_end`` ({label: [triangle ids]}) and the seed triangle ids. It returns a new ``Map_end`` (the **same object** when nothing was
changed, so the default output is bitwise unchanged) and a ``SurgeryReport`` listing every
move.

Failure modes
-------------
A pinch with no admissible candidate within ``max_changes`` triangles is left in place and
listed in ``report.unresolved_pinch_vertices`` (the DCEL then still fails). A material whose
boundary has several loops (a medium in several pieces, a cell enclosing a gap) is listed in
``report.labels_with_several_loops``: the legacy ``DCEL_Data`` (``allow_several_loops=False``)
cannot represent it; ``geometry_reconstruction_2d.return_dcel`` passes
``allow_several_loops=True`` whenever this repair was requested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np


@dataclass
class SurgeryReport:
    """What ``repair_labelled_triangulation`` did.

    Attributes:
        passes: number of passes run.
        fragment_moves: one dict per absorbed fragment: ``label_from``, ``label_to``,
            ``n_triangles``, ``shared_score``.
        pinch_moves: one dict per repaired pinch: ``vertex``, ``cycle_before``, ``cycle_after``,
            ``triangles`` and ``labels_from`` / ``labels_to``, ``agreement`` (the maximised
            score), ``new_pinches``.
        unresolved_pinch_vertices: vertices still pinched at the end.
        labels_with_several_components: labels still split into components at the end
            (medium components joined through the outside count as one).
        labels_with_several_loops: labels whose boundary still has more than one connected
            piece (inner loops), which the legacy ``DCEL_Data`` cannot represent.
        n_triangles_changed: number of distinct triangles whose label changed.
    """

    passes: int = 0
    fragment_moves: list = field(default_factory=list)
    pinch_moves: list = field(default_factory=list)
    unresolved_pinch_vertices: list = field(default_factory=list)
    labels_with_several_components: dict = field(default_factory=dict)
    labels_with_several_loops: dict = field(default_factory=dict)
    n_triangles_changed: int = 0


class _Triangulation:
    """Adjacency of a dw2d ``Delaunay_Graph`` (built once, read-only)."""

    def __init__(self, delaunay_graph):
        self.triangles = np.asarray(delaunay_graph.Nodes, dtype=np.int64)
        self.n = len(self.triangles)
        self.neighbours = [[] for _ in range(self.n)]  # (neighbour triangle, edge score)
        for (a, b), s in zip(delaunay_graph.Nodes_linked_by_Edges, delaunay_graph.Scores):
            a, b = int(a), int(b)
            self.neighbours[a].append((b, float(s)))
            self.neighbours[b].append((a, float(s)))
        self.on_hull = np.zeros(self.n, dtype=bool)
        if len(delaunay_graph.Nodes_linked_by_lone_edges):
            self.on_hull[np.asarray(delaunay_graph.Nodes_linked_by_lone_edges, dtype=np.int64)] = True
        self.vertices = np.asarray(delaunay_graph.Vertices, dtype=np.float64)
        self.fans = _ordered_fans(self.triangles)
        self.triangles_of_vertex = {v: fan for v, (fan, _closed) in self.fans.items()}


def _ordered_fans(triangles):
    """{vertex: (triangle ids in angular order, closed)}; open fans belong to hull vertices."""
    incident = defaultdict(list)
    for t, tri in enumerate(triangles):
        for v in tri:
            incident[int(v)].append(t)
    fans = {}
    for v, ts in incident.items():
        by_other = defaultdict(list)
        for t in ts:
            for w in triangles[t]:
                if w != v:
                    by_other[int(w)].append(t)
        nbr = defaultdict(list)
        for tt in by_other.values():
            if len(tt) == 2:
                nbr[tt[0]].append(tt[1])
                nbr[tt[1]].append(tt[0])
        ends = [t for t in ts if len(nbr[t]) < 2]
        closed = not ends
        start = ts[0] if closed else ends[0]
        order, prev, cur = [start], None, start
        while True:
            nxt = [u for u in nbr[cur] if u != prev]
            if not nxt or (closed and nxt[0] == start):
                break
            prev, cur = cur, nxt[0]
            order.append(cur)
        if len(order) != len(ts):
            raise ValueError(f"vertex {v}: its incident triangles do not form one fan")
        fans[v] = (order, closed)
    return fans


_OUTSIDE = -1  # position marker of the virtual outside slot of a hull vertex


def _cycle(fan, closed, labels):
    """Labels around a vertex; a hull vertex's fan is closed by one outside slot of label 0."""
    cyc = [int(labels[t]) for t in fan]
    return cyc if closed else cyc + [0]


def _has_repeated_run(cyc):
    """True when some label occupies two or more maximal circular runs of ``cyc``."""
    n = len(cyc)
    if n < 4:
        # three slots carrying a label twice are always one circular run of that label
        return False
    run_labels = [cyc[i] for i in range(n) if cyc[i] != cyc[i - 1]]
    return len(run_labels) != len(set(run_labels))


def find_pinch_vertices(tri: _Triangulation, labels: np.ndarray) -> list[int]:
    """Vertices whose fan holds some label in two or more separate circular runs."""
    return sorted(v for v, (fan, closed) in tri.fans.items() if _has_repeated_run(_cycle(fan, closed, labels)))


def _components(tri: _Triangulation, labels: np.ndarray, members) -> list[list[int]]:
    """Edge-connected components of the triangles ``members`` (all of one label)."""
    members = set(int(m) for m in members)
    seen, comps = set(), []
    for m in sorted(members):
        if m in seen:
            continue
        comp, stack = [], [m]
        seen.add(m)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w, _s in tri.neighbours[u]:
                if w in members and w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def _members_by_label(labels):
    out = defaultdict(set)
    for t, L in enumerate(labels):
        out[int(L)].add(t)
    return out


def _medium_component_groups(tri, comps, seed_set):
    """Split the medium's components into those joined through the outside and the inner ones."""
    outer, inner = [], []
    for c in comps:
        if tri.on_hull[c].any() or (seed_set & set(c)):
            outer.append(c)
        else:
            inner.append(c)
    return outer, inner


def _absorb(tri, labels, comp, own_label):
    """Relabel a fragment to the neighbouring label it shares the largest summed score with."""
    comp_set = set(comp)
    shared = defaultdict(float)
    for u in comp:
        for w, s in tri.neighbours[u]:
            if w not in comp_set and labels[w] != own_label:
                shared[int(labels[w])] += s
    if not shared:
        return None
    best = max(sorted(shared), key=lambda L: shared[L])  # argmax; ties -> smallest label id
    labels[np.asarray(comp, dtype=np.int64)] = best
    return best, shared[best]


def _fragment_pass(tri, labels, seed_set, report):
    """Absorb every fragment of every cell label (the medium, label 0, is never absorbed)."""
    changed = False
    for L in sorted(_members_by_label(labels)):
        if L == 0:
            continue
        # members are read now, not before the loop: an earlier absorption may have joined them
        comps = _components(tri, labels, np.flatnonzero(labels == L).tolist())
        if len(comps) < 2:
            continue
        with_seed = [c for c in comps if seed_set & set(c)]
        kept = with_seed[0] if with_seed else max(comps, key=len)
        for c in comps:
            if c is kept:
                continue
            res = _absorb(tri, labels, c, L)
            if res is not None:
                changed = True
                report.fragment_moves.append(dict(label_from=int(L), label_to=int(res[0]), n_triangles=len(c),
                                                  shared_score=float(res[1])))
    return changed


def _agreement(tri, labels, changed_triangles):
    """Sum of edge scores over the changed triangles' edges whose two sides now agree."""
    total, seen = 0.0, set()
    for t in changed_triangles:
        for w, s in tri.neighbours[t]:
            key = (min(t, w), max(t, w))
            if key in seen:
                continue
            seen.add(key)
            if labels[t] == labels[w]:
                total += s
    return total


def _n_components(tri, labels, label, seed_set):
    """Number of components of ``label``; the medium's hull-touching components count as one."""
    members = set(np.flatnonzero(labels == label).tolist())
    if not members:
        return 0
    comps = _components(tri, labels, members)
    if label == 0:
        outer, inner = _medium_component_groups(tri, comps, seed_set)
        return (1 if outer else 0) + len(inner)
    return len(comps)


def mask_contacts(mask_labels):
    """Label pairs (sorted) that touch in a label image: 4-adjacent pixels, plus (0, cell) for every
    cell touching the image border (the outside of the triangulation is medium)."""
    m = np.asarray(mask_labels)
    pairs = set()
    for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1, :], m[1:, :])):
        d = a != b
        pairs |= {(int(min(x, y)), int(max(x, y))) for x, y in zip(a[d], b[d])}
    border = set(np.unique(np.r_[m[0], m[-1], m[:, 0], m[:, -1]]).tolist()) - {0}
    return pairs | {(0, int(L)) for L in border}


def _contacts(tri, labels):
    """Label pairs adjacent across at least one Delaunay edge (hull edges: (0, cell))."""
    pairs = set()
    for t in range(tri.n):
        for w, _s in tri.neighbours[t]:
            if labels[t] != labels[w]:
                pairs.add((int(min(labels[t], labels[w])), int(max(labels[t], labels[w]))))
    pairs |= {(0, int(L)) for L in labels[tri.on_hull] if L != 0}
    return pairs


def _repair_one_pinch(tri, labels, v, seed_set, max_changes, allowed_contacts=None):
    fan, closed = tri.fans[v]
    cyc0 = _cycle(fan, closed, labels)
    positions = [i for i, t in enumerate(fan) if t not in seed_set]
    fan_labels = sorted(set(cyc0))
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    baseline = {}
    existing = _contacts(tri, labels) if allowed_contacts is not None else None
    fallback = None
    for k in range(1, max_changes + 1):
        best = None
        for pos in combinations(positions, k):
            options = [[L for L in fan_labels if L != cyc0[i]] for i in pos]
            for new in product(*options):
                cyc = list(cyc0)
                for i, L in zip(pos, new):
                    cyc[i] = L
                if _has_repeated_run(cyc):
                    continue
                ts = [fan[i] for i in pos]
                old = [int(labels[t]) for t in ts]
                # never remove a label's last triangle
                loss = defaultdict(int)
                for L in old:
                    loss[L] += 1
                if any(counts[L] - n <= 0 for L, n in loss.items()):
                    continue
                trial = labels.copy()
                trial[np.asarray(ts, dtype=np.int64)] = np.asarray(new, dtype=np.int64)
                affected = set(loss) | set(new)
                if any(_n_components(tri, trial, L, seed_set) > baseline.setdefault(L, _n_components(tri, labels, L, seed_set))
                       for L in affected):
                    continue  # never split a label (nor the medium) into more pieces
                others = {int(w) for t in ts for w in tri.triangles[t] if int(w) != v}
                new_pinches = sum(_has_repeated_run(_cycle(*tri.fans[w], trial)) for w in others)
                old_pinches = sum(_has_repeated_run(_cycle(*tri.fans[w], labels)) for w in others)
                created = max(0, new_pinches - old_pinches)
                agreement = _agreement(tri, trial, ts)
                foreign = 0
                if allowed_contacts is not None:
                    new_pairs = set()
                    for t in ts:
                        for w, _s in tri.neighbours[t]:
                            if trial[t] != trial[w]:
                                new_pairs.add((int(min(trial[t], trial[w])), int(max(trial[t], trial[w]))))
                    foreign = len({p for p in new_pairs - existing if p not in allowed_contacts})
                # fewer contacts the mask does not have, then fewer new pinches, then the largest agreement
                key = (-foreign, -created, agreement)
                if best is None or key > best[0]:
                    best = (key, ts, old, list(new), cyc, created, agreement, foreign)
        if best is None:
            continue
        if allowed_contacts is None or best[-1] == 0:
            return _apply_pinch_move(labels, v, cyc0, best)
        # every repair with k changes invents a contact the mask does not have: look further (more
        # changes) for one that does not, and fall back to the fewest invented contacts
        if fallback is None or best[-1] < fallback[-1]:
            fallback = best
    if fallback is not None:
        return _apply_pinch_move(labels, v, cyc0, fallback)
    return None


def _apply_pinch_move(labels, v, cyc0, best):
    _key, ts, old, new, cyc, created, agreement, foreign = best
    labels[np.asarray(ts, dtype=np.int64)] = np.asarray(new, dtype=np.int64)
    return dict(vertex=int(v), cycle_before=cyc0, cycle_after=cyc, triangles=[int(t) for t in ts],
                labels_from=old, labels_to=[int(x) for x in new], agreement=float(agreement),
                new_pinches=int(created), contacts_absent_from_the_mask=int(foreign))


def _labels_with_several_loops(tri, labels, delaunay_graph):
    edges = defaultdict(list)
    for (v1, v2), (a, b) in zip(delaunay_graph.Edges, delaunay_graph.Nodes_linked_by_Edges):
        la, lb = int(labels[a]), int(labels[b])
        if la != lb:
            edges[la].append((int(v1), int(v2)))
            edges[lb].append((int(v1), int(v2)))
    for (v1, v2), t in zip(delaunay_graph.Lone_edges, delaunay_graph.Nodes_linked_by_lone_edges):
        lt = int(labels[t])
        if lt != 0:
            edges[lt].append((int(v1), int(v2)))
            edges[0].append((int(v1), int(v2)))
    out = {}
    for L, es in edges.items():
        adj = defaultdict(set)
        for a, b in es:
            adj[a].add(b)
            adj[b].add(a)
        seen, pieces = set(), 0
        for s in adj:
            if s in seen:
                continue
            pieces += 1
            stack = [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
        if pieces > 1:
            out[int(L)] = pieces
    return out


def repair_labelled_triangulation(delaunay_graph, map_end, seed_nodes, max_passes=10, max_changes=3,
                                  mask_labels=None):
    """Absorb fragments and repair pinch vertices of a labelled 2D Delaunay triangulation.

    Args:
        delaunay_graph: dw2d ``Delaunay_Graph`` (see the module docstring for the fields used).
        map_end: {label: list of triangle ids} from the seeded watershed (after its zero-node
            override), as stored in ``geometry_reconstruction_2d.Map_end``.
        seed_nodes: triangle ids of the watershed seeds (never relabelled).
        mask_labels: optional label image the reconstruction was built from. When given, a pinch
            repair that would make two labels touch although they do not touch in the mask is
            ranked after every repair that does not (see the module docstring).
        max_passes: bound on the number of fragment + pinch passes.
        max_changes: largest number of fan triangles one pinch repair may relabel.

    Returns:
        (new_map_end, SurgeryReport). ``new_map_end is map_end`` when nothing changed.
    """
    tri = _Triangulation(delaunay_graph)
    labels = np.full(tri.n, -1, dtype=np.int64)
    for L, nodes in map_end.items():
        labels[np.asarray(nodes, dtype=np.int64)] = int(L)
    if (labels < 0).any():
        raise ValueError("every triangle must carry a label")
    original = labels.copy()
    seed_set = set(int(s) for s in np.atleast_1d(seed_nodes))
    allowed = mask_contacts(mask_labels) if mask_labels is not None else None
    report = SurgeryReport()
    for _pass in range(max_passes):
        report.passes += 1
        changed = _fragment_pass(tri, labels, seed_set, report)
        for v in find_pinch_vertices(tri, labels):
            fan, closed = tri.fans[v]
            if not _has_repeated_run(_cycle(fan, closed, labels)):
                continue  # repaired as a side effect of an earlier move in this pass
            move = _repair_one_pinch(tri, labels, v, seed_set, max_changes, allowed)
            if move is not None:
                changed = True
                report.pinch_moves.append(move)
        if not changed:
            break
    report.unresolved_pinch_vertices = find_pinch_vertices(tri, labels)
    comps = {}
    for L, members in _members_by_label(labels).items():
        cs = _components(tri, labels, members)
        if L == 0:
            outer, inner = _medium_component_groups(tri, cs, seed_set)
            n = (1 if outer else 0) + len(inner)
        else:
            n = len(cs)
        if n > 1:
            comps[int(L)] = n
    report.labels_with_several_components = comps
    report.labels_with_several_loops = _labels_with_several_loops(tri, labels, delaunay_graph)
    report.n_triangles_changed = int((labels != original).sum())
    if report.n_triangles_changed == 0:
        return map_end, report
    new_map = {}
    for idx, L in enumerate(labels.tolist()):
        new_map.setdefault(L, []).append(idx)
    return new_map, report
