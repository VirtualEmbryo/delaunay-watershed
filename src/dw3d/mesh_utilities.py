"""Module for Mesh creation and cleaning.

Decoupling the surface vertex set from the tesselation point set
------------------------------------------------------------------
`dw3d`'s extraction is **primal**: a mesh triangle is a face of the tetrahedralisation, so
every mesh vertex *is* a seed point. The boundary layer exploits that the other way
round — it adds offset points a distance `delta` into each adjacent material precisely so
that near-interface tetrahedra stop having all four vertices on the interface, which it
achieves spectacularly (all-surface tets 40 % -> 0.8 %). But the offsets are then surface
vertices too, and they sit `delta` voxels off the surface **by design**.

Measurement showed what that costs: the offsets are 30-35 % of mesh vertices, carry
`epsilon_rms = 2.51` voxels against the other families' 0.58-0.60, and account for **90 % of
`epsilon^2`** — where `epsilon` is the vertex-to-true-surface distance that sets the
junction-angle error through the calibrated `error = C * epsilon / ell`, `C = 33.2 +- 1.2` deg.

`exclude_offsets_from_surface` removes them from the surface while leaving the tesselation,
the watershed and the labelling untouched: each offset is *merged onto its parent* — the
interface sample it is a `delta`-displacement of, which lies on the interface — and the
triangles that collapse as a result are dropped. The surface vertex set becomes a subset of
the interface samples and junction samples, i.e. of the points that were placed *on* the
geometry, and the tesselation keeps every point it needs.

Two measurements decided this construction over the more literal alternative (move the
vertex to "the EDT zero-set" along a tet edge):

* **The EDT has no zero-set at the interface.** `compute_edt_classical` builds a *valley*,
  not a signed distance: interface samples sit at EDT 0.0-1.24 (median 0.82) with the level
  set by the local boundary thickness, and cell interiors are `distance + max(edt_2)`. The
  interface is a local *minimum* set, so "project to the zero level set" is ill-posed; the
  well-posed operation is a 1-D minimisation across the sheet.
* **Doing that minimisation lands each offset 0.026 voxel (median) from an interface sample
  that is already a mesh vertex.** The offsets carry no independent surface geometry: they
  are duplicates of the samples they were displaced from. Projecting them therefore produced
  1234 triangles under 1 deg of minimum angle, 92 of exactly zero area and 409 normal flips
  on `3.tif` alone — while merging them is exact, needs no interpolation and no threshold.

Sacha Ichbiah 2021.
Matthieu Perez 2024.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dw3d.tesselation_graph import TesselationGraph


#####
#####
# Mesh creation
#####
#####


def labeled_mesh_from_labeled_graph(
    tesselation_graph: "TesselationGraph",
    map_label_to_nodes_ids: dict[int, list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
    """Extract a labeled mesh from a labeled tesselation graph.

    Args:
        tesselation_graph (TesselationGraph): TesselationGraph object
        map_label_to_nodes_ids (dict[int, list[int]]): Labels on this graph.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]: points, triangles, and labels (materials)
    """
    # Take a Segmentation class as entry

    (
        points,
        triangles,
        labels,
        nodes_idx_in_graph_linked_to_triangle,
    ) = _retrieve_mesh_multimaterial_multitracker_format(
        tesselation_graph,
        map_label_to_nodes_ids,
    )

    # sort label (and nodes)
    for i, label in enumerate(labels):
        if label[0] > label[1]:  # if label0 > label1 we swap them
            labels[i] = labels[i, [1, 0]]
            nodes_idx_in_graph_linked_to_triangle[i] = nodes_idx_in_graph_linked_to_triangle[i][[1, 0]]
    # reorient triangles to have coherent normals (for plotting mostly)
    triangles = _reorient_triangles(
        triangles,
        tesselation_graph.vertices,
        tesselation_graph.tetrahedrons,
        nodes_idx_in_graph_linked_to_triangle,
    )

    return (points, triangles, labels)


def _retrieve_mesh_multimaterial_multitracker_format(
    tesselation_graph: "TesselationGraph",
    map_label_to_nodes: dict[int, list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint], NDArray[np.uint]]:
    """Extract multi-material mesh from the tesselation graph with every nodes (tetrahedrons) marked with a material.

    The extracted surface mesh is composed of all triangles that are faces of 2 tetrahedrons with different materials.
    Note that there will be a lot of unused points in this mesh. A filtering step will be necessary.

    Args:
        tesselation_graph (TesselationGraph): the tesselation graph.
        map_label_to_nodes (dict[int, list[int]]): The map that links materials to tetrahedrons id.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.uint], list[int], NDArray[np.uint]]:
            - an array of vertices of the mesh. Note that there are some unused points, to be filtered later.
            - an array of triangles (id p1, id p2, id p3)
            - an array of labels (material 1, material 2)
            - the list of ids of nodes (tetrahedrons) in graph.nodes that are linked to selected triangles.
    """
    # faces = triangles ; nodes = tetras, with map linking regions to list of tetra/node ids
    map_nodes_to_labels: dict[int, int] = {}
    for key in map_label_to_nodes:
        for node_idx in map_label_to_nodes[key]:
            map_nodes_to_labels[node_idx] = key
    triangles: list[list[int]] = []  # p1, p2, p3, selected faces
    labels: list[list[int]] = []  # l1, l2 of selected faces
    nodes_linked_by_face: list[list[int]] = []  # list nodes ids linked to triangles from selected graph faces

    for idx, face in enumerate(tesselation_graph.triangle_faces):
        # faces are triangles, nodes linked are the 2 adjacent tetrahedrons (nodes).
        nodes_linked = tesselation_graph.nodes_linked_by_faces[idx]

        # corresponding regions
        cluster_1 = map_nodes_to_labels[nodes_linked[0]]
        cluster_2 = map_nodes_to_labels[nodes_linked[1]]
        cells = [cluster_1, cluster_2]

        if cluster_1 != cluster_2:
            # some faces belong to 2 tetras of two different regions : those are the triangles in final mesh !
            triangles.append([face[0], face[1], face[2]])  # tri p1, p2, p3, label l1, l2
            labels.append([cells[0], cells[1]])
            nodes_linked_by_face.append(nodes_linked)  # 2 tetras

    # Matthieu Perez:
    # Apparently, there are triangles on only one tetra in the tesselation graph,
    # and sometimes they might belong to the mesh ? If segmented cell touches the border of the image ?
    for idx in range(len(tesselation_graph.lone_faces)):
        face = tesselation_graph.lone_faces[idx]
        node_linked = tesselation_graph.nodes_linked_by_lone_faces[idx]
        cluster_1 = map_nodes_to_labels[node_linked]
        # We incorporate all these edges because they are border edges
        if cluster_1 != 0:
            cells = [0, cluster_1]
            triangles.append([face[0], face[1], face[2]])
            labels.append([cells[0], cells[1]])
            nodes_linked_by_face.append(nodes_linked)

    # Note that the extraction might lead to a mesh that is non-manifold where it is not expected to,
    # if the labeling of nodes is not perfect. Some kind of "mesh surgery" might be necessary to improve
    # extracted mesh quality.
    return (
        tesselation_graph.vertices,
        np.array(triangles, dtype=np.uint),
        np.array(labels, dtype=np.uint),
        np.array(nodes_linked_by_face, dtype=np.uint),
    )


GUARD_MAX_ROUNDS = 8
COLLAPSE_MAX_ROUNDS = 8

WELD = "weld"
LINK_CONDITION = "link_condition"
COLLAPSE_RULES = (WELD, LINK_CONDITION)


def exclude_offsets_from_surface(
    points: NDArray[np.float64],
    triangles: NDArray[np.uint],
    labels: NDArray[np.uint],
    surface_merge_target: NDArray[np.int64],
    guard_duplicate_faces: bool = True,
    collapse_rule: str = WELD,
) -> tuple[NDArray[np.uint], NDArray[np.uint], dict]:
    """Merge every boundary-layer offset vertex onto its parent interface sample.

    See the module docstring for why this is a merge and not a displacement. Each merge is a
    half-edge collapse, and they are **independent**: `surface_merge_target` is identity
    except on the offset families and no offset is ever another offset's target, so the merge
    itself is one vectorised re-indexing with no ordering to choose.

    Geometry is not touched — no point moves — so `points` is returned to the caller
    unchanged and the surviving vertices keep the exact coordinates the tesselation gave
    them. What changes is which points the triangles reference.

    **Which merges are allowed is set by `collapse_rule`**, and that is the whole difference
    between the unconditional weld and the link-condition-checked collapse:

    * `"weld"` (the historical, unconditional-merge behaviour) merges every offset it can, with only the
      duplicate-face guard below standing in the way. It is *not* topology preserving: welding
      two distinct vertices makes formerly distinct edges the same edge, so their triangle
      incidences add. Measured over the 51-case benchmark this took valence->=4 edges
      115 -> 184 and abnormal non-manifold edges 7 -> 76.
    * `"link_condition"` (the topology-preserving rule) accepts a merge only when contracting the edge
      `(offset, parent)` cannot change the topology of the surface. See
      `_refuse_by_link_condition` for the three tests and why each is needed. It is the
      recommended rule; `"weld"` is kept reachable so the original offset-exclusion
      measurements stay reproducible.

    **The guard, and why it is on by default.** (`"weld"` only — the link condition subsumes
    it, see condition 3 in `_refuse_by_link_condition`.) A few merges land two triangles on the same
    vertex triple: a flap flattened onto the film it hangs from. Measured three ways on
    `3.tif`, keeping one copy of each such pair opens **4 boundary
    edges** where the flap was what closed the surface; keeping both copies leaves **55
    valence-4 edges** (the coincident pairs) against the default's 0; dropping both opens 42.
    Refusing the handful of merges that would create a duplicate instead leaves the edge
    valence histogram **identical to the default's** (0 boundary edges, 227 valence-3, 0
    valence-≥4) at the cost of leaving **38 of 650 offsets** in the surface — 5.8 %, whose
    residual contribution to `epsilon` was measured separately. That is the trade taken,
    and `guard_duplicate_faces=False` reproduces the unguarded construction so it stays
    measurable rather than merely asserted.

    The guard iterates because un-merging one offset can leave another duplicate behind; it
    converges in **one** round on every case measured, and `GUARD_MAX_ROUNDS` bounds it.
    Same-label duplicates that survive the guard are still deduplicated (one copy kept) so
    the return value is well defined either way.

    Args:
        points (NDArray[np.float64]): tesselation vertices (unchanged; read only to report
            the surface area before and after).
        triangles (NDArray[np.uint]): extracted surface triangles, indexing `points`.
        labels (NDArray[np.uint]): the (material, material) pair of each triangle.
        surface_merge_target (NDArray[np.int64]): per tesselation point, the point the
            surface should use in its place; identity for every non-offset point. From the
            point-placing scheme's `point_metadata` (`dw3d.points_on_edt`).
        guard_duplicate_faces (bool, optional): refuse a merge that would put two triangles
            on one vertex triple. Defaults to True; see above. Ignored — and reported as
            ignored — when `collapse_rule` is `"link_condition"`, which forbids the same
            thing on stronger grounds.
        collapse_rule (str, optional): `"weld"` (the unconditional-merge default, unchanged) or
            `"link_condition"` (the topology-preserving rule). See above.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint], dict]: the re-indexed triangles, their
            labels, and a report:
                collapse_rule          — the rule that was applied;
                n_triangles_before / n_triangles_after;
                n_triangles_collapsed  — dropped for having a repeated vertex after the merge;
                n_merges_refused       — offsets left in the surface by the guard or the
                    link condition;
                n_guard_rounds         — iterations the guard / link-condition sweep needed
                    (0 when it is off or when there was nothing to refuse);
                n_refused_no_edge / n_refused_link_vertices / n_refused_link_edges /
                n_refused_valence      — the link condition's refusal breakdown, all 0 under
                    `"weld"`; see `_refuse_by_link_condition`;
                n_duplicate_triples    — distinct vertex-triples still carrying more than one
                    surviving triangle;
                n_duplicates_dropped   — triangles dropped as same-label duplicates;
                n_duplicate_conflicts  — duplicate triples whose triangles disagree on the
                    label pair. **Reported, not repaired**: a conflict would mean the merge
                    has fused two different interfaces, so it must be visible rather than
                    silently deduplicated.
                n_surface_vertices_before / n_surface_vertices_after;
                n_offset_vertices_removed — surface vertices that were offsets and are gone;
                area_before / area_after.
    """
    if collapse_rule not in COLLAPSE_RULES:
        message = f"collapse_rule must be one of {COLLAPSE_RULES}, got {collapse_rule!r}"
        raise ValueError(message)

    triangles = np.asarray(triangles, dtype=np.int64)
    target = np.asarray(surface_merge_target, dtype=np.int64).copy()
    if len(triangles) == 0:
        report = _empty_exclusion_report()
        report["collapse_rule"] = collapse_rule
        return triangles.astype(np.uint), np.asarray(labels, dtype=np.uint), report

    # No offset may be another offset's target: both rules apply the accepted merges as a
    # single vectorised re-indexing `target[triangles]`, which is only equal to applying them
    # one after another when there is no chain to follow. Cheap to check, and it is the
    # assumption the whole construction rests on.
    if not np.array_equal(target[target], target):
        message = "surface_merge_target is chained: some point's target is itself a merged point"
        raise ValueError(message)

    before = np.unique(triangles)
    refusals: dict[str, int] = {}
    if collapse_rule == LINK_CONDITION:
        n_refused, n_rounds, refusals = _refuse_by_link_condition(triangles, target)
    elif guard_duplicate_faces:
        n_refused, n_rounds = _refuse_duplicating_merges(triangles, target)
    else:
        n_refused, n_rounds = 0, 0
    merged = target[triangles]

    # A triangle whose three vertices are no longer distinct has collapsed to an edge or a
    # point: it is exactly the "flap" the offset held open, and it carries no area.
    distinct = (merged[:, 0] != merged[:, 1]) & (merged[:, 1] != merged[:, 2]) & (merged[:, 0] != merged[:, 2])
    kept, kept_labels = merged[distinct], np.asarray(labels)[distinct]

    _keys, first, inverse, counts = _unique_triples(kept)
    duplicate_groups = np.nonzero(counts > 1)[0]
    n_conflicts = 0
    for group in duplicate_groups:
        members = np.nonzero(inverse == group)[0]
        if len({tuple(sorted(int(x) for x in kept_labels[i])) for i in members}) > 1:
            n_conflicts += 1
    # Same-label duplicates are one film meeting itself: keep one copy, since two coincident
    # triangles bound zero volume, and dropping both would open a hole.
    duplicate_members = np.isin(inverse, duplicate_groups)
    survives = ~duplicate_members
    survives[first[duplicate_groups]] = True

    n_duplicates_dropped = int((~survives).sum())
    kept, kept_labels = kept[survives], kept_labels[survives]

    after = np.unique(kept)
    report = {
        "collapse_rule": collapse_rule,
        "guard_duplicate_faces": bool(guard_duplicate_faces) and collapse_rule == WELD,
        **{f"n_refused_{reason}": refusals.get(reason, 0) for reason in REFUSAL_REASONS},
        **{f"n_fired_{reason}": refusals.get(f"fired_{reason}", 0) for reason in REFUSAL_REASONS},
        "n_triangles_before": len(triangles),
        "n_triangles_after": len(kept),
        "n_triangles_collapsed": int((~distinct).sum()),
        "n_merges_refused": n_refused,
        "n_guard_rounds": n_rounds,
        "n_duplicate_triples": len(duplicate_groups),
        "n_duplicates_dropped": n_duplicates_dropped,
        "n_duplicate_conflicts": int(n_conflicts),
        "n_surface_vertices_before": len(before),
        "n_surface_vertices_after": len(after),
        "n_offset_vertices_removed": int(np.count_nonzero(target[before] != before)),
        # The area the flaps were carrying. Not a cost: a flap reaching `delta` voxels off the
        # surface and back adds area no interface has, so the drop is the wrinkle coming out.
        "area_before": float(_triangle_areas(points, triangles).sum()),
        "area_after": float(_triangle_areas(points, kept).sum()),
    }
    return kept.astype(np.uint), np.asarray(kept_labels, dtype=np.uint), report


def _refuse_duplicating_merges(triangles: NDArray[np.int64], target: NDArray[np.int64]) -> tuple[int, int]:
    """Un-merge, **in place in `target`**, the offsets whose collapse duplicates a face.

    A duplicate means two triangles landing on one vertex triple. Rather than choose between
    keeping one copy (which opens a boundary edge where the flap was closing the surface) and
    keeping both (which creates a valence-4 edge), refuse those few merges: the offsets stay
    in the surface, carrying their `epsilon`, and the topology is exactly the default's. See
    `exclude_offsets_from_surface` for the measured comparison.

    Every offset appearing in *any* triangle of a duplicate group is refused together, which
    makes the result independent of the order the groups are visited. Refusing one offset can
    reveal a new duplicate, so this iterates; it converges in one round on every case measured
    and stops at `GUARD_MAX_ROUNDS` regardless, leaving any residue to the deduplication (and
    to `n_duplicate_triples` in the report).

    Returns `(n_merges_refused, n_rounds)`.
    """
    identity = np.arange(len(target))
    n_refused = 0
    for round_index in range(GUARD_MAX_ROUNDS):
        merged = target[triangles]
        distinct = (merged[:, 0] != merged[:, 1]) & (merged[:, 1] != merged[:, 2]) & (merged[:, 0] != merged[:, 2])
        _keys, _first, inverse, counts = _unique_triples(merged[distinct])
        duplicate_groups = np.nonzero(counts > 1)[0]
        if len(duplicate_groups) == 0:
            return n_refused, round_index
        rows = np.nonzero(distinct)[0][np.isin(inverse, duplicate_groups)]
        involved = np.unique(triangles[rows])
        refuse = involved[target[involved] != identity[involved]]
        if len(refuse) == 0:
            # The duplicate does not come from a merge, so refusing cannot remove it.
            return n_refused, round_index
        target[refuse] = refuse
        n_refused += len(refuse)
    return n_refused, GUARD_MAX_ROUNDS


def _refuse_by_link_condition(
    triangles: NDArray[np.int64],
    target: NDArray[np.int64],
) -> tuple[int, int, dict[str, int]]:
    """Un-merge, **in place in `target`**, every merge that is not a safe edge contraction.

    The unconditional weld merged each offset onto its parent regardless of topology. Welding two vertices makes
    formerly distinct edges the same edge, so their triangle incidences *add*: edge `(a, p)`
    inherits the incidences of `(a, O1)`, `(a, O2)`, ... That is exactly the regression
    measured for the unconditional weld over the 51-case benchmark — valence->=4 edges
    115 -> 184, abnormal non-manifold edges 7 -> 76, almost all of the added valence->=4 edges
    being the 3-material "pinched triple line" kind.

    A weld of `O` onto `p` is a **contraction of the edge `(O, p)`** of the surface complex,
    and there is a standard test for when that cannot change the topology: the *link
    condition* `lk(O) & lk(p) == lk(Op)` (Dey, Edelsbrunner, Guha & Nekhyev 1999,
    *Topology preserving edge contraction*). For the 2-complex here, `lk(v)` is the vertices
    joined to `v` by an edge together with the opposite edges of the triangles at `v`, and
    `lk(Op)` is just the third vertices of the triangles on the edge `(O, p)`. This function
    applies that test, plus one more, because on **this** complex — a 2-complex that is
    non-manifold by design, with triple lines everywhere — the link condition as stated above
    is satisfied by configurations that still change the topology (see test 4):

    1. **`(O, p)` must be an edge of the surface.** Otherwise the merge is not a contraction
       at all but an identification of two vertices that were not joined, which pinches two
       sheets together. Refused, counted as `no_edge` (measured: 21-24 offsets per case).
    2. **Vertex part of the link condition:** every `a` joined by an edge to both `O` and `p`
       must have `(O, p, a)` as a triangle. Counted as `link_vertices` (15-17 per case). This
       is what keeps the arithmetic in test 3 exact: it forces the number of triangles
       carrying all three of `a`, `O`, `p` to be exactly one for each such `a`.
    3. **Edge part of the link condition:** no pair `(w, x)` may have both `(O, w, x)` and
       `(p, w, x)` as triangles, since the contraction would make those the same triangle.
       Counted as `link_edges`. **This subsumes the weld's duplicate-face guard** — it forbids
       creating a duplicate triple — which is why `guard_duplicate_faces` is ignored under
       this rule.
    4. **Edge valences must be preserved.** Tests 1-3 do not give this here, and the
       counterexample is exactly the configuration that the weld hit. (Whether that contradicts
       DEGN's sufficiency claim or merely falls outside its hypotheses has not been checked
       against the paper; that question is still open. The test
       is cheap and it is the invariant the acceptance criterion measures, so it is asserted
       directly rather than deduced.) Take a triangle `(O, p, a)` whose edges `(a, O)` and `(a, p)` are both triple
       lines: `lk(O) & lk(p) == lk(Op) == {a}` holds, yet after the contraction `(a, p)`
       carries `val(a, O) + val(a, p) - 2 == 4` triangles — a fresh valence-4 edge with only
       3 materials on it, i.e. an abnormal non-manifold edge. So the valence bookkeeping is
       checked directly: by test 2 the only edges whose valence can change are the `(a, p)`
       for `a` in `lk(Op)`, each ending at `val(a, O) + val(a, p) - 2`, so the merge is
       accepted only when `min(val(a, O), val(a, p)) == 2` for every such `a` — the condition
       under which that expression is one of the two inputs and no edge valence moves.
       Counted as `valence`.

    With 1-4 no edge valence in the surface can *increase*, so `holes`, `valence->=4 edges`
    and `abnormal non-manifold edges` are bounded above by the un-merged surface's. That is a
    guarantee from the construction, not an observation — but it is asserted against the
    measurement anyway in `benchmarks/analyze_offset_exclusion.py`.

    The sweep is **sequential**: each candidate is tested against the complex as the merges
    accepted before it have left it, so two offsets sharing a parent cannot both spend the
    same valence budget. Candidates are visited in ascending tesselation index, which is the
    only ordering choice made and is deterministic. Refused candidates are retried, because
    accepting one merge can make a neighbouring one legal (it converges in 3 rounds on every
    case measured); `COLLAPSE_MAX_ROUNDS` bounds it.

    Returns `(n_merges_refused, n_rounds, counts)`. `counts` carries, per test, both the
    number of candidates *finally* refused by it (`no_edge`, `link_vertices`, `link_edges`,
    `valence`) and the number of times it fired at all across every round
    (`fired_no_edge`, ...), since a test that only ever fires on candidates that a later
    round accepts is doing no work and should be visible as such.
    """
    surface = _SurfaceComplex(triangles)
    identity = np.arange(len(target))
    pending = sorted(int(v) for v in np.unique(triangles) if target[v] != identity[v])
    reasons: dict[str, int] = dict.fromkeys(REFUSAL_REASONS, 0)
    fired: dict[str, int] = {f"fired_{reason}": 0 for reason in REFUSAL_REASONS}
    n_rounds = 0

    for round_index in range(COLLAPSE_MAX_ROUNDS):
        n_rounds = round_index + 1
        still_pending: list[int] = []
        accepted_this_round = 0
        reasons = dict.fromkeys(REFUSAL_REASONS, 0)

        for offset in pending:
            parent = int(target[offset])
            if not surface.is_used(offset):
                continue  # the offset has already left the surface; there is nothing to refuse
            reason = _collapse_verdict(surface, offset, parent)
            if reason is not None:
                reasons[reason] += 1
                fired[f"fired_{reason}"] += 1
                still_pending.append(offset)
                continue
            surface.contract(offset, parent)
            accepted_this_round += 1

        pending = still_pending
        if accepted_this_round == 0 or not pending:
            break

    if pending:
        indices = np.asarray(pending, dtype=np.int64)
        target[indices] = indices
    return len(pending), n_rounds, {**reasons, **fired}


REFUSAL_REASONS = ("no_edge", "link_vertices", "link_edges", "valence")


class _SurfaceComplex:
    """The extracted surface as a mutable 2-complex, indexed for the link-condition tests.

    Only what `_refuse_by_link_condition` needs: which triangles touch a vertex, and the
    ability to contract one vertex onto another. Triangles are never renumbered, so a
    contracted-away triangle is simply dropped from every incidence set and its stale row is
    never read again.
    """

    def __init__(self, triangles: NDArray[np.int64]) -> None:
        self.rows: list[list[int]] = [[int(a), int(b), int(c)] for a, b, c in triangles.tolist()]
        self.at: dict[int, set[int]] = {}
        for index, row in enumerate(self.rows):
            for vertex in row:
                self.at.setdefault(vertex, set()).add(index)

    def is_used(self, vertex: int) -> bool:
        """Does any surviving triangle reference this vertex?"""
        return bool(self.at.get(vertex))

    def triangles_on_edge(self, first: int, second: int) -> set[int]:
        """The triangles carrying both endpoints; its size is the edge's valence."""
        return self.at.get(first, set()) & self.at.get(second, set())

    def neighbours(self, vertex: int) -> set[int]:
        """The vertex part of `lk(vertex)`: everything joined to it by an edge."""
        found: set[int] = set()
        for index in self.at[vertex]:
            found.update(self.rows[index])
        found.discard(vertex)
        return found

    def opposite_edges(self, vertex: int) -> set[tuple[int, int]]:
        """The edge part of `lk(vertex)`: the opposite side of each triangle at it."""
        found: set[tuple[int, int]] = set()
        for index in self.at[vertex]:
            rest = [v for v in self.rows[index] if v != vertex]
            if len(rest) == 2:  # a degenerate row contributes no link edge
                found.add((min(rest), max(rest)))
        return found

    def contract(self, vertex: int, onto: int) -> None:
        """Contract the edge `(vertex, onto)`, keeping `onto`. Assumes the verdict passed."""
        for index in list(self.at[vertex]):
            row = self.rows[index]
            if onto in row:
                # The triangle collapses onto the edge it already shared with `onto`.
                for corner in row:
                    self.at[corner].discard(index)
            else:
                self.rows[index] = [onto if corner == vertex else corner for corner in row]
                self.at[vertex].discard(index)
                self.at[onto].add(index)
        self.at[vertex].clear()


def _collapse_verdict(surface: _SurfaceComplex, offset: int, parent: int) -> str | None:
    """Return the reason to refuse contracting `(offset, parent)`, or None to accept it.

    The four tests are documented, with their justification, in `_refuse_by_link_condition`.
    """
    shared = surface.triangles_on_edge(offset, parent)
    if not shared:
        return "no_edge"

    link_of_edge = {v for index in shared for v in surface.rows[index] if v not in (offset, parent)}
    if surface.neighbours(offset) & surface.neighbours(parent) != link_of_edge:
        return "link_vertices"

    if surface.opposite_edges(offset) & surface.opposite_edges(parent):
        return "link_edges"

    # Test 2 has just established one triangle on (apex, offset, parent) for each apex, so
    # the merged edge (apex, parent) ends with val(apex, offset) + val(apex, parent) - 2
    # triangles; requiring one of the two to be 2 is requiring that to be the other one.
    for apex in link_of_edge:
        valence_to_offset = len(surface.triangles_on_edge(apex, offset))
        valence_to_parent = len(surface.triangles_on_edge(apex, parent))
        if min(valence_to_offset, valence_to_parent) != 2:
            return "valence"
    return None


def _triangle_areas(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.float64]:
    """Area of each triangle."""
    corners = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    return 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )


def _unique_triples(triangles: NDArray[np.int64]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Group triangles by their *unordered* vertex triple."""
    sorted_triples = np.sort(triangles, axis=1)
    keys, first, inverse, counts = np.unique(
        sorted_triples,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return keys, first, inverse.reshape(-1), counts


def _empty_exclusion_report() -> dict:
    return {
        "guard_duplicate_faces": False,
        **{f"n_refused_{reason}": 0 for reason in REFUSAL_REASONS},
        **{f"n_fired_{reason}": 0 for reason in REFUSAL_REASONS},
        "n_merges_refused": 0,
        "n_guard_rounds": 0,
        "n_triangles_before": 0,
        "n_triangles_after": 0,
        "n_triangles_collapsed": 0,
        "n_duplicate_triples": 0,
        "n_duplicates_dropped": 0,
        "n_duplicate_conflicts": 0,
        "n_surface_vertices_before": 0,
        "n_surface_vertices_after": 0,
        "n_offset_vertices_removed": 0,
        "area_before": 0.0,
        "area_after": 0.0,
    }


###############
# Mesh Cleaning
###############
def set_points_min_max(
    points: NDArray[np.float64],
    global_min: float,
    global_max: float,
) -> NDArray[np.float64]:
    """Return a homogeneously scaled points array such that its min & max values are global_min and global_max."""
    current_min = points.min()
    current_max = points.max()
    new_points = (
        (np.copy(points) - current_min) / (current_max - current_min) * (global_max - global_min)
    ) + global_min

    return new_points


def set_pixel_size(
    points: NDArray[np.float64],
    xy_pixel_size: float,
    z_pixel_size: float,
) -> NDArray[np.float64]:
    """Return a scaled points array from pixel coordinates space to real coordinates.

    Given microscope's xy and z pixel size.
    """
    new_points = np.copy(points)
    new_points[:, :2] *= xy_pixel_size
    new_points[:, 2] *= z_pixel_size

    return new_points


def center_around_origin(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a centered points array around 0."""
    current_min = points.min(axis=0)
    current_max = points.max(axis=0)

    return np.copy(points) - (current_max + current_min) / 2.0


def _reorient_triangles(
    triangles: NDArray[np.uint],
    vertices: NDArray[np.float64],
    tetrahedrons: NDArray[np.intc],
    nodes_linked: NDArray[np.uint],
) -> NDArray[np.uint]:
    """Swap point order in triangles such that all normals points in the same direction.

    Args:
        triangles (NDArray[np.uint]): Triangles of a multimaterial mesh.
        vertices (NDArray[np.float64]): Vertices of a 3D tesselation (geometry).
        tetrahedrons (NDArray[np.intc]): Tetrahedrons of the same 3D tesselation (topology).
        nodes_linked (NDArray[np.uint]): Tetrahedrons id on the tesselation linked to each triangles of the mesh.

    Returns:
        NDArray[np.uint]: reoriented triangles.
    """
    # Thumb rule for all the faces

    normals = _compute_normal_faces(vertices, triangles)

    points = vertices[triangles]
    centroids_faces = np.mean(points, axis=1)  # center of tirangles
    centroids_nodes = np.mean(
        vertices[tetrahedrons[nodes_linked[:, 0]]],
        axis=1,
    )  # center of "first" adjacent tetrahedron in Tesselation Graph

    vectors = centroids_nodes - centroids_faces

    dot_product = np.sum(np.multiply(vectors, normals), axis=1)
    normals_sign = np.sign(dot_product)

    # Reorientation according to the normal sign
    reoriented_triangles = triangles.copy()

    # Matthieu Perez: one liner is quicker when there's more than 100 faces (ie. always)
    reoriented_triangles[normals_sign > 0] = reoriented_triangles[normals_sign > 0][:, [0, 2, 1]]
    return reoriented_triangles


def _compute_normal_faces(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> NDArray[np.float64]:
    """Return the normalized normals for each triangles."""
    positions = points[triangles]
    sides_1 = positions[:, 1] - positions[:, 0]
    sides_2 = positions[:, 2] - positions[:, 1]
    normals = np.cross(sides_1, sides_2, axis=1)
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.array([norms] * 3).transpose()
    return normals


def filter_unused_points(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """Take a mesh made from points and triangles and remove points not indexed in triangles. Re-index triangles.

    Return the filtered points and reindexed triangles.
    """
    used_points_id = np.unique(triangles)
    used_points = np.copy(points[used_points_id])
    idx_mapping = np.arange(len(used_points))
    mapping = dict(zip(used_points_id, idx_mapping, strict=True))

    reindexed_triangles = np.fromiter(
        (mapping[xi] for xi in triangles.reshape(-1)),
        dtype=np.ulonglong,
        count=3 * len(triangles),
    ).reshape((-1, 3))

    return (used_points, reindexed_triangles)
