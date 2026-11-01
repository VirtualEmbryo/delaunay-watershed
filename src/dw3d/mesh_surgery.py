"""Module for mesh surgery after mesh reconstruction.

The mesh reconstruction algorithm can lead to non-manifold edges that should be manifold.
We try to correct that.

Matthieu Perez 2024
"""

from collections.abc import Iterable
from itertools import chain, combinations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dw3d.mesh_utilities import labeled_mesh_from_labeled_graph

if TYPE_CHECKING:
    from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm
    from dw3d.tesselation_graph import TesselationGraph


class AbnormalEdgeWarning(UserWarning):
    """A finished mesh still has abnormal non-manifold edges.

    Emitted by :meth:`dw3d.reconstruction_algorithm.MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`
    **once per reconstruction**, when the final mesh has an edge carried by four or more triangles
    that do not separate as many regions (`_find_abnormal_non_manifold_edges`): a cell surface that
    touches itself, or two cells that meet twice, along that edge. The message gives their number
    and the configuration that made the mesh (point placement and its parameters, mesh surgery,
    tetrahedron order, offset exclusion). Such edges come from mesh surgery being off, from an edge
    surgery could not remove without adding another, or from a step after surgery (the offset
    exclusion of the boundary-layer configurations). Downstream, a junction reader may fail or
    skip that edge. A subclass of :class:`UserWarning`, so it can be silenced or promoted to an
    error on its own with :func:`warnings.filterwarnings`.
    """


def post_process_mesh_surgery(reconstruction_algorithm: "MeshReconstructionAlgorithm", max_iter: int = 20) -> None:
    """Optional part of post-process mesh surgery, separated from MeshReconstructionAlgorithm for clarity.

    Each iteration finds the abnormal non-manifold edges of the mesh and, for each, relabels one or
    a few tetrahedra of the ring around it. Every relabelling is scored by how many abnormal edges
    the mesh gains or loses with it, counted exactly on the edges of the relabelled tetrahedra (the
    only edges whose incident triangles a relabelling can change; `_RingNeighbourhood`), and the
    first of these is applied (`_select_relabelling`):

    1. the rule's own choice (`_rule_relabelling`: the single- or double-switch candidates below),
       if it lowers the count, or leaves it unchanged without giving a tetrahedron a label it has
       already held during this surgery -- so wherever the rule repaired the mesh without undoing
       itself, the mesh is the one the rule alone gave;
    2. otherwise the alternative (`_alternative_relabellings`) that lowers the count most;
    3. otherwise an alternative that leaves the count unchanged, never one that gives a tetrahedron
       a label it has already held;
    4. otherwise nothing: the edge is left as it is.

    A relabelling that raises the count is never applied, and no step that leaves it unchanged can
    undo an earlier one, so surgery cannot oscillate. It stops as soon as no abnormal edge is left or
    an iteration changes nothing; `max_iter` (20 since 0.7.1, 3 before) is only a bound. The old bound
    of 3 cut short repairs that move a defect along before removing it: on simulated tissues at a real
    cell size, 3 meshes of 168 kept an abnormal edge at 3 iterations and none after 5 or 6.

    Every walk is strictly descending: the steps after the last one that lowered the count -- steps
    that each kept the count, in the hope of moving a defect to where it can be removed -- are
    reverted when surgery stops, and the tetrahedra get back their labels, in the label maps' own
    order. So surgery returns either a mesh with fewer abnormal edges than it started with, every
    unchanged-count step of which led to a lower count, or, when it lowered nothing, the mesh it was
    given. On one golden-master mesh of the boundary-layer configuration (image 3, `min_distance` 3),
    surgery lowers the count from 54 to 1 and then took 6 unchanged-count steps that never reached a
    repair; they are now reverted. Edges surgery cannot remove are reported when the reconstruction
    ends (`AbnormalEdgeWarning`).

    Why: the rule gives a switched tetrahedron the label of its *previous* neighbour in the ring,
    and the ring's direction and starting point follow the tesselation's row order. On some rings
    the rule's choice repairs the edge but creates an abnormal edge next to it, whose own repair by
    the rule gives the same tetrahedron its label back and so recreates the first edge: a period-2
    oscillation that `max_iter` stopped at an arbitrary point. Which direction the ring was walked in
    decided whether it happened: on the benchmark's anisotropic routes the same edge was left by 2 of
    9 tetrahedron row orders of the same tetrahedra and by the canonical order, and repaired by the 7
    others, through the same unchanged-count first step.
    """
    graph = reconstruction_algorithm._tesselation_graph
    held: dict[int, set[int]] = {}  # tetrahedron -> the labels it has held during this surgery
    walk = _UncommittedWalk(reconstruction_algorithm)
    for _iter in range(max_iter):
        points, triangles, labels = reconstruction_algorithm.last_constructed_mesh
        abnormal_edges = _find_abnormal_non_manifold_edges(points, triangles, labels)
        if len(abnormal_edges) == 0:
            break

        change_made = False

        for abnormal_edge in abnormal_edges:
            # find ordered cycle of tetrahedrons around edge (& labels)
            tetra_cycle = graph.find_tetra_cycle_around_edge(abnormal_edge)
            label_cycle = list(reconstruction_algorithm._map_node_id_to_label[tetra_cycle])
            selected = _select_relabelling(reconstruction_algorithm, tetra_cycle, label_cycle, held)
            if selected is None:
                continue
            kind, selected_switch, relabelling, change = selected
            labels_now = reconstruction_algorithm._map_node_id_to_label
            for tetrahedron in relabelling:
                held.setdefault(tetrahedron, set()).add(int(labels_now[tetrahedron]))
            walk.record(tetra_cycle)
            _apply_selected(reconstruction_algorithm, tetra_cycle, kind, selected_switch, relabelling)
            if change < 0:
                walk.commit()
            change_made = True

        # switch label => what does it change for the mesh ? do we recompute everything ? that the simple way
        if not change_made:
            break
        _rebuild_mesh(reconstruction_algorithm)

    if walk.revert():
        _rebuild_mesh(reconstruction_algorithm)


def _apply_selected(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    tetra_cycle: list[int],
    kind: str,
    selected_switch: tuple[int, ...],
    relabelling: dict[int, int],
) -> None:
    """Apply what `_select_relabelling` chose; the rule's choice exactly as the rule always applied it."""
    if kind == "single":
        _apply_one_label_switch(reconstruction_algorithm, tetra_cycle, selected_switch)
    elif kind == "double":
        _apply_two_label_switch(reconstruction_algorithm, tetra_cycle, selected_switch)
    else:
        _apply_relabelling(reconstruction_algorithm, relabelling)


def _rebuild_mesh(reconstruction_algorithm: "MeshReconstructionAlgorithm") -> None:
    """Extract the mesh again from the tesselation graph and the current labels."""
    reconstruction_algorithm._points, reconstruction_algorithm._triangles, reconstruction_algorithm._labels = (
        labeled_mesh_from_labeled_graph(
            reconstruction_algorithm._tesselation_graph,
            reconstruction_algorithm._map_label_to_nodes_ids,
        )
    )


class _UncommittedWalk:
    """The steps applied since surgery last lowered the abnormal-edge count, so that they can be reverted.

    Before a step, `record` saves the label of every tetrahedron of the ring and the node list of
    every label on it (a superset of what the step changes: the rule's switches move a tetrahedron
    between two lists of the ring's labels, even when it ends with its own label), the first time
    each is touched since the last `commit`; `revert` puts them back, so the label maps are exactly
    as they were after the last step that lowered the count, list order included (it sets the order
    of the extracted triangles).
    """

    def __init__(self, reconstruction_algorithm: "MeshReconstructionAlgorithm") -> None:
        self._algorithm = reconstruction_algorithm
        self._labels: dict[int, int] = {}
        self._lists: dict[int, list[int]] = {}

    def record(self, tetra_cycle: list[int]) -> None:
        """Save what a step on this ring can change, unless already saved since the last commit."""
        node_labels = self._algorithm._map_node_id_to_label
        lists = self._algorithm._map_label_to_nodes_ids
        for tetrahedron in tetra_cycle:
            label = int(node_labels[tetrahedron])
            self._labels.setdefault(int(tetrahedron), label)
            if label not in self._lists:
                self._lists[label] = list(lists[label])

    def commit(self) -> None:
        """Keep every step so far: the count has just been lowered."""
        self._labels.clear()
        self._lists.clear()

    def revert(self) -> bool:
        """Undo every step since the last commit; whether there was any."""
        if not self._labels:
            return False
        for tetrahedron, label in self._labels.items():
            self._algorithm._map_node_id_to_label[tetrahedron] = label
        for label, nodes in self._lists.items():
            self._algorithm._map_label_to_nodes_ids[label][:] = nodes
        self.commit()
        return True


def _select_relabelling(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    tetra_cycle: list[int],
    label_cycle: list[int],
    held: dict[int, set[int]],
) -> tuple[str, tuple[int, ...], dict[int, int], int] | None:
    """Steps 1-4 of `post_process_mesh_surgery` for one ring: `(kind, switch, relabelling, change)` or None.

    `kind` is `"single"` or `"double"` for the rule's choice (then `switch` is its subset of
    positions) and `"alternative"` otherwise (`switch` empty); `change` is the change of the
    abnormal-edge count the relabelling makes (negative or zero). Among alternatives, the lower change
    in the abnormal-edge count wins, then the higher `_evaluate_cycle` score of the ring after the
    relabelling, then fewer relabelled tetrahedra, then the order of `_alternative_relabellings`. A
    relabelling that would take the last tetrahedron of a region away is never selected.
    """
    neighbourhood = _RingNeighbourhood(
        reconstruction_algorithm._tesselation_graph,
        reconstruction_algorithm._map_node_id_to_label,
        tetra_cycle,
    )
    rule = _rule_relabelling(tetra_cycle, label_cycle)
    if rule is not None and rule[2]:
        rule_change = neighbourhood.change_in_abnormal_edges(rule[2])
        if rule_change < 0 or (rule_change == 0 and not _undoes(rule[2], held)):
            return (*rule, rule_change)

    scored = []
    for order, relabelling in enumerate(_alternative_relabellings(tetra_cycle, label_cycle)):
        if _empties_a_region(reconstruction_algorithm, relabelling):
            continue
        change = neighbourhood.change_in_abnormal_edges(relabelling)
        if change > 0:
            continue
        resulting = [relabelling.get(int(t), int(lab)) for t, lab in zip(tetra_cycle, label_cycle, strict=True)]
        scored.append(((change, -_evaluate_cycle(resulting), len(relabelling), order), relabelling))
    improving = [item for item in scored if item[0][0] < 0]
    for candidates in (improving, [item for item in scored if not _undoes(item[1], held)]):
        if candidates:
            key, relabelling = min(candidates, key=lambda item: item[0])
            return "alternative", (), relabelling, key[0]
    return None


def _undoes(relabelling: dict[int, int], held: dict[int, set[int]]) -> bool:
    """Whether a relabelling gives a tetrahedron a label it has already held during this surgery."""
    return any(label in held.get(tetrahedron, ()) for tetrahedron, label in relabelling.items())


def _empties_a_region(reconstruction_algorithm: "MeshReconstructionAlgorithm", relabelling: dict[int, int]) -> bool:
    """Whether a relabelling would take every remaining tetrahedron away from one region."""
    taken: dict[int, int] = {}
    for tetrahedron in relabelling:
        old = int(reconstruction_algorithm._map_node_id_to_label[tetrahedron])
        taken[old] = taken.get(old, 0) + 1
    return any(len(reconstruction_algorithm._map_label_to_nodes_ids[old]) <= n for old, n in taken.items())


def _rule_relabelling(
    tetra_cycle: list[int],
    label_cycle: list[int],
) -> tuple[str, tuple[int, ...], dict[int, int]] | None:
    """The rule's choice for one ring: `(kind, selected switch, {tetrahedron: new label})`, or None.

    The single-switch candidates first, the double-switch candidates if there is none; among several
    candidates, the subset whose result `_evaluate_cycle` scores highest. The returned relabelling is
    what `_apply_one_label_switch` / `_apply_two_label_switch` do with that subset, which applies the
    switches one after the other, each reading its neighbour's label as already switched (this can
    differ from the cycle `_label_switching` scored); tetrahedra whose label does not change are left out.
    """
    candidates_id = _find_candidate_for_label_switching(label_cycle)
    if len(candidates_id) > 0:
        selected_switch = tuple(candidates_id)  # when only one candidate
        # when more than one candidate, we try all possibilities
        if len(candidates_id) > 1:
            switches_to_test = _powerset(candidates_id)
            results_of_switch = [_label_switching(label_cycle, to_switch) for to_switch in switches_to_test]
            evaluation_of_switch = [_evaluate_cycle(cycle) for cycle in results_of_switch]
            # argMAX, not argmin: `_evaluate_cycle` is a score to be maximised -- its own
            # docstring says "The higher the better ... We want to maximize this minimum",
            # and it returns 0 for the all-one-label cycle precisely to mean "avoid at all
            # cost", because that outcome deletes the edge. 0 is also the global minimum,
            # so selecting with argmin chose the degenerate collapse the score exists to
            # prevent whenever it was among the candidates. Do not re-invert this.
            selected_switch = switches_to_test[np.argmax(evaluation_of_switch)]
        return "single", selected_switch, _relabelling_of_switch(tetra_cycle, label_cycle, selected_switch, width=1)

    # no candidate for one label switching, try double label switching
    candidates_id = _find_candidate_for_two_labels_switching(label_cycle)
    if len(candidates_id) > 0:
        selected_switch = tuple(candidates_id)  # when only one candidate
        if len(candidates_id) > 1:
            switches_to_test = _powerset(candidates_id)
            results_of_switch = [_double_label_switching(label_cycle, to_switch) for to_switch in switches_to_test]
            evaluation_of_switch = [_evaluate_cycle(cycle) for cycle in results_of_switch]
            # argMAX, not argmin -- same inversion, same reason as above.
            selected_switch = switches_to_test[np.argmax(evaluation_of_switch)]
        return "double", selected_switch, _relabelling_of_switch(tetra_cycle, label_cycle, selected_switch, width=2)
    return None


def _relabelling_of_switch(
    tetra_cycle: list[int],
    label_cycle: list[int],
    selected_switch: tuple[int, ...],
    width: int,
) -> dict[int, int]:
    """What `_apply_one_label_switch` (`width` 1) or `_apply_two_label_switch` (`width` 2) do to a ring."""
    current = list(label_cycle)
    size = len(current)
    for candidate in selected_switch:
        new_label = current[candidate - 1]
        for offset in range(width):
            current[(candidate + offset) % size] = new_label
    return {int(tetra_cycle[i]): int(current[i]) for i in range(size) if current[i] != label_cycle[i]}


def _alternative_relabellings(tetra_cycle: list[int], label_cycle: list[int]) -> list[dict[int, int]]:
    """The relabellings surgery tries when the rule's own choice does not lower the abnormal-edge count.

    In this order, duplicates dropped: every subset of the rule's single- and double-switch candidates,
    on the ring as walked and on the ring walked the other way (so a switched tetrahedron takes its
    next neighbour's label as well as its previous one's); then every single tetrahedron of the ring
    given the label of either of its two neighbours.
    """
    out: list[dict[int, int]] = []
    seen: set[frozenset] = set()

    def add(relabelling: dict[int, int]) -> None:
        key = frozenset(relabelling.items())
        if relabelling and key not in seen:
            seen.add(key)
            out.append(relabelling)

    for tetrahedra, cycle in (
        (list(tetra_cycle), list(label_cycle)),
        (list(tetra_cycle)[::-1], list(label_cycle)[::-1]),
    ):
        for subset in _powerset(_find_candidate_for_label_switching(cycle)):
            add(_relabelling_of_switch(tetrahedra, cycle, subset, width=1))
        for subset in _powerset(_find_candidate_for_two_labels_switching(cycle)):
            add(_relabelling_of_switch(tetrahedra, cycle, subset, width=2))
    size = len(label_cycle)
    for i in range(size):
        for neighbour in (label_cycle[i - 1], label_cycle[(i + 1) % size]):
            if neighbour != label_cycle[i]:
                add({int(tetra_cycle[i]): int(neighbour)})
    return out


def _apply_relabelling(reconstruction_algorithm: "MeshReconstructionAlgorithm", relabelling: dict[int, int]) -> None:
    """Give each tetrahedron of `relabelling` its new label, in the two label maps of the algorithm."""
    for tetrahedron, new_label in relabelling.items():
        old_label = reconstruction_algorithm._map_node_id_to_label[tetrahedron]
        reconstruction_algorithm._map_node_id_to_label[tetrahedron] = new_label
        reconstruction_algorithm._map_label_to_nodes_ids[old_label].remove(tetrahedron)
        reconstruction_algorithm._map_label_to_nodes_ids[new_label].append(tetrahedron)


class _RingNeighbourhood:
    """The tesselation faces around one ring of tetrahedra, to count abnormal edges under a relabelling.

    A relabelling of some tetrahedra of the ring changes which faces are mesh triangles only among
    the faces of those tetrahedra, hence the incident triangles of their edges only. Every face that
    contains an edge of a ring tetrahedron has two of its vertices among the ring's vertices, so the
    faces kept here are all the faces any such edge has. On them an edge is abnormal exactly as
    `_find_abnormal_non_manifold_edges` defines it on the mesh: a face between two tetrahedra of
    different labels is a triangle with those two labels, a face on the tesselation's hull whose
    tetrahedron is not labelled 0 is a triangle with labels (0, that label), and an edge with `k >= 4`
    triangles whose labels are not `k` distinct regions is abnormal.
    """

    def __init__(self, graph: "TesselationGraph", node_labels: NDArray, tetra_cycle: list[int]) -> None:
        self._node_labels = node_labels
        ring_vertices = np.unique(graph.tetrahedrons[np.asarray(tetra_cycle)])
        shared = np.isin(graph.triangle_faces.astype(np.int64), ring_vertices).sum(axis=1) >= 2
        lone = np.isin(graph.lone_faces, ring_vertices).sum(axis=1) >= 2
        self._faces = np.vstack(
            (graph.triangle_faces[shared].astype(np.int64), np.asarray(graph.lone_faces[lone], dtype=np.int64)),
        )
        hull_side = np.full(int(lone.sum()), -1, dtype=np.int64)
        self._nodes = np.vstack(
            (
                graph.nodes_linked_by_faces[shared].astype(np.int64),
                np.column_stack((np.asarray(graph.nodes_linked_by_lone_faces[lone], dtype=np.int64), hull_side)),
            ),
        )
        self._tetrahedrons = graph.tetrahedrons

    def _edges_of(self, tetrahedra: Iterable[int]) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for tetrahedron in tetrahedra:
            vertices = sorted(int(v) for v in self._tetrahedrons[tetrahedron])
            edges.update(combinations(vertices, 2))
        return edges

    def _n_abnormal(self, edges: set[tuple[int, int]], relabelling: dict[int, int]) -> int:
        def label(node: int) -> int:
            return 0 if node < 0 else relabelling.get(node, int(self._node_labels[node]))

        count = 0
        for a, b in edges:
            rows = np.flatnonzero((self._faces == a).any(axis=1) & (self._faces == b).any(axis=1))
            regions: set[int] = set()
            n_triangles = 0
            for row in rows:
                first, second = (label(int(node)) for node in self._nodes[row])
                if first != second:
                    n_triangles += 1
                    regions.update((first, second))
            if n_triangles > 3 and len(regions) != n_triangles:
                count += 1
        return count

    def change_in_abnormal_edges(self, relabelling: dict[int, int]) -> int:
        """How many abnormal edges the mesh gains (positive) or loses (negative) with this relabelling."""
        edges = self._edges_of(relabelling)
        return self._n_abnormal(edges, relabelling) - self._n_abnormal(edges, {})


def _apply_one_label_switch(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    tetra_cycle: list[int],
    selected_switch: tuple[int, ...],
) -> None:
    """Apply the selected switch to modify the map label <-> tetras for the algorithm (one label switched per value)."""
    for candidate in selected_switch:
        tetra_to_change1 = tetra_cycle[candidate]
        old_label1 = reconstruction_algorithm._map_node_id_to_label[tetra_to_change1]
        new_label = reconstruction_algorithm._map_node_id_to_label[tetra_cycle[candidate - 1]]

        reconstruction_algorithm._map_node_id_to_label[tetra_to_change1] = new_label

        reconstruction_algorithm._map_label_to_nodes_ids[old_label1].remove(tetra_to_change1)
        reconstruction_algorithm._map_label_to_nodes_ids[new_label].append(tetra_to_change1)


def _apply_two_label_switch(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    tetra_cycle: list[int],
    selected_switch: tuple[int, ...],
) -> None:
    """Apply the selected switch to modify the map label <-> tetras for the algorithm (two label switched per value)."""
    for candidate in selected_switch:
        new_label = reconstruction_algorithm._map_node_id_to_label[tetra_cycle[candidate - 1]]

        tetra_to_change1 = tetra_cycle[candidate]
        old_label1 = reconstruction_algorithm._map_node_id_to_label[tetra_to_change1]

        reconstruction_algorithm._map_node_id_to_label[tetra_to_change1] = new_label

        reconstruction_algorithm._map_label_to_nodes_ids[old_label1].remove(tetra_to_change1)
        reconstruction_algorithm._map_label_to_nodes_ids[new_label].append(tetra_to_change1)

        tetra_to_change2 = tetra_cycle[(candidate + 1) % len(tetra_cycle)]
        old_label2 = reconstruction_algorithm._map_node_id_to_label[tetra_to_change2]

        reconstruction_algorithm._map_node_id_to_label[tetra_to_change2] = new_label

        reconstruction_algorithm._map_label_to_nodes_ids[old_label2].remove(tetra_to_change2)
        reconstruction_algorithm._map_label_to_nodes_ids[new_label].append(tetra_to_change2)


def _find_abnormal_non_manifold_edges(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> NDArray[np.uint]:
    """Find edges in mesh that appear 4 or more times but are not at the intersection of 4 or more cells.

    Returns:
        NDArray[np.uint]: an array of those edges (id_p1, id_p2)
    """
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    edges = np.sort(edges, axis=1)
    edges_key = edges[:, 0] * (len(points) + 1) + edges[:, 1]
    _, index_first_occurence, index_counts = np.unique(
        edges_key,
        return_index=True,
        return_counts=True,
    )
    more_than_trijunctions_id = index_counts > 3
    suspicious_edges = edges[index_first_occurence[more_than_trijunctions_id]]
    suspicious_counts = index_counts[more_than_trijunctions_id]
    abnormal_mask = np.ones(len(suspicious_counts), dtype=np.bool_)

    for i, edge in enumerate(suspicious_edges):
        if _number_of_adjacent_cells_of_edge(triangles, labels, edge) == suspicious_counts[i]:
            abnormal_mask[i] = False

    return suspicious_edges[abnormal_mask]


def _number_of_adjacent_cells_of_edge(
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    edge: tuple[int, int],
) -> int:
    """Return the number of adjacent cells of an edge."""
    pid1, pid2 = edge

    adjacent_triangles_id = np.where(
        np.logical_and(
            (triangles == pid1).any(axis=1),
            (triangles == pid2).any(axis=1),
        ),
    )[0]

    adjacent_labels = labels[adjacent_triangles_id]

    return len(np.unique(adjacent_labels))


def _find_candidate_for_label_switching(label_cycle: list[int]) -> list[int]:
    """Find label-cycle positions that can be switched to their previous neighbour's label.

    The `elif` branch below (from a later step of the identifiability-floor work,
    see the project's research history) widens this beyond the
    original rule's single case. It targets the "AABB shape": an edge already flagged abnormal
    by `_find_abnormal_non_manifold_edges`, whose tetrahedron label cycle has exactly one
    material `A` occupying exactly two runs, at least one of length 1, whose two flanking
    neighbours are two *different* materials `B != C` -- at the mesh level, 4 incident triangles,
    3 materials, label-pair multiset `{(A,B): 2, (A,C): 2}` (two films each doubled, sharing one
    material). 64/87 (74%) of abnormal edges dataset-wide share this shape, including all three
    md=3 survivors (cases 011, 013, 036).

    The original `if` branch declines this shape because it only fires when the label *before*
    the run equals the label *after* it (`label_cycle[next_id] == label_cycle[previous_id]`),
    which is false here by construction (`B != C`). The widened `elif` instead accepts a
    length-1 run when its label recurs elsewhere in the cycle (`count(...) >= 2`), which is
    exactly what identifies the second, doubled occurrence of `A`.

    Effect and reuse: a widened candidate is switched to its previous neighbour's label via the
    same `_apply_one_label_switch` / powerset-and-evaluate machinery already used for the
    original rule's candidates -- no new action code.

    Blast radius on legitimate edges is zero by construction: this function is only ever called
    on edges `_find_abnormal_non_manifold_edges` already flagged, so a genuine trijunction or
    quadjunction is never passed in regardless of what this condition would return on it.

    Failure mode: on already-abnormal edges, measured full-batch, the widened rule resolves
    every abnormal edge to zero at md 2, 3, 5 and 7 in that work's own
    47-case/24-clean-control validation, but is *not* uniformly an improvement at md=1
    (a spacing disqualified regardless by that validation's acceptance criteria): 72 -> 26,
    with 3 of 35 cases getting strictly worse, because relabelling one
    tetrahedron in a denser, defect-clustered md=1 mesh can create a fresh anomaly at a
    neighbouring edge that was previously fine.

    A later full-47-case/5-rung verification of this fix (see the project's research history)
    found one exception to the "clean at md 2,3,5,7" reading above: case 031 at md=3 goes from 0 to 1
    abnormal edges. Root cause, traced with `post_process_mesh_surgery`'s own iteration log: the
    widened rule turns a single forced candidate into three (`[0,1,2]` instead of `[1]`) on one
    edge; the powerset-argmin selection in `post_process_mesh_surgery`/`_evaluate_cycle` picks a
    different one than the original rule would have, which creates a new abnormal edge whose own
    repair recreates the first -- a genuine 2-cycle (confirmed non-convergent to 5 periods at
    `max_iter=10`, diagnostic only), left mid-cycle by production's `max_iter=3`. This is the
    same class of pre-existing oscillation `test_abnormal_edge_survivors.py` already documents
    for two other cases under the *original* rule ("Mechanism 2"), and the risk the widening's own
    specification flagged and left unresolved (whether `_evaluate_cycle`'s
    `np.argmin` can pick the worst-scored split when several candidates compete): case
    031 is that risk materialising. Not fixed here -- out of scope for the widening, and a
    fix would need to change candidate selection, not this function's precondition. Since
    0.7.1, `post_process_mesh_surgery` scores every relabelling by the abnormal edges it adds
    or removes and never applies one that adds more than it removes, nor undoes an earlier
    one with no gain, so such a 2-cycle can no longer happen; this function is unchanged.
    """
    candidates = []
    nb_labels = len(label_cycle)
    for current_id in range(nb_labels):
        previous_id = (current_id - 1) % nb_labels
        if label_cycle[previous_id] == label_cycle[current_id]:
            continue
        next_id = (current_id + 1) % nb_labels
        if label_cycle[next_id] == label_cycle[current_id]:
            continue
        # here current id != prev and next
        if label_cycle[next_id] == label_cycle[previous_id]:
            # and they are the same !
            candidates.append(current_id)
        elif label_cycle.count(label_cycle[current_id]) >= 2:
            # AABB shape: A's other run is elsewhere in the cycle.
            candidates.append(current_id)

    return candidates


def _find_candidate_for_two_labels_switching(label_cycle: list[int]) -> list[int]:
    candidates = []
    nb_labels = len(label_cycle)

    for current_id in range(nb_labels):
        p1 = current_id
        p2 = (current_id + 1) % nb_labels
        if label_cycle[p1] != label_cycle[p2]:
            continue

        pp1 = (current_id + 2) % nb_labels
        pp2 = (current_id + 3) % nb_labels
        if label_cycle[pp1] != label_cycle[pp2] or label_cycle[pp1] == label_cycle[p2]:
            continue
        pm1 = (current_id - 1) % nb_labels
        pm2 = (current_id - 2) % nb_labels
        if (
            label_cycle[pm1] != label_cycle[pm2]
            or label_cycle[pm1] != label_cycle[pp1]
            or label_cycle[pm2] == label_cycle[p1]
        ):
            continue

        candidates.append(current_id)

    return candidates


def _powerset(iterable: Iterable) -> list[tuple]:
    """powerset([1,2,3]) → (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)."""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


def _label_switching(label_cycle: list[int], to_switch: list[int]) -> list[int]:
    """Return new label cycle with label switched."""
    result = label_cycle.copy()

    for i in to_switch:
        result[i] = label_cycle[i - 1]

    return result


def _double_label_switching(label_cycle: list[int], to_switch: list[int]) -> list[int]:
    """Return new label cycle with two consecutive label switched."""
    result = label_cycle.copy()

    for i in to_switch:
        result[i] = label_cycle[i - 1]
        result[(i + 1) % len(label_cycle)] = label_cycle[i - 1]

    return result


def _evaluate_cycle(label_cycle: list[int]) -> int:
    """The higher the better. Try to balance labels in a cycle and avoid at all cost having only one label.

    The balance is achieved by computing the minimum of consecutives labels. We want to maximize this minimum.
    """
    # we avoid solutions that lead to deletion of edge by giving a bad evaluation if all labels are the same
    if all(x == label_cycle[0] for x in label_cycle):
        return 0

    label_cycle = label_cycle.copy()
    # Here not all values are the same.
    # We cycle until the first element is different from the last
    while label_cycle[0] == label_cycle[-1]:
        label_cycle.append(label_cycle[0])
        del label_cycle[0]

    # count differences now
    size = len(label_cycle)
    current = label_cycle[0]
    current_consecutives = 1
    min_consecutives = size
    for i in range(1, size):  # there is at least two (different) elements
        if label_cycle[i] == current:
            current_consecutives += 1
        else:
            if min_consecutives > current_consecutives:
                min_consecutives = current_consecutives
            current_consecutives = 1
            current = label_cycle[i]
    if min_consecutives > current_consecutives:
        min_consecutives = current_consecutives

    return min_consecutives
