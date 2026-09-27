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


def post_process_mesh_surgery(reconstruction_algorithm: "MeshReconstructionAlgorithm", max_iter: int = 3) -> None:
    """Optional part of post-process mesh surgery, separated from MeshReconstructionAlgorithm for clarity."""
    for _iter in range(max_iter):
        points, triangles, labels = reconstruction_algorithm.last_constructed_mesh
        abnormal_edges = _find_abnormal_non_manifold_edges(points, triangles, labels)
        if len(abnormal_edges) == 0:
            break

        change_made = False

        for abnormal_edge in abnormal_edges:
            # find ordered cycle of tetrahedrons around edge (& labels)
            tetra_cycle = reconstruction_algorithm._tesselation_graph.find_tetra_cycle_around_edge(abnormal_edge)
            label_cycle = list(reconstruction_algorithm._map_node_id_to_label[tetra_cycle])
            # find where we can switch label and switch
            candidates_id = _find_candidate_for_label_switching(label_cycle)

            if len(candidates_id) > 0:
                change_made = True
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

                # Apply change
                _apply_one_label_switch(reconstruction_algorithm, tetra_cycle, selected_switch)

            else:  # no candidate for one label switching, try double label switching
                cm = _search_and_make_double_switching(
                    reconstruction_algorithm,
                    tetra_cycle,
                    label_cycle,
                )
                change_made = change_made or cm  # beware of lazy evaluation ! Compute cm first.

        # switch label => what does it change for the mesh ? do we recompute everything ? that the simple way
        if change_made:
            reconstruction_algorithm._points, reconstruction_algorithm._triangles, reconstruction_algorithm._labels = (
                labeled_mesh_from_labeled_graph(
                    reconstruction_algorithm._tesselation_graph,
                    reconstruction_algorithm._map_label_to_nodes_ids,
                )
            )

    # To check remaining abnormal edges
    # points, triangles, labels = reconstruction_algorithm.last_constructed_mesh
    # abnormal_edges = _find_abnormal_non_manifold_edges(points, triangles, labels)
    # for abnormal_edge in abnormal_edges:
    #     # find ordered cycle of tetrahedrons around edge (& labels)
    #     tetra_cycle = reconstruction_algorithm._tesselation_graph.find_tetra_cycle_around_edge(abnormal_edge)
    #     label_cycle = list(reconstruction_algorithm._map_node_id_to_label[tetra_cycle])
    #     print(label_cycle, "for edge", abnormal_edge)


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


def _search_and_make_double_switching(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    tetra_cycle: list[int],
    label_cycle: list[int],
) -> bool:
    """Search if there is two consecutive switches that could help."""
    candidates_id = _find_candidate_for_two_labels_switching(label_cycle)
    change_made = False
    if len(candidates_id) > 0:
        change_made = True
        selected_switch = tuple(candidates_id)  # when only one candidate

        # when more than one candidate, we try all possibilities
        if len(candidates_id) > 1:
            switches_to_test = _powerset(candidates_id)
            results_of_switch = [_double_label_switching(label_cycle, to_switch) for to_switch in switches_to_test]
            evaluation_of_switch = [_evaluate_cycle(cycle) for cycle in results_of_switch]
            # argMAX, not argmin -- same inversion, same reason as in `post_process_mesh_surgery`
            # above: `_evaluate_cycle`'s docstring says "The higher the better", and its 0 is the
            # reserved "avoid at all cost" score for the edge-deleting all-one-label cycle.
            selected_switch = switches_to_test[np.argmax(evaluation_of_switch)]

        # Apply change
        _apply_two_label_switch(reconstruction_algorithm, tetra_cycle, selected_switch)
    return change_made


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
    fix would need to change candidate selection, not this function's precondition.
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
