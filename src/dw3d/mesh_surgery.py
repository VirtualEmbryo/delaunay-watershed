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


def post_process_mesh_surgery(
    reconstruction_algorithm: "MeshReconstructionAlgorithm",
    max_iter: int = 3,
) -> None:
    """Optional part of post-process mesh surgery, separated from MeshReconstructionAlgorithm for clarity."""
    points, triangles, labels = reconstruction_algorithm.last_constructed_mesh
    abnormal_edges = _find_abnormal_non_manifold_edges(points, triangles, labels)

    for abnormal_edge in abnormal_edges:
        # find ordered cycle of tetrahedrons around edge (& labels)
        tetra_cycle = reconstruction_algorithm._tesselation_graph.find_tetra_cycle_around_edge(abnormal_edge)
        label_cycle = reconstruction_algorithm._map_node_id_to_label[tetra_cycle]
        # find where we can switch label and switch
        current_nb_iter = 0
        candidates_id = _find_candidate_for_label_switching(label_cycle)
        print(label_cycle, candidates_id)

        while len(candidates_id) > 0 and current_nb_iter < max_iter:
            candidate = candidates_id[0]
            tetra_to_change = tetra_cycle[candidate]
            old_label = reconstruction_algorithm._map_node_id_to_label[tetra_to_change]
            new_label = reconstruction_algorithm._map_node_id_to_label[tetra_cycle[candidate - 1]]

            reconstruction_algorithm._map_node_id_to_label[tetra_to_change] = new_label

            reconstruction_algorithm._map_label_to_nodes_ids[old_label].remove(tetra_to_change)
            reconstruction_algorithm._map_label_to_nodes_ids[new_label].append(tetra_to_change)

            label_cycle = reconstruction_algorithm._map_node_id_to_label[tetra_cycle]
            candidates_id = _find_candidate_for_label_switching(label_cycle)
            current_nb_iter += 1

            # sometimes there's more than one candidates and it's not obvious to me which one to choose...
            # so I arbitrarily choose the first for now.

    # switch label => what does it change for the mesh ? do we recompute everything ? that the simple way
    reconstruction_algorithm._points, reconstruction_algorithm._triangles, reconstruction_algorithm._labels = (
        labeled_mesh_from_labeled_graph(
            reconstruction_algorithm._tesselation_graph,
            reconstruction_algorithm._map_label_to_nodes_ids,
        )
    )


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
    edges_key = edges[:, 0] * len(points) + edges[:, 1]
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


def _evaluate_cycle(label_cycle: list[int]) -> int:
    """The higher the better."""
    label_cycle = np.array(label_cycle, dtype=np.int64)

    _, index_counts = np.unique(
        label_cycle,
        return_counts=True,
    )
    return np.min(index_counts)
