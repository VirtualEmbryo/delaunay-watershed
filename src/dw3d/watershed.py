"""Module defind the watershed algorithm on a networkx graph.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import networkx
import numpy as np
from numpy.typing import NDArray


#####
# SEEDED WATERSHED
#####
def seeded_watershed_map(
    nx_graph: networkx.Graph,
    seeds_nodes: NDArray[np.uint],
    indices_labels: NDArray[np.uint8],
    zero_nodes: NDArray[np.uint] | None = None,
) -> tuple[dict[int, list[int]], NDArray[np.uint]]:
    """Perform Watershed algorithm on tetrahedron nodes to label them.

    Return the map label -> indices of nodes and inverse map.
    """
    # Init the map node -> label with seeds
    map_node_id_to_label = np.zeros(len(nx_graph.nodes), dtype=int) - 1

    # Seeds are expressed as labels of the nodes
    for i, seed_node in enumerate(seeds_nodes):
        map_node_id_to_label[seed_node] = indices_labels[i]

    map_node_id_to_label = _seeded_watershed_aggregation(nx_graph, map_node_id_to_label)

    # Matthieu Perez: next 2 lines seems useless, test without it
    if zero_nodes is None:
        map_node_id_to_label[zero_nodes] = 0

    map_label_to_nodes = _build_map_label_to_node_ids(map_node_id_to_label)
    return map_label_to_nodes, map_node_id_to_label


def _seeded_watershed_aggregation(
    nx_graph: networkx.Graph,
    map_node_id_to_label: NDArray[np.uint],
) -> NDArray[np.uint8]:
    """Perform Watershed algorithm on tetrahedron nodes to label them. Return the labels array."""
    groups = {}
    number_group = np.zeros(len(nx_graph.nodes), dtype=int) - 1
    num_group = 0

    scores = -np.array(list(nx_graph.edges.data("score")))[:, 2]
    args = np.argsort(scores)  # Note : edges.data('score') gives [node edge 1, node edge 2, score data]
    edges = list(nx_graph.edges)
    for arg in args:
        a, b = edges[arg]
        if map_node_id_to_label[a] != -1 and map_node_id_to_label[b] != -1:
            continue
        elif map_node_id_to_label[a] != -1 and map_node_id_to_label[b] == -1:
            group = groups.get(number_group[b], [b])
            map_node_id_to_label[group] = map_node_id_to_label[a]
        elif map_node_id_to_label[b] != -1 and map_node_id_to_label[a] == -1:
            group = groups.get(number_group[a], [a])
            map_node_id_to_label[group] = map_node_id_to_label[b]
        else:  # here labels are both -1, unknown.
            # the triangles has a high score but both tetras it belongs to have not been seen before
            if number_group[a] != -1:  # maybe we identified a group
                if number_group[a] == number_group[b]:
                    continue
                elif number_group[b] != -1:
                    old_b_group = groups.pop(number_group[b])
                    groups[number_group[a]] += old_b_group
                    number_group[old_b_group] = number_group[a]
                else:
                    groups[number_group[a]].append(b)
                    number_group[b] = number_group[a]
            else:
                if number_group[b] != -1:
                    groups[number_group[b]].append(a)
                    number_group[a] = number_group[b]
                else:
                    number_group[a] = num_group
                    number_group[b] = num_group
                    groups[num_group] = [a, b]
                    num_group += 1
    return map_node_id_to_label


def _build_map_label_to_node_ids(map_node_id_to_label: NDArray[np.uint8]) -> dict[int, list[int]]:
    """Reverse the map node id to label to give a map label to node indices."""
    map_label_to_node_ids: dict[int, list[int]] = {}
    for idx, label in enumerate(map_node_id_to_label):
        map_label_to_node_ids[label] = map_label_to_node_ids.get(label, [])
        map_label_to_node_ids[label].append(idx)
    return map_label_to_node_ids
