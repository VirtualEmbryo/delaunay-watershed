"""The seeded watershed visits edges of equal score in edge order (stable sort), on every platform.

Pins the fix of 0.8.0: the default unstable ``np.argsort`` broke ties of equal edge scores differently in
NumPy 1.26 and 2.5, so the meshes of large images depended on the NumPy version.
"""
import networkx
import numpy as np

from dw2d.Networkx_functions import seeded_watershed_aggregation


def test_equal_scores_are_visited_in_edge_order():
    # a path 0 - 1 - 2 - 3 with every score equal; seeds at both ends with different labels.
    G = networkx.Graph()
    G.add_nodes_from(range(4))
    for a, b in ((0, 1), (1, 2), (2, 3)):
        G.add_edge(a, b, score=1.0)
    labels = seeded_watershed_aggregation(G, [0, 3], [10, 20])
    # edge (0, 1) first gives 1 the label of 0; then (1, 2) gives 2 that label too; (2, 3) joins two labelled nodes.
    assert labels.tolist() == [10, 10, 10, 20]


def test_many_equal_scores_give_the_same_labels_as_an_explicit_stable_order():
    rng = np.random.default_rng(0)
    G = networkx.gnm_random_graph(300, 900, seed=1)
    for a, b in G.edges:
        G.edges[a, b]["score"] = float(rng.integers(0, 3))  # many ties
    seeds = [0, 1, 2]
    lab = seeded_watershed_aggregation(G, seeds, [1, 2, 3])
    lab2 = seeded_watershed_aggregation(G, seeds, [1, 2, 3])
    assert np.array_equal(lab, lab2)
