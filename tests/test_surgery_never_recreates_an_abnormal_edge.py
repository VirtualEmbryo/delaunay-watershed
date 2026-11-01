"""Mesh surgery never applies a relabelling that adds abnormal edges, and never undoes itself.

The defect: the switch rule gives a tetrahedron the label of its *previous* neighbour in the ring
around an abnormal edge, and the ring's direction follows the tesselation's row order. On benchmark
case 037 meshed at k = 5/3 (the anisotropic, resampled route), the rule repairs one edge by
relabelling a tetrahedron, which creates a second abnormal edge, whose repair gives the same
tetrahedron its label back and recreates the first: `max_iter = 3` stopped this period-2 oscillation
with one abnormal edge left, under the canonical tetrahedron order and 2 of 9 row orders of the same
tetrahedra. The 7 other orders walk the ring the other way and end with none.

The fixture is that mesh's labelled tesselation just before surgery (canonical order;
`tests/golden/surgery_case_037_resampled_k5_3_before_surgery.npz`, 133 kB, written by the research
record's `save_the_pre_surgery_tesselation_of_case_037.py`), so these tests need neither the
benchmark dataset nor a reconstruction. Pinned:

* the rule alone, applied as before, leaves 1 abnormal edge on the fixture, and the guarded surgery
  leaves 0, under the canonical order and 9 row permutations of the same tetrahedra;
* every relabelling the guarded surgery applies lowers the abnormal-edge count or leaves it
  unchanged, and the unchanged ones never give a tetrahedron a label it held before;
* the local count of `_RingNeighbourhood` equals the change of the whole mesh's count by
  `_find_abnormal_non_manifold_edges`, on every candidate relabelling around the fixture's edges.
"""

from pathlib import Path

import numpy as np
import pytest

import dw3d.mesh_surgery as mesh_surgery
from dw3d.mesh_surgery import (
    _alternative_relabellings,
    _find_abnormal_non_manifold_edges,
    _RingNeighbourhood,
    _rule_relabelling,
    post_process_mesh_surgery,
)
from dw3d.mesh_utilities import labeled_mesh_from_labeled_graph
from dw3d.tesselation import canonical_tetrahedron_order
from dw3d.tesselation_graph import TesselationGraph

FIXTURE = Path(__file__).parent / "golden" / "surgery_case_037_resampled_k5_3_before_surgery.npz"
N_ROW_PERMUTATIONS = 9
ROW_PERMUTATION_SEED = 20261022


class _LabelledTesselation:
    """The part of `MeshReconstructionAlgorithm` that surgery reads and writes."""

    def __init__(self, vertices: np.ndarray, tetrahedra: np.ndarray, labels: np.ndarray) -> None:
        graph = TesselationGraph.__new__(TesselationGraph)
        graph.vertices = vertices
        graph.nodes = tetrahedra
        graph.n_simplices = len(tetrahedra)
        graph._construct_edges(graph._construct_edges_table())
        self._tesselation_graph = graph
        self._map_node_id_to_label = labels.copy()
        self._map_label_to_nodes_ids = {
            int(label): [int(i) for i in np.flatnonzero(labels == label)] for label in np.unique(labels)
        }
        self._points, self._triangles, self._labels = labeled_mesh_from_labeled_graph(
            graph,
            self._map_label_to_nodes_ids,
        )

    @property
    def last_constructed_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._points, self._triangles, self._labels

    def n_abnormal(self) -> int:
        return len(_find_abnormal_non_manifold_edges(*self.last_constructed_mesh))


@pytest.fixture(scope="module")
def stored() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def _permuted(stored: dict[str, np.ndarray], permutation: np.ndarray | None) -> _LabelledTesselation:
    if permutation is None:
        return _LabelledTesselation(stored["vertices"], stored["tetrahedra"], stored["labels"])
    return _LabelledTesselation(stored["vertices"], stored["tetrahedra"][permutation], stored["labels"][permutation])


def _row_permutations(n: int) -> list[np.ndarray | None]:
    rng = np.random.default_rng(ROW_PERMUTATION_SEED)
    return [None] + [rng.permutation(n) for _ in range(N_ROW_PERMUTATIONS)]


def _rule_only_surgery(algorithm: _LabelledTesselation, max_iter: int = 3) -> None:
    """Surgery as it was before the guard: the rule's choice, applied whatever it does to the count."""
    graph = algorithm._tesselation_graph
    for _ in range(max_iter):
        edges = _find_abnormal_non_manifold_edges(*algorithm.last_constructed_mesh)
        if len(edges) == 0:
            break
        changed = False
        for edge in edges:
            cycle = graph.find_tetra_cycle_around_edge(edge)
            rule = _rule_relabelling(cycle, list(algorithm._map_node_id_to_label[cycle]))
            if rule is None:
                continue
            kind, switch, _ = rule
            apply = mesh_surgery._apply_one_label_switch if kind == "single" else mesh_surgery._apply_two_label_switch
            apply(algorithm, cycle, switch)
            changed = True
        if changed:
            algorithm._points, algorithm._triangles, algorithm._labels = labeled_mesh_from_labeled_graph(
                graph,
                algorithm._map_label_to_nodes_ids,
            )


def test_the_fixture_is_in_canonical_order_and_has_abnormal_edges(stored):
    np.testing.assert_array_equal(canonical_tetrahedron_order(stored["tetrahedra"]), stored["tetrahedra"])
    assert _permuted(stored, None).n_abnormal() == 6


def test_the_rule_alone_oscillates_and_leaves_one_abnormal_edge(stored):
    algorithm = _permuted(stored, None)
    _rule_only_surgery(algorithm)
    assert algorithm.n_abnormal() == 1


def test_the_guarded_surgery_leaves_none(stored):
    algorithm = _permuted(stored, None)
    post_process_mesh_surgery(algorithm)
    assert algorithm.n_abnormal() == 0


def test_the_guarded_surgery_leaves_none_under_every_row_order(stored):
    """The rule alone leaves 1 edge under some of these orders and 0 under others; the guard, 0 under all."""
    rule_only, guarded = [], []
    for permutation in _row_permutations(len(stored["tetrahedra"])):
        algorithm = _permuted(stored, permutation)
        _rule_only_surgery(algorithm)
        rule_only.append(algorithm.n_abnormal())
        algorithm = _permuted(stored, permutation)
        post_process_mesh_surgery(algorithm)
        guarded.append(algorithm.n_abnormal())
    assert guarded == [0] * (N_ROW_PERMUTATIONS + 1)
    assert 1 in rule_only  # the defect is present in this set of orders, so the test can see it


def test_every_applied_relabelling_lowers_or_keeps_the_count_and_never_undoes(stored, monkeypatch):
    algorithm = _permuted(stored, None)
    steps: list[tuple[int, dict[int, int], dict[int, int]]] = []
    select = mesh_surgery._select_relabelling

    def recording(reconstruction_algorithm, tetra_cycle, label_cycle, held):
        selected = select(reconstruction_algorithm, tetra_cycle, label_cycle, held)
        if selected is not None:
            relabelling = selected[2]
            before = {t: int(reconstruction_algorithm._map_node_id_to_label[t]) for t in relabelling}
            neighbourhood = _RingNeighbourhood(
                reconstruction_algorithm._tesselation_graph,
                reconstruction_algorithm._map_node_id_to_label,
                tetra_cycle,
            )
            steps.append((neighbourhood.change_in_abnormal_edges(relabelling), before, dict(relabelling)))
        return selected

    monkeypatch.setattr(mesh_surgery, "_select_relabelling", recording)
    post_process_mesh_surgery(algorithm)
    assert steps
    assert all(change <= 0 for change, _, _ in steps)
    held: dict[int, set[int]] = {}
    for change, before, after in steps:
        for tetrahedron, label in after.items():
            if change == 0:
                assert label not in held.get(tetrahedron, set())
            held.setdefault(tetrahedron, set()).add(before[tetrahedron])
    assert sum(change for change, _, _ in steps) == -6


def test_the_local_count_is_the_change_of_the_whole_mesh(stored):
    algorithm = _permuted(stored, None)
    graph = algorithm._tesselation_graph
    whole_before = algorithm.n_abnormal()
    checked = 0
    for edge in _find_abnormal_non_manifold_edges(*algorithm.last_constructed_mesh):
        cycle = graph.find_tetra_cycle_around_edge(edge)
        labels = list(algorithm._map_node_id_to_label[cycle])
        neighbourhood = _RingNeighbourhood(graph, algorithm._map_node_id_to_label, cycle)
        for relabelling in _alternative_relabellings(cycle, labels):
            trial = algorithm._map_node_id_to_label.copy()
            for tetrahedron, label in relabelling.items():
                trial[tetrahedron] = label
            by_label = {int(label): [int(i) for i in np.flatnonzero(trial == label)] for label in np.unique(trial)}
            whole_after = len(_find_abnormal_non_manifold_edges(*labeled_mesh_from_labeled_graph(graph, by_label)))
            assert neighbourhood.change_in_abnormal_edges(relabelling) == whole_after - whole_before
            checked += 1
    assert checked == 39  # every alternative around the 6 edges of the fixture
