"""Every walk of mesh surgery is strictly descending, and a finished mesh with abnormal edges is never silent.

Surgery may take steps that keep the abnormal-edge count, to move a defect to where it can be removed. Such a walk
is kept only if it ends lower: the steps after the last one that lowered the count are reverted when surgery stops,
and the label maps (list order included) are exactly as they were after that step.

Fixture: the labelled tesselation of the boundary-layer golden configuration (image 3, `min_distance` 3, canonical
order) just before surgery (`tests/golden/surgery_boundary_layer_image_3_min_distance_3_before_surgery.npz`, 183 kB,
written by the research record's `save_the_pre_surgery_tesselation_of_boundary_layer_image_3.py`). On it surgery lowers
the count from 54 to 1, then walks through unchanged-count steps that never reach a repair: they must be reverted.
R161's fixture of case 037 ends with no abnormal edge, so nothing is reverted there.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import dw3d
import dw3d.mesh_surgery as mesh_surgery
from dw3d.mesh_surgery import AbnormalEdgeWarning, post_process_mesh_surgery
from tests.test_surgery_never_recreates_an_abnormal_edge import _LabelledTesselation

GOLDEN = Path(__file__).parent / "golden"
BOUNDARY_LAYER_FIXTURE = GOLDEN / "surgery_boundary_layer_image_3_min_distance_3_before_surgery.npz"
CASE_037_FIXTURE = GOLDEN / "surgery_case_037_resampled_k5_3_before_surgery.npz"


def _load(path: Path) -> _LabelledTesselation:
    with np.load(path) as data:
        return _LabelledTesselation(data["vertices"], data["tetrahedra"], data["labels"])


def _state(algorithm: _LabelledTesselation) -> tuple[np.ndarray, dict[int, list[int]]]:
    return (
        algorithm._map_node_id_to_label.copy(),
        {label: list(nodes) for label, nodes in algorithm._map_label_to_nodes_ids.items()},
    )


def _surgery_with_its_steps(algorithm: _LabelledTesselation, monkeypatch) -> tuple[list[int], list, int]:
    """Run surgery; return the count change of each applied step, the state after each commit, and the reverts."""
    changes: list[int] = []
    committed: list = []
    reverted: list[bool] = []
    select = mesh_surgery._select_relabelling
    commit = mesh_surgery._UncommittedWalk.commit
    revert = mesh_surgery._UncommittedWalk.revert

    def recording_select(reconstruction_algorithm, tetra_cycle, label_cycle, held):
        selected = select(reconstruction_algorithm, tetra_cycle, label_cycle, held)
        if selected is not None:
            changes.append(selected[3])
        return selected

    def recording_commit(walk):
        commit(walk)
        if changes:  # `revert` also commits; only the commits after a step that lowered the count are snapshots
            committed.append(_state(algorithm))

    def recording_revert(walk):
        done = revert(walk)
        reverted.append(done)
        return done

    monkeypatch.setattr(mesh_surgery, "_select_relabelling", recording_select)
    monkeypatch.setattr(mesh_surgery._UncommittedWalk, "commit", recording_commit)
    monkeypatch.setattr(mesh_surgery._UncommittedWalk, "revert", recording_revert)
    post_process_mesh_surgery(algorithm)
    return changes, committed, sum(reverted)


def test_a_walk_that_does_not_end_lower_is_reverted(monkeypatch):
    algorithm = _load(BOUNDARY_LAYER_FIXTURE)
    assert algorithm.n_abnormal() == 54
    changes, committed, n_reverts = _surgery_with_its_steps(algorithm, monkeypatch)
    last_decrease = max(i for i, change in enumerate(changes) if change < 0)
    assert all(change <= 0 for change in changes)
    assert len(changes) > last_decrease + 1  # unchanged-count steps after the last decrease: the fixture exercises it
    assert n_reverts == 1
    assert algorithm.n_abnormal() == 1
    labels, lists = _state(algorithm)
    np.testing.assert_array_equal(labels, committed[-1][0])
    assert lists == committed[-1][1]  # list order too: it sets the order of the extracted triangles


def test_the_mesh_is_the_one_extracted_after_the_last_decrease(monkeypatch):
    algorithm = _load(BOUNDARY_LAYER_FIXTURE)
    _surgery_with_its_steps(algorithm, monkeypatch)
    rebuilt = mesh_surgery.labeled_mesh_from_labeled_graph(
        algorithm._tesselation_graph,
        algorithm._map_label_to_nodes_ids,
    )
    for got, expected in zip(algorithm.last_constructed_mesh, rebuilt, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_nothing_is_reverted_when_surgery_ends_with_no_abnormal_edge(monkeypatch):
    algorithm = _load(CASE_037_FIXTURE)
    changes, _, n_reverts = _surgery_with_its_steps(algorithm, monkeypatch)
    assert algorithm.n_abnormal() == 0
    assert changes[-1] < 0
    assert n_reverts == 0


def test_a_surgery_that_lowers_nothing_returns_the_mesh_it_was_given(monkeypatch):
    """Only unchanged-count steps allowed: every one of them is reverted, and the mesh is the input's."""
    algorithm = _load(BOUNDARY_LAYER_FIXTURE)
    before = _state(algorithm)
    mesh_before = [array.copy() for array in algorithm.last_constructed_mesh]
    select = mesh_surgery._select_relabelling

    def only_neutral(reconstruction_algorithm, tetra_cycle, label_cycle, held):
        selected = select(reconstruction_algorithm, tetra_cycle, label_cycle, held)
        return selected if selected is not None and selected[3] == 0 else None

    monkeypatch.setattr(mesh_surgery, "_select_relabelling", only_neutral)
    post_process_mesh_surgery(algorithm)
    labels, lists = _state(algorithm)
    np.testing.assert_array_equal(labels, before[0])
    assert lists == before[1]
    for got, expected in zip(algorithm.last_constructed_mesh, mesh_before, strict=True):
        np.testing.assert_array_equal(got, expected)


def _two_cubes_touching_along_an_edge() -> np.ndarray:
    """Two cells whose only contact is one edge line: the mesh has an abnormal edge there without surgery."""
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[5:20, 5:20, 5:35] = 1
    mask[20:35, 20:35, 5:35] = 2
    return mask


def test_the_finished_mesh_warns_with_the_count_and_the_configuration():
    algorithm = dw3d.MeshReconstructionAlgorithmFactory.get_default_algorithm(min_distance=3)
    algorithm.perform_mesh_postprocess_surgery = False
    algorithm.relocate_junctions = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algorithm.construct_mesh_from_segmentation_mask(_two_cubes_touching_along_an_edge())
    found = [w for w in caught if issubclass(w.category, AbnormalEdgeWarning)]
    if algorithm._n_abnormal_edges == 0:
        pytest.skip("this mask gave no abnormal edge with surgery off; the warning cannot be exercised on it")
    assert len(found) == 1
    message = str(found[0].message)
    assert f"{algorithm._n_abnormal_edges} abnormal non-manifold edge" in message
    assert "min_distance=3" in message
    assert "mesh surgery off" in message
    assert "canonical tetrahedron order on" in message
    assert "offset exclusion off" in message


def test_a_finished_mesh_without_abnormal_edges_does_not_warn():
    algorithm = dw3d.MeshReconstructionAlgorithmFactory.get_default_algorithm(min_distance=3)
    algorithm.relocate_junctions = False
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    mask[5:15, 5:25, 5:25] = 1
    mask[15:25, 5:25, 5:25] = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algorithm.construct_mesh_from_segmentation_mask(mask)
    assert algorithm._n_abnormal_edges == 0
    assert not [w for w in caught if issubclass(w.category, AbnormalEdgeWarning)]


def test_the_warning_is_exported():
    assert dw3d.AbnormalEdgeWarning is AbnormalEdgeWarning
    assert issubclass(AbnormalEdgeWarning, UserWarning)
