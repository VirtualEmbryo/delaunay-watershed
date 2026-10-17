"""Junction relocation never turns a cell inside out.

The relocation guard tested each incident triangle's normal against its previous normal and its
area. On a real embryo timepoint that let a one-tetrahedron cell turn inside out: one vertex moved
6.7 voxels across the opposite face while every incident triangle kept a normal within 49 degrees
of its previous one, and the cell's volume went from +35 to -5.8 cubic voxels. The surface stayed
closed, oriented, manifold and free of crossings. The guard now also refuses a move that would leave
a cell it touches with a signed volume of zero or less, and nothing else: shrinking is allowed.

Where the clause never fires the relocated mesh must be bitwise the one without it.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

import dw3d.junction_relocation as relocation
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.junction_relocation import JunctionRelocationReport, _apply_moves
from tests.conftest import REPO_ROOT, load_image_or_skip

# Cell 1 is the tetrahedron (0, 1, 2, 3) inside the medium 0: base in z = 0, apex at height 3,
# volume 8. Labels `(0, 1)`: each normal points out of the medium, into the cell.
POINTS = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 1.0, 3.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
LABELS = np.array([[0, 1]] * 4)


def _volume(points: np.ndarray, triangles: np.ndarray = TRIANGLES) -> float:
    corners = points[triangles]
    return float(-np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)


def _move_apex(height: float, labels, triangles: np.ndarray = TRIANGLES):
    report = JunctionRelocationReport()
    moved = _apply_moves(POINTS, triangles, {3: np.array([1.0, 1.0, height])}, {}, report, labels)
    return moved, report


def test_the_cell_reads_its_volume():
    assert _volume(POINTS) == pytest.approx(8.0)


def test_the_triangle_clauses_alone_let_the_cell_turn_inside_out():
    """Apex lowered through the base, to z = -0.1: every normal stays within 90 degrees, no area collapses."""
    moved, report = _move_apex(-0.1, None)
    assert report.n_accepted == 1
    assert report.n_rejected_inversion == 0
    assert _volume(moved) == pytest.approx(-0.8 / 3.0)


def test_a_move_that_turns_the_cell_inside_out_is_refused():
    moved, report = _move_apex(-0.1, LABELS)
    assert np.array_equal(moved, POINTS)  # the vertex stays where it was
    assert report.n_accepted == 0
    assert report.n_rejected_cell_inversion == 1
    assert report.cells_kept_from_inverting == [1]


def test_a_move_to_zero_volume_counts_as_inverted():
    moved, report = _move_apex(0.0, LABELS)
    assert np.array_equal(moved, POINTS)
    assert report.n_rejected_cell_inversion == 1


def test_shrinking_without_inverting_is_allowed():
    """Unlike a volume floor, the clause lets relocation shrink a cell as far as it likes above zero."""
    moved, report = _move_apex(0.3, LABELS)
    assert report.n_accepted == 1
    assert report.n_rejected_cell_inversion == 0
    assert _volume(moved) == pytest.approx(0.8)


def test_a_cell_already_inverted_before_the_pass_is_not_constrained():
    reversed_cell = TRIANGLES[:, [0, 2, 1]]  # cell 1 now reads -8
    _, report = _move_apex(-0.1, LABELS, reversed_cell)
    assert report.n_cells_not_positive_before_relocation == 1
    assert report.n_rejected_cell_inversion == 0


def test_where_the_clause_never_fires_the_mesh_is_bitwise_unchanged(monkeypatch):
    """In-repo image 1 with the default algorithm, against the same run with the clause switched off."""
    mask = load_image_or_skip("1.tif")
    first = get_default_mesh_reconstruction_algorithm()
    points, triangles, labels = first.construct_mesh_from_segmentation_mask(mask)
    assert first.relocation_report.n_rejected_cell_inversion == 0
    monkeypatch.setattr(relocation._CellVolumes, "cells_inverted_by", lambda self, points, faces: [])  # noqa: ARG005
    second = get_default_mesh_reconstruction_algorithm()
    points_off, triangles_off, labels_off = second.construct_mesh_from_segmentation_mask(mask)
    assert np.array_equal(points, points_off)
    assert np.array_equal(triangles, triangles_off)
    assert np.array_equal(labels, labels_off)


def _experimental_mask_or_skip(series: str, filename: str) -> np.ndarray:
    """A mask of `experimental-dataset`: the repository's sibling folder, or `EXPERIMENTAL_DATASET`."""
    candidates = [REPO_ROOT.parent / "experimental-dataset"]
    if os.environ.get("EXPERIMENTAL_DATASET"):
        candidates.append(Path(os.environ["EXPERIMENTAL_DATASET"]))
    for directory in candidates:
        path = directory / "segmentation_masks" / series / filename
        if path.exists():
            return tifffile.imread(path)
    pytest.skip(f"experimental-dataset/segmentation_masks/{series}/{filename} not found")
    return None


def test_the_cell_relocation_turned_inside_out_stays_positive():
    """`st1_mem_crop2` timepoint 244, cell 490: +35 before relocation, -5.8 after it without the clause.

    The frame is one the real-data gates leave out as over-segmented; as a case for the code it is
    still valid, because no mask, however poor, should come out with a cell turned inside out.
    """
    mask = _experimental_mask_or_skip("st1_mem_crop2", "Cam_long_00244.lux_crop.tif")
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    report = algorithm.relocation_report
    assert report.n_rejected_cell_inversion >= 1
    assert 490 in report.cells_kept_from_inverting
    corners = points[np.asarray(triangles, dtype=np.int64)]
    det = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))
    labels = np.asarray(labels, dtype=np.int64)
    volume = (det[labels[:, 0] == 490].sum() - det[labels[:, 1] == 490].sum()) / 6.0
    assert volume > 0
    assert all(cell["reason"] != "non-positive" for cell in algorithm.cell_volume_report.cells)
