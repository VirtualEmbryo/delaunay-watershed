"""Opt-in repair of the labelled triangulation (``dw2d.mesh_surgery``), the 2D analogue of dw3d's
abnormal-edge surgery.

Pins: pinch detection on a hand-built fan; fragment absorption; that a seed triangle is never
relabelled; that the candidate selection **maximises** its score agreement (dw3d's surgery once
minimised a "higher is better" score; the same inversion here would pick the worst repair); that a
repaired tissue mask builds a DCEL the legacy code cannot build; that the repair is a bitwise no-op
where nothing needs repairing; and that several boundary loops per material are oriented so cell
areas come out right (holes subtracted).
"""
import contextlib
import io

import numpy as np
import pytest
import skimage.io as skio

from .conftest import MASKS as DATA
from dw2d import DCEL_Data, geometry_reconstruction_2d
from dw2d import mesh_surgery as MS


class HexagonFan:
    """Six triangles around vertex 0 (a hexagon 1..6 with its centre), dw2d ``Delaunay_Graph`` fields.

    Triangle ``p`` is (0, p+1, p+2 mod 6); triangles ``p`` and ``p+1`` share the edge (0, p+2), whose
    score is ``scores[p]``. Hull edges (p+1, p+2) are lone edges of triangle ``p``.
    """

    def __init__(self, scores):
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        self.Vertices = np.vstack([[0.0, 0.0], np.column_stack([np.cos(ang), np.sin(ang)])])
        self.Nodes = np.array([[0, p + 1, (p + 1) % 6 + 1] for p in range(6)])
        self.Edges = np.array([[0, (p + 1) % 6 + 1] for p in range(6)])
        self.Nodes_linked_by_Edges = np.array([[p, (p + 1) % 6] for p in range(6)])
        self.Scores = np.asarray(scores, dtype=float)
        self.Lone_edges = np.array([[p + 1, (p + 1) % 6 + 1] for p in range(6)])
        self.Nodes_linked_by_lone_edges = np.arange(6)


def map_of(labels):
    out = {}
    for t, L in enumerate(labels):
        out.setdefault(L, []).append(t)
    return out


def labels_of(map_end, n):
    lab = np.empty(n, dtype=int)
    for L, ts in map_end.items():
        lab[ts] = L
    return lab


def test_pinch_detection_on_a_hand_built_fan():
    tri = MS._Triangulation(HexagonFan(np.ones(6)))
    assert MS.find_pinch_vertices(tri, np.array([1, 2, 1, 2, 2, 2])) == [0]
    assert MS.find_pinch_vertices(tri, np.array([1, 1, 2, 2, 2, 2])) == []
    # a hull vertex sees the outside as medium: [cell, cell] + outside is one run each
    assert MS.find_pinch_vertices(tri, np.array([1, 1, 1, 1, 1, 1])) == []


def test_a_fragment_is_absorbed_and_the_pinch_disappears():
    g = HexagonFan(np.ones(6))
    lab0 = [1, 2, 1, 2, 2, 2]
    new_map, rep = MS.repair_labelled_triangulation(g, map_of(lab0), seed_nodes=[0, 4])
    lab = labels_of(new_map, 6)
    assert lab[0] == 1 and lab[4] == 2  # seeds kept
    assert MS.find_pinch_vertices(MS._Triangulation(g), lab) == []
    assert len(rep.fragment_moves) == 1 and rep.fragment_moves[0]["label_from"] == 1
    assert rep.unresolved_pinch_vertices == []


def _one_pinch_move(scores, seeds):
    g = HexagonFan(scores)
    tri = MS._Triangulation(g)
    lab = np.array([1, 2, 1, 2, 2, 2])
    move = MS._repair_one_pinch(tri, lab, 0, set(seeds), max_changes=3)
    return move, lab


def test_the_selection_maximises_the_score_agreement():
    """Three single-triangle repairs exist; their agreements are 2 (relabel 1), 1.1 (relabel 0) and
    11 (relabel 2). The argmax is triangle 2; an argmin would pick triangle 0."""
    #        e(0,1) e(1,2) e(2,3) e(3,4) e(4,5) e(5,0)
    scores = [1.0, 1.0, 10.0, 0.5, 0.5, 0.1]
    move, lab = _one_pinch_move(scores, seeds=[])
    assert move["triangles"] == [2] and move["labels_to"] == [2]
    assert move["agreement"] == pytest.approx(11.0)
    assert list(lab) == [1, 2, 2, 2, 2, 2]


def test_a_seed_triangle_is_never_relabelled():
    scores = [1.0, 1.0, 10.0, 0.5, 0.5, 0.1]
    move, lab = _one_pinch_move(scores, seeds=[1, 2])  # the two best candidates are seeds
    assert move["triangles"] == [0]
    assert lab[1] == 2 and lab[2] == 1


def _build(name, md, ex, repair):
    lab = skio.imread(DATA / name)
    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(lab, min_dist=md, expansion_labels=ex, repair_pinch_vertices=repair)
    return dw


def test_a_repaired_tissue_mask_builds_a_dcel_the_legacy_code_cannot_build():
    """mask_Cells_8 at min_dist 5 with expansion 6: one pinch vertex in the legacy labelling."""
    legacy = _build("mask_Cells_8.tif", 5, 6, False)
    with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
        legacy.return_dcel()
    repaired = _build("mask_Cells_8.tif", 5, 6, True)
    assert repaired.surgery_report.unresolved_pinch_vertices == []
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = repaired.return_dcel()
    areas = mesh.compute_area_faces()
    assert all(areas[L] > 0 for L in areas if L != 0)


def test_the_repair_is_a_bitwise_no_op_where_nothing_needs_repairing():
    off = _build("mask_Cells_1.tif", 5, 0, False)
    on = _build("mask_Cells_1.tif", 5, 0, True)
    assert on.surgery_report.n_triangles_changed == 0
    v0, e0 = off.return_mesh()
    v1, e1 = on.return_mesh()
    assert np.array_equal(v0, v1) and np.array_equal(e0, e1)
    with contextlib.redirect_stdout(io.StringIO()):
        a0, a1 = off.return_dcel().compute_angles(), on.return_dcel().compute_angles()
    assert a0 == a1


def _disk(shape, centre, radius):
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    return (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2 <= radius ** 2


def test_several_loops_per_material_are_oriented_so_areas_are_right():
    """Two separate cells (the medium has two loops) and an annular cell (the cell has a hole):
    the legacy DCEL refuses both; with ``allow_several_loops`` every cell area is positive and
    within 15 % of its pixel count, the annulus's hole subtracted."""
    shape = (120, 160)
    lab = np.zeros(shape, dtype=np.int32)
    lab[_disk(shape, (40, 40), 22)] = 1
    lab[_disk(shape, (40, 120), 18)] = 2
    ring = _disk(shape, (90, 80), 25) & ~_disk(shape, (90, 80), 12)
    lab[ring] = 3
    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(lab, min_dist=3, repair_pinch_vertices=True)
        v, e = dw.return_mesh()
        with pytest.raises(AssertionError):
            DCEL_Data(v, e)
        mesh = DCEL_Data(v, e, allow_several_loops=True)
    areas = mesh.compute_area_faces()
    for L in (1, 2, 3):
        px = int((lab == L).sum())
        assert areas[L] > 0
        assert abs(areas[L] - px) / px < 0.15, (L, areas[L], px)
