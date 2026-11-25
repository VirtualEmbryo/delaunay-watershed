"""Tests of the junction relocation (``dw2d.junction_relocation``, opt-in when written, on by default since 0.8.0).

Pinned: (1) the relocation is on by default since 0.8.0 (the call without the option gives the arrays
of ``relocate_junctions=True``; the legacy golden master pins ``relocate_junctions=False`` against dw2d
0.1.5); (2) the mask's junction is found on the dual lattice, and a mesh junction is moved onto it;
(3) the guards refuse a move that makes edges cross and leave the vertex where it was; (4) dw2d's
oriented edges give every cell a signed area of one sign (what the area guard relies on); (5) on
the shipped example masks the relocation never raises the whole-mesh crossing count and the
half-edge structure still builds; (6) connectivity is frozen: the edge array is unchanged.
"""
import contextlib
import io

import numpy as np
import pytest
import skimage.io as skio

from .conftest import DATA, MASKS
from dw2d import geometry_reconstruction_2d
from dw2d import junction_relocation as JR


def _quiet(labels, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return geometry_reconstruction_2d(labels, **kw)


def _two_squares():
    L = np.zeros((40, 50), int)
    L[10:30, 8:25] = 1
    L[10:30, 25:42] = 2
    return L


def test_relocation_is_on_by_default():
    L = _two_squares()
    with contextlib.redirect_stdout(io.StringIO()):
        v0, e0 = _quiet(L, min_dist=3).return_mesh()
        v1, e1 = _quiet(L, min_dist=3, relocate_junctions=True).return_mesh()
        v2, _ = _quiet(L, min_dist=3, relocate_junctions=False).return_mesh()
    assert np.array_equal(v0, v1) and np.array_equal(e0, e1)
    assert not np.array_equal(v0, v2)


def test_mask_junction_is_the_dual_corner():
    T = JR.mask_junctions(_two_squares())
    tri = sorted(tuple(t["position"]) for t in T if t["materials"] == (0, 1, 2))
    assert tri == [(9.5, 24.5), (29.5, 24.5)]


def test_junction_is_moved_onto_the_dual_corner_and_edges_are_frozen():
    L = _two_squares()
    dw = _quiet(L, min_dist=3, relocate_junctions=True)
    with contextlib.redirect_stdout(io.StringIO()):
        v, e = dw.return_mesh()
        v0, e0 = _quiet(L, min_dist=3, relocate_junctions=False).return_mesh()
    assert np.array_equal(e, e0)
    rep = dw.junction_relocation_report
    assert rep["moved"] >= 1 and rep["crossings_after"] <= rep["crossings_before"]
    J = JR.mesh_junction_vertices(e)
    targets = {(9.5, 24.5), (29.5, 24.5)}
    moved = [tuple(np.round(v[k], 9)) for k, tri in J.items() if tri == (0, 1, 2) and not np.allclose(v[k], v0[k])]
    assert moved and all(p in targets for p in moved)


def test_crossing_move_is_refused():
    L = _two_squares()
    with contextlib.redirect_stdout(io.StringIO()):
        v, e = _quiet(L, min_dist=3, relocate_junctions=False).return_mesh()
    mover = JR.GuardedVertexMover(v, e)
    j = sorted(JR.mesh_junction_vertices(e))[0]
    far = mover.v[j] + np.array([0.0, 30.0]) * (1 if mover.v[j][1] < 25 else -1)
    assert mover.try_move(j, far) != "" and np.allclose(mover.v[j], v[j])
    assert mover.try_move(j, mover.v[j] + np.array([0.3, 0.2])) == ""
    assert JR.whole_mesh_crossings(mover.v, e) == 0


def test_signed_areas_have_one_sign_per_cell():
    L = _two_squares()
    with contextlib.redirect_stdout(io.StringIO()):
        v, e = _quiet(L, min_dist=3, relocate_junctions=False).return_mesh()
    A = JR.signed_areas(v, e)
    assert A[1] * A[2] > 0 and abs(abs(A[1]) - (L == 1).sum()) < 0.1 * (L == 1).sum()


@pytest.mark.parametrize("name, path, expansion", [("c_elegans", DATA / "c_elegans_label.png", 0),
                                                   ("mask_Cells_1", MASKS / "mask_Cells_1.tif", 0),
                                                   ("mask_Cells_3", MASKS / "mask_Cells_3.tif", 0),
                                                   ("mask_Cells_2_expansion_6", MASKS / "mask_Cells_2.tif", 6)])
@pytest.mark.parametrize("md", [3, 5])
def test_shipped_masks_never_gain_a_crossing(name, path, expansion, md):
    lab = skio.imread(path)
    if lab.ndim == 3:
        lab = lab[..., 0]
    dw = _quiet(lab, min_dist=md, expansion_labels=expansion, relocate_junctions=True)
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = dw.return_dcel()
    rep = dw.junction_relocation_report
    assert rep["crossings_after"] <= rep["crossings_before"]
    assert len(mesh.faces) == len(np.unique(dw.return_mesh()[1][:, 2:4]))
