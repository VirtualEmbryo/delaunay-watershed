"""The opt-in locus projection: what it moves, what it refuses, and that it is off by default.

Pinned: (1) a vertex of a straight interface between two pixel rows lands on the mid-line between them;
(2) asking for the projection without the junction relocation raises (alone it was measured to make the
angles worse); (3) the default constructor still produces the legacy mesh (also pinned by the golden
master), and the projection never raises the whole-mesh crossing count on the shipped example masks.
"""
import contextlib
import io

import numpy as np
import pytest
import skimage.io as skio

from .conftest import MASKS
from dw2d import geometry_reconstruction_2d
from dw2d.junction_relocation import whole_mesh_crossings
from dw2d.locus_projection import _project_on_edges as project_interface_vertices


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def test_vertex_lands_on_the_mid_line_between_two_label_rows():
    L = np.zeros((10, 10), int); L[:5] = 1; L[5:] = 2
    verts = np.array([[5.0, 2.0], [5.0, 5.0], [5.0, 8.0]]); edges = np.array([[0, 1, 1, 2], [1, 2, 1, 2]])
    v2, rep = project_interface_vertices(verts, edges, L)
    assert rep["moved"] == 1 and abs(v2[1, 0] - 4.5) < 1e-6 and v2[1, 1] == 5.0


def test_projection_requires_the_relocation():
    L = np.zeros((20, 20), int); L[5:15, 5:10] = 1; L[5:15, 10:15] = 2
    with pytest.raises(ValueError):
        _quiet(geometry_reconstruction_2d, L, min_dist=3, relocate_junctions=False, project_interface_vertices=True)


@pytest.mark.parametrize("name", ["mask_Cells_1.tif", "mask_Cells_3.tif"])
def test_off_by_default_and_never_adds_a_crossing(name):
    lab = skio.imread(MASKS / name)
    lab = lab[..., 0] if lab.ndim == 3 else lab
    base = _quiet(geometry_reconstruction_2d, lab, min_dist=3, relocate_junctions=False)
    v0, e0 = base.return_mesh()
    assert base.locus_projection_report is None
    dw = _quiet(geometry_reconstruction_2d, lab, min_dist=3, relocate_junctions=True, project_interface_vertices=True)
    v, e = dw.return_mesh()
    assert np.array_equal(e, e0)
    rep = dw.locus_projection_report
    assert rep["moved"] > 0 and rep["crossings_after"] <= rep["crossings_before"]
    assert whole_mesh_crossings(v, e) <= whole_mesh_crossings(v0, e0)
