"""The default options together, and the opt-in ones on top of them.

The default route is ``deterministic=True, repair_pinch_vertices=True``; the junction relocation and the
locus projection are opt-in. Pinned:

* the repair is on by default, and on every golden case it is a bitwise no-op on the default route and on
  the legacy route (given the golden triangulation, so on every machine);
* on the shipped masks where the unrepaired labelling cannot build the half-edge structure, the default
  builds it, with positive cell areas;
* the relocation and the projection run on top of a repaired mesh (several boundary loops for a material),
  keep its connectivity, never add an edge crossing, and still give a half-edge structure.
"""
import contextlib
import io
import types

import numpy as np
import pytest
import skimage.io as skio
from .conftest import GOLDEN, MASKS, golden_cases

from dw2d import geometry_reconstruction_2d
from dw2d.junction_relocation import whole_mesh_crossings

CASES = golden_cases()
IDS = [c[0] for c in CASES]


@pytest.fixture(scope="session")
def golden_triangulations():
    return np.load(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_triangulations.npz")


def _build(labels, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return geometry_reconstruction_2d(labels, **kwargs)


def _triangle_labels(dw):
    out = np.full(len(dw.tri.simplices), np.iinfo(np.int64).min, dtype=np.int64)
    for label, triangles in dw.Map_end.items():
        out[np.asarray(triangles, dtype=np.int64)] = int(label)
    return out


def _assert_same_reconstruction(a, b):
    assert np.array_equal(_triangle_labels(a), _triangle_labels(b))
    va, ea = a.return_mesh()
    vb, eb = b.return_mesh()
    assert np.array_equal(va, vb) and np.array_equal(ea, eb)


def _mask(name):
    lab = skio.imread(MASKS / name)
    return lab[..., 0] if lab.ndim == 3 else lab


def test_the_repair_is_on_by_default():
    L = np.zeros((20, 20), int); L[5:15, 5:10] = 1; L[5:15, 10:15] = 2
    dw = _build(L, min_dist=3)
    assert dw.deterministic and dw.surgery_report is not None
    assert dw.relocate_junctions and not dw.project_interface_vertices  # relocation on by default since 0.8.0


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_repair_is_a_no_op_on_every_golden_case_of_the_default_route(name, labels, min_dist, expansion):
    on = _build(labels, min_dist=min_dist, expansion_labels=expansion)
    off = _build(labels, min_dist=min_dist, expansion_labels=expansion, repair_pinch_vertices=False)
    assert on.surgery_report.n_triangles_changed == 0
    _assert_same_reconstruction(on, off)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_repair_is_a_no_op_on_every_golden_case_of_the_legacy_route(golden_triangulations, monkeypatch,
                                                                         name, labels, min_dist, expansion):
    import dw2d.Geometric_utilities as geometric_utilities

    qhull = geometric_utilities.Delaunay
    monkeypatch.setattr(geometric_utilities, "Delaunay",
                        lambda points: types.SimpleNamespace(points=qhull(points).points,
                                                             simplices=golden_triangulations[name]))
    on = _build(labels, min_dist=min_dist, expansion_labels=expansion, deterministic=False)
    off = _build(labels, min_dist=min_dist, expansion_labels=expansion, deterministic=False,
                 repair_pinch_vertices=False)
    assert on.surgery_report.n_triangles_changed == 0
    _assert_same_reconstruction(on, off)


@pytest.mark.parametrize("name, min_dist, expansion", [("mask_Cells_2.tif", 3, 0), ("mask_Cells_7.tif", 3, 0),
                                                       ("mask_Cells_8.tif", 5, 6)])
def test_the_default_builds_the_half_edge_structure_the_unrepaired_labelling_cannot(name, min_dist, expansion):
    lab = _mask(name)
    unrepaired = _build(lab, min_dist=min_dist, expansion_labels=expansion, repair_pinch_vertices=False)
    with pytest.raises(ValueError), contextlib.redirect_stdout(io.StringIO()):
        unrepaired.return_dcel()
    default = _build(lab, min_dist=min_dist, expansion_labels=expansion)
    assert default.surgery_report.unresolved_pinch_vertices == []
    with contextlib.redirect_stdout(io.StringIO()):
        areas = default.return_dcel().compute_area_faces()
    assert all(areas[k] > 0 for k in areas if k != 0)


@pytest.mark.parametrize("name, min_dist", [("mask_Cells_7.tif", 3), ("mask_Cells_8.tif", 3), ("mask_Cells_3.tif", 5)])
@pytest.mark.parametrize("project", [False, True])
def test_relocation_and_projection_on_top_of_the_default(name, min_dist, project):
    lab = _mask(name)
    base = _build(lab, min_dist=min_dist)
    moved = _build(lab, min_dist=min_dist, relocate_junctions=True, project_interface_vertices=project)
    assert np.array_equal(_triangle_labels(base), _triangle_labels(moved))
    v0, e0 = base.return_mesh()
    v1, e1 = moved.return_mesh()
    assert np.array_equal(e0, e1) and v0.shape == v1.shape
    assert moved.junction_relocation_report is not None
    assert (moved.locus_projection_report is not None) == project
    assert whole_mesh_crossings(v1, e1) <= whole_mesh_crossings(v0, e0)
    with contextlib.redirect_stdout(io.StringIO()):
        areas = moved.return_dcel().compute_area_faces()
    assert all(areas[k] > 0 for k in areas if k != 0)
