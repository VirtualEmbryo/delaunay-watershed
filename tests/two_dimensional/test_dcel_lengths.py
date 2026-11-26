"""``DCEL_Data.compute_lengths`` is the sum of the lengths of each interface's edges.

``compute_lengths`` reads each interface along the half-edges of its first material's faces, from junction to
junction; the edges of a material pair, read straight from the mesh, give the same number. Until 0.8.0 the two
disagreed: on faces whose loop had been reversed, the walk cut every interface one edge off its junctions (103 of the
146 interfaces of ``mask_Cells_2`` with ``expansion_labels=6`` at ``min_dist=3``), and a closed loop without junction
(a cell enclosed by the medium) lost its last edge. Checked on every golden case and every shipped mask, with the
labels as they are (cells enclosed by the medium) and grown by 6 px (cells touching one another and the image
border), on the default route (junctions relocated) and the plain one.
"""
import contextlib
import io
from collections import defaultdict

import numpy as np
import pytest
from .conftest import MASKS, golden_cases

ROUTES = {"relocated": {"relocate_junctions": True}, "plain": {"relocate_junctions": False}}


def _cases():
    import skimage.io as skio

    cases = [(name, lab, md, exp) for name, lab, md, exp in golden_cases()]
    for path in sorted(MASKS.glob("mask_Cells_*.tif")):
        lab = skio.imread(path)
        lab = lab[..., 0] if lab.ndim == 3 else lab
        for expansion in (0, 6):
            cases.append((f"{path.stem}__expansion{expansion}__md3", lab, 3, expansion))
    return cases


CASES = _cases()


def summed_edge_lengths(vertices, edges):
    out = defaultdict(float)
    for v1, v2, a, b in np.asarray(edges).tolist():
        out[(min(a, b), max(a, b))] += float(np.linalg.norm(np.asarray(vertices[v1], float) - vertices[v2]))
    return dict(out)


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(("name", "labels", "min_dist", "expansion"), CASES, ids=[c[0] for c in CASES])
def test_compute_lengths_is_the_sum_of_the_edge_lengths(name, labels, min_dist, expansion, route):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        mesh = geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion,
                                          **ROUTES[route]).return_dcel()
        lengths = {(int(a), int(b)): float(value) for (a, b), value in mesh.compute_lengths().items()}
    expected = summed_edge_lengths(np.asarray(mesh.v, float), mesh.e)
    assert lengths.keys() == expected.keys()
    for pair, value in expected.items():
        assert lengths[pair] == pytest.approx(value, rel=1e-12, abs=1e-12), pair


def test_a_cell_enclosed_by_the_medium_has_its_whole_perimeter():
    """A closed loop without junction keeps its last edge: a single disc in the medium."""
    from dw2d import geometry_reconstruction_2d

    rows, cols = np.mgrid[:60, :60]
    labels = (((rows - 29.5) ** 2 + (cols - 29.5) ** 2) < 15 ** 2).astype(np.int64)
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = geometry_reconstruction_2d(labels, min_dist=3).return_dcel()
        lengths = mesh.compute_lengths()
    expected = summed_edge_lengths(np.asarray(mesh.v, float), mesh.e)
    assert set(expected) == {(0, 1)}
    assert float(lengths[(0, 1)]) == pytest.approx(expected[(0, 1)], rel=1e-12)
    assert 2 * np.pi * 14 < expected[(0, 1)] < 2 * np.pi * 16 * 1.1
