"""``DCEL_Data.compute_curvatures`` is the length-weighted Menger curvature of each interface's interior vertices.

``compute_curvatures`` reads each interface along the half-edges of its first material's faces, from junction to
junction; the same sum read straight from the mesh's edges (a vertex is interior when exactly two edges meet there)
gives the same number. Until 0.8.0 the two disagreed on every closed loop without junction (a cell enclosed by the
medium, a pocket of medium inside a cell): the walk left out the loop's first and last vertices (228 of 228 such
interfaces on the shipped masks and golden cases, median relative error 3.5 %, largest 20 %). Checked on every
golden case and every shipped mask, with the labels as they are (cells enclosed by the medium) and grown by 6 px,
on the default route (junctions relocated) and the plain one.
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


def _menger(p, q, r):
    twice_area = abs(p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1]))
    return 2.0 * twice_area / max(np.linalg.norm(q - p) * np.linalg.norm(r - q) * np.linalg.norm(r - p), 1e-10)


def curvatures_from_edges(vertices, edges):
    """``{(a, b): sum_i c_i (l_i^- + l_i^+) / sum_i (l_i^- + l_i^+)}`` over the vertices where exactly two edges meet."""
    degree = defaultdict(int)
    neighbours = defaultdict(lambda: defaultdict(list))
    for v1, v2, a, b in np.asarray(edges).tolist():
        pair = (min(a, b), max(a, b))
        degree[v1] += 1
        degree[v2] += 1
        neighbours[pair][v1].append(v2)
        neighbours[pair][v2].append(v1)
    out = {}
    for pair, around in neighbours.items():
        num = den = 0.0
        for v, (u, w) in ((v, nb) for v, nb in around.items() if degree[v] == 2 and len(nb) == 2):
            p, q, r = vertices[u], vertices[v], vertices[w]
            weight = np.linalg.norm(p - q) + np.linalg.norm(q - r)
            num += _menger(p, q, r) * weight
            den += weight
        out[pair] = num / max(den, 1e-10)
    return out


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(("name", "labels", "min_dist", "expansion"), CASES, ids=[c[0] for c in CASES])
def test_compute_curvatures_is_the_curvature_read_from_the_edges(name, labels, min_dist, expansion, route):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        mesh = geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion,
                                          **ROUTES[route]).return_dcel()
        curvatures = {(int(a), int(b)): float(value) for (a, b), value in mesh.compute_curvatures().items()}
    expected = curvatures_from_edges(np.asarray(mesh.v, float), mesh.e)
    assert curvatures.keys() == expected.keys()
    for pair, value in expected.items():
        assert curvatures[pair] == pytest.approx(value, rel=1e-12, abs=1e-15), pair


def test_a_cell_enclosed_by_the_medium_has_the_curvature_of_its_whole_loop():
    """A closed loop without junction reads every vertex: a disc of radius 15 px in the medium."""
    from dw2d import geometry_reconstruction_2d

    rows, cols = np.mgrid[:60, :60]
    labels = (((rows - 29.5) ** 2 + (cols - 29.5) ** 2) < 15**2).astype(np.int64)
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = geometry_reconstruction_2d(labels, min_dist=3).return_dcel()
        curvatures = mesh.compute_curvatures()
    expected = curvatures_from_edges(np.asarray(mesh.v, float), mesh.e)
    assert set(expected) == {(0, 1)}
    assert float(curvatures[(0, 1)]) == pytest.approx(expected[(0, 1)], rel=1e-12)
    assert 1 / 17 < expected[(0, 1)] < 1 / 13
