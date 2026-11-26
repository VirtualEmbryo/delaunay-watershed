"""The orientation contract of dw2d's edges, which foambryo's 2D pressures rely on.

Every edge ``[v1, v2, m1, m2]`` returned by ``return_mesh`` (and stored in ``DCEL_Data.e``) has ``m1 < m2``, and is
the positively oriented boundary of ``m1`` in the index frame ``(row, col)``: material ``m1`` lies on the left of
``v1 -> v2`` in ``(row, col)``, the rule of dw3d's triangles (since 0.8.0; before, every edge was stored the other way
round). Checked on every edge, from the mesh alone, in ``(row, col)``: a point just to the right of the edge's midpoint
must lie inside the polygon of ``m2`` (even-odd rule over ``m2``'s edges) and a point just to its left must not; where
``m1`` is a cell, the point to the left must lie inside ``m1``'s polygon. Cases: every golden case, and every shipped
mask (several cells cut by the image border, whose border edges were oriented arbitrarily before 0.8.0), with
relocation off, on, and on with the projection.
"""
import contextlib
import io

import numpy as np
import pytest
from .conftest import MASKS, golden_cases

ROUTES = {"plain": {"relocate_junctions": False}, "relocated": {"relocate_junctions": True},
          "relocated_and_projected": {"relocate_junctions": True, "project_interface_vertices": True}}


def _inside(points, edges_of_material, p):
    a, b = points[edges_of_material[:, 0]], points[edges_of_material[:, 1]]
    crossing = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
    x = a[crossing, 0] + (p[1] - a[crossing, 1]) * (b[crossing, 0] - a[crossing, 0]) / (b[crossing, 1] - a[crossing, 1])
    return int(np.sum(x > p[0])) % 2 == 1


def orientation_violations(vertices, edges):
    """Indices of the edges that break the contract, from the mesh alone (in the index frame (row, col))."""
    xy = np.asarray(vertices, float)  # (row, col): "left" below is the left of v1 -> v2 in this frame
    edges = np.asarray(edges, np.int64)
    of = {m: edges[(edges[:, 2] == m) | (edges[:, 3] == m)][:, :2] for m in np.unique(edges[:, 2:])}
    bad = []
    for k, (v1, v2, m1, m2) in enumerate(edges.tolist()):
        d = xy[v2] - xy[v1]
        length = float(np.hypot(*d))
        left = np.array([-d[1], d[0]]) / length
        mid = 0.5 * (xy[v1] + xy[v2])
        p_left, p_right = mid + 1e-4 * length * left, mid - 1e-4 * length * left
        ok = m1 < m2 and _inside(xy, of[m2], p_right) and not _inside(xy, of[m2], p_left)
        if ok and m1 != 0:
            ok = _inside(xy, of[m1], p_left) and not _inside(xy, of[m1], p_right)
        if not ok:
            bad.append(k)
    return bad


def _cases():
    import skimage.io as skio

    cases = [(name, lab, md, exp) for name, lab, md, exp in golden_cases()]
    for path in sorted(MASKS.glob("mask_Cells_*.tif")):
        lab = skio.imread(path)
        cases.append((f"{path.stem}__md3", lab[..., 0] if lab.ndim == 3 else lab, 3, 0))
    return cases


CASES = _cases()


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(("name", "labels", "min_dist", "expansion"), CASES, ids=[c[0] for c in CASES])
def test_m1_lies_on_the_left_of_every_edge(name, labels, min_dist, expansion, route):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion, **ROUTES[route])
        vertices, edges = dw.return_mesh()
    assert orientation_violations(vertices, edges) == []


def test_border_edges_follow_the_contract():
    """A mask whose cells are cut by the image border: its border edges have the medium (outside) on their left."""
    from dw2d import geometry_reconstruction_2d

    labels = np.ones((40, 60), np.int64)
    labels[:, 30:] = 2
    labels[10:30, 10:50][(np.add.outer(np.arange(20) - 9.5, np.zeros(40)) ** 2 + np.add.outer(np.zeros(20), np.arange(40) - 19.5) ** 2) < 64] = 3
    with contextlib.redirect_stdout(io.StringIO()):
        vertices, edges = geometry_reconstruction_2d(labels, min_dist=3).return_mesh()
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    on_border = (vertices[:, 0] == lo[0]) | (vertices[:, 0] == hi[0]) | (vertices[:, 1] == lo[1]) | (vertices[:, 1] == hi[1])
    border = [k for k, (v1, v2, _, _) in enumerate(edges.tolist()) if on_border[v1] and on_border[v2]]
    assert len(border) > 0 and all(edges[k, 2] == 0 for k in border)
    assert orientation_violations(vertices, edges) == []
