"""Golden master of dw2d's default route (``deterministic=True``, the default since 0.8.0).

The default route re-triangulates the cocircular cells by a fixed rule, lists triangles and dual-graph
edges in canonical order and sorts the watershed's edges stably, so it does not depend on the
machine. On every golden case, with no skip:

* the reconstruction is bitwise the golden one (mesh vertices and edges, seeds, edge scores,
  triangles, the label of each triangle), and the DCEL quantities equal the golden ones to 1e-12
  relative (the last bits of ``arccos`` may differ between NumPy versions, as for the legacy route);
* the same outputs come from the x86-64 triangulation stored with the legacy golden master
  as from this machine's Qhull, so the golden outputs do not depend on the machine that recorded them.

The golden outputs were recorded on macOS arm64 by ``golden/record_the_deterministic_route_outputs.py``.
"""
import contextlib
import io
import json
import types

import numpy as np
import pytest
from .conftest import GOLDEN, golden_cases

CASES = golden_cases()
IDS = [c[0] for c in CASES]
ARRAYS = ("verts", "edges", "seeds_coords", "seeds_indices", "edge_scores", "triangles", "triangle_labels")


@pytest.fixture(scope="session")
def golden_route():
    return (np.load(GOLDEN / "dw2d_deterministic_route_outputs.npz"),
            json.load(open(GOLDEN / "dw2d_deterministic_route_outputs.json")))


@pytest.fixture(scope="session")
def stored_x86_triangulations():
    return np.load(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_triangulations.npz")


def label_of_each_triangle(dw):
    out = np.full(len(dw.tri.simplices), np.iinfo(np.int64).min, dtype=np.int64)
    for label, triangles in dw.Map_end.items():
        out[np.asarray(triangles, dtype=np.int64)] = int(label)
    return out


def reconstruct(labels, min_dist, expansion):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        return geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion)


def outputs(dw):
    verts, edges = dw.return_mesh()
    return dict(verts=verts, edges=edges, seeds_coords=dw.seeds_coords, seeds_indices=dw.seeds_indices,
                edge_scores=dw.Delaunay_Graph.Scores, triangles=dw.tri.simplices, triangle_labels=label_of_each_triangle(dw))


def assert_bitwise(got, arrays, name):
    for key in ARRAYS:
        expected = arrays[f"{name}__{key}"]
        assert np.asarray(got[key]).dtype == expected.dtype, key
        assert np.array_equal(np.asarray(got[key]), expected), key


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_default_route_is_bitwise_the_golden_one(golden_route, name, labels, min_dist, expansion):
    arrays, record = golden_route
    dw = reconstruct(labels, min_dist, expansion)
    assert dw.deterministic
    assert_bitwise(outputs(dw), arrays, name)
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = dw.return_dcel()
        for quantity, compute in (("angles", mesh.compute_angles), ("lengths", mesh.compute_lengths),
                                  ("curvatures", mesh.compute_curvatures), ("cell_areas", mesh.compute_area_faces)):
            got = {"|".join(str(int(x)) for x in (k if isinstance(k, tuple) else (k,))): float(v) for k, v in compute().items()}
            want = record["cases"][name][quantity]
            assert got.keys() == want.keys(), quantity
            for k, value in want.items():
                assert got[k] == pytest.approx(value, rel=1e-12, abs=1e-12), (quantity, k)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_stored_x86_triangulation_gives_the_same_outputs(golden_route, stored_x86_triangulations, monkeypatch,
                                                            name, labels, min_dist, expansion):
    import dw2d.Geometric_utilities as geometric_utilities

    arrays, _ = golden_route
    qhull = geometric_utilities.Delaunay

    def stored(points):
        return types.SimpleNamespace(points=qhull(points).points, simplices=stored_x86_triangulations[name])

    monkeypatch.setattr(geometric_utilities, "Delaunay", stored)
    assert_bitwise(outputs(reconstruct(labels, min_dist, expansion)), arrays, name)


def segments(verts, edges):
    v, e = np.asarray(verts), np.asarray(edges)
    return {(tuple(sorted((tuple(v[a]), tuple(v[b])))), tuple(sorted((int(m), int(n))))) for a, b, m, n in e}


def test_negative_control_the_legacy_route_gives_another_mesh_on_doublet_03(golden_route, stored_x86_triangulations, monkeypatch):
    """The comparison has teeth: on doublet_03 (md 3) the rule takes the other diagonal of a cocircular cell on
    the interface, so the legacy route on the stored triangles gives another set of mesh segments (3 in only one of the two, an
    interface length changes by 0.06 %)."""
    import dw2d.Geometric_utilities as geometric_utilities
    from dw2d import geometry_reconstruction_2d

    arrays, _ = golden_route
    name, labels, md, exp = next(c for c in CASES if c[0] == "doublet_03__md3")
    qhull = geometric_utilities.Delaunay
    monkeypatch.setattr(geometric_utilities, "Delaunay",
                        lambda points: types.SimpleNamespace(points=qhull(points).points, simplices=stored_x86_triangulations[name]))
    with contextlib.redirect_stdout(io.StringIO()):
        legacy = geometry_reconstruction_2d(labels, min_dist=md, expansion_labels=exp, deterministic=False)
    verts, edges = legacy.return_mesh()
    assert len(segments(verts, edges) ^ segments(arrays[f"{name}__verts"], arrays[f"{name}__edges"])) == 3
