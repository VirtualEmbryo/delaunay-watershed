"""Golden master of the legacy route (``deterministic=False``): dw2d 0.1.5 run on SciPy 1.13 with interp2d.

Since 0.8.0 the default route is the deterministic one, whose golden master is
``test_deterministic_route.py``; these tests pin the legacy route, kept as an option.

Pins the replacement of ``scipy.interpolate.interp2d`` (removed in SciPy 1.14) and the
vectorised edge scores: mesh vertices, mesh edges, watershed seeds and edge scores must be
bitwise identical to the legacy outputs (the edges after the two orientation changes of 0.8.0, which reverse the
order of the two vertices within edges and nothing else: ``legacy_edges_in_the_orientation_of_0_8_0``), and the DCEL
angles and cell areas equal to 1e-12 relative (the
last bits of ``arccos`` differ between NumPy versions); the interface lengths equal the sum of the legacy mesh's
edge lengths (``QUANTITIES_OF_THE_LEGACY_RECORD`` says why). A passing golden master is a
regression guard, not evidence of scientific correctness.

What is bitwise on every machine, and what is not (as measured when dw2d 0.8.0 was prepared):

* Everything but the triangulation is computed the same on every machine (NumPy, scikit-image's
  stable peak sort, and the bilinear kernel in NumPy ufuncs). Given the golden triangulation,
  the reconstruction is therefore bitwise the golden one everywhere:
  ``test_reconstruction_on_the_golden_triangulation_is_the_legacy_one``.
* Qhull's triangulation is not: the points are integer pixels, often cocircular, and Qhull picks
  the triangulation of each cocircular polygon by floating-point rounding. On macOS arm64 it
  differs from the golden (x86-64) one in 13 of the 18 cases, always only inside cocircular
  polygons, which ``test_qhull_differs_from_the_golden_triangulation_only_inside_cocircular_polygons``
  checks in exact integer arithmetic on every machine.
* The end-to-end tests compare bitwise wherever Qhull returns the golden triangulation, and are
  skipped, with the number of polygons triangulated differently, where it does not.
"""
import contextlib
import hashlib
import io
import types

import numpy as np
import pytest
from .conftest import GOLDEN, golden_cases
from .exact_delaunay_checks import canonical, differences_inside_cocircular_polygons

CASES = golden_cases()
IDS = [c[0] for c in CASES]


@pytest.fixture(scope="session")
def golden_triangulations():
    return np.load(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_triangulations.npz")


def stage_hash(a):
    """The stage hash of the golden record: SHA-256 of dtype, shape and bytes."""
    a = np.ascontiguousarray(np.asarray(a))
    m = hashlib.sha256()
    m.update(str(a.dtype).encode())
    m.update(str(a.shape).encode())
    m.update(a.tobytes())
    return m.hexdigest()


def reconstruct(labels, min_dist, expansion):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        return geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion, deterministic=False,
                                          relocate_junctions=False, repair_pinch_vertices=False,
                                          relabel_near_tie_slivers=False)


def skip_unless_qhull_returns_the_golden_triangulation(dw, golden_triangles):
    if not np.array_equal(canonical(dw.tri.simplices), golden_triangles):
        n = differences_inside_cocircular_polygons(dw.tri.points, dw.tri.simplices, golden_triangles)
        pytest.skip(f"Qhull triangulates {n} cocircular polygon(s) differently from the golden on this machine; "
                    "the reconstruction is checked on the golden triangulation instead")


def legacy_edges_with_the_border_orientation_fixed(verts, legacy_edges):
    """dw2d 0.1.5's edges with its border edges reversed where they break the orientation contract.

    Until 0.8.0 an image-border edge took the triangles of the previous interior edge (a stale loop variable), so
    its two vertices came in an arbitrary order; since 0.8.0 the medium is on its left
    (``test_edge_orientation_contract.py``). This is the only change to the legacy edges: rows, materials and
    every other edge are compared bitwise.
    """
    v, e = np.asarray(verts, float), np.asarray(legacy_edges).copy()
    lo, hi = v.min(axis=0), v.max(axis=0)
    on_border = (v[:, 0] == lo[0]) | (v[:, 0] == hi[0]) | (v[:, 1] == lo[1]) | (v[:, 1] == hi[1])
    for k, (a, b, m1, _) in enumerate(e.tolist()):
        if not (on_border[a] and on_border[b] and m1 == 0):
            continue
        d = v[b] - v[a]
        inside = 0.5 * (v[a] + v[b]) - 1e-3 * np.array([d[1], -d[0]])  # the right of a -> b in (x, y) = (col, row)
        if not (lo[0] < inside[0] < hi[0] and lo[1] < inside[1] < hi[1]):
            e[k, :2] = e[k, [1, 0]]
    return e


def legacy_edges_in_the_orientation_of_0_8_0(verts, legacy_edges):
    """dw2d 0.1.5's edges as dw2d 0.8.0 stores them: border edges fixed, then every edge reversed.

    Since 0.8.0 each edge is the positively oriented boundary of its first material in the index frame (row, col),
    dw3d's rule; dw2d 0.1.5 (and 0.8.0's development versions) stored every edge the other way round. Only the order of
    the two vertices within each row changes: rows, materials and vertices are compared bitwise.
    """
    return legacy_edges_with_the_border_orientation_fixed(verts, legacy_edges)[:, [1, 0, 2, 3]]


def assert_bitwise_the_legacy_outputs(dw, name, arrays):
    verts, edges = dw.return_mesh()
    for key, value in (("verts", verts), ("edges", edges), ("seeds_coords", dw.seeds_coords),
                       ("seeds_indices", dw.seeds_indices), ("edge_scores", dw.Delaunay_Graph.Scores)):
        expected = arrays[f"{name}__{key}"]
        if key == "edges":
            expected = legacy_edges_in_the_orientation_of_0_8_0(arrays[f"{name}__verts"], expected)
        assert np.asarray(value).dtype == expected.dtype, key
        assert np.array_equal(np.asarray(value), expected), key


QUANTITIES = ("angles", "lengths", "curvatures", "cell_areas")
#: Compared with dw2d 0.1.5's record. Its lengths and curvatures were read along half-edge loops that dw2d 0.1.5
#: reversed without swapping their endpoints, which cut most interfaces one edge off their junctions and dropped the
#: last edge of a closed loop (fixed in 0.8.0, ``reverse_closed_line``): the legacy route's lengths are compared with
#: the sum of the legacy mesh's edge lengths instead, and its curvatures with nothing (the default route's golden
#: master pins them). The angles and the cell areas did not depend on it and are compared with the record.
QUANTITIES_OF_THE_LEGACY_RECORD = ("angles", "cell_areas")


def _dcel_quantities(mesh):
    out = {}
    for quantity, compute in zip(QUANTITIES, (mesh.compute_angles, mesh.compute_lengths, mesh.compute_curvatures,
                                              mesh.compute_area_faces)):
        out[quantity] = {"|".join(str(int(x)) for x in (k if isinstance(k, tuple) else (k,))): float(v)
                         for k, v in compute().items()}
    return out


def assert_dcel_quantities_are_the_legacy_ones(dw, name, record, arrays):
    """The DCEL quantities of the golden record; where the legacy mesh has border edges whose orientation 0.8.0
    corrects (``legacy_edges_with_the_border_orientation_fixed``), those of the legacy mesh with that correction."""
    from dw2d.Dcel import DCEL_Data

    legacy_v, legacy_e = arrays[f"{name}__verts"], arrays[f"{name}__edges"]
    fixed_e = legacy_edges_with_the_border_orientation_fixed(legacy_v, legacy_e)
    if np.array_equal(fixed_e, legacy_e):
        expected = record["cases"][name]
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            expected = _dcel_quantities(DCEL_Data(legacy_v, fixed_e[:, [1, 0, 2, 3]],
                                                  allow_several_loops=dw.surgery_report is not None))
    with contextlib.redirect_stdout(io.StringIO()):
        got_all = _dcel_quantities(dw.return_dcel())
    for quantity in QUANTITIES_OF_THE_LEGACY_RECORD:
        got = got_all[quantity]
        want = {k: float(v) for k, v in expected[quantity].items()}
        assert got.keys() == want.keys(), quantity
        for k, value in want.items():
            assert got[k] == pytest.approx(value, rel=1e-12, abs=1e-12), (quantity, k)
    summed = {}
    for a, b, m, n in np.asarray(legacy_e).tolist():
        key = f"{min(m, n)}|{max(m, n)}"
        summed[key] = summed.get(key, 0.0) + float(np.linalg.norm(np.asarray(legacy_v[a], float) - legacy_v[b]))
    assert got_all["lengths"].keys() == summed.keys() == set(record["cases"][name]["lengths"])
    for k, value in summed.items():
        assert got_all["lengths"][k] == pytest.approx(value, rel=1e-12, abs=1e-12), ("lengths", k)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_reconstruction_is_bitwise_the_legacy_one(golden, golden_triangulations, name, labels, min_dist, expansion):
    arrays, _ = golden
    dw = reconstruct(labels, min_dist, expansion)
    skip_unless_qhull_returns_the_golden_triangulation(dw, golden_triangulations[name])
    assert_bitwise_the_legacy_outputs(dw, name, arrays)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_dcel_quantities_match_the_legacy_ones(golden, golden_triangulations, name, labels, min_dist, expansion):
    _, record = golden
    dw = reconstruct(labels, min_dist, expansion)
    skip_unless_qhull_returns_the_golden_triangulation(dw, golden_triangulations[name])
    assert_dcel_quantities_are_the_legacy_ones(dw, name, record, golden[0])


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_reconstruction_on_the_golden_triangulation_is_the_legacy_one(golden, golden_triangulations, monkeypatch,
                                                                      name, labels, min_dist, expansion):
    """Bitwise on every machine: Qhull's points are kept, its triangles replaced by the golden ones."""
    import dw2d.Geometric_utilities as geometric_utilities

    arrays, record = golden
    qhull = geometric_utilities.Delaunay

    def golden_triangulation(points):
        tri = qhull(points)
        return types.SimpleNamespace(points=tri.points, simplices=golden_triangulations[name])

    monkeypatch.setattr(geometric_utilities, "Delaunay", golden_triangulation)
    dw = reconstruct(labels, min_dist, expansion)
    stages = record["cases"][name]["stages"]
    assert stage_hash(dw.tri.points) == stages["tesselation_points"]
    assert stage_hash(dw.Delaunay_Graph.Edges) == stages["graph_edges"], "the stored triangulation is not the legacy one"
    assert_bitwise_the_legacy_outputs(dw, name, arrays)
    assert_dcel_quantities_are_the_legacy_ones(dw, name, record, golden[0])


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_qhull_differs_from_the_golden_triangulation_only_inside_cocircular_polygons(golden_triangulations, name, labels,
                                                                                     min_dist, expansion):
    from skimage.segmentation import expand_labels

    from dw2d.Geometric_utilities import build_triangulation

    _, _, tri, _ = build_triangulation(expand_labels(labels, expansion), min_distance=min_dist)
    differences_inside_cocircular_polygons(tri.points, tri.simplices, golden_triangulations[name])
