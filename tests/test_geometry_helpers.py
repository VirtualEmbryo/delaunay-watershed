"""Unit tests for pure-geometry / IO helpers.

These exercise small, hand-picked cases with known answers, independent of any full
mesh reconstruction run (that is covered by test_golden_master.py).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from dw3d.io import load_rec, save_rec
from dw3d.mesh_utilities import filter_unused_points
from dw3d.points_on_edt import _give_corners
from dw3d.tesselation_graph import TesselationGraph, intersect_line_triangle

# ---------------------------------------------------------------------------
# _compute_volumes / _compute_areas (TesselationGraph)
# ---------------------------------------------------------------------------
# These are pure functions of self.vertices / self.tetrahedrons / self.triangle_faces,
# so we call them unbound on a minimal stand-in object rather than constructing a full
# TesselationGraph (which needs an EDT image + score function unrelated to this math).


def test_compute_volumes_unit_tetrahedron():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tetrahedrons = np.array([[0, 1, 2, 3]], dtype=np.int64)
    stub = SimpleNamespace(vertices=vertices, tetrahedrons=tetrahedrons)

    volumes = TesselationGraph._compute_volumes(stub)

    assert volumes == pytest.approx([1 / 6])


def test_compute_volumes_scales_with_cube():
    # A cube of side 2 split by the tetrahedron (0,0,0),(2,0,0),(0,2,0),(0,0,2): volume = 2^3 / 6
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
    tetrahedrons = np.array([[0, 1, 2, 3]], dtype=np.int64)
    stub = SimpleNamespace(vertices=vertices, tetrahedrons=tetrahedrons)

    volumes = TesselationGraph._compute_volumes(stub)

    assert volumes == pytest.approx([8 / 6])


def test_compute_areas_right_triangle():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    triangle_faces = np.array([[0, 1, 2]], dtype=np.uint)
    stub = SimpleNamespace(vertices=vertices, triangle_faces=triangle_faces)

    areas = TesselationGraph._compute_areas(stub)

    assert areas == pytest.approx([0.5])


def test_compute_areas_equilateral_triangle():
    # Side length 2 equilateral triangle: area = sqrt(3)
    vertices = np.array([[0, 0, 0], [2, 0, 0], [1, np.sqrt(3), 0]], dtype=np.float64)
    triangle_faces = np.array([[0, 1, 2]], dtype=np.uint)
    stub = SimpleNamespace(vertices=vertices, triangle_faces=triangle_faces)

    areas = TesselationGraph._compute_areas(stub)

    assert areas == pytest.approx([np.sqrt(3)])


# ---------------------------------------------------------------------------
# filter_unused_points
# ---------------------------------------------------------------------------


def test_filter_unused_points_removes_and_reindexes():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float64)
    triangles = np.array([[0, 2, 4]], dtype=np.ulonglong)  # only points 0, 2, 4 are used

    filtered_points, reindexed_triangles = filter_unused_points(points, triangles)

    assert len(filtered_points) == 3
    np.testing.assert_array_equal(filtered_points, points[[0, 2, 4]])
    np.testing.assert_array_equal(reindexed_triangles, np.array([[0, 1, 2]]))


def test_filter_unused_points_preserves_used_points():
    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.ulonglong)

    filtered_points, reindexed_triangles = filter_unused_points(points, triangles)

    np.testing.assert_array_equal(filtered_points, points)
    np.testing.assert_array_equal(reindexed_triangles, triangles)


# ---------------------------------------------------------------------------
# _give_corners
# ---------------------------------------------------------------------------


def test_give_corners_shape_and_values():
    image = np.zeros((2, 3, 4))

    corners = _give_corners(image)

    assert corners.shape == (8, 3)
    expected = {(0, 0, 0), (0, 0, 3), (0, 2, 0), (0, 2, 3), (1, 0, 0), (1, 0, 3), (1, 2, 0), (1, 2, 3)}
    assert {tuple(int(x) for x in c) for c in corners} == expected


# ---------------------------------------------------------------------------
# save_rec / load_rec round trip
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mesh():
    points = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    labels = np.array([[0, 1], [0, 1], [1, 2], [1, 2]], dtype=np.int64)
    return points, triangles, labels


@pytest.mark.parametrize("binary_mode", [False, True])
def test_rec_round_trip(tmp_path, sample_mesh, binary_mode):
    points, triangles, labels = sample_mesh
    filename = tmp_path / "mesh.rec"

    save_rec(filename, points, triangles, labels, binary_mode=binary_mode)
    loaded_points, loaded_triangles, loaded_labels = load_rec(filename)

    np.testing.assert_allclose(loaded_points, points, rtol=1e-10)
    np.testing.assert_array_equal(loaded_triangles, triangles)
    np.testing.assert_array_equal(loaded_labels, labels)


def test_load_rec_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rec(tmp_path / "does_not_exist.rec")


def test_load_rec_wrong_extension_raises(tmp_path):
    bad_file = tmp_path / "mesh.txt"
    bad_file.write_text("not a rec file")
    with pytest.raises(ValueError, match=r"not a \.rec or \.arec file"):
        load_rec(bad_file)


# ---------------------------------------------------------------------------
# intersect_line_triangle (ported from the module's own inline __main__ self-test)
# ---------------------------------------------------------------------------

_T1 = np.array([0, 0, 0])
_T2 = np.array([1, 0, 0])
_T3 = np.array([0, 1, 0])


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [
        # Exactly over a point: no intersection.
        (np.array([0, 0, 1]), np.array([0, 0, -1]), None),
        # Exactly over a segment of the triangle: no intersection.
        (np.array([0.5, 0, 1]), np.array([0.5, 0, -1]), None),
        # Same-side points: no intersection.
        (np.array([0.25, 0.25, 1]), np.array([0.26, 0.24, 1.1]), None),
        # Coplanar points: no intersection.
        (np.array([0.25, 0.25, 0]), np.array([0.25, 0.25, -1]), None),
        # Inside the triangle: intersection at (0.25, 0.25, 0).
        (np.array([0.25, 0.25, 1]), np.array([0.25, 0.25, -1]), np.array([0.25, 0.25, 0])),
        (np.array([0.26, 0.23, 1]), np.array([0.24, 0.27, -1]), np.array([0.25, 0.25, 0])),
    ],
)
def test_intersect_line_triangle(l1, l2, expected):
    result = intersect_line_triangle(l1, l2, _T1, _T2, _T3)
    if expected is None:
        assert result is None
    else:
        np.testing.assert_array_equal(result, expected)
