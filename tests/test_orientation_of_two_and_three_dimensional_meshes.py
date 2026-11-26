"""One orientation rule for dw3d's triangles and dw2d's edges, read in index order.

The rule: `labels[n] = (a, b)`, `a < b`, and element `n` is the positively oriented boundary of `a` in the mask's
index frame (`(z, y, x)` for a stack, `(row, column)` for an image), taken right-handed. For a triangle `(p0, p1, p2)`
its right-hand normal `(p1 - p0) x (p2 - p0)` points out of `a` and into `b`; for an edge `p0 -> p1` with tangent
`t = (dr, dc)`, `a` lies on its left and the normal `(dc, -dr)` (the tangent turned clockwise) points out of `a` and
into `b`.

The test that decides it: one cell (label 1) alone in the medium (label 0), so every element carries `(a, b) = (0, 1)`
and is the positive boundary of the medium. By the divergence theorem its signed volume
`V = (1/6) sum det(p0, p1, p2)` and its signed area `A = (1/2) sum (r0 c1 - c0 r1)` are minus the cell's volume and
area: both negative. Measured here on a ball of radius 9 voxels and a disk of radius 20 pixels. dw2d stored each edge
the other way round until 0.8.0 (the disk's signed area was positive), the opposite rule in index order.
"""

import numpy as np

import dw2d
import dw2d.io
import dw3d


def _ball(n: int = 28, radius: float = 9.0) -> np.ndarray:
    z, y, x = np.indices((n, n, n))
    c = (n - 1) / 2
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2).astype(np.int64)


def _disk(n: int = 60, radius: float = 20.0) -> np.ndarray:
    r, c = np.indices((n, n))
    centre = (n - 1) / 2
    return ((r - centre) ** 2 + (c - centre) ** 2 <= radius**2).astype(np.int64)


def test_dw3d_s_normals_point_out_of_the_first_label_so_a_cell_in_the_medium_has_a_negative_signed_volume():
    mask = _ball()
    points, triangles, labels = dw3d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask)
    assert np.all(labels == [0, 1])
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]])
    centre = (np.array(mask.shape) - 1) / 2
    assert np.all(np.einsum("ij,ij->i", normals, centre - points[triangles].mean(axis=1)) > 0)
    volume = np.einsum("ij,ij->i", points[triangles[:, 0]],
                       np.cross(points[triangles[:, 1]], points[triangles[:, 2]])).sum() / 6
    assert volume < 0
    assert 0.6 * mask.sum() < -volume < 1.2 * mask.sum()


def test_dw2d_s_normals_point_out_of_the_first_label_so_a_cell_in_the_medium_has_a_negative_signed_area():
    mask = _disk()
    points, segments, labels = dw2d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask)
    assert np.all(labels == [0, 1])
    start, end = points[segments[:, 0]], points[segments[:, 1]]
    area = 0.5 * np.sum(start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0])  # (row, column) frame
    assert area < 0
    assert 0.9 * mask.sum() < -area < 1.1 * mask.sum()
    # The normal (t_column, -t_row), the tangent turned clockwise in (row, column), points out of the first label
    # (the medium) into the cell on every edge, as the right-hand normal of a dw3d triangle does on the ball.
    tangents = end - start
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    centre = (np.array(mask.shape) - 1) / 2
    assert np.all(np.einsum("ij,ij->i", normals, centre - (start + end) / 2) > 0)


def test_one_rule_through_every_route_that_returns_a_two_dimensional_mesh(tmp_path):
    """dw2d's algorithm, its half-edge structure, its compressed segmentation, its `.rec` file and dw3d's dispatch."""
    mask = _disk()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    reference = np.hstack((segments, labels))

    def signed_area(p, s):
        return 0.5 * np.sum(p[s[:, 0], 0] * p[s[:, 1], 1] - p[s[:, 0], 1] * p[s[:, 1], 0])

    assert np.array_equal(algorithm.last_constructed_dcel.e, reference)
    compressed = algorithm.last_compressed_segmentation
    assert np.array_equal(np.hstack((compressed["segments"], compressed["labels"])), reference)
    path = tmp_path / "disk.rec"
    algorithm.save_to_rec_mesh(path)
    p, s, lab = dw2d.io.load_rec(path)
    assert np.array_equal(np.hstack((s, lab)), reference)
    assert dw2d.io.load_rec_appendix(path)["orientation"] == dw2d.io.ORIENTATION_RECORDED == "boundary_of_first_label"
    p3, s3, l3 = dw3d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask)
    assert np.array_equal(np.hstack((s3, l3)), reference)
    for pts, seg in ((points, segments), (p, s), (p3, s3)):
        assert signed_area(pts, seg) < 0
