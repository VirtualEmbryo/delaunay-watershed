"""The axis order of dw3d's mesh points, and `set_pixel_size` scaling the right columns.

Pins: column `i` of a reconstructed mesh's points is the voxel index along the mask's axis `i` (a mask whose shape
and cell differ along each axis); `set_pixel_size` scales column 0 by the z size and columns 1 and 2 by the xy size,
the `(z, y, x)` order of a microscope stack; the function, the method and `dw3d.set_points_pixel_size` agree
bitwise; and a sphere sampled with a z step three times its pixel size comes back round after scaling, read by the
second moments of the volume its mesh encloses (before 0.7.0 it came back about seven times wider along x than
along z; what remains, about 20 % longer along z, is the coarse z sampling of the native grid).
"""

from __future__ import annotations

import numpy as np

from dw3d import get_default_mesh_reconstruction_algorithm, set_points_pixel_size
from dw3d.mesh_utilities import set_pixel_size


def ellipsoid(shape: tuple[int, int, int], centre: tuple[float, ...], semi_axes: tuple[float, ...]) -> np.ndarray:
    grid = np.indices(shape, dtype=float)
    inside = sum(((g - c) / a) ** 2 for g, c, a in zip(grid, centre, semi_axes, strict=True)) < 1
    return inside.astype(np.uint16)


def cell_vertices(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray, cell: int) -> np.ndarray:
    on_cell = (labels == cell).any(axis=1)
    return points[np.unique(triangles[on_cell])]


def rms_radius_per_axis(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Square root of the enclosed volume's second moment about its centre, per axis (R / sqrt(5) for a sphere)."""
    origin = points[triangles].reshape(-1, 3).mean(axis=0)
    a, b, c = (points[triangles[:, k]] - origin for k in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    second_moment = (volume[:, None] * (a * a + b * b + c * c + (a + b + c) ** 2)).sum(axis=0) / 20
    return np.sqrt(second_moment / volume.sum())


def test_point_column_i_is_mask_axis_i() -> None:
    centre, semi_axes = (19.0, 29.0, 41.0), (12.0, 20.0, 30.0)
    mask = ellipsoid((40, 60, 84), centre, semi_axes)
    points, triangles, labels = get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(
        mask,
    )
    vertices = cell_vertices(points, triangles, labels, 1)
    extent = np.ptp(vertices, axis=0)
    middle = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    assert np.allclose(extent, 2 * np.array(semi_axes), atol=3.5)
    assert np.allclose(middle, centre, atol=1.0)


def test_column_0_is_scaled_by_z_and_columns_1_2_by_xy() -> None:
    points = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    scaled = set_pixel_size(points, xy_pixel_size=0.25, z_pixel_size=2.0)
    assert np.array_equal(scaled, np.array([[2.0, 0.5, 0.75], [20.0, 5.0, 7.5]]))
    assert np.array_equal(points, np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]))  # the input is not modified


def test_function_method_and_top_level_alias_agree_bitwise() -> None:
    mask = ellipsoid((24, 30, 36), (12.0, 15.0, 18.0), (6.0, 9.0, 11.0))
    algorithm = get_default_mesh_reconstruction_algorithm()
    points, _, _ = algorithm.construct_mesh_from_segmentation_mask(mask)
    expected = set_pixel_size(points, 0.3, 0.9)
    assert np.array_equal(set_points_pixel_size(points, xy_pixel_size=0.3, z_pixel_size=0.9), expected)
    algorithm.set_pixel_size(0.3, 0.9)
    assert np.array_equal(algorithm.last_constructed_mesh[0], expected)


def test_a_sphere_sampled_with_a_coarse_z_step_comes_back_round() -> None:
    # A sphere of radius 5.4 um imaged with 0.3 um pixels and a 0.9 um z step: (z, y, x) stack, z every 3rd plane.
    xy, z_step, radius = 0.3, 0.9, 5.4
    stack = ellipsoid((90, 60, 60), (45.0, 30.0, 30.0), (radius / xy,) * 3)[::3]
    points, triangles, _ = get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(stack)
    rms = rms_radius_per_axis(set_pixel_size(points, xy, z_step), triangles)
    assert np.allclose(rms, radius / np.sqrt(5), rtol=0.2)
    assert rms.max() / rms.min() < 1.4
