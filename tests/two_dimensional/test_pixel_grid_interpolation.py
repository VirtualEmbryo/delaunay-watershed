"""The interp2d replacement is bilinear interpolation on the pixel grid, with interp2d's call convention."""
import math

import numpy as np
import pytest

from dw2d.pixel_grid_interpolation import BilinearOnPixelGrid


def bilinear(image, row, col):
    """Textbook bilinear interpolation between the four pixel centres around (row, col)."""
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1, c1 = min(r0 + 1, image.shape[0] - 1), min(c0 + 1, image.shape[1] - 1)
    fr, fc = row - r0, col - c0
    return ((1 - fr) * (1 - fc) * image[r0, c0] + (1 - fr) * fc * image[r0, c1]
            + fr * (1 - fc) * image[r1, c0] + fr * fc * image[r1, c1])


def test_equals_bilinear_interpolation_at_random_points():
    rng = np.random.default_rng(20261018)
    image = rng.normal(size=(17, 23))
    f = BilinearOnPixelGrid(image)
    points = rng.uniform([0, 0], [16, 22], size=(500, 2))
    for row, col in points:
        assert f(row, col)[0] == pytest.approx(bilinear(image, row, col), rel=1e-12, abs=1e-12)


def test_exact_at_pixel_centres_and_call_convention():
    image = np.arange(12.0).reshape(3, 4)
    f = BilinearOnPixelGrid(image)
    assert f(2, 3).shape == (1,)
    for i in range(3):
        for j in range(4):
            assert f(i, j)[0] == image[i, j]


def test_vectorised_evaluation_is_bitwise_the_per_point_one():
    rng = np.random.default_rng(1)
    image = rng.normal(size=(30, 40))
    f = BilinearOnPixelGrid(image)
    points = rng.uniform([0, 0], [29, 39], size=(200, 2))
    assert np.array_equal(f.at(points), np.array([f(r, c)[0] for r, c in points]))


def test_integer_label_images_read_zero_only_inside_the_medium():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[2:, 2:] = 3
    f = BilinearOnPixelGrid(labels)
    assert f(0.5, 0.5)[0] == 0
    assert f(1.5, 1.5)[0] > 0


def test_query_outside_the_grid_raises():
    f = BilinearOnPixelGrid(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        f(-0.5, 1.0)


def scipy_formula(image, row, col, multiply_add):
    """SciPy's ``evaluate_linear_2d`` for one point, in Python floats: ``multiply_add(a, b, c)`` is a*b + c."""
    i = min(int(np.floor(row)), image.shape[0] - 2)
    j = min(int(np.floor(col)), image.shape[1] - 2)
    y0, y1 = row - i, col - j
    a0, a1 = 1 - y0, 1 - y1
    value = 0.0
    value = multiply_add(float(image[i, j]) * a0, a1, value)
    value = multiply_add(float(image[i, j + 1]) * a0, y1, value)
    value = multiply_add(float(image[i + 1, j]) * y0, a1, value)
    return multiply_add(float(image[i + 1, j + 1]) * y0, y1, value)


def unfused(a, b, c):
    return a * b + c


def probe_points(rng, shape, n):
    """Random points, pixel centres and points on the last row and column of the grid."""
    random = rng.uniform([0, 0], [shape[0] - 1, shape[1] - 1], size=(n, 2))
    centres = np.array([(i, j) for i in range(shape[0]) for j in range(shape[1])], dtype=np.float64)
    edges = np.array([(shape[0] - 1, c) for c in random[:20, 1]] + [(r, shape[1] - 1) for r in random[:20, 0]])
    return np.vstack([random, centres, edges])


def test_kernel_is_bitwise_scipys_formula_without_fused_multiply_adds():
    """Pins the kernel on every machine: Python floats round each operation, as NumPy ufuncs do."""
    rng = np.random.default_rng(20261019)
    image = rng.normal(size=(23, 31))
    points = probe_points(rng, image.shape, 5000)
    expected = np.array([scipy_formula(image, r, c, unfused) for r, c in points])
    assert np.array_equal(BilinearOnPixelGrid(image).at(points), expected)
    labels = rng.integers(0, 6, size=(9, 7))
    points = probe_points(rng, labels.shape, 500)
    assert np.array_equal(BilinearOnPixelGrid(labels).at(points), np.array([scipy_formula(labels, r, c, unfused) for r, c in points]))


def test_kernel_agrees_with_scipys_compiled_kernel():
    """SciPy's compiled kernel is either bitwise this one (no contraction) or bitwise its fused version.

    SciPy's macOS arm64 wheels contract the multiply-adds into fused multiply-adds (measured when
    dw2d 0.8.0 was prepared); then the two kernels differ by rounding only, and when ``math.fma`` exists
    (Python >= 3.13) the difference is checked to be exactly that contraction.
    """
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(20261018)
    image = rng.random((40, 40))
    points = probe_points(rng, image.shape, 20000)
    grid = (np.linspace(0, 39, 40), np.linspace(0, 39, 40))
    compiled = RegularGridInterpolator(grid, image, method="linear")(points)
    ours = BilinearOnPixelGrid(image).at(points)
    if np.array_equal(compiled, ours):
        return
    np.testing.assert_allclose(compiled, ours, rtol=0, atol=4 * np.finfo(np.float64).eps * np.abs(image).max())
    if hasattr(math, "fma"):
        fused = np.array([scipy_formula(image, r, c, math.fma) for r, c in points])
        assert np.array_equal(compiled, fused), "SciPy's kernel is neither the unfused nor the fused formula"
