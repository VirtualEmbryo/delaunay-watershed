"""Bilinear interpolation of an image on its pixel grid (replaces ``scipy.interpolate.interp2d``).

``geometry_reconstruction_2d`` used ``interp2d(x, y, image.T, kind="linear")`` to read the
distance transform (edge scores) and the label image (exterior triangles) at arbitrary
points. ``interp2d`` was deprecated in SciPy 1.10 and removed in SciPy 1.14, so dw2d 0.1.5
cannot run on current SciPy. ``interp2d(kind="linear")`` on a regular grid computes bilinear
interpolation, which is what ``scipy.interpolate.RegularGridInterpolator(method="linear")``
computes; on every golden case the whole pipeline is bitwise identical to dw2d
0.1.5 running on SciPy 1.13 with ``interp2d``.

The kernel is written with NumPy ufuncs rather than called from SciPy, in the operation order
of SciPy's compiled ``evaluate_linear_2d``. SciPy's kernel is compiled C, and a compiler may
contract its multiply-adds into fused multiply-adds: SciPy's macOS arm64 wheels do, so the
edge scores differed from x86-64 by 1-2 ulp there. NumPy ufuncs round
every operation separately on every platform, so this kernel is bitwise SciPy's unfused one,
and the same on every machine.
"""

import numpy as np
from numpy.typing import NDArray


class BilinearOnPixelGrid:
    """Bilinear interpolant of a 2D image whose samples sit at integer pixel centres.

    Args:
        image: 2D array indexed ``image[row, col]``, at least 2 x 2; sample ``(i, j)`` sits at
            the point ``(row, col) = (i, j)`` in pixel units. Integer images (labels) are
            interpolated as numbers, exactly as ``interp2d`` did: only a value of exactly 0 is
            meaningful for a label image (all four neighbouring pixels are 0).

    Calling ``f(row, col)`` returns a 1-element array, the call convention of ``interp2d``
    that dw2d relies on (it indexes the result with ``[0]``). ``f.at(points)`` evaluates an
    ``(n, 2)`` array of ``(row, col)`` points at once.

    For a point with ``i = min(floor(row), n_rows - 2)``, ``y0 = row - i`` (same for the
    column, ``j`` and ``y1``), the value is accumulated as in SciPy:
    ``0 + v[i, j] (1-y0)(1-y1) + v[i, j+1] (1-y0) y1 + v[i+1, j] y0 (1-y1) + v[i+1, j+1] y0 y1``,
    left to right, every product and sum rounded separately.

    Failure mode: a query outside ``[0, n_rows - 1] x [0, n_cols - 1]`` raises ``ValueError``
    (``interp2d`` extrapolated silently). dw2d only queries points on Delaunay edges and
    triangle centroids, which lie in the convex hull of pixel centres, hence inside the grid.
    """

    def __init__(self, image: NDArray) -> None:
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2 or min(image.shape) < 2:
            raise ValueError(f"bilinear interpolation needs a 2D image of at least 2 x 2 pixels, got shape {image.shape}")
        self._image = image

    def __call__(self, row: float, col: float) -> NDArray[np.float64]:
        """Value at one point ``(row, col)``, as a 1-element array."""
        return self.at(np.array([[float(row), float(col)]]))

    def at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Values at an ``(n, 2)`` array of ``(row, col)`` points."""
        points = np.asarray(points, dtype=np.float64)
        n_rows, n_cols = self._image.shape
        rows, cols = points[:, 0], points[:, 1]
        if not (np.all((rows >= 0) & (rows <= n_rows - 1)) and np.all((cols >= 0) & (cols <= n_cols - 1))):
            raise ValueError("a query point lies outside the pixel grid")
        i = np.minimum(np.floor(rows), n_rows - 2).astype(np.intp)
        j = np.minimum(np.floor(cols), n_cols - 2).astype(np.intp)
        y0 = rows - i
        y1 = cols - j
        a0 = 1 - y0
        a1 = 1 - y1
        v = self._image
        value = 0.0 + v[i, j] * a0 * a1
        value = value + v[i, j + 1] * a0 * y1
        value = value + v[i + 1, j] * y0 * a1
        value = value + v[i + 1, j + 1] * y0 * y1
        return value
