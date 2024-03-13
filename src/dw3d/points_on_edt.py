"""Module with methods to place points on an EDT image for future tesselation.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max


def peak_local_points(
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
) -> NDArray[np.uint]:
    """Peak local min and max points (+ corner points) from an EDT image.

    Args:
        edt_image (NDArray[np.float64]): 3D image of an Euclidean Distance Transform.
        min_distance (int): Minimum distance between peaks (local min and local max)
        print_info (bool, optional): Print details about the algorithm. Defaults to False.

    Returns:
        NDArray[np.uint]: An array of 3D pixel coordinates of local min & max of the EDT.
    """
    corners = _give_corners(edt_image)

    if print_info:
        print("Searching local extremas ...")

    nx, ny, nz = edt_image.shape

    # fix seed as we place points with some randomness
    # rng = np.random.default_rng(42)
    # edt = total_edt + rng.random((nx, ny, nz)) * 1e-5
    # Note: we keep the old way to be sure we have the same results across our tests
    np.random.seed(42)  # noqa: NPY002
    edt = edt_image + np.random.rand(nx, ny, nz) * 1e-5  # noqa: NPY002

    local_mins = peak_local_max(-edt, min_distance=min_distance, exclude_border=False).astype(np.uint)
    if print_info:
        print("Number of local minimas :", len(local_mins))

    local_maxes = peak_local_max(edt, min_distance=min_distance, exclude_border=False).astype(np.uint)
    if print_info:
        print("Number of local maxes :", len(local_maxes))

    all_points = np.vstack((corners, local_maxes, local_mins), dtype=np.uint)
    return all_points


def _give_corners(img: NDArray) -> NDArray[np.uint]:
    """Give the eight corners pixels coordinates of a 3D image."""
    corners = np.zeros((8, 3), dtype=np.uint)
    index = 0
    a, b, c = img.shape
    for i in [0, a - 1]:
        for j in [0, b - 1]:
            for k in [0, c - 1]:
                corners[index] = np.array([i, j, k])
                index += 1
    return corners
