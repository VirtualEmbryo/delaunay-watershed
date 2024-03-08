"""Module to compute euclidean distance transforms, place points and build tesselation from image segmentation.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from time import time

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from skimage.feature import peak_local_max


def tesselation_from_edt(
    edt_image: NDArray[np.float64],
    min_distance: int = 5,
    print_info: bool = False,
) -> Delaunay:
    """Build a Delaunay tesselation on points at extrema of the Euclidean Distance Transform of the segmented image.

    Args:
        edt_image (NDArray[np.uint8]): Segmented image.
        min_distance (int, optional): Minimal distance between two points of the Delaunay tesselation. Defaults to 5.
        print_info (bool, optional): Whether to print some intermediate results and details. Defaults to False.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint8], Delaunay, NDArray[np.float64]]:
            - array of seed coordinates in the pixel images (points)
            - array of seed values
            - Delaunay tesselation constructed
            - Euclidean Distance Transform of segmented image.
    """
    if print_info:
        print("Mode == Skimage")
        print("min_distance =", min_distance)

    corners = _give_corners(edt_image)

    t3 = time()
    if print_info:
        print("Searching local extremas ...")

    nx, ny, nz = edt_image.shape

    # fix seed as we place points with some randomness
    # rng = np.random.default_rng(42)
    # edt = total_edt + rng.random((nx, ny, nz)) * 1e-5
    # Note: we keep the old way to be sure we have the same results across our tests
    np.random.seed(42)  # noqa: NPY002
    edt = edt_image + np.random.rand(nx, ny, nz) * 1e-5  # noqa: NPY002

    local_mins = peak_local_max(-edt, min_distance=min_distance, exclude_border=False)
    if print_info:
        print("Number of local minimas :", len(local_mins))

    local_maxes = peak_local_max(edt, min_distance=min_distance, exclude_border=False)
    if print_info:
        print("Number of local maxes :", len(local_maxes))

    t4 = time()
    if print_info:
        print("Local minimas computed in ", np.round(t4 - t3, 2))

    all_points = np.vstack((corners, local_maxes, local_mins))

    if print_info:
        print("Starting triangulation..")

    tesselation = Delaunay(all_points)

    t5 = time()
    if print_info:
        print("Triangulation build in ", np.round(t5 - t4, 2))

    return tesselation


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
