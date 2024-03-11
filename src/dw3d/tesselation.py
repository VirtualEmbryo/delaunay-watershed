"""Module to compute euclidean distance transforms, place points and build tesselation from image segmentation.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay


def simple_delaunay_tesselation(
    points_for_tesselation: NDArray[np.uint],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a Delaunay tesselation on points at extrema of the Euclidean Distance Transform of the segmented image.

    Args:
        points_for_tesselation (NDArray[np.uint]): Points to construct the tesselation.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64]]:
            - tesselation's points
            - tesselation's tetrahedrons as array of point indices.
    """
    tesselation = Delaunay(points_for_tesselation)

    return tesselation.points, tesselation.simplices.astype(np.int64)
