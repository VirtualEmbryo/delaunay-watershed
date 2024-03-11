"""Score computation functions on a TesselationGraph, for the Watershed algorithm.

Sacha Ichibiah 2021
Matthieu Perez 2024
"""


import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


def compute_scores_by_mean_value(
    edt_image: NDArray[np.float64],
    vertices: NDArray[np.float64],
    triangle_faces: NDArray[np.uint],
) -> NDArray[np.float64]:
    """Compute scores on triangle faces of a tesselation for a Watershed algorithm.

    Scores are based on the mean value of the EDT image at some points of the triangles.

    Args:
        edt_image (NDArray[np.float64]): EDT image integrated to computed the scores.
        vertices (NDArray[np.float64]): Points positions (in EDT pixels coordinates space)
        triangle_faces (NDArray[np.uint]): Triangles faces as point indices.

    Returns:
        NDArray[np.float64]: Array of score for each triangle face of the tesselation.
    """
    f = _interpolate_image(edt_image)
    alpha = np.linspace(0, 1, 5)[1:-1]
    beta = np.linspace(0, 1, 5)[1:-1]
    gamma = np.linspace(0, 1, 5)[1:-1]

    vertices = vertices.copy()
    scale = np.amax(vertices, axis=0) / 2
    vertices -= scale
    vertices *= 1 - 1e-4
    vertices += scale * (1 - 1e-4)
    v = vertices[triangle_faces]

    v1 = v[:, 0]
    v2 = v[:, 1]
    v3 = v[:, 2]
    count = 0
    count_bad = 0
    score_faces = np.zeros(len(triangle_faces), dtype=np.float64)
    for a in alpha:
        for b in beta:
            for c in gamma:
                try:
                    s = a + b + c
                    l1 = a / s
                    l2 = b / s
                    l3 = c / s

                    score_faces += np.array(f(v1 * l1 + v2 * l2 + v3 * l3))

                    # Test Matthieu Perez: take score max (seems to improve a bit the results)
                    # Score_Faces = np.maximum(
                    #     Score_Faces, f(V1 * l1 + V2 * l2 + V3 * l3)
                    # )
                    count += 1
                except:  # except what ?? what can happen badly ??  # noqa: E722
                    count_bad += 1
    score_faces /= count
    return score_faces


def compute_scores_by_max_value(
    edt_image: NDArray[np.float64],
    vertices: NDArray[np.float64],
    triangle_faces: NDArray[np.uint],
) -> NDArray[np.float64]:
    """Compute scores on triangle faces of a tesselation for a Watershed algorithm.

    Scores are based on the max value of the EDT image at some points of the triangles.

    Args:
        edt_image (NDArray[np.float64]): EDT image integrated to computed the scores.
        vertices (NDArray[np.float64]): Points positions (in EDT pixels coordinates space)
        triangle_faces (NDArray[np.uint]): Triangles faces as point indices.

    Returns:
        NDArray[np.float64]: Array of score for each triangle face of the tesselation.
    """
    f = _interpolate_image(edt_image)
    alpha = np.linspace(0, 1, 5)[1:-1]
    beta = np.linspace(0, 1, 5)[1:-1]
    gamma = np.linspace(0, 1, 5)[1:-1]

    vertices = vertices.copy()
    scale = np.amax(vertices, axis=0) / 2
    vertices -= scale
    vertices *= 1 - 1e-4
    vertices += scale * (1 - 1e-4)
    v = vertices[triangle_faces]

    v1 = v[:, 0]
    v2 = v[:, 1]
    v3 = v[:, 2]

    score_faces = np.zeros(len(triangle_faces), dtype=np.float64)
    for a in alpha:
        for b in beta:
            for c in gamma:
                s = a + b + c
                l1 = a / s
                l2 = b / s
                l3 = c / s

                # Test Matthieu Perez: take score max (seems to improve a bit the results)
                score_faces = np.maximum(score_faces, f(v1 * l1 + v2 * l2 + v3 * l3))

    return score_faces


def _interpolate_image(image: NDArray[np.uint8]) -> RegularGridInterpolator:
    """Return an interpolated image, a function with values based on pixels."""
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    image_interp = RegularGridInterpolator((x, y, z), image)
    return image_interp
