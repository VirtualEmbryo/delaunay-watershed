"""Operations on segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
import numpy as np
from numpy.typing import NDArray


def extract_seed_coords_and_indices(
    segmented_image: NDArray[np.uint],
    edt_image: NDArray[np.float64],
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Extract seeds from a segmented image and corresponding euclidean distance transform (EDT).

    Seeds are placed at the most inner points (found via the EDT) of each segmented region.

    Args:
        segmented_image (NDArray[np.uint]): Segmented image.
        edt_image (NDArray[np.float64]): Corresponding euclidean distance transform.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]:
            - array of coordinates for each seed,
            - array of corresponding index in segmented image.
    """
    seeds_coords = []

    nx, ny, nz = segmented_image.shape
    table_coords = _pixels_coords(nx, ny, nz)
    values_lbls = np.unique(segmented_image)

    flat_edt = edt_image.flatten()
    flat_labels = segmented_image.flatten()
    for i in values_lbls:
        f_i = flat_labels == i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])

    seeds_coords = np.array(seeds_coords, dtype=np.uint)
    seeds_indices = values_lbls
    return seeds_coords, seeds_indices


def _pixels_coords(nx: int, ny: int, nz: int) -> NDArray[np.int64]:
    """Create all pixels coordinates for an image of size nx*ny*nz."""
    xv = np.linspace(0, nx - 1, nx)
    yv = np.linspace(0, ny - 1, ny)
    zv = np.linspace(0, nz - 1, nz)
    xvv, yvv, zvv = np.meshgrid(xv, yv, zv)
    xvv = np.transpose(xvv, (1, 0, 2)).flatten()
    yvv = np.transpose(yvv, (1, 0, 2)).flatten()
    zvv = zvv.flatten()
    points = np.vstack([xvv, yvv, zvv]).transpose().astype(int)
    return points
