"""Module to compute euclidean distance transforms, place points and build Delaunay tesselation from image segmentation.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from time import time

import numpy as np
from edt import edt as euclidean_dt
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries


def _recover_ignore_index(
    input_img: NDArray[np.uint],
    orig: NDArray[np.uint],
    ignore_index: None | int,
) -> NDArray[np.uint]:
    """Put back the ignored_index in input_image ?"""
    if ignore_index is not None:
        mask = orig == ignore_index
        input_img[mask] = ignore_index

    return input_img


class StandardLabelToBoundary:
    """Class-function because why not ? The goal is to extract pixels boundaries in a multi-labeled image."""

    def __init__(
        self,
        ignore_index: int | None = None,
        append_label: bool = False,
        mode: str = "thick",
        foreground: bool = False,
    ) -> None:
        """Register the parameters for the class-function.

        Args:
            ignore_index (int | None, optional): Consider the ignored index as the background ?. Defaults to None.
            append_label (bool, optional): Append original input data to the result. Defaults to False.
            mode (str, optional): {'thick', 'inner', 'outer', 'subpixel'}
                How to mark the boundaries:

                thick: any pixel not completely surrounded by pixels of the same label (defined by connectivity)
                       is marked as a boundary. This results in boundaries that are 2 pixels thick.
                inner: outline the pixels *just inside* of objects, leaving background pixels untouched.
                outer: outline pixels in the background around object boundaries. When two objects touch,
                       their boundary is also marked.
                subpixel: return a doubled image, with pixels *between* the original pixels
                          marked as boundary where appropriate.
                Defaults to "thick".
            foreground (bool, optional): Extract the foreground and put it in the result. Defaults to False.
        """
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m: NDArray[np.uint]) -> NDArray[np.uint]:
        """Call the function to extract pixels boundaries in a multi-labeled image.

        The output is an image with boundaries marked but it can change with parameters...
        """
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype("int32")

        results = []
        if self.foreground:
            foreground = (m > 0).astype("uint8")
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


def interpolate_image(image: NDArray[np.uint8]) -> RegularGridInterpolator:
    """Return an interpolated image, a function with values based on pixels."""
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    image_interp = RegularGridInterpolator((x, y, z), image)
    return image_interp


def pad_mask(mask: NDArray[np.uint8], pad_size: int = 1) -> NDArray[np.uint8]:
    """Pad a mask with ones on the borders."""
    padded_mask = mask.copy()[
        pad_size:-pad_size,
        pad_size:-pad_size,
        pad_size:-pad_size,
    ]
    padded_mask = np.pad(
        padded_mask,
        ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)),
        "constant",
        constant_values=1,
    )
    return padded_mask


def give_corners(img: NDArray) -> NDArray[np.uint]:
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


def compute_edt_base(labels: NDArray[np.uint8], prints: bool = False) -> NDArray[np.float64]:
    """Compute the Euclidean Distance Transorm of a segmented image.

    It will be 0 at borders and cells boundaries. Bigger when getting far from these points.
    """
    if prints:
        print("Computing EDT ...")
    t1 = time()

    b = StandardLabelToBoundary()(labels)[0]  # "thick" boundaries are marked by 1, 0 outside
    mask_2 = b
    edt_2 = euclidean_dt(mask_2)  # EDT of the thick boundaries (0 elsewhere)
    b = pad_mask(b)  # exterior bbox is marked as 1
    mask_1 = 1 - b  # 1 everywhere except bbox and boundaries
    edt_1 = euclidean_dt(
        mask_1,
    )  # main part of the final EDT. Both inside cells and outside cells. 0 in boundaries & bbox
    inv = (
        np.amax(edt_2) - edt_2
    )  # max EDT2 everywhere except on thick boundaries where it decreases to 0 on the mid of boundaries
    total_edt = (edt_1 + np.amax(edt_2)) * mask_1 + inv * mask_2  # total EDT is valid also on thick boundaries

    # # Matthieu Perez try 2: augment constrast in EDT
    # Total_EDT = 255 * ((Total_EDT / 255) ** 0.5)

    t2 = time()
    if prints:
        print("EDT computed in ", np.round(t2 - t1, 2))

    return total_edt


# Matthieu Perez : tests bias EDT
# def compute_edt_with_bias(labels, prints=False):
#     if prints:
#         print("Computing EDT (bias) ...")
#     t1 = time()

#     region_indices = np.unique(labels)
#     total_boundaries = np.zeros(labels.shape)

#     for index in region_indices:
#         if index == 0:
#             region_labels = np.where(labels == 0, 1, 0)
#         else:
#             region_labels = np.where(labels == index, labels, 0)
#         total_boundaries += StandardLabelToBoundary()(region_labels)[0]

#     # total_boundaries *= 3
#     total_boundaries = np.amax(total_boundaries) - total_boundaries

#     # "thick" boundaries are marked by 1, 0 outside
#     b = StandardLabelToBoundary()(labels)[0]
#     mask_2 = b
#     # EDT of the thick boundaries (0 elsewhere)
#     EDT_2 = euclidean_dt(mask_2)
#     b = pad_mask(b)  # exterior bbox is marked as 1
#     mask_1 = 1 - b  # 1 everywhere except bbox and boundaries
#     # main part of the final EDT. Both inside cells and outside cells. 0 in boundaries & bbox
#     EDT_1 = euclidean_dt(mask_1)
#     # max EDT2 everywhere except on thick boundaries where it decreases to 0 on the mid of boundaries
#     # + total_boundaries which is less on interesting parts of the mesh
#     inv = np.amax(EDT_2) - EDT_2 + total_boundaries
#     # total EDT is valid also on thick boundaries
#     Total_EDT = (EDT_1 + np.amax(inv)) * mask_1 + inv * mask_2

#     # # Matthieu Perez try 2: augment constrast in EDT
#     # Total_EDT = 255 * ((Total_EDT / 255) ** 0.5)

#     t2 = time()
#     if prints:
#         print("EDT computed in ", np.round(t2 - t1, 2))

#     return Total_EDT


def build_triangulation(
    labels: NDArray[np.uint8],
    min_distance: int = 5,
    prints: bool = False,
) -> tuple[NDArray[np.uint], NDArray[np.uint8], Delaunay, NDArray[np.float64]]:
    """Build a Delaunay tesselation on points at extrema of the Euclidean Distance Transform of the segmented image.

    Args:
        labels (NDArray[np.uint8]): Segmented image.
        min_distance (int, optional): Minimal distance between two points of the Delaunay tesselation. Defaults to 5.
        prints (bool, optional): Whether to print some intermediate results and details. Defaults to False.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint8], Delaunay, NDArray[np.float64]]:
            - array of seed coordinates in the pixel images (points)
            - array of seed values
            - Delaunay tesselation constructed
            - Euclidean Distance Transform of segmented image.
    """
    if prints:
        print("Mode == Skimage")
        print("min_distance =", min_distance)

    total_edt = compute_edt_base(labels, prints)
    # total_edt = compute_edt_with_bias(labels, prints)

    seeds_coords = []

    nx, ny, nz = labels.shape
    table_coords = _pixels_coords(nx, ny, nz)
    values_lbls = np.unique(labels)

    flat_edt = total_edt.flatten()
    flat_labels = labels.flatten()
    for i in values_lbls:
        f_i = flat_labels == i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])

    seeds_coords = np.array(seeds_coords, dtype=np.uint)
    seeds_indices = values_lbls

    corners = give_corners(total_edt)

    t3 = time()
    if prints:
        print("Searching local extremas ...")

    # fix seed as we place points with some randomness
    # rng = np.random.default_rng(42)
    # edt = total_edt + rng.random((nx, ny, nz)) * 1e-5
    # Note: we keep the old way to be sure we have the same results across our tests
    np.random.seed(42)  # noqa: NPY002
    edt = total_edt + np.random.rand(nx, ny, nz) * 1e-5  # noqa: NPY002

    local_mins = peak_local_max(-edt, min_distance=min_distance, exclude_border=False)
    if prints:
        print("Number of local minimas :", len(local_mins))

    local_maxes = peak_local_max(edt, min_distance=min_distance, exclude_border=False)
    if prints:
        print("Number of local maxes :", len(local_maxes))

    t4 = time()
    if prints:
        print("Local minimas computed in ", np.round(t4 - t3, 2))

    all_points = np.vstack((corners, local_maxes, local_mins))

    # # Matthieu Perez try 1:
    # # regular grid
    # XV = np.linspace(0, nx - 1, int(nx / 20))
    # YV = np.linspace(0, ny - 1, int(ny / 20))
    # ZV = np.linspace(0, nz - 1, int(nz / 20))
    # xvv, yvv, zvv = np.meshgrid(XV, YV, ZV)
    # xvv = np.transpose(xvv, (1, 0, 2)).flatten()
    # yvv = np.transpose(yvv, (1, 0, 2)).flatten()
    # zvv = zvv.flatten()
    # added_points = np.vstack(([xvv, yvv, zvv])).transpose().astype(int)
    # print(XV)
    # all_points = np.vstack((all_points, added_points))  # to do: unique ?
    # print(len(all_points), "points")

    if prints:
        print("Starting triangulation..")

    tesselation = Delaunay(all_points)

    t5 = time()
    if prints:
        print("Triangulation build in ", np.round(t5 - t4, 2))

    return (seeds_coords, seeds_indices, tesselation, total_edt)
