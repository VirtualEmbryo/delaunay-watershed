"""Compute Euclidean Distance Trasnform out of segmentation image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from time import time

import numpy as np
from edt import edt as euclidean_dt
from numpy.typing import NDArray
from skimage.segmentation import find_boundaries


def compute_edt_classical(segmentation_mask: NDArray[np.uint], print_info: bool = False) -> NDArray[np.float64]:
    """Compute the Euclidean Distance Transorm of a segmented image.

    It will be 0 at borders and cells boundaries. Bigger when getting far from these points.
    """
    if print_info:
        print("Computing EDT ...")
    t1 = time()

    b = _StandardLabelToBoundary()(segmentation_mask)[0]  # "thick" boundaries are marked by 1, 0 outside
    mask_2 = b
    edt_2 = euclidean_dt(mask_2)  # EDT of the thick boundaries (0 elsewhere)
    b = _pad_mask(b)  # exterior bbox is marked as 1
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
    if print_info:
        print("EDT computed in ", np.round(t2 - t1, 2))

    return total_edt


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


class _StandardLabelToBoundary:
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


def _pad_mask(mask: NDArray[np.uint8], pad_size: int = 1) -> NDArray[np.uint8]:
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
