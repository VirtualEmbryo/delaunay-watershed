"""Operations on segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

#: Voxels handled per block by `extract_seed_coords_and_indices`. A block holds one dense label index
#: (8 bytes) and one comparison flag (1 byte) per voxel, so about 150 MB of working memory. Fixed;
#: not a tuning parameter of the result, which does not depend on it.
_SEED_BLOCK_VOXELS = 2**24

#: Integer labels up to this value are counted through a direct look-up table (8 bytes per possible
#: label) instead of `np.unique`'s full sort of the volume. The result is the same either way.
_DIRECT_LABEL_TABLE_MAX = 2**24

_VOLUME_NDIM = 3


def extract_seed_coords_and_indices(
    segmented_image: NDArray[np.uint],
    edt_image: NDArray[np.float64],
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Extract seeds from a segmented image and corresponding euclidean distance transform (EDT).

    Seeds are placed at the most inner points (found via the EDT) of each segmented region: for
    each label, the voxel of largest EDT value, and among equal values the first one in C order
    (the order of `segmented_image.ravel()`).

    The result is bitwise the same as the original per-label loop,
    `argmax(where(labels.ravel() == i, edt.ravel(), 0))` for each label `i`, but it reads the volume
    in two passes instead of one pass per label: the first pass takes each label's maximum, the
    second the first voxel that attains it. Both run on blocks of `_SEED_BLOCK_VOXELS` voxels, so the
    extra memory is bounded and does not grow with the volume. A label whose maximum is not
    strictly positive, or whose values include a NaN, is the one case in which the loop's
    expression can return a voxel outside the label (its `where` fills the other labels with 0);
    for such a label the loop's expression itself is evaluated, so the result stays identical.

    Args:
        segmented_image (NDArray[np.uint]): Segmented image, 3-D, one integer label per voxel.
        edt_image (NDArray[np.float64]): Corresponding euclidean distance transform, same shape.
            Any real dtype; values are compared in that dtype.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]:
            - array of coordinates for each seed, `(n_labels, 3)`, dtype `np.uint`, voxel indices
              along the image's three axes in order,
            - array of corresponding label values (`np.unique(segmented_image)`, same dtype as the
              image), sorted ascending, aligned with the coordinates.

    Raises:
        ValueError: if the image is not 3-D or the EDT does not have the image's shape.
    """
    if segmented_image.ndim != _VOLUME_NDIM:
        message = f"segmented_image must be 3-D, got shape {segmented_image.shape}"
        raise ValueError(message)
    if edt_image.shape != segmented_image.shape:
        message = f"edt_image has shape {edt_image.shape}, segmented_image {segmented_image.shape}"
        raise ValueError(message)

    values_lbls, dense_index = _label_values_and_dense_index(segmented_image)
    blocks = _BlockReader(segmented_image, edt_image, dense_index)
    maxima = _label_maxima(blocks, len(values_lbls), np.result_type(edt_image.dtype, 0))
    first_index = _first_index_of_maximum(blocks, maxima)

    # The loop fills other labels with 0, so its answer is the label's own first maximum only when
    # that maximum is strictly positive. Otherwise evaluate the loop's own expression.
    exact = ~(maxima > 0)
    if exact.any():
        flat_labels = segmented_image.ravel()
        flat_edt = edt_image.ravel()
        for position in np.flatnonzero(exact):
            first_index[position] = np.argmax(np.where(flat_labels == values_lbls[position], flat_edt, 0))

    seeds_coords = np.stack(np.unravel_index(first_index, segmented_image.shape), axis=1).astype(np.uint)
    return seeds_coords, values_lbls


class _BlockReader:
    """Iterate a label image and its EDT in blocks of whole planes along axis 0, in C order."""

    def __init__(
        self,
        segmented_image: NDArray,
        edt_image: NDArray,
        dense_index: Callable[[NDArray], NDArray[np.intp]],
    ) -> None:
        self.labels, self.edt, self.dense_index = segmented_image, edt_image, dense_index
        self.plane = segmented_image.shape[1] * segmented_image.shape[2]
        self.rows = max(1, _SEED_BLOCK_VOXELS // max(1, self.plane))

    def __iter__(self) -> Iterator[tuple[int, NDArray[np.intp], NDArray]]:
        """Yield `(flat offset of the block, dense label ids, EDT values)`, both 1-D."""
        for start in range(0, self.labels.shape[0], self.rows):
            stop = min(start + self.rows, self.labels.shape[0])
            yield start * self.plane, self.dense_index(self.labels[start:stop].ravel()), self.edt[start:stop].ravel()


def _label_maxima(blocks: _BlockReader, n_labels: int, dtype: np.dtype) -> NDArray:
    """Each label's maximum EDT value, in `dtype`. A NaN propagates (`np.maximum`)."""
    if dtype.kind == "f":
        maxima = np.full(n_labels, -np.inf, dtype=dtype)
    else:
        maxima = np.full(n_labels, np.iinfo(dtype).min, dtype=dtype)
    for _offset, ids, values in blocks:
        np.maximum.at(maxima, ids, values.astype(dtype, copy=False))
    return maxima


def _first_index_of_maximum(blocks: _BlockReader, maxima: NDArray) -> NDArray[np.int64]:
    """The flat C-order index of the first voxel at which each label attains its maximum (-1: none)."""
    first_index = np.full(len(maxima), -1, dtype=np.int64)
    for offset, ids, values in blocks:
        at_maximum = np.flatnonzero(values.astype(maxima.dtype, copy=False) == maxima[ids])
        if not len(at_maximum):
            continue
        hit, first = np.unique(ids[at_maximum], return_index=True)
        new = first_index[hit] < 0
        first_index[hit[new]] = offset + at_maximum[first[new]]
    return first_index


def _label_values_and_dense_index(segmented_image: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray[np.intp]]]:
    """The sorted distinct labels, as `np.unique` returns them, and a map from labels to their rank.

    Non-negative integer labels no larger than `_DIRECT_LABEL_TABLE_MAX` are counted block by
    block with `np.bincount` and ranked through a look-up table; any other image falls back to
    `np.unique` and `np.searchsorted`. Both give `np.unique(segmented_image)` exactly (same values,
    order and dtype).

    Returns:
        tuple: `(values, dense_index)`, where `dense_index(block)` maps a 1-D block of labels to
        their positions in `values` (`np.intp`).
    """
    image = segmented_image
    if image.dtype.kind in "iu" and image.size:
        low, high = int(image.min()), int(image.max())
        if low >= 0 and high <= _DIRECT_LABEL_TABLE_MAX:
            counts = np.zeros(high + 1, dtype=np.int64)
            plane = image.shape[1] * image.shape[2]
            rows = max(1, _SEED_BLOCK_VOXELS // max(1, plane))
            for start in range(0, image.shape[0], rows):
                counts += np.bincount(image[start:start + rows].ravel().astype(np.intp), minlength=high + 1)
            values = np.flatnonzero(counts).astype(image.dtype)
            table = np.full(high + 1, -1, dtype=np.intp)
            table[values] = np.arange(len(values), dtype=np.intp)
            return values, lambda block: table[block]
    values = np.unique(image)
    return values, lambda block: np.searchsorted(values, block)


def _linear_to_3d_index(idx: int, ny: int, nz: int) -> NDArray[np.int64]:
    z = idx // (ny * nz)
    idx -= z * ny * nz
    y = idx // nz
    x = idx % nz
    return np.array([z, y, x])
