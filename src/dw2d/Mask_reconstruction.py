"""Rebuild an approximate label image from a compressed segmentation (a 2-D mesh, its seeds and the image shape).

The rebuild draws every mesh edge into a pixel mask (Bresenham's line between the two ends, each rounded to the
nearest pixel centre), then grows one region per seed by a watershed on the distance to the drawn lines. It is an
approximation: measured on one shipped mask, 95.6 % of the pixels agree with the mask the mesh was built from. dw3d's
rebuild of a 3-D mask is exact; dw2d's is not, yet.

The names mirror dw3d's: `reconstruct_mask_from_dict(dictionary)` and `reconstruct_mask_from_saved_file_dict(filename)`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from dw2d._matplotlib import pyplot as plt  # imported on first use: matplotlib is in the `viewing` extra


def create_mesh_semantic_masks(Verts, Edges, image_shape):
    """A float mask of the image's shape, 1 on the pixels every edge of the mesh crosses (see `trace_line`).

    Each edge is traced from its second vertex to its first. Bresenham's walk breaks its ties towards the end it starts
    from, so the drawn pixels depend on the direction: this is the direction dw2d stored each edge in before 0.8.0
    (since then every edge is stored the other way round, the orientation rule of dw3d's triangles), so the rebuilt mask
    of a mesh is the one dw2d drew before. A compressed segmentation written before 0.8.0 is traced from its edges'
    other ends, and may differ on tie pixels.
    """
    Mask = np.zeros(image_shape)
    for edge in Edges:
        v1, v2 = edge[1], edge[0]
        c = trace_line(Verts[v1], Verts[v2], image_shape)
        Mask[tuple(c.T)] = 1
    return Mask


def _lattice_point(point, image_shape=None) -> tuple[int, int]:
    """The pixel centre nearest to `point`, clipped into the image when its shape is given."""
    row, column = (int(v) for v in np.rint(np.asarray(point, dtype=np.float64)))
    if image_shape is not None:
        row = min(max(row, 0), int(image_shape[0]) - 1)
        column = min(max(column, 0), int(image_shape[1]) - 1)
    return row, column


def trace_line(v0, v1, image_shape=None):
    """The pixels of the segment `v0 -> v1` by Bresenham's line algorithm, both ends included.

    `v0` and `v1` are `(row, column)` points. Each end is first rounded to the nearest pixel centre (`numpy.rint`: a
    half-integer goes to the even neighbour) and, when `image_shape` is given, clipped into the image. Before 0.8.0
    the ends were used as given and the walk stepped by one pixel until it met the far end exactly, so it never ended
    when the two ends differed by a non-integer, as they do after the default junction relocation, which moves
    junctions to pixel corners (half-integers). On integer ends the result is unchanged.

    Modified from https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python.

    Returns:
        NDArray[np.int64]: `(n, 2)` pixel indices `(row, column)`, from `v0` to `v1`.
    """
    x0, y0 = _lattice_point(v0, image_shape)
    x1, y1 = _lattice_point(v1, image_shape)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append([x, y])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append([x, y])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append([x, y])
    return np.array(points, dtype=np.int64)


def create_mesh_instance_masks(Verts, Edges, image_shape, seeds, print_info=False):
    """Grow one region per seed (region `k + 1` from `seeds[k]`) by a watershed on the distance to the drawn mesh."""
    semantic_mask = create_mesh_semantic_masks(Verts, Edges, image_shape)
    distance = ndi.distance_transform_edt(1 - semantic_mask)
    markers = np.zeros(distance.shape, dtype=np.int_)
    for i in range(len(seeds)):
        markers[tuple(seeds[i].T)] = i + 1
    if print_info:  # printed unconditionally before 0.8.0
        print(distance.shape, markers.shape)
    return watershed(-distance, markers)


def _mesh_of(dict_mask: dict) -> tuple[NDArray, NDArray]:
    """`(points, segments)` from a compressed segmentation: the keys of 0.8.0 or those of dw2d 0.1.5."""
    if "points" in dict_mask:
        return np.asarray(dict_mask["points"]), np.asarray(dict_mask["segments"])
    edges = np.asarray(dict_mask["Edges"])
    return np.asarray(dict_mask["Verts"]), edges[:, :2]


def reconstruct_mask_from_dict(dict_mask: dict, plot: bool = False) -> NDArray[np.int64]:
    """Rebuild an approximate label image from a compressed segmentation dictionary.

    Args:
        dict_mask (dict): the dictionary `MeshReconstructionAlgorithm.last_compressed_segmentation` returns (keys
            `points`, `segments`, `labels`, `seeds`, `image_shape`), or one saved by dw2d 0.1.5's
            `export_segmentation` (keys `Verts`, `Edges`, `seeds`, `image_shape`). The points must be in pixels of
            the image (read the dictionary before any scaling). Deprecated, through 0.8.x: a file name, read with
            `numpy.load`; use `reconstruct_mask_from_saved_file_dict`.
        plot (bool, optional): draw the result with matplotlib. Defaults to False.

    Returns:
        NDArray[np.int64]: the label image; the seed at row `k` of `seeds` fills label `k` (seeds are stored one per
            label, in ascending label order, label 0 first).
    """
    if isinstance(dict_mask, (str, Path)):
        warnings.warn(
            "reconstruct_mask_from_dict given a file name is deprecated and will be removed in 0.9; "
            "use dw2d.reconstruct_mask_from_saved_file_dict(filename), or pass the dictionary.",
            DeprecationWarning,
            stacklevel=2,
        )
        dict_mask = np.load(dict_mask, allow_pickle=True).item()
    points, segments = _mesh_of(dict_mask)
    labels = create_mesh_instance_masks(points, segments, dict_mask["image_shape"], dict_mask["seeds"]) - 1
    if plot:
        plt.figure(figsize=(4, 4))
        plt.imshow(labels, plt.cm.nipy_spectral)
        plt.axis("off")
        plt.title("Reconstructed mask")
    return labels


def reconstruct_mask_from_saved_file_dict(filename: str | Path) -> NDArray[np.int64]:
    """Rebuild an approximate label image from a compressed segmentation saved with `dw2d.save_compressed_segmentation`.

    Args:
        filename (str | Path): the `.npy` file (a pickled dictionary; see `reconstruct_mask_from_dict`).

    Returns:
        NDArray[np.int64]: the label image, as `reconstruct_mask_from_dict` returns it.
    """
    return reconstruct_mask_from_dict(np.load(filename, allow_pickle=True).item())
