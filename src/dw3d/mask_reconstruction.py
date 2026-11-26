"""Rebuild a segmentation mask from a compressed segmentation (mesh, seeds and image shape).

Every voxel centre is assigned to the region of space the mesh's triangles enclose it in, and each region is
named by the seed it holds (see `dw3d.exact_voxelisation`). The decision is exact and uses only the mesh's own
triangles, so the rebuilt mask does not depend on the NumPy or SciPy version, nor on the machine. It
approximates the original mask as closely as the mesh approximates the mask's interfaces; it is not a lossless
copy.

Sacha Ichbiah, 2021.
Matthieu Perez, 2024.
Herve Turlier's group, 2026.
"""

import numpy as np
from numpy.typing import NDArray

from dw3d.exact_voxelisation import enclosed_regions_of_voxel_centres


def _links_between_regions(regions: NDArray[np.int64], n_regions: int) -> NDArray[np.int64]:
    """Count the links (6-neighbour pairs of voxel centres) between every two different regions.

    Returns:
        NDArray[np.int64]: `(k, 3)` rows `(region, neighbour, count)`, both orders, sorted.
    """
    pairs = []
    for axis in range(regions.ndim):
        lower = np.take(regions, np.arange(regions.shape[axis] - 1), axis=axis).ravel()
        upper = np.take(regions, np.arange(1, regions.shape[axis]), axis=axis).ravel()
        differ = lower != upper
        pairs.append(lower[differ] * n_regions + upper[differ])
        pairs.append(upper[differ] * n_regions + lower[differ])
    keys, counts = np.unique(np.concatenate(pairs), return_counts=True)
    return np.stack([keys // n_regions, keys % n_regions, counts], axis=1)


def _squared_depth(regions: NDArray[np.int64], region: int, voxel: NDArray[np.int64]) -> int:
    """Squared distance, exact in integers, from a voxel centre of `region` to the nearest centre outside it.

    The centres beyond the image count as outside. The nearest outside centre is always a 6-neighbour of the
    region (a step towards the voxel along an axis where they differ would be nearer), so only those are read.
    """
    inside = np.pad(regions == region, 1, constant_values=False)
    shell = ~inside & (
        np.roll(inside, 1, 0)
        | np.roll(inside, -1, 0)
        | np.roll(inside, 1, 1)
        | np.roll(inside, -1, 1)
        | np.roll(inside, 1, 2)
        | np.roll(inside, -1, 2)
    )
    difference = np.argwhere(shell) - 1 - np.asarray(voxel, dtype=np.int64)
    return int(np.min(np.sum(difference * difference, axis=1)))


def _label_every_region(regions: NDArray[np.int64], n_regions: int, seeds: NDArray[np.integer]) -> NDArray[np.int64]:
    """The label of each region: its seed's index, or that of the labelled region it shares most links with.

    A region holding several seeds takes the index of the seed deepest inside it (the largest distance to the
    nearest centre outside the region), ties to the lowest index: the other seeds belong to cells the mesh
    does not hold (a cell no tetrahedron lies inside), whose seed lies at the edge of a neighbour's region. A
    region holding none (a few voxel centres the lattice cuts off, for example inside a sliver thinner than a
    voxel) takes the label of the labelled region it shares the most links with, ties to the lowest label,
    repeated until every region is labelled.
    """
    label_of_region = np.full(n_regions, -1, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    seed_regions = regions[tuple(seeds.T)]
    for region in np.unique(seed_regions):
        held = np.flatnonzero(seed_regions == region)
        if len(held) > 1:
            depths = [_squared_depth(regions, int(region), seeds[index]) for index in held]
            label_of_region[region] = held[int(np.argmax(depths))]
        else:
            label_of_region[region] = held[0]
    if np.all(label_of_region >= 0):
        return label_of_region
    links = _links_between_regions(regions, n_regions)
    while np.any(label_of_region < 0):
        candidate = links[(label_of_region[links[:, 0]] < 0) & (label_of_region[links[:, 1]] >= 0)]
        if len(candidate) == 0:
            break
        neighbour_label = label_of_region[candidate[:, 1]]
        # Per unlabelled region: most links first, then the lowest label.
        order = np.lexsort((neighbour_label, -candidate[:, 2], candidate[:, 0]))
        candidate, neighbour_label = candidate[order], neighbour_label[order]
        first = np.concatenate([[True], candidate[1:, 0] != candidate[:-1, 0]])
        label_of_region[candidate[first, 0]] = neighbour_label[first]
    return label_of_region


def reconstruct_mask_from_dict(dict_mask: dict[str]) -> NDArray[np.int64]:
    """Reconstruct a segmentation image from a saved dict of points, triangles, seeds coords and desired image shape.

    Each voxel centre takes the label of the region of space the mesh's triangles enclose it in: two
    neighbouring centres are in the same region when the segment between them crosses no triangle, and a
    region takes the index of the seed it holds. A centre lying exactly on a triangle belongs to the region it
    enters when moved infinitesimally towards increasing index along axis 0, then 1, then 2 (towards
    decreasing index on the last voxel of an axis, so that a cell cut by the image box keeps its outermost
    voxels). A region holding no seed (a few centres cut off by a sliver thinner than a voxel) takes the label
    of the region it shares the most neighbouring centres with. Every test is exact, so the result does not
    depend on the NumPy or SciPy version.

    Args:
        dict_mask (dict[str]): the dictionary `MeshReconstructionAlgorithm.last_compressed_segmentation`
            returns, with keys `points`, `triangles`, `seeds` and `image_shape`. The points must be in
            voxel units of the mask, as the reconstruction returns them (not scaled by `set_pixel_size`
            or built by `construct_mesh_in_physical_units`). After `construct_mesh_in_physical_units`, divide
            the points by the resampled spacing (the shortest spacing) first: the seeds and `image_shape` are
            those of the resampled cubic grid, and that is the mask this rebuilds.

    Returns:
        NDArray[np.int64]: the rebuilt mask, of shape `image_shape`, in 64-bit integer labels (not the original
            mask's dtype). Seed `k` (in the ascending label order of the original mask, label 0 included when
            present) fills its region with label `k`, so the labels are those of the original mask only when
            these were `0, 1, ..., n - 1`.
    """
    points = np.asarray(dict_mask["points"], dtype=np.float64)
    triangles = np.asarray(dict_mask["triangles"], dtype=np.int64)
    seeds = np.asarray(dict_mask["seeds"])
    image_shape = tuple(int(n) for n in dict_mask["image_shape"])

    regions, n_regions = enclosed_regions_of_voxel_centres(points, triangles, image_shape)
    label_of_region = _label_every_region(regions, n_regions, seeds)
    labels: NDArray[np.int64] = label_of_region[regions]
    return labels


def reconstruct_mask_from_saved_file_dict(filename_dict: str) -> NDArray[np.int64]:
    """Reconstruct a segmentation image from a saved dict of points, triangles, seeds coords and desired image shape.

    Args:
        filename_dict (str): a `.npy` file written by `dw3d.save_compressed_segmentation`. It is
            unpickled (`allow_pickle=True`), so read only files you trust.

    Returns:
        NDArray[np.int64]: the rebuilt mask, as `reconstruct_mask_from_dict` returns it.
    """
    dict_mask: dict[str, NDArray] = np.load(filename_dict, allow_pickle=True).item()
    return reconstruct_mask_from_dict(dict_mask)
