from time import time

import numpy as np
import torch
from edt import edt as euclidean_dt
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


class StandardLabelToBoundary:
    def __init__(
        self,
        ignore_index=None,
        append_label=False,
        mode="thick",
        foreground=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
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


def interpolate_image(image):
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    image_interp = RegularGridInterpolator((x, y, z), image)
    return image_interp


def pad_mask(mask, pad_size=1):
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


def give_corners(img):
    Points = np.zeros((8, 3))
    index = 0
    a, b, c = img.shape
    for i in [0, a - 1]:
        for j in [0, b - 1]:
            for k in [0, c - 1]:
                Points[index] = np.array([i, j, k])
                index += 1
    return Points


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


def compute_edt_base(labels, prints=False):
    if prints:
        print("Computing EDT ...")
    t1 = time()

    b = StandardLabelToBoundary()(labels)[0]  # "thick" boundaries are marked by 1, 0 outside
    mask_2 = b
    EDT_2 = euclidean_dt(mask_2)  # EDT of the thick boundaries (0 elsewhere)
    b = pad_mask(b)  # exterior bbox is marked as 1
    mask_1 = 1 - b  # 1 everywhere except bbox and boundaries
    EDT_1 = euclidean_dt(
        mask_1,
    )  # main part of the final EDT. Both inside cells and outside cells. 0 in boundaries & bbox
    inv = (
        np.amax(EDT_2) - EDT_2
    )  # max EDT2 everywhere except on thick boundaries where it decreases to 0 on the mid of boundaries
    Total_EDT = (EDT_1 + np.amax(EDT_2)) * mask_1 + inv * mask_2  # total EDT is valid also on thick boundaries

    # # Matthieu Perez try 2: augment constrast in EDT
    # Total_EDT = 255 * ((Total_EDT / 255) ** 0.5)

    t2 = time()
    if prints:
        print("EDT computed in ", np.round(t2 - t1, 2))

    return Total_EDT


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


def build_triangulation(labels, min_distance=5, mode="torch", prints=False):
    if mode == "torch":
        return build_triangulation_torch(labels, (min_distance // 2) * 2 + 1, prints)
    else:
        return build_triangulation_skimage(labels, min_distance, prints)


def build_triangulation_torch(
    labels,
    min_distance=5,
    prints=False,
):  # ,size_shell=2,dist_shell=4):
    if prints:
        print("Mode == Torch")
        print("Kernel size =", min_distance)

    Total_EDT = compute_edt_base(labels, prints)
    # Total_EDT = compute_edt_with_bias(labels, prints)

    seeds_coords = []

    nx, ny, nz = labels.shape
    table_coords = _pixels_coords(nx, ny, nz)
    values_lbls = np.unique(labels)

    flat_edt = Total_EDT.flatten()
    flat_labels = labels.flatten()
    for i in values_lbls:
        f_i = flat_labels == i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])

    seeds_coords = np.array(seeds_coords)
    seeds_indices = values_lbls

    corners = give_corners(Total_EDT)

    t3 = time()
    if prints:
        print("Searching local extremas ...")
    np.random.seed(42)
    EDT = Total_EDT + np.random.rand(nx, ny, nz) * 1e-5
    T = torch.tensor(EDT).unsqueeze(0)
    kernel_size = min_distance
    padding = kernel_size // 2
    F = torch.nn.MaxPool3d(
        (kernel_size, kernel_size, kernel_size),
        stride=(1, 1, 1),
        padding=padding,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    )

    minpooled = F(-T)[0].numpy()
    markers_min = (EDT + minpooled) == 0
    local_mins = table_coords[markers_min.flatten()]
    if prints:
        print("Number of local minimas :", len(local_mins))

    maxpooled = F(T)[0].numpy()
    markers_max = (EDT - maxpooled) == 0
    local_maxes = table_coords[markers_max.flatten()]
    if prints:
        print("Number of local maxes :", len(local_maxes))

    t4 = time()
    if prints:
        print("Local minimas computed in ", np.round(t4 - t3, 2))

    all_points = np.vstack((corners, local_maxes, local_mins))

    if prints:
        print("Starting triangulation..")

    tesselation = Delaunay(all_points)

    t5 = time()
    if prints:
        print("Triangulation build in ", np.round(t5 - t4, 2))

    return (seeds_coords, seeds_indices, tesselation, Total_EDT)


def build_triangulation_skimage(labels, min_distance=5, prints=False):
    if prints:
        print("Mode == Skimage")
        print("min_distance =", min_distance)

    Total_EDT = compute_edt_base(labels, prints)
    # Total_EDT = compute_edt_with_bias(labels, prints)

    seeds_coords = []

    nx, ny, nz = labels.shape
    table_coords = _pixels_coords(nx, ny, nz)
    values_lbls = np.unique(labels)

    flat_edt = Total_EDT.flatten()
    flat_labels = labels.flatten()
    for i in values_lbls:
        f_i = flat_labels == i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])

    seeds_coords = np.array(seeds_coords)
    seeds_indices = values_lbls

    corners = give_corners(Total_EDT)

    t3 = time()
    if prints:
        print("Searching local extremas ...")
    np.random.seed(42)

    # # try Matthieu Perez add labels to peak local max
    # region_indices = np.unique(labels)
    # total_boundaries = np.zeros(labels.shape)

    # for index in region_indices:
    #     if index == 0:
    #         region_labels = np.where(labels == 0, 1, 0)
    #     else:
    #         region_labels = np.where(labels == index, labels, 0)
    #     total_boundaries += StandardLabelToBoundary()(region_labels)[0]

    # total_boundaries = np.amax(total_boundaries) - total_boundaries
    # bv = np.unique(total_boundaries).astype(np.int64)

    # noise = np.random.rand(nx, ny, nz) * 1e-5

    # EDT = Total_EDT + noise

    # local_mins = None
    # for value in bv[:-2]:
    #     print("value ==", value)
    #     new_local_mins = peak_local_max(
    #         -EDT,
    #         min_distance=value + 1,
    #         exclude_border=False,
    #         labels=(total_boundaries == value),
    #     )
    #     if local_mins is None:
    #         local_mins = new_local_mins
    #     else:
    #         local_mins = np.vstack((local_mins, new_local_mins))

    # value = bv[-2]
    # new_local_mins = peak_local_max(
    #     -EDT,
    #     min_distance=min_distance,
    #     exclude_border=False,
    #     labels=(total_boundaries == value),
    # )
    # if local_mins is None:
    #     local_mins = new_local_mins
    # else:
    #     local_mins = np.vstack((local_mins, new_local_mins))

    EDT = Total_EDT + np.random.rand(nx, ny, nz) * 1e-5

    local_mins = peak_local_max(-EDT, min_distance=min_distance, exclude_border=False)
    if prints:
        print("Number of local minimas :", len(local_mins))

    local_maxes = peak_local_max(EDT, min_distance=min_distance, exclude_border=False)
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

    return (seeds_coords, seeds_indices, tesselation, Total_EDT)
