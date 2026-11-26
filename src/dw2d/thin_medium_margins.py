"""A thin margin of medium between a cell and the image border: why the cell took it, and the padding that prevents it.

The same defect as in dw3d (`dw3d.thin_medium_margins`), and the same rule.

The defect
----------
dw2d's distance field is 0 on the image's outer ring of pixels (`dw2d.Geometric_utilities.pad_mask` makes the image
border a boundary), at or below the level of the membranes, and a membrane point is placed only where no lower value
lies within `min_distance` (`skimage.feature.peak_local_max`). So the membrane between a cell and a side of the image a
few pixels away loses its points, the triangles joining the cell's interior to the border span the medium, and the
watershed gives them to the cell. Measured before 0.8.0 on single squares (side 12, 20, 32 px), squares with the margin
on one side, a doublet of 20-px squares and disks (radius 10 and 16 px), at `min_distance` 3, 4 and 5: at margins of 2
px up to about `min_distance`, the membrane facing the near side carried no point, the cell had edges on a side its
pixels do not reach, and its area was 1.04 to 1.73 times its pixel count; from `min_distance + 1` or `+ 2` px on, every
cell was back at 0.84 to 0.99 (its value at a margin of 16 px).

The fix
-------
dw2d places its points on the field without a random dither, so the mask can be padded before it is meshed: every side
of the image that no cell reaches and whose margin (its smallest gap to a cell) is below
`thin_margin_bound(min_distance)`, `2 * min_distance + 2` as in dw3d, is padded with label 0 up to that bound, and the
mesh is translated back into the mask's frame. The pixels of the mask keep the field and the points they have
unpadded, except within the padded sides' reach. A mask whose cells reach every side or stay at least that far from it
is meshed exactly as before. A side that a cell reaches is a crop and is never padded. A cell left with edges on a side
its pixels do not reach is named by `ThinMediumMarginWarning`. On a padded side the mesh is that of the same tissue
imaged with a wider margin, so a cell's boundary may reach past the image border there, by at most the padding.

dw3d decides from a first mesh (a vertex on a face no cell reaches), because its meshes cost more and its dither makes
a padded mesh a new draw. In 2D that test would miss the defect twice over: at a margin of `min_distance` the cell took
the medium without reaching the side (1.28 times its pixels at `min_distance` 4, its nearest vertex 2.5 px beyond the
membrane), and a 12-px square 3 or 4 px from one side collapsed to a triangle of 0.23 times its pixels, with no vertex
near the side, at `min_distance` 5 (0.84 at a margin of 12 px).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: The sides of the image, in the order of `numpy.pad`'s widths: (axis 0 low, axis 0 high, axis 1 low, axis 1 high).
IMAGE_SIDES = ("axis0_low", "axis0_high", "axis1_low", "axis1_high")


def thin_margin_bound(min_distance: int) -> int:
    """The medium margin, in pixels, below which a side no cell reaches is padded: `2 * min_distance + 2` (dw3d's)."""
    return 2 * int(min_distance) + 2


def cells_on_each_image_side(mask: NDArray[np.integer]) -> tuple[tuple[int, ...], ...]:
    """The cells (non-zero labels) with a pixel on each side of the image, in the order of `IMAGE_SIDES`."""
    mask = np.asarray(mask)
    rings = (mask[0], mask[-1], mask[:, 0], mask[:, -1])
    return tuple(tuple(int(v) for v in np.unique(ring) if v != 0) for ring in rings)


def side_margins(mask: NDArray[np.integer]) -> tuple[int | None, ...]:
    """The medium margin of each side: the pixels of label 0 between it and the nearest cell along its normal.

    Returns:
        tuple[int | None, ...]: in the order of `IMAGE_SIDES`; 0 when a cell reaches the side, `None` without a cell.
    """
    occupied = np.asarray(mask) != 0
    out: list[int | None] = []
    for axis in range(2):
        filled = np.flatnonzero(occupied.any(axis=1 - axis))
        if not len(filled):
            out += [None, None]
            continue
        out += [int(filled[0]), int(occupied.shape[axis] - 1 - filled[-1])]
    return tuple(out)


def padding_of_thin_margins(
    margins: tuple[int | None, ...],
    cells_on_sides: tuple[tuple[int, ...], ...],
    bound: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """The `numpy.pad` widths that bring every side no cell reaches to a margin of `bound` pixels, or `None`."""
    widths = [[0, 0], [0, 0]]
    for index, margin in enumerate(margins):
        if cells_on_sides[index] or margin is None or margin >= bound:
            continue
        axis, side = divmod(index, 2)
        widths[axis][side] = bound - margin
    if not any(any(pair) for pair in widths):
        return None
    return ((int(widths[0][0]), int(widths[0][1])), (int(widths[1][0]), int(widths[1][1])))


def unsupported_border_edges(
    points: NDArray[np.floating],
    segments: NDArray[np.integer],
    labels: NDArray[np.integer],
    mask: NDArray[np.integer],
) -> list[tuple[int, str]]:
    """`(cell, side)` for each cell with an edge lying on a side of the image that its pixels do not reach."""
    points, segments, labels = np.asarray(points), np.asarray(segments, np.int64), np.asarray(labels, np.int64)
    shape = np.shape(mask)
    reached = cells_on_each_image_side(mask)
    out = set()
    for index, side in enumerate(IMAGE_SIDES):
        axis, high = divmod(index, 2)
        plane = 0 if not high else shape[axis] - 1
        on = (points[segments[:, 0], axis] == plane) & (points[segments[:, 1], axis] == plane) if len(segments) else []
        for pair in labels[on] if len(segments) else []:
            for cell in pair:
                if cell != 0 and int(cell) not in reached[index]:
                    out.add((int(cell), side))
    return sorted(out)
