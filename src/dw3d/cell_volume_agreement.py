"""List the cells whose reconstructed mesh volume disagrees with their voxel count.

Why this exists
---------------
A reconstruction can lose a cell or misplace part of it while the mesh stays closed, manifold and
free of self-intersections, so no topological check sees it. Measured with the default algorithm at
`min_distance` 3: on the 47 benchmark masks 2 of 240 cells, on two 246-timepoint real embryo series
169 of 5 038, and on a 1 866-cell tissue 20 cells, have a mesh volume below half or above one and a
half times their voxel count, a non-positive volume, or no surface at all (the benchmark and
embryo counts measured before 0.7.1, whose meshes differ slightly; the 1 866-cell tissue's 20 are the
same cells on 0.7.0's and on 0.7.1's meshes). Almost all of them are small cells (a few hundred
voxels, about two voxels thick) that the seeded watershed never grows out of; before 0.6.0 one of
the 1 866-cell tissue's was a single-tetrahedron cell that junction relocation turned inside out,
which relocation now refuses to do. Their interfaces are missing or misplaced, and a pressure inferred for such a
cell is not a measurement.

This module changes no geometry. It compares each label's mesh volume with its voxel count and
reports the labels that fall outside a fixed range, so that a user, or foambryo, can see which
cells to distrust.

Conventions
-----------
The mesh volume of a label is the divergence-theorem volume with the package's winding
convention: `+det(p0, p1, p2) / 6` for every triangle whose first label it is, `-det / 6` for every
triangle whose second label it is. Volumes are in cubic voxels, so the comparison is made on the
mesh in mask voxel units, before any pixel-size scaling. Label 0 is the exterior medium and is
never listed, nor is an enclosed medium (a lumen) the reconstruction made its own region
(`media`). The range `[0.5, 1.5]` is inclusive; its bounds are fixed, not tuning parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

#: The inclusive range of mesh volume over voxel count inside which a cell is not listed. Fixed:
#: the bounds the disagreement was measured and classified with, not a tuning parameter.
CELL_VOLUME_RATIO_RANGE = (0.5, 1.5)

#: Why a label is listed, in the order the reasons are tested (the first that applies is given).
ABSENT = "absent"
NON_POSITIVE = "non-positive"
BELOW = "below"
ABOVE = "above"

#: Labels up to this value are counted with `np.bincount` instead of `np.unique`'s full sort.
_DIRECT_COUNT_MAX = 2**24
#: Voxels counted per block, so the counting's working memory does not grow with the volume.
_COUNT_BLOCK_VOXELS = 2**24


class CellVolumeDisagreementWarning(UserWarning):
    """Some cells of the last reconstruction have a mesh volume that disagrees with their voxel count.

    Emitted by
    :meth:`dw3d.reconstruction_algorithm.MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`
    **once per reconstruction**, when :func:`compare_cell_volumes_with_mask` lists at least one cell.
    The list is the algorithm's ``cell_volume_report``. The mesh itself is unchanged. The warning is
    a subclass of :class:`UserWarning`, so it can be silenced or promoted to an error on its own with
    :func:`warnings.filterwarnings`.
    """


@dataclass
class CellVolumeReport:
    """The labels whose mesh volume disagrees with their voxel count.

    Attributes:
        ratio_range: the inclusive range of mesh volume over voxel count that is not listed.
        n_cells: non-zero labels in the mask.
        n_cells_in_mesh: those that at least one mesh triangle carries.
        cells: one dict per listed label, in ascending label order, with `label`, `voxels`,
            `mesh_volume` (cubic voxels, `None` when absent), `ratio` (`mesh_volume / voxels`,
            `None` when absent) and `reason` (`"absent"`, `"non-positive"`, `"below"` or `"above"`,
            the first that applies in that order).
    """

    ratio_range: tuple[float, float] = CELL_VOLUME_RATIO_RANGE
    n_cells: int = 0
    n_cells_in_mesh: int = 0
    cells: list[dict] = field(default_factory=list)

    @property
    def labels(self) -> list[int]:
        """The listed labels, ascending."""
        return [cell["label"] for cell in self.cells]

    def counts(self) -> dict[str, int]:
        """How many labels are listed for each reason."""
        reasons = (ABSENT, NON_POSITIVE, BELOW, ABOVE)
        return {reason: sum(cell["reason"] == reason for cell in self.cells) for reason in reasons}

    def as_dict(self) -> dict:
        """A JSON-serialisable copy of the report, with the counts per reason."""
        return {
            "ratio_range": list(self.ratio_range),
            "n_cells": self.n_cells,
            "n_cells_in_mesh": self.n_cells_in_mesh,
            "n_listed": len(self.cells),
            "counts": self.counts(),
            "cells": [dict(cell) for cell in self.cells],
        }


def voxel_counts(segmented_mask: NDArray) -> dict[int, int]:
    """Voxel count of every label present in the mask, the medium 0 included when present.

    Non-negative integer labels up to `_DIRECT_COUNT_MAX` are counted block by block with
    `np.bincount`; any other mask falls back to `np.unique`. Both give the same counts.

    Args:
        segmented_mask (NDArray): the label image, any shape.

    Returns:
        dict[int, int]: label to voxel count, labels ascending.
    """
    mask = np.asarray(segmented_mask)
    if mask.dtype.kind in "iu" and mask.size:
        low, high = int(mask.min()), int(mask.max())
        if low >= 0 and high <= _DIRECT_COUNT_MAX:
            flat = mask.reshape(-1)
            counts = np.zeros(high + 1, dtype=np.int64)
            for start in range(0, flat.size, _COUNT_BLOCK_VOXELS):
                counts += np.bincount(flat[start : start + _COUNT_BLOCK_VOXELS].astype(np.intp), minlength=high + 1)
            present = np.flatnonzero(counts)
            return dict(zip(present.tolist(), counts[present].tolist(), strict=True))
    values, counts = np.unique(mask, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def mesh_cell_volumes(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
) -> dict[int, float]:
    """Divergence-theorem volume of every label a triangle carries, with the package's winding.

    Args:
        points (NDArray[np.float64]): `(n, 3)` vertex coordinates, voxel units.
        triangles (NDArray[np.integer]): `(m, 3)` vertex indices.
        labels (NDArray[np.integer]): `(m, 2)`; a triangle's normal points out of `labels[:, 0]`.

    Returns:
        dict[int, float]: label to volume in cubic voxels, every label of `labels` (0 included).
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(triangles):
        return {}
    corners = points[triangles]
    det = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))
    present, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.reshape(labels.shape)
    volume = np.bincount(inverse[:, 0], weights=det, minlength=len(present))
    volume -= np.bincount(inverse[:, 1], weights=det, minlength=len(present))
    return dict(zip(present.tolist(), (volume / 6.0).tolist(), strict=True))


def compare_cell_volumes_with_mask(
    segmented_mask: NDArray,
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    ratio_range: tuple[float, float] = CELL_VOLUME_RATIO_RANGE,
    *,
    media: tuple[int, ...] | list[int] = (0,),
) -> CellVolumeReport:
    """List every non-zero label whose mesh volume disagrees with its voxel count.

    A label is listed when no triangle carries it (`"absent"`), when its mesh volume is `<= 0`
    (`"non-positive"`), or when its mesh volume over voxel count is below (`"below"`) or above
    (`"above"`) the inclusive `ratio_range`, tested in that order.

    Args:
        segmented_mask (NDArray): the label image the mesh was reconstructed from; 0 is the medium.
        points (NDArray[np.float64]): mesh vertex coordinates in mask voxel units (before any
            pixel-size scaling, or the ratio is off by the voxel volume).
        triangles (NDArray[np.integer]): `(m, 3)` vertex indices.
        labels (NDArray[np.integer]): `(m, 2)`, the two materials each triangle separates.
        ratio_range (tuple[float, float]): the inclusive range that is not listed.
        media (tuple[int, ...] | list[int]): the labels that are media, not cells (the exterior 0 and the enclosed
            media a reconstruction made, `MeshReconstructionAlgorithm.medium_labels`); never listed nor counted.
            Keyword-only. Defaults to `(0,)`.

    Returns:
        CellVolumeReport: the listed labels, ascending, with their voxel count, volume and reason.
    """
    lower, upper = ratio_range
    voxels = voxel_counts(segmented_mask)
    volumes = mesh_cell_volumes(points, triangles, labels)
    report = CellVolumeReport(ratio_range=(float(lower), float(upper)))
    skipped = {int(label) for label in media}
    for label, count in voxels.items():
        if label in skipped:
            continue
        report.n_cells += 1
        volume = volumes.get(label)
        if volume is None:
            report.cells.append({"label": label, "voxels": count, "mesh_volume": None, "ratio": None, "reason": ABSENT})
            continue
        report.n_cells_in_mesh += 1
        ratio = volume / count
        if volume <= 0:
            reason = NON_POSITIVE
        elif ratio < lower:
            reason = BELOW
        elif ratio > upper:
            reason = ABOVE
        else:
            continue
        report.cells.append({"label": label, "voxels": count, "mesh_volume": volume, "ratio": ratio, "reason": reason})
    return report


def disagreement_message(report: CellVolumeReport) -> str:
    """The text of :class:`CellVolumeDisagreementWarning` for one report."""
    lower, upper = report.ratio_range
    shown = ", ".join(str(label) for label in report.labels[:10])
    more = f" and {len(report.cells) - 10} more" if len(report.cells) > 10 else ""
    return (
        f"{len(report.cells)} of {report.n_cells} cells have a mesh volume that disagrees with their voxel "
        f"count (outside [{lower:g}, {upper:g}] times it, non-positive, or absent from the mesh): labels "
        f"{shown}{more}. Their interfaces are missing or misplaced and a pressure inferred for them is not "
        "a measurement; see the algorithm's `cell_volume_report`."
    )
