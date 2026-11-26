"""Every label of the meshed mask that the finished mesh lacks is named, whatever the cause.

The mask: four cells filling a ball of radius 14 voxels that the z = 0 face of a 36^3 image cuts (foambryo's
command-line test mask "four cells cut by the image box"). At `min_distance` 3 cell 1 (1 648 voxels, 262 of them on
the cut face) is lost: its seed is the tetrahedron whose centroid is nearest to its deepest voxel, a sliver against
the image box whose faces all score below the face through which the rest of the cell, grouped without the seed,
reaches cell 2 in the watershed; the group goes to cell 2 whole, the cell keeps only its seed tetrahedron, and mesh
surgery relabels it. Before 0.8.0 only `CellVolumeDisagreementWarning` listed the cell.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dw3d import UnresolvedCellWarning, get_default_mesh_reconstruction_algorithm
from dw3d.enclosed_media import (
    REMOVED_BY_MESH_SURGERY,
    EnclosedMediumReport,
    UnresolvedRegion,
    unresolved_message,
)
from dw3d.io import load_rec_appendix
from tests.conftest import assert_every_mask_cell_is_meshed


def four_cells_in_a_ball(n: int = 36, centre_z: float | None = None) -> np.ndarray:
    """Four Voronoi cells filling a ball of radius 14 voxels; `centre_z` near 0 makes the image cut it."""
    z, y, x = np.indices((n, n, n))
    c = (n - 1) / 2
    cz = c if centre_z is None else centre_z
    ball = (z - cz) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= 14**2
    seeds = np.array([[cz - 6, c - 5, c], [cz + 6, c - 5, c + 1], [cz, c + 7, c - 6], [cz + 1, c + 4, c + 8]])
    distances = ((np.stack([z, y, x], axis=-1)[..., None, :] - seeds) ** 2).sum(axis=-1)
    return np.where(ball, distances.argmin(axis=-1) + 1, 0).astype(np.uint16)


def _reconstruct(mask: np.ndarray, min_distance: int = 3, *, pad_thin_medium_margins: bool = True):
    algorithm = get_default_mesh_reconstruction_algorithm(min_distance, False)
    algorithm.pad_thin_medium_margins = pad_thin_medium_margins
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    return algorithm, mesh, [w.category for w in caught]


@pytest.mark.parametrize("min_distance", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("centre_z", [6.0, None])
def test_the_mask_labels_are_the_mesh_labels_and_the_unresolved_ones(min_distance, centre_z) -> None:
    """Each label of the meshed mask is either carried by a triangle or named unresolved, never both."""
    algorithm, mesh, caught = _reconstruct(four_cells_in_a_ball(centre_z=centre_z), min_distance)
    in_mask = {int(label) for label in np.unique(algorithm._segmented_image)}
    in_mesh = {int(label) for label in np.unique(mesh[2])}
    unresolved = set(algorithm.enclosed_medium_report.dropped_seeds)
    assert in_mesh | unresolved == in_mask
    assert not in_mesh & unresolved
    assert (UnresolvedCellWarning in caught) == bool(unresolved)
    assert caught.count(UnresolvedCellWarning) <= 1


def test_a_cut_cell_lost_by_the_watershed_is_named(tmp_path) -> None:
    """The known issue's mesh: cell 1 is absent, named with its cause, in the report, the warning and the file.

    Meshed without the padding of thin medium margins: since 0.8.0 the default pads this mask's four free faces (the
    ball is 4 voxels from them), and the second mesh keeps 4 triangles of cell 1, a fragment of 0.010 of its voxel
    volume inside the box of voxel centres, while cell 2 holds 1.25 times its own; the watershed's defect is unchanged.
    """
    algorithm, mesh, caught = _reconstruct(four_cells_in_a_ball(centre_z=6.0), 3, pad_thin_medium_margins=False)
    assert 1 not in np.unique(mesh[2])
    report = algorithm.enclosed_medium_report
    (region,) = report.unresolved
    assert region == UnresolvedRegion(1, 1648, "cell", REMOVED_BY_MESH_SURGERY, 1, 33)
    assert report.unresolved_cells == (1,)
    assert not report.changes_the_mesh  # naming a lost cell changes nothing in the mesh
    assert UnresolvedCellWarning in caught
    assert algorithm.cell_volume_report.labels == [1]
    path = tmp_path / "cut.rec"
    algorithm.save_to_rec_mesh(path)
    appendix = load_rec_appendix(path)
    assert appendix["unresolved_regions"] == (1,)
    assert appendix["unresolved_regions_voxels"] == (1648,)


@pytest.mark.xfail(strict=True, reason="known issue: the seeded watershed loses cell 1 of this cut mask at "
                   "min_distance 3 (it keeps only its seed tetrahedron, which surgery relabels)")
def test_every_cell_of_the_cut_ball_is_meshed_at_the_default_spacing() -> None:
    # Without the padding of thin medium margins, which keeps cell 1 on this mask by a change of tesselation, not by
    # a fix of the watershed (see the test above).
    algorithm, mesh, _ = _reconstruct(four_cells_in_a_ball(centre_z=6.0), 3, pad_thin_medium_margins=False)
    assert_every_mask_cell_is_meshed(algorithm, mesh[2], "four cells cut by the image box")


@pytest.mark.parametrize("min_distance", [1, 2])
def test_every_cell_of_the_cut_ball_is_meshed_at_a_finer_spacing(min_distance) -> None:
    algorithm, mesh, _ = _reconstruct(four_cells_in_a_ball(centre_z=6.0), min_distance)
    assert_every_mask_cell_is_meshed(algorithm, mesh[2], f"four cells cut by the image box, md {min_distance}")


def test_the_warning_says_why_a_seeded_region_is_absent() -> None:
    report = EnclosedMediumReport(
        unresolved=(UnresolvedRegion(9, 27, "cell"), UnresolvedRegion(1, 1648, "cell", REMOVED_BY_MESH_SURGERY, 1, 33)),
        min_distance=3,
    )
    text = unresolved_message(report)
    assert "cell 9 (27 voxels)" in text  # the dropped seed's sentence, as in 0.7.1
    assert "no tetrahedron whose centroid lies in its voxels" in text
    assert "cell 1 (1648 voxels; the watershed left it 1 of the 33 tetrahedra centred in its voxels" in text
    assert "mesh surgery relabelled it" in text
    assert text.count("min_distance 3") == 2
