"""The cells whose mesh volume disagrees with their voxel count are listed, and the mesh is untouched.

A reconstruction can lose a small cell, give part of a cell to its neighbour, or (once measured)
turn a one-tetrahedron cell inside out, while the mesh stays closed, manifold and free of
self-intersections. No topological check sees that; only the cell's volume does. The algorithm
therefore compares every label's mesh volume with its voxel count after the last stage and lists
the labels outside `[0.5, 1.5]`, non-positive or absent, with one warning, changing no geometry.

The first tests pin the rule on a hand-made mesh; the last one pins it on the real timepoint where
both an absent and an inverted cell were measured.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

from dw3d import (
    CellVolumeDisagreementWarning,
    compare_cell_volumes_with_mask,
    get_default_mesh_reconstruction_algorithm,
)
from dw3d.cell_volume_agreement import ABOVE, ABSENT, BELOW, NON_POSITIVE, mesh_cell_volumes, voxel_counts
from tests.conftest import REPO_ROOT

# A closed tetrahedron of volume 36 cubic voxels bounding cell 1 against the medium 0. Labels are
# `(0, 1)`, so each normal must point out of the medium and into the cell, i.e. inwards.
CORNERS = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
INWARD = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
LABELS = np.array([[0, 1]] * 4)


def _mask(voxels_of_label: dict[int, int]) -> np.ndarray:
    """A flat label image with the given voxel count per label; the geometry plays no part."""
    values = np.concatenate([np.full(count, label, dtype=np.uint16) for label, count in voxels_of_label.items()])
    return values.reshape(1, 1, -1)


def test_the_tetrahedron_reads_its_own_volume():
    assert mesh_cell_volumes(CORNERS, INWARD, LABELS)[1] == pytest.approx(36.0)


@pytest.mark.parametrize(
    ("voxels", "triangles", "reason"),
    [
        (36, INWARD, None),  # ratio 1: not listed
        (72, INWARD, None),  # ratio 0.5: the range is inclusive
        (24, INWARD, None),  # ratio 1.5: inclusive
        (100, INWARD, BELOW),  # ratio 0.36
        (20, INWARD, ABOVE),  # ratio 1.8
        (36, INWARD[:, [0, 2, 1]], NON_POSITIVE),  # every triangle reversed: volume -36
    ],
)
def test_the_rule_on_one_cell(voxels, triangles, reason):
    report = compare_cell_volumes_with_mask(_mask({0: 10, 1: voxels}), CORNERS, triangles, LABELS)
    assert report.n_cells == 1
    assert report.n_cells_in_mesh == 1
    if reason is None:
        assert report.cells == []
    else:
        assert [(cell["label"], cell["reason"]) for cell in report.cells] == [(1, reason)]
        assert report.cells[0]["voxels"] == voxels


def test_a_label_no_triangle_carries_is_listed_absent_and_the_medium_never_is():
    report = compare_cell_volumes_with_mask(_mask({0: 10, 1: 36, 7: 5}), CORNERS, INWARD, LABELS)
    assert report.cells == [{"label": 7, "voxels": 5, "mesh_volume": None, "ratio": None, "reason": ABSENT}]
    assert report.counts() == {ABSENT: 1, NON_POSITIVE: 0, BELOW: 0, ABOVE: 0}
    assert report.as_dict()["n_listed"] == 1


def test_voxel_counts_agree_with_unique_on_both_paths():
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 9, size=(7, 8, 9)).astype(np.uint16)
    values, counts = np.unique(mask, return_counts=True)
    expected = dict(zip(values.tolist(), counts.tolist(), strict=True))
    assert voxel_counts(mask) == expected  # the bincount path
    assert voxel_counts(mask.astype(np.float64)) == expected  # the np.unique fallback


# The real timepoint the disagreement was traced on.


def _experimental_mask_or_skip(series: str, filename: str) -> np.ndarray:
    """A mask of `experimental-dataset`: the repository's sibling folder, or `EXPERIMENTAL_DATASET`."""
    candidates = [REPO_ROOT.parent / "experimental-dataset"]
    if os.environ.get("EXPERIMENTAL_DATASET"):
        candidates.append(Path(os.environ["EXPERIMENTAL_DATASET"]))
    for directory in candidates:
        path = directory / "segmentation_masks" / series / filename
        if path.exists():
            return tifffile.imread(path)
    pytest.skip(f"experimental-dataset/segmentation_masks/{series}/{filename} not found")
    return None


def test_the_absent_and_the_inverted_cell_of_a_real_timepoint_are_named():
    """`st1_mem_crop2`, timepoint 244: two cells that no surface check flags.

    Cell 490 (246 voxels) receives a single tetrahedron, 35 voxels' worth: its mesh volume is a seventh
    of its voxel count (junction relocation used to turn it inside out as well, which its guard now
    refuses). Cell 367 (249 voxels) has its seed on the same tetrahedron as cell 490, loses it, and has
    no surface at all. Both must be named, with those reasons, and the warning must be emitted
    exactly once; running the comparison again must leave the mesh bitwise as it was.
    """
    mask = _experimental_mask_or_skip("st1_mem_crop2", "Cam_long_00244.lux_crop.tif")
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    disagreement = [w for w in caught if issubclass(w.category, CellVolumeDisagreementWarning)]
    assert len(disagreement) == 1
    assert "cell_volume_report" in str(disagreement[0].message)

    report = algorithm.cell_volume_report
    reasons = {cell["label"]: cell["reason"] for cell in report.cells}
    assert reasons[367] == ABSENT
    assert reasons[490] == BELOW
    by_label = {cell["label"]: cell for cell in report.cells}
    assert by_label[490]["voxels"] == 246
    assert 0 < by_label[490]["mesh_volume"] < 0.5 * 246

    before = (points.copy(), np.asarray(triangles).copy(), np.asarray(labels).copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CellVolumeDisagreementWarning)
        algorithm._compare_cell_volumes_with_mask()
    after = algorithm.last_constructed_mesh
    assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))
    assert algorithm.cell_volume_report.as_dict() == report.as_dict()
