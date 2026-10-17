"""Seed extraction in two passes over the voxels gives exactly the per-label loop's seeds.

`extract_seed_coords_and_indices` used to run one masked `argmax` over the whole volume per label,
so its cost grew as labels x voxels (27 minutes on a 1 866-cell tissue). It now reads the volume
twice, in blocks: once for each label's maximum, once for the first voxel attaining it. That is a
pure speed change, so the seeds must not move by a single voxel. These tests compare it, bitwise,
with the original loop kept here verbatim, on volumes built to stress the tie-break: integer EDT
values (many equal maxima), labels whose maximum is zero or negative (where the loop's `where`
fills other labels with 0 and can return a voxel outside the label), NaN, several label dtypes,
negative labels, and block sizes small enough to split ties across blocks.
"""

import numpy as np
import pytest

from dw3d import segmentation
from dw3d.edt import compute_edt_classical
from dw3d.segmentation import extract_seed_coords_and_indices


def per_label_loop(segmented_image, edt_image):
    """The original implementation, verbatim apart from its commented-out lines."""
    _nx, ny, nz = segmented_image.shape
    values_lbls = np.unique(segmented_image)
    flat_edt = edt_image.flatten()
    flat_labels = segmented_image.flatten()
    seeds_coords = [
        segmentation._linear_to_3d_index(np.argmax(np.where(flat_labels == i, flat_edt, 0)), ny, nz)
        for i in values_lbls
    ]
    return np.array(seeds_coords, dtype=np.uint), values_lbls


def assert_bitwise_equal(expected, got):
    for e, g in zip(expected, got, strict=True):
        assert e.dtype == g.dtype
        assert e.shape == g.shape
        assert np.array_equal(e, g)


@pytest.mark.parametrize("seed", range(40))
def test_random_volumes_with_ties_match_the_loop(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in rng.integers(1, 12, size=3))
    dtype = [np.int32, np.uint16, np.uint64, np.int64, np.uint8][seed % 5]
    labels = rng.integers(0, int(rng.integers(1, 10)), size=shape).astype(dtype)
    edt = rng.integers(0, 4, size=shape).astype([np.float64, np.float32][seed % 2])
    monkeypatch.setattr(segmentation, "_SEED_BLOCK_VOXELS", int(rng.integers(1, 300)))
    assert_bitwise_equal(per_label_loop(labels, edt), extract_seed_coords_and_indices(labels, edt))


@pytest.mark.parametrize("case", ["zero_maximum", "negative", "nan", "negative_labels", "one_label"])
def test_the_cases_where_the_loop_can_leave_its_label(case):
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 5, size=(6, 7, 8)).astype(np.int32)
    edt = rng.integers(0, 3, size=labels.shape).astype(np.float64)
    if case == "zero_maximum":
        edt[labels == 2] = 0.0
    elif case == "negative":
        edt[labels == 3] = -1.0 - edt[labels == 3]
    elif case == "nan":
        edt[(labels == 1) & (rng.random(labels.shape) < 0.3)] = np.nan
    elif case == "negative_labels":
        labels -= 2
    elif case == "one_label":
        labels[:] = 4
        edt[:] = 0.0
    assert_bitwise_equal(per_label_loop(labels, edt), extract_seed_coords_and_indices(labels, edt))


def test_a_voronoi_tissue_with_its_classical_edt_matches_the_loop(monkeypatch):
    """Real EDT plateaus, an exterior region, and blocks that cut through cells."""
    rng = np.random.default_rng(1977)
    side = 40
    centres = rng.uniform(0, side, size=(30, 3))
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    labels = (np.argmin(((grid[:, None, :] - centres[None]) ** 2).sum(-1), axis=1) + 1).astype(np.int32)
    labels = labels.reshape(side, side, side)
    labels[:, :, :4] = 0
    edt = compute_edt_classical(labels)
    monkeypatch.setattr(segmentation, "_SEED_BLOCK_VOXELS", 3 * side * side + 17)
    assert_bitwise_equal(per_label_loop(labels, edt), extract_seed_coords_and_indices(labels, edt))


def test_the_label_values_are_np_unique_whichever_path_counts_them(monkeypatch):
    labels = np.array([[[5, 0], [5, 9]]], dtype=np.uint16)
    edt = np.ones(labels.shape)
    expected = np.unique(labels)
    _coords, indices = extract_seed_coords_and_indices(labels, edt)
    assert indices.dtype == expected.dtype
    assert np.array_equal(indices, expected)
    monkeypatch.setattr(segmentation, "_DIRECT_LABEL_TABLE_MAX", 3)  # forces np.unique
    _coords, indices = extract_seed_coords_and_indices(labels, edt)
    assert indices.dtype == expected.dtype
    assert np.array_equal(indices, expected)


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="3-D"):
        extract_seed_coords_and_indices(np.zeros((3, 3), dtype=np.int32), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        extract_seed_coords_and_indices(np.zeros((3, 3, 3), dtype=np.int32), np.zeros((3, 3, 2)))
