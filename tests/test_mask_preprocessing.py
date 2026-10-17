"""`dw3d.mask_preprocessing.resample_to_isotropic` on synthetic masks only.

Pins: the bitwise identity for cubic voxels (an isotropic mask must go through untouched), the frame and the
output length, that the input slices are reproduced exactly at their own centres, that shape-based interpolation
recovers a known isotropic geometry better than repeating slices (the reason it is the default: see the module
docstring), and the refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from dw3d.mask_preprocessing import coarse_axis_and_factor, isotropic_spacing, resample_to_isotropic


def three_cells(n: int = 61) -> np.ndarray:
    """Two touching balls and a third one, in a cube of side `n`, on cubic voxels (`uint16`, labels 0-3)."""
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[(z - 30) ** 2 + (y - 22) ** 2 + (x - 30) ** 2 < 14**2] = 1
    mask[(z - 27) ** 2 + (y - 40) ** 2 + (x - 30) ** 2 < 12**2] = 2
    mask[(z - 34) ** 2 + (y - 31) ** 2 + (x - 44) ** 2 < 9**2] = 3
    return mask


def test_cubic_voxels_return_the_input_array_itself() -> None:
    mask = three_cells()
    for spacing in ((1.0, 1.0, 1.0), (0.3, 0.3, 0.3), (0.3, 0.3 * (1 + 1e-8), 0.3)):
        out = resample_to_isotropic(mask, spacing)
        assert out is mask
    assert coarse_axis_and_factor((2.0, 2.0, 2.0)) is None


def test_output_length_dtype_and_labels() -> None:
    mask = three_cells()[::3]  # 21 slices, z spacing 3
    out = resample_to_isotropic(mask, (3.0, 1.0, 1.0))
    assert out.shape == ((21 - 1) * 3 + 1, 61, 61)
    assert out.dtype == mask.dtype
    assert set(np.unique(out)) <= set(np.unique(mask))
    out = resample_to_isotropic(three_cells()[:40], (5 / 3, 1.0, 1.0))
    assert out.shape[0] == int(np.floor(39 * 5 / 3 + 1e-9)) + 1


@pytest.mark.parametrize("method", ["shape", "nearest"])
@pytest.mark.parametrize("k", [2, 3])
def test_input_slices_are_reproduced_at_their_centres(method: str, k: int) -> None:
    mask = three_cells()[::k]
    out = resample_to_isotropic(mask, (float(k), 1.0, 1.0), method=method)
    assert np.array_equal(out[::k], mask)


def test_shape_based_recovers_the_isotropic_geometry_better_than_nearest() -> None:
    truth = three_cells()
    k = 3
    sampled = truth[::k]  # voxel j of the stack has its centre at z = 3 j, as the module's frame says
    shape = resample_to_isotropic(sampled, (float(k), 1.0, 1.0), method="shape")
    nearest = resample_to_isotropic(sampled, (float(k), 1.0, 1.0), method="nearest")
    reference = truth[: shape.shape[0]]
    err_shape = np.count_nonzero(shape != reference)
    err_nearest = np.count_nonzero(nearest != reference)
    # Measured on this fixture: 1 037 against 1 551 voxels wrong (ratio 0.67).
    assert err_shape < 0.75 * err_nearest, (err_shape, err_nearest)
    # Shape-based interpolation shrinks small convex cells a little (linear interpolation of a concave distance
    # profile underestimates it): -1.2 to -4.6 % here, balls of radius 9-14 voxels at k = 3. Pinned at 6 %.
    for label in (1, 2, 3):
        ratio = np.count_nonzero(shape == label) / np.count_nonzero(reference == label)
        assert 0.94 < ratio <= 1.0, (label, ratio)


def test_the_coarse_axis_can_be_any_axis() -> None:
    mask = three_cells()[::2]
    along_0 = resample_to_isotropic(mask, (2.0, 1.0, 1.0))
    along_2 = resample_to_isotropic(np.moveaxis(mask, 0, 2), (1.0, 1.0, 2.0))
    assert np.array_equal(np.moveaxis(along_2, 2, 0), along_0)


def test_physical_units_do_not_matter() -> None:
    mask = three_cells()[::3]
    assert np.array_equal(resample_to_isotropic(mask, (3.0, 1.0, 1.0)), resample_to_isotropic(mask, (0.9, 0.3, 0.3)))
    assert isotropic_spacing((0.9, 0.3, 0.3)) == 0.3


def test_refusals() -> None:
    mask = three_cells()
    with pytest.raises(ValueError, match="two different fine axes"):
        resample_to_isotropic(mask, (3.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="positive finite"):
        resample_to_isotropic(mask, (0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="unknown resampling method"):
        resample_to_isotropic(mask, (2.0, 1.0, 1.0), method="linear")
    with pytest.raises(ValueError, match="3-D"):
        resample_to_isotropic(mask[0], (2.0, 1.0, 1.0))
