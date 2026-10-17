"""`MeshReconstructionAlgorithm.construct_mesh_in_physical_units` on a small synthetic mask.

Pins: for cubic voxels of size 1 it is bitwise the ordinary reconstruction (nothing changes for an isotropic
input); otherwise its points are the reconstruction of the resampled mask times the short spacing, exactly; the
`.rec` appendix records the physical frame, a later ordinary reconstruction clears it, and an ordinary mesh's file
has no appendix at all (byte for byte what it was before this method existed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_rec, load_rec_appendix
from dw3d.mask_preprocessing import resample_to_isotropic


def three_cells(n: int = 61) -> np.ndarray:
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[(z - 30) ** 2 + (y - 22) ** 2 + (x - 30) ** 2 < 14**2] = 1
    mask[(z - 27) ** 2 + (y - 40) ** 2 + (x - 30) ** 2 < 12**2] = 2
    mask[(z - 34) ** 2 + (y - 31) ** 2 + (x - 44) ** 2 < 9**2] = 3
    return mask


def test_cubic_unit_voxels_give_the_ordinary_mesh_bitwise() -> None:
    mask = three_cells()
    ordinary = get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask)
    physical = get_default_mesh_reconstruction_algorithm().construct_mesh_in_physical_units(mask, (1.0, 1.0, 1.0))
    for a, b in zip(ordinary, physical, strict=True):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b)


def test_points_are_the_resampled_reconstruction_times_the_short_spacing() -> None:
    stack = three_cells()[::3]
    spacing = (0.9, 0.3, 0.3)
    expected = get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(
        resample_to_isotropic(stack, spacing),
    )
    points, triangles, labels = get_default_mesh_reconstruction_algorithm().construct_mesh_in_physical_units(
        stack,
        spacing,
    )
    assert np.array_equal(points, expected[0] * 0.3)
    assert np.array_equal(triangles, expected[1])
    assert np.array_equal(labels, expected[2])


def test_the_appendix_records_the_physical_frame_and_a_later_reconstruction_clears_it(tmp_path: Path) -> None:
    stack = three_cells()[::3]
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_in_physical_units(stack, (0.9, 0.3, 0.3), spacing_unit="um")
    algorithm.save_to_rec_mesh(tmp_path / "physical.rec")
    appendix = load_rec_appendix(tmp_path / "physical.rec")
    assert appendix["coordinate_frame"] == "physical"
    assert appendix["spacing"] == (0.9, 0.3, 0.3)
    assert appendix["spacing_unit"] == "um"
    assert appendix["source_shape"] == stack.shape
    assert appendix["resampling"] == "shape"
    assert appendix["resampled_spacing"] == 0.3
    points, _, _ = load_rec(tmp_path / "physical.rec")
    assert np.allclose(points, algorithm.last_constructed_mesh[0])

    algorithm.construct_mesh_from_segmentation_mask(three_cells())
    algorithm.save_to_rec_mesh(tmp_path / "ordinary.rec")
    assert load_rec_appendix(tmp_path / "ordinary.rec") is None


def test_an_ordinary_mesh_file_is_unchanged(tmp_path: Path) -> None:
    algorithm = get_default_mesh_reconstruction_algorithm()
    points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(three_cells())
    algorithm.save_to_rec_mesh(tmp_path / "via_method.rec")
    from dw3d.io import save_rec

    save_rec(tmp_path / "direct.rec", points, triangles, labels)
    assert (tmp_path / "via_method.rec").read_bytes() == (tmp_path / "direct.rec").read_bytes()


def test_resample_to_isotropic_is_exported_at_the_top_level() -> None:
    import dw3d
    from dw3d import mask_preprocessing

    assert dw3d.resample_to_isotropic is mask_preprocessing.resample_to_isotropic
    assert dw3d.UnmeasuredAnisotropyWarning is mask_preprocessing.UnmeasuredAnisotropyWarning
    assert {"resample_to_isotropic", "UnmeasuredAnisotropyWarning"} <= set(dw3d.__all__)


def test_a_ratio_above_three_warns_and_still_meshes() -> None:
    import warnings

    from dw3d import UnmeasuredAnisotropyWarning

    stack = three_cells()[::4]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        points, _, _ = get_default_mesh_reconstruction_algorithm().construct_mesh_in_physical_units(
            stack,
            (4.0, 1.0, 1.0),
        )
    assert sum(issubclass(w.category, UnmeasuredAnisotropyWarning) for w in record) == 1
    assert len(points) > 0


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (3.0, 1.0, 1.0), (0.9, 0.3, 0.3), (3.0 * (1 + 1e-9), 1.0, 1.0)])
def test_a_ratio_up_to_three_does_not_warn(spacing: tuple[float, float, float]) -> None:
    import warnings

    from dw3d import UnmeasuredAnisotropyWarning

    stack = three_cells()[:: round(spacing[0] / spacing[1])]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        get_default_mesh_reconstruction_algorithm().construct_mesh_in_physical_units(stack, spacing)
    assert not any(issubclass(w.category, UnmeasuredAnisotropyWarning) for w in record)
