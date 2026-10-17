"""After `construct_mesh_in_physical_units`, the compressed segmentation's entries are in the frames documented.

`last_compressed_segmentation` returns the mesh's current points, which this route has scaled to physical units,
with the seeds and the shape of the resampled cubic grid that was meshed. Mixing the two frames is documented rather
than changed (changing the points it returns would change a number users may have stored). Pins: the frames the
docstrings state, that dividing the points by the resampled spacing rebuilds the resampled mask as well as the
cubic-voxel route does, and that both docstrings say so.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dw3d import get_default_mesh_reconstruction_algorithm, reconstruct_mask_from_dict
from dw3d.mask_preprocessing import isotropic_spacing, resample_to_isotropic
from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm

SPACING = (0.6, 0.3, 0.3)


def two_cells_with_long_z(nz: int = 20, n: int = 40) -> np.ndarray:
    z, y, x = np.mgrid[:nz, :n, :n].astype(float)
    z = 2.0 * z  # voxel centres in units of the short spacing
    mask = np.zeros((nz, n, n), dtype=np.uint16)
    mask[(z - 20) ** 2 + (y - 14) ** 2 + (x - 20) ** 2 < 9**2] = 1
    mask[(z - 20) ** 2 + (y - 27) ** 2 + (x - 20) ** 2 < 8**2] = 2
    return mask


@pytest.fixture(scope="module")
def physical_route() -> tuple[MeshReconstructionAlgorithm, np.ndarray]:
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_in_physical_units(two_cells_with_long_z(), SPACING)
    return algorithm, resample_to_isotropic(two_cells_with_long_z(), SPACING, method="shape")


def test_points_are_the_physical_mesh_and_seeds_the_resampled_grid(physical_route) -> None:
    algorithm, cubic = physical_route
    compressed = algorithm.last_compressed_segmentation
    assert compressed["points"] is algorithm.last_constructed_mesh[0]
    assert tuple(compressed["image_shape"]) == cubic.shape
    seeds = np.asarray(compressed["seeds"])
    assert np.issubdtype(seeds.dtype, np.integer)
    assert np.all((seeds >= 0) & (seeds < np.array(cubic.shape)))


def test_points_divided_by_the_resampled_spacing_rebuild_the_resampled_mask(physical_route) -> None:
    algorithm, cubic = physical_route
    compressed = algorithm.last_compressed_segmentation
    h = isotropic_spacing(SPACING)
    rebuilt = reconstruct_mask_from_dict({**compressed, "points": compressed["points"] / h})
    as_is = reconstruct_mask_from_dict(compressed)
    # Measured: 0.962 after the division, 0.074 as they are (the mesh rasterised 0.3 times too small).
    assert np.mean((rebuilt > 0) == (cubic > 0)) > 0.9
    assert np.mean((as_is > 0) == (cubic > 0)) < 0.5


def test_the_docstrings_name_the_frame_of_each_entry() -> None:
    property_doc = MeshReconstructionAlgorithm.last_compressed_segmentation.__doc__ or ""
    assert "resampled_spacing" in property_doc
    assert "physical units" in property_doc
    method_doc = MeshReconstructionAlgorithm.construct_mesh_in_physical_units.__doc__ or ""
    assert "last_compressed_segmentation" in method_doc
    assert "resampled spacing" in (reconstruct_mask_from_dict.__doc__ or "")
