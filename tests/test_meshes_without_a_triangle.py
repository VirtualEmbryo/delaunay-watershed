"""A mask whose mesh has no triangle is returned as an empty mesh, and its cells are named; it no longer raises.

Before 0.8.0 a cube of side 3 or 4 voxels at `min_distance` 3 made dw3d raise `numpy.exceptions.AxisError` ("axisa:
axis 1 is out of bounds for array of dimension 1") in `mesh_utilities._compute_normal_faces`: no tetrahedron centroid
falls in so small a cell, so its watershed seed is dropped, every tetrahedron gets label 0, the mesh has no triangle,
and the empty triangle list became a `(0,)` array instead of `(0, 3)`. The same happened to a mask of label 0 only.
These tests raise that `AxisError` on the 0.7 code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dw3d import UnresolvedCellWarning, get_default_mesh_reconstruction_algorithm
from dw3d.enclosed_media import NO_TETRAHEDRON_IN_ITS_VOXELS


def reconstruct(mask: np.ndarray):
    algorithm = get_default_mesh_reconstruction_algorithm(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    return algorithm, mesh, [w.category for w in caught]


@pytest.mark.parametrize("side", [1, 2, 3, 4])
@pytest.mark.parametrize("margin", [4, 8])
def test_a_cell_too_small_for_a_tetrahedron_is_named_not_raised(side, margin) -> None:
    """The mesh is empty with the usual column counts, and the cell is named, with its cause."""
    mask = np.zeros((side + 2 * margin,) * 3, dtype=np.uint16)
    mask[margin : margin + side, margin : margin + side, margin : margin + side] = 1
    algorithm, (points, triangles, labels), caught = reconstruct(mask)
    assert np.shape(triangles) == (0, 3)
    assert np.shape(labels) == (0, 2)
    assert np.shape(points)[1:] == (3,)
    assert UnresolvedCellWarning in caught
    unresolved = {region.label: region for region in algorithm.enclosed_medium_report.unresolved}
    assert unresolved[1].cause == NO_TETRAHEDRON_IN_ITS_VOXELS
    assert unresolved[1].voxels == side**3


def test_two_cells_too_small_for_a_tetrahedron_are_both_named() -> None:
    """Two 4 x 2 x 4 boxes side by side, 4 voxels from every face, which raised `AxisError` on 0.7."""
    mask = np.zeros((12, 12, 12), dtype=np.uint16)
    mask[4:8, 4:6, 4:8] = 1
    mask[4:8, 6:8, 4:8] = 2
    algorithm, (_, triangles, _), caught = reconstruct(mask)
    assert np.shape(triangles) == (0, 3)
    assert UnresolvedCellWarning in caught
    unresolved = algorithm.enclosed_medium_report.unresolved
    assert {region.label for region in unresolved if region.cause == NO_TETRAHEDRON_IN_ITS_VOXELS} == {1, 2}


def test_a_mask_of_medium_only_gives_an_empty_mesh(tmp_path) -> None:
    """A mask with no cell gives an empty mesh, which `save_to_rec_mesh` writes."""
    algorithm, (_, triangles, labels), caught = reconstruct(np.zeros((12, 12, 12), dtype=np.uint16))
    assert np.shape(triangles) == (0, 3)
    assert np.shape(labels) == (0, 2)
    assert algorithm.cell_volume_report.cells == []
    assert UnresolvedCellWarning in caught
    algorithm.save_to_rec_mesh(tmp_path / "empty.rec")
    assert (tmp_path / "empty.rec").stat().st_size > 0


@pytest.mark.parametrize("extension", [".vtk", ".vtu"])
@pytest.mark.parametrize("binary_mode", [False, True])
def test_an_empty_mesh_is_refused_by_save_to_vtk_mesh_with_a_clear_error(tmp_path, extension, binary_mode) -> None:
    """`save_to_vtk_mesh` refuses a mesh without triangles and writes nothing; `save_to_rec_mesh` writes it.

    Before 0.8.0 meshio 5.3 raised `IndexError` writing such a `.vtk` file (binary or not), and wrote `.vtu` files that
    its own reader then failed to read; it cannot read an empty binary legacy `.vtk` file either, so no empty file of
    these formats would come back as a mesh.
    """
    algorithm, _, _ = reconstruct(np.zeros((12, 12, 12), dtype=np.uint16))
    target = tmp_path / f"empty{extension}"
    with pytest.raises(ValueError, match=r"the mesh has no triangle.*save_rec"):
        algorithm.save_to_vtk_mesh(target, binary_mode=binary_mode)
    assert not target.exists()
    algorithm.save_to_rec_mesh(tmp_path / "empty.rec")
    from dw3d.io import load_rec

    points, triangles, labels = load_rec(tmp_path / "empty.rec")
    assert np.shape(points) == (0, 3)
    assert np.shape(triangles) == (0, 3)
    assert np.shape(labels) == (0, 2)
