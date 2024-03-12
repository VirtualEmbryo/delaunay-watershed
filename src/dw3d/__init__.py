"""Main DW3D module."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dw3d.io import save_rec, save_vtk
from dw3d.mask_reconstruction import reconstruct_mask_from_dict, reconstruct_mask_from_saved_file_dict
from dw3d.mesh_utilities import center_around_origin, set_pixel_size, set_points_min_max
from dw3d.reconstruction_algorithm_factory import MeshReconstructionAlgorithmFactory

get_default_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_default_algorithm


def center_points_around_zero(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center mesh points around 0."""
    return center_around_origin(points)


def set_points_pixel_size(
    points: NDArray[np.float64],
    xy_pixel_size: float,
    z_pixel_size: float,
) -> NDArray[np.float64]:
    """Scales the points from pixel coordinates to real coordinates, given microscope's xy and z pixel size."""
    return set_pixel_size(points, xy_pixel_size, z_pixel_size)


def set_points_global_min_max(points: NDArray[np.float64], global_min: float, global_max: float) -> NDArray[np.float64]:
    """Scales homogeneously the points such that their min & max values are global_min and global_max."""
    return set_points_min_max(points, global_min, global_max)


def save_mesh_to_rec_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    filename: str | Path,
    binary_mode: bool = False,
) -> None:
    """Save the output mesh on disk in the rec format."""
    save_rec(filename, points, triangles, labels, binary_mode)


def save_mesh_to_vtk_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    filename: str | Path,
    binary_mode: bool = False,
) -> None:
    """Save the output mesh on disk in the vtk format."""
    save_vtk(filename, points, triangles, labels, binary_mode)


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str]) -> None:
    """Save a compressed segmentation on disk with numpy.save."""
    np.save(filename, compressed_segmentation)


__all__ = (
    "MeshReconstructionAlgorithmFactory",
    "get_default_mesh_reconstruction_algorithm",
    "center_points_around_zero",
    "set_points_pixel_size",
    "set_points_global_min_max",
    "save_mesh_to_rec_mesh",
    "save_mesh_to_vtk_mesh",
    "save_compressed_segmentation",
    "reconstruct_mask_from_dict",
    "reconstruct_mask_from_saved_file_dict",
)
