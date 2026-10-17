"""Delaunay-Watershed 3D (`dw3d`): build a multimaterial surface mesh from a 3-D segmentation mask.

The top-level names build a reconstruction algorithm (`get_default_mesh_reconstruction_algorithm`,
`MeshReconstructionAlgorithmFactory`), save and reload its meshes, and read extra information from
the mask (junction curves, crop faces, cell volumes).

Conventions used everywhere in the package:

* The mask is a 3-D integer array; label `0` is the medium (the outside of the tissue), every other
  label is a cell.
* Mesh points are in the mask's index order: column `i` of `points` is the position along the
  mask's axis `i`. For a microscope stack stored as `(z, y, x)`, the order dw3d assumes, the points
  are `(z, y, x)`. `construct_mesh_from_segmentation_mask` returns them in voxel units: the centre
  of voxel `(i, j, k)` is the point `(i, j, k)`.
* `spacing` is the voxel size along the mask's axes 0, 1, 2, so `(dz, dy, dx)` for a `(z, y, x)`
  stack.
* `labels[n] = (a, b)`, with `a < b`, are the two regions triangle `n` separates; its normal
  (right-hand rule on its three vertices) points out of `a` and into `b`.
* `min_distance` (the point placement's minimum distance between points) counts voxels of the grid
  the mask is meshed on.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dw3d.cell_volume_agreement import CellVolumeDisagreementWarning, CellVolumeReport, compare_cell_volumes_with_mask
from dw3d.crop_faces import IMAGE_FACES, CropFaceReport
from dw3d.io import save_rec, save_vtk

# An additional, purely additive read of the mask a mesh was reconstructed from. Nothing
# in the reconstruction pipeline calls it and no default changes; see
# dw3d/junction_curves.py.
from dw3d.junction_curves import JunctionCurve, JunctionCurves, extract_junction_curves

# A post-process on a finished mesh, on by default since 0.5.0: it moves trijunction vertices onto
# the curves the mask itself gives, under a guard that provably cannot create a self-intersection
# anywhere in the mesh. `relocate_junctions=False` reproduces 0.4. See dw3d/junction_relocation.py
# for what it costs and what it was measured to buy.
from dw3d.junction_relocation import (
    CoarseSpacingRelocationWarning,
    JunctionRelocationReport,
    relocate_junction_vertices,
)
from dw3d.mask_preprocessing import UnmeasuredAnisotropyWarning, resample_to_isotropic
from dw3d.mask_reconstruction import reconstruct_mask_from_dict, reconstruct_mask_from_saved_file_dict
from dw3d.mesh_utilities import center_around_origin, set_pixel_size, set_points_min_max
from dw3d.reconstruction_algorithm_factory import MeshReconstructionAlgorithmFactory

# The exact nonlocal triangle-triangle predicate the relocation certifies itself with, and the
# whole-mesh count built on it.
from dw3d.triangle_intersection import count_self_intersections, find_self_intersections

# The default is the dithered configuration, the published dw3d 0.3.6 configuration. The
# default was for a time the junction-protected, link-checked configuration
# (get_link_checked_algorithm); that was reverted because end-to-end tension-inference
# accuracy is worse with the revised meshing than with the dithered one on every method
# measured, despite the revised mesh's better geometry in isolation. The revised pipeline
# stays reachable as get_link_checked_mesh_reconstruction_algorithm, an opt-in variant for
# when mesh quality matters more than today's downstream inference accuracy; see the
# factory's docstring and BENCHMARKS.md at the repository root.
get_default_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_default_algorithm
# The historical algorithm (randomly dithered point placement); this is the default. See the
# factory's docstring.
get_dithered_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_dithered_algorithm
# The deterministic algorithm (no dither): the default before the boundary-layer and
# junction-protection placements were added, kept as the baseline those placements are measured
# against.
get_deterministic_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_deterministic_algorithm
# The boundary-layer algorithm (opt-in variant, not the default); see the factory.
get_boundary_layer_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_boundary_layer_algorithm
_factory = MeshReconstructionAlgorithmFactory
# Boundary layer + junction protection, and the only way to reach the junction-protection
# ablation switches. At its defaults it carries a measured regression (+15.7 % on every
# interface area); see the factory.
get_junction_protected_mesh_reconstruction_algorithm = _factory.get_junction_protected_algorithm
# The "revised meshing" variant: the same exclusion, taken as a link-condition-checked edge
# collapse so that it cannot create non-manifold topology. Was the default for a time; now
# the opt-in variant for when mesh quality matters more than today's downstream
# tension-inference accuracy. See the factory's docstring.
get_link_checked_mesh_reconstruction_algorithm = _factory.get_link_checked_algorithm


def center_points_around_zero(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Translate mesh points so that the centre of their bounding box is the origin.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points.

    Returns:
        NDArray[np.float64]: a new `(n, 3)` array; `points` is not modified.
    """
    return center_around_origin(points)


def set_points_pixel_size(
    points: NDArray[np.float64],
    xy_pixel_size: float,
    z_pixel_size: float,
) -> NDArray[np.float64]:
    """Scale points from voxel coordinates to physical coordinates, given the microscope's xy and z pixel sizes.

    The points are in the mask's `(z, y, x)` index order: column 0 is scaled by `z_pixel_size`, columns 1 and 2 by
    `xy_pixel_size` (see `dw3d.mesh_utilities.set_pixel_size`). Changed in 0.7.0: before, z was multiplied by the
    xy size and x by the z size. For a mask whose voxels are longer along z, prefer
    `MeshReconstructionAlgorithm.construct_mesh_in_physical_units`, which is more accurate.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points in voxel units, `(z, y, x)` order.
        xy_pixel_size (float): the pixel size in the image plane, which multiplies columns 1 and 2.
        z_pixel_size (float): the z step, which multiplies column 0.

    Returns:
        NDArray[np.float64]: a new `(n, 3)` array; `points` is not modified.
    """
    return set_pixel_size(points, xy_pixel_size, z_pixel_size)


def set_points_global_min_max(points: NDArray[np.float64], global_min: float, global_max: float) -> NDArray[np.float64]:
    """Scale and shift points, by one factor on every axis, so their coordinates span `[global_min, global_max]`.

    The minimum and maximum are taken over all coordinates of all points together, not per axis, so
    the shape is kept: the smallest coordinate becomes `global_min` and the largest `global_max`.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points.
        global_min (float): the new smallest coordinate.
        global_max (float): the new largest coordinate.

    Returns:
        NDArray[np.float64]: a new `(n, 3)` array; `points` is not modified.
    """
    return set_points_min_max(points, global_min, global_max)


def save_mesh_to_rec_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    filename: str | Path,
    binary_mode: bool = False,
    crop_faces: NDArray[np.bool_] | None = None,
) -> None:
    """Save a mesh on disk in the rec format.

    `crop_faces` (the algorithm's `crop_faces`), when given, puts the crop faces in the last block
    of triangle rows and records the block in the file's appendix; see `dw3d.io.save_rec`. No other
    metadata is written: `MeshReconstructionAlgorithm.save_to_rec_mesh` also records the physical
    frame and the physical image faces.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points, written as given.
        triangles (NDArray[np.int64]): `(m, 3)` vertex indices.
        labels (NDArray[np.int64]): `(m, 2)` the two regions each triangle separates.
        filename (str | Path): where to write the file.
        binary_mode (bool, optional): write the binary variant of the format. Defaults to False.
        crop_faces (NDArray[np.bool_] | None, optional): `(m,)` bool, `True` on the crop faces.
            Defaults to None, which writes no appendix.
    """
    save_rec(filename, points, triangles, labels, binary_mode, crop_faces=crop_faces)


def save_mesh_to_vtk_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    filename: str | Path,
    binary_mode: bool = False,
    crop_faces: NDArray[np.bool_] | None = None,
) -> None:
    """Save a mesh on disk in the vtk format (or any format meshio infers from the file extension).

    `crop_faces`, when given, is written as the cell array `crop_face`; see `dw3d.io.save_vtk`.

    Args:
        points (NDArray[np.float64]): `(n, 3)` mesh points, written as given.
        triangles (NDArray[np.int64]): `(m, 3)` vertex indices.
        labels (NDArray[np.int64]): `(m, 2)` the two regions each triangle separates, written as the
            cell arrays `label1` and `label2`.
        filename (str | Path): where to write the file.
        binary_mode (bool, optional): write a binary file. Defaults to False.
        crop_faces (NDArray[np.bool_] | None, optional): `(m,)` bool, `True` on the crop faces.
            Defaults to None.
    """
    save_vtk(filename, points, triangles, labels, binary_mode, crop_faces=crop_faces)


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str]) -> None:
    """Save a compressed segmentation on disk with `numpy.save`.

    The dictionary is pickled into a `.npy` file (numpy adds the extension when `filename` has
    none). Read it back with `reconstruct_mask_from_saved_file_dict`.

    Args:
        filename (str | Path): where to write the file.
        compressed_segmentation (dict[str]): the dictionary
            `MeshReconstructionAlgorithm.last_compressed_segmentation` returns.
    """
    np.save(filename, compressed_segmentation)


__all__ = (
    "MeshReconstructionAlgorithmFactory",
    "get_default_mesh_reconstruction_algorithm",
    "get_dithered_mesh_reconstruction_algorithm",
    "get_deterministic_mesh_reconstruction_algorithm",
    "get_boundary_layer_mesh_reconstruction_algorithm",
    "get_junction_protected_mesh_reconstruction_algorithm",
    "get_link_checked_mesh_reconstruction_algorithm",
    "center_points_around_zero",
    "set_points_pixel_size",
    "set_points_global_min_max",
    "save_mesh_to_rec_mesh",
    "save_mesh_to_vtk_mesh",
    "save_compressed_segmentation",
    "reconstruct_mask_from_dict",
    "reconstruct_mask_from_saved_file_dict",
    "resample_to_isotropic",
    "UnmeasuredAnisotropyWarning",
    "extract_junction_curves",
    "JunctionCurves",
    "JunctionCurve",
    "relocate_junction_vertices",
    "JunctionRelocationReport",
    "CoarseSpacingRelocationWarning",
    "compare_cell_volumes_with_mask",
    "CellVolumeReport",
    "CellVolumeDisagreementWarning",
    "CropFaceReport",
    "IMAGE_FACES",
    "find_self_intersections",
    "count_self_intersections",
)
