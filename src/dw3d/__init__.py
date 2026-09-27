"""Main DW3D module."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
# junction-protection work, kept as the baseline that every acceptance criterion for that
# work is stated against.
get_deterministic_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_deterministic_algorithm
# The boundary-layer algorithm (opt-in variant, not the default); see the factory.
get_boundary_layer_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_boundary_layer_algorithm
_factory = MeshReconstructionAlgorithmFactory
# Boundary layer + junction protection, and the only way to reach the junction-protection
# ablation switches. At its defaults it carries a measured regression (+15.7 % on every
# interface area); see the factory.
get_junction_protected_mesh_reconstruction_algorithm = _factory.get_junction_protected_algorithm
# Three exact synonyms of keyword calls on get_junction_protected_mesh_reconstruction_algorithm,
# deprecated since 0.5.0 and removed in 0.6.0; each emits a DeprecationWarning naming its
# replacement. offset_included = the defaults; cubic_score = spline_order=3; offset_excluded =
# exclude_offsets_from_surface=True.
get_offset_included_mesh_reconstruction_algorithm = _factory.get_offset_included_algorithm
get_cubic_score_mesh_reconstruction_algorithm = _factory.get_cubic_score_algorithm
get_offset_excluded_mesh_reconstruction_algorithm = _factory.get_offset_excluded_algorithm
# The "revised meshing" variant: the same exclusion, taken as a link-condition-checked edge
# collapse so that it cannot create non-manifold topology. Was the default for a time; now
# the opt-in variant for when mesh quality matters more than today's downstream
# tension-inference accuracy. See the factory's docstring.
get_link_checked_mesh_reconstruction_algorithm = _factory.get_link_checked_algorithm


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
    "get_dithered_mesh_reconstruction_algorithm",
    "get_deterministic_mesh_reconstruction_algorithm",
    "get_boundary_layer_mesh_reconstruction_algorithm",
    "get_offset_included_mesh_reconstruction_algorithm",
    "get_junction_protected_mesh_reconstruction_algorithm",
    "get_cubic_score_mesh_reconstruction_algorithm",
    "get_offset_excluded_mesh_reconstruction_algorithm",
    "get_link_checked_mesh_reconstruction_algorithm",
    "center_points_around_zero",
    "set_points_pixel_size",
    "set_points_global_min_max",
    "save_mesh_to_rec_mesh",
    "save_mesh_to_vtk_mesh",
    "save_compressed_segmentation",
    "reconstruct_mask_from_dict",
    "reconstruct_mask_from_saved_file_dict",
    "extract_junction_curves",
    "JunctionCurves",
    "JunctionCurve",
    "relocate_junction_vertices",
    "JunctionRelocationReport",
    "CoarseSpacingRelocationWarning",
    "find_self_intersections",
    "count_self_intersections",
)
