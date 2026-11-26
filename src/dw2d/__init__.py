"""Delaunay-Watershed 2D (`dw2d`): build a multimaterial polyline mesh from a 2-D segmentation mask.

The top-level names mirror dw3d's (`get_default_mesh_reconstruction_algorithm`, `MeshReconstructionAlgorithmFactory`,
`MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`, `save_to_rec_mesh`, the compressed segmentation,
`reconstruct_mask_from_dict`), so that dw2d is used exactly like dw3d; dw3d's algorithm also hands a 2-D mask to dw2d.

Conventions used everywhere in the package:

* The mask is a 2-D integer array; label `0` is the medium, every other label a cell.
* A mesh is `(points, segments, labels)`: `points` `(n, 2)` float64 in the mask's index order `(row, column)`, in
  pixels (the centre of pixel `(i, j)` is the point `(i, j)`); `segments` `(m, 2)` int64 vertex indices; `labels`
  `(m, 2)` int64, the two materials each edge separates, smaller first.
* Orientation, one rule with dw3d's (since 0.8.0): each edge is the positively oriented boundary of its first label in
  the index frame. Material `labels[n, 0]` lies on the left of `segments[n, 0] -> segments[n, 1]` in `(row, column)`,
  so the normal `(dc, -dr)` of the edge (its tangent `(dr, dc)` turned clockwise) points out of `labels[n, 0]` and
  into `labels[n, 1]`, as a dw3d triangle's right-hand normal does. A cell alone in the medium therefore has a
  negative signed area, as it has a negative signed volume in 3D (the user guide's page on orientation). Until 0.8.0
  each edge was stored the other way round.
* `min_distance` counts pixels and defaults to 4 (dw3d: 3 voxels).

Names that dw2d 0.1.5 exported and that are not listed in `__all__` (`geometry_reconstruction_2d`, the `.rec`
functions `write_mesh_bin`, `read_rec_file_bin`, ..., the half-edge and plotting helpers of `dw2d.Dcel`) still work
through 0.8.x with a `DeprecationWarning` naming their replacement or their module, and will be removed in 0.9. The
third-party names 0.1.5 leaked by its star imports (`dw2d.np`, `dw2d.plt`, `dw2d.pickle`, ...) are gone.
"""

from __future__ import annotations

import importlib
import warnings

from dw2d.Dcel import DCEL_Data
from dw2d.junction_relocation import relocate_junction_vertices
from dw2d.locus_projection import project_interface_vertices
from dw2d.Mask_reconstruction import reconstruct_mask_from_dict, reconstruct_mask_from_saved_file_dict
from dw2d.reconstruction_algorithm import (
    DEFAULT_MIN_DISTANCE,
    MeshReconstructionAlgorithm,
    save_compressed_segmentation,
    save_mesh_to_rec_mesh,
)
from dw2d.reconstruction_algorithm_factory import (
    MeshReconstructionAlgorithmFactory,
    get_default_mesh_reconstruction_algorithm,
)
from dw2d.reports import (
    EnclosedMediumPocket,
    EnclosedMediumWarning,
    InterfaceProjectionReport,
    JunctionRelocationReport,
    ThinMediumMarginWarning,
    UnresolvedCellWarning,
)

__all__ = (
    "DEFAULT_MIN_DISTANCE",
    "DCEL_Data",
    "EnclosedMediumPocket",
    "EnclosedMediumWarning",
    "InterfaceProjectionReport",
    "JunctionRelocationReport",
    "MeshReconstructionAlgorithm",
    "MeshReconstructionAlgorithmFactory",
    "ThinMediumMarginWarning",
    "UnresolvedCellWarning",
    "get_default_mesh_reconstruction_algorithm",
    "project_interface_vertices",
    "reconstruct_mask_from_dict",
    "reconstruct_mask_from_saved_file_dict",
    "relocate_junction_vertices",
    "save_compressed_segmentation",
    "save_mesh_to_rec_mesh",
)

# Every name dw2d 0.1.5 exported from its top level (its star imports of `dw2d.functions` and `dw2d.Dcel`) that is not
# in `__all__`, with the module that holds it now. Each still resolves through 0.8.x, with a DeprecationWarning.
_DEPRECATED_0_1_5_NAMES = {
    "geometry_reconstruction_2d": "dw2d.functions",
    **dict.fromkeys(
        (
            "Area", "Clean_mesh", "Compute_local_curvature", "Face", "Green_area", "HalfEdge", "Menger_Curvature",
            "Vertex", "change_idx_cells", "compute_angles", "compute_area_faces", "compute_areas", "compute_curvatures",
            "compute_curvatures_vertices", "compute_lengths", "compute_normal_Edges", "create_halfedges_from_edges",
            "find_closed_faces", "find_interfaces", "find_trijunctions", "make_closed_oriented_closed_line",
            "make_vertices_list", "open_mesh_multitracker", "plot_DCEL_bw", "plot_DCEL_polyline", "plot_line",
            "plot_line_no_marker", "plot_lines", "plot_local_curvature", "plot_polylines", "plot_polylines_no_marker",
            "read_rec_file_bin", "read_rec_file_num", "renormalize_verts", "reorient_edges",
            "retrieve_mesh_multimaterial_multitracker_format", "separate_edges", "separate_edges_with_edge_idx",
            "write_mesh_bin", "write_mesh_text",
        ),
        "dw2d.Dcel",
    ),
    "trace_line": "dw2d.Mask_reconstruction",
    "Delaunay_Graph": "dw2d.Graph_functions",
    "build_triangulation": "dw2d.Geometric_utilities",
    "seeded_watershed_map": "dw2d.Networkx_functions",
    **dict.fromkeys(
        (
            "compute_seeds_idx_from_pixel_coords", "plot_polylines_ax", "plot_triangles_coloured_ax",
            "plot_triangulation_ax", "retrieve_border_lines",
        ),
        "dw2d.Plotting_tools",
    ),
}

# The deprecated names that warn when they are used (called or constructed), naming their replacement themselves.
_REPLACEMENTS = frozenset(
    (
        "geometry_reconstruction_2d", "write_mesh_bin", "write_mesh_text", "read_rec_file_bin", "read_rec_file_num",
        "open_mesh_multitracker",
    ),
)


def __getattr__(name: str) -> object:
    """Resolve a name dw2d 0.1.5 exported, with a DeprecationWarning (through 0.8.x; removed in 0.9)."""
    module = _DEPRECATED_0_1_5_NAMES.get(name)
    if module is None:
        error = f"module 'dw2d' has no attribute {name!r}"
        raise AttributeError(error)
    value = getattr(importlib.import_module(module), name)
    if name in _REPLACEMENTS:  # these warn themselves, when called: one warning per use, not two
        return value
    warnings.warn(
        f"dw2d.{name} is deprecated and will be removed in 0.9; import it from {module}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
