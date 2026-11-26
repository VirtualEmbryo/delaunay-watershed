"""Read and write dw2d meshes in the 2-D `.rec` format, through iorec. The names and arrays mirror `dw3d.io`'s.

A mesh is `(points, segments, labels)`, as `dw2d.MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`
returns it: `points` `(n, 2)` float64, one point per row, in the mask's index order `(row, column)` (as dw3d's points
are in the mask's `(z, y, x)`), in pixels unless built in physical units; `segments` `(m, 2)` int64, the two vertex
indices of each edge; `labels` `(m, 2)` int64, the two materials each edge separates, smaller first (0 is the medium).
The files store the points as given.

The file layout is iorec's 2-D `.rec` (`iorec.load_2d_rec`, `iorec.save_2d_rec`). Binary: the number of points
(uint64), the points (2 float64 each), the number of edges (uint64), then per edge its 2 vertex indices (int64) and
its 2 materials (int32), little-endian. Text: the point count, one line of 2 coordinates per point, the edge count, one
`v1 v2 m1 m2` line per edge; every value is written exactly (the shortest text that reads back to the same float64).
Either layout may be followed by the metadata appendix (version 1, as dw3d writes it), which records `dimension = 2`
when written by dw2d's algorithm, so that a reader can tell a 2-D file from a 3-D one without probing its layout
(`dw3d.io.rec_dimension`), and, since 0.8.0, `orientation = boundary_of_first_label` (`ORIENTATION_RECORDED`): each
edge is the positively oriented boundary of its first label in `(row, column)`, the rule of dw3d's triangles.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import iorec
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "APPENDIX_KEY_TYPES",
    "APPENDIX_VERSION_WRITTEN",
    "ORIENTATION_RECORDED",
    "load_rec",
    "load_rec_appendix",
    "save_rec",
]

#: The appendix version dw2d writes: 1, as dw3d (version 2 loses every key in delaunay-watershed and foambryo 0.7.1).
APPENDIX_VERSION_WRITTEN = 1

#: The orientation rule a mesh written by dw2d's algorithm follows, recorded as the appendix key `orientation`: each
#: edge is the positively oriented boundary of its first label in index order `(row, column)` (the first label lies on
#: the left of `v1 -> v2`), dw3d's rule for triangles. Files written before 0.8.0 do not carry the key, and their edges
#: are stored the other way round.
ORIENTATION_RECORDED = "boundary_of_first_label"

#: The type of dw2d's own appendix keys in a version-1 appendix, where values are text.
APPENDIX_KEY_TYPES: dict[str, object] = {
    "dimension": int,
    "spacing": tuple[float, ...],
    "source_shape": tuple[int, ...],
    "image_border_vertices": tuple[int, ...],
    "unresolved_regions": tuple[int, ...],
}


def _dw2d_version() -> str:
    """The installed dw2d version (distribution `delaunay-watershed-2d`), or "unknown"."""
    try:
        return version("delaunay-watershed-2d")
    except PackageNotFoundError:
        return "unknown"


def load_rec(filename: str | Path) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a 2-D `.rec` (or `.arec`) mesh file, binary or text (detected from the content).

    A trailing metadata appendix, if any, is ignored; read it with `load_rec_appendix`.

    Args:
        filename (str | Path): the path to the file on disk.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the path is not a `.rec` or `.arec` file.

    Returns:
        tuple: `(points, segments, labels)`: points `(n, 2)` float64, segments `(m, 2)` int64 and labels `(m, 2)` int64.
    """
    points, segments, labels = iorec.load_2d_rec(filename)
    return points, np.asarray(segments, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def load_rec_appendix(filename: str | Path) -> dict[str, Any] | None:
    """Read the trailing metadata appendix of a `.rec`/`.arec` file, if present, with dw2d's keys typed.

    Returns:
        dict[str, Any] | None: the appendix fields (`dimension` an int, `spacing` a tuple of floats, ...), or `None`
            when the file has none.
    """
    return iorec.load_rec_appendix(filename, metadata_types=APPENDIX_KEY_TYPES)


def save_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a 2-D mesh to a `.rec` file, binary or text.

    Binary mode is the more compact; text mode can be read by eye. The layouts are in this module's docstring.

    Args:
        filename (str | Path): where to write the file.
        points (NDArray[np.float64]): `(n, 2)` vertex coordinates, written as given (float64).
        segments (NDArray[np.int64]): `(m, 2)` vertex indices of each edge.
        labels (NDArray[np.int64]): `(m, 2)` the two materials each edge separates.
        binary_mode (bool, optional): write the binary layout. Defaults to False (text).
        metadata (dict[str, Any] | None, optional): appendix fields (`dimension`, `axis_order`, `coordinate_frame`,
            `spacing`, `spacing_unit`, `image_border_vertices`, ...), written as a version-1 appendix after the
            geometry. Defaults to None, which writes no appendix (the file dw2d 0.1.5 wrote, byte for byte, in binary).

    Raises:
        ValueError: if `points` is not `(n, 2)`, or `segments` or `labels` not `(m, 2)`.
    """
    points = np.asarray(points)
    segments = np.asarray(segments)
    labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] != 2 or segments.shape != labels.shape or segments.ndim != 2 or segments.shape[1] != 2:
        error = (
            "expected points of shape (n, 2), segments and labels of shape (m, 2), got "
            f"{points.shape}, {segments.shape} and {labels.shape}"
        )
        raise ValueError(error)
    if metadata is None:
        iorec.save_2d_rec(filename, points, segments, labels, binary_mode)
        return
    iorec.save_2d_rec(
        filename,
        points,
        segments,
        labels,
        binary_mode,
        metadata=metadata,
        written_by=_dw2d_version(),
        appendix_version=APPENDIX_VERSION_WRITTEN,
    )
