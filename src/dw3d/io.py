"""IO for .rec meshes. IO for .vtk/.vtu meshes.

Copy of private iorec package v1.0.5 for rec part.
Matthieu Perez 2024
"""

from __future__ import annotations

import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import meshio
import numpy as np
from numpy.typing import NDArray

# The `.rec` metadata appendix. See CHANGELOG.md / README.md for the format. The
# appendix is a trailing block of ASCII `#`-prefixed lines -- see `write_rec_appendix` for
# why it must stay trailing, never leading.
_APPENDIX_MAGIC = "rec-appendix"
_APPENDIX_VERSION = 1
_APPENDIX_SEQUENCE_INT_KEYS = frozenset({"source_shape", "crop_offset"})
_APPENDIX_SEQUENCE_FLOAT_KEYS = frozenset({"spacing"})


def _dw3d_version() -> str:
    """Return the installed dw3d version, or "unknown" if it can't be determined.

    The importable module is `dw3d`, but the distribution is registered as
    `delaunay_watershed_3d` (see `setup.cfg`), which is the name `importlib.metadata`
    needs.
    """
    try:
        return version("delaunay_watershed_3d")
    except PackageNotFoundError:
        return "unknown"


def load_meshio_mesh(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a mesh file using meshio. Error if there are no labels data in the mesh.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the mesh has no "labels" data saved alongside points and triangles.
        meshio.ReadError: If meshio can't read the file for any reason.

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (ulonglong)
        - labels: np.array of shape (num_triangles, 2) (uint)
    """
    filename = Path(filename).resolve()
    mesh: meshio.Mesh = meshio.read(filename)

    label1 = mesh.get_cell_data("label1", "triangle")
    label2 = mesh.get_cell_data("label2", "triangle")
    labels = np.stack((label1, label2), axis=-1)

    if len(labels) == 0:
        error = f"{filename} has no labels data."
        raise ValueError(error)

    points = mesh.points
    triangles = mesh.get_cells_type("triangle")

    return (points, triangles, labels)


def save_vtk(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): Mesh points.
        triangles (NDArray[np.int64]): Mesh triangles (topology).
        labels (NDArray[np.int64]): Triangles multimaterial labels.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
    """
    cells = [
        ("triangle", triangles),
    ]
    cell_data = {
        "label1": [
            labels[:, 0],
        ],
        "label2": [
            labels[:, 1],
        ],
    }

    mesh = meshio.Mesh(
        points,
        cells,
        # Each item in cell data must match the cells array
        cell_data=cell_data,
    )

    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
        binary=binary_mode,
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )


def load_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a rec mesh file.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (ulonglong)
        - labels: np.array of shape (num_triangles, 2) (uint)
    """
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix == ".rec" or filename.suffix == ".arec":
        return _load_rec_file(filename)
    else:
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)


def _load_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if _is_binary(filename):
        return _load_binary_rec_file(filename)
    else:
        return _load_text_rec_file(filename)


def _load_binary_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        (number_of_points,) = struct.unpack("Q", f.read(8))
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(f, count=3 * number_of_points, dtype=np.float64).reshape(
            (number_of_points, 3),
        )

        # Number of triangles
        (number_of_triangles,) = struct.unpack("Q", f.read(8))
        # Each line defines a triangle and 2 regions labels
        dtype_triangles_and_labels = np.dtype(
            [("triangles", np.int64, (3,)), ("labels", np.int32, (2,))],
        )
        triangles_and_labels = np.fromfile(
            f,
            count=number_of_triangles,
            dtype=dtype_triangles_and_labels,
        )

    return (
        points,
        triangles_and_labels["triangles"].astype(np.int64),
        triangles_and_labels["labels"].astype(np.int64),
    )


def _load_text_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename) as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(
            f,
            count=3 * number_of_points,
            sep="\n",
            dtype=np.float64,
        ).reshape((number_of_points, 3))

        # Number of triangles
        number_of_triangles = int(f.readline().strip())
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = np.fromfile(
            f,
            count=5 * number_of_triangles,
            sep="\n",
            dtype=np.int64,
        ).reshape(
            (number_of_triangles, 5),
        )

    return (
        points,
        triangles_and_labels[:, 0:3].astype(np.int64),
        triangles_and_labels[:, 3:5].astype(np.int64),
    )


def _format_appendix_value(value: object) -> str:
    if isinstance(value, (tuple, list, np.ndarray)):
        return " ".join(str(v) for v in value)
    return str(value)


def write_rec_appendix(metadata: dict[str, Any], *, written_by: str) -> bytes:
    """Build the trailing `.rec`/`.arec` metadata appendix block.

    The block is a sequence of ASCII `#`-prefixed `key = value` lines, headed by the
    `rec-appendix <version>` line. It must be appended at the very end of the file: a
    trailing `#` block is invisible to every reader that predates it (both text and
    binary mode), since those readers read a count and then `np.fromfile(..., count=n)`,
    which stops exactly there. A leading block breaks both readers. See
    `tests/test_rec_appendix.py::test_legacy_reader_ignores_trailing_appendix`.

    Args:
        metadata: appendix fields, e.g. `coordinate_frame`, `axis_order`, `spacing`,
            `spacing_unit`, `source_shape`, `crop_offset`. Sequence-valued fields
            (`spacing`, `source_shape`, `crop_offset`) are written space-separated.
        written_by: the dw3d version string that produced this file. Stamped by the
            writer -- the appendix is never hand-edited.

    Returns:
        bytes: the ASCII appendix block (including a trailing newline), ready to be
        concatenated onto the geometry bytes/text already written.
    """
    lines = [f"# {_APPENDIX_MAGIC} {_APPENDIX_VERSION}", f"# written_by = {written_by}"]
    for key, raw_value in metadata.items():
        lines.append(f"# {key} = {_format_appendix_value(raw_value)}")
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_rec_appendix(block_text: str) -> dict[str, Any]:
    """Parse a `.rec`/`.arec` metadata appendix block into a dict.

    Args:
        block_text: the appendix block: one `# key = value` line per line, headed by the
            `# rec-appendix <version>` line.

    Raises:
        ValueError: if `block_text` does not start with a `rec-appendix <version>` header.

    Returns:
        dict[str, Any]: parsed fields, plus `"_version"` (the schema version, an int).
        Sequence-valued fields known to the schema (`spacing`, `source_shape`,
        `crop_offset`) are parsed into tuples of `float`/`int`; unknown keys are kept as
        raw strings, never dropped, so a future rewrite can carry them forward.
    """
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    header_body = header[1:].strip() if header.startswith("#") else ""
    header_parts = header_body.split()
    if len(header_parts) != 2 or header_parts[0] != _APPENDIX_MAGIC:
        error = f"Not a rec-appendix block, expected a header line, got: {header!r}"
        raise ValueError(error)

    metadata: dict[str, Any] = {"_version": int(header_parts[1])}
    for line in lines[1:]:
        content = line[1:].strip() if line.startswith("#") else line
        key, _, raw_value = content.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if key in _APPENDIX_SEQUENCE_INT_KEYS:
            metadata[key] = tuple(int(v) for v in raw_value.split())
        elif key in _APPENDIX_SEQUENCE_FLOAT_KEYS:
            metadata[key] = tuple(float(v) for v in raw_value.split())
        else:
            metadata[key] = raw_value
    return metadata


def load_rec_appendix(filename: str | Path) -> dict[str, Any] | None:
    """Read the trailing metadata appendix from a `.rec`/`.arec` file, if present.

    Works uniformly for text- and binary-mode files: the appendix is located by finding
    the last `# rec-appendix ` marker in the file. It is written only ever as the literal
    first line of the trailing appendix block (see `write_rec_appendix`), so its last
    occurrence marks the start of the block -- note that binary-mode payloads have no
    guaranteed newline before this marker, so the search cannot require one.

    Args:
        filename (str | Path): the path to the file on disk.

    Returns:
        dict[str, Any] | None: the parsed appendix fields (see `parse_rec_appendix`), or
        `None` if the file has no appendix. A `None` return must be treated by the caller
        as `coordinate_frame == "unknown"`, never as isotropic spacing -- a legacy file's
        absent appendix carries no information about spacing at all.
    """
    filename = Path(filename).resolve()
    raw = filename.read_bytes()
    marker = f"# {_APPENDIX_MAGIC} ".encode("ascii")
    idx = raw.rfind(marker)
    if idx == -1:
        return None
    return parse_rec_appendix(raw[idx:].decode("ascii"))


def _is_binary(file_name: Path) -> bool:
    try:
        with Path.open(file_name, encoding="utf-8") as f:
            f.read()
            return False
    except UnicodeDecodeError:
        return True


def save_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): Mesh points.
        triangles (NDArray[np.int64]): Mesh triangles (topology).
        labels (NDArray[np.int64]): Triangles multimaterial labels.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
        metadata (dict[str, Any] | None, optional): appendix fields (`coordinate_frame`,
            `axis_order`, `spacing`, `spacing_unit`, `source_shape`, `crop_offset`, ...).
            Defaults to `None`, which writes nothing extra -- byte-for-byte the same file
            as before this option existed. When supplied, a trailing metadata appendix is
            appended after the geometry data; see `write_rec_appendix`.
    """
    if binary_mode:
        _save_rec_binary(filename, points, triangles, labels)
    else:
        _save_rec_text(filename, points, triangles, labels)

    if metadata is not None:
        appendix_bytes = write_rec_appendix(metadata, written_by=_dw3d_version())
        with Path(filename).open("ab") as file:
            file.write(appendix_bytes)


def _save_rec_binary(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Binary only."""
    bytes_content = struct.pack("Q", len(points))
    bytes_content += points.flatten().tobytes()
    bytes_content += struct.pack("Q", len(triangles))
    dt = np.dtype([("triangles", np.int64, (3,)), ("labels", np.int32, (2,))])

    def map_tri_labels_func(
        i: int,
    ) -> tuple[Any, Any]:
        # -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64]:
        return triangles[i], labels[i]

    triangles_and_labels = np.fromiter(
        (map_tri_labels_func(xi) for xi in np.arange(len(triangles))),
        dtype=dt,
        count=len(triangles),
    )

    bytes_content += triangles_and_labels.tobytes()
    with Path(filename).open("wb") as file:
        file.write(bytes_content)


def _save_rec_text(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Text only."""
    content = ""
    content += str(len(points)) + "\n"

    for i in range(len(points)):
        # content += f"{points[i][0]:.5f} {points[i][1]:.5f} {points[i][2]:.5f}\n"
        content += f"{points[i][0]} {points[i][1]} {points[i][2]}\n"

    content += str(len(triangles)) + "\n"

    for i in range(len(triangles)):
        content += f"{triangles[i][0]} {triangles[i][1]} {triangles[i][2]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"

    with Path(filename).open("w") as file:
        file.write(content)
