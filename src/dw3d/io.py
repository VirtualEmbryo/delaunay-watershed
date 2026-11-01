"""Read and write dw3d meshes: the `.rec` format with its metadata appendix, and `.vtk`/`.vtu` through meshio.

A mesh is `(points, triangles, labels)`: `(n, 3)` points, `(m, 3)` vertex indices and `(m, 2)` labels,
the two regions each triangle separates (label 0 is the medium). The files store the points as given:
these functions neither scale nor reorder their columns. A file therefore keeps the mask's order, `(z, y, x)` for a
microscope stack; a program that reads its columns as `(x, y, z)`, as ParaView, MeshLab and other VTK readers do,
shows the mirror image of the tissue, unless it exchanges columns 0 and 2 (and two corners of every triangle, so
that each still faces the same side), as `dw3d.viewing.plot_cells_polyscope` does for what it draws.

The `.rec` reader and writer are copied from iorec, version 1.0.5.
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
_APPENDIX_SEQUENCE_INT_KEYS = frozenset(
    {"source_shape", "crop_offset", "crop_faces", "media", "enclosed_media_voxels", "unresolved_regions",
     "unresolved_regions_voxels"},
)
_APPENDIX_SEQUENCE_FLOAT_KEYS = frozenset({"spacing", "enclosed_media_inscribed_radius"})
_APPENDIX_SCALAR_FLOAT_KEYS = frozenset({"resampled_spacing"})


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
    """Read a mesh file using meshio; the labels come from the triangle cell arrays `label1` and `label2`.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the file has no `label1` or `label2` triangle cell array, or if those arrays are
            empty. (Before 0.7.0 a missing array raised meshio's `KeyError`.)
        meshio.ReadError: If meshio can't read the file for any reason.

    Returns:
        tuple[np.array, np.array, np.array]: `(points, triangles, labels)`: points `(num_points, 3)`,
            triangles `(num_triangles, 3)` and labels `(num_triangles, 2)`, with the dtypes stored in
            the file (float64 and int64 for a file written by `save_vtk` from dw3d's own arrays).
            A `crop_face` array is not read; use `load_meshio_mesh_with_crop_faces`.
    """
    filename = Path(filename).resolve()
    mesh: meshio.Mesh = meshio.read(filename)

    try:
        label1 = mesh.get_cell_data("label1", "triangle")
        label2 = mesh.get_cell_data("label2", "triangle")
    except KeyError as missing:
        error = f"{filename} has no labels data: no triangle cell array {missing} (label1 and label2 are needed)."
        raise ValueError(error) from missing
    labels = np.stack((label1, label2), axis=-1)

    if len(labels) == 0:
        error = f"{filename} has no labels data."
        raise ValueError(error)

    points = mesh.points
    triangles = mesh.get_cells_type("triangle")

    return (points, triangles, labels)


def load_meshio_mesh_with_crop_faces(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_] | None]:
    """`load_meshio_mesh`, plus the crop faces a `crop_face` cell array records.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the file has no `label1` or `label2` triangle cell array, or if those arrays are empty.

    Returns:
        tuple: `(points, triangles, labels, crop_faces)`; `crop_faces` is `(n_triangles,)` bool,
            or `None` when the file has no `crop_face` array (a file written before crop faces were
            recorded, or by another program). `None` says nothing about whether the mesh was cropped.
    """
    points, triangles, labels = load_meshio_mesh(filename)
    mesh: meshio.Mesh = meshio.read(Path(filename).resolve())
    try:
        crop = mesh.get_cell_data("crop_face", "triangle")
    except KeyError:
        return (points, triangles, labels, None)
    return (points, triangles, labels, np.asarray(crop) != 0)


def save_vtk(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    crop_faces: NDArray[np.bool_] | None = None,
) -> None:
    """Save a vtk mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood. meshio picks the format
    from the file extension (`.vtk`, `.vtu`, ...). The labels are written as the triangle cell
    arrays `label1` and `label2`.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): `(n, 3)` mesh points, written as given.
        triangles (NDArray[np.int64]): `(m, 3)` mesh triangles (topology).
        labels (NDArray[np.int64]): `(m, 2)` the two regions each triangle separates.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
        crop_faces (NDArray[np.bool_] | None, optional): `(n_triangles,)`, `True` on the crop
            faces (see `dw3d.crop_faces`). When given, written as the cell array `crop_face`
            (1 on a crop face, 0 elsewhere) beside `label1` and `label2`. Defaults to `None`,
            which writes the file exactly as before this option existed.

    Raises:
        ValueError: if `crop_faces` is not a boolean array with one entry per triangle.
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
    if crop_faces is not None:
        crop_faces = _require_crop_mask(crop_faces, len(triangles))
        cell_data["crop_face"] = [crop_faces.astype(np.int64)]

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
    """Read a rec mesh file, text or binary (detected from the content).

    A metadata appendix, if any, is ignored; read it with `load_rec_appendix`. Crop faces are
    returned as ordinary triangles, in the file's row order; read them with
    `load_rec_with_crop_faces`.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: `(points, triangles, labels)`: points
            `(num_points, 3)` float64, triangles `(num_triangles, 3)` int64 and labels
            `(num_triangles, 2)` int64.
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


def load_rec_with_crop_faces(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_] | None]:
    """Read a rec mesh file and the crop faces its metadata appendix records.

    A file written by `MeshReconstructionAlgorithm.save_to_rec_mesh`, or by `save_rec` with
    `crop_faces`, holds its crop faces as the last block of triangle rows and records the block
    in the appendix as `crop_faces = <first row> <count>`. This reads that range back as a mask
    over the triangle rows, in the file's own row order.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: if the appendix's range is not the last block of the file's triangles, if
            a crop face does not carry the medium label 0, or as `load_rec` (not a .rec or .arec
            file).
        FileNotFoundError: if the file doesn't exist.

    Returns:
        tuple: `(points, triangles, labels, crop_faces)`, the first three as `load_rec` returns
            them; `crop_faces` is `(n_triangles,)` bool, or `None` when the file records no
            `crop_faces` key (a file written before crop faces were recorded, or by another program).
            `None` says nothing about whether the mesh was cropped.
    """
    points, triangles, labels = load_rec(filename)
    appendix = load_rec_appendix(filename)
    if appendix is None or "crop_faces" not in appendix:
        return (points, triangles, labels, None)
    return (points, triangles, labels, crop_faces_from_appendix_range(appendix["crop_faces"], labels))


def crop_faces_from_appendix_range(
    crop_range: tuple[int, ...],
    labels: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Turn the appendix's `crop_faces = <first row> <count>` into a mask over the triangle rows.

    Args:
        crop_range: `(first, count)` as parsed from the appendix.
        labels: `(n_triangles, 2)` the file's labels, to check the range against.

    Raises:
        ValueError: if the range is not two integers ending at the last row, or if a row in it
            does not carry the medium label 0.

    Returns:
        NDArray[np.bool_]: `(n_triangles,)`, `True` on rows `first` to `first + count - 1`.
    """
    n_triangles = len(labels)
    if len(crop_range) != 2:
        error = f"crop_faces must be '<first row> <count>', got {crop_range!r}"
        raise ValueError(error)
    first, count = (int(v) for v in crop_range)
    if count < 0 or first < 0 or first + count != n_triangles:
        error = (
            f"crop_faces = {first} {count} is not the last block of the file's {n_triangles} triangles; "
            "the file is inconsistent with its appendix"
        )
        raise ValueError(error)
    mask = np.zeros(n_triangles, dtype=bool)
    mask[first:] = True
    if count and not np.all(np.asarray(labels)[mask].min(axis=1) == 0):
        error = "a row the appendix names as a crop face does not carry the medium label 0"
        raise ValueError(error)
    return mask


def _require_crop_mask(crop_faces: NDArray[np.bool_], n_triangles: int) -> NDArray[np.bool_]:
    """Check that a crop-face mask is a boolean vector over the triangles."""
    crop_faces = np.asarray(crop_faces)
    if crop_faces.dtype != bool or crop_faces.shape != (n_triangles,):
        error = (
            f"crop_faces must be a boolean array of shape ({n_triangles},), got {crop_faces.dtype} {crop_faces.shape}"
        )
        raise ValueError(error)
    return crop_faces


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
            `spacing_unit`, `source_shape`, `crop_offset`, `crop_faces`. Sequence-valued fields
            (`spacing`, `source_shape`, `crop_offset`, `crop_faces`) are written space-separated.
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
            Sequence-valued fields known to the schema are parsed into tuples (`spacing` and
            `enclosed_media_inscribed_radius` of `float`; `source_shape`, `crop_offset`, `crop_faces`, `media`,
            `enclosed_media_voxels`, `unresolved_regions` and `unresolved_regions_voxels` of `int`), and
            `resampled_spacing` into a `float`; every other key is kept as a raw string, never
            dropped, so a future rewrite can carry it forward.
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
        elif key in _APPENDIX_SCALAR_FLOAT_KEYS:
            metadata[key] = float(raw_value)
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
    crop_faces: NDArray[np.bool_] | None = None,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood. Its layout, in the machine's
    byte order (little-endian on x86 and ARM), is: the number of points (uint64); the points, 3 float64
    each; the number of triangles (uint64); per triangle, its 3 vertex indices (int64) then its 2 labels
    (int32). The points are always written as float64, the dtype `load_rec` reads, whatever their dtype
    in memory (before 0.7.0 they were written in their own dtype, so float32 points did not read back).
    The text layout is the point count, one `x y z` line per point, the triangle count, and one
    `v0 v1 v2 label1 label2` line per triangle. Either layout may be followed by the metadata appendix.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): `(n, 3)` mesh points, written as given.
        triangles (NDArray[np.int64]): `(m, 3)` mesh triangles (topology).
        labels (NDArray[np.int64]): `(m, 2)` the two regions each triangle separates.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
        metadata (dict[str, Any] | None, optional): appendix fields (`coordinate_frame`,
            `axis_order`, `spacing`, `spacing_unit`, `source_shape`, `crop_offset`, ...).
            Defaults to `None`, which writes nothing extra -- byte-for-byte the same file
            as before this option existed. When supplied, a trailing metadata appendix is
            appended after the geometry data; see `write_rec_appendix`.
        crop_faces (NDArray[np.bool_] | None, optional): `(n_triangles,)`, `True` on the crop
            faces (see `dw3d.crop_faces`). When given, the crop faces are written as the **last
            block of triangle rows** (the other rows first, both blocks in their original order)
            and the appendix records `crop_faces = <first row> <count>` (`<count>` may be 0), so
            the file's rows are in a different order from the arrays passed in. A reader that
            ignores the appendix still reads a valid mesh, in which the crop faces look like
            contact with the medium; `load_rec_with_crop_faces` reads them back. Defaults to
            `None`, which writes the file as before this option existed.

    Raises:
        ValueError: if `crop_faces` is not a boolean array with one entry per triangle.
    """
    if crop_faces is not None:
        crop_faces = _require_crop_mask(crop_faces, len(triangles))
        order = np.concatenate([np.flatnonzero(~crop_faces), np.flatnonzero(crop_faces)])
        triangles = np.asarray(triangles)[order]
        labels = np.asarray(labels)[order]
        n_crop = int(np.count_nonzero(crop_faces))
        metadata = {**(metadata or {}), "crop_faces": (len(order) - n_crop, n_crop)}

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
    """Save a rec file at filename. Binary only; the layout is in `save_rec`'s docstring."""
    # float64 whatever the points' own dtype: the reader reads float64, so float32 points written as they were
    # did not read back. For float64 points (dw3d's own) the bytes are what they always were.
    bytes_content = struct.pack("Q", len(points))
    bytes_content += np.ascontiguousarray(points, dtype=np.float64).tobytes()
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
