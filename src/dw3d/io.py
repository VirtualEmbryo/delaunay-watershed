"""Read and write dw3d meshes: the `.rec` format with its metadata appendix, and `.vtk`/`.vtu` through meshio.

A mesh is `(points, triangles, labels)`: `(n, 3)` points, `(m, 3)` vertex indices and `(m, 2)` labels,
the two regions each triangle separates (label 0 is the medium). The files store the points as given:
these functions neither scale nor reorder their columns. A file therefore keeps the mask's order, `(z, y, x)` for a
microscope stack; a program that reads its columns as `(x, y, z)`, as ParaView, MeshLab and other VTK readers do,
shows the mirror image of the tissue, unless it exchanges columns 0 and 2 (and two corners of every triangle, so
that each still faces the same side), as `dw3d.viewing.plot_cells_polyscope` does for what it draws.

The `.rec` reader and writer are iorec's (`iorec.load_rec`, `iorec.save_rec`, `iorec.load_rec_appendix`); this module
keeps dw3d's names and signatures, types dw3d's own appendix keys when it reads a version-1 appendix, and writes the
appendix in version 1 (see `APPENDIX_VERSION_WRITTEN`).
Matthieu Perez 2024
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import iorec
import iorec.rec_appendix
import meshio
import numpy as np
from numpy.typing import NDArray

# The appendix version this module writes. iorec 1.4.1 writes version 2 by default (`# key: <JSON>` lines);
# delaunay-watershed-3d 0.7.1 and foambryo 0.7.1 read such a block without an error but lose every key of it
# (their parser splits each line on `=`): a file's crop faces, media, unresolved regions and physical spacing
# would vanish silently for anyone still on 0.7.1. Version 1 (`# key = value` lines) reads the same in both.
APPENDIX_VERSION_WRITTEN = 1

# The type of each of dw3d's own appendix keys in a version-1 appendix, where values are text (a version-2
# appendix carries its types): sequences of int or float are space-separated, `resampled_spacing` is one float.
# Every other key is kept as a string.
_INTEGER_SEQUENCE_KEYS = (
    "source_shape",
    "crop_offset",
    "crop_faces",
    "media",
    "enclosed_media_voxels",
    "unresolved_regions",
    "unresolved_regions_voxels",
)
APPENDIX_KEY_TYPES: dict[str, object] = {
    **dict.fromkeys(_INTEGER_SEQUENCE_KEYS, tuple[int, ...]),
    # Written by both packages' algorithms since 0.8.0 (dw3d: whenever it writes an appendix; dw2d: always), so that a
    # reader can tell a 2-D file from a 3-D one without probing its layout (`rec_dimension`).
    "dimension": int,
    # dw2d's: the vertices on the image border.
    "image_border_vertices": tuple[int, ...],
    "spacing": tuple[float, ...],
    "enclosed_media_inscribed_radius": tuple[float, ...],
    "resampled_spacing": float,
}


def _dw3d_version() -> str:
    """Return the installed dw3d version, or "unknown" if it can't be determined.

    The importable module is `dw3d`; `importlib.metadata` needs the name of the distribution that carries it,
    `delaunay-watershed-3d` (the metapackage `delaunay-watershed` only requires it at the same version).
    """
    try:
        return version("delaunay-watershed-3d")
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
        ValueError: if `crop_faces` is not a boolean array with one entry per triangle, or if the mesh has no triangle
            (since 0.8.0; before, meshio raised `IndexError` while writing a `.vtk` file, and wrote `.vtu` files that
            meshio cannot read back). Nothing is written then; `save_rec` writes an empty `.rec` file, which
            `load_rec` reads back.
    """
    if len(triangles) == 0:
        error = (
            f"cannot write {Path(filename).name}: the mesh has no triangle (a mask whose cells are all too small to "
            "hold a tetrahedron, or a mask without cells; see the UnresolvedCellWarning of the reconstruction). meshio "
            "cannot write an empty .vtk file, nor read back the empty .vtu files it writes; save_rec (or the "
            "reconstruction's save_to_rec_mesh) writes an empty .rec file"
        )
        raise ValueError(error)
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


_APPENDIX_MARKER = b"# rec-appendix "


def _layouts_of_a_binary_rec(filename: Path) -> list[int]:
    """The dimensions (2, 3) whose binary layout accounts exactly for the file's geometry bytes.

    A binary `.rec` file is `n` (uint64), `n` points of `d` float64, `m` (uint64), then `m` records of `d` vertex indices
    (int64) and 2 labels (int32), optionally followed by the appendix, which starts with `# rec-appendix `. Layout `d`
    fits exactly when its geometry ends at the end of the file or at the start of an appendix; when neither fits exactly
    (stray bytes after the geometry), the layouts that leave less than one of their records unread are returned.
    """
    size = filename.stat().st_size
    exact, within = [], []
    with filename.open("rb") as f:
        header = f.read(8)
        if len(header) < 8:
            return []
        n = int(np.frombuffer(header, dtype="<u8")[0])
        for d in (2, 3):
            at = 8 + 8 * d * n
            if at + 8 > size:
                continue
            f.seek(at)
            m = int(np.frombuffer(f.read(8), dtype="<u8")[0])
            end = at + 8 + (8 * d + 8) * m
            if end > size:
                continue
            f.seek(end)
            if end == size or f.read(len(_APPENDIX_MARKER)) == _APPENDIX_MARKER:
                exact.append(d)
            elif size - end < 8 * d + 8:
                within.append(d)
    # A file with a few stray bytes after its geometry (which iorec's readers ignore) fits no layout exactly; then a
    # layout that leaves less than one of its own records unread decides (the other reads its counts from coordinate
    # bits and lands elsewhere).
    return exact or within


def _layouts_of_a_text_rec(filename: Path) -> list[int]:
    """The dimensions whose text layout the file's first rows have: 2 or 3 coordinates per point row (or, with no
    point, 4 or 5 integers per segment or triangle row)."""
    with filename.open() as f:
        rows = []
        for line in f:
            if line.strip() and not line.startswith("#"):
                rows.append(line.split())
            if len(rows) == 3:
                break
    if len(rows) < 2:
        return []
    n = int(rows[0][0])
    if n > 0:
        return [len(rows[1])] if len(rows[1]) in (2, 3) else []
    if len(rows) < 3:
        return []
    return [len(rows[2]) - 2] if len(rows[2]) in (4, 5) else []


def rec_dimension(filename: str | Path) -> int:
    """Whether a `.rec`/`.arec` file holds a 2-D mesh (segments) or a 3-D mesh (triangles).

    The `.rec` format carries no dimension marker. Since 0.8.0 both packages' algorithms record `dimension` in the
    metadata appendix (dw2d always, dw3d whenever it writes an appendix); when the key is present it decides. Otherwise
    the layout does: in a text file, the number of coordinates on the point rows; in a binary file, which of the two
    layouts accounts exactly for the geometry's bytes before the appendix (`_layouts_of_a_binary_rec`). A file both
    layouts would fit (possible in principle for a binary file, never observed) is refused rather than guessed.

    Args:
        filename (str | Path): the path to the file.

    Returns:
        int: 2 or 3.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the appendix's `dimension` is not 2 or 3, or if the layout fits neither or both dimensions.
    """
    path = Path(filename)
    if not path.is_file():
        error = f"no such file: {path}"
        raise FileNotFoundError(error)
    appendix = load_rec_appendix(path)
    if appendix is not None and "dimension" in appendix:
        dimension = int(appendix["dimension"])
        if dimension not in (2, 3):
            error = f"{path}: the appendix records dimension = {dimension}, not 2 or 3"
            raise ValueError(error)
        return dimension
    fitting = _layouts_of(path)
    if len(fitting) != 1:
        error = (
            f"{path}: cannot tell whether this .rec file is 2-D or 3-D from its layout "
            f"({'both layouts fit' if fitting else 'neither layout fits'}); it records no dimension in its appendix"
        )
        raise ValueError(error)
    return fitting[0]


def load_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a rec mesh file, text or binary (detected from the content).

    A metadata appendix, if any, is ignored; read it with `load_rec_appendix`. Crop faces are
    returned as ordinary triangles, in the file's row order; read them with
    `load_rec_with_crop_faces`. A 2-D file (`rec_dimension`) is refused: read it with `dw2d.io.load_rec`.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file, or if it holds a 2-D mesh.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: `(points, triangles, labels)`: points
            `(num_points, 3)` float64, triangles `(num_triangles, 3)` int64 and labels
            `(num_triangles, 2)` int64.
    """
    path = Path(filename)
    if path.is_file() and path.suffix in (".rec", ".arec") and _is_two_dimensional(path):
        error = f"{path} holds a 2-D mesh (segments, not triangles): read it with dw2d.io.load_rec"
        raise ValueError(error)
    return iorec.load_rec(filename)


def _layouts_of(path: Path) -> list[int]:
    with path.open("rb") as f:
        binary = b"\x00" in f.read(64)
    return _layouts_of_a_binary_rec(path) if binary else _layouts_of_a_text_rec(path)


def _is_two_dimensional(path: Path) -> bool:
    """Whether a file holds a 2-D mesh, from its layout (a few bytes read), the appendix only when the layout cannot
    tell; a file neither can classify is left to iorec's reader, which reports what is wrong with it."""
    fitting = _layouts_of(path)
    if len(fitting) == 1:
        return fitting[0] == 2
    try:
        return rec_dimension(path) == 2
    except ValueError:
        return False


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


def write_rec_appendix(metadata: dict[str, Any], *, written_by: str) -> bytes:
    """Build the trailing `.rec`/`.arec` metadata appendix block.

    The block is a sequence of ASCII `#`-prefixed `key = value` lines (appendix version 1,
    `APPENDIX_VERSION_WRITTEN`), headed by the `rec-appendix 1` line, built by
    `iorec.write_rec_appendix`. It must be appended at the very end of the file: a
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
    return iorec.write_rec_appendix(metadata, written_by=written_by, appendix_version=APPENDIX_VERSION_WRITTEN)


def parse_rec_appendix(block_text: str) -> dict[str, Any]:
    """Parse a `.rec`/`.arec` metadata appendix block into a dict.

    Args:
        block_text: the appendix block, headed by the `# rec-appendix <version>` line: version 1
            (`# key = value` lines) or version 2 (`# key: <JSON>` lines).

    Raises:
        ValueError: if `block_text` does not start with a `rec-appendix <version>` header, if the
            version is not 1 or 2, or if a value cannot be read as its key's type.

    Returns:
        dict[str, Any]: parsed fields, plus `"_version"` (the schema version, an int).
            Sequence-valued fields known to the schema are parsed into tuples (`spacing` and
            `enclosed_media_inscribed_radius` of `float`; `source_shape`, `crop_offset`, `crop_faces`, `media`,
            `enclosed_media_voxels`, `unresolved_regions` and `unresolved_regions_voxels` of `int`), and
            `resampled_spacing` into a `float` (`APPENDIX_KEY_TYPES`); every other key is kept as a
            raw string, never dropped, so a future rewrite can carry it forward. A version-2 block
            gives each value the type its JSON holds, sequences as tuples.
    """
    # iorec's parser, which has no public name for a block already in memory; pinned by tests/test_rec_appendix.py.
    return iorec.rec_appendix._parse_rec_appendix(block_text, APPENDIX_KEY_TYPES)


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
    return iorec.load_rec_appendix(filename, metadata_types=APPENDIX_KEY_TYPES)


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

    Binary mode is more compact but might not be universally understood. Its layout, little-endian
    on every machine (iorec's), is: the number of points (uint64); the points, 3 float64
    each; the number of triangles (uint64); per triangle, its 3 vertex indices (int64) then its 2 labels
    (int32). The points are always written as float64, the dtype `load_rec` reads, whatever their dtype
    in memory (before 0.7.0 they were written in their own dtype, so float32 points did not read back).
    The text layout is the point count, one line of 3 coordinates per point (each the shortest text
    that reads back to the same float64), the triangle count, and one `v0 v1 v2 label1 label2` line per
    triangle. Either layout may be followed by the metadata appendix (version 1, `APPENDIX_VERSION_WRITTEN`).
    The file is written by `iorec.save_rec`.

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

    # iorec stamps dw3d's version on `written_by` and its own on `writer_library`.
    iorec.save_rec(
        filename,
        points,
        triangles,
        labels,
        binary_mode,
        metadata=metadata,
        written_by=_dw3d_version(),
        appendix_version=APPENDIX_VERSION_WRITTEN,
    )
