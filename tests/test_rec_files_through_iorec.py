"""dw3d.io reads and writes `.rec` files through iorec, and writes the appendix in version 1.

Since 0.8.0 `dw3d.io` calls iorec (>= 1.4.1) instead of carrying a copy of iorec 1.0.5. iorec writes appendix version 2
(`# key: <JSON>` lines) by default; delaunay-watershed-3d 0.7.1 and foambryo 0.7.1 read such a block without an
error but lose every key of it (measured: a file's crop faces, media, unresolved regions and physical spacing),
so dw3d writes version 1 (`# key = value`), which they read. These tests pin that choice, the typing of dw3d's own
version-1 keys, the reading of a version-2 file, and the bytes of a file as the copy of iorec 1.0.5 wrote it.
"""

from __future__ import annotations

import importlib.metadata
import struct

import numpy as np
import pytest

from dw3d.io import (
    APPENDIX_KEY_TYPES,
    APPENDIX_VERSION_WRITTEN,
    load_rec,
    load_rec_appendix,
    load_rec_with_crop_faces,
    parse_rec_appendix,
    save_rec,
)

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 1 / 3, 2.5e-7]])
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4]])
# The last two rows carry the medium, so that `crop_faces = 3 2` names a valid last block.
LABELS = np.array([[0, 1], [1, 2], [0, 1], [0, 2], [0, 2]])

# One value for every key of dw3d's registry, of its type, and three of dw3d's string keys.
EVERY_KEY = {
    "source_shape": (21, 61, 61),
    "crop_offset": (0, 4, 4),
    "media": (0, 9),
    "enclosed_media_voxels": (0, 125),
    "unresolved_regions": (7,),
    "unresolved_regions_voxels": (12,),
    "crop_faces": (3, 2),
    "spacing": (0.9, 0.3, 0.3),
    "enclosed_media_inscribed_radius": (0.0, 2.5),
    "resampled_spacing": 0.3,
    "coordinate_frame": "physical",
    "spacing_unit": "um",
    "physical_image_faces": "none",
    "dimension": 3,
    "image_border_vertices": (0, 5, 9),
}


def _bytes_of_the_vendored_writer(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray, binary: bool) -> bytes:
    """A file without appendix as the copy of iorec 1.0.5 in dw3d 0.7.1 wrote it."""
    if binary:
        data = struct.pack("<Q", len(points)) + np.asarray(points, dtype="<f8").tobytes()
        data += struct.pack("<Q", len(triangles))
        for triangle, label in zip(triangles, labels, strict=True):
            data += struct.pack("<qqqii", *triangle, *label)
        return data
    text = f"{len(points)}\n" + "".join(f"{x} {y} {z}\n" for x, y, z in points)
    rows = zip(triangles, labels, strict=True)
    text += f"{len(triangles)}\n" + "".join(f"{a} {b} {c} {d} {e}\n" for (a, b, c), (d, e) in rows)
    return text.encode()


def test_the_appendix_version_written_is_1() -> None:
    assert APPENDIX_VERSION_WRITTEN == 1


@pytest.mark.parametrize("binary_mode", [False, True])
def test_a_file_without_appendix_is_the_one_dw3d_0_7_1_wrote(tmp_path, binary_mode) -> None:
    save_rec(tmp_path / "mesh.rec", POINTS, TRIANGLES, LABELS, binary_mode=binary_mode)
    expected = _bytes_of_the_vendored_writer(POINTS, TRIANGLES, LABELS, binary_mode)
    assert (tmp_path / "mesh.rec").read_bytes() == expected


@pytest.mark.parametrize("binary_mode", [False, True])
def test_the_appendix_is_written_in_version_1_with_both_versions_stamped(tmp_path, binary_mode) -> None:
    save_rec(tmp_path / "mesh.rec", POINTS, TRIANGLES, LABELS, binary_mode=binary_mode, metadata={"media": (0, 9)})
    raw = (tmp_path / "mesh.rec").read_bytes()
    block = raw[raw.rindex(b"# rec-appendix ") :].decode("ascii").splitlines()
    assert block[0] == "# rec-appendix 1"
    assert block[1] == f"# written_by = {importlib.metadata.version('delaunay-watershed-3d')}"
    assert block[2] == f"# writer_library = iorec {importlib.metadata.version('iorec')}"
    assert block[3:] == ["# media = 0 9"]


@pytest.mark.parametrize("binary_mode", [False, True])
def test_every_key_of_the_registry_comes_back_typed(tmp_path, binary_mode) -> None:
    assert set(APPENDIX_KEY_TYPES) <= set(EVERY_KEY)
    save_rec(tmp_path / "mesh.rec", POINTS, TRIANGLES, LABELS, binary_mode=binary_mode, metadata=EVERY_KEY)
    appendix = load_rec_appendix(tmp_path / "mesh.rec")
    assert appendix["_version"] == 1
    for key, value in EVERY_KEY.items():
        assert appendix[key] == value, key
        assert type(appendix[key]) is type(value), key
        if isinstance(value, tuple):
            assert [type(v) for v in appendix[key]] == [type(v) for v in value], key


@pytest.mark.parametrize("binary_mode", [False, True])
def test_a_version_2_file_reads_to_the_same_typed_values(tmp_path, binary_mode) -> None:
    iorec = pytest.importorskip("iorec")
    metadata = EVERY_KEY
    iorec.save_rec(tmp_path / "v2.rec", POINTS, TRIANGLES, LABELS, binary_mode, metadata=metadata, written_by="x")
    save_rec(tmp_path / "v1.rec", POINTS, TRIANGLES, LABELS, binary_mode=binary_mode, metadata=metadata)
    v2, v1 = load_rec_appendix(tmp_path / "v2.rec"), load_rec_appendix(tmp_path / "v1.rec")
    assert v2["_version"] == 2
    assert v1["_version"] == 1
    for key in metadata:
        assert v2[key] == v1[key], key
        assert type(v2[key]) is type(v1[key]), key
    assert np.array_equal(load_rec_with_crop_faces(tmp_path / "v2.rec")[3], [False, False, False, True, True])
    for a, b in zip(load_rec(tmp_path / "v2.rec"), load_rec(tmp_path / "v1.rec"), strict=True):
        assert np.array_equal(a, b)


def test_text_points_are_written_exactly(tmp_path) -> None:
    save_rec(tmp_path / "mesh.rec", POINTS, TRIANGLES, LABELS)
    points, triangles, labels = load_rec(tmp_path / "mesh.rec")
    assert np.array_equal(points, POINTS)
    assert np.array_equal(triangles, TRIANGLES)
    assert np.array_equal(labels, LABELS)


def test_parse_rec_appendix_reads_a_block_in_memory() -> None:
    block = "# rec-appendix 1\n# written_by = 0.7.1\n# crop_faces = 10 2\n# spacing = 1 0.5 0.5\n# other = a b\n"
    assert parse_rec_appendix(block) == {
        "_version": 1,
        "written_by": "0.7.1",
        "crop_faces": (10, 2),
        "spacing": (1.0, 0.5, 0.5),
        "other": "a b",
    }
    with pytest.raises(ValueError, match="rec-appendix"):
        parse_rec_appendix("# not an appendix\n")
