"""dw2d's .rec reader and writer, `dw2d.io.load_rec` and `dw2d.io.save_rec` (through iorec), and the deprecated aliases.

Since 0.8.0 they take and return dw3d's three arrays, `(points, segments, labels)`; the 0.1.5 aliases still take and
return `(verts, edges)` with `edges` `(m, 4)`.

The binary layout is pinned byte for byte against the layout dw2d 0.1.5's `write_mesh_bin` wrote (rebuilt here with
`struct`); a text file as 0.1.5's `write_mesh_text` wrote it (coordinates with 5 decimals) reads to the values it
holds. The aliases of 0.1.5 (`read_rec_file_bin`, `read_rec_file_num`, `open_mesh_multitracker`, `write_mesh_bin`,
`write_mesh_text`) warn, name their replacement, and return what they returned before.
"""

import struct
import warnings

import numpy as np
import pytest
import skimage.io

import dw2d
from dw2d import Dcel, geometry_reconstruction_2d
from dw2d.io import load_rec, save_rec

from .conftest import MASKS


@pytest.fixture(scope="module")
def mesh() -> tuple[np.ndarray, np.ndarray]:
    labels = skimage.io.imread(MASKS / "mask_Cells_3.tif")
    return geometry_reconstruction_2d(labels, expansion_labels=0, min_dist=5).return_mesh()


def _save(path, verts, edges, binary_mode=False):
    save_rec(path, verts, edges[:, :2], edges[:, 2:], binary_mode=binary_mode)


def _load(path):
    points, segments, labels = load_rec(path)
    assert segments.dtype == labels.dtype == np.int64
    return points, np.hstack((segments, labels))


def _layout_of_0_1_5(verts: np.ndarray, edges: np.ndarray) -> bytes:
    """The bytes dw2d 0.1.5's `write_mesh_bin` wrote: counts uint64, points float64, per edge 2 uint64 + 2 int32."""
    data = struct.pack("<Q", len(verts)) + np.asarray(verts, dtype="<f8").tobytes()
    data += struct.pack("<Q", len(edges))
    for v1, v2, m1, m2 in np.asarray(edges):
        data += struct.pack("<QQii", v1, v2, m1, m2)
    return data


def _text_of_0_1_5(verts: np.ndarray, edges: np.ndarray) -> str:
    """The text dw2d 0.1.5's `write_mesh_text` wrote: coordinates with 5 decimals."""
    lines = [str(len(verts))] + [f"{x:.5f} {y:.5f}" for x, y in verts]
    lines += [str(len(edges))] + [" ".join(str(v) for v in row) for row in edges]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("binary_mode", [True, False])
def test_a_mesh_reads_back_exactly(mesh, tmp_path, binary_mode) -> None:
    verts, edges = mesh
    path = tmp_path / "mesh.rec"
    _save(path, verts, edges, binary_mode=binary_mode)
    read_verts, read_edges = _load(path)
    assert read_verts.dtype == np.float64
    assert read_edges.dtype == np.int64
    assert np.array_equal(read_verts, verts)
    assert np.array_equal(read_edges, edges)


def test_the_binary_layout_is_the_one_of_0_1_5(mesh, tmp_path) -> None:
    verts, edges = mesh
    _save(tmp_path / "mesh.rec", verts, edges, binary_mode=True)
    assert (tmp_path / "mesh.rec").read_bytes() == _layout_of_0_1_5(verts, edges)


def test_a_text_file_of_0_1_5_reads_to_its_values(mesh, tmp_path) -> None:
    verts, edges = mesh
    path = tmp_path / "old.rec"
    path.write_text(_text_of_0_1_5(verts, edges))
    read_verts, read_edges = _load(path)
    assert np.array_equal(read_verts, np.round(verts, 5))
    assert np.array_equal(read_edges, edges)


def test_text_writes_every_coordinate_exactly(tmp_path) -> None:
    verts = np.array([[0.1, 1 / 3], [2.0, np.pi], [1e-7, 12345.678901234]])
    edges = np.array([[0, 1, 0, 1], [1, 2, 1, 2]])
    _save(tmp_path / "exact.rec", verts, edges)
    read_verts, _ = _load(tmp_path / "exact.rec")
    assert np.array_equal(read_verts, verts)


def test_export_mesh_writes_what_save_rec_writes(tmp_path) -> None:
    labels = skimage.io.imread(MASKS / "mask_Cells_3.tif")
    reconstruction = geometry_reconstruction_2d(labels, expansion_labels=0, min_dist=5)
    verts, edges = reconstruction.return_mesh()
    for mode, binary_mode in (("bin", True), ("txt", False)):
        reconstruction.export_mesh(str(tmp_path / f"export.{mode}.rec"), mode=mode)
        _save(tmp_path / f"save.{mode}.rec", verts, edges, binary_mode=binary_mode)
        assert (tmp_path / f"export.{mode}.rec").read_bytes() == (tmp_path / f"save.{mode}.rec").read_bytes()


def test_save_rec_refuses_arrays_of_the_wrong_shape(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        save_rec(tmp_path / "bad.rec", np.zeros((3, 3)), np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        save_rec(tmp_path / "bad.rec", np.zeros((3, 2)), np.zeros((2, 4), dtype=int), np.zeros((2, 2), dtype=int))


def test_the_names_mirror_dw3d_io() -> None:
    """`dw2d.io` has dw3d.io's reader, writer and appendix reader; the package's top level has the writer by dw3d's
    top-level name, `save_mesh_to_rec_mesh` (the io names were added for 0.8.0 and are not at the top level)."""
    import dw2d.io

    assert {"load_rec", "save_rec", "load_rec_appendix"} <= set(dir(dw2d.io))
    assert callable(dw2d.save_mesh_to_rec_mesh)
    assert "load_rec" not in dw2d.__all__


def _call_warning_once(name: str, *args: object) -> tuple:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(Dcel, name)(*args)
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 1, messages
    assert f"dw2d.Dcel.{name} is deprecated" in messages[0]
    assert "removed in 0.9" in messages[0]
    assert "dw2d.io." in messages[0]
    return result


def test_the_readers_of_0_1_5_warn_and_return_what_they_returned(mesh, tmp_path) -> None:
    verts, edges = mesh
    _save(tmp_path / "mesh.bin.rec", verts, edges, binary_mode=True)
    _save(tmp_path / "mesh.txt.rec", verts, edges)
    counts = np.array([len(verts), len(edges)])
    for name, args, shift in (
        ("read_rec_file_bin", (str(tmp_path / "mesh.bin.rec"),), 0),
        ("read_rec_file_num", (str(tmp_path / "mesh.txt.rec"),), 0),
        ("read_rec_file_num", (str(tmp_path / "mesh.txt.rec"), 3), 3),
        ("open_mesh_multitracker", ("bin", str(tmp_path / "mesh.bin.rec")), 0),
        ("open_mesh_multitracker", ("txt", str(tmp_path / "mesh.txt.rec")), 0),
    ):
        read_verts, read_edges, read_counts = _call_warning_once(name, *args)
        expected_edges = edges.copy()
        expected_edges[:, 2:] += shift
        assert np.array_equal(read_verts, verts)
        assert read_verts.dtype == np.float64
        assert np.array_equal(read_edges, expected_edges)
        assert read_edges.dtype == np.int64
        assert np.array_equal(read_counts, counts)
        assert read_counts.dtype == np.int64


def test_the_writers_of_0_1_5_warn_and_write_what_save_rec_writes(mesh, tmp_path) -> None:
    verts, edges = mesh
    _call_warning_once("write_mesh_bin", str(tmp_path / "alias.bin.rec"), verts, edges)
    _call_warning_once("write_mesh_text", str(tmp_path / "alias.txt.rec"), verts, edges)
    assert (tmp_path / "alias.bin.rec").read_bytes() == _layout_of_0_1_5(verts, edges)
    _save(tmp_path / "save.txt.rec", verts, edges)
    assert (tmp_path / "alias.txt.rec").read_bytes() == (tmp_path / "save.txt.rec").read_bytes()


def test_open_mesh_multitracker_has_no_default_file() -> None:
    with pytest.raises(TypeError):
        Dcel.open_mesh_multitracker("bin")
