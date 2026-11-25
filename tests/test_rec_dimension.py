"""`dw3d.io.rec_dimension`: a `.rec` file is told 2-D or 3-D by its appendix's `dimension`, else by its layout.

The `.rec` format carries no dimension marker. Pinned: (1) the appendix key decides when present (both packages'
algorithms write it: dw2d always, dw3d whenever it writes an appendix); (2) without it, the layout decides: the number
of coordinates on a text file's point rows, and which binary layout accounts exactly for the geometry's bytes, before
an appendix or the end of the file; (3) `dw3d.io.load_rec` refuses a 2-D file with a message naming dw2d's reader;
(4) a 3-D file with an appendix records `dimension = 3`; a 3-D file without one is written byte for byte as before.
"""

import numpy as np
import pytest

import dw3d
import dw3d.io

POINTS_2D = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.5]])
SEGMENTS = np.array([[0, 1], [1, 2], [2, 3]])
LABELS_2D = np.array([[0, 1], [1, 2], [0, 2]])
POINTS_3D = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3]])
LABELS_3D = np.array([[0, 1], [0, 1]])


def _write_2d(path, binary, metadata=None):
    import iorec

    if metadata is None:
        iorec.save_2d_rec(path, POINTS_2D, SEGMENTS, LABELS_2D, binary)
    else:
        iorec.save_2d_rec(path, POINTS_2D, SEGMENTS, LABELS_2D, binary, metadata=metadata, appendix_version=1)


@pytest.mark.parametrize("binary", [False, True])
def test_the_layout_tells_the_dimension_without_an_appendix(tmp_path, binary):
    _write_2d(tmp_path / "two.rec", binary)
    dw3d.io.save_rec(tmp_path / "three.rec", POINTS_3D, TRIANGLES, LABELS_3D, binary)
    assert dw3d.io.load_rec_appendix(tmp_path / "two.rec") is None
    assert dw3d.io.rec_dimension(tmp_path / "two.rec") == 2
    assert dw3d.io.rec_dimension(tmp_path / "three.rec") == 3


@pytest.mark.parametrize("binary", [False, True])
def test_the_layout_is_read_up_to_an_appendix(tmp_path, binary):
    _write_2d(tmp_path / "two.rec", binary, metadata={"spacing": (1.0, 1.0)})
    assert dw3d.io.rec_dimension(tmp_path / "two.rec") == 2


def test_the_appendix_key_decides_when_present(tmp_path):
    _write_2d(tmp_path / "two.rec", False, metadata={"dimension": 2})
    assert dw3d.io.load_rec_appendix(tmp_path / "two.rec")["dimension"] == 2
    assert dw3d.io.rec_dimension(tmp_path / "two.rec") == 2
    _write_2d(tmp_path / "wrong.rec", False, metadata={"dimension": 4})
    with pytest.raises(ValueError, match="dimension = 4"):
        dw3d.io.rec_dimension(tmp_path / "wrong.rec")


def test_a_file_of_neither_layout_is_refused(tmp_path):
    (tmp_path / "bad.rec").write_text("2\n1 2 3 4\n5 6 7 8\n")
    with pytest.raises(ValueError, match="neither layout"):
        dw3d.io.rec_dimension(tmp_path / "bad.rec")
    with pytest.raises(FileNotFoundError):
        dw3d.io.rec_dimension(tmp_path / "missing.rec")


@pytest.mark.parametrize("binary", [False, True])
def test_load_rec_refuses_a_2d_file(tmp_path, binary):
    _write_2d(tmp_path / "two.rec", binary)
    with pytest.raises(ValueError, match="dw2d.io.load_rec"):
        dw3d.io.load_rec(tmp_path / "two.rec")


def test_a_3d_appendix_records_the_dimension_and_a_file_without_one_is_unchanged(tmp_path):
    mask = np.zeros((24, 24, 24), dtype=np.uint16)
    mask[4:20, 4:12, 4:20] = 1
    mask[4:20, 12:20, 4:20] = 2
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_in_physical_units(mask, (1.0, 1.0, 1.0))
    algorithm.save_to_rec_mesh(tmp_path / "physical.rec")
    assert dw3d.io.load_rec_appendix(tmp_path / "physical.rec")["dimension"] == 3
    algorithm.construct_mesh_from_segmentation_mask(mask)
    algorithm.save_to_rec_mesh(tmp_path / "plain.rec")
    points, triangles, labels = algorithm.last_constructed_mesh
    dw3d.io.save_rec(tmp_path / "direct.rec", points, triangles, labels)
    assert (tmp_path / "plain.rec").read_bytes() == (tmp_path / "direct.rec").read_bytes()
    assert dw3d.io.rec_dimension(tmp_path / "plain.rec") == 3


def test_stray_bytes_after_a_binary_geometry_leave_the_dimension_readable(tmp_path):
    """iorec's readers ignore bytes after the geometry; so does the probe, when one layout alone leaves less than one of
    its records unread. On a mesh with real coordinates the other layout reads its counts from coordinate bits and lands
    far away (on a mesh of a few points at small integers both may fit, and the probe refuses rather than guess)."""
    import iorec

    rng = np.random.default_rng(0)
    iorec.save_rec(tmp_path / "three.rec", rng.uniform(0, 30, (40, 3)), rng.integers(0, 40, (60, 3)),
                   np.tile([0, 1], (60, 1)), True)
    iorec.save_2d_rec(tmp_path / "two.rec", rng.uniform(0, 30, (40, 2)), rng.integers(0, 40, (60, 2)),
                      np.tile([0, 1], (60, 1)), True)
    for name, dimension in (("three.rec", 3), ("two.rec", 2)):
        (tmp_path / name).write_bytes((tmp_path / name).read_bytes() + b"\n")
        assert dw3d.io.rec_dimension(tmp_path / name) == dimension
