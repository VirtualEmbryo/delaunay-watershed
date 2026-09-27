"""The `.rec` metadata appendix.

Format, rationale and API are fixed by the design this module implements and does not
relitigate. In short: a trailing block of ASCII `#`-prefixed `key = value` lines, appended
only after the geometry data.
Trailing, because a leading block breaks both `.rec` readers (measured: `ValueError` in
text mode, `OverflowError` in binary mode) while a trailing block is invisible to every
reader already in the wild, including dw3d 0.3.6 and released foambryo.
"""

from __future__ import annotations

import numpy as np
import pytest

from dw3d.io import load_rec, load_rec_appendix, save_rec

POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
)
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
LABELS = np.array([[0, 1], [0, 2], [0, 1], [0, 2]])

METADATA = {
    "coordinate_frame": "voxel_index",
    "axis_order": "zyx",
    "spacing": (1.0, 0.325, 0.325),
    "spacing_unit": "um",
    "source_shape": (194, 200, 199),
    "crop_offset": (0, 0, 0),
}


@pytest.mark.parametrize("binary_mode", [False, True])
@pytest.mark.parametrize("suffix", [".rec", ".arec"])
def test_appendix_round_trip(tmp_path, binary_mode, suffix):
    """Round trip: every metadata field survives write -> read, both modes, both extensions."""
    path = tmp_path / f"mesh{suffix}"
    save_rec(path, POINTS, TRIANGLES, LABELS, binary_mode=binary_mode, metadata=METADATA)

    appendix = load_rec_appendix(path)

    assert appendix is not None
    for key, value in METADATA.items():
        assert appendix[key] == value


@pytest.mark.parametrize("binary_mode", [False, True])
def test_legacy_file_has_no_appendix_and_reads_as_unknown(tmp_path, binary_mode):
    """A file written without metadata has no appendix, which must read as `unknown`.

    Not as isotropic spacing: if an absent appendix ever silently became `(1, 1, 1)`,
    every legacy file would start asserting isotropy, and the ability to detect the
    anisotropic case -- the entire reason this appendix exists -- would be lost.
    """
    path = tmp_path / "legacy.rec"
    save_rec(path, POINTS, TRIANGLES, LABELS, binary_mode=binary_mode)

    appendix = load_rec_appendix(path)

    assert appendix is None
    # Explicitly not the isotropic default some other format might silently assume:
    assert appendix != {"coordinate_frame": "isotropic", "spacing": (1.0, 1.0, 1.0)}


@pytest.mark.parametrize("binary_mode", [False, True])
def test_legacy_reader_ignores_trailing_appendix(tmp_path, binary_mode):
    """The backward-compatibility guarantee, pinned.

    A file written *with* an appendix must load through `load_rec` -- the same
    count-then-`np.fromfile(..., count=n)` reader every prior version of this format
    uses, and which knows nothing about appendices -- byte-identical (as arrays) to the
    same file written without one. This is the test that stops someone later moving the
    appendix block to the top of the file for tidiness: a leading block breaks both
    readers (measured `ValueError` in text mode, `OverflowError` in binary mode), which
    is exactly why the format requires it trailing.
    """
    with_appendix = tmp_path / "with_appendix.rec"
    without_appendix = tmp_path / "without_appendix.rec"
    save_rec(with_appendix, POINTS, TRIANGLES, LABELS, binary_mode=binary_mode, metadata=METADATA)
    save_rec(without_appendix, POINTS, TRIANGLES, LABELS, binary_mode=binary_mode)

    points_a, triangles_a, labels_a = load_rec(with_appendix)
    points_b, triangles_b, labels_b = load_rec(without_appendix)

    np.testing.assert_array_equal(points_a, points_b)
    np.testing.assert_array_equal(triangles_a, triangles_b)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_np_fromfile_sep_stops_at_count(tmp_path):
    r"""Numpy fragility, pinned.

    The trailing-appendix guarantee (see `test_legacy_reader_ignores_trailing_appendix`)
    depends entirely on `np.fromfile`'s `count=` parameter stopping reading exactly
    there in text mode, leaving trailing content (the appendix) untouched and the file
    position right after the last consumed value. `dw3d`'s text `.rec` reader calls this
    with `sep="\n"`; a `sep` made only of whitespace matches runs of *any* whitespace,
    not just literal newlines, which is what lets "x y z\n"-per-line data parse at all.
    That behaviour is exactly the kind of thing a future numpy release could change
    (e.g. by reading ahead into an internal buffer). If it ever does, this test fails
    here, at the numpy layer, instead of silently corrupting `.rec` reads.
    """
    path = tmp_path / "probe.txt"
    path.write_text("1 2 3\n4 5 6\n# not-a-number\n")

    with path.open() as f:
        values = np.fromfile(f, count=3, sep="\n", dtype=np.int64)
        remainder = f.read()

    assert values.tolist() == [1, 2, 3]
    assert remainder == "4 5 6\n# not-a-number\n"
