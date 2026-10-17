"""The binary `.rec` layout stores the points as float64 whatever their dtype in memory, so a file reads back.

Before 0.7.0 `save_rec(..., binary_mode=True)` wrote the points' bytes in their own dtype while `load_rec` reads
float64: float32 points (as a mesh made by another program can carry) read back as garbage, or not at all. Pins:
float32, float64 and non-contiguous points read back equal to their float64 values; for float64 points the file is
byte for byte the documented layout (so every file dw3d wrote before is unchanged); the text layout is unchanged.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from dw3d.io import load_rec, save_rec

POINTS = np.array([[0.0, 0.0, 0.0], [1.5, 0.25, 0.0], [0.0, 1.0, 0.1], [0.3, 0.0, 1.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
LABELS = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)


def documented_binary_layout(points: np.ndarray) -> bytes:
    """The layout `save_rec`'s docstring describes, built independently of the writer."""
    rows = np.zeros(len(TRIANGLES), dtype=[("triangles", np.int64, (3,)), ("labels", np.int32, (2,))])
    rows["triangles"] = TRIANGLES
    rows["labels"] = LABELS
    return (
        struct.pack("Q", len(points))
        + np.asarray(points, dtype=np.float64).tobytes()
        + struct.pack("Q", len(TRIANGLES))
        + rows.tobytes()
    )


@pytest.mark.parametrize(
    "points",
    [POINTS.astype(np.float32), POINTS, np.asfortranarray(POINTS), POINTS.T.copy().T],
    ids=["float32", "float64", "float64 Fortran order", "float64 transposed view"],
)
def test_binary_points_read_back_as_their_float64_values(tmp_path: Path, points: np.ndarray) -> None:
    path = tmp_path / "mesh.rec"
    save_rec(path, points, TRIANGLES, LABELS, binary_mode=True)
    read_points, read_triangles, read_labels = load_rec(path)
    assert read_points.dtype == np.float64
    assert np.array_equal(read_points, points.astype(np.float64))
    assert np.array_equal(read_triangles, TRIANGLES)
    assert np.array_equal(read_labels, LABELS)


def test_float64_binary_file_is_the_documented_layout(tmp_path: Path) -> None:
    path = tmp_path / "mesh.rec"
    save_rec(path, POINTS, TRIANGLES, LABELS, binary_mode=True)
    assert path.read_bytes() == documented_binary_layout(POINTS)


def test_text_layout_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "mesh.rec"
    save_rec(path, POINTS.astype(np.float32), TRIANGLES, LABELS)
    lines = path.read_text().splitlines()
    assert lines[0] == "4"
    assert lines[1:5] == [f"{p[0]} {p[1]} {p[2]}" for p in POINTS.astype(np.float32)]
    assert lines[5] == "4"
    assert lines[6:] == [f"{t[0]} {t[1]} {t[2]} {lab[0]} {lab[1]}" for t, lab in zip(TRIANGLES, LABELS, strict=True)]
