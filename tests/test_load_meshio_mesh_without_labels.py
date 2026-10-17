"""A meshio file without the `label1` / `label2` triangle arrays is refused with a `ValueError` naming the file.

Before 0.7.0 `load_meshio_mesh` let meshio's `KeyError` through, while its docstring and the empty-array case
already used `ValueError`; a caller catching `ValueError` for "not a dw3d mesh" missed it. Pins: no label array,
one of the two missing, and the crop-face reader, which goes through the same function.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest

from dw3d.io import load_meshio_mesh, load_meshio_mesh_with_crop_faces

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def write_vtk(path: Path, cell_data: dict) -> Path:
    meshio.Mesh(POINTS, [("triangle", TRIANGLES)], cell_data=cell_data).write(path)
    return path


@pytest.mark.parametrize(
    "cell_data",
    [{}, {"label1": [np.zeros(4, dtype=np.int64)]}, {"label2": [np.ones(4, dtype=np.int64)]}],
    ids=["no label array", "label2 missing", "label1 missing"],
)
@pytest.mark.parametrize("reader", [load_meshio_mesh, load_meshio_mesh_with_crop_faces])
def test_missing_labels_raise_a_value_error(tmp_path: Path, cell_data: dict, reader) -> None:
    path = write_vtk(tmp_path / "unlabelled.vtk", cell_data)
    with pytest.raises(ValueError, match="has no labels data"):
        reader(path)


def test_a_labelled_file_still_reads(tmp_path: Path) -> None:
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
    path = write_vtk(tmp_path / "labelled.vtk", {"label1": [labels[:, 0]], "label2": [labels[:, 1]]})
    points, triangles, read_labels = load_meshio_mesh(path)
    assert np.array_equal(points, POINTS)
    assert np.array_equal(triangles, TRIANGLES)
    assert np.array_equal(read_labels, labels)
