"""`plot_cells_polyscope` draws the tissue, not its mirror image, and keeps every triangle facing its own side.

dw3d's mesh points are in the mask's order, `(z, y, x)`. Up to 0.7.0 the viewer handed them to Polyscope as its
`(x, y, z)`: an exchange of two axes, a reflection, so every view was the mirror image of the tissue. It now draws
`(x, y, z)` = the columns `(2, 1, 0)`, the mask's (column, row, slice), and hands each triangle `(a, b, c)` as
`(a, c, b)`. Pins, on a chirality phantom (three arms of different lengths along z, y and x, and a letter F):

* the map from the mask's (column, row, slice) to the drawn points is the identity: determinant +1 (the former map,
  the stored points as they are, has determinant -1);
* the arms, told apart by their lengths alone, are drawn right-handed, along Polyscope's x, y, z;
* each drawn triangle's normal is the mesh's normal mapped into the drawn frame (orientation kept), and the cells'
  signed volumes from the drawn arrays are those of the stored mesh;
* the reconstruction's mesh is unchanged.

polyscope is replaced by a recorder (the real viewer needs an OpenGL context), so nothing is displayed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.viewing import plot_cells_polyscope

ARMS = {"x": 2, "y": 3, "z": 4}
ARM_LENGTHS = {"x": 56, "y": 36, "z": 16}


def chirality_phantom() -> np.ndarray:
    """`(z, y, x)` labels: a 16-voxel cube (1), arms along x (2, 56 voxels), y (3, 36) and z (4, 16), a letter F (5).

    The same phantom as foambryo's `tests/chirality_phantom.py`. The F lies in slices 10 to 21, its top bar against
    the x arm; it reads as an F shown as an image (row down, column right).
    """
    mask = np.zeros((48, 72, 88), dtype=np.uint16)
    low, high = 8, 24
    mask[low:high, low:high, low:high] = 1
    mask[low:high, low:high, high:high + ARM_LENGTHS["x"]] = ARMS["x"]
    mask[low:high, high:high + ARM_LENGTHS["y"], low:high] = ARMS["y"]
    mask[high:high + ARM_LENGTHS["z"], low:high, low:high] = ARMS["z"]
    letter = np.zeros((72, 88), dtype=bool)
    letter[24:56, 32:40] = True
    letter[24:32, 32:64] = True
    letter[38:46, 32:56] = True
    mask[10:22][:, letter] = 5
    return mask


@pytest.fixture(scope="module")
def phantom_algorithm():
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(chirality_phantom())
    return algorithm


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Replace polyscope by a module that records the vertices and triangles of every registered surface."""
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    recorder = types.ModuleType("polyscope")

    def register(name: str, vertices: np.ndarray, faces: np.ndarray, **_kwargs: object) -> None:
        surfaces[name] = (np.array(vertices), np.array(faces))

    recorder.register_surface_mesh = register
    for name in ("init", "remove_all_structures", "set_ground_plane_mode", "set_transparency_mode", "show"):
        setattr(recorder, name, lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "polyscope", recorder)
    return surfaces


def test_the_drawn_frame_is_the_images_column_row_and_slice(shown, phantom_algorithm):
    points, _, _ = phantom_algorithm.last_constructed_mesh
    plot_cells_polyscope(phantom_algorithm, show=False)
    drawn = shown["Cell 1"][0]
    column_row_slice = np.asarray(points)[:, ::-1]  # voxel units: the mask's (column, row, slice)
    for candidate, expected in ((drawn, +1.0), (np.asarray(points), -1.0)):  # the fix, and up to 0.7.0
        linear_map, *_ = np.linalg.lstsq(column_row_slice, candidate, rcond=None)
        assert np.allclose(column_row_slice @ linear_map, candidate, rtol=0, atol=1e-9)
        assert np.linalg.det(linear_map) == pytest.approx(expected, abs=1e-9)
    assert np.array_equal(drawn, column_row_slice)


def test_the_phantoms_arms_are_drawn_right_handed(shown, phantom_algorithm):
    plot_cells_polyscope(phantom_algorithm, show=False)

    def centre(label: int) -> np.ndarray:
        vertices, triangles = shown[f"Cell {label}"]
        return vertices[np.unique(triangles)].mean(axis=0)

    arms = {label: centre(label) - centre(1) for label in ARMS.values()}
    by_length = sorted(arms, key=lambda label: -np.linalg.norm(arms[label]))
    assert by_length == [ARMS["x"], ARMS["y"], ARMS["z"]]  # told apart by length alone, as in a picture
    directions = np.array([arms[label] / np.linalg.norm(arms[label]) for label in by_length])
    assert np.linalg.det(directions) > 0.95  # right-handed; the mirror image gives about -1
    # x, y, z each mostly along Polyscope's own x, y, z, in the positive direction
    assert np.argmax(np.abs(directions), axis=1).tolist() == [0, 1, 2]
    assert (np.diag(directions) > 0.9).all()


def test_every_triangle_faces_its_own_side_and_the_mesh_is_unchanged(shown, phantom_algorithm):
    points, triangles, labels = (np.array(a) for a in phantom_algorithm.last_constructed_mesh)
    plot_cells_polyscope(phantom_algorithm, show=False)
    plot_cells_polyscope(phantom_algorithm, view="Scattered", show=False)
    for label in range(1, 6):
        stored = triangles[np.any(labels == label, axis=1)]
        vertices, drawn = shown[f"Cell {label}"]
        assert np.array_equal(drawn, stored[:, [0, 2, 1]])
        a, b, c = (points[stored[:, k]] for k in range(3))
        stored_normals = np.cross(b - a, c - a)
        a, b, c = (vertices[drawn[:, k]] for k in range(3))
        assert np.allclose(np.cross(b - a, c - a), stored_normals[:, ::-1], rtol=0, atol=1e-9)
    # The signed volumes from the drawn arrays: +det to the first label, -det to the second, as the stored mesh's.
    drawn_points = points[:, ::-1]
    for label in range(1, 6):
        rows = np.any(labels == label, axis=1)
        sign = np.where(labels[rows, 0] == label, 1.0, -1.0)
        stored_volume = (sign * np.linalg.det(points[triangles[rows]])).sum() / 6
        drawn_volume = (sign * np.linalg.det(drawn_points[triangles[rows][:, [0, 2, 1]]])).sum() / 6
        assert stored_volume > 0
        assert drawn_volume == pytest.approx(stored_volume, rel=1e-12)
    assert np.array_equal(phantom_algorithm.last_constructed_mesh[0], points)
    assert np.array_equal(phantom_algorithm.last_constructed_mesh[1], triangles)
