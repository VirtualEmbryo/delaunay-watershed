"""`plot_cells_polyscope` draws a 2-D mesh in a 2-D window: the mesh decides the dimension, as for the reconstruction.

Pins, on a three-cell mask whose cells are told apart by their place in the image (an L-shaped cell at the top left, a
square at the top right, a bar along the bottom):

* the three inputs give the same scene: dw3d's algorithm after the 2-D mask, dw2d's algorithm, dw2d's three arrays;
* each cell is one curve network, its own closed boundary (every node of degree 2, every segment of the label once), in
  the colour the 3-D view gives the same label;
* the frame is `(x, y) = (column, -row)`: a rotation (determinant +1), row 0 at the top and column 0 at the left;
* the camera looks straight down on the plane, the navigation is planar while the window is open and restored after;
* Scattered moves each cell by the 3-D view's formula; the image, when given, lies under the curves on the pixel grid,
  scaled by the pixel size of a mesh built in physical units;
* an option without a 2-D meaning, and an image under a 3-D mesh, raise `ValueError` naming the option;
* the stored mesh is unchanged.

polyscope is replaced by a recorder (the real viewer needs an OpenGL context), so nothing is displayed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import dw2d
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.viewing import plot_cells_polyscope

LABELS = (1, 2, 3)


def three_cell_mask() -> np.ndarray:
    """`(row, column)` labels: an L-shaped cell 1 at the top left, a square 2 at the top right, a bar 3 below."""
    mask = np.zeros((70, 90), dtype=np.uint16)
    mask[8:40, 8:30] = 1
    mask[8:22, 30:44] = 1
    mask[8:40, 48:80] = 2
    mask[44:62, 8:80] = 3
    return mask


def three_cube_mask() -> np.ndarray:
    """A 3-D mask with the same three labels, for the colours of the 3-D view."""
    mask = np.zeros((24, 24, 60), dtype=np.uint16)
    mask[4:20, 4:20, 4:20] = 1
    mask[4:20, 4:20, 22:38] = 2
    mask[4:20, 4:20, 40:56] = 3
    return mask


class Recorder(types.ModuleType):
    """A stand-in for polyscope that records what is registered and how the window is set up."""

    def __init__(self) -> None:  # noqa: D107 - the records, empty
        super().__init__("polyscope")
        self.curves: dict[str, dict] = {}
        self.surfaces: dict[str, dict] = {}
        self.navigation = ["turntable"]
        self.looks: list[tuple] = []
        self.calls: list[str] = []

    def register_curve_network(self, name: str, nodes: np.ndarray, edges: np.ndarray, **kwargs: object) -> None:
        """Record a curve network's nodes, edges and options."""
        self.curves[name] = {"nodes": np.array(nodes), "edges": np.array(edges), **kwargs}

    def register_surface_mesh(self, name: str, vertices: np.ndarray, faces: np.ndarray, **kwargs: object) -> object:
        """Record a surface's vertices, faces and quantities; return a structure that records them."""
        record = {"vertices": np.array(vertices), "faces": np.array(faces), "quantities": {}, **kwargs}
        self.surfaces[name] = record

        class Structure:
            def add_parameterization_quantity(self, quantity: str, values: np.ndarray, **kw: object) -> None:
                """Record a parameterisation."""
                record["quantities"][quantity] = (np.array(values), kw)

            def add_color_quantity(self, quantity: str, values: np.ndarray, **kw: object) -> None:
                """Record a colour quantity."""
                record["quantities"][quantity] = (np.array(values), kw)

            def set_material(self, *_args: object) -> None:
                """Ignore the material."""
                pass

            def set_back_face_policy(self, *_args: object) -> None:
                """Ignore the back-face policy."""
                pass

        return Structure()

    def get_navigation_style(self) -> str:
        """The current navigation style."""
        return self.navigation[-1]

    def set_navigation_style(self, style: str) -> None:
        """Record a navigation style."""
        self.navigation.append(style)

    def look_at(self, camera: tuple, target: tuple) -> None:
        """Record the camera's position and target."""
        self.looks.append((camera, target))

    def get_window_size(self) -> tuple[int, int]:
        """A window size, as polyscope's."""
        return (1024, 768)

    def get_vertical_fov_degrees(self) -> float:
        """A vertical field of view, as polyscope's."""
        return 45.0

    def show(self) -> None:
        """Record the navigation style the window opens with."""
        self.calls.append(f"show with navigation {self.navigation[-1]}")

    def __getattr__(self, name: str) -> object:
        """Any other polyscope function: record its name."""
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *_args, **_kwargs: self.calls.append(name)


@pytest.fixture
def ps(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setitem(sys.modules, "polyscope", recorder)
    return recorder


@pytest.fixture(scope="module")
def dw3d_algorithm():
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(three_cell_mask())
    return algorithm


@pytest.fixture(scope="module")
def dw2d_algorithm():
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(three_cell_mask())
    return algorithm


def _scene(recorder: Recorder) -> dict[str, tuple]:
    return {name: (c["nodes"], c["edges"], tuple(np.asarray(c["color"]))) for name, c in recorder.curves.items()}


def test_the_three_inputs_give_the_same_scene(ps, dw3d_algorithm, dw2d_algorithm):
    scenes = []
    for given in (dw3d_algorithm, dw2d_algorithm, dw2d_algorithm.last_constructed_mesh):
        ps.curves.clear()
        plot_cells_polyscope(given, show=False)
        scenes.append(_scene(ps))
    assert sorted(scenes[0]) == [f"Cell {k}" for k in LABELS]
    for other in scenes[1:]:
        assert other.keys() == scenes[0].keys()
        for name in scenes[0]:
            for a, b in zip(scenes[0][name], other[name], strict=True):
                np.testing.assert_array_equal(a, b)
    assert not ps.surfaces  # no image given, no cell filled


def test_each_cell_is_its_closed_boundary_in_the_colour_of_the_3d_view(ps, dw3d_algorithm):
    points, segments, labels = dw3d_algorithm.last_constructed_mesh
    plot_cells_polyscope(dw3d_algorithm, show=False)
    colours_2d = {name: c["color"] for name, c in ps.curves.items()}
    for k in LABELS:
        curve = ps.curves[f"Cell {k}"]
        own = segments[(labels == k).any(axis=1)]
        assert len(curve["edges"]) == len(own)
        degree = np.bincount(curve["edges"].reshape(-1), minlength=len(curve["nodes"]))
        assert np.all(degree == 2)  # a closed boundary: every node on two of its segments
        drawn = {tuple(p) for p in curve["nodes"][:, :2].tolist()}
        expected = {(float(c), -float(r)) for r, c in points[np.unique(own)]}
        assert drawn == expected

    algorithm_3d = get_default_mesh_reconstruction_algorithm()
    algorithm_3d.construct_mesh_from_segmentation_mask(three_cube_mask())
    shown_3d: dict[str, np.ndarray] = {}
    ps.register_surface_mesh = lambda name, _v, _f, **kw: shown_3d.__setitem__(name, kw["color"])
    plot_cells_polyscope(algorithm_3d, show=False)
    for k in LABELS:
        np.testing.assert_array_equal(colours_2d[f"Cell {k}"], shown_3d[f"Cell {k}"])


def test_the_frame_is_column_and_minus_row_a_rotation(ps, dw3d_algorithm):
    plot_cells_polyscope(dw3d_algorithm, show=False)
    centre = {k: ps.curves[f"Cell {k}"]["nodes"].mean(axis=0) for k in LABELS}
    # cell 1 at the top left, 2 at the top right, 3 at the bottom: x grows to the right, y upwards on screen
    assert centre[1][0] < centre[2][0]
    assert centre[3][1] < min(centre[1][1], centre[2][1])
    assert all(np.all(ps.curves[f"Cell {k}"]["nodes"][:, 2] == 0.0) for k in LABELS)
    rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])  # (row, column) -> (column, -row)
    assert np.linalg.det(rotation) == 1.0
    points, _, _ = dw3d_algorithm.last_constructed_mesh
    from dw3d.viewing import _drawn_planar_points

    np.testing.assert_array_equal(_drawn_planar_points(points)[:, :2], points @ rotation.T)


def test_the_camera_looks_straight_down_with_planar_navigation_restored_after(ps, dw3d_algorithm):
    plot_cells_polyscope(dw3d_algorithm, show=True)
    assert "show with navigation planar" in ps.calls
    assert ps.navigation[-1] == "turntable"
    ((camera, target),) = ps.looks
    assert camera[:2] == target[:2]
    assert target[2] == 0.0
    assert camera[2] > 0.0


def test_scattered_moves_each_cell_by_the_formula_of_the_3d_view(ps, dw3d_algorithm):
    points, segments, labels = dw3d_algorithm.last_constructed_mesh
    plot_cells_polyscope(dw3d_algorithm, show=False)
    simple = {name: c["nodes"].copy() for name, c in ps.curves.items()}
    plot_cells_polyscope(dw3d_algorithm, show=False, view="Scattered", scattering_coeff=0.5, image=three_cell_mask())
    centroid_mesh = points[segments].reshape(-1, 2).mean(axis=0)
    for k in LABELS:
        own = segments[(labels == k).any(axis=1)]
        move = 0.5 * (points[own].reshape(-1, 2).mean(axis=0) - centroid_mesh)
        np.testing.assert_allclose(
            ps.curves[f"Cell {k}"]["nodes"][:, :2] - simple[f"Cell {k}"][:, :2],
            np.tile([move[1], -move[0]], (len(simple[f"Cell {k}"]), 1)),
            rtol=0,
            atol=1e-12,
        )
    assert "Image" not in ps.surfaces  # the image is not drawn in the scattered view


@pytest.mark.parametrize("pixel_size", [1.0, 0.5])
def test_the_image_lies_under_the_curves_on_the_pixel_grid(ps, pixel_size):
    mask = three_cell_mask()
    algorithm = get_default_mesh_reconstruction_algorithm()
    if pixel_size == 1.0:
        algorithm.construct_mesh_from_segmentation_mask(mask)
    else:
        algorithm.construct_mesh_in_physical_units(mask, spacing=(pixel_size, pixel_size), spacing_unit="um")
    plot_cells_polyscope(algorithm, show=False, image=mask)
    quad = ps.surfaces["Image"]
    rows, columns = mask.shape
    corners = np.array([[-0.5, -0.5], [-0.5, columns - 0.5], [rows - 0.5, columns - 0.5], [rows - 0.5, -0.5]])
    drawn_corners = np.column_stack([corners[:, 1], -corners[:, 0]]) * pixel_size
    np.testing.assert_array_equal(quad["vertices"][:, :2], drawn_corners)
    assert np.all(quad["vertices"][:, 2] < 0.0)  # under the curves
    pixels, options = quad["quantities"]["pixels"]
    assert pixels.shape == (rows, columns, 3)
    assert options["defined_on"] == "texture"
    assert options["filter_mode"] == "nearest"
    # the mask's labels as grey levels: the medium darkest, label 3 lightest
    assert pixels[0, 0, 0] == 0.0
    assert pixels[50, 40, 0] == 1.0


def test_options_without_a_meaning_in_the_mesh_dimension_raise_naming_them(ps, dw3d_algorithm):
    with pytest.raises(ValueError, match="anisotropy_factor"):
        plot_cells_polyscope(dw3d_algorithm, anisotropy_factor=2.0, show=False)
    with pytest.raises(ValueError, match="image"):
        plot_cells_polyscope(dw3d_algorithm, image=np.zeros((3, 3, 2)), show=False)
    with pytest.raises(ValueError, match="view"):
        plot_cells_polyscope(dw3d_algorithm, view="Exploded", show=False)
    algorithm_3d = get_default_mesh_reconstruction_algorithm()
    algorithm_3d.construct_mesh_from_segmentation_mask(three_cube_mask())
    with pytest.raises(ValueError, match="image"):
        plot_cells_polyscope(algorithm_3d, image=np.zeros((24, 24)), show=False)
    with pytest.raises(ValueError, match="three arrays"):
        plot_cells_polyscope(object(), show=False)
    assert not ps.curves  # every refusal happens before anything is drawn


@pytest.mark.usefixtures("ps")
def test_the_stored_mesh_is_unchanged(dw3d_algorithm):
    before = [np.array(a, copy=True) for a in dw3d_algorithm.last_constructed_mesh]
    plot_cells_polyscope(dw3d_algorithm, show=False, view="Scattered")
    plot_cells_polyscope(dw3d_algorithm, show=False, image=three_cell_mask())
    for a, b in zip(before, dw3d_algorithm.last_constructed_mesh, strict=True):
        np.testing.assert_array_equal(a, b)
