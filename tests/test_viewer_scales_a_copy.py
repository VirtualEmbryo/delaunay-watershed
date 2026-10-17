"""`plot_cells_polyscope(anisotropy_factor != 1)` shows a scaled copy and never scales the stored mesh.

Before 0.7.0 it multiplied column 0 of the reconstruction's own points in place: each call compounded the factor,
and a later `save_to_rec_mesh` wrote the scaled points. Pins: two calls show the same points, scaled once; the
reconstruction's points are unchanged, bitwise; a later `save_to_rec_mesh` writes the unscaled points. polyscope is
replaced by a recorder (the real viewer needs an OpenGL context), so nothing is displayed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_rec
from dw3d.viewing import plot_cells_polyscope


def two_cells(n: int = 40) -> np.ndarray:
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[(z - 20) ** 2 + (y - 14) ** 2 + (x - 20) ** 2 < 9**2] = 1
    mask[(z - 20) ** 2 + (y - 27) ** 2 + (x - 20) ** 2 < 8**2] = 2
    return mask


@pytest.fixture
def shown_vertices(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Replace polyscope by a module that records the vertices of every registered surface."""
    shown: list[np.ndarray] = []
    recorder = types.ModuleType("polyscope")
    recorder.register_surface_mesh = lambda _name, vertices, _faces, **_kwargs: shown.append(np.array(vertices))
    for name in ("init", "remove_all_structures", "set_ground_plane_mode", "set_transparency_mode", "show"):
        setattr(recorder, name, lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "polyscope", recorder)
    return shown


def test_two_calls_show_the_same_points_and_leave_the_mesh_unchanged(shown_vertices: list[np.ndarray]) -> None:
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(two_cells())
    stored = algorithm.last_constructed_mesh[0].copy()

    plot_cells_polyscope(algorithm, anisotropy_factor=3.0, show=False)
    first = shown_vertices[0]
    plot_cells_polyscope(algorithm, anisotropy_factor=3.0, show=False)
    second = shown_vertices[-1]

    assert np.array_equal(first, second)
    assert np.array_equal(first, stored * np.array([3.0, 1.0, 1.0]))
    assert np.array_equal(algorithm.last_constructed_mesh[0], stored)


def test_a_later_save_writes_the_unscaled_points(shown_vertices: list[np.ndarray], tmp_path: Path) -> None:
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(two_cells())
    stored = algorithm.last_constructed_mesh[0].copy()

    plot_cells_polyscope(algorithm, anisotropy_factor=3.0, show=False)
    plot_cells_polyscope(algorithm, anisotropy_factor=3.0, view="Scattered", show=False)
    algorithm.save_to_rec_mesh(tmp_path / "after_viewing.rec")

    assert shown_vertices  # the viewer did run
    points, _, _ = load_rec(tmp_path / "after_viewing.rec")
    assert np.allclose(points, stored)
    assert np.array_equal(algorithm.last_constructed_mesh[0], stored)
