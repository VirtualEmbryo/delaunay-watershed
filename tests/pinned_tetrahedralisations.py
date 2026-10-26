"""The stored tetrahedralisations of the golden images, per configuration, and helpers to inject them.

See `tests/test_canonical_tetrahedron_order.py` for why the golden masters are pinned to them.
Every configuration with golden fixtures places its points on the integer lattice, where Qhull's
tetrahedralisation of a set of cospherical points (of co-power-spherical weighted points, for the
configurations with junction protection) depends on floating-point rounding and differs between
machines; the mesh, hence the fixture, follows that choice. The fixture tests therefore reconstruct
from a stored tetrahedralisation, and a separate check per configuration certifies, exactly, that
Qhull's own tetrahedralisation on the machine running the tests differs from the stored one only by
another such choice.

* `tests/golden/default_tetrahedralisations.npz`: the default configuration, per golden image and
  `min_distance` (3, 5), its points and Qhull's tetrahedra in canonical order, as uint16; made on macOS
  arm64 with SciPy 1.18.1.
* `tests/golden/opt_in_configuration_tetrahedralisations.npz`: the same for the point sets of the
  opt-in configurations (`TESSELATED_POINT_SETS`), with the weights of the weighted ones; made by
  `tests/make_opt_in_configuration_tetrahedralisations.py` on macOS arm64 with SciPy 1.18.1.
"""

from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest

import dw3d
from dw3d.tesselation import simple_delaunay_tesselation
from tests import exact_delaunay_checks as exact
from tests.conftest import GOLDEN_DIR, load_image_or_skip

PINNED_TETRAHEDRALISATIONS = GOLDEN_DIR / "default_tetrahedralisations.npz"
OPT_IN_TETRAHEDRALISATIONS = GOLDEN_DIR / "opt_in_configuration_tetrahedralisations.npz"

#: The start of the skip message of every test that needs Qhull's own tetrahedralisation to be the stored
#: one. The CI reads it: such a skip is reported as a warning, any other skip of the cross-machine checks
#: fails the job (`tests/report_cross_machine_checks.py`).
COSPHERICAL_SKIP = "Qhull triangulates the cospherical sets differently here"

#: The configurations with golden fixtures, by the name their fixture directory and tests use, and the
#: point set each one tesselates. The dithered configuration places the default's points, and the four
#: junction-protected variants place the same weighted points (checked by the script that stores them
#: and by the injected function), so each group shares one stored tetrahedralisation.
CONFIGURATION_GETTERS = {
    "default": dw3d.get_default_mesh_reconstruction_algorithm,
    "deterministic": dw3d.get_deterministic_mesh_reconstruction_algorithm,
    "dithered": dw3d.get_dithered_mesh_reconstruction_algorithm,
    "boundary_layer": dw3d.get_boundary_layer_mesh_reconstruction_algorithm,
    "junction_protected": dw3d.get_junction_protected_mesh_reconstruction_algorithm,
    "link_checked": dw3d.get_link_checked_mesh_reconstruction_algorithm,
    "cubic_score": partial(dw3d.get_junction_protected_mesh_reconstruction_algorithm, spline_order=3),
    "offset_excluded": partial(
        dw3d.get_junction_protected_mesh_reconstruction_algorithm,
        exclude_offsets_from_surface=True,
    ),
}
TESSELATED_POINT_SETS = {
    "default": "default",
    "deterministic": "deterministic",
    "dithered": "default",
    "boundary_layer": "boundary_layer",
    "junction_protected": "junction_protected",
    "link_checked": "junction_protected",
    "cubic_score": "junction_protected",
    "offset_excluded": "junction_protected",
}
#: One stored tetrahedralisation per point set of an opt-in configuration.
OPT_IN_POINT_SETS = ("deterministic", "boundary_layer", "junction_protected")


def _key(image_name: str, min_distance: int) -> str:
    return f"{Path(image_name).stem}_min_distance_{min_distance}"


def pinned_tetrahedralisation(image_name: str, min_distance: int) -> tuple[np.ndarray, np.ndarray]:
    """The stored points and canonical tetrahedra of the default configuration on one golden image."""
    points, tetrahedra, _ = pinned_configuration_tetrahedralisation(image_name, min_distance, "default")
    return points, tetrahedra


def pinned_configuration_tetrahedralisation(
    image_name: str,
    min_distance: int,
    configuration: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The stored points, canonical tetrahedra and weights (None if unweighted) of one configuration on one image."""
    point_set = TESSELATED_POINT_SETS[configuration]
    if point_set == "default":
        with np.load(PINNED_TETRAHEDRALISATIONS) as stored:
            key = _key(image_name, min_distance)
            return stored[f"{key}_points"].astype(np.float64), stored[f"{key}_tetrahedra"].astype(np.int64), None
    with np.load(OPT_IN_TETRAHEDRALISATIONS) as stored:
        key = f"{point_set}_{_key(image_name, min_distance)}"
        weights = stored[f"{key}_weights"] if f"{key}_weights" in stored.files else None
        return stored[f"{key}_points"].astype(np.float64), stored[f"{key}_tetrahedra"].astype(np.int64), weights


def pinned_tesselation_function(image_name: str, min_distance: int, configuration: str = "default"):
    """A tesselation function giving the stored tetrahedra, after checking the points and weights are stored ones."""
    stored_points, stored_tetrahedra, stored_weights = pinned_configuration_tetrahedralisation(
        image_name,
        min_distance,
        configuration,
    )

    def tesselate(points, weights=None):
        points = np.asarray(points, dtype=np.float64)
        assert np.array_equal(points, stored_points), "the point placement no longer gives the stored points"
        if stored_weights is not None:
            assert np.array_equal(np.asarray(weights, dtype=np.float64), stored_weights), (
                "the point placement no longer gives the stored weights"
            )
        return points, stored_tetrahedra.copy()

    return tesselate


def placed_points_and_weights(
    image_name: str,
    min_distance: int,
    configuration: str = "default",
) -> tuple[np.ndarray, np.ndarray | None]:
    """The points (and weights, None if the configuration places none) a configuration hands to its tesselation."""
    mask = load_image_or_skip(image_name)
    algorithm = CONFIGURATION_GETTERS[configuration](min_distance=min_distance, print_info=False)
    placed = algorithm.point_placing_function(mask, algorithm.edt_creation_function(mask))
    weights = placed[2] if len(placed) >= 3 else None
    return np.asarray(placed[0], dtype=np.float64), None if weights is None else np.asarray(weights, dtype=np.float64)


def _placed_points(image_name: str, min_distance: int) -> np.ndarray:
    """The points the default configuration hands to Qhull on one golden image."""
    return placed_points_and_weights(image_name, min_distance)[0]


def live_tetrahedralisation(image_name: str, min_distance: int, configuration: str = "default") -> np.ndarray:
    """The tetrahedra the configuration's own tesselation function (Qhull) gives on this machine."""
    points, weights = placed_points_and_weights(image_name, min_distance, configuration)
    algorithm = CONFIGURATION_GETTERS[configuration](min_distance=min_distance, print_info=False)
    _, tetrahedra = algorithm.tesselation_creation_function(points, weights)
    return np.asarray(tetrahedra, dtype=np.int64)


@cache
def qhull_difference_from_the_pinned_tetrahedralisation(
    image_name: str,
    min_distance: int,
    configuration: str = "default",
) -> tuple[int, bool]:
    """How Qhull's tetrahedralisation differs from the stored one on this machine.

    Returns:
        The number of tetrahedra in one and not the other (both directions), and whether every one of
        them lies inside a cospherical set (co-power-spherical for weighted points; exact checks; True
        when there is no difference).
    """
    if configuration == "default":
        points = placed_points_and_weights(image_name, min_distance)[0]
        _, live = simple_delaunay_tesselation(points)
    else:
        points = placed_points_and_weights(image_name, min_distance, configuration)[0]
        live = live_tetrahedralisation(image_name, min_distance, configuration)
    stored_points, stored, weights = pinned_configuration_tetrahedralisation(image_name, min_distance, configuration)
    assert np.array_equal(points, stored_points), "the point placement no longer gives the stored points"
    n_differing = exact.differing_tetrahedra(stored, live)
    if not n_differing:
        return 0, True
    integer = exact.integer_points(points)
    if weights is None:
        return n_differing, exact.difference_lies_inside_cospherical_sets(integer, stored, live)
    return n_differing, exact.difference_lies_inside_copower_sets(integer, weights, stored, live)


def qhull_differs_from_the_pinned_tetrahedralisation(image_name: str, min_distance: int) -> int:
    """The number of tetrahedra in which Qhull's tetrahedralisation differs from the stored one, on this machine."""
    return qhull_difference_from_the_pinned_tetrahedralisation(image_name, min_distance)[0]


def skip_where_qhull_differs_inside_cospherical_sets(
    image_name: str,
    min_distance: int,
    configuration: str = "default",
) -> None:
    """Skip a test needing Qhull's own tetrahedralisation where it is not the stored one; fail outside cospherical sets.

    A difference inside cospherical sets is another Delaunay choice, made by floating-point rounding:
    the test cannot run here and says so (`COSPHERICAL_SKIP`, read by the CI). A difference anywhere
    else means Qhull's tetrahedralisation is not a Delaunay one of these points, a defect, so the test fails.
    """
    n_differing, inside = qhull_difference_from_the_pinned_tetrahedralisation(image_name, min_distance, configuration)
    if not n_differing:
        return
    if not inside:
        pytest.fail(
            "Qhull's tetrahedralisation differs from the stored one outside cospherical sets: "
            f"{n_differing} tetrahedra",
        )
    pytest.skip(f"{COSPHERICAL_SKIP}: {n_differing} tetrahedra differ, all inside cospherical sets (exact check)")
