"""The canonical tetrahedron order, and the tetrahedralisation the default golden master is pinned to.

Qhull returns the tetrahedra of a Delaunay tetrahedralisation in a row order that depends on the
order of the input points, and dw3d used to read that order (the watershed's tie-break between
equal scores, the seed lookup, mesh surgery's tetrahedron cycles). Since 0.7.1 the tetrahedra are
put in canonical order first (`dw3d.tesselation.canonical_tetrahedron_order`), so the mesh is a
function of the tetrahedralisation alone. `canonical_tetrahedron_order=False` keeps Qhull's row
order; `tests/golden/qhull_tetrahedron_order/` (and its twin without relocation) hold the default
configuration's 0.7 fixtures, moved byte for byte, to show that the legacy order still reproduces
them.

What remains machine-dependent is the tetrahedralisation itself: dw3d's points lie on the integer
lattice, where five or more points are often cospherical, and inside such a set Qhull chooses
among several Delaunay tetrahedralisations by floating-point rounding. The default golden master
is therefore checked against a stored tetrahedralisation (`tests/golden/default_tetrahedralisations.npz`:
the points and Qhull's tetrahedra in canonical order, for the default configuration on the four
golden images at `min_distance` 3 and 5, made on macOS arm64 with SciPy 1.18.1). Given it, the
golden master does not depend on Qhull. Exact integer checks verify, on every machine, that the
stored set is a Delaunay tetrahedralisation of the points and that Qhull's own differs from it
only by another Delaunay choice; the end-to-end test with Qhull's own tetrahedralisation runs where
the two are equal and is skipped, with the number of differing tetrahedra, where they differ only
inside cospherical sets (checked exactly); a difference anywhere else fails it.
"""

import warnings

import numpy as np
import pytest

import dw3d
from dw3d.tesselation import canonical_tetrahedron_order, regular_tesselation, simple_delaunay_tesselation
from tests import exact_delaunay_checks as exact
from tests.conftest import IMAGE_NAMES, load_image_or_skip
from tests.pinned_tetrahedralisations import (
    _placed_points,
    pinned_configuration_tetrahedralisation,
    pinned_tesselation_function,
    pinned_tetrahedralisation,
    skip_where_qhull_differs_inside_cospherical_sets,
)

MIN_DISTANCES = [3, 5]
SMALL_CASE = ("1.tif", 5)


def _row_permuted(seed: int, tesselation=None):
    """`tesselation`'s tetrahedra (Qhull's by default) with their rows permuted."""
    tesselation = simple_delaunay_tesselation if tesselation is None else tesselation

    def tesselate(points, weights=None):
        points, tetrahedra = tesselation(points, weights)
        return points, tetrahedra[np.random.default_rng(seed).permutation(len(tetrahedra))]

    return tesselate


def _reconstruct(mask: np.ndarray, min_distance: int, canonical: bool, tesselation=None) -> tuple:
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=min_distance)
    algorithm.canonical_tetrahedron_order = canonical
    if tesselation is not None:
        algorithm.tesselation_creation_function = tesselation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dw3d.CoarseSpacingRelocationWarning)
        mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    return tuple(np.array(a) for a in mesh)


def _same(a: tuple, b: tuple) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_canonical_order_sorts_rows_and_tetrahedra_and_keeps_the_set():
    tetrahedra = np.array([[5, 1, 3, 2], [0, 9, 4, 1], [3, 2, 1, 0]])
    result = canonical_tetrahedron_order(tetrahedra)
    np.testing.assert_array_equal(result, [[0, 1, 2, 3], [0, 1, 4, 9], [1, 2, 3, 5]])
    assert result.dtype == np.int64
    permuted = tetrahedra[[2, 0, 1]][:, [3, 1, 0, 2]]
    np.testing.assert_array_equal(canonical_tetrahedron_order(permuted), result)


def test_the_factory_and_every_named_configuration_default_to_the_canonical_order():
    assert dw3d.MeshReconstructionAlgorithmFactory().make_algorithm().canonical_tetrahedron_order is True
    legacy = dw3d.MeshReconstructionAlgorithmFactory().set_canonical_tetrahedron_order(False).make_algorithm()
    assert legacy.canonical_tetrahedron_order is False
    for getter in (
        dw3d.get_default_mesh_reconstruction_algorithm,
        dw3d.get_deterministic_mesh_reconstruction_algorithm,
        dw3d.get_dithered_mesh_reconstruction_algorithm,
        dw3d.get_boundary_layer_mesh_reconstruction_algorithm,
        dw3d.get_junction_protected_mesh_reconstruction_algorithm,
        dw3d.get_link_checked_mesh_reconstruction_algorithm,
    ):
        assert getter().canonical_tetrahedron_order is True, getter.__name__


def test_the_mesh_does_not_depend_on_the_tetrahedra_row_order():
    """Three row permutations of the tetrahedra give the bitwise-identical mesh under the canonical order.

    On the stored tetrahedralisation, so that the tetrahedra permuted are the same on every machine.
    """
    image_name, min_distance = SMALL_CASE
    mask = load_image_or_skip(image_name)
    stored = pinned_tesselation_function(image_name, min_distance)
    reference = _reconstruct(mask, min_distance, canonical=True, tesselation=stored)
    for seed in (1, 2, 3):
        permuted = _row_permuted(seed, stored)
        assert _same(_reconstruct(mask, min_distance, canonical=True, tesselation=permuted), reference)


def test_negative_control_qhull_row_order_changes_the_mesh():
    """With the legacy order, the same row permutations do change the mesh: the test above is not vacuous.

    On the stored tetrahedralisation: whether three given permutations change the mesh depends on which
    tetrahedra are permuted, and Qhull's own split of cospherical point sets differs between machines.
    """
    image_name, min_distance = SMALL_CASE
    mask = load_image_or_skip(image_name)
    stored = pinned_tesselation_function(image_name, min_distance)
    reference = _reconstruct(mask, min_distance, canonical=False, tesselation=stored)
    changed = [
        not _same(_reconstruct(mask, min_distance, canonical=False, tesselation=_row_permuted(seed, stored)), reference)
        for seed in (1, 2, 3)
    ]
    assert any(changed)


def test_the_exact_insphere_sign_is_calibrated():
    assert exact.INSIDE_SIGN in (-1, 1)
    points = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [9, 9, 9]], dtype=np.int64)
    tetrahedron = np.array([[0, 1, 2, 3]])
    insphere = exact._insphere(points, tetrahedron, np.array([4]))[0]
    outside = np.sign(insphere * exact.orientation(points, tetrahedron)[0])
    assert outside == -exact.INSIDE_SIGN


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
def test_the_pinned_tetrahedralisation_is_delaunay_and_qhull_differs_only_by_another_delaunay_choice(
    image_name,
    min_distance,
):
    """Exact, on every machine: the stored set and Qhull's are both Delaunay tetrahedralisations of the same points."""
    points = _placed_points(image_name, min_distance)
    stored_points, stored = pinned_tetrahedralisation(image_name, min_distance)
    np.testing.assert_array_equal(points, stored_points)
    np.testing.assert_array_equal(canonical_tetrahedron_order(stored), stored)
    integer = exact.integer_points(points)
    _, live = simple_delaunay_tesselation(points)
    assert exact.count_non_delaunay_faces(integer, stored) == 0
    assert exact.count_non_delaunay_faces(integer, live) == 0
    assert exact.covered_volume_times_six(integer, stored) == exact.covered_volume_times_six(integer, live)


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
def test_qhull_gives_the_pinned_tetrahedralisation(image_name, min_distance):
    """Qhull gives the stored tetrahedralisation here; where it does not, the live golden checks skip and say so."""
    skip_where_qhull_differs_inside_cospherical_sets(image_name, min_distance)
    points = _placed_points(image_name, min_distance)
    _, stored = pinned_tetrahedralisation(image_name, min_distance)
    _, live = simple_delaunay_tesselation(points)
    np.testing.assert_array_equal(canonical_tetrahedron_order(live), stored)


def _two_triangulations_of_a_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cube's eight (cospherical) corners and two of its triangulations, around two different main diagonals.

    Each is the six-tetrahedra Kuhn split: one tetrahedron per order in which the three axes are walked
    from one corner to the opposite one. Every tetrahedron of either is inscribed in the cube's sphere.
    """
    corners = [(x, y, z) for x in (0, 2) for y in (0, 2) for z in (0, 2)]
    index = {corner: i for i, corner in enumerate(corners)}

    def kuhn(start: tuple[int, int, int], steps: tuple[int, int, int]) -> np.ndarray:
        tetrahedra = []
        for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            vertex, path = list(start), [index[start]]
            for axis in order:
                vertex[axis] += steps[axis]
                path.append(index[tuple(vertex)])
            tetrahedra.append(path)
        return np.array(tetrahedra)

    points = np.array(corners, dtype=np.int64)
    return points, kuhn((0, 0, 0), (2, 2, 2)), kuhn((2, 0, 0), (-2, 2, 2))


def test_a_difference_inside_a_cospherical_set_is_certified_and_one_outside_is_not():
    """The exact certificate behind the CI's warning: a cube split two ways passes, a non-Delaunay split fails."""
    points, first, second = _two_triangulations_of_a_cube()
    assert exact.differing_tetrahedra(first, second) == 12
    assert exact.difference_lies_inside_cospherical_sets(points, first, second)
    assert exact.difference_lies_inside_cospherical_sets(points, first, first)
    # Negative control: five points in general position, triangulated by the Delaunay two-tetrahedra
    # split and by the non-Delaunay three-tetrahedra split of the same bipyramid.
    bipyramid = np.array([[0, 0, 0], [5, 1, 0], [1, 4, 0], [2, 2, 3], [2, 1, -4]], dtype=np.int64)
    two = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    three = np.array([[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]])
    delaunay = two if exact.count_non_delaunay_faces(bipyramid, two) == 0 else three
    other = three if delaunay is two else two
    assert exact.count_non_delaunay_faces(bipyramid, delaunay) == 0
    assert not exact.difference_lies_inside_cospherical_sets(bipyramid, delaunay, other)


def _flip_one_strictly_delaunay_face(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Replace two tetrahedra sharing a face, no fifth point on their sphere, by the three around their far edge."""
    first, second, opposite = exact.face_neighbours(tetrahedra)
    for t1, t2, e in zip(first.tolist(), second.tolist(), opposite.tolist(), strict=True):
        face = [v for v in tetrahedra[t1] if v in tetrahedra[t2]]
        (d,) = [v for v in tetrahedra[t1] if v not in face]
        if exact._insphere(points, tetrahedra[[t1]], np.array([e]))[0] == 0:
            continue
        a, b, c = face
        three = np.array([[a, b, d, e], [b, c, d, e], [c, a, d, e]])
        if np.any(exact.orientation(points, three) == 0):
            continue
        two_volume = exact.covered_volume_times_six(points, tetrahedra[[t1, t2]])
        if exact.covered_volume_times_six(points, three) != two_volume:
            continue  # the edge d-e does not cross the face: not a valid flip
        keep = np.ones(len(tetrahedra), dtype=bool)
        keep[[t1, t2]] = False
        return np.vstack([tetrahedra[keep], three])
    message = "no flippable face found"
    raise AssertionError(message)


def test_the_certificate_on_a_golden_image_accepts_another_qhull_choice_and_rejects_a_flipped_face():
    """Real lattice points: Qhull on permuted input splits cospherical sets otherwise (accepted); a 2-3 flip is not."""
    image_name, min_distance = SMALL_CASE
    points = _placed_points(image_name, min_distance)
    integer = exact.integer_points(points)
    _, stored = pinned_tetrahedralisation(image_name, min_distance)
    permutation = np.random.default_rng(1).permutation(len(points))
    _, permuted = simple_delaunay_tesselation(points[permutation])
    other_choice = permutation[permuted]
    assert exact.differing_tetrahedra(stored, other_choice) > 0  # 18 on macOS arm64, SciPy 1.18.1
    assert exact.difference_lies_inside_cospherical_sets(integer, stored, other_choice)
    flipped = _flip_one_strictly_delaunay_face(integer, stored)
    assert exact.differing_tetrahedra(stored, flipped) == 5
    assert not exact.difference_lies_inside_cospherical_sets(integer, stored, flipped)


def _flip_one_strictly_regular_face(points: np.ndarray, weights: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """The weighted `_flip_one_strictly_delaunay_face`: a 2-3 flip of a face that no fifth point makes degenerate."""
    scaled, scale = exact.exact_weights(weights)
    first, second, opposite = exact.face_neighbours(tetrahedra)
    for t1, t2, e in zip(first.tolist(), second.tolist(), opposite.tolist(), strict=True):
        face = [v for v in tetrahedra[t1] if v in tetrahedra[t2]]
        (d,) = [v for v in tetrahedra[t1] if v not in face]
        if exact._power_insphere(points, scaled, scale, tetrahedra[[t1]], np.array([e]))[0] == 0:
            continue
        a, b, c = face
        three = np.array([[a, b, d, e], [b, c, d, e], [c, a, d, e]])
        if np.any(exact.orientation(points, three) == 0):
            continue
        two_volume = exact.covered_volume_times_six(points, tetrahedra[[t1, t2]])
        if exact.covered_volume_times_six(points, three) != two_volume:
            continue
        keep = np.ones(len(tetrahedra), dtype=bool)
        keep[[t1, t2]] = False
        return np.vstack([tetrahedra[keep], three])
    message = "no flippable face found"
    raise AssertionError(message)


def test_the_weighted_certificate_reduces_to_the_unweighted_one_and_rejects_a_flipped_face():
    """Zero weights give the Delaunay certificate; on real weighted points another Qhull choice passes, a flip fails.

    On a junction-protected golden image, another Qhull choice of the weighted points is accepted and a 2-3
    flip of a strictly regular face is not.
    """
    points, first, second = _two_triangulations_of_a_cube()
    zero = np.zeros(len(points))
    assert exact.difference_lies_inside_copower_sets(points, zero, first, second)
    assert exact.count_non_regular_faces(points, zero, first) == exact.count_non_delaunay_faces(points, first) == 0

    image_name, min_distance = SMALL_CASE
    stored_points, stored, weights = pinned_configuration_tetrahedralisation(
        image_name,
        min_distance,
        "junction_protected",
    )
    assert weights is not None
    assert np.any(weights)
    integer = exact.integer_points(stored_points)
    assert exact.count_non_regular_faces(integer, weights, stored) == 0
    permutation = np.random.default_rng(1).permutation(len(stored_points))
    other_choice = permutation[regular_tesselation(stored_points[permutation], weights[permutation])]
    assert exact.difference_lies_inside_copower_sets(integer, weights, stored, other_choice)
    flipped = _flip_one_strictly_regular_face(integer, weights, stored)
    assert exact.differing_tetrahedra(stored, flipped) == 5
    assert not exact.difference_lies_inside_copower_sets(integer, weights, stored, flipped)
