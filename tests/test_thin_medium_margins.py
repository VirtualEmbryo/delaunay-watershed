"""A cell a few voxels from the image box no longer takes the medium between them.

Before 0.8.0, at `min_distance` 3, a 12-voxel cube with a margin of 4 voxels of medium on every face meshed to 2.98
times its voxel count, and the two 16 x 8 x 16 boxes of a 24-voxel smoke mask to 2.46 and 2.44 times: the distance
field is 0 on the image box's outer voxels, below the membranes, so the membrane facing the box got no point and the
tetrahedra joining the cell to the box went to the cell (`dw3d.thin_medium_margins`). Since 0.8.0 a mask whose first
mesh has a vertex on a face that no cell reaches is meshed again, padded with label 0 on every such face thinner
than `2 * min_distance + 2` voxels, and the mesh is translated back; the mask's own voxels keep the dither they have
unpadded, so only the mesh near the padded faces changes. These tests fail on the 0.7 code: the volume tests by the
ratio, the others because `medium_margin_padding`, `pad_thin_medium_margins` and `dither_padding` did not exist.

The cubes' points lie on the integer lattice, with cospherical sets whose tetrahedralisation Qhull chooses by
floating-point rounding, so the mesh can differ between machines (on Linux the unpadded 12-voxel cube 4 voxels away
meshes to 1.52 times its voxels, not 2.98, and the 8-voxel cube 5 voxels away is not padded). A test that checks an
exact value reconstructs from the tetrahedralisation stored on macOS (`tests/make_thin_margin_tetrahedralisations.py`);
a test with Qhull's own tetrahedralisation checks only what holds for every Delaunay choice; and Qhull's choice on the
machine running the tests is checked to differ from the stored one only inside cospherical sets.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from dw3d import (
    CellVolumeDisagreementWarning,
    ThinMediumMarginWarning,
    get_default_mesh_reconstruction_algorithm,
    reconstruct_mask_from_dict,
)
from dw3d.crop_faces import IMAGE_FACES, cells_on_each_image_face
from dw3d.edt import compute_edt_classical
from dw3d.points_on_edt import _dithered, peak_local_points_dithered
from dw3d.tesselation import canonical_tetrahedron_order, simple_delaunay_tesselation
from dw3d.thin_medium_margins import (
    cell_margin_to_face,
    face_margins,
    needs_padding,
    padding_of_thin_margins,
    thin_margin_bound,
)
from tests import exact_delaunay_checks as exact
from tests.make_thin_margin_tetrahedralisations import (
    THIN_MARGIN_CASES,
    stored_tesselation_function,
    stored_tesselations,
)

#: The volume band of `dw3d.cell_volume_agreement` (mesh volume over voxel count).
BAND = (0.5, 1.5)


def cube(side: int, margin: int) -> np.ndarray:
    """One cell, a cube of `side` voxels, `margin` voxels of label 0 from every face of the array."""
    mask = np.zeros((side + 2 * margin,) * 3, dtype=np.uint16)
    mask[margin : margin + side, margin : margin + side, margin : margin + side] = 1
    return mask


def smoke_mask() -> np.ndarray:
    """The smoke mask of the upgrade checks: two 16 x 8 x 16 boxes, 4 voxels from every face of a 24-voxel array."""
    mask = np.zeros((24, 24, 24), dtype=np.uint16)
    mask[4:20, 4:12, 4:20] = 1
    mask[4:20, 12:20, 4:20] = 2
    return mask


def cell_volumes(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Each label's enclosed volume by the divergence theorem, triangles oriented out of their first label."""
    det = np.einsum("ij,ij->i", points[triangles[:, 0]], np.cross(points[triangles[:, 1]], points[triangles[:, 2]])) / 6
    volumes: dict[int, float] = {}
    for column, sign in ((0, 1.0), (1, -1.0)):
        for label in np.unique(labels[:, column]):
            volumes[int(label)] = volumes.get(int(label), 0.0) + sign * float(det[labels[:, column] == label].sum())
    return volumes


def reconstruct(mask: np.ndarray, stored: str | None = None, **settings: bool):
    """Mesh `mask` by the default configuration at `min_distance` 3; with `stored`, from that case's tetrahedra."""
    algorithm = get_default_mesh_reconstruction_algorithm(3)
    for name, value in settings.items():
        setattr(algorithm, name, value)
    if stored is not None:
        algorithm.tesselation_creation_function = stored_tesselation_function(stored)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    return algorithm, (points, triangles, labels), [w.category for w in caught]


@pytest.mark.parametrize("side", [12, 16, 24])
@pytest.mark.parametrize("margin", [2, 3, 4, 5])
def test_a_cube_near_the_image_box_keeps_its_volume(side, margin) -> None:
    """A cube 2 to 5 voxels from every face meshes to a volume inside the agreement band, with no volume warning.

    On 0.7, 7 of these 12 cubes were outside the band: 1.89, 2.10, 2.98, 4.42 (side 12, margins 2 to 5), 1.63, 2.21
    (side 16, margins 2 and 3), 1.74 (side 24, margin 3).
    """
    _, mesh, caught = reconstruct(cube(side, margin))
    ratio = cell_volumes(*mesh)[1] / side**3
    assert BAND[0] <= ratio <= BAND[1], ratio
    assert CellVolumeDisagreementWarning not in caught
    assert ThinMediumMarginWarning not in caught


@pytest.mark.parametrize("margin", [2, 3, 4, 5])
def test_an_eight_voxel_cube_near_the_image_box_no_longer_takes_the_medium(margin) -> None:
    """An 8-voxel cube 2 to 5 voxels from every face no longer takes the medium: at most 1.5 times its voxels.

    On 0.7: 2.38, 2.70, 1.58 and 0.49 at margins 2 to 5. Its volume can fall below the band, as that of any cell with
    one or two interior points does: the same cube 8 voxels from every face, with no padding, meshes to 0.096 and
    0.144 of its voxels with 2 of 20 dither seeds (42 to 61), and so does the padded cube at margins 4 and 5.

    From the stored tetrahedralisation, which pads the mask at every margin. With another Delaunay choice the cube at
    margin 5 can collapse before the padding is decided, to a mesh with no vertex on the image box, and is then not
    padded (Linux; the known issue on the decision from the first mesh's vertices).
    """
    algorithm, mesh, caught = reconstruct(cube(8, margin), stored=f"cube_8_margin_{margin}")
    assert cell_volumes(*mesh)[1] / 8**3 <= BAND[1]
    assert algorithm.medium_margin_padding == dict.fromkeys(IMAGE_FACES, 8 - margin)
    assert ThinMediumMarginWarning not in caught


@pytest.mark.parametrize("margin", [2, 3, 4, 5])
def test_an_eight_voxel_cube_near_the_image_box_takes_no_medium_with_qhull_s_own_choice(margin) -> None:
    """With Qhull's own tetrahedralisation, padded or not, the cube stays at most 1.5 times its voxels.

    And no cell is left on a face it does not reach. This holds for every Delaunay choice of these points.
    """
    _, mesh, caught = reconstruct(cube(8, margin))
    assert cell_volumes(*mesh)[1] / 8**3 <= BAND[1]
    assert ThinMediumMarginWarning not in caught


def test_the_two_boxes_of_the_smoke_mask_keep_their_volume() -> None:
    """Both boxes of the smoke mask mesh inside the band (0.7: 2.46 and 2.44 times their voxel count)."""
    mask = smoke_mask()
    algorithm, mesh, caught = reconstruct(mask)
    volumes = cell_volumes(*mesh)
    for cell in (1, 2):
        assert BAND[0] <= volumes[cell] / np.count_nonzero(mask == cell) <= BAND[1]
    assert CellVolumeDisagreementWarning not in caught
    assert algorithm.medium_margin_padding == dict.fromkeys(IMAGE_FACES, 4)


def test_a_padded_field_keeps_the_dither_of_the_mask_s_own_voxels() -> None:
    """With `dither_padding`, the mask's voxels get the dither they get unpadded; the padding gets the next draw."""
    widths = ((5, 0), (0, 2), (3, 3))
    shape = (20, 12, 9)
    padded = tuple(n + low + high for n, (low, high) in zip(shape, widths, strict=True))
    inner = (slice(5, 25), slice(0, 12), slice(3, 12))
    kept = _dithered(np.zeros(padded), 42, widths)
    np.testing.assert_array_equal(kept[inner], _dithered(np.zeros(shape), 42))
    redrawn = _dithered(np.zeros(padded), 42)
    assert not np.array_equal(redrawn[inner], kept[inner])
    np.testing.assert_array_equal(_dithered(np.zeros(shape), 42, None), _dithered(np.zeros(shape), 42))


def test_the_points_away_from_a_padded_face_are_those_of_the_unpadded_mask() -> None:
    """Padding one face moves no point placed 16 or more voxels from it; redrawing the dither moved 459 of them.

    Two cubes along a long axis, the first 3 voxels from its face (padded by 5), the second 31 voxels from it. The
    distance field is the same beyond 12 voxels of the padded face, and with the dither kept in the mask's frame so are
    the 255 points placed there; a dither drawn on the padded array (the first form of this padding) placed 268 points
    there, 459 differing.
    """
    mask = np.zeros((80, 32, 32), dtype=np.uint16)
    mask[3:19, 8:24, 8:24] = 1
    mask[50:71, 8:24, 8:24] = 2
    widths = ((5, 0), (0, 0), (0, 0))
    unpadded_edt, padded_edt = compute_edt_classical(mask), compute_edt_classical(np.pad(mask, widths))

    def far_points(points: np.ndarray, shift: int) -> set[tuple[int, ...]]:
        moved = np.asarray(points, dtype=np.int64) - np.array([shift, 0, 0])
        return {tuple(int(x) for x in p) for p in moved if p[0] >= 16}

    unpadded = far_points(peak_local_points_dithered(None, unpadded_edt, 3)[0], 0)
    kept = far_points(peak_local_points_dithered(None, padded_edt, 3, dither_padding=widths)[0], 5)
    redrawn = far_points(peak_local_points_dithered(None, padded_edt, 3)[0], 5)
    assert kept == unpadded
    assert len(kept) == 255
    assert redrawn != unpadded


def test_the_padded_run_hands_its_widths_to_the_point_placement() -> None:
    """The reconstruction passes the padding to a placement that takes `dither_padding`, on its padded run only."""
    calls: list[object] = []

    def spy(mask, edt, min_distance, print_info=False, seed=42, dither_padding=None):
        calls.append(dither_padding)
        return peak_local_points_dithered(mask, edt, min_distance, print_info, seed, dither_padding)

    algorithm = get_default_mesh_reconstruction_algorithm(3)
    algorithm.point_placing_function = functools.partial(spy, min_distance=3, seed=42, print_info=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(cube(12, 4))
    assert calls == [None, ((4, 4), (4, 4), (4, 4))]
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(cube(12, 8))
    assert calls == [None]


def test_the_reports_are_in_the_frame_of_the_mask() -> None:
    """The compressed segmentation, the crop-face report and the rebuilt mask have the mask's shape and frame."""
    mask = cube(16, 3)
    algorithm, _, _ = reconstruct(mask)
    compressed = algorithm.last_compressed_segmentation
    assert tuple(compressed["image_shape"]) == mask.shape
    seeds = np.asarray(compressed["seeds"], dtype=np.int64)
    assert np.all(seeds >= 0)
    assert np.all(seeds < np.array(mask.shape))
    assert algorithm.crop_face_report.image_shape == mask.shape
    assert algorithm.cut_cells == ()
    rebuilt = reconstruct_mask_from_dict(compressed)
    assert rebuilt.shape == mask.shape
    assert np.mean(rebuilt == mask) > 0.9


def test_switching_the_padding_off_gives_the_previous_mesh() -> None:
    """`pad_thin_medium_margins = False` meshes the mask as given (the 0.7 mesh, 2.98 times the voxel count).

    From the stored tetrahedralisation: Qhull's own choice gives 1.51 to 2.98 times the voxel count (seven meshes over
    300 orders of the same points on macOS; 1.52 on Linux).
    """
    algorithm, mesh, caught = reconstruct(
        cube(12, 4),
        stored="cube_12_margin_4_unpadded",
        pad_thin_medium_margins=False,
    )
    assert algorithm.medium_margin_padding == {}
    assert cell_volumes(*mesh)[1] / 12**3 == pytest.approx(2.9827, abs=1e-3)
    assert CellVolumeDisagreementWarning in caught
    assert ThinMediumMarginWarning in caught


def test_switching_the_padding_off_lets_the_cube_take_the_medium_with_qhull_s_own_choice() -> None:
    """With Qhull's own tetrahedralisation, the unpadded cube still takes the medium: above the band, both warnings.

    This holds for every Delaunay choice of these points seen (1.51 to 2.98 times the voxel count).
    """
    algorithm, mesh, caught = reconstruct(cube(12, 4), pad_thin_medium_margins=False)
    assert algorithm.medium_margin_padding == {}
    assert cell_volumes(*mesh)[1] / 12**3 > BAND[1]
    assert CellVolumeDisagreementWarning in caught
    assert ThinMediumMarginWarning in caught


@pytest.mark.parametrize("case", sorted(THIN_MARGIN_CASES))
def test_qhull_s_choice_here_differs_from_the_stored_one_only_inside_cospherical_sets(case) -> None:
    """Qhull's tetrahedralisation of each stored point set here is the stored one or another Delaunay choice.

    Every tetrahedron in one and not the other lies inside a cospherical set (exact check).
    """
    for points, stored in stored_tesselations(case):
        live = canonical_tetrahedron_order(simple_delaunay_tesselation(points)[1])
        if exact.differing_tetrahedra(stored, live):
            assert exact.difference_lies_inside_cospherical_sets(exact.integer_points(points), stored, live)


@pytest.mark.parametrize(("side", "margin"), [(24, 16), (16, 8), (8, 12)])
def test_a_mask_without_a_thin_margin_is_meshed_as_before(side, margin) -> None:
    """No padding, and the mesh bitwise that of the padding switched off."""
    algorithm, (points, triangles, labels), _ = reconstruct(cube(side, margin))
    _, (old_points, old_triangles, old_labels), _ = reconstruct(cube(side, margin), pad_thin_medium_margins=False)
    assert algorithm.medium_margin_padding == {}
    assert np.array_equal(points, old_points)
    assert np.array_equal(triangles, old_triangles)
    assert np.array_equal(labels, old_labels)


def test_a_face_a_cell_reaches_is_never_padded_and_a_cell_left_on_it_is_named() -> None:
    """Cell 1 reaches the face axis2_low (a crop); cell 2, 3 voxels from that face, is named with its margin."""
    mask = np.zeros((40, 40, 28), dtype=np.uint16)
    mask[2:12, 2:12, 0:10] = 1
    mask[16:32, 16:32, 3:19] = 2
    algorithm, _, caught = reconstruct(mask)
    assert "axis2_low" not in algorithm.medium_margin_padding
    assert algorithm.cut_cells == (1,)
    assert ThinMediumMarginWarning in caught
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        algorithm.construct_mesh_from_segmentation_mask(mask)
    message = next(str(w.message) for w in again if w.category is ThinMediumMarginWarning)
    assert "cell 2 lies on the image face axis2_low, 3 voxels of medium away" in message
    assert "a crop" in message


def test_the_margin_helpers() -> None:
    """Margins per face and per cell, the bound, and the padding rule, on a small mask with a cut cell."""
    mask = np.zeros((20, 20, 20), dtype=np.uint16)
    mask[4:16, 5:16, 3:17] = 1
    mask[0:2, 8:10, 8:10] = 2
    assert face_margins(mask) == (0, 4, 5, 4, 3, 3)
    assert face_margins(np.zeros((5, 5, 5), dtype=np.uint8)) == (None,) * 6
    assert cell_margin_to_face(mask, 1, "axis0_low") == 4
    assert cell_margin_to_face(mask, 1, "axis2_high") == 3
    assert cell_margin_to_face(mask, 3, "axis0_low") == -1
    assert thin_margin_bound(3) == 8
    assert thin_margin_bound(None) == 8
    assert thin_margin_bound(5) == 12
    cells_on_faces = cells_on_each_image_face(mask)
    assert cells_on_faces[0] == (2,)
    assert padding_of_thin_margins(face_margins(mask), cells_on_faces, 8) == ((0, 4), (3, 4), (5, 5))
    assert padding_of_thin_margins(face_margins(mask), cells_on_faces, 3) is None


def test_needs_padding_reads_the_vertices_on_faces_no_cell_reaches() -> None:
    """A vertex on an unreached face triggers; a vertex on a reached face, or none on the box, does not."""
    shape = (10, 10, 10)
    cells_on_faces = ((1,), (), (), (), (), ())
    triangle = np.array([[0, 1, 2]])
    on_reached_face = np.array([[0.0, 3, 3], [0, 4, 3], [0, 3, 4]])
    on_free_face = np.array([[9.0, 3, 3], [9, 4, 3], [9, 3, 4]])
    inside = np.array([[1.0, 3, 3], [1, 4, 3], [1, 3, 4]])
    assert not needs_padding(on_reached_face, triangle, cells_on_faces, shape)
    assert needs_padding(on_free_face, triangle, cells_on_faces, shape)
    assert not needs_padding(inside, triangle, cells_on_faces, shape)
    assert not needs_padding(inside, np.zeros((0, 3), dtype=np.int64), cells_on_faces, shape)


@pytest.mark.parametrize("pad", [True, False])
def test_a_two_dimensional_mask_is_padded_by_dw2d_with_this_algorithm_s_setting(pad) -> None:
    """dw3d's algorithm hands a 2-D mask to dw2d with its padding setting; `medium_margin_padding` is then dw2d's."""
    pytest.importorskip("dw2d")
    mask = np.zeros((24, 24), dtype=np.int64)
    mask[2:22, 2:22] = 1
    algorithm, _, _ = reconstruct(mask, pad_thin_medium_margins=pad)
    sides = ("axis0_low", "axis0_high", "axis1_low", "axis1_high")
    expected = dict.fromkeys(sides, 6) if pad else {}  # the bound 2 * 3 + 2, minus the margin 2
    assert algorithm.medium_margin_padding == expected


# The resampled route (`construct_mesh_in_physical_units` with voxels longer along one axis) is not padded by default:
# on a four-cell ball with voxels twice as long along z, over dither seeds 42-141, the padding made a cell keep less
# than half its voxels on 20 of 100 seeds against 3 of 100 unpadded (McNemar exact p = 1.5e-5; cubic voxels 13 against
# 10, p = 0.51), lost in the watershed by a mechanism not yet found. These tests fail on the code before that decision
# (the resampled mask was padded, and the keyword did not exist).

ANISOTROPIC = (1.0, 0.5, 0.5)


def reconstruct_in_physical_units(mask: np.ndarray, spacing: tuple[float, float, float], **keywords: bool):
    algorithm = get_default_mesh_reconstruction_algorithm(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = algorithm.construct_mesh_in_physical_units(mask, spacing, **keywords)
    return algorithm, mesh, caught


def test_a_resampled_mask_is_not_padded_by_default_and_its_thin_margin_is_named() -> None:
    """A 12-voxel cube 4 voxels from every face, z twice as long: unpadded, named, the algorithm's setting kept."""
    algorithm, mesh, caught = reconstruct_in_physical_units(cube(12, 4), ANISOTROPIC)
    assert algorithm.medium_margin_padding == {}
    assert algorithm.pad_thin_medium_margins is True
    messages = [str(w.message) for w in caught if w.category is ThinMediumMarginWarning]
    assert len(messages) == 1
    assert "off by default on a resampled mask" in messages[0]
    unpadded = get_default_mesh_reconstruction_algorithm(3)
    unpadded.pad_thin_medium_margins = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = unpadded.construct_mesh_in_physical_units(cube(12, 4), ANISOTROPIC)
    for ours, theirs in zip(mesh, reference, strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_a_resampled_mask_is_padded_when_asked() -> None:
    algorithm, _, caught = reconstruct_in_physical_units(cube(12, 4), ANISOTROPIC, pad_thin_medium_margins=True)
    assert algorithm.medium_margin_padding
    assert set(algorithm.medium_margin_padding) <= set(IMAGE_FACES)
    assert not any("off by default" in str(w.message) for w in caught)
    assert algorithm.pad_thin_medium_margins is True


def test_cubic_voxels_in_physical_units_are_padded_as_the_voxel_route() -> None:
    """Unit cubic voxels: padded by default, and bitwise the mesh of `construct_mesh_from_segmentation_mask`."""
    algorithm, mesh, _ = reconstruct_in_physical_units(cube(12, 4), (1.0, 1.0, 1.0))
    voxel_route, reference, _ = reconstruct(cube(12, 4))
    assert algorithm.medium_margin_padding == voxel_route.medium_margin_padding != {}
    for ours, theirs in zip(mesh, reference, strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


def test_the_keyword_overrides_the_setting_for_one_call_only() -> None:
    algorithm, _, _ = reconstruct_in_physical_units(cube(12, 4), (1.0, 1.0, 1.0), pad_thin_medium_margins=False)
    assert algorithm.medium_margin_padding == {}
    assert algorithm.pad_thin_medium_margins is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(cube(12, 4))
    assert algorithm.medium_margin_padding


def test_the_warning_gives_each_named_cells_volume_ratio():
    """Each named cell's mesh volume over its voxel count, as `cell_volume_report` computes it, to two decimals.

    A real 1 866-cell stack had cells named whose volumes were normal; the ratio lets a reader tell them from a cell
    that took the medium. Here a 12-voxel cube 3 voxels from every face, unpadded, takes it: about 2.1 times its voxels.
    """
    import re

    import dw3d
    from dw3d.cell_volume_agreement import mesh_cell_volumes
    from dw3d.thin_medium_margins import ThinMediumMarginWarning

    cube = np.zeros((18, 18, 18), dtype=np.uint16)
    cube[3:15, 3:15, 3:15] = 1
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm()
    algorithm.pad_thin_medium_margins = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(cube)
    (message,) = [str(w.message) for w in caught if issubclass(w.category, ThinMediumMarginWarning)]
    expected = mesh_cell_volumes(points, triangles, labels)[1] / np.count_nonzero(cube == 1)
    ratios = re.findall(r"cell 1 lies on the image face \w+, .*?, mesh volume ([0-9.]+) times its voxel count", message)
    assert len(ratios) == 6  # one clause per face, each with the cell's ratio
    assert {float(r) for r in ratios} == {round(expected, 2)}
    assert expected > 1.5  # the medium taken shows in the ratio
