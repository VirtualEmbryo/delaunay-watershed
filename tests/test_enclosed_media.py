"""Enclosed pockets of medium become their own regions, flagged as media, and every exterior piece is seeded.

The regression pinned: dw3d placed one watershed seed per label value, so an enclosed pocket of label 0 (a lumen)
had no seed and the first cell its tetrahedra reached took all of it; and when the pocket was deeper than the
exterior (a tight crop), label 0's one seed fell in the pocket and a cell took the exterior. Measured before the fix
on a 697-cell embryo: a 389 918-voxel lumen inside one cell at 2.40 times its voxel volume; on the hollow shell
below, the cell taking a 4 207-voxel core at 1.19 times, and, in a tight crop, a cell taking the exterior at 22-29
times. A seed whose nearest tetrahedron lay in another region grew there (phantom regions of up to 64 times their
voxels when seeds were put in pockets too small to hold a tetrahedron).

Geometry: a ball of radius 30 voxels cut into its 8 octants (cells 1-8, about 13 000 voxels each), centred off the
lattice so that no voxel centre lies on a cutting plane, around a hollow core of label 0; `margin` voxels of medium
between the ball and the image box. Default algorithm, `min_distance` 3 (size bound: inscribed radius >= 4 voxels).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dw3d import (
    EnclosedMediumWarning,
    MeshReconstructionAlgorithmFactory,
    UnresolvedCellWarning,
    find_enclosed_media,
)
from dw3d.cell_volume_agreement import (
    CellVolumeDisagreementWarning,
    compare_cell_volumes_with_mask,
    mesh_cell_volumes,
    voxel_counts,
)
from dw3d.io import load_rec_appendix
from dw3d.reconstruction_algorithm import _seeds_inside_their_labels

BALL_RADIUS = 30.0


def hollow_shell(margin: int, core_radius: float, core_position: str = "centre") -> np.ndarray:
    """The 8 octant cells of a ball around a core of label 0 (centred, or inside octant 8)."""
    n = int(2 * (BALL_RADIUS + margin)) + 1
    centre = np.full(3, n // 2) + np.array([0.31, 0.17, 0.43])
    z, y, x = np.indices((n, n, n), dtype=np.float64)
    dz, dy, dx = z - centre[0], y - centre[1], x - centre[2]
    mask = np.zeros((n, n, n), dtype=np.uint16)
    inside = dz**2 + dy**2 + dx**2 < BALL_RADIUS**2
    mask[inside] = (1 + (dz > 0) + 2 * (dy > 0) + 4 * (dx > 0))[inside]
    if core_radius > 0:
        core = centre if core_position == "centre" else centre + 0.6 * BALL_RADIUS / np.sqrt(3.0)
        mask[(z - core[0]) ** 2 + (y - core[1]) ** 2 + (x - core[2]) ** 2 < core_radius**2] = 0
    return mask


def reconstruct(mask: np.ndarray, **attributes: object) -> tuple:
    """Default reconstruction at `min_distance` 3; returns the algorithm, the mesh and the warnings' classes."""
    algorithm = MeshReconstructionAlgorithmFactory.get_default_algorithm(min_distance=3)
    for name, value in attributes.items():
        setattr(algorithm, name, value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    return algorithm, mesh, [w.category for w in caught]


def volume_ratios(algorithm: object, mesh: tuple) -> dict[int, float]:
    """Mesh volume over voxel count of every non-zero label of the meshed mask."""
    volumes = mesh_cell_volumes(*mesh)
    voxels = voxel_counts(algorithm._segmented_image)
    return {label: volumes.get(label, 0.0) / count for label, count in voxels.items() if label != 0}


@pytest.fixture(scope="module")
def lumen_case() -> tuple:
    mask = hollow_shell(margin=30, core_radius=10)
    return (mask, mask.copy(), *reconstruct(mask))


def test_an_enclosed_pocket_lined_by_cells_is_its_own_medium(lumen_case: tuple) -> None:
    """The core is region 9, a medium lining all 8 cells; no cell keeps the core's volume (1.19 before)."""
    _, _, algorithm, mesh, caught = lumen_case
    assert algorithm.medium_labels == (0, 9)
    (medium,) = algorithm.enclosed_medium_report.media
    assert (medium.label, medium.voxels, medium.lining_cells) == (9, 4207, tuple(range(1, 9)))
    assert medium.inscribed_radius == pytest.approx(9.4868, abs=1e-4)
    ratios = volume_ratios(algorithm, mesh)
    assert all(0.85 <= ratios[cell] <= 1.1 for cell in range(1, 9)), ratios
    assert 0.5 <= ratios[9] <= 1.1
    labels = mesh[2]
    assert {int(a) for a, b in labels.tolist() if b == 9} == set(range(1, 9))
    assert EnclosedMediumWarning not in caught
    assert CellVolumeDisagreementWarning not in caught


def test_the_callers_mask_is_not_modified(lumen_case: tuple) -> None:
    mask, copy, algorithm, _, _ = lumen_case
    np.testing.assert_array_equal(mask, copy)
    assert algorithm._segmented_image is not mask


def test_the_compressed_segmentation_keeps_one_seed_per_label(lumen_case: tuple) -> None:
    """Media included, so `reconstruct_mask_from_dict` rebuilds the relabelled mask's labels 0-9."""
    _, _, algorithm, _, _ = lumen_case
    assert len(algorithm.last_compressed_segmentation["seeds"]) == 10


def test_the_rec_file_records_the_media(lumen_case: tuple, tmp_path: object) -> None:
    _, _, algorithm, _, _ = lumen_case
    path = tmp_path / "lumen.rec"
    algorithm.save_to_rec_mesh(path)
    appendix = load_rec_appendix(path)
    assert appendix["media"] == (0, 9)
    assert appendix["enclosed_media_voxels"] == (4207,)
    assert appendix["enclosed_media_inscribed_radius"][0] == pytest.approx(9.4868, abs=1e-4)


def test_a_mesh_without_enclosed_medium_writes_no_appendix(tmp_path: object) -> None:
    """A file without an enclosed medium is what it was before 0.7.1: no appendix at all here."""
    algorithm, _, caught = reconstruct(hollow_shell(margin=30, core_radius=0))
    assert algorithm.medium_labels == (0,)
    assert not algorithm.enclosed_medium_report.changes_the_mesh
    assert EnclosedMediumWarning not in caught
    path = tmp_path / "plain.rec"
    algorithm.save_to_rec_mesh(path)
    assert load_rec_appendix(path) is None


def test_a_pocket_below_the_size_bound_is_left_and_named() -> None:
    """Inscribed radius 2.8 < 4: no tetrahedron would hold it; it stays to the cells, with the warning."""
    mask = hollow_shell(margin=30, core_radius=3)
    relabelled, report = find_enclosed_media(mask, 4.0)
    assert relabelled is mask
    assert report.media == ()
    algorithm, _, caught = reconstruct(mask)
    assert algorithm.medium_labels == (0,)
    (left,) = algorithm.enclosed_medium_report.pockets_left_to_cells
    assert (left.voxels, left.reason) == (117, "below the size bound")
    assert left.absorbed_by in range(1, 9)
    assert EnclosedMediumWarning in caught


def test_a_pocket_lined_by_one_cell_is_left_to_it() -> None:
    """A hole in one cell: as its own region it would have no triple line and no inferable tension."""
    algorithm, _, caught = reconstruct(hollow_shell(margin=30, core_radius=6, core_position="inside one cell"))
    (left,) = algorithm.enclosed_medium_report.pockets_left_to_cells
    assert (left.lining_cells, left.reason, left.absorbed_by) == ((8,), "lined by one cell", 8)
    assert algorithm.medium_labels == (0,)
    assert EnclosedMediumWarning in caught


def test_the_exterior_keeps_its_seed_in_a_tight_crop() -> None:
    """In a tight crop the exterior keeps its seed, with or without the pocket made a region.

    Margin 6, core radius 16: the core (EDT 15.9) is deeper than the exterior (12.7). Before the fix label 0's seed
    fell in the core and a cell took the exterior (23.6 times its voxels). With the pocket left to the cells
    (`label_enclosed_media = False`) the exterior still gets its own seed and every cell meets it.
    """
    mask = hollow_shell(margin=6, core_radius=16)
    for switch in (True, False):
        algorithm, mesh, _ = reconstruct(mask, label_enclosed_media=switch)
        ratios = volume_ratios(algorithm, mesh)
        assert max(ratios[cell] for cell in range(1, 9)) < (1.1 if switch else 3.0), ratios
        assert {int(b) for a, b in mesh[2].tolist() if a == 0} == set(range(1, 9))
        assert algorithm.enclosed_medium_report.exterior_pieces == 1


def test_every_piece_of_exterior_is_seeded() -> None:
    """A wall of two cells cuts the box in two: both halves of medium stay medium.

    Before the fix label 0 had one seed, in the half holding its first EDT maximum; cell 1 took 471 of the other
    half's 476 tetrahedra (measured at fe02db4). The wall cells are cut by the image box, so their volume is not a
    measure here: the tetrahedra's labels are.
    """
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[:, :, 17:23] = 1
    mask[:20, :, 17:23] = 2
    algorithm, _, _ = reconstruct(mask)
    assert algorithm.enclosed_medium_report.exterior_pieces == 2
    centroids = np.clip(np.rint(algorithm._tesselation_graph.compute_nodes_centroids()).astype(int), 0, 39)
    labels = algorithm._map_node_id_to_label
    for half in (centroids[:, 2] < 17, centroids[:, 2] > 22):
        assert np.mean(labels[half] == 0) > 0.98


def test_new_labels_follow_the_lowest_raster_index_and_widen_the_dtype() -> None:
    """Two pockets get 256 and 257 in raster order of their first voxel; uint8 is widened to hold them."""
    mask = np.full((30, 30, 60), 255, dtype=np.uint8)
    mask[:15] = 254
    mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1] = 0, 0, 0, 0, 0, 0
    mask[10:20, 10:20, 40:50] = 0  # first voxel at raster index of (10, 10, 40)
    mask[12:22, 10:20, 5:15] = 0  # first voxel at (12, 10, 5): later in raster order
    relabelled, report = find_enclosed_media(mask, 4.0)
    assert relabelled.dtype == np.uint16
    assert [m.label for m in report.media] == [256, 257]
    assert relabelled[15, 15, 45] == 256
    assert relabelled[15, 15, 10] == 257
    assert [m.lining_cells for m in report.media] == [(254, 255), (254, 255)]
    assert (mask[15, 15, 45], mask[15, 15, 10]) == (0, 0)


def test_compare_cell_volumes_skips_the_media() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 5
    mask[0, 0, 0] = 7
    empty = np.zeros((0, 3))
    report = compare_cell_volumes_with_mask(mask, empty, np.zeros((0, 3), int), np.zeros((0, 2), int), media=(0, 7))
    assert report.n_cells == 1
    assert report.labels == [5]


def test_a_seed_outside_its_label_moves_inside_or_is_dropped() -> None:
    """Seed of label 1 nearest to a tetrahedron in label 2 moves to the nearest in label 1; label 3 has none."""
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[:, :5] = 1
    mask[:, 5:] = 2
    mask[9, 9, 9] = 3
    centroids = np.array([[5.0, 5.0, 5.0], [5.0, 2.0, 5.0], [5.0, 1.0, 5.0], [5.0, 8.0, 5.0]])
    seeds = np.array([[5, 4, 5], [5, 8, 5], [9, 9, 9]])
    nodes = np.array([0, 3, 3])
    labels = np.array([1, 2, 3], dtype=np.uint8)
    new_nodes, new_labels, dropped, moved = _seeds_inside_their_labels(centroids, nodes, seeds, labels, mask)
    assert new_nodes.tolist() == [1, 3]
    assert new_labels.tolist() == [1, 2]
    assert dropped == (3,)
    assert moved == (1,)


def test_seeds_already_inside_their_labels_are_returned_unchanged() -> None:
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    centroids = np.array([[1.0, 1.0, 1.0]])
    nodes = np.array([0])
    labels = np.array([1], dtype=np.uint8)
    out = _seeds_inside_their_labels(centroids, nodes, np.array([[1, 1, 1]]), labels, mask)
    assert out[0] is nodes
    assert out[1] is labels
    assert out[2] == ()
    assert out[3] == ()


def test_a_cell_with_no_tetrahedron_of_its_own_is_absent_and_named(tmp_path: object) -> None:
    """A 3x3x3-voxel cell inside cell 8 holds no tetrahedron centroid at min_distance 3: its seed is dropped.

    Before 0.7.1 the seed's nearest tetrahedron, in cell 8, grew label 9 inside cell 8 as a phantom region (on the
    benchmark: cell 2 of case 031 and cell 7 of case 038, at every min_distance from 1 to 7).
    """
    mask = hollow_shell(margin=30, core_radius=0)
    corner = np.array(mask.shape) // 2 + 10
    mask[corner[0] : corner[0] + 3, corner[1] : corner[1] + 3, corner[2] : corner[2] + 3] = 9
    algorithm, mesh, caught = reconstruct(mask)
    report = algorithm.enclosed_medium_report
    (region,) = report.unresolved
    assert (region.label, region.voxels, region.kind) == (9, 27, "cell")
    assert report.unresolved_cells == (9,)
    assert report.min_distance == 3
    assert report.changes_the_mesh
    assert 9 not in np.unique(mesh[2])
    assert UnresolvedCellWarning in caught
    assert algorithm.cell_volume_report.labels == [9]
    path = tmp_path / "unresolved.rec"
    algorithm.save_to_rec_mesh(path)
    appendix = load_rec_appendix(path)
    assert appendix["unresolved_regions"] == (9,)
    assert appendix["unresolved_regions_voxels"] == (27,)
    assert "media" not in appendix


def test_the_unresolved_warning_names_the_cell_and_the_min_distance() -> None:
    from dw3d.enclosed_media import EnclosedMediumReport, UnresolvedRegion, unresolved_message

    report = EnclosedMediumReport(unresolved=(UnresolvedRegion(9, 27, "cell"),), min_distance=3)
    text = unresolved_message(report)
    assert "cell 9 (27 voxels)" in text
    assert "min_distance 3" in text
    assert "a smaller min_distance may mesh it" in text
