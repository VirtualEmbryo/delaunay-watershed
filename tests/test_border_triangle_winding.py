"""Border triangles are wound from their first label into their second, like every other triangle.

A mesh triangle with labels `(l1, l2)` must have its right-hand-rule normal pointing out of `l1`
and into `l2`: foambryo's cell volume adds `+det` to `l1` and `-det` to `l2`, which is a cell's
volume only under that convention. A *border triangle* is a face of the tesselation's hull (a
*lone face*, with one tetrahedron) whose tetrahedron carries a cell label; it enters the mesh
with labels `(0, cell)`, so its normal must point into the cell, i.e. into the hull.

Up to 0.5.0 every border triangle was wound the other way: the orientation turned each normal
away from a reference tetrahedron, as is right when that tetrahedron is on the first label's
side, but for a hull face the reference was a stale tetrahedron from the previous loop, and any
tetrahedron lies inside the hull. A cell bounded by the image box then read a negative volume,
and every mask cut through its cells gave some. Fixed by reading which label the reference
tetrahedron carries; who found the defect is credited in `CHANGELOG.md`.

The first three tests are the smallest triggers; the others check cut masks end to end.
"""

import warnings

import numpy as np
import pytest
import tifffile
from scipy.spatial import Delaunay

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.mesh_utilities import NO_TETRAHEDRON, _reorient_triangles, labeled_mesh_from_labeled_graph
from dw3d.tesselation_graph import TesselationGraph
from tests.conftest import IMAGE_NAMES, dataset_masks, load_image_or_skip

# The ten benchmark cases the defect was measured on, each cut at its mid-plane.
CUT_BENCHMARK_CASES = [f"{i:03d}" for i in range(0, 47, 5)]


def _zero_scores(edt_image, vertices, faces):  # noqa: ARG001 -- the score function's signature
    return np.zeros(len(faces))


def _mesh(points: np.ndarray, cell_of_tet):
    """Label a Delaunay tesselation of `points` by `cell_of_tet(centroids)` and extract its mesh."""
    tets = Delaunay(points).simplices.astype(np.int64)
    graph = TesselationGraph(
        tesselation_points=points,
        tesselation_tetrahedrons=tets,
        indices_of_sorted_maxes=np.zeros(0, dtype=np.uint),
        score_computation_function=_zero_scores,
        edt_image=np.zeros((3, 3, 3)),
    )
    labels_of_tet = cell_of_tet(points[graph.tetrahedrons].mean(axis=1))
    label_map = {int(c): [int(i) for i in np.flatnonzero(labels_of_tet == c)] for c in np.unique(labels_of_tet)}
    vertices, triangles, labels = labeled_mesh_from_labeled_graph(graph, label_map)
    return vertices, triangles.astype(np.int64), labels.astype(np.int64), graph, labels_of_tet


def _cube_with_centre() -> np.ndarray:
    corners = np.array([[x, y, z] for x in (0.0, 2.0) for y in (0.0, 2.0) for z in (0.0, 2.0)])
    return np.vstack([corners, [[1.0, 1.0, 1.0]]])


def _two_cells_in_a_box() -> np.ndarray:
    rng = np.random.default_rng(89)
    corners = np.array([[x, y, z] for x in (0.0, 4.0) for y in (0.0, 2.0) for z in (0.0, 2.0)])
    interior = rng.uniform([0.3, 0.3, 0.3], [3.7, 1.7, 1.7], size=(24, 3))
    return np.vstack([corners, interior])


def _signed_volumes(vertices, triangles, labels) -> dict[int, float]:
    """Signed cell volumes as foambryo computes them: `+det/6` to the first label, `-det/6` to the second."""
    dets = np.linalg.det(vertices[triangles]) / 6
    volumes: dict[int, float] = {}
    for det, (m1, m2) in zip(dets, labels, strict=True):
        volumes[int(m1)] = volumes.get(int(m1), 0.0) + float(det)
        volumes[int(m2)] = volumes.get(int(m2), 0.0) - float(det)
    return volumes


def _closed_cells(triangles, labels) -> list[int]:
    """Cells whose own triangles use every edge an even number of times (a closed surface)."""
    closed = []
    for cell in np.unique(labels):
        if cell == 0:
            continue
        own = triangles[(labels == cell).any(axis=1)]
        edges = np.sort(np.vstack([own[:, [0, 1]], own[:, [1, 2]], own[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        if len(own) and not (counts % 2).any():
            closed.append(int(cell))
    return closed


def _misoriented(vertices, triangles, labels, graph, labels_of_tet) -> tuple[int, int, int, int]:
    """Count triangles whose normal does not point into the second label, split border / interior."""
    tets_of_face: dict[tuple[int, ...], list[int]] = {}
    for t, tet in enumerate(graph.tetrahedrons):
        a, b, c, d = (int(x) for x in tet)
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            tets_of_face.setdefault(tuple(sorted(face)), []).append(t)
    border_bad = border = interior_bad = interior = 0
    for tri, (_, m2) in zip(triangles, labels, strict=True):
        p = vertices[tri]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        tets = tets_of_face[tuple(sorted(int(x) for x in tri))]
        on_m2 = [t for t in tets if labels_of_tet[t] == m2]
        assert len(on_m2) == 1
        side = float(normal @ (vertices[graph.tetrahedrons[on_m2[0]]].mean(axis=0) - p.mean(axis=0)))
        if len(tets) == 1:
            border += 1
            border_bad += side <= 0
        else:
            interior += 1
            interior_bad += side <= 0
    return border_bad, border, interior_bad, interior


def _cut_at_mid_plane(mask: np.ndarray) -> np.ndarray:
    """Keep axis 0 up to the middle of the labelled extent, so the cut runs through the cells."""
    occupied = np.flatnonzero((mask != 0).any(axis=(1, 2)))
    cut = (occupied[0] + occupied[-1]) // 2
    return np.ascontiguousarray(mask[: cut + 1])


def _reconstruct(mask: np.ndarray):
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the coarse-spacing and deprecation warnings are not under test
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    return points, triangles.astype(np.int64), labels.astype(np.int64), algorithm


def _border_triangles_not_into_their_cell(algorithm, triangles, labels) -> tuple[int, int]:
    """(mis-wound, total) border triangles, judged on the tesselation's own, unrelocated vertices.

    A border triangle is a mesh triangle with label 0 first whose vertex triple is a lone face
    of the tesselation; its cell is its one tetrahedron, which must be on its normal's side. The
    dot product is exactly 0 for a flat tetrahedron, so those are judged against a point inside
    the hull instead, as the rule does.
    """
    graph = algorithm._tesselation_graph
    tess = np.asarray(algorithm._surface_point_ids, dtype=np.int64)[triangles]
    lone_tet = {
        tuple(sorted(int(x) for x in face)): int(t)
        for face, t in zip(graph.lone_faces, graph.nodes_linked_by_lone_faces, strict=True)
    }
    inside = graph.vertices[np.unique(graph.tetrahedrons)].mean(axis=0)
    bad = total = 0
    for i in np.flatnonzero(labels[:, 0] == 0):
        tet = lone_tet.get(tuple(sorted(int(x) for x in tess[i])))
        if tet is None:
            continue
        p = graph.vertices[tess[i]]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        side = float(normal @ (graph.vertices[graph.tetrahedrons[tet]].mean(axis=0) - p.mean(axis=0)))
        if side == 0.0:
            side = float(normal @ (inside - p.mean(axis=0)))
        total += 1
        bad += side <= 0
    return bad, total


def _assert_cut_mask_is_wound_and_positive(points, triangles, labels, algorithm, mask, name: str) -> None:
    bad, total = _border_triangles_not_into_their_cell(algorithm, triangles, labels)
    assert total > 0, f"{name}: no border triangle, so the cut is not under test"
    assert bad == 0, f"{name}: {bad} of {total} border triangles point out of their cell"
    volumes = _signed_volumes(points, triangles, labels)
    closed = _closed_cells(triangles, labels)
    border_cells = set(np.unique(mask[-1][mask[-1] != 0]).tolist())
    assert border_cells & set(closed), f"{name}: no closed cell on the cut face, so the cut is not under test"
    negative = {c: volumes[c] for c in closed if volumes[c] <= 0}
    assert not negative, f"{name}: closed cells with a non-positive volume: {negative}"


# Smallest triggers.


def test_single_cell_touching_the_hull_has_positive_volume():
    """A cube's corners and centre, all twelve tetrahedra in one cell: every triangle is a border one."""
    vertices, triangles, labels, *_ = _mesh(_cube_with_centre(), lambda c: np.ones(len(c), dtype=np.int64))
    assert len(triangles) == 12
    volume = _signed_volumes(vertices, triangles, labels)[1]
    assert np.isclose(volume, 8.0), f"cell volume {volume:+.6f}, expected +8"


def test_every_border_triangle_points_into_its_cell():
    mesh = _mesh(_two_cells_in_a_box(), lambda c: np.where(c[:, 0] < 2.0, 1, 2))
    border_bad, border, interior_bad, interior = _misoriented(*mesh)
    assert border > 0
    assert interior > 0
    assert interior_bad == 0, f"{interior_bad}/{interior} interior triangles mis-wound"
    assert border_bad == 0, f"{border_bad}/{border} border triangles mis-wound"


def test_cell_volumes_match_their_tetrahedra():
    vertices, triangles, labels, graph, labels_of_tet = _mesh(
        _two_cells_in_a_box(),
        lambda c: np.where(c[:, 0] < 2.0, 1, 2),
    )
    tet_points = vertices[graph.tetrahedrons]
    tet_volumes = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                tet_points[:, 1] - tet_points[:, 0],
                np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
            ),
        )
        / 6
    )
    volumes = _signed_volumes(vertices, triangles, labels)
    for cell in (1, 2):
        expected = tet_volumes[labels_of_tet == cell].sum()
        assert np.isclose(volumes[cell], expected), (
            f"cell {cell}: mesh volume {volumes[cell]:+.4f}, tetrahedra {expected:.4f}"
        )


def test_every_closed_cell_of_the_minimal_inputs_is_positive():
    for points, cell_of_tet in (
        (_cube_with_centre(), lambda c: np.ones(len(c), dtype=np.int64)),
        (_two_cells_in_a_box(), lambda c: np.where(c[:, 0] < 2.0, 1, 2)),
    ):
        vertices, triangles, labels, *_ = _mesh(points, cell_of_tet)
        volumes = _signed_volumes(vertices, triangles, labels)
        closed = _closed_cells(triangles, labels)
        assert closed
        assert all(volumes[c] > 0 for c in closed), {c: volumes[c] for c in closed}


# The rule itself, on one triangle.


def test_the_rule_reads_the_label_side_of_its_tetrahedron_not_its_position():
    """One tetrahedron, one face: the same face and tetrahedron give opposite windings by side.

    Face `(0, 1, 2)` in the plane z = 0, tetrahedron apex at z = +1. With the tetrahedron on the
    first label's side the normal must point to -z (away from it). As a border triangle (no
    tetrahedron on the first side) it must point into the hull, to +z. Both input windings.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 1.0]])
    tetrahedrons = np.array([[0, 1, 2, 3]])
    for triangle in ([0, 1, 2], [0, 2, 1]):
        triangles = np.array([triangle], dtype=np.uint)
        for sides, expected_z in (([0, NO_TETRAHEDRON], -1.0), ([NO_TETRAHEDRON, 0], 1.0)):
            wound = _reorient_triangles(triangles, vertices, tetrahedrons, np.array([sides], dtype=np.int64))
            p = vertices[wound[0].astype(np.int64)]
            normal_z = np.cross(p[1] - p[0], p[2] - p[0])[2]
            assert np.sign(normal_z) == expected_z, (triangle, sides, normal_z)


def test_a_border_triangle_on_a_flat_tetrahedron_points_into_the_hull():
    """Qhull emits flat tetrahedra lying in a hull face; their centroid gives no side.

    Measured on the benchmark masks cut through their cells: four coplanar vertices on the cut
    plane form a tetrahedron of zero volume, two of whose faces are border triangles. Here the
    flat tetrahedron `(0, 1, 2, 3)` is the unit square in z = 0 and `(0, 1, 3, 4)` closes the
    hull with an apex at z = 1. The border triangle `(0, 1, 2)`, whose one tetrahedron is the
    flat one, must point into the hull (+z) whichever way it was extracted.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]])
    tetrahedrons = np.array([[0, 1, 2, 3], [0, 1, 3, 4]])
    for triangle in ([0, 1, 2], [0, 2, 1]):
        wound = _reorient_triangles(
            np.array([triangle], dtype=np.uint),
            vertices,
            tetrahedrons,
            np.array([[NO_TETRAHEDRON, 0]], dtype=np.int64),
        )
        p = vertices[wound[0].astype(np.int64)]
        assert np.cross(p[1] - p[0], p[2] - p[0])[2] > 0, triangle


def test_a_triangle_with_no_tetrahedron_on_either_side_is_refused():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        _reorient_triangles(
            np.array([[0, 1, 2]], dtype=np.uint),
            vertices,
            np.array([[0, 1, 2, 3]]),
            np.array([[NO_TETRAHEDRON, NO_TETRAHEDRON]], dtype=np.int64),
        )


# Cut masks, end to end with the default algorithm.


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
def test_every_closed_cell_is_positive_on_a_cut_reference_image(image_name):
    mask = _cut_at_mid_plane(load_image_or_skip(image_name))
    _assert_cut_mask_is_wound_and_positive(*_reconstruct(mask), mask, f"{image_name} cut")


@pytest.mark.parametrize("case", CUT_BENCHMARK_CASES)
def test_every_closed_cell_is_positive_on_a_cut_benchmark_mask(case):
    found = {path.name.split("_")[0]: path for path in dataset_masks()}
    if case not in found:
        pytest.skip("benchmarking-dataset not found (set BENCHMARKING_DATASET)")
    mask = _cut_at_mid_plane(tifffile.imread(found[case]))
    _assert_cut_mask_is_wound_and_positive(*_reconstruct(mask), mask, f"benchmark {case} cut")
