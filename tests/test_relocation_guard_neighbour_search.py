"""The guard's new neighbour search returns exactly the pair sets, and the guard exactly the decisions.

`build_vertex_pair_sets` used to run one k-d tree ball query per incident triangle, at a radius
inflated by the largest reach in the whole mesh, and `_crossing_count` built the whole mesh's
corner array for every move it judged. On a 1 866-cell tissue those took 167 s and 222 s. The
search is now one batched query per pair of reach bins, filtered by the same exact criterion, and
the predicate gathers only the triangles its pairs involve. Both are pure speed changes: these
tests compare them, bitwise, with the original functions kept here verbatim, on a reconstructed
mesh with its real targets, on the same mesh with targets spread over three orders of magnitude of
displacement (so the reach bins matter), and on the whole relocation pass.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial import cKDTree

import dw3d
from dw3d import junction_relocation
from dw3d.junction_curves import extract_junction_curves
from dw3d.junction_relocation import (
    _DISPLACEMENT_INFLATION,
    _crossing_count,
    build_vertex_pair_sets,
    matched_curve_targets,
    relocate_junction_vertices,
    vertex_incidence,
)
from dw3d.triangle_intersection import (
    DEGENERATE_TWICE_AREA,
    drop_adjacent_pairs,
    pair_intersection_flags,
    triangle_centroids_and_circumradii,
    twice_areas,
)
from tests import conftest


# ---------------------------------------------------------------- the original functions, verbatim
def reference_build_vertex_pair_sets(
    points: NDArray[np.float64],
    triangles: NDArray[np.integer],
    targets: dict[int, NDArray[np.float64]],
) -> tuple[dict[int, NDArray[np.int64]], float]:
    """For each vertex with a target, a superset of the pairs its move can change the status of.

    The superset is built from the *base* centroids with an inflation that covers every displacement
    the pass can ever propose, so one static tree serves the whole pass: a triangle's centroid moves
    by at most the largest displacement `d` of its own three vertices, its radius about that
    centroid grows by at most `2 d`, and two triangles can meet only when their centroids lie within
    the sum of their radii. A pair that can ever meet therefore satisfies
    `|c_f - c_t| <= r_f + r_t + 3 (d_f + d_t)` at base.

    Args:
        points (NDArray[np.float64]): the base vertex coordinates, voxel units.
        triangles (NDArray[np.integer]): triangle vertex indices, never modified.
        targets (dict[int, NDArray[np.float64]]): vertex index to proposed position; only these
            vertices move, and their displacements set the inflation.

    Returns:
        tuple: `(pairs_by_vertex, seconds)`. Each value is a `(k, 2)` array of triangle index pairs
        with adjacency already removed, ready for the exact predicate. A vertex whose move can
        change nothing maps to an empty array.
    """
    clock = time.perf_counter()
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    centroids, radii = triangle_centroids_and_circumradii(points[triangles])

    displacement = np.zeros(len(points), dtype=np.float64)
    for vertex, target in targets.items():
        displacement[vertex] = float(np.linalg.norm(np.asarray(target) - points[vertex]))
    reach = radii + _DISPLACEMENT_INFLATION * displacement[triangles].max(axis=1)

    tree = cKDTree(centroids)
    incident = vertex_incidence(triangles, len(points))
    worst_partner_reach = float(reach.max()) if len(reach) else 0.0

    pairs_by_vertex: dict[int, NDArray[np.int64]] = {}
    neighbours_of_triangle: dict[int, NDArray[np.int64]] = {}
    for vertex in sorted(targets):
        faces = incident[vertex]
        collected: list[NDArray[np.int64]] = []
        for face in faces:
            cached = neighbours_of_triangle.get(face)
            if cached is None:
                found = np.asarray(
                    tree.query_ball_point(centroids[face], reach[face] + worst_partner_reach),
                    dtype=np.int64,
                )
                if len(found):
                    # exact filter against each partner's own reach, so the superset stays tight
                    separation = np.linalg.norm(centroids[found] - centroids[face], axis=1)
                    found = found[separation <= reach[face] + reach[found]]
                cached = found[found != face]
                neighbours_of_triangle[face] = cached
            if len(cached):
                collected.append(
                    np.stack([np.full(len(cached), face, dtype=np.int64), cached], axis=1),
                )
        if not collected:
            pairs_by_vertex[vertex] = np.zeros((0, 2), dtype=np.int64)
            continue
        pairs = np.unique(np.sort(np.concatenate(collected, axis=0), axis=1), axis=0)
        pairs_by_vertex[vertex] = drop_adjacent_pairs(pairs, triangles)
    return pairs_by_vertex, time.perf_counter() - clock


def reference_crossing_count(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    pairs: NDArray[np.int64],
) -> tuple[int, int]:
    """Intersections among an explicit pair list, and how many of its triangles are degenerate."""
    involved = np.unique(pairs)
    n_degenerate = int((twice_areas(points[triangles[involved]]) <= DEGENERATE_TWICE_AREA).sum())
    intersects, _touching = pair_intersection_flags(points[triangles], pairs)
    return int(intersects.sum()), n_degenerate


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def reconstructed():
    """A reconstructed reference image, before relocation, with its mask and its real targets."""
    mask = conftest.load_image_or_skip("3.tif")
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm()
    algorithm.relocate_junctions = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(mask)
    points = np.asarray(algorithm._points, dtype=np.float64)
    triangles = np.asarray(algorithm._triangles, dtype=np.int64)
    labels = np.asarray(algorithm._labels, dtype=np.int64)
    curves = extract_junction_curves(mask, points, triangles, labels)
    targets, _proposed, _quadruple, _diagnostics = matched_curve_targets(curves, points, triangles, labels)
    assert len(targets) > 100
    return mask, points, triangles, labels, curves, targets


def assert_same_pair_sets(expected: dict, got: dict) -> None:
    assert sorted(expected) == sorted(got)
    for vertex in expected:
        e, g = expected[vertex], got[vertex]
        assert e.dtype == g.dtype, vertex
        assert e.shape == g.shape, vertex
        assert np.array_equal(e, g), vertex


# ---------------------------------------------------------------- the tests


def test_pair_sets_match_the_original_on_real_targets(reconstructed):
    _mask, points, triangles, _labels, _curves, targets = reconstructed
    expected, _ = reference_build_vertex_pair_sets(points, triangles, targets)
    got, _ = build_vertex_pair_sets(points, triangles, targets)
    assert_same_pair_sets(expected, got)
    assert sum(len(v) for v in got.values()) > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pair_sets_match_the_original_with_displacements_over_three_decades(reconstructed, seed):
    _mask, points, triangles, _labels, _curves, _targets = reconstructed
    rng = np.random.default_rng(seed)
    vertices = rng.choice(len(points), size=min(200, len(points)), replace=False)
    spread = {}
    for vertex in vertices:
        direction = rng.normal(size=3)
        spread[int(vertex)] = points[vertex] + direction / np.linalg.norm(direction) * 10 ** rng.uniform(-3, 0.7)
    expected, _ = reference_build_vertex_pair_sets(points, triangles, spread)
    got, _ = build_vertex_pair_sets(points, triangles, spread)
    assert_same_pair_sets(expected, got)


def test_pair_sets_match_the_original_with_a_repeated_vertex_and_an_isolated_target():
    """A triangle listing a vertex twice, and a target vertex with no triangle at all."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [5, 6, 5], [6, 5, 5], [9, 9, 9]], float)
    triangles = np.array([[0, 1, 2], [0, 0, 1], [3, 4, 5], [1, 2, 3]], dtype=np.int64)
    targets = {0: np.array([0.1, 0.1, 0.1]), 6: np.array([9.5, 9, 9]), 3: np.array([4, 4, 4.0])}
    expected, _ = reference_build_vertex_pair_sets(points, triangles, targets)
    got, _ = build_vertex_pair_sets(points, triangles, targets)
    assert_same_pair_sets(expected, got)


def test_the_crossing_count_on_gathered_triangles_matches_the_whole_corner_array(reconstructed):
    _mask, points, triangles, _labels, _curves, targets = reconstructed
    pairs_by_vertex, _ = build_vertex_pair_sets(points, triangles, targets)
    moved = points.copy()
    for vertex, target in targets.items():
        moved[vertex] = target
    checked = 0
    for pairs in pairs_by_vertex.values():
        if len(pairs):
            for configuration in (points, moved):
                assert _crossing_count(configuration, triangles, pairs) == reference_crossing_count(
                    configuration, triangles, pairs)
            checked += 1
    assert checked > 0


def test_the_whole_pass_takes_the_same_decisions_as_with_the_original_search(reconstructed, monkeypatch):
    mask, points, triangles, labels, curves, _targets = reconstructed
    moved_new, report_new = relocate_junction_vertices(mask, points, triangles, labels, curves=curves)
    monkeypatch.setattr(junction_relocation, "build_vertex_pair_sets", reference_build_vertex_pair_sets)
    monkeypatch.setattr(junction_relocation, "_crossing_count", reference_crossing_count)
    moved_old, report_old = relocate_junction_vertices(mask, points, triangles, labels, curves=curves)
    assert moved_new.tobytes() == moved_old.tobytes()
    for field in ("n_targets", "n_accepted", "n_rejected_degenerate", "n_rejected_inversion",
                  "n_rejected_self_intersection", "n_pairs_tested", "n_degenerate_encountered",
                  "n_self_intersections_before", "n_self_intersections_after"):
        assert getattr(report_new, field) == getattr(report_old, field), field
    assert np.array_equal(report_new.applied_displacement_voxels, report_old.applied_displacement_voxels)
