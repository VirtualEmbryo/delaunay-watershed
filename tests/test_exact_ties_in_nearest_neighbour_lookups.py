"""Exact ties in the nearest-neighbour lookups go to the lowest index, whatever the SciPy build.

`cKDTree.query` breaks an exact tie by its tree traversal, which differs between SciPy builds: on
`3.tif` the Mac and Linux snapped a one-voxel 0-stratum component to different skeleton voxels, so the
junction-protected configurations placed one more point on Linux. Shuffling the input of the k-d tree
changes its traversal the same way, so a lookup that does not change under a shuffle (once re-sorted)
does not depend on the build either.
"""

import itertools

import numpy as np
import pytest

from dw3d.junctions import _corner_representatives_on_skeleton
from dw3d.mask_reconstruction import _create_coords, _make_mask
from dw3d.nearest_neighbours import candidates_holding_every_nearest, nearest_with_ties_to_the_lowest_index
from dw3d.reconstruction_algorithm import _compute_seeds_idx_from_voxel_coords

SHUFFLES = range(8)


def _lattice_points_at_squared_distance(centre, squared_distance):
    span = range(-3, 4)
    return [
        tuple(int(c + o) for c, o in zip(centre, offset, strict=True))
        for offset in itertools.product(span, span, span)
        if sum(o * o for o in offset) == squared_distance
    ]


def _snap(is_corner, skeleton, seed=None):
    skeleton = np.asarray(sorted(skeleton), dtype=np.int64)
    if seed is not None:
        skeleton = skeleton[np.random.default_rng(seed).permutation(len(skeleton))]
    return _corner_representatives_on_skeleton(is_corner, skeleton)


@pytest.mark.parametrize(
    ("corner_voxels", "tied", "farther"),
    [
        # A one-voxel component and three skeleton voxels at distance exactly 1: the case of 3.tif.
        ([(4, 4, 4)], [(3, 4, 4), (4, 4, 5), (5, 4, 4)], [(4, 6, 4), (6, 6, 6)]),
        # A two-voxel component, centroid on a half-integer: four skeleton voxels at d^2 = 5/4.
        ([(4, 4, 4), (4, 4, 5)], [(3, 4, 4), (3, 4, 5), (5, 4, 4), (5, 4, 5)], [(4, 6, 4)]),
        # Thirty skeleton voxels at distance exactly 3, more than the k-d tree's first 8 candidates.
        ([(4, 4, 4)], _lattice_points_at_squared_distance((4, 4, 4), 9), [(0, 0, 0), (8, 8, 8)]),
    ],
    ids=["one_voxel", "half_integer_centroid", "thirty_way_tie"],
)
def test_corner_snapping_takes_the_lowest_raster_index_among_the_exact_minima(corner_voxels, tied, farther):
    is_corner = np.zeros((9, 9, 9), dtype=bool)
    is_corner[tuple(np.array(corner_voxels).T)] = True
    expected = np.array([min(tied)], dtype=np.int64)  # raster order is lexicographic order

    np.testing.assert_array_equal(_snap(is_corner, tied + farther), expected)
    for seed in SHUFFLES:
        np.testing.assert_array_equal(_snap(is_corner, tied + farther, seed), expected)


def test_corner_snapping_breaks_an_exact_tie_at_a_centroid_that_floating_point_rounds():
    # A 7-voxel component with centroid (29/7, 29/7, 29/7), not representable: the two skeleton voxels,
    # mirror images through the plane x = y, are at exactly the same distance, but their floating-point
    # distances sum the same three squares in another order and can differ in the last bit.
    corner_voxels = [(4, 4, 4), (4, 4, 5), (4, 5, 4), (5, 4, 4), (4, 4, 3), (4, 3, 4), (4, 5, 5)]
    is_corner = np.zeros((10, 10, 10), dtype=bool)
    is_corner[tuple(np.array(corner_voxels).T)] = True
    total = np.array(corner_voxels).sum(axis=0).tolist()
    n = len(corner_voxels)

    def scaled(voxel):
        return sum((n * x - s) ** 2 for x, s in zip(voxel, total, strict=True))

    tied = [(2, 7, 4), (7, 2, 4)]
    assert total == [29, 29, 29]
    assert scaled(tied[0]) == scaled(tied[1])
    skeleton = [*tied, (0, 0, 9), (9, 9, 0)]
    for seed in [None, *SHUFFLES]:
        np.testing.assert_array_equal(_snap(is_corner, skeleton, seed), np.array([min(tied)]))


def test_corner_snapping_keeps_one_sample_per_component_and_merges_shared_ones():
    is_corner = np.zeros((9, 9, 9), dtype=bool)
    is_corner[2, 2, 2] = is_corner[2, 2, 4] = is_corner[6, 6, 6] = True
    skeleton = [(2, 2, 3), (6, 6, 7), (6, 7, 6)]
    # The first two components both snap to (2, 2, 3), one sample; the third has a tie at distance 1
    # and takes (6, 6, 7), the lower raster index.
    for seed in [None, *SHUFFLES]:
        np.testing.assert_array_equal(_snap(is_corner, skeleton, seed), np.array([(2, 2, 3), (6, 6, 7)]))


def test_the_candidates_hold_every_exact_minimum():
    from scipy.spatial import cKDTree

    points = np.array([*_lattice_points_at_squared_distance((0, 0, 0), 9), (9, 9, 9)], dtype=np.float64)
    candidate = candidates_holding_every_nearest(cKDTree(points), np.zeros((1, 3)))
    assert set(range(30)) <= set(candidate[0].tolist())


def test_nearest_point_takes_the_lowest_index_on_an_exact_tie_in_any_order():
    tied = np.array([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)])
    others = np.array([(2.0, 0.0, 0.0), (0.0, -2.0, 0.0)])
    queries = np.zeros((1, 3))
    for seed in [None, *SHUFFLES]:
        points = np.vstack([others, tied])
        order = np.arange(len(points)) if seed is None else np.random.default_rng(seed).permutation(len(points))
        points = points[order]
        is_tied = np.isin(order, np.arange(len(others), len(points)))
        assert nearest_with_ties_to_the_lowest_index(points, queries)[0] == np.flatnonzero(is_tied).min()


def test_seed_lookup_takes_the_lowest_tetrahedron_on_an_exact_tie():
    # Centroids on quarter-integers, the last three at d^2 = 3/16 from the seed: an exact three-way tie.
    centroids = np.array([(9.0, 9.0, 9.0), (1.25, 1.25, 0.75), (0.75, 0.75, 1.25), (0.75, 1.25, 1.25)])
    seeds = np.array([(1, 1, 1)], dtype=np.uint)
    assert _compute_seeds_idx_from_voxel_coords(centroids, seeds).tolist() == [1]
    assert _compute_seeds_idx_from_voxel_coords(centroids[[0, 3, 2, 1]], seeds).tolist() == [1]


def test_mask_rebuild_marks_the_lower_index_pixel_halfway_between_two():
    shape = np.array([3, 3, 3])
    grid = _create_coords(3, 3, 3)
    # Halfway between two neighbouring grid points along each axis, and one point exactly on the grid.
    pairs = [(0, 1), (0, 3), (0, 9), (13, 14)]
    verts = np.array([(grid[a] + grid[b]) / 2 for a, b in pairs] + [grid[26]])
    membrane = _make_mask(verts, shape).ravel()
    assert sorted(np.flatnonzero(membrane).tolist()) == sorted({min(a, b) for a, b in pairs} | {26})
