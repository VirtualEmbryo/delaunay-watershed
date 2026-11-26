"""The deterministic triangulation (``dw2d.deterministic_triangulation``, used by the default route since 0.8.0).

The rule depends only on the points: any valid Delaunay triangulation, in any row order and with the
vertices of each row in any rotation, gives the same triangle array; the fan apex is a documented
function of the cell's integer coordinates (SHA-256 of a fixed byte encoding), pinned by a test
vector. The end-to-end checks on the golden cases are in ``test_deterministic_route.py``.
"""
import hashlib
import math
import struct

import numpy as np
import pytest
from scipy.spatial import Delaunay

from dw2d.deterministic_triangulation import (canonical_order, cocircular_cells, deterministic_triangles,
                                              fan_apex)



def lattice(n, seed, offset=(0, 0)):
    P = np.array([(i + offset[0], j + offset[1]) for i in range(n) for j in range(n)], dtype=float)
    P = P[np.random.default_rng(seed).permutation(len(P))]
    return P, Delaunay(P - np.array(offset, float)).simplices


def another_valid_triangulation(P, S, rng):
    """Every cocircular cell re-triangulated as a fan from a random vertex, rows permuted and rotated."""
    T = np.asarray(S, dtype=np.int64)
    keep, new = np.ones(len(T), bool), []
    for c in cocircular_cells(P, S):
        keep[c["triangles"]] = False
        k = int(rng.integers(len(c["ring"])))
        r = c["ring"][k:] + c["ring"][:k]
        new += [(r[0], r[i], r[i + 1]) for i in range(1, len(r) - 1)]
    out = np.vstack([T[keep], np.array(new, dtype=np.int64).reshape(-1, 3)])[rng.permutation(len(T))]
    return np.array([np.roll(t, int(rng.integers(3))) for t in out])


def test_the_cells_of_a_lattice_are_its_unit_squares():
    P, S = lattice(6, 0)
    cells = cocircular_cells(P, S)
    assert len(cells) == 25 and all(len(c["ring"]) == 4 for c in cells)


def test_the_result_does_not_depend_on_the_triangulation_qhull_returned():
    P, S = lattice(7, 1)
    ref = deterministic_triangles(P, S)
    assert np.array_equal(ref, canonical_order(ref))
    rng = np.random.default_rng(2)
    for _ in range(20):
        assert np.array_equal(deterministic_triangles(P, another_valid_triangulation(P, S, rng)), ref)


def test_exact_integers_are_used_for_large_coordinates():
    P, S = lattice(5, 3, offset=(40000, 70000))
    small, _ = lattice(5, 3)
    rings = [c["ring"] for c in cocircular_cells(P, S)]
    assert rings == [c["ring"] for c in cocircular_cells(small, S)]
    rng = np.random.default_rng(4)
    assert np.array_equal(deterministic_triangles(P, another_valid_triangulation(P, S, rng)), deterministic_triangles(P, S))


def test_each_diagonal_of_a_lattice_square_is_chosen_about_half_the_time():
    count = 0
    for r in range(60):
        for c in range(60):
            P = np.array([(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)], dtype=float)
            count += fan_apex(P, [0, 3, 2, 1]) % 2 == 0  # apex (r, c) or (r+1, c+1): the diagonal at 45 degrees
    assert abs(count - 1800) < 4 * math.sqrt(900)


def test_a_non_delaunay_triangulation_is_refused():
    P = np.array([(0, 0), (0, 4), (3, 2), (-3, 2)], dtype=float)
    with pytest.raises(ValueError, match="not Delaunay"):
        cocircular_cells(P, np.array([[0, 2, 3], [1, 2, 3]]))


def test_non_integer_points_are_refused():
    with pytest.raises(ValueError, match="not integer"):
        cocircular_cells(np.array([(0, 0), (0, 1.5), (1, 0)]), np.array([[0, 1, 2]]))


def test_the_fan_apex_is_sha256_of_the_documented_encoding():
    """Test vector: the apex depends only on SHA-256 of the sorted (row, col) pairs as little-endian int64."""
    P = np.array([(7, 3), (7, 4), (8, 4), (8, 3)], dtype=float)
    ring = [0, 3, 2, 1]
    encoding = b"".join(struct.pack("<qq", r, c) for r, c in sorted((7 + i, 3 + j) for i in (0, 1) for j in (0, 1)))
    digest = hashlib.sha256(encoding).hexdigest()
    assert digest == "1d6492f00ea8cc05188fee547098c2d4fefdb82e5208aa635beb27065a8bb6ee"
    assert fan_apex(P, ring) == int.from_bytes(bytes.fromhex(digest)[:8], "little") % 4 == 1


def test_negative_control_without_the_rule_two_valid_triangulations_differ():
    """On every machine: two valid triangulations of a lattice differ in canonical order alone, and agree once the rule is applied."""
    P, S = lattice(8, 6)
    other = another_valid_triangulation(P, S, np.random.default_rng(5))
    assert not np.array_equal(canonical_order(S), canonical_order(other))
    assert np.array_equal(deterministic_triangles(P, S), deterministic_triangles(P, other))
