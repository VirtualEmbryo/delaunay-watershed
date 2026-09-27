"""Regression tests for `TesselationGraph._construct_edges` (the dropped-trailing-row defect).

The defect: the loop bound was `while index < n - 1`, so the last row of the lexicographically
sorted face table was never examined. When that row is a *lone* face (a face belonging to
only one tetrahedron, i.e. on the tesselation boundary) it was silently dropped from
`lone_faces` / `nodes_linked_by_lone_faces`, and its tetrahedron was not marked in
`nodes_on_the_border`.

The property that pins this down is a counting invariant: every one of the `4 * n_tets`
rows of the face table is either half of a shared-face pair or a lone face, so

    2 * len(triangle_faces) + len(lone_faces) == 4 * n_tets

The old code violates this exactly when it drops a row. The invariant is what these tests
assert, rather than a hand-copied "expected" number, because it holds for any input and so
cannot rot.

Measured note on blast radius: on all 8 golden-master configurations (`data/Images/*.tif`
at `min_distance` 3 and 5) the last row happens to be the second half of a matched pair,
so the invariant already held and fixing the defect changed no output there.

Which of these tests actually discriminate, checked by running the whole file against the
pre-fix code at commit 432566c: only the two "keeps all four / keeps the trailing" cases
fail there (4 vs 3 and 8 vs 7 lone faces). The border-marking and random-Delaunay cases
pass either way -- they are guards against regression in the surrounding logic, not
evidence about the defect -- and that asymmetry is the point: on realistic tesselations the bug is
rare, which is exactly why it survived unnoticed.
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from dw3d.tesselation_graph import TesselationGraph


def _zero_scores(_edt_image, _vertices, triangle_faces):
    """Stand-in ScoreComputationFunction: these tests are about connectivity, not scores."""
    return np.zeros(len(triangle_faces), dtype=np.float64)


def _build(points, tetrahedrons):
    return TesselationGraph(
        np.asarray(points, dtype=np.float64),
        np.asarray(tetrahedrons, dtype=np.int64),
        indices_of_sorted_maxes=np.array([], dtype=np.uint),
        score_computation_function=_zero_scores,
        edt_image=None,
    )


def _assert_face_count_invariant(graph, n_tets):
    n_shared = len(graph.triangle_faces)
    n_lone = len(graph.lone_faces)
    assert 2 * n_shared + n_lone == 4 * n_tets, (
        f"face table not fully accounted for: 2*{n_shared} + {n_lone} != 4*{n_tets}"
    )
    # the two lone-face arrays are built in lockstep and must stay aligned
    assert len(graph.nodes_linked_by_lone_faces) == n_lone


# --- a single tetrahedron: every face is lone, so the last row is certainly a lone face ---


def test_single_tetrahedron_keeps_all_four_lone_faces():
    """One isolated tetrahedron has 4 lone faces. The loop before the fix reported only 3."""
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    graph = _build(points, [[0, 1, 2, 3]])

    assert len(graph.triangle_faces) == 0
    assert len(graph.lone_faces) == 4
    _assert_face_count_invariant(graph, n_tets=1)
    assert graph.nodes_on_the_border[0] == 1


# --- two disjoint tetrahedra: 8 rows, all lone, and the trailing one is the interesting one ---


def test_two_disjoint_tetrahedra_keep_the_trailing_lone_face():
    """The lexicographically largest face belongs to the far tetrahedron and must survive.

    With the old `index < n - 1` bound this returned 7 lone faces out of 8; the dropped one
    is precisely the highest-indexed face, `(5, 6, 7)`.
    """
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (10, 10, 10),
        (11, 10, 10),
        (10, 11, 10),
        (10, 10, 11),
    ]
    graph = _build(points, [[0, 1, 2, 3], [4, 5, 6, 7]])

    assert len(graph.lone_faces) == 8
    _assert_face_count_invariant(graph, n_tets=2)

    lone = {tuple(sorted(face)) for face in graph.lone_faces}
    assert (5, 6, 7) in lone, "the trailing lone face was dropped"
    assert np.all(graph.nodes_on_the_border == 1)


def test_trailing_lone_face_tetrahedron_is_marked_on_the_border():
    """`nodes_on_the_border` is driven by the same loop, so it must cover the last row too.

    Does not discriminate the defect by itself: in this configuration both tetrahedra own several
    lone faces, so both get flagged from earlier rows even when the last is dropped. It
    guards the lone-face/border-flag correspondence, which the fix must not break.
    """
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (10, 10, 10),
        (11, 10, 10),
        (10, 11, 10),
        (10, 10, 11),
    ]
    graph = _build(points, [[0, 1, 2, 3], [4, 5, 6, 7]])

    # every lone face's tetrahedron is flagged, including the one owning the trailing row
    for tet_id in graph.nodes_linked_by_lone_faces:
        assert graph.nodes_on_the_border[tet_id] == 1


# --- a real tesselation with a mix of shared and lone faces ---


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_invariant_holds_on_a_random_delaunay_tesselation(seed):
    """A genuine tesselation exercises both loop branches, thousands of rows at a time."""
    rng = np.random.default_rng(seed)
    points = rng.random((60, 3)) * 20
    tesselation = Delaunay(points)

    graph = _build(tesselation.points, tesselation.simplices)

    n_tets = len(tesselation.simplices)
    _assert_face_count_invariant(graph, n_tets=n_tets)
    # a convex-hull tesselation always has boundary faces, so this is not vacuous
    assert len(graph.lone_faces) > 0
    assert len(graph.triangle_faces) > 0


def test_shared_faces_are_shared_by_exactly_two_distinct_tetrahedra():
    """Sanity check on the other loop branch, so the invariant cannot be satisfied wrongly."""
    rng = np.random.default_rng(7)
    tesselation = Delaunay(rng.random((60, 3)) * 20)
    graph = _build(tesselation.points, tesselation.simplices)

    pairs = graph.nodes_linked_by_faces
    assert pairs.shape == (len(graph.triangle_faces), 2)
    assert np.all(pairs[:, 0] != pairs[:, 1])
