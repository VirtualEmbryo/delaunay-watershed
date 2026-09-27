"""M6 must count nonlocal triangle-triangle self-intersections, and must count them exactly.

Every other validity number in this package is local or combinatorial. Watertightness, abnormal
non-manifold edges, degenerate and duplicate faces, quadjunction edges, reflex wedges and interface
identity are all read from a vertex's own neighbourhood or from the label structure, so a mesh whose
two *distant* sheets pass through each other satisfies every one of them.

That is not hypothetical. A post-process that moved junction vertices under a guard checking only
the triangles incident to the moved vertex left all twelve local predicates bit-identical on 40 of
40 benchmark cases while 23 nonlocal crossings existed on 5 of them. **A local guard cannot certify
a nonlocal property**, and until this predicate existed nothing in either repository could see one.

Pinned here:

1. the six counts are emitted by `compute_m6_validity`, and every pre-existing key of that block
   is still emitted with its own name;
2. the count needs **no ground truth** -- `compute_self_intersections` takes only the mesh;
3. an analytic pair that genuinely crosses is counted, a separated pair is not, and an *adjacent*
   pair -- which touches by construction, as three triangles do along every trijunction edge -- is
   never counted and never even tested;
4. the exact predicate agrees with an independent Moller-Trumbore oracle that shares no arithmetic
   with it;
5. the lesson itself: a phantom whose two closed shells are translated through each other leaves
   **every** M6 value bit-identical while the new count rises from zero.

Units are the mesh's own; the phantoms below are dimensionless.
"""

from __future__ import annotations

import numpy as np
import pytest

from dw3d_benchmarks.mesh_quality_comparison import case_metrics, self_intersection

#: The M6 keys that existed before this predicate was added. The addition is additive, so every
#: one of them must still be emitted; this list is what makes that a test rather than a promise.
_PRE_EXISTING_M6_KEYS = (
    "watertight",
    "n_boundary_edges",
    "n_abnormal_non_manifold_edges",
    "n_degenerate_faces",
    "n_duplicate_faces",
    "n_triple_line_edges",
    "n_malformed_triple_line_edges",
    "n_quadjunction_edges_note",
    "n_quadjunction_edges",
    "n_junction_edges",
    "n_reflex_junction_edges",
    "reflex_junction_edge_fraction",
    "reflex_excess_mean_deg",
    "reflex_excess_p90_deg",
    "reflex_excess_max_deg",
    "valence_geq4_split",
    "sliver_fraction",
    "identity",
)


def _tetrahedral_shell(offset: np.ndarray, material: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One closed tetrahedral surface separating `material` from the exterior medium `0`.

    Args:
        offset: translation applied to all four corners.
        material: the label on the inside; the outside is material `0`.

    Returns:
        tuple: `(points, triangles, labels)`, watertight and consistently wound.
    """
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) + offset
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    labels = np.array([[0, material]] * 4)
    return points, triangles, labels


def _two_shells(separation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two closed shells whose centres lie `separation` apart along `x`.

    At a large separation they are disjoint; at a small one they pass through each other. The
    connectivity, the labels and the winding are **identical** in both configurations, which is
    precisely why every combinatorial predicate cannot tell them apart.
    """
    first_points, first_triangles, first_labels = _tetrahedral_shell(np.zeros(3), 1)
    second_points, second_triangles, second_labels = _tetrahedral_shell(
        np.array([separation, 0.0, 0.0]), 2,
    )
    points = np.vstack([first_points, second_points])
    triangles = np.vstack([first_triangles, second_triangles + len(first_points)])
    labels = np.vstack([first_labels, second_labels])
    return points, triangles, labels


def _self_as_ground_truth(triangles: np.ndarray, labels: np.ndarray) -> dict:
    """A ground truth equal to the mesh itself, so interface identity is trivially clean."""
    return {"triangles": triangles, "labels": labels}


def test_a_crossing_pair_is_counted_and_a_separated_pair_is_not():
    """A blade driven through a flat triangle crosses it; translated away, it does not."""
    crossing = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],       # a triangle in the z = 0 plane
        [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 0.0],      # a blade through it
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    assert self_intersection.compute_self_intersections(crossing, triangles)[
        "n_self_intersecting_triangle_pairs"
    ] == 1

    apart = crossing.copy()
    apart[3:, 2] += 50.0
    assert self_intersection.compute_self_intersections(apart, triangles)[
        "n_self_intersecting_triangle_pairs"
    ] == 0


def test_an_adjacent_pair_is_never_counted_and_never_tested():
    """An adjacent pair touches by construction and is excluded before the predicate runs.

    Three triangles meet along every trijunction edge of a correct multimaterial mesh, so counting
    those would report the mesh's intended non-manifold structure as a defect.
    """
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 1, 3]])
    got = self_intersection.compute_self_intersections(points, triangles)
    assert got["n_self_intersecting_triangle_pairs"] == 0
    assert got["n_self_intersection_tested_pairs"] == 0
    assert got["n_self_intersection_candidate_pairs"] == 1, "the pair was a candidate, then dropped"


def test_the_count_reads_nothing_but_the_mesh():
    """No ground truth, no reference mesh, no labels: two arrays in, six numbers out.

    That is what makes the predicate usable on experimental data and as an acceptance gate for a
    construction, exactly as the reflex count is.
    """
    points, triangles, _labels = _two_shells(separation=0.4)
    got = self_intersection.compute_self_intersections(points, triangles)
    assert set(got) == set(self_intersection.VALIDITY_KEYS)
    assert got["n_self_intersecting_triangle_pairs"] > 0


def test_the_exact_predicate_agrees_with_an_independent_oracle():
    """Moller-Trumbore over six edges, sharing no arithmetic with the plane-interval construction."""
    report = self_intersection.differential_test(n_pairs=400, seed=20260913)
    assert report["n_disagreements"] == 0
    assert report["n_intersecting"] > 0, "a comparison in which nothing crossed would prove nothing"


def test_m6_emits_the_counts_and_keeps_every_pre_existing_key():
    """The addition is additive: six new keys, and not one old key renamed, removed or reordered."""
    points, triangles, labels = _two_shells(separation=0.4)
    validity = case_metrics.compute_m6_validity(
        points, triangles, labels, _self_as_ground_truth(triangles, labels),
    )
    for key in _PRE_EXISTING_M6_KEYS:
        assert key in validity, f"{key} disappeared from the M6 block"
    for key in self_intersection.VALIDITY_KEYS:
        assert key in validity, f"{key} was not emitted"
    assert validity["n_self_intersecting_triangle_pairs"] > 0


def test_every_local_predicate_is_blind_to_a_nonlocal_crossing():
    """The lesson, as a test.

    Two closed shells are translated through each other. Connectivity, labels and winding do not
    change, so **every** pre-existing M6 value is bit-identical -- the mesh is still watertight,
    still has no boundary edge, no degenerate face and no duplicate face -- while the new count
    rises from zero. A validity set without this predicate would call the crossed mesh perfect.
    """
    separated_points, triangles, labels = _two_shells(separation=5.0)
    crossed_points, _t, _l = _two_shells(separation=0.4)
    truth = _self_as_ground_truth(triangles, labels)

    separated = case_metrics.compute_m6_validity(separated_points, triangles, labels, truth)
    crossed = case_metrics.compute_m6_validity(crossed_points, triangles, labels, truth)

    for key in _PRE_EXISTING_M6_KEYS:
        assert separated[key] == crossed[key], f"{key} moved; the phantom is not isolating the point"

    assert separated["n_self_intersecting_triangle_pairs"] == 0
    assert crossed["n_self_intersecting_triangle_pairs"] > 0


def test_a_degenerate_triangle_is_excluded_and_counted_rather_than_skipped():
    """A zero-area triangle has no plane; silently dropping one would let a sliver hide a crossing."""
    points = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
        [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 0.0],
        [3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [5.0, 3.0, 3.0],   # collinear: zero area
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    got = self_intersection.compute_self_intersections(points, triangles)
    assert got["n_self_intersection_degenerate_triangles"] == 1
    assert got["n_self_intersecting_triangle_pairs"] == 1, "the healthy crossing is still found"


@pytest.mark.parametrize("n_triangles", [0, 1])
def test_a_mesh_with_no_pair_returns_zeros_rather_than_raising(n_triangles: int):
    """Fewer than two triangles is a valid input with no pair to test, not an error."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]])[:n_triangles]
    got = self_intersection.compute_self_intersections(points, triangles)
    assert got["n_self_intersecting_triangle_pairs"] == 0
    assert got["n_self_intersection_tested_pairs"] == 0
