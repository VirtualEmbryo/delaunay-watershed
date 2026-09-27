"""Connected-line identity in M5: what the triple-keyed path sees, and what it drops.

The regression these pin is not numerical drift, it is a **coverage** claim. Production's angle
and junction dictionaries are keyed by material triple with no connected-component identity, so
two disconnected trijunction lines that happen to separate the same three materials are one key.
`compute_m5_junction_geometry` reproduces that and, per triple, scores only the single longest
chain. `compute_m5_component_geometry` keeps each connected line separate. Both are reported;
neither replaces the other.

Two facts are pinned:

1. On an **ordinary single-component triple** the two paths score the identical chain, so every
   number they emit agrees exactly. Without this, the new path could not be trusted to be the
   same measurement.
2. On **two disconnected lines sharing one material triple** they must differ: the triple-keyed
   path sees one triple and scores one line; the component path sees and matches two, and reports
   the arclength fraction the triple-keyed path covers as about a half rather than one.

The fixture is built analytically, from `dw3d_benchmarks.angle_error_budget.build_wedge`, so it
needs no image, no dataset and no reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest

from dw3d_benchmarks.angle_error_budget import build_wedge
from dw3d_benchmarks.mesh_quality_comparison import case_metrics

# Materials 1, 2, 3 for the first wedge; the same three for the second, so both copies share
# one material triple while being geometrically disconnected.
SHARED_TRIPLE = (1, 2, 3)
WEDGE_EDGE_LENGTH = 1.0
WEDGE_SEGMENTS = 6
#: Far enough apart that no vertex or edge of one copy touches the other.
COPY_OFFSET = np.array([50.0, 50.0, 0.0])


def _one_wedge() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, triangles, labels, _is_line = build_wedge(
        edge_length=WEDGE_EDGE_LENGTH, n_segments=WEDGE_SEGMENTS, n_rings=1,
    )
    return np.asarray(points, dtype=np.float64), np.asarray(triangles), np.asarray(labels)


def _two_disconnected_wedges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two copies of the same wedge, translated apart: one material triple, two components."""
    points, triangles, labels = _one_wedge()
    shifted = points + COPY_OFFSET
    return (
        np.vstack([points, shifted]),
        np.vstack([triangles, triangles + len(points)]),
        np.vstack([labels, labels]),
    )


def _as_ground_truth(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> dict:
    return {"points": points, "triangles": triangles, "labels": labels, "tensions": None}


def test_single_wedge_has_exactly_one_triple_and_one_component() -> None:
    points, triangles, labels = _one_wedge()
    coverage = case_metrics.compute_m1_component_coverage(points, triangles, labels)
    assert coverage["n_triples"] == 1
    assert coverage["n_components"] == 1
    assert coverage["n_multi_component_triples"] == 0
    assert coverage["arclength_fraction_on_multi_component_triples"] == pytest.approx(0.0)


def test_two_disconnected_lines_sharing_a_triple_are_one_key_and_two_components() -> None:
    """The distinction the triple key cannot express: one key, two separate lines."""
    points, triangles, labels = _two_disconnected_wedges()
    coverage = case_metrics.compute_m1_component_coverage(points, triangles, labels)
    assert coverage["n_triples"] == 1, "the two lines share one material triple, so one key"
    assert coverage["n_components"] == 2, "and are two disconnected lines"
    assert coverage["n_multi_component_triples"] == 1
    # Every wedge angle on this mesh is pooled across both lines by production's convention.
    assert coverage["arclength_fraction_on_multi_component_triples"] == pytest.approx(1.0)


def test_component_path_agrees_with_triple_keyed_path_on_a_single_component_triple() -> None:
    """Same chain in, same numbers out -- the precondition for trusting the component path."""
    points, triangles, labels = _one_wedge()
    ground_truth = _as_ground_truth(points, triangles, labels)
    identity = {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3), "residual_mean_voxels": 0.0}

    triple_keyed = case_metrics.compute_m5_junction_geometry(points, triangles, labels, ground_truth, identity)
    per_component = case_metrics.compute_m5_component_geometry(points, triangles, labels, ground_truth, identity)

    for key in (
        "tortuosity_ratio_median",
        "tangent_error_mean_deg",
        "tangent_error_p90_deg",
        "curvature_ratio_median",
        "line_position_error_median_voxels",
    ):
        left, right = triple_keyed[key], per_component[key]
        # `curvature_ratio_median` is `nan` on this fixture and must be on both sides: the
        # reference line is exactly straight, so the ratio's denominator (total turning per unit
        # arclength) is zero and the existing estimator returns `nan` by its own rule. `nan` is
        # the agreement here, not a gap in it -- so compare it as such rather than numerically.
        if left is not None and np.isnan(left):
            assert right is not None, key
            assert np.isnan(right), key
            continue
        assert left == pytest.approx(right, abs=1e-12, rel=0), key
    assert per_component["n_matched_components"] == 1
    assert per_component["n_unmatched_components_reconstruction"] == 0
    assert per_component["n_unmatched_components_ground_truth"] == 0
    # A single component whose vertices all have degree <= 2 is one chain, so the triple-keyed
    # path covers all of it.
    assert per_component["arclength_fraction_scored_by_triple_keyed_path"] == pytest.approx(1.0)


def test_component_path_scores_both_lines_where_the_triple_keyed_path_scores_one() -> None:
    """The coverage claim, on the fixture built to expose it."""
    points, triangles, labels = _two_disconnected_wedges()
    ground_truth = _as_ground_truth(points, triangles, labels)
    identity = {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3), "residual_mean_voxels": 0.0}

    triple_keyed = case_metrics.compute_m5_junction_geometry(points, triangles, labels, ground_truth, identity)
    per_component = case_metrics.compute_m5_component_geometry(points, triangles, labels, ground_truth, identity)

    assert triple_keyed["n_common_triples"] == 1, "one material triple, so one triple-keyed comparison"
    assert per_component["n_components_reconstruction"] == 2
    assert per_component["n_components_ground_truth"] == 2
    assert per_component["n_matched_components"] == 2, "both lines matched, one to one"
    assert per_component["n_unmatched_components_reconstruction"] == 0
    assert per_component["n_unmatched_components_ground_truth"] == 0
    # The triple-keyed path scores the longest chain of one component out of two equal ones.
    assert per_component["arclength_fraction_scored_by_triple_keyed_path"] == pytest.approx(0.5, abs=0.02)
