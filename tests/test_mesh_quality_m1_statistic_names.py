"""M1's statistics must stay named for what they measure, and the old names must keep working.

The regression these pin is a **naming** regression, and it has already cost this project a
ranking. Two quantities the comparison trades off are both *bias* statistics -- the flattening
slope is the differential bias against a wedge's deviation from 120 degrees, and the "angle
median" is `median(signed error)`, the net bias, which alternating errors of opposite sign
cancel out of. They were read under accuracy-sounding names, and no accuracy statistic was
emitted at all.

Pinned here:

1. every historical key is still emitted, so results JSON written before the renaming stays
   comparable and no reader's extractor breaks;
2. `median_abs_error_deg` really is `median(|e|)`, and differs from `|median(e)|` on a
   distribution with scatter -- the two are not interchangeable;
3. `net_offset_deg` is exactly `flattening_slope_length_weighted["intercept"]` -- a name for an
   existing number, not a second computation that could drift from it;
4. `net_offset_ci95` is labelled `wedge`-level, because a per-case block cannot compute the
   case-level interval this package's cohort statistics use;
5. every emitted statistic has a `_statistics` descriptor saying what it measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from dw3d_benchmarks.mesh_quality_comparison import case_metrics

_HISTORICAL_KEYS = (
    "n_common_wedges",
    "signed_error_median_deg",
    "abs_error_p90_deg",
    "flattening_slope_unweighted",
    "flattening_slope_length_weighted",
)
_ADDED_KEYS = ("median_abs_error_deg", "abs_error_iqr_deg", "net_offset_deg", "net_offset_ci95")


#: Eight analytic wedges, each on its own material triple, so the block carries 24 wedge keys
#: rather than 3. One wedge gives three errors that can all share a sign by chance, and then a
#: net-bias statistic and an accuracy statistic coincide numerically -- which would let the test
#: pass while proving nothing about the distinction it exists to pin.
_N_WEDGE_COPIES = 8


def _wedge_block() -> dict:
    """One M1 reference block, measured on a mesh whose vertices are perturbed analytically.

    Eight copies of an exact 120-degree analytic wedge, each labelled with its own three
    materials and translated clear of the others, then every vertex displaced by i.i.d. noise.
    The reference is the unperturbed mesh, so the errors have real scatter with a small net bias
    -- the configuration in which a net-bias statistic and an accuracy statistic differ most,
    which is the whole point of separating them.
    """
    from foambryo.dcel import DcelData

    from dw3d_benchmarks.angle_error_budget import build_wedge

    base_points, base_triangles, base_labels, _is_line = build_wedge(edge_length=8.0, n_segments=8, n_rings=1)
    base_points = np.asarray(base_points, dtype=np.float64)
    base_triangles = np.asarray(base_triangles)
    base_labels = np.asarray(base_labels)

    points_blocks, triangle_blocks, label_blocks = [], [], []
    for copy in range(_N_WEDGE_COPIES):
        points_blocks.append(base_points + np.array([300.0 * copy, 0.0, 0.0]))
        triangle_blocks.append(base_triangles + copy * len(base_points))
        # Materials 1,2,3 for the first copy, 4,5,6 for the second, and so on, so each copy is
        # its own material triple and contributes three distinct wedge keys.
        label_blocks.append(base_labels + 3 * copy)
    points = np.vstack(points_blocks)
    triangles = np.vstack(triangle_blocks)
    labels = np.vstack(label_blocks)

    rng = np.random.default_rng(20260901)
    perturbed = points + rng.normal(0.0, 0.5, size=points.shape)
    mesh = DcelData(perturbed, triangles, labels)
    gt_mesh = DcelData(points, triangles, labels)
    result = case_metrics.compute_m1_contact_angles(mesh, gt_mesh, None, 1.9573)
    return result["vs_ground_truth_mesh"]


def test_every_historical_key_is_still_emitted() -> None:
    block = _wedge_block()
    for key in _HISTORICAL_KEYS:
        assert key in block, key
    for key in ("raw_true_minus_120_deg", "raw_signed_error_deg", "raw_length_weight"):
        assert key in block, key


def test_added_keys_are_present() -> None:
    block = _wedge_block()
    for key in _ADDED_KEYS:
        assert key in block, key


def test_accuracy_statistic_is_the_median_absolute_error_not_the_absolute_median() -> None:
    block = _wedge_block()
    errors = np.asarray(block["raw_signed_error_deg"], dtype=np.float64)
    assert block["median_abs_error_deg"] == pytest.approx(float(np.median(np.abs(errors))), abs=1e-12, rel=0)
    assert block["signed_error_median_deg"] == pytest.approx(float(np.median(errors)), abs=1e-12, rel=0)
    # The two are different statistics, and on a scattered distribution they are not close.
    assert abs(block["median_abs_error_deg"]) > abs(block["signed_error_median_deg"])


def test_spread_statistic_is_the_interquartile_range_of_the_absolute_error() -> None:
    block = _wedge_block()
    absolute = np.abs(np.asarray(block["raw_signed_error_deg"], dtype=np.float64))
    expected = float(np.percentile(absolute, 75) - np.percentile(absolute, 25))
    assert block["abs_error_iqr_deg"] == pytest.approx(expected, abs=1e-12, rel=0)


def test_net_offset_is_the_length_weighted_intercept_and_not_a_second_computation() -> None:
    block = _wedge_block()
    assert block["net_offset_deg"] == pytest.approx(
        block["flattening_slope_length_weighted"]["intercept"], abs=1e-12, rel=0,
    )


def test_net_offset_interval_declares_itself_wedge_level() -> None:
    block = _wedge_block()
    assert block["net_offset_ci95"]["level"] == "wedge"
    assert block["net_offset_ci95"]["seed"] == 20260901
    assert block["net_offset_ci95"]["lo"] <= block["net_offset_deg"] <= block["net_offset_ci95"]["hi"]


def test_every_emitted_statistic_has_a_descriptor_saying_what_it_measures() -> None:
    block = _wedge_block()
    described = block["_statistics"]
    for key in (*_HISTORICAL_KEYS, *_ADDED_KEYS, "matched_spacing_floor_deg"):
        assert key in described, key
        assert described[key]["measures"], key
        assert described[key]["unit"], key
    # The floor's descriptor must state both defects, so a reader meeting the number in a table
    # cannot take it for a measured, cross-arm-comparable quantity.
    floor_note = described["matched_spacing_floor_deg"]["note"]
    assert "asserted" in floor_note
    assert "1/ell" in floor_note
    assert "NOT comparable" in floor_note
