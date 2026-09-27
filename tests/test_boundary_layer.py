"""The boundary-layer point placement, its orientation, and its measured effect.

The boundary-layer scheme keeps the deterministic extrema and adds offset
points a fixed distance into each material adjacent to a genuine interface, to break up
the all-surface tetrahedra that starve the watershed. It is an **opt-in** variant: the
default remains the deterministic extrema-only rule (behaviour changes ship as opt-in
variants with their own golden masters), so nothing here touches the default golden masters.

Three things are pinned, in order:

1. **Determinism.** The scheme uses no random number; two runs agree bit-for-bit on all
   seven fingerprinted arrays even with the global RNG disturbed between them, and the
   placement leaves the global RNG untouched. Same discipline as `test_determinism.py`.
2. **Orientation, verified not assumed** (the scheme is falsified if it fails). Every emitted
   offset is checked against the *label image*: it must straddle two genuinely-adjacent
   materials. On `3.tif`/md=3, 1172 of 5228 EDT minima sit on a real >=2-label boundary and
   100 % of those produce a correctly-oriented pair; the rest are bounding-box/background
   plateau minima that correctly get no offset.
3. **The acceptance numbers as measured** on `3.tif`/md=3, pinned so the mixed outcome of the
   boundary-layer work is a regression fixture rather than a claim. Two criteria pass (all-surface tets, sliver
   fraction), and the score-gap-below-1e-3 target does **not** (it rises, a density effect
   of ~2x more faces; the finer 1e-5 degeneracy that motivated the work falls). The
   asserts below record the true values and their pass/fail against the boundary-layer
   targets; 8 configurations were measured in all, and work on the scheme stopped at this
   mixed outcome.
"""

import numpy as np
import pytest

from dw3d_benchmarks import metrics as m
from dw3d_benchmarks.fingerprint import FINGERPRINT_FIELDS, fingerprint_reconstruction
from dw3d import get_boundary_layer_mesh_reconstruction_algorithm
from dw3d.edt import compute_edt_classical
from dw3d.points_on_edt import peak_local_points_boundary_layer
from tests.conftest import load_image_or_skip

MIN_DISTANCE = 3
CASE_IMAGE = "3.tif"


@pytest.fixture(scope="module")
def edt_3tif():
    return compute_edt_classical(load_image_or_skip(CASE_IMAGE))


@pytest.fixture(scope="module")
def reconstruction_3tif():
    mask = load_image_or_skip(CASE_IMAGE)
    algo = get_boundary_layer_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    return algo, points, triangles, labels


# --------------------------------------------------------------------------------------
# 1. Determinism
# --------------------------------------------------------------------------------------


def test_boundary_layer_is_bit_identical_across_runs_and_rng_states():
    mask = load_image_or_skip(CASE_IMAGE)

    def factory():
        return get_boundary_layer_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)

    np.random.seed(0)  # noqa: NPY002 - deliberately perturb the legacy global RNG
    first = fingerprint_reconstruction(mask, factory)
    np.random.seed(999)  # noqa: NPY002
    np.random.rand(1000)  # noqa: NPY002 - advance the stream too
    second = fingerprint_reconstruction(mask, factory)

    for field in FINGERPRINT_FIELDS:
        assert first[field] == second[field], field
    assert first["joint"] == second["joint"]


def test_boundary_layer_placement_does_not_touch_the_global_rng(edt_3tif):
    mask = load_image_or_skip(CASE_IMAGE)
    np.random.seed(1234)  # noqa: NPY002
    expected = np.random.rand(5)  # noqa: NPY002

    np.random.seed(1234)  # noqa: NPY002
    peak_local_points_boundary_layer(mask, edt_3tif, MIN_DISTANCE)
    after = np.random.rand(5)  # noqa: NPY002

    np.testing.assert_array_equal(after, expected)


# --------------------------------------------------------------------------------------
# 2. Orientation, verified against the label image
# --------------------------------------------------------------------------------------


def test_every_emitted_offset_is_correctly_oriented(edt_3tif):
    mask = load_image_or_skip(CASE_IMAGE)
    stats = m.boundary_layer_orientation_stats(mask, edt_3tif, MIN_DISTANCE)

    # These pin the numbers measured for the boundary-layer work; they move only if the EDT,
    # the plateau rule, or the offset construction changes.
    assert stats["n_minima"] == 5228
    assert stats["n_on_interface"] == 1172
    assert stats["n_offsets"] == 2344  # 2 per straddling sample, none lost to dedup here

    # The point of the test: orientation is *verified* against the label image, and every
    # emitted offset passes. Check, do not assume.
    assert stats["n_orientation_checked"] == stats["n_on_interface"]
    assert stats["orientation_ok_fraction"] == 1.0
    # ~22 % of EDT minima are on a real interface; the rest are the outer shell (no offset).
    assert stats["interface_yield"] == pytest.approx(0.224, abs=0.01)


# --------------------------------------------------------------------------------------
# 3. Acceptance metrics as measured (mixed outcome; see the module docstring)
# --------------------------------------------------------------------------------------


def _surface_and_score_stats(algo):
    counts = m.point_placement_counts(
        algo._edt_image,
        MIN_DISTANCE,
        point_placing_function=algo.point_placing_function,
        segmented_image=algo._segmented_image,
    )
    point_classes = m.classify_tesselation_points(len(algo._tesselation_graph.vertices), counts["n_interior_points"])
    surf = m.tetrahedron_surface_stats(algo._tesselation_graph, point_classes)
    quality = m.tetrahedron_quality(algo._tesselation_graph)
    sliv = m.sliver_stats(quality, surf["_all_surface_mask"])
    score = m.score_gap_stats(algo._tesselation_graph)
    return surf, sliv, score


def test_all_surface_tet_fraction_meets_a3_target(reconstruction_3tif):
    """Boundary-layer target < 15 %, deterministic "before" 42.0 %. PASSES."""
    algo, _, _, _ = reconstruction_3tif
    surf, _, _ = _surface_and_score_stats(algo)
    assert surf["fraction_all_surface_tets"] == pytest.approx(0.0837, abs=0.003)  # deterministic: 0.420
    assert surf["fraction_all_surface_tets"] < 0.15  # boundary-layer target


def test_sliver_fraction_holds_below_target(reconstruction_3tif):
    """Boundary-layer target < 7 % (hold, not regress). Deterministic 6.25 % -> boundary layer ~6.4 %. HELD."""
    algo, _, _, _ = reconstruction_3tif
    _, sliv, _ = _surface_and_score_stats(algo)
    assert sliv["sliver_fraction"] == pytest.approx(0.0638, abs=0.003)  # deterministic: 0.0625
    assert sliv["sliver_fraction"] < 0.07  # boundary-layer target


def test_score_gap_below_1e3_does_not_meet_target(reconstruction_3tif):
    """Boundary-layer target < 40 %; the boundary layer moves it the WRONG way (density effect).

    Pinned as a negative result, not a pass. The finer 1e-5 degeneracy — the bit-level
    ambiguity measured on the dithered baseline — falls (48.9 % -> 37.0 %); the 1e-3 fraction rises because
    ~2x more tesselation faces pack the same score range.
    """
    algo, _, _, _ = reconstruction_3tif
    _, _, score = _surface_and_score_stats(algo)
    assert score["fraction_score_gaps_below_1e-3"] == pytest.approx(0.886, abs=0.005)  # deterministic: 0.774
    assert score["fraction_score_gaps_below_1e-3"] > 0.40  # target NOT met (recorded, not tuned)
    assert score["fraction_score_gaps_below_1e-5"] == pytest.approx(0.370, abs=0.005)  # deterministic: 0.489 (improved)
