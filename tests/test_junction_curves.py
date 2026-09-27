"""The junction-curve API's contract, on geometry whose answer is known by construction.

`extract_junction_curves` reads a label mask and a reconstructed mesh and returns one polyline
per material triple, split by whether the mesh agrees that trijunction exists. The measured
accuracy claims live in the project's research history, against the 47-case benchmark and
a registered reference; this file pins the parts that must hold for *any* input:

1. **The detector is exactly right** on a synthetic three-material wedge whose junction line is
   a known lattice column, and on a four-material block whose quadruple point is known -- the
   discriminating test, since every downstream number depends on it. Label `0` is an ordinary
   material and must not be special-cased away.
2. **Identity is the mesh's**, not the mask's: a triple the mesh does not have lands in
   `invented` and never in `matched`, and the two lists are never merged.
3. **The estimator's vectorised azimuth scan is bit-identical** to the junction-curve-extraction
   work's own scalar loop. That
   optimisation exists only because it is measurably faster, so it may not change an answer;
   this is the regression fixture for it.
4. **Determinism**, to the same standard as the deterministic algorithm and junction
   protection: bit-identical across runs with the global RNG disturbed in between.
5. **The refusals are refusals**: a bad estimator name and a mesh outside the mask's frame both
   raise rather than returning something plausible.
"""

import numpy as np
import pytest

from dw3d import extract_junction_curves
from dw3d.junction_curve_estimator import (
    _refine_azimuths,
    _refine_azimuths_scalar,
    normal_plane_basis,
    voxels_in_window,
    wedge_structure,
)
from dw3d.junction_curves import (
    detect_junction_spines,
    dual_label_sets,
    mesh_junction_spacing,
    mesh_trijunction_components,
    subsample_indices,
)


def three_material_wedge(size=24):
    """A mask whose trijunction of (0, 1, 2) is the lattice column x = y = size // 2.

    Three axis-aligned quadrants around that column carry labels 0, 1 and 2; the fourth is
    label 2 as well, so exactly three materials meet along the column and the answer is known.
    """
    mask = np.zeros((size, size, size), dtype=np.uint16)
    half = size // 2
    mask[:half, :half] = 0
    mask[half:, :half] = 1
    mask[:, half:] = 2
    return mask, half


def a_quadruple_block(size=24):
    """Four materials meeting at one point: the octants of a cube, two of them merged."""
    mask = np.zeros((size, size, size), dtype=np.uint16)
    half = size // 2
    mask[half:, :half, :half] = 1
    mask[:half, half:, :] = 2
    mask[half:, half:, :] = 3
    mask[:half, :half, half:] = 1
    return mask


def a_mesh_for(triples, points_per_line=5):
    """A tiny multimaterial mesh carrying exactly the trijunctions in `triples`.

    Each triple gets its own chain of three triangles per edge, so the edge is shared by three
    triangles spanning three materials -- dw3d's own trijunction predicate -- and the chains do
    not touch each other.
    """
    points = []
    triangles = []
    labels = []
    for index, triple in enumerate(triples):
        base = len(points)
        offset = 10.0 * index
        spine = [[2.0 + offset, 2.0, 2.0 + step] for step in range(points_per_line)]
        points += spine
        wings = [
            [2.0 + offset + 1.0, 2.0, 2.0],
            [2.0 + offset, 2.0 + 1.0, 2.0],
            [2.0 + offset - 1.0, 2.0 - 1.0, 2.0],
        ]
        wing_base = len(points)
        points += wings
        a, b, c = sorted(triple)
        pairs = [(a, b), (b, c), (a, c)]
        for step in range(points_per_line - 1):
            for wing, pair in enumerate(pairs):
                triangles.append([base + step, base + step + 1, wing_base + wing])
                labels.append(list(pair))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


# ---------------------------------------------------------------------------------------
# 1 - the detector
# ---------------------------------------------------------------------------------------
def test_dual_label_sets_finds_exactly_the_wedge_column():
    mask, half = three_material_wedge()
    codes, table = dual_label_sets(mask)
    occupied = np.argwhere(codes >= 0)
    assert len(occupied) > 0
    # every junction point sits on the column x = y = half - 1 of the dual lattice
    assert set(occupied[:, 0]) == {half - 1}
    assert set(occupied[:, 1]) == {half - 1}
    assert all(table[c] == frozenset({0, 1, 2}) for c in codes[codes >= 0])


def test_background_label_zero_is_an_ordinary_material():
    mask, _half = three_material_wedge()
    detected = detect_junction_spines(mask)
    assert list(detected["by_triple"]) == [(0, 1, 2)]


def test_the_spine_lies_on_the_true_line_to_within_the_lattice_bound():
    mask, half = three_material_wedge()
    detected = detect_junction_spines(mask)
    spine = detected["by_triple"][(0, 1, 2)][0]
    # the dual point (half - 1, half - 1, k) is emitted at (half - 0.5, half - 0.5, k + 0.5)
    assert np.allclose(spine[:, 0], half - 0.5)
    assert np.allclose(spine[:, 1], half - 0.5)
    # and the spine runs the length of the volume, in order
    assert len(spine) >= mask.shape[2] - 2
    assert np.all(np.diff(spine[:, 2]) > 0) or np.all(np.diff(spine[:, 2]) < 0)


def test_a_quadruple_point_belongs_to_every_triple_it_contains():
    mask = a_quadruple_block()
    codes, table = dual_label_sets(mask)
    quadruple = [table[c] for c in np.unique(codes[codes >= 0]) if len(table[c]) == 4]
    assert quadruple, "the block should carry a point where four materials meet"
    detected = detect_junction_spines(mask)
    # each of the four triples contained in {0, 1, 2, 3} must have a curve reaching that point
    for triple in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        assert triple in detected["by_triple"], triple


def test_subsample_indices_are_ascending_and_keep_both_ends():
    path = np.stack([np.arange(40.0), np.zeros(40), np.zeros(40)], axis=1)
    for spacing in (0.0, 1.0, 3.0, 7.5, 100.0):
        kept = subsample_indices(path, spacing)
        assert kept[0] == 0
        assert kept[-1] == len(path) - 1
        assert np.all(np.diff(kept) > 0)
        assert np.array_equal(path[kept][0], path[0])


def test_subsample_indices_hit_the_requested_spacing():
    path = np.stack([np.arange(100.0), np.zeros(100), np.zeros(100)], axis=1)
    kept = subsample_indices(path, 8.0)
    steps = np.diff(path[kept][:, 0])
    assert abs(float(np.median(steps)) - 8.0) <= 1.0


# ---------------------------------------------------------------------------------------
# 2 - identity is the mesh's
# ---------------------------------------------------------------------------------------
def test_mesh_trijunctions_use_dw3d_s_own_predicate():
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    components = mesh_trijunction_components(points, triangles, labels)
    assert list(components) == [(0, 1, 2)]
    assert len(components[(0, 1, 2)]) == 1
    assert components[(0, 1, 2)][0]["length"] == pytest.approx(4.0)
    assert mesh_junction_spacing(components) == pytest.approx(1.0)


def test_a_triple_the_mesh_lacks_is_invented_and_never_matched():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 3)])  # not the mask's (0, 1, 2)
    result = extract_junction_curves(mask, points, triangles, labels)
    assert [c.triple for c in result.invented] == [(0, 1, 2)]
    assert result.matched == ()
    assert result.counts["invented"] == 1
    assert result.counts["missed"] == 1
    assert result.match_rate == 0.0


def test_a_triple_the_mesh_has_is_matched_and_carries_the_mesh_s_own_triple():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    result = extract_junction_curves(mask, points, triangles, labels)
    assert [c.triple for c in result.matched] == [(0, 1, 2)]
    assert result.invented == ()
    assert result.fragmented == ()
    assert result.match_rate == 1.0
    assert all(curve.matched for curve in result.matched)
    assert result.matched[0].arclength > 0


def test_matched_and_invented_are_disjoint_lists():
    mask = a_quadruple_block()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    result = extract_junction_curves(mask, points, triangles, labels)
    matched = {(c.triple, c.component) for c in result.matched}
    invented = {(c.triple, c.component) for c in result.invented}
    fragmented = {(c.triple, c.component) for c in result.fragmented}
    assert not (matched & invented)
    assert not (matched & fragmented)
    assert len(result.all_curves()) == len(matched) + len(invented) + len(fragmented)


def test_the_reading_spacing_comes_from_the_mesh_and_not_from_a_parameter():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    result = extract_junction_curves(mask, points, triangles, labels)
    assert result.spacing_voxels == pytest.approx(
        mesh_junction_spacing(mesh_trijunction_components(points, triangles, labels)),
    )


# ---------------------------------------------------------------------------------------
# 3 - the vectorised azimuth scan against the junction-curve-extraction work's own loop
# ---------------------------------------------------------------------------------------
def _a_window(seed):
    """A window of labelled voxel offsets around a synthetic junction, and its seed azimuths."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    e1, e2 = normal_plane_basis(direction)
    mask, half = three_material_wedge(28)
    centre = np.array([half - 0.5, half - 0.5, 14.0])
    offsets, materials = voxels_in_window(mask, centre, direction, 3)
    x, y = offsets @ e1, offsets @ e2
    keep = np.hypot(x, y) > 1e-9
    theta = np.arctan2(y[keep], x[keep])
    labels = materials[keep]
    base = rng.uniform(0, 2 * np.pi)
    interfaces = [
        {"pair": pair, "azimuth": float(base + k * 2 * np.pi / 3)}
        for k, pair in enumerate([(0, 1), (1, 2), (0, 2)])
    ]
    return interfaces, theta, labels


@pytest.mark.parametrize("seed", range(12))
def test_vectorised_azimuth_refinement_is_bit_identical_to_the_scalar_loop(seed):
    interfaces, theta, labels = _a_window(seed)
    if len(theta) < 20:
        pytest.skip("window too small for the estimator's own guard")
    fast = _refine_azimuths(interfaces, theta, labels, (0, 1, 2))
    slow = _refine_azimuths_scalar(interfaces, theta, labels, (0, 1, 2))
    assert (fast is None) == (slow is None)
    if fast is None:
        return
    for quick, careful in zip(fast, slow, strict=True):
        assert quick["azimuth"] == careful["azimuth"]
        assert quick["pair"] == careful["pair"]


def test_the_refined_azimuths_still_describe_convex_wedges():
    interfaces, theta, labels = _a_window(0)
    refined = _refine_azimuths(interfaces, theta, labels, (0, 1, 2))
    if refined is not None:
        assert wedge_structure(refined, (0, 1, 2)) is not None


# ---------------------------------------------------------------------------------------
# 4 - determinism
# ---------------------------------------------------------------------------------------
def test_extraction_is_bit_identical_across_runs_with_the_rng_disturbed():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    first = extract_junction_curves(mask, points, triangles, labels)
    # disturb both RNGs the codebase could plausibly touch, exactly as the determinism
    # fixture in `test_determinism.py` does, so a hidden dependence on either would show up
    # as a different curve
    np.random.default_rng(1234).normal(size=10_000)
    np.random.seed(7)  # noqa: NPY002 - the point is to disturb the legacy global RNG
    np.random.random(10_000)  # noqa: NPY002 - see above
    second = extract_junction_curves(mask, points, triangles, labels)
    assert len(first.matched) == len(second.matched)
    for a, b in zip(first.matched, second.matched, strict=True):
        assert a.triple == b.triple
        assert np.array_equal(a.points, b.points)
        assert a.arclength == b.arclength


def test_the_call_does_not_mutate_the_arrays_it_is_given():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    before = (mask.copy(), points.copy(), triangles.copy(), labels.copy())
    extract_junction_curves(mask, points, triangles, labels)
    for original, given in zip(before, (mask, points, triangles, labels), strict=True):
        assert np.array_equal(original, given)


# ---------------------------------------------------------------------------------------
# 5 - the refusals
# ---------------------------------------------------------------------------------------
def test_an_unknown_estimator_raises():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    with pytest.raises(ValueError, match="estimator must be"):
        extract_junction_curves(mask, points, triangles, labels, estimator="deepest")


def test_a_mesh_outside_the_mask_frame_raises_rather_than_measuring_the_permutation():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    with pytest.raises(ValueError, match="mask frame"):
        extract_junction_curves(mask, points * 1000.0, triangles, labels)


def test_the_default_estimator_runs_no_linear_program():
    mask, _half = three_material_wedge()
    points, triangles, labels = a_mesh_for([(0, 1, 2)])
    result = extract_junction_curves(mask, points, triangles, labels)
    assert result.estimator == "plain"
    assert all(curve.n_estimated == 0 for curve in result.all_curves())
    assert result.diagnostics["estimator_statuses"] == {}
