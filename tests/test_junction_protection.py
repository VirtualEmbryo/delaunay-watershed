"""Junction protection: stratum detection, junction sampling, protection, and the measured outcome.

Junction protection adds 0-/1-junction detection from the label image (after Boltcheva et al.),
protecting balls around the samples, a regular (weighted) triangulation that consumes the
ball radii, and shell coarsening of the bounding-box minima. Like the boundary layer it is
an **opt-in** variant, so nothing here touches the default golden masters (behaviour changes
ship as opt-in variants with their own golden masters).

What is pinned here, in order:

1. **The detector is exactly right**, on a synthetic label cube whose `|L|` is known by
   construction — the discriminating test, since every downstream number depends on it.
   Plus the open question of reusing `edt.get_total_boundaries`, resolved in the negative
   with a demonstration rather than an assertion: `edt.get_total_boundaries` is *not* the label multiplicity.
2. **Sampling invariants**: one sample per 0-stratum component, spacing respected along the
   1-strata, and every recorded pair genuinely consecutive.
3. **Determinism**, to the same standard as the deterministic algorithm and the boundary layer: bit-identical across
runs with the
   global RNG disturbed in between, and the placement leaves that RNG untouched.
4. **The weighted tesselation's contract**: all-zero (and all-equal) weights reproduce plain
   Delaunay exactly, and hiding a point needs the surrounding balls to *enclose* it — one
   heavier neighbour is not enough. The second of those corrects the reasoning junction
   protection was first
   written on and is why the protecting-ball exclusion is done explicitly.
5. **Junction preservation, measured not assumed** — an explicit requirement of the
   specification, since Boltcheva's guarantee is for Delaunay *refinement* and `dw3d` is one-shot.
6. **The falsified part of the specification, as a standing regression fixture.** The junction
   boundary layer (the boundary layer around junction samples) makes topology worse, not
   better; the test asserts the failure so it cannot be quietly forgotten or silently "fixed" by a change
   that does not actually address the mechanism.
"""

import numpy as np
import pytest

from dw3d_benchmarks import metrics as m
from dw3d_benchmarks.fingerprint import FINGERPRINT_FIELDS, fingerprint_reconstruction
from dw3d import get_junction_protected_mesh_reconstruction_algorithm
from dw3d.edt import compute_edt_classical, get_total_boundaries
from dw3d.junctions import label_multiplicity, sample_junction_network
from dw3d.points_on_edt import junction_protected_families, peak_local_points_junction_protected
from dw3d.tesselation import regular_tesselation, simple_delaunay_tesselation, weighted_delaunay_tesselation
from tests.conftest import load_image_or_skip

MIN_DISTANCE = 3
CASE_IMAGE = "3.tif"
JUNCTION_SPACING = 2 * MIN_DISTANCE + 1  # the effective interface spacing h_S; see points_on_edt


@pytest.fixture(scope="module")
def edt_3tif():
    return compute_edt_classical(load_image_or_skip(CASE_IMAGE))


@pytest.fixture(scope="module")
def reconstruction_3tif():
    mask = load_image_or_skip(CASE_IMAGE)
    algo = get_junction_protected_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
    # Junction preservation is measured on the construction itself -- whether the protected junction
    # samples survive as mesh vertices and edges, matched by position. Junction relocation (on by
    # default since 0.5.0) then moves those vertices onto the mask's curves by design, so it is
    # switched off explicitly here; tests/test_default_junction_relocation.py covers the relocated path.
    algo.relocate_junctions = False
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    return algo, points, triangles, labels


def _quadrant_cube(size: int = 12) -> np.ndarray:
    """A cube split into four labelled quadrants about one axis-aligned line.

    Labels 1..4 occupy the four `(y, z)` quadrants and run the full length of `x`, so:

    * the line `y = z = size/2` is a genuine **1-D quadruple junction** — four materials
      meet along it, giving `|L| = 4` on the dual lattice all along it;
    * the two half-planes `y = size/2` and `z = size/2` are ordinary interfaces, `|L| = 2`;
    * everywhere else is material interior, `|L| = 1`.

    Deliberately a *quadruple line* rather than a quadruple point: it makes `|L| = 4`
    exactly enumerable (`size - 1` dual positions) so the test can assert a count, not a
    property. Nothing else in the suite depends on this shape being physical — a real dry
    foam has no quadruple line, which is exactly why `dw3d` reports one as a defect.
    """
    half = size // 2
    mask = np.zeros((size, size, size), dtype=np.int32)
    mask[:, :half, :half] = 1
    mask[:, :half, half:] = 2
    mask[:, half:, :half] = 3
    mask[:, half:, half:] = 4
    return mask


# --------------------------------------------------------------------------------------
# 1. The detector, against a case whose answer is known by construction
# --------------------------------------------------------------------------------------


def test_label_multiplicity_is_exact_on_a_constructed_cube():
    size = 12
    mask = _quadrant_cube(size)
    multiplicity = label_multiplicity(mask)

    assert multiplicity.shape == (size - 1, size - 1, size - 1)
    assert multiplicity.min() == 1
    assert multiplicity.max() == 4

    half = size // 2
    # The quadruple line: dual index (i, half-1, half-1) for every i.
    quadruple = np.argwhere(multiplicity >= 4)
    assert len(quadruple) == size - 1
    np.testing.assert_array_equal(np.unique(quadruple[:, 1]), [half - 1])
    np.testing.assert_array_equal(np.unique(quadruple[:, 2]), [half - 1])

    # The two interface half-planes, minus the shared line, are |L| == 2; nothing is 3.
    assert not (multiplicity == 3).any()
    assert (multiplicity == 2).sum() == 2 * (size - 1) * (size - 2)


def test_label_multiplicity_matches_a_brute_force_count():
    """The vectorised 28-comparison count agrees with the obvious slow definition."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 4, size=(7, 8, 9))
    multiplicity = label_multiplicity(mask)
    for index in np.ndindex(multiplicity.shape):
        block = mask[index[0] : index[0] + 2, index[1] : index[1] + 2, index[2] : index[2] + 2]
        assert multiplicity[index] == len(np.unique(block)), index


def test_get_total_boundaries_is_not_the_label_multiplicity():
    """An open question, resolved in the negative — demonstrated, not asserted in prose.

    `get_total_boundaries` was a candidate for reuse as the `|L|` detector. It is a per-label
    sum of thick `find_boundaries` masks, inverted by `max - x`, so it lives on the primal
    lattice, is dilated relative to `|L|`, and does not take the same values. The test shows
    the disagreement rather than trusting the argument.
    """
    mask = _quadrant_cube()
    multiplicity = label_multiplicity(mask)
    boundaries = get_total_boundaries(mask)

    assert boundaries.shape == mask.shape  # primal lattice, not the dual one
    # Compare on the overlapping index range; the value sets are not even the same.
    overlap = boundaries[:-1, :-1, :-1]
    assert set(np.unique(overlap).tolist()) != set(np.unique(multiplicity).astype(float).tolist())
    assert not np.array_equal(overlap, multiplicity.astype(overlap.dtype))


# --------------------------------------------------------------------------------------
# 2. Sampling invariants
# --------------------------------------------------------------------------------------


def test_junction_sampling_covers_the_constructed_quadruple_line():
    mask = _quadrant_cube(size=24)
    network = sample_junction_network(mask, spacing=4.0)

    samples = network["points"].astype(int)
    assert len(samples) > 0
    # Every sample sits on the quadruple line, i.e. at the two central dual indices.
    np.testing.assert_array_equal(np.unique(samples[:, 1]), [11])
    np.testing.assert_array_equal(np.unique(samples[:, 2]), [11])
    # The line is 23 dual positions long and sampled at spacing 4, so ~6 intervals.
    assert 5 <= len(samples) <= 9

    # Recorded pairs must join samples that really are ~`spacing` apart along the line.
    pairs = network["pairs"]
    assert len(pairs) == len(samples) - 1
    gaps = np.linalg.norm(samples[pairs[:, 0]] - samples[pairs[:, 1]], axis=1)
    assert gaps.max() <= 4.0 + 1.0  # a voxel of slack from snapping targets to skeleton voxels


def test_junction_sampling_is_deterministic():
    mask = load_image_or_skip(CASE_IMAGE)
    first = sample_junction_network(mask, JUNCTION_SPACING)
    second = sample_junction_network(mask, JUNCTION_SPACING)
    np.testing.assert_array_equal(first["points"], second["points"])
    np.testing.assert_array_equal(first["pairs"], second["pairs"])
    np.testing.assert_array_equal(first["is_corner_sample"], second["is_corner_sample"])


def test_junction_sampling_rejects_a_non_positive_spacing():
    with pytest.raises(ValueError, match="spacing must be positive"):
        sample_junction_network(_quadrant_cube(), spacing=0.0)


# --------------------------------------------------------------------------------------
# 3. Determinism of the whole variant
# --------------------------------------------------------------------------------------


def test_junction_protected_is_bit_identical_across_runs_and_rng_states():
    mask = load_image_or_skip(CASE_IMAGE)

    def factory():
        return get_junction_protected_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)

    np.random.seed(0)  # noqa: NPY002 - deliberately perturb the legacy global RNG
    first = fingerprint_reconstruction(mask, factory)
    np.random.seed(999)  # noqa: NPY002
    np.random.rand(1000)  # noqa: NPY002 - advance the stream too
    second = fingerprint_reconstruction(mask, factory)

    for field in FINGERPRINT_FIELDS:
        assert first[field] == second[field], field
    assert first["joint"] == second["joint"]


def test_junction_protected_placement_does_not_touch_the_global_rng(edt_3tif):
    mask = load_image_or_skip(CASE_IMAGE)
    np.random.seed(1234)  # noqa: NPY002
    expected = np.random.rand(5)  # noqa: NPY002

    np.random.seed(1234)  # noqa: NPY002
    peak_local_points_junction_protected(mask, edt_3tif, MIN_DISTANCE)
    after = np.random.rand(5)  # noqa: NPY002

    np.testing.assert_array_equal(after, expected)


# --------------------------------------------------------------------------------------
# 4. The weighted (regular) tesselation's contract
# --------------------------------------------------------------------------------------


def test_zero_weights_reproduce_plain_delaunay_exactly():
    rng = np.random.default_rng(1)
    points = rng.integers(0, 40, size=(200, 3)).astype(np.uint64)
    plain_points, plain_tets = simple_delaunay_tesselation(points)

    for weights in (None, np.zeros(len(points))):
        weighted_points, weighted_tets = weighted_delaunay_tesselation(points, weights)
        np.testing.assert_array_equal(weighted_points, plain_points)
        np.testing.assert_array_equal(weighted_tets, plain_tets)


def test_uniform_weights_change_nothing():
    """Only weight *differences* matter: a constant shifts the whole lift vertically.

    This is why a protected junction cluster, whose members all carry `r_J**2`, is
    triangulated internally exactly as it would have been unweighted — the property
    junction protection relies on to keep every member of a cluster present.
    """
    rng = np.random.default_rng(3)
    points = np.unique(rng.integers(0, 40, size=(200, 3)), axis=0).astype(float)
    reference = np.unique(np.sort(regular_tesselation(points, np.zeros(len(points))), axis=1), axis=0)
    for weight in (7.0, 123.4):
        tets = np.unique(np.sort(regular_tesselation(points, np.full(len(points), weight)), axis=1), axis=0)
        np.testing.assert_array_equal(tets, reference)


def test_hiding_needs_enclosure_not_a_single_heavy_neighbour():
    """Corrects the reasoning junction protection was first written on, and pins what is actually true.

    A weighted point is hidden only when its power cell is *empty*. One heavier neighbour can
    never do that — the power bisector of two weighted points is a plane, so each keeps a
    half-space. It takes a surrounding shell of balls that together enclose the point.

    This is why junction protection excludes the crowding points from a protecting ball
    **explicitly** instead of trusting the weights to hide them, and why the ablation measured the weights
    as contributing almost nothing on their own. The test walks the radius up so the threshold
    behaviour is visible rather than asserted at one lucky value.
    """
    distance = 6.0
    centre = np.array([20.0, 20.0, 20.0])
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    shell = centre + distance * directions
    far = np.array(
        [[0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40], [40, 40, 40], [0, 40, 40], [40, 0, 40], [40, 40, 0]],
        dtype=float,
    )
    points = np.vstack((far, shell, centre[None, :]))
    victim = len(points) - 1

    present = {}
    for radius in (0.0, 5.0, 7.0):
        weights = np.zeros(len(points))
        weights[len(far) : len(far) + len(shell)] = radius**2
        _, tets = weighted_delaunay_tesselation(points, weights)
        present[radius] = victim in set(np.unique(tets).tolist())

    assert present[0.0], "unweighted, the enclosed point is an ordinary Delaunay vertex"
    assert present[5.0], "a heavy neighbour at distance 6 does not hide a point at radius 5"
    assert not present[7.0], "once the shell balls enclose it (radius > distance) it is hidden"


# --------------------------------------------------------------------------------------
# 5. Shell coarsening and junction preservation, on 3.tif
# --------------------------------------------------------------------------------------


def test_shell_classification_reproduces_the_boundary_layers_independent_measurement(edt_3tif):
    """Junction protection's stratum detector must agree with the boundary layer's straddle test.

    Both decide which minima are real.

    The boundary layer measured, by probing `p +- delta*n` and reading the labels, that
    1172 of `3.tif`'s 5228 EDT minima lie on a genuine >=2-label interface. Junction
    protection reaches the same split from a
    completely different direction — the 2x2x2 label multiplicity, no EDT, no normal, no
    offset — and must get the same numbers. Agreement between two independent derivations
    is the reason to believe either.
    """
    mask = load_image_or_skip(CASE_IMAGE)
    families = junction_protected_families(mask, edt_3tif, MIN_DISTANCE, shell_coarsening=1)
    shell = families["shell"]
    assert shell["n_minima_before"] == 5228
    assert shell["n_interface_minima"] == 1172
    assert shell["n_shell_minima"] == 4056
    # 77.6 % of the "interface" point budget is the bounding-box / background shell.
    assert shell["n_shell_minima"] / shell["n_minima_before"] == pytest.approx(0.776, abs=0.005)


def test_shell_coarsening_cuts_the_shell_budget_without_touching_real_interfaces(edt_3tif):
    mask = load_image_or_skip(CASE_IMAGE)
    coarse = junction_protected_families(mask, edt_3tif, MIN_DISTANCE, shell_coarsening=3)["shell"]
    assert coarse["n_interface_minima"] == 1172  # unchanged: only |L| == 1 minima are re-packed
    assert coarse["n_shell_minima_kept"] < coarse["n_shell_minima"] / 5
    assert coarse["n_minima_after"] < 0.4 * coarse["n_minima_before"]


def test_junction_preservation_is_measured_and_high(reconstruction_3tif):
    """An explicit requirement of the specification: *measure* preservation, do not assume it.

    Boltcheva's protecting-ball theorem is stated for Delaunay refinement; `dw3d` is
    one-shot, so all we are entitled to expect is a strong bias. The measurement says the
    bias is in practice near-total on this case, and the numbers are pinned so a future
    change that quietly loses the junction network is caught.
    """
    algo, points, triangles, _ = reconstruction_3tif
    mask = algo._segmented_image
    families = junction_protected_families(
        mask,
        algo._edt_image,
        MIN_DISTANCE,
        shell_coarsening=3,
        junction_boundary_layer=False,
    )
    stats = m.junction_preservation_stats(families, algo._tesselation_graph, points, triangles)

    assert stats["n_junction_samples"] > 100
    assert stats["n_sampled_pairs"] >= stats["n_junction_samples"]  # the network has cycles
    assert stats["tesselation_vertex_preservation"] == 1.0
    assert stats["tesselation_edge_preservation"] > 0.95
    # The mesh-edge figure is the one that matters and is bounded by the two above.
    assert stats["mesh_edge_preservation"] > 0.90
    assert stats["mesh_edge_preservation"] <= stats["tesselation_edge_preservation"]


def test_junction_preservation_is_near_zero_without_protection(reconstruction_3tif):
    """The control for the test above: a high preservation figure must mean something.

    Measure the *same* sampled junction network against the mesh the **deterministic**
    algorithm builds — which places no junction samples at all. If the metric were near 1
    there too, it would
    be measuring the metric's own leniency rather than the protection. It is near 0, so the
    ~1.0 above is a real property of the protected construction.
    """
    from dw3d import get_deterministic_mesh_reconstruction_algorithm

    algo, _, _, _ = reconstruction_3tif
    mask = algo._segmented_image
    families = junction_protected_families(
        mask,
        algo._edt_image,
        MIN_DISTANCE,
        shell_coarsening=3,
        junction_boundary_layer=False,
    )

    unprotected = get_deterministic_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE)
    points, triangles, _ = unprotected.construct_mesh_from_segmentation_mask(mask)
    stats = m.junction_preservation_stats(families, unprotected._tesselation_graph, points, triangles)

    assert stats["vertex_preservation"] < 0.05
    assert stats["mesh_edge_preservation"] < 0.05


# --------------------------------------------------------------------------------------
# 6. The falsified part of the specification, pinned as a negative result
# --------------------------------------------------------------------------------------


def test_the_junction_boundary_layer_makes_topology_worse(edt_3tif):
    """The junction boundary layer is FALSIFIED; this fixture records it with numbers.

    The specification asks for the boundary layer around junction samples on the grounds that it
    prevents Perez's "more points locally -> more small tets" failure. It causes it: the
    junction offsets land inside the protecting balls the same construction has just
    emptied, so the net effect is the crowding without the protection. Asserted here as a
    *failure*, so that turning the layer back on cannot silently pass the suite.
    """
    mask = load_image_or_skip(CASE_IMAGE)
    without = get_junction_protected_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE)
    with_layer = get_junction_protected_mesh_reconstruction_algorithm(
        min_distance=MIN_DISTANCE,
        junction_boundary_layer=True,
    )
    counts = []
    for algo in (without, with_layer):
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
        counts.append(
            (
                m.edge_topology_stats(points, triangles, labels)["n_valence_geq4_edges"],
                m.abnormal_non_manifold_edge_count(points, triangles, labels),
            ),
        )
    (valence_without, abnormal_without), (valence_with, abnormal_with) = counts

    assert valence_with > valence_without, "step 5 was measured to be harmful; this must stay recorded"
    assert abnormal_with >= abnormal_without

    # And the offsets really are emitted when asked for — the layer is implemented, not
    # stubbed out, so the negative result is about the method and not about dead code.
    families = junction_protected_families(mask, edt_3tif, MIN_DISTANCE, junction_boundary_layer=True)
    assert len(families["junction_offsets"]) > 100
