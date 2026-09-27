"""The pipeline is run-to-run deterministic, and the tie-breaks are the reason.

Golden masters cannot carry this. They assert `rtol=1e-10` on sorted interface areas,
cell volumes and junction lengths, which says nothing at all about the watershed's
tetrahedron labelling, the raw face scores or the tesselation-boundary bookkeeping. These
tests work on the bit-exactness fingerprint instead (`benchmarks/fingerprint.py`), which
hashes all seven arrays.

The properties pinned here, in order:

1. Run-to-run determinism, whatever the caller's global numpy RNG is doing. The default is
   `dithered`, which *does* place points with a dither — but through a private,
   freshly-seeded `RandomState` rather than the legacy global one (`seed` defaults to 42
   every call), so it is bit-identical across runs and independent of ambient global RNG
   state exactly like the RNG-free deterministic/offset-included/link-checked paths are,
   just by a different mechanism. Checked by perturbing the global numpy RNG between two
   runs of the default and asserting the point-placement functions do not read or write it.
2. The plateau rule's own guarantees (the deterministic, non-dithered point placer that
   every configuration *other than* `dithered` uses): candidate set identical to
   `peak_local_max`'s, and the accepted points pairwise `>= 2*min_distance+1` apart in
   Chebyshev distance.
3. The watershed ordering is a *total* order, so it does not depend on the sort kind.
   This one has teeth: the same test run against the historical `np.argsort(scores)` fails,
   because that ordering genuinely changes with the sort algorithm. This is independent of
   which point-placement function is in use, `dithered`'s included.
"""

import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage.feature import peak_local_max

from dw3d_benchmarks.fingerprint import FINGERPRINT_FIELDS, fingerprint_reconstruction
from dw3d import (
    get_default_mesh_reconstruction_algorithm,
    get_deterministic_mesh_reconstruction_algorithm,
    get_dithered_mesh_reconstruction_algorithm,
    get_link_checked_mesh_reconstruction_algorithm,
)
from dw3d.edt import compute_edt_classical
from dw3d.points_on_edt import peak_local_points, peak_local_points_dithered, plateau_packing_extrema
from dw3d.watershed import _seeded_watershed_aggregation, seeded_watershed_map
from tests.conftest import load_image_or_skip

MIN_DISTANCE = 3
CASE_IMAGE = "3.tif"


@pytest.fixture(scope="module")
def edt_3tif():
    return compute_edt_classical(load_image_or_skip(CASE_IMAGE))


# --------------------------------------------------------------------------------------
# 1. The default path is deterministic and RNG-free
# --------------------------------------------------------------------------------------


def test_default_reconstruction_is_bit_identical_across_runs_and_rng_states():
    """Two runs agree on all seven fingerprinted arrays, whatever the global RNG is doing.

    Disturbing `numpy.random` between the runs is the discriminating part: it is what
    would have made the historical default differ, had its seeding not been hard-wired, and
    it is what still makes the dithered variant depend on its `seed` argument.
    """
    mask = load_image_or_skip(CASE_IMAGE)

    def factory():
        return get_default_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)

    np.random.seed(0)  # noqa: NPY002 - deliberately perturbing the legacy global RNG
    first = fingerprint_reconstruction(mask, factory)
    np.random.seed(123456)  # noqa: NPY002
    np.random.rand(1000)  # noqa: NPY002 - advance the stream too, not just reseed it
    second = fingerprint_reconstruction(mask, factory)

    for field in FINGERPRINT_FIELDS:
        assert first[field] == second[field], field
    assert first["joint"] == second["joint"]


def test_default_point_placement_does_not_touch_the_global_rng(edt_3tif):
    """The historical default called `np.random.seed(42)`, resetting its caller's RNG."""
    np.random.seed(1234)  # noqa: NPY002
    expected = np.random.rand(5)  # noqa: NPY002

    np.random.seed(1234)  # noqa: NPY002
    peak_local_points(None, edt_3tif, MIN_DISTANCE)
    after_default = np.random.rand(5)  # noqa: NPY002

    np.testing.assert_array_equal(after_default, expected)

    # The dithered variant must not disturb it either: it draws from a private RandomState.
    np.random.seed(1234)  # noqa: NPY002
    peak_local_points_dithered(None, edt_3tif, MIN_DISTANCE)
    after_dithered = np.random.rand(5)  # noqa: NPY002
    np.testing.assert_array_equal(after_dithered, expected)


def test_dithered_variant_still_depends_on_its_seed(edt_3tif):
    """Control for the two tests above: the dithered path is *not* seed-independent.

    Without this, "the default is identical across runs" could pass vacuously on an image
    with no plateaus at all.
    """
    points_42, _ = peak_local_points_dithered(None, edt_3tif, MIN_DISTANCE, seed=42)
    points_7, _ = peak_local_points_dithered(None, edt_3tif, MIN_DISTANCE, seed=7)
    assert points_42.shape != points_7.shape or not np.array_equal(points_42, points_7)


def test_dithered_variant_reproduces_the_historical_global_rng_stream(edt_3tif):
    """`RandomState(seed).rand(...)` is the same stream `np.random.seed(seed)` produced.

    This is what makes `get_dithered_algorithm` a faithful restoration of the v0.3 point set
    rather than an approximation of it.
    """
    np.random.seed(42)  # noqa: NPY002
    historical = edt_3tif + np.random.rand(*edt_3tif.shape) * 1e-5  # noqa: NPY002
    expected_mins = peak_local_max(-historical, min_distance=MIN_DISTANCE, exclude_border=False)

    points, indices = peak_local_points_dithered(None, edt_3tif, MIN_DISTANCE, seed=42)
    n_maxima = len(indices)
    np.testing.assert_array_equal(points[8 + n_maxima :], expected_mins.astype(np.uint))


# --------------------------------------------------------------------------------------
# 2. The plateau rule's own guarantees
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("maximise", [True, False])
def test_plateau_candidates_are_exactly_peak_local_max_candidates(edt_3tif, maximise):
    """The rule changes *which* plateau representative is chosen, not what a plateau is.

    `plateau_packing_extrema` rebuilds `peak_local_max`'s candidate mask with a separable
    `size=` extremum filter instead of a boolean `footprint=` one (same box, faster). This
    asserts the two agree, so the rule cannot silently drift away from skimage's notion of
    a peak.
    """
    size = 2 * MIN_DISTANCE + 1
    image = edt_3tif if maximise else -edt_3tif
    footprint = np.ones((size,) * 3, dtype=bool)
    skimage_mask = (image == ndi.maximum_filter(image, footprint=footprint, mode="nearest")) & (image > image.min())

    if maximise:
        ours = (edt_3tif == ndi.maximum_filter(edt_3tif, size=size, mode="nearest")) & (edt_3tif > edt_3tif.min())
    else:
        ours = (edt_3tif == ndi.minimum_filter(edt_3tif, size=size, mode="nearest")) & (edt_3tif < edt_3tif.max())

    np.testing.assert_array_equal(ours, skimage_mask)


@pytest.mark.parametrize("maximise", [True, False])
def test_plateau_representatives_are_a_valid_packing(edt_3tif, maximise):
    """Accepted points are pairwise >= 2*min_distance+1 apart, and all are real candidates."""
    size = 2 * MIN_DISTANCE + 1
    points = plateau_packing_extrema(edt_3tif, MIN_DISTANCE, maximise=maximise).astype(np.int64)
    assert len(points) > 0

    # Every accepted point is an extremum of its own filter box.
    values = edt_3tif[points[:, 0], points[:, 1], points[:, 2]]
    if maximise:
        filtered = ndi.maximum_filter(edt_3tif, size=size, mode="nearest")
    else:
        filtered = ndi.minimum_filter(edt_3tif, size=size, mode="nearest")
    np.testing.assert_allclose(values, filtered[points[:, 0], points[:, 1], points[:, 2]])

    # Separation: no two accepted points within Chebyshev distance `size`.
    from scipy.spatial import cKDTree

    tree = cKDTree(points.astype(float))
    assert len(tree.query_pairs(r=size - 1e-9, p=np.inf)) == 0


def test_plateau_representatives_are_run_to_run_identical(edt_3tif):
    first_max = plateau_packing_extrema(edt_3tif, MIN_DISTANCE, maximise=True)
    second_max = plateau_packing_extrema(edt_3tif, MIN_DISTANCE, maximise=False)
    np.testing.assert_array_equal(first_max, plateau_packing_extrema(edt_3tif, MIN_DISTANCE, maximise=True))
    np.testing.assert_array_equal(second_max, plateau_packing_extrema(edt_3tif, MIN_DISTANCE, maximise=False))


def test_the_plateau_rule_preserves_the_dithered_point_budget(edt_3tif):
    """The determinism fix must not rebalance the surface/interior point ratio — that is the boundary layer's job.

    Bounds are deliberately loose (they hold on all four in-repo images at
    `min_distance` 3 and 5) and one-sided where it matters: the interface budget must not
    *grow*, since that is what a naive dither removal does (5.2x, see `points_on_edt`).
    """
    determinstic, det_indices = peak_local_points(None, edt_3tif, MIN_DISTANCE)
    dithered, dith_indices = peak_local_points_dithered(None, edt_3tif, MIN_DISTANCE)

    det_interface = len(determinstic) - 8 - len(det_indices)
    dith_interface = len(dithered) - 8 - len(dith_indices)
    assert 0.8 <= det_interface / dith_interface <= 1.0
    assert 0.8 <= len(determinstic) / len(dithered) <= 1.0
    # The interior count does move, by up to ~2x; pinned so the drift is visible, not hidden.
    assert 1.4 <= len(det_indices) / len(dith_indices) <= 2.2


# --------------------------------------------------------------------------------------
# 3. The watershed ordering is total
# --------------------------------------------------------------------------------------


def _aggregation_ordering_is_total(scores: np.ndarray, node_a: np.ndarray, node_b: np.ndarray) -> bool:
    """True when no two graph edges share the full (score, node_a, node_b) sort key."""
    keyed = np.stack([scores, node_a, node_b], axis=1)
    return len(np.unique(keyed, axis=0)) == len(keyed)


@pytest.mark.parametrize(
    ("variant", "min_exact_tie_fraction"),
    [("deterministic", 0.25), ("default", 0.25), ("link_checked", 0.05)],
)
def test_watershed_ordering_key_is_unique_on_a_real_graph(variant, min_exact_tie_fraction):
    """A total order is what makes the sort kind irrelevant; a sort on score alone is not.

    Also records the degeneracy that makes this necessary: a substantial fraction of
    consecutive score gaps are *exactly* zero, so score alone leaves much of the order
    undecided. The guard on that fraction keeps the totality assertion from being vacuous.

    `deterministic` and `default` (`dithered`) share a threshold because neither
    de-degenerates the score field — both sit around 31-45 % exact ties on `3.tif`,
    `min_distance=3` (measured). `link_checked` (boundary layer + junction protection +
    link-condition-checked offset exclusion, a former default) is lower because
    **the boundary layer and junction protection materially de-degenerate the score
    field**, which is a result, not a nuisance: the exactly tied fraction falls to 9.2 %,
    the same direction as the primary `< 1e-5` score-gap metric (0.489 -> 0.208). One edge in
    eleven is still an exact tie there, so the tie-break is still load-bearing even on that path.
    """
    getter = {
        "deterministic": get_deterministic_mesh_reconstruction_algorithm,
        "default": get_default_mesh_reconstruction_algorithm,
        "link_checked": get_link_checked_mesh_reconstruction_algorithm,
    }[variant]
    mask = load_image_or_skip(CASE_IMAGE)
    algo = getter(min_distance=MIN_DISTANCE, print_info=False)
    algo.construct_mesh_from_segmentation_mask(mask)
    graph = algo._tesselation_graph.to_networkx_graph()

    edge_data = np.array(list(graph.edges.data("score")))
    scores = -edge_data[:, 2]
    node_a = edge_data[:, 0].astype(np.int64)
    node_b = edge_data[:, 1].astype(np.int64)

    exact_ties = int(np.sum(np.diff(np.sort(scores)) == 0))
    assert exact_ties > min_exact_tie_fraction * len(scores), (
        f"{variant}: score field unexpectedly non-degenerate ({exact_ties}/{len(scores)}); the rest is vacuous"
    )
    assert _aggregation_ordering_is_total(scores, node_a, node_b)


def test_watershed_labelling_is_independent_of_the_edge_enumeration_order():
    """The tie-break is what makes the aggregation independent of how the graph was built.

    Re-running the aggregation on a graph holding the same edges inserted in reverse order
    is a direct proxy for "a different sorting kernel": with a total order the labels must
    be identical, whereas a sort on the score alone resolves its ~30 % exact ties by
    position in the edge list and therefore cannot be. The final assertion of this test
    demonstrates precisely that, by running the historical ordering on the same two graphs.
    """
    import networkx

    from dw3d.reconstruction_algorithm import _compute_seeds_idx_from_voxel_coords

    mask = load_image_or_skip(CASE_IMAGE)
    algo = get_default_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
    algo.construct_mesh_from_segmentation_mask(mask)
    graph = algo._tesselation_graph.to_networkx_graph()

    reversed_graph = networkx.Graph()
    reversed_graph.add_nodes_from(graph.nodes)
    reversed_graph.add_edges_from(reversed(list(graph.edges(data=True))))

    seeds_nodes = _compute_seeds_idx_from_voxel_coords(
        algo._tesselation_graph.compute_nodes_centroids(),
        algo._seeds_coords,
    )

    def initial_labels() -> np.ndarray:
        labels = np.zeros(len(graph.nodes), dtype=int) - 1
        for i, seed_node in enumerate(seeds_nodes):
            labels[seed_node] = algo._seeds_indices[i]
        return labels

    forward = _seeded_watershed_aggregation(graph, initial_labels())
    backward = _seeded_watershed_aggregation(reversed_graph, initial_labels())
    np.testing.assert_array_equal(forward, backward)
    # Against the pipeline's own labelling, taken from `seeded_watershed_map` rather than
    # from `algo._map_node_id_to_label`: mesh surgery mutates that map in place after the
    # watershed has run (30 of 25 997 tetrahedra on this case under the junction-protected
    # default), so it is no longer the watershed's output by the time the reconstruction
    # returns.
    _, pipeline_labels = seeded_watershed_map(graph, seeds_nodes, algo._seeds_indices)
    np.testing.assert_array_equal(forward, pipeline_labels)

    # And the discriminating half: the historical ordering does *not* have this property.
    assert not np.array_equal(
        _historical_aggregation(graph, initial_labels()),
        _historical_aggregation(reversed_graph, initial_labels()),
    )


def _historical_aggregation(nx_graph, map_node_id_to_label: np.ndarray) -> np.ndarray:
    """dw3d 0.3.6's `_seeded_watershed_aggregation`, verbatim apart from formatting.

    Kept here, in the test file, so that the claim "the old ordering was not a total
    order" is demonstrated against the actual old code rather than asserted about a
    commit. It is not importable from `dw3d`; nothing else may use it.
    """
    groups: dict[int, list[int]] = {}
    number_group = np.zeros(len(nx_graph.nodes), dtype=int) - 1
    num_group = 0

    scores = -np.array(list(nx_graph.edges.data("score")))[:, 2]
    args = np.argsort(scores)
    edges = list(nx_graph.edges)
    for arg in args:
        a, b = edges[arg]
        if map_node_id_to_label[a] != -1 and map_node_id_to_label[b] != -1:
            continue
        if map_node_id_to_label[a] != -1:
            map_node_id_to_label[groups.get(number_group[b], [b])] = map_node_id_to_label[a]
        elif map_node_id_to_label[b] != -1:
            map_node_id_to_label[groups.get(number_group[a], [a])] = map_node_id_to_label[b]
        elif number_group[a] != -1:
            if number_group[a] == number_group[b]:
                continue
            if number_group[b] != -1:
                old_b_group = groups.pop(number_group[b])
                groups[number_group[a]] += old_b_group
                number_group[old_b_group] = number_group[a]
            else:
                groups[number_group[a]].append(b)
                number_group[b] = number_group[a]
        elif number_group[b] != -1:
            groups[number_group[b]].append(a)
            number_group[a] = number_group[b]
        else:
            number_group[a] = num_group
            number_group[b] = num_group
            groups[num_group] = [a, b]
            num_group += 1
    return map_node_id_to_label


def test_committed_fingerprints_reproduce():
    """The strongest regression net in the repo: bit-level, on all 8 configurations.

    `benchmarks/baseline/fingerprints.json` is a committed artefact of the determinism
    fix, taken on numpy 2.5.1 / scipy 1.18.0 / scikit-image 0.26.0 and verified identical
    on numpy 1.26.1 / scipy 1.16.1 / scikit-image 0.22.0. Golden masters allow
    `rtol=1e-10`; this allows nothing. Any future change that moves a number will fail
    here first and must re-baseline deliberately, with
    `python -m benchmarks.run_fingerprints --out ...`.

    Wall times are stored alongside the digests and are deliberately not compared.
    """
    import json

    from tests.conftest import REPO_ROOT

    committed_path = REPO_ROOT / "benchmarks" / "baseline" / "fingerprints.json"
    if not committed_path.exists():
        pytest.skip(f"{committed_path} not present")
    committed = json.loads(committed_path.read_text())

    mask = load_image_or_skip(CASE_IMAGE)
    for min_distance in (3, 5):
        key = f"{CASE_IMAGE.removesuffix('.tif')}_md{min_distance}"
        # `run_fingerprints --repeats N` nests the runs; `--repeats 1` stores the run flat.
        # The junction-protection re-baseline was taken with `--repeats 2` (which is also
        # how the across-repeat reproducibility was checked), so accept both shapes.
        expected = committed["cases"][key]
        expected = expected["runs"][0] if "runs" in expected else expected
        actual = fingerprint_reconstruction(
            mask,
            lambda md=min_distance: get_default_mesh_reconstruction_algorithm(min_distance=md, print_info=False),
        )
        for field in FINGERPRINT_FIELDS:
            assert actual[field] == expected[field], f"{key}: {field}"
        assert actual["joint"] == expected["joint"], key


def test_dithered_and_link_checked_algorithms_produce_different_meshes():
    """Control: the meshing pipeline really did change, so the tests above are not vacuous.

    The default *is* `dithered` (see `get_default_algorithm`'s docstring), so the contrast
    this test now needs is against `get_link_checked_algorithm`, the opt-in "revised
    meshing" variant that carries the determinism fix through the offset-exclusion work's
    changes.
    """
    mask = load_image_or_skip(CASE_IMAGE)
    link_checked = fingerprint_reconstruction(
        mask,
        lambda: get_link_checked_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False),
    )
    dithered = fingerprint_reconstruction(
        mask,
        lambda: get_dithered_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False),
    )
    assert link_checked["joint"] != dithered["joint"]
    assert link_checked["n_points"] != dithered["n_points"]
