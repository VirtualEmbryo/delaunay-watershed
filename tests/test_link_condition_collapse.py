"""The link-condition-checked collapse: removing the offsets must not change the surface's topology.

The unconditional weld welded each boundary-layer offset onto its parent unconditionally.
Welding two vertices makes formerly distinct edges the same edge, so their triangle
incidences *add*, and over the 51-case benchmark that took valence->=4 edges 115 -> 184 and
abnormal non-manifold edges 7 -> 76. The link-condition-checked
collapse keeps the weld's candidate merges and its geometry and only changes the test that
decides which candidates are taken: `dw3d.mesh_utilities._refuse_by_link_condition`.

What these tests are for:

1. **Each of the four refusal tests fires on a case whose answer is readable by hand**, and
   for test 4 — the one the classical link condition does not give — the same hand case is
   run through the `"weld"` rule as a **control**, so the test proves the configuration is
   genuinely dangerous rather than merely being refused. Without that control the whole file
   could pass while the predicate refused everything for no reason.
2. **The invariant that matters is asserted directly on a real mask**: no edge valence in
   the link-condition-checked surface exceeds what the unexcluded offset-included surface
   already had, so holes, valence->=4 edges and abnormal non-manifold edges are all bounded
   by the offset-included surface's. That is the acceptance criterion the unconditional
   weld failed.
3. **The refusal bookkeeping is internally consistent** — the per-reason counts must add up to
   the reported total. An earlier version of this measurement had a check that passed
   *vacuously* because both sides of a comparison were `None`; every count here is asserted against
   something computed independently of it.
4. **The link-condition-checked collapse is not the weld with everything refused.** The
   share of offsets actually removed is pinned from below, and the offset-included
   configuration is shown to have those offsets in its surface in the first place.
"""

import numpy as np
import pytest

from dw3d import (
    get_junction_protected_mesh_reconstruction_algorithm,
    get_link_checked_mesh_reconstruction_algorithm,
)
from dw3d.mesh_surgery import _find_abnormal_non_manifold_edges
from dw3d.mesh_utilities import LINK_CONDITION, WELD, exclude_offsets_from_surface
from dw3d.points_on_edt import OFFSET_FAMILIES
from tests.conftest import load_image_or_skip

CASE_IMAGE = "3.tif"
MIN_DISTANCE = 3


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def edge_valences(triangles) -> dict[tuple[int, int], int]:
    """Number of triangles on each undirected edge."""
    triangles = np.asarray(triangles, dtype=np.int64)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in triangles.tolist():
        for u, v in ((a, b), (b, c), (a, c)):
            key = (min(u, v), max(u, v))
            counts[key] = counts.get(key, 0) + 1
    return counts


def valence_histogram(triangles) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for valence in edge_valences(triangles).values():
        histogram[valence] = histogram.get(valence, 0) + 1
    return histogram


def collapse(points, triangles, labels, target, rule=LINK_CONDITION, **kwargs: bool):
    return exclude_offsets_from_surface(
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.uint),
        np.asarray(labels, dtype=np.uint),
        np.asarray(target, dtype=np.int64),
        collapse_rule=rule,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# 1. Each refusal test, on a hand-built case
# --------------------------------------------------------------------------------------

# The counterexample in `_refuse_by_link_condition`'s docstring, and the reason test 4 exists.
# Vertex 0 is the offset, vertex 1 its parent, vertex 2 the apex they share. Both (2, 0) and
# (2, 1) are triple lines carrying materials {1, 2, 3}; the triangle (0, 1, 2) joins them.
# The classical link condition is satisfied — and welding still fuses the two triple lines
# into one valence-4 edge that only three materials meet along, i.e. an abnormal edge.
PINCH_POINTS = np.array(
    [[0.0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 1], [0, 2, -1], [2, 2, 1], [2, 2, -1]],
)
PINCH_TRIANGLES = np.array([[0, 1, 2], [2, 0, 3], [2, 0, 4], [2, 1, 5], [2, 1, 6]], dtype=np.uint)
PINCH_LABELS = np.array([[1, 2], [2, 3], [3, 1], [2, 3], [3, 1]], dtype=np.uint)
PINCH_TARGET = np.array([1, 1, 2, 3, 4, 5, 6])


def test_the_weld_really_does_fuse_two_triple_lines_into_an_abnormal_edge():
    """Control for the test below: the configuration is genuinely dangerous.

    Both (0, 2) and (1, 2) are valence-3 triple lines before. Welding 0 onto 1 merges them
    into a single valence-4 edge with only 3 materials on it, which is exactly what
    `_find_abnormal_non_manifold_edges` — and the unconditional weld's 7 -> 76 regression —
    counts.
    """
    assert edge_valences(PINCH_TRIANGLES)[(0, 2)] == 3
    assert edge_valences(PINCH_TRIANGLES)[(1, 2)] == 3
    assert len(_find_abnormal_non_manifold_edges(PINCH_POINTS, np.asarray(PINCH_TRIANGLES), PINCH_LABELS)) == 0

    kept, kept_labels, info = collapse(PINCH_POINTS, PINCH_TRIANGLES, PINCH_LABELS, PINCH_TARGET, rule=WELD)
    assert info["n_merges_refused"] == 0
    assert edge_valences(kept)[(1, 2)] == 4, "the weld was expected to fuse the two triple lines"
    abnormal = _find_abnormal_non_manifold_edges(PINCH_POINTS, np.asarray(kept, dtype=np.int64), kept_labels)
    assert len(abnormal) == 1


def test_the_link_condition_refuses_that_merge_on_the_valence_test():
    """The same case under the link-condition-checked collapse: refused, by test 4, and nothing moves.

    The classical link condition passes here (there is exactly one shared neighbour, 2, and
    (0, 1, 2) is a triangle; no two triangles would coincide). Only the explicit valence
    bookkeeping catches it, which is why the predicate carries a fourth test.
    """
    kept, kept_labels, info = collapse(PINCH_POINTS, PINCH_TRIANGLES, PINCH_LABELS, PINCH_TARGET)
    assert info["n_refused_valence"] == 1
    assert info["n_refused_no_edge"] == 0
    assert info["n_refused_link_vertices"] == 0
    assert info["n_refused_link_edges"] == 0
    assert info["n_merges_refused"] == 1
    np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(PINCH_TRIANGLES, dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(kept_labels), np.asarray(PINCH_LABELS))
    assert len(_find_abnormal_non_manifold_edges(PINCH_POINTS, np.asarray(kept, dtype=np.int64), kept_labels)) == 0


def test_a_merge_across_a_non_edge_is_refused():
    """Two disjoint triangles: merging a vertex of one onto a vertex of the other is a pinch.

    It is not an edge contraction at all — there is no edge to contract — so it is refused
    before any link test is reached.
    """
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 2]], dtype=np.uint)
    target = np.array([3, 1, 2, 3, 4, 5])  # vertex 0 "merges" onto vertex 3
    kept, _labels, info = collapse(points, triangles, labels, target)
    assert info["n_refused_no_edge"] == 1
    assert info["n_merges_refused"] == 1
    np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(triangles, dtype=np.int64))


def test_the_vertex_part_of_the_link_condition_refuses_a_shared_neighbour_with_no_triangle():
    """The unconditional weld's duplicate-face hand case, refused here for a stronger reason.

    Vertices 0 and 3 share the neighbours 1 *and* 2, but only (0, 1, 3) is a triangle — there
    is no (0, 2, 3). Contracting would therefore make the two distinct edges (0, 2) and
    (2, 3) into one, which is the vertex part of the link condition failing. The
    unconditional weld needed a special duplicate-face guard to notice this case; here it
    falls out of the general test.
    """
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 3], [1, 1, 0]])
    triangles = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2], [1, 3, 4]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 2]], dtype=np.uint)
    target = np.array([0, 1, 2, 0, 4])
    kept, _labels, info = collapse(points, triangles, labels, target)
    assert info["n_refused_link_vertices"] == 1
    assert info["n_duplicate_triples"] == 0
    assert info["n_triangles_collapsed"] == 0
    np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(triangles, dtype=np.int64))


def test_the_edge_part_of_the_link_condition_refuses_a_would_be_duplicate_face():
    """A closed tetrahedron: contracting any edge would fold two faces onto each other.

    (0, 2, 3) and (1, 2, 3) are both triangles, so (2, 3) is an edge of both links and the
    contraction would make them the same face. This is the test that subsumes the
    unconditional weld's duplicate-face guard.
    """
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.uint)
    target = np.array([1, 1, 2, 3])
    kept, _labels, info = collapse(points, triangles, labels, target)
    assert info["n_refused_link_edges"] == 1
    np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(triangles, dtype=np.int64))


def test_a_safe_collapse_is_accepted_and_preserves_every_edge_valence():
    """The positive control: an octahedron, contracting the apex onto an equator vertex.

    Every edge of an octahedron is manifold, the two shared neighbours are both apexes of the
    contracted edge, and no two faces would coincide — all four tests pass. The result is
    still a closed surface with every edge at valence 2, and two faces have gone.
    """
    points = np.array(
        [[0.0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    )
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]],
        dtype=np.uint,
    )
    labels = np.zeros((8, 2), dtype=np.uint)
    labels[:, 1] = 1
    target = np.array([1, 1, 2, 3, 4, 5])
    kept, _labels, info = collapse(points, triangles, labels, target)
    assert info["n_merges_refused"] == 0
    assert info["n_triangles_collapsed"] == 2
    assert len(kept) == 6
    assert 0 not in set(np.asarray(kept, dtype=np.int64).reshape(-1).tolist())
    assert valence_histogram(kept) == {2: 9}, "the contracted surface must stay closed and manifold"


def test_the_guard_switch_is_ignored_under_the_link_condition():
    """`guard_duplicate_faces` cannot change a `link_condition` result, and says so."""
    for guard in (True, False):
        kept, _labels, info = collapse(
            PINCH_POINTS,
            PINCH_TRIANGLES,
            PINCH_LABELS,
            PINCH_TARGET,
            guard_duplicate_faces=guard,
        )
        assert info["guard_duplicate_faces"] is False
        assert info["n_merges_refused"] == 1
        np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(PINCH_TRIANGLES, dtype=np.int64))


def test_an_unknown_collapse_rule_is_rejected():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2]], dtype=np.uint)
    labels = np.array([[0, 1]], dtype=np.uint)
    with pytest.raises(ValueError, match="collapse_rule"):
        collapse(points, triangles, labels, np.arange(3), rule="contract_everything")


def test_a_chained_merge_target_is_rejected():
    """Both rules apply the accepted merges as one re-indexing, which needs a flat target."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2]], dtype=np.uint)
    labels = np.array([[0, 1]], dtype=np.uint)
    with pytest.raises(ValueError, match="chained"):
        collapse(points, triangles, labels, np.array([1, 2, 2]))


def test_the_refusal_reasons_add_up_to_the_reported_total():
    """Bookkeeping check: the per-reason counts must reconstruct `n_merges_refused`.

    Not a tautology — the reasons are counted in the sweep's last round and the total is the
    length of the pending list at the end, computed separately.
    """
    _kept, _labels, info = collapse(PINCH_POINTS, PINCH_TRIANGLES, PINCH_LABELS, PINCH_TARGET)
    reasons = ("no_edge", "link_vertices", "link_edges", "valence")
    by_reason = sum(info[f"n_refused_{reason}"] for reason in reasons)
    assert by_reason == info["n_merges_refused"]


# --------------------------------------------------------------------------------------
# 2-4. The invariant, on a real mask
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mask():
    return load_image_or_skip(CASE_IMAGE)


# The control these tests compare against is the **unexcluded** surface, which since the
# link-checked default flip is no longer `get_default_...` -- the default *is* the excluded
# one. It is `get_offset_included_...`, the boundary layer + junction protection
# configuration that was a former default. Pointing it at the default would make every
# "offset exclusion removes the offsets" assertion below compare a mesh against itself and
# pass vacuously.
@pytest.fixture(scope="module")
def reconstructions(mask):
    """Boundary layer + junction protection (the control) and the link-checked variant, on one mask, built once."""
    out = {}
    for name, getter in (
        ("offset_included", get_junction_protected_mesh_reconstruction_algorithm),
        ("linkcheck", get_link_checked_mesh_reconstruction_algorithm),
    ):
        algo = getter(min_distance=MIN_DISTANCE, print_info=False)
        # This test specifies the reconstruction *before* any post-process: which tesselation
        # points become mesh vertices, and where. Junction relocation (on by default since 0.5.0)
        # moves trijunction vertices by design, so it is switched off explicitly here; the
        # relocated path is covered by tests/test_default_junction_relocation.py and the golden masters.
        algo.relocate_junctions = False
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
        out[name] = (algo, points, triangles, labels)
    return out


def test_no_edge_valence_exceeds_what_offset_included_already_had(reconstructions):
    """The acceptance criterion, asserted directly rather than through a summary statistic.

    The link-condition-checked collapse's contract is that no edge valence can increase, so
    for every valence `k >= 3` the link-checked surface must carry no more edges at that
    valence than the offset-included surface's does. Both meshes are re-indexed, so this
    compares the *histograms*, not edge identities.
    """
    before = valence_histogram(reconstructions["offset_included"][2])
    after = valence_histogram(reconstructions["linkcheck"][2])
    assert before, "the offset-included histogram is empty: this test would be vacuous"
    assert after
    for valence, count in after.items():
        if valence != 2:
            assert count <= before.get(valence, 0), f"valence {valence}: {count} > {before.get(valence, 0)}"
    assert sum(c for v, c in after.items() if v >= 4) <= sum(c for v, c in before.items() if v >= 4)
    assert after.get(1, 0) <= before.get(1, 0)


def test_the_abnormal_non_manifold_edge_count_does_not_rise(reconstructions):
    """The other metric the unconditional weld regressed, measured with the library's own detector."""
    counts = {}
    for name in ("offset_included", "linkcheck"):
        _algo, points, triangles, labels = reconstructions[name]
        counts[name] = len(
            _find_abnormal_non_manifold_edges(points, np.asarray(triangles, dtype=np.int64), np.asarray(labels)),
        )
    assert counts["linkcheck"] <= counts["offset_included"], counts


def test_the_merged_surface_has_no_hole_and_no_label_conflict(reconstructions):
    algo, _points, triangles, _labels = reconstructions["linkcheck"]
    assert valence_histogram(triangles).get(1, 0) == 0
    info = algo._surface_exclusion_info
    assert info is not None
    assert info["collapse_rule"] == LINK_CONDITION
    assert info["n_duplicate_conflicts"] == 0
    assert info["n_duplicate_triples"] == 0


def test_most_of_the_offsets_are_still_removed(reconstructions):
    """The link-condition-checked collapse is the weld minus the dangerous merges, not a weld refusing everything.

    Pinned from below at 80 % (measured: 94 %), and against the *default's* offset count so
    that the denominator cannot shrink to make the ratio look good.
    """
    algo = reconstructions["linkcheck"][0]
    info = algo._surface_exclusion_info
    removed, refused = info["n_offset_vertices_removed"], info["n_merges_refused"]
    assert removed + refused > 100, "too few offsets to make this meaningful"
    assert removed >= 0.8 * (removed + refused), f"only {removed} of {removed + refused} offsets removed"

    remaining = np.isin(algo._point_metadata["family"][algo._surface_point_ids], OFFSET_FAMILIES).sum()
    assert int(remaining) == refused


def test_offset_included_does_use_offsets_as_surface_vertices(reconstructions):
    """Control: without the exclusion a third of the surface vertices are offsets."""
    offset_included_algo = reconstructions["offset_included"][0]
    metadata = reconstructions["linkcheck"][0]._point_metadata
    share = np.isin(metadata["family"][offset_included_algo._surface_point_ids], OFFSET_FAMILIES).mean()
    assert share > 0.2, f"only {100 * share:.1f} % of the offset-included surface vertices are offsets"


def test_the_tesselation_labelling_and_geometry_are_untouched(reconstructions):
    """The link-condition-checked collapse changes the extraction and nothing upstream of it."""
    offset_included_algo = reconstructions["offset_included"][0]
    link_checked_algo, link_checked_points, *_ = reconstructions["linkcheck"]
    np.testing.assert_array_equal(
        offset_included_algo._tesselation_graph.vertices,
        link_checked_algo._tesselation_graph.vertices,
    )
    np.testing.assert_array_equal(
        offset_included_algo._tesselation_graph.tetrahedrons,
        link_checked_algo._tesselation_graph.tetrahedrons,
    )
    np.testing.assert_array_equal(
        offset_included_algo._map_node_id_to_label,
        link_checked_algo._map_node_id_to_label,
    )
    # no point moves: every surviving vertex keeps the coordinates the tesselation gave it
    np.testing.assert_array_equal(
        link_checked_points,
        offset_included_algo._tesselation_graph.vertices[link_checked_algo._surface_point_ids],
    )


def test_no_degenerate_triangle_survives(reconstructions):
    points, triangles = reconstructions["linkcheck"][1], reconstructions["linkcheck"][2]
    corners = points[np.asarray(triangles, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    assert areas.min() > 0.0
    assert np.all(triangles[:, 0] != triangles[:, 1])
    assert np.all(triangles[:, 1] != triangles[:, 2])
    assert np.all(triangles[:, 0] != triangles[:, 2])


def test_the_collapse_is_deterministic(mask):
    """No ordering choice is left to a set iteration: two runs must be bit-identical."""
    meshes = []
    for _ in range(2):
        algo = get_link_checked_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
        meshes.append(algo.construct_mesh_from_segmentation_mask(mask))
    for first, second in zip(meshes[0], meshes[1], strict=True):
        np.testing.assert_array_equal(first, second)
