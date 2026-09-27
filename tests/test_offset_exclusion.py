"""Offset exclusion: the extracted surface must not use the boundary layer's offsets as vertices.

What these tests are for, in order of how much they would cost to get wrong:

1. **The merge target is exact, not a nearest-neighbour guess.** Every offset is
   `rint(p +- delta * n)` for a specific interface sample `p`, and offset exclusion merges
   it onto *that* `p`. The bookkeeping that carries `p` through de-duplication and the
   protecting-ball filter is the part that can silently drift, so it is checked against the
   geometry: the distance from each offset to its recorded parent must be consistent with
   `delta` and the rounding, and the parent must be an interface sample.
2. **Offset exclusion changes the surface and nothing else.** Same EDT, same points, same
   tesselation, same watershed labels, same cell volumes. If any of those moved, the
   before/after comparison of the offset exclusion would not be attributable to the
   extraction.
3. **The surface stays a closed multi-material mesh.** Merging removes 30-35 % of surface
   vertices and drops the triangles that collapse; that is where topological damage would
   show up. Asserted directly (no boundary edges, no label conflicts), not inferred.
4. **No offset survives in the surface.** The point of the exercise.
"""

from functools import partial

import numpy as np
import pytest

from dw3d import get_junction_protected_mesh_reconstruction_algorithm
from dw3d.mesh_utilities import exclude_offsets_from_surface
from dw3d.points_on_edt import (
    FAMILY_INTERFACE_MINIMUM,
    FAMILY_INTERFACE_OFFSET,
    FAMILY_JUNCTION_SAMPLE,
    OFFSET_FAMILIES,
    _dedup_against,
    _dedup_against_index,
    junction_protected_families,
    peak_local_points_junction_protected,
)
from tests.conftest import load_image_or_skip

CASE_IMAGE = "3.tif"
MIN_DISTANCE = 3
# `get_junction_protected_algorithm`'s defaults, which are what the offset-included control
# and the offset-excluded variant both use. Kept here so a change to them makes these tests
# fail rather than quietly measure a different configuration.
SHELL_COARSENING = 3
JUNCTION_BOUNDARY_LAYER = False


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
    """Boundary layer + junction protection (the control) and the offset-excluded variant, on one mask, built once."""
    out = {}
    for name, getter in (
        ("offset_included", get_junction_protected_mesh_reconstruction_algorithm),
        (
            "offset_excluded",
            partial(get_junction_protected_mesh_reconstruction_algorithm, exclude_offsets_from_surface=True),
        ),
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


@pytest.fixture(scope="module")
def families(mask, reconstructions):
    algo = reconstructions["offset_included"][0]
    return junction_protected_families(
        mask,
        algo._edt_image,
        MIN_DISTANCE,
        shell_coarsening=SHELL_COARSENING,
        junction_boundary_layer=JUNCTION_BOUNDARY_LAYER,
    )


# --------------------------------------------------------------------------------------
# 1. The parent bookkeeping
# --------------------------------------------------------------------------------------


def test_dedup_index_and_dedup_agree():
    """`_dedup_against_index` must select exactly the rows `_dedup_against` returns.

    The parent array is carried through de-duplication by indexing with the former while the
    coordinates go through the latter, so any divergence would silently mis-assign parents.
    """
    rng = np.random.default_rng(20260731)
    points = rng.integers(0, 6, size=(400, 3))
    forbidden = (rng.integers(0, 6, size=(30, 3)),)
    np.testing.assert_array_equal(points[_dedup_against_index(points, forbidden)], _dedup_against(points, forbidden))


def test_dedup_index_is_stable_on_an_empty_input():
    empty = np.zeros((0, 3), dtype=np.int64)
    assert len(_dedup_against_index(empty, ())) == 0
    assert len(_dedup_against(empty, ())) == 0


def test_every_offset_has_a_parent_and_the_counts_line_up(families):
    parents = families["offset_parents"]
    assert len(parents) == len(families["offsets"])
    assert len(parents) > 0, "no offsets emitted: the rest of this file would be vacuous"
    assert parents.min() >= 0
    assert parents.max() < len(families["minima"])


def test_each_offset_sits_delta_from_its_recorded_parent(families):
    """The recorded parent must be the sample the offset actually displaces.

    `offset = rint(parent +- delta * n_hat)` with `|n_hat| = 1`, so the distance is `delta`
    up to the rounding of a 3-vector, i.e. within `sqrt(3)/2` of `delta`. This is a check on
    the *bookkeeping*, not on the construction: a mis-carried parent index would land the
    offset next to an unrelated sample and break this bound.
    """
    offsets = np.asarray(families["offsets"], dtype=np.float64)
    parents = np.asarray(families["minima"], dtype=np.float64)[families["offset_parents"]]
    distance = np.linalg.norm(offsets - parents, axis=1)
    delta = float(MIN_DISTANCE)
    tolerance = np.sqrt(3.0) / 2.0
    assert distance.max() <= delta + tolerance, f"worst {distance.max():.3f} > {delta + tolerance:.3f}"
    assert distance.min() >= delta - tolerance


def test_the_merge_target_is_identity_off_the_offset_families(reconstructions):
    algo = reconstructions["offset_excluded"][0]
    metadata = algo._point_metadata
    family, target = metadata["family"], metadata["surface_merge_target"]
    scaffolding = np.isin(family, OFFSET_FAMILIES)
    identity = np.arange(len(target))
    np.testing.assert_array_equal(target[~scaffolding], identity[~scaffolding])
    assert np.all(target[scaffolding] != identity[scaffolding])
    # and every target is itself a surface-eligible point, so the merge cannot cascade
    assert not np.isin(family[target[scaffolding]], OFFSET_FAMILIES).any()


def test_interface_offsets_merge_onto_interface_minima(reconstructions):
    metadata = reconstructions["offset_excluded"][0]._point_metadata
    family, target = metadata["family"], metadata["surface_merge_target"]
    offsets = family == FAMILY_INTERFACE_OFFSET
    assert offsets.any()
    assert np.all(family[target[offsets]] == FAMILY_INTERFACE_MINIMUM)


# --------------------------------------------------------------------------------------
# 2. Offset exclusion changes the surface and nothing else
# --------------------------------------------------------------------------------------


def test_asking_for_the_metadata_does_not_change_the_point_set(mask, reconstructions):
    """The metadata is a *declaration*, so emitting it must not move a single point.

    This is what lets the ε-by-family decomposition be measured on the offset-included
    configuration with the same switch offset exclusion runs with
    (`benchmarks/angle_error_budget.epsilon_by_point_family`).
    """
    edt = reconstructions["offset_included"][0]._edt_image
    plain = peak_local_points_junction_protected(
        mask,
        edt,
        MIN_DISTANCE,
        shell_coarsening=SHELL_COARSENING,
        junction_boundary_layer=JUNCTION_BOUNDARY_LAYER,
    )
    annotated = peak_local_points_junction_protected(
        mask,
        edt,
        MIN_DISTANCE,
        shell_coarsening=SHELL_COARSENING,
        junction_boundary_layer=JUNCTION_BOUNDARY_LAYER,
        exclude_offsets_from_surface=True,
    )
    assert len(plain) == 3
    assert len(annotated) == 4
    for i in range(3):
        np.testing.assert_array_equal(plain[i], annotated[i])


def test_the_tesselation_and_the_labelling_are_untouched(reconstructions):
    """Same tetrahedra and same watershed labels: only the extraction differs."""
    offset_included_algo = reconstructions["offset_included"][0]
    offset_excluded_algo = reconstructions["offset_excluded"][0]
    np.testing.assert_array_equal(
        offset_included_algo._tesselation_graph.vertices,
        offset_excluded_algo._tesselation_graph.vertices,
    )
    np.testing.assert_array_equal(
        offset_included_algo._tesselation_graph.tetrahedrons,
        offset_excluded_algo._tesselation_graph.tetrahedrons,
    )
    np.testing.assert_array_equal(
        offset_included_algo._map_node_id_to_label,
        offset_excluded_algo._map_node_id_to_label,
    )


def test_cell_volumes_are_unchanged(reconstructions):
    """Cell volume comes from the labelled tetrahedra, which offset exclusion does not touch.

    A regression here would mean the merge had reached back into the tesselation.
    """
    from dw3d_benchmarks import metrics as m

    volumes = [
        m.cell_volumes_from_tetrahedra(algo._tesselation_graph, algo._map_label_to_nodes_ids)
        for algo, *_ in (reconstructions["offset_included"], reconstructions["offset_excluded"])
    ]
    assert volumes[0].keys() == volumes[1].keys()
    for key in volumes[0]:
        assert volumes[0][key] == pytest.approx(volumes[1][key], rel=1e-12)


def test_no_point_moves(reconstructions):
    """Offset exclusion is a re-indexing, not a displacement: every surviving vertex keeps its coordinates."""
    offset_included_algo = reconstructions["offset_included"][0]
    offset_excluded_algo, offset_excluded_points, *_ = reconstructions["offset_excluded"]
    surviving = offset_excluded_algo._surface_point_ids
    np.testing.assert_array_equal(offset_excluded_points, offset_included_algo._tesselation_graph.vertices[surviving])
    # Every surviving vertex was either already in the offset-included surface or is the
    # parent of one of its offsets -- a parent that no surface triangle referenced before
    # the merge becomes a surface vertex through it, which is why this is not a plain subset.
    target = offset_excluded_algo._point_metadata["surface_merge_target"]
    admissible = set(offset_included_algo._surface_point_ids.tolist()) | set(
        target[offset_included_algo._surface_point_ids].tolist(),
    )
    assert set(surviving.tolist()) <= admissible


# --------------------------------------------------------------------------------------
# 3. The surface stays a closed multi-material mesh
# --------------------------------------------------------------------------------------


def _edge_valences(triangles):
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return counts


def test_the_merged_surface_has_no_boundary_edge(reconstructions):
    """A valence-1 edge is a hole. The default has none and offset exclusion must not open any."""
    for name in ("offset_included", "offset_excluded"):
        counts = _edge_valences(np.asarray(reconstructions[name][2], dtype=np.int64))
        assert int((counts == 1).sum()) == 0, name


def test_the_merge_reports_no_label_conflict(reconstructions):
    """Two coincident triangles with different materials must be reported, never repaired.

    That would mean the merge had fused two different interfaces, so it must not be able to
    pass silently.
    """
    info = reconstructions["offset_excluded"][0]._surface_exclusion_info
    assert info is not None
    assert info["n_duplicate_conflicts"] == 0, info


def test_no_degenerate_triangle_survives(reconstructions):
    points, triangles = reconstructions["offset_excluded"][1], reconstructions["offset_excluded"][2]
    corners = points[np.asarray(triangles, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    assert areas.min() > 0.0
    # every triangle references three distinct vertices
    assert np.all(triangles[:, 0] != triangles[:, 1])
    assert np.all(triangles[:, 1] != triangles[:, 2])
    assert np.all(triangles[:, 0] != triangles[:, 2])


def test_triangle_shape_quality_does_not_regress(reconstructions):
    """Removing the flaps must not leave worse-shaped triangles behind.

    Measured, not assumed: the merge deletes the flap triangles outright rather than
    flattening them, so the surviving population is the one that was already on the
    interface. The minimum angle over the whole mesh must not fall.
    """

    def min_angles(points, triangles):
        corners = points[np.asarray(triangles, dtype=np.int64)]
        sides = np.sort(
            np.stack([np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)], axis=1),
            axis=1,
        )
        shortest, mid, longest = sides[:, 0], sides[:, 1], sides[:, 2]
        cosine = (mid**2 + longest**2 - shortest**2) / (2 * mid * longest)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    before = min_angles(reconstructions["offset_included"][1], reconstructions["offset_included"][2])
    after = min_angles(reconstructions["offset_excluded"][1], reconstructions["offset_excluded"][2])
    assert after.min() >= before.min() - 1e-9
    assert np.median(after) >= np.median(before)


# --------------------------------------------------------------------------------------
# 4. No offset survives, and the construction's own arithmetic
# --------------------------------------------------------------------------------------


def test_only_the_guard_refused_offsets_remain_in_the_surface(reconstructions):
    """The offsets are gone except the handful the duplicate-face guard refused to merge.

    Pinned as an identity rather than a threshold: the number of offsets left in the surface
    must equal the number of merges the guard reports refusing. If those two ever disagreed,
    either the guard is refusing merges that were not needed or offsets are surviving for some
    other reason, and both would be invisible in a "less than 10 %" style assertion.
    """
    algo = reconstructions["offset_excluded"][0]
    family_of_surface_vertex = algo._point_metadata["family"][algo._surface_point_ids]
    remaining = int(np.isin(family_of_surface_vertex, OFFSET_FAMILIES).sum())
    assert remaining == algo._surface_exclusion_info["n_merges_refused"]
    # apart from those, the surface is drawn from the families that sit *on* the geometry
    assert set(np.unique(family_of_surface_vertex).tolist()) <= {
        FAMILY_INTERFACE_MINIMUM,
        FAMILY_JUNCTION_SAMPLE,
        FAMILY_INTERFACE_OFFSET,
    }


def test_the_guard_leaves_only_a_small_minority_of_the_offsets(reconstructions):
    """The guard must be a correction, not the rule: it may keep only a few per cent.

    Measured 38 of 650 on `3.tif`. The bound is deliberately loose — this guards against the
    guard quietly swallowing the exclusion, not against a small drift.
    """
    algo = reconstructions["offset_excluded"][0]
    info = algo._surface_exclusion_info
    kept = info["n_merges_refused"]
    excluded = info["n_offset_vertices_removed"]
    assert kept < 0.2 * (kept + excluded), f"guard kept {kept} of {kept + excluded} offsets"


def test_the_unguarded_construction_removes_every_offset(mask):
    """Control on the guard: with it off, no offset survives at all.

    This is what makes the test above a statement about the guard rather than about the merge.
    """
    algo = get_junction_protected_mesh_reconstruction_algorithm(
        min_distance=MIN_DISTANCE,
        print_info=False,
        exclude_offsets_from_surface=True,
        guard_duplicate_faces=False,
    )
    algo.construct_mesh_from_segmentation_mask(mask)
    family_of_surface_vertex = algo._point_metadata["family"][algo._surface_point_ids]
    assert not np.isin(family_of_surface_vertex, OFFSET_FAMILIES).any()
    assert algo._surface_exclusion_info["n_merges_refused"] == 0


def test_offset_included_does_use_offsets_as_surface_vertices(reconstructions):
    """The control: without offset exclusion a third of the surface vertices are offsets.

    Without this the test above could pass because the offsets were never in the surface to
    begin with, and offset exclusion would be measuring nothing.
    """
    offset_included_algo = reconstructions["offset_included"][0]
    offset_excluded_algo = reconstructions["offset_excluded"][0]
    # The offset-included placer emits no metadata, so borrow the offset-excluded one's
    # (same points, see the test above)
    family_of_surface_vertex = offset_excluded_algo._point_metadata["family"][offset_included_algo._surface_point_ids]
    share = np.isin(family_of_surface_vertex, OFFSET_FAMILIES).mean()
    assert share > 0.2, f"only {100 * share:.1f} % of the offset-included surface vertices are offsets"


def test_the_reported_vertex_counts_are_internally_consistent(reconstructions):
    """The report's own arithmetic, and the one subtlety in it.

    The net vertex count does *not* fall by `n_offset_vertices_removed`: a parent that no
    surface triangle referenced before the merge becomes a surface vertex through it. So the
    identity to check is `before - after = removed - newly_arrived`, and both sides are
    computed here from the meshes rather than from the report.
    """
    offset_included_algo = reconstructions["offset_included"][0]
    offset_excluded_algo = reconstructions["offset_excluded"][0]
    info = offset_excluded_algo._surface_exclusion_info
    before = set(offset_included_algo._surface_point_ids.tolist())
    after = set(offset_excluded_algo._surface_point_ids.tolist())
    assert info["n_surface_vertices_before"] == len(before)
    assert info["n_surface_vertices_after"] == len(after)
    assert info["n_offset_vertices_removed"] == len(before - after)
    assert len(before) - len(after) == len(before - after) - len(after - before)


def test_the_flaps_carried_area_that_no_interface_has(reconstructions):
    """The surface area must fall, and the drop is the wrinkle coming out.

    Not a regression: a flap reaching `delta` voxels off the surface and back adds area the
    interface does not have. Interface areas measured against the ground truth are what
    decides whether the drop is an improvement; that comparison is made against ground truth on
    the benchmark cases, not on these images (see `get_link_checked_algorithm`'s docstring).
    """
    info = reconstructions["offset_excluded"][0]._surface_exclusion_info
    assert info["area_after"] < info["area_before"]


def test_exclusion_is_a_no_op_when_nothing_is_scaffolding():
    """`surface_merge_target = identity` must leave the mesh exactly as it was."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.uint)
    kept, kept_labels, info = exclude_offsets_from_surface(points, triangles, labels, np.arange(4))
    np.testing.assert_array_equal(kept, triangles)
    np.testing.assert_array_equal(kept_labels, labels)
    assert info["n_triangles_collapsed"] == 0
    assert info["n_offset_vertices_removed"] == 0
    assert info["n_merges_refused"] == 0


def test_exclusion_drops_exactly_the_collapsing_triangles():
    """A hand-built case with a known answer: vertex 3 is scaffolding above vertex 0.

    Triangles containing both 3 and 0 must vanish; the rest must be re-indexed onto 0.
    """
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 3], [1, 1, 0]])
    triangles = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2], [1, 3, 4]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 2]], dtype=np.uint)
    target = np.array([0, 1, 2, 0, 4])
    kept, kept_labels, info = exclude_offsets_from_surface(
        points,
        triangles,
        labels,
        target,
        guard_duplicate_faces=False,
    )
    assert info["n_triangles_collapsed"] == 1  # [0, 1, 3] -> [0, 1, 0]
    # [1, 2, 3] -> [1, 2, 0], which duplicates [0, 1, 2] with the same label pair: one survives
    assert info["n_duplicate_triples"] == 1
    assert info["n_duplicates_dropped"] == 1
    assert info["n_duplicate_conflicts"] == 0
    assert len(kept) == 2
    assert 3 not in set(kept.reshape(-1).tolist())
    assert sorted(tuple(sorted(t)) for t in kept.tolist()) == [(0, 1, 2), (0, 1, 4)]
    assert len(kept_labels) == 2


def test_the_guard_refuses_the_merge_that_would_duplicate_a_face():
    """Same hand-built case with the guard on: vertex 3 stays, and no face is duplicated.

    The guard's whole contract, on a case whose answer can be read off by hand.
    """
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 3], [1, 1, 0]])
    triangles = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2], [1, 3, 4]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 2]], dtype=np.uint)
    target = np.array([0, 1, 2, 0, 4])
    kept, _kept_labels, info = exclude_offsets_from_surface(points, triangles, labels, target)
    assert info["n_merges_refused"] == 1
    assert info["n_guard_rounds"] == 1
    assert info["n_duplicate_triples"] == 0
    assert info["n_triangles_collapsed"] == 0
    np.testing.assert_array_equal(np.asarray(kept, dtype=np.int64), np.asarray(triangles, dtype=np.int64))


def test_a_label_conflict_is_reported_and_not_silently_merged():
    """Two triangles on one triple with different materials is a fused interface: report it."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 3]])
    triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint)
    labels = np.array([[0, 1], [0, 2]], dtype=np.uint)
    target = np.array([0, 1, 2, 0])
    kept, _kept_labels, info = exclude_offsets_from_surface(
        points,
        triangles,
        labels,
        target,
        guard_duplicate_faces=False,
    )
    assert info["n_duplicate_conflicts"] == 1
    assert len(kept) == 1  # still deduplicated, but the conflict is on the record
