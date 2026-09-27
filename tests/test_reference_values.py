"""Reproduce the dw3d 0.3.6 reference measurement on `data/Images/3.tif`, min_distance=3.

These are the baseline numbers every later meshing change was measured against.

**Which algorithm these numbers describe.** They were measured on dw3d 0.3.6, whose point
placement resolved EDT plateaus with a random dither. The determinism fix replaced that
with a deterministic geometric rule and re-baselined the golden masters, so those
figures are now the reference values of `get_dithered_mesh_reconstruction_algorithm`, not
of the default. They are checked against that factory here, which is what keeps the
baseline measurement — the quantitative basis of all the later meshing work — alive and
falsifiable rather than quietly superseded.

The **deterministic baseline that the boundary-layer, regular-triangulation and
junction-protection work are judged against** is asserted in the second section, against
`get_deterministic_mesh_reconstruction_algorithm`. The boundary-layer work targets
all-surface tets < 15 %, slivers < 7 % and score gaps below 1e-3 under 40 %, quoting the
v0.3 reference figures of 58.0 %, 14.1 % and 80.4 % as its "before". The determinism fix
already moved two of those, so reading the boundary-layer work off the v0.3 numbers
would mis-attribute the difference. This section
was checked against `get_default_...` until the default moved to the junction-protected
configuration; the numbers are unchanged, only the factory they are read from.

The **"revised meshing" variant** (boundary layer + junction protection + link-condition-
checked offset exclusion, `get_link_checked_mesh_reconstruction_algorithm`, a former
default, now the opt-in variant — see `get_default_algorithm`'s docstring for why) is
asserted in the third section, so its numbers stay pinned by absolute values on the one
case all the later meshing work is anchored to and not only by the golden masters.
Every metric that moved at each of the three meshing-pipeline flips was quantified at the
time; `BENCHMARKS.md` summarises the dithered vs link-checked comparison. **The current
default is `dithered` again**, so its reference values are the ones in the first section,
unchanged.

The fourth section pins the **offset-included configuration the link-checked flip
demoted**, against `get_offset_included_mesh_reconstruction_algorithm`. Only two numbers
separate it from the third section — the mesh point and triangle counts — because the
link-condition-checked collapse changes the *extraction* and nothing upstream of it.
Keeping both on the same case is what makes that claim checkable rather than asserted:
everything except the surface must be bit-identical between them.

Total tetrahedra count is allowed a tolerance of a few units: Qhull can resolve
near-degenerate (cospherical) point configurations differently between
environments/versions (observed: it reproduces Delaunay to within 3 tetrahedra).
"""

import pytest

from dw3d_benchmarks import metrics as m
from dw3d import (
    get_deterministic_mesh_reconstruction_algorithm,
    get_dithered_mesh_reconstruction_algorithm,
    get_junction_protected_mesh_reconstruction_algorithm,
    get_link_checked_mesh_reconstruction_algorithm,
)
from tests.conftest import load_image_or_skip

# dw3d 0.3.6 reference values, on data/Images/3.tif, min_distance=3:
#   interface pts 5911 | interior pts 133 | tets 24153 | all-surface tets 58.0%
#   tets with >=1 interior vertex 41.5% | tet qual median 0.0232 | slivers 14.1%
#   mesh pts 1580 | triangles 3311 | valence-3 edges 163 | score gaps <1e-3: 80.4%


def _reconstruct(algorithm_getter):
    mask = load_image_or_skip("3.tif")
    algo = algorithm_getter(min_distance=3, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    return algo, points, triangles, labels


@pytest.fixture(scope="module")
def reconstruction_3tif_dithered():
    return _reconstruct(get_dithered_mesh_reconstruction_algorithm)


@pytest.fixture(scope="module")
def reconstruction_3tif_deterministic():
    return _reconstruct(get_deterministic_mesh_reconstruction_algorithm)


@pytest.fixture(scope="module")
def reconstruction_3tif_link_checked():
    return _reconstruct(get_link_checked_mesh_reconstruction_algorithm)


@pytest.fixture(scope="module")
def reconstruction_3tif_offset_included():
    # The configuration the deprecated synonym `get_offset_included_...` names.
    return _reconstruct(get_junction_protected_mesh_reconstruction_algorithm)


def _counts_and_surface_stats(algo):
    """Point-placement counts and tetrahedron classification for whichever variant `algo` is.

    The point-placing function is taken from the algorithm rather than assumed, so these
    helpers measure the variant under test and not the current default.
    """
    counts = m.point_placement_counts(
        algo._edt_image,
        min_distance=3,
        point_placing_function=algo.point_placing_function,
        segmented_image=algo._segmented_image,
    )
    point_classes = m.classify_tesselation_points(len(algo._tesselation_graph.vertices), counts["n_interior_points"])
    return counts, m.tetrahedron_surface_stats(algo._tesselation_graph, point_classes)


# --------------------------------------------------------------------------------------
# The dw3d 0.3.6 reference values, measured on the dithered algorithm
# --------------------------------------------------------------------------------------


def test_point_placement_counts(reconstruction_3tif_dithered):
    algo, _, _, _ = reconstruction_3tif_dithered
    counts, _ = _counts_and_surface_stats(algo)
    assert counts["n_interface_points"] == 5911
    assert counts["n_interior_points"] == 133


def test_tetrahedron_count(reconstruction_3tif_dithered):
    algo, _, _, _ = reconstruction_3tif_dithered
    n_tetrahedra = len(algo._tesselation_graph.tetrahedrons)
    assert abs(n_tetrahedra - 24153) <= 3


def test_mesh_size(reconstruction_3tif_dithered):
    _, points, triangles, _ = reconstruction_3tif_dithered
    assert len(points) == 1580
    assert len(triangles) == 3311


def test_all_surface_and_interior_tet_fractions(reconstruction_3tif_dithered):
    algo, _, _, _ = reconstruction_3tif_dithered
    _, surf = _counts_and_surface_stats(algo)
    assert surf["fraction_all_surface_tets"] == pytest.approx(0.580, abs=0.003)
    assert surf["fraction_tets_with_interior_vertex"] == pytest.approx(0.415, abs=0.003)


def test_tet_quality_and_slivers(reconstruction_3tif_dithered):
    algo, _, _, _ = reconstruction_3tif_dithered
    _, surf = _counts_and_surface_stats(algo)
    quality = m.tetrahedron_quality(algo._tesselation_graph)
    sliv = m.sliver_stats(quality, surf["_all_surface_mask"])
    assert sliv["tet_quality_median"] == pytest.approx(0.0232, abs=0.0005)
    assert sliv["sliver_fraction"] == pytest.approx(0.141, abs=0.003)


def test_valence_3_edge_count(reconstruction_3tif_dithered):
    _, points, triangles, labels = reconstruction_3tif_dithered
    edge_stats = m.edge_topology_stats(points, triangles, labels)
    assert edge_stats["valence_histogram"].get(3, 0) == 163


def test_score_gap_fraction_below_1e3(reconstruction_3tif_dithered):
    algo, _, _, _ = reconstruction_3tif_dithered
    sg = m.score_gap_stats(algo._tesselation_graph)
    assert sg["fraction_score_gaps_below_1e-3"] == pytest.approx(0.804, abs=0.003)


# --------------------------------------------------------------------------------------
# The deterministic baseline: an earlier default, same case
# --------------------------------------------------------------------------------------


def test_deterministic_point_placement_counts(reconstruction_3tif_deterministic):
    algo, _, _, _ = reconstruction_3tif_deterministic
    counts, _ = _counts_and_surface_stats(algo)
    assert counts["n_interface_points"] == 5228  # v0.3: 5911 (0.88x)
    assert counts["n_interior_points"] == 224  # v0.3: 133 (1.68x)


def test_deterministic_mesh_size_and_tetrahedron_count(reconstruction_3tif_deterministic):
    algo, points, triangles, _ = reconstruction_3tif_deterministic
    assert len(points) == 1172  # v0.3: 1580
    assert len(triangles) == 2477  # v0.3: 3311
    assert abs(len(algo._tesselation_graph.tetrahedrons) - 20912) <= 3  # v0.3: 24152


def test_deterministic_all_surface_and_sliver_fractions(reconstruction_3tif_deterministic):
    """The two boundary-layer acceptance metrics the determinism fix moves, pinned as its true "before"."""
    algo, _, _, _ = reconstruction_3tif_deterministic
    _, surf = _counts_and_surface_stats(algo)
    quality = m.tetrahedron_quality(algo._tesselation_graph)
    sliv = m.sliver_stats(quality, surf["_all_surface_mask"])
    assert surf["fraction_all_surface_tets"] == pytest.approx(0.420, abs=0.003)  # 0.3.6: 0.580; target < 0.15
    assert sliv["sliver_fraction"] == pytest.approx(0.062, abs=0.003)  # v0.3: 0.141, boundary-layer target < 0.07
    assert sliv["tet_quality_median"] == pytest.approx(0.0293, abs=0.0005)  # v0.3: 0.0232


def test_deterministic_score_gap_fractions(reconstruction_3tif_deterministic):
    """Score degeneracy is essentially unmoved: it is a property of the EDT's plateaus."""
    algo, _, _, _ = reconstruction_3tif_deterministic
    sg = m.score_gap_stats(algo._tesselation_graph)
    assert sg["fraction_score_gaps_below_1e-3"] == pytest.approx(0.774, abs=0.003)  # v0.3: 0.804
    assert sg["fraction_score_gaps_below_1e-5"] == pytest.approx(0.489, abs=0.003)  # v0.3: 0.380


# --------------------------------------------------------------------------------------
# The "revised meshing" variant: boundary layer + junction protection + link-condition-
# checked offset exclusion (get_link_checked_mesh_reconstruction_algorithm; a former
# default, now opt-in -- see get_default_algorithm's docstring for why)
# --------------------------------------------------------------------------------------


def test_link_checked_point_placement_counts(reconstruction_3tif_link_checked):
    """The point budget is *smaller* than the deterministic one's despite adding two families of points.

    Shell coarsening removes 89 % of the bounding-box / background shell minima, which is
    where 77.6 % of the deterministic configuration's "interface" budget went; the boundary
    layer's offsets and junction protection's junction samples together add back less than
    that. The interface/interior split also inverts, because both new families are
    classified as interior (they carry a non-zero EDT value).
    """
    algo, _, _, _ = reconstruction_3tif_link_checked
    counts, _ = _counts_and_surface_stats(algo)
    assert counts["n_interface_points"] == 1534  # deterministic: 5228 (0.29x)
    assert counts["n_interior_points"] == 2655  # deterministic: 224 (11.9x)
    assert counts["n_interface_points"] + counts["n_interior_points"] == 4189  # deterministic: 5452 (0.77x)


def test_link_checked_mesh_size_and_tetrahedron_count(reconstruction_3tif_link_checked):
    """The link-checked flip shows up here and nowhere else in this section.

    The link-condition-checked collapse removes the boundary layer's offsets from the
    *extracted surface* while leaving them in the tesselation, so the tetrahedron count is
    untouched (25 997, exactly the offset-included configuration's) and the mesh shrinks by
    31 %. Every other assertion in this section is unchanged from the junction-protected
    default, which is the point: nothing upstream of the extraction moved.
    """
    algo, points, triangles, _ = reconstruction_3tif_link_checked
    # A later fix to the surgery selector's comparator (it was inverted -- an `np.argmin`
    # on a score its own docstring says to maximise) changes which repair is applied at one
    # competitive decision on this case, moving these two counts by +1 point and +4
    # triangles (was 1363 / 2939). The mesh is equivalent on every blocking acceptance
    # criterion -- 0 abnormal non-manifold edges before and after, watertight, identical
    # material-pair set, no degenerate triangle, no unclosed cell. The attribution claim
    # this test exists to protect is untouched: the tetrahedron count below is unchanged,
    # and `test_offset_included_and_link_checked_differ_only_in_the_extraction` still passes.
    assert len(points) == 1364  # offset_included: 1980, deterministic: 1172, v0.3: 1580
    assert len(triangles) == 2943  # offset_included: 4175, deterministic: 2477, v0.3: 3311
    assert abs(len(algo._tesselation_graph.tetrahedrons) - 25997) <= 3  # offset_included: 25997, deterministic: 20912


def test_link_checked_all_surface_and_sliver_fractions(reconstruction_3tif_link_checked):
    """The boundary layer's headline metric, and the sliver fraction it had to hold while moving it."""
    algo, _, _, _ = reconstruction_3tif_link_checked
    _, surf = _counts_and_surface_stats(algo)
    quality = m.tetrahedron_quality(algo._tesselation_graph)
    sliv = m.sliver_stats(quality, surf["_all_surface_mask"])
    # The boundary layer's target was < 0.15; deterministic sat at 0.420 and v0.3 at 0.580.
    assert surf["fraction_all_surface_tets"] == pytest.approx(0.0085, abs=0.003)
    assert sliv["sliver_fraction"] == pytest.approx(0.053, abs=0.003)  # deterministic: 0.062, v0.3: 0.141
    assert sliv["tet_quality_median"] == pytest.approx(0.0444, abs=0.0005)  # deterministic: 0.0293


def test_link_checked_score_gap_fractions(reconstruction_3tif_link_checked):
    """`< 1e-5` is the primary figure; the `< 1e-3` one is face-count-confounded.

    The outcome of the boundary-layer work explains why: the
    boundary layer roughly doubles the face count, so consecutive gaps shrink mechanically
    and the absolute 1e-3 threshold reports a worsening while the bit-level degeneracy
    improves. Both are asserted, with the confound stated, rather than quietly reporting
    only the flattering one.
    """
    algo, _, _, _ = reconstruction_3tif_link_checked
    sg = m.score_gap_stats(algo._tesselation_graph)
    assert sg["fraction_score_gaps_below_1e-5"] == pytest.approx(0.208, abs=0.003)  # deterministic: 0.489
    assert sg["fraction_score_gaps_below_1e-3"] == pytest.approx(0.852, abs=0.003)  # deterministic: 0.774


def test_link_checked_has_no_quadjunction_edge_and_no_hole_on_this_case(reconstruction_3tif_link_checked):
    """The deterministic configuration leaves 3 valence->=4 edges on 3.tif; the link-checked collapse leaves none.

    Not a 51-case claim -- the totals over the full set are 133 (deterministic), 115
    (offset-included), 184 (the unconditional weld) and 114 (link-checked) -- but it is the
    single case the baseline measurement and every reference value above are anchored to, so it is pinned
    here too. The unconditional weld is the reason this is worth pinning: an extraction
    change *can* add edge valence, and the link-condition-checked collapse rule is what
    forbids it.
    """
    _, points, triangles, labels = reconstruction_3tif_link_checked
    edge_stats = m.edge_topology_stats(points, triangles, labels)
    assert edge_stats["n_valence_geq4_edges"] == 0
    assert edge_stats["n_holes"] == 0
    assert edge_stats["valence_histogram"].get(3, 0) == 227  # deterministic: 163 at v0.3, more triple lines resolved


# --------------------------------------------------------------------------------------
# Boundary layer + junction protection: a former default, kept pinned as a measured regression
# --------------------------------------------------------------------------------------


def test_offset_included_mesh_size(reconstruction_3tif_offset_included):
    """The two numbers the link-checked flip moves, pinned on the configuration it demoted.

    31 % of the mesh vertices and 29 % of the triangles the junction-protected default
    emitted came from the boundary layer's offsets, which sit `delta = 3` voxels off the
    surface by design. Over the 47 ground-truth cases they inflate every interface area by
    a median +15.7 %. Pinning them here means the
    regression stays measurable, not merely documented.
    """
    _, points, triangles, _ = reconstruction_3tif_offset_included
    # A later fix to the surgery selector's comparator (the same one and the same evidence
    # as the link_checked section above) applies a different repair at one competitive
    # decision on this case, moving these two counts by +6 points and +14 triangles (was
    # 1974 / 4161); the mesh is equivalent on every blocking acceptance criterion (0
    # abnormal non-manifold edges before and after, watertight, identical material-pair
    # set). The regression this test exists to keep measurable is unaffected in kind: the
    # boundary layer's offsets still account for ~31% of the vertices and ~30% of the
    # triangles that the junction-protected default emitted over the link-checked default's.
    assert len(points) == 1980  # the link-checked default: 1364
    assert len(triangles) == 4175  # the link-checked default: 2943


def test_offset_included_and_link_checked_differ_only_in_the_extraction(
    reconstruction_3tif_offset_included,
    reconstruction_3tif_link_checked,
):
    """The attribution claim, asserted rather than inferred from two tables.

    The link-condition-checked collapse touches `mesh_utilities`' surface extraction and
    nothing before it, so the point set, the tesselation, the tetrahedron count, the score
    field and the labelling must all be identical between the two, and only the surface may
    differ. If a future change to that collapse reached upstream, this fails.
    """
    offset_included_algo, offset_included_points, offset_included_triangles, _ = reconstruction_3tif_offset_included
    link_checked_algo, link_checked_points, link_checked_triangles, _ = reconstruction_3tif_link_checked

    offset_included_tess = offset_included_algo._tesselation_graph
    link_checked_tess = link_checked_algo._tesselation_graph
    assert len(offset_included_tess.vertices) == len(link_checked_tess.vertices)
    assert (offset_included_tess.vertices == link_checked_tess.vertices).all()
    assert (offset_included_tess.tetrahedrons == link_checked_tess.tetrahedrons).all()
    assert (offset_included_algo._map_node_id_to_label == link_checked_algo._map_node_id_to_label).all()
    # ...and the surface really is different, so the assertions above are not vacuous.
    assert (len(link_checked_points), len(link_checked_triangles)) != (
        len(offset_included_points),
        len(offset_included_triangles),
    )
