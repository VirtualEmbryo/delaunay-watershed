"""The relabelling of near-tie membrane slivers between the seeded watershed and the overrides.

Pinned:

1. on a hand-made pair of flat membrane triangles that makes a Z-shaped interface, the triangle whose
   watershed choice is a near tie takes its partner's label and the Z is gone; the triangle with a
   clear margin keeps its label; a margin equal to the threshold is not a near tie; a seed or zero-node
   triangle is never relabelled; triangles that are not flat change nothing;
2. on an equilibrium-foam image (cells of 40 px radius, ``min_dist`` 3, benchmark tissue n008_slot01,
   the image stored for the contact guard) the default reconstruction relabels the two triangles the
   research prototype relabelled, with the prototype's counts, and the locus projection on that mesh
   then has no move to refuse;
3. the default: on for the plain reconstruction, off when ``relocate_junctions=True``, and the
   explicit ``True`` / ``False`` override it;
4. on every golden case the rule relabels nothing, on the default route and on the legacy route
   (given the golden triangulation, so on every machine): the outputs are bitwise those without it.
"""
import contextlib
import io
import types

import numpy as np
import pytest
from .conftest import DATA, GOLDEN, golden_cases

from dw2d import geometry_reconstruction_2d
from dw2d.sliver_relabelling import (NEAR_TIE_SCORE_MARGIN, relabel_near_tie_slivers,
                                     valley_sliver_triangles)

FIXTURE = DATA / "equilibrium_foam_with_a_flattened_z_fold.npz"
CASES = golden_cases()
IDS = [c[0] for c in CASES]


def _build(labels, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(labels, **kwargs)
        dw.return_mesh()
    return dw


def _triangle_labels(dw):
    out = np.full(len(dw.tri.simplices), np.iinfo(np.int64).min, dtype=np.int64)
    for label, triangles in dw.Map_end.items():
        out[np.asarray(triangles, dtype=np.int64)] = int(label)
    return out


# ---------------------------------------------------------------- hand-made sliver pair
#
#            T (label 1)
#   m0 ---- m1 ------------ m3        m1 and m3 at height 0.3, m0 and m2 at 0: the two flat
#     \  S1  \     S2      /          triangles S1 = (m0, m1, m2) and S2 = (m1, m2, m3) fill the gap
#      ------ m2 ----------           between the upper path m0-m1-m3 and the lower path m0-m2-m3.
#            B (label 2)
#
# With S1 labelled 1 and S2 labelled 2 the interface runs m0 -> m2 -> m1 -> m3 and turns back on itself
# at m2 and at m1 (two reversals: the Z). Giving S2 the label 1 (or S1 the label 2) leaves a straight
# interface along one of the two paths.
M0, M1, M2, M3, T, B = range(6)
POINTS = np.array([[0.0, 0.0], [2.0, 0.3], [4.0, 0.0], [6.0, 0.3], [3.0, 3.0], [3.0, -3.0]])
TRIANGLES = np.sort(np.array([[M0, M1, M2], [M1, M2, M3], [M0, M1, T], [M1, M3, T], [M0, M2, B], [M2, M3, B]]), axis=1)
S1, S2, TOP1, TOP2, BOTTOM1, BOTTOM2 = range(6)
# dual edges: (triangle, triangle, Delaunay edge)
DUAL = [(S1, S2, (M1, M2)), (S1, TOP1, (M0, M1)), (S1, BOTTOM1, (M0, M2)), (S2, TOP2, (M1, M3)),
        (S2, BOTTOM2, (M2, M3)), (TOP1, TOP2, (M1, T)), (BOTTOM1, BOTTOM2, (M2, B))]
WATERSHED = np.array([1, 2, 1, 1, 2, 2])


def _hand_made_graph(scores):
    """A ``Delaunay_Graph`` stand-in: the rule reads only these five attributes."""
    return types.SimpleNamespace(Vertices=POINTS, Tris=TRIANGLES,
                                 Nodes_linked_by_Edges=np.array([[a, b] for a, b, _ in DUAL]),
                                 Edges=np.array([sorted(e) for _, _, e in DUAL]),
                                 Scores=np.asarray(scores, float))


def _scores(s1_to_top=2.0, s1_to_bottom=1.0, s2_to_top=1.5, s2_to_bottom=1.55, shared=1.0):
    """S1's margin towards label 2 is s1_to_top - max(s1_to_bottom, shared); S2's towards 1 is
    s2_to_bottom - max(s2_to_top, shared)."""
    return [shared, s1_to_top, s1_to_bottom, s2_to_top, s2_to_bottom, 5.0, 5.0]


def test_the_hand_made_pair_is_a_pair_of_valley_slivers():
    sliver = valley_sliver_triangles(POINTS, TRIANGLES, n_membrane_points=4)
    assert sliver.tolist() == [True, True, False, False, False, False]
    assert not valley_sliver_triangles(POINTS, TRIANGLES, n_membrane_points=3)[S2]  # m3 not a membrane point


def test_the_near_tie_sliver_takes_its_partners_label_and_the_clear_one_keeps_its_own():
    labels, report = relabel_near_tie_slivers(_hand_made_graph(_scores()), WATERSHED, 4, [])
    # S1's margin towards 2 is 1.0 (kept); S2's towards 1 is 0.05 < 0.1 (relabelled to S1's label 1)
    assert labels.tolist() == [1, 1, 1, 1, 2, 2]
    assert report.relabelled_triangles == [S2] and report.triangles_changed == 1
    assert (report.flips, report.candidates_examined, report.below_margin, report.refused_no_gain) == (1, 2, 1, 0)
    assert report.alternating_pairs_before == 1 and report.converged and report.passes == 2
    assert WATERSHED.tolist() == [1, 2, 1, 1, 2, 2]  # the input is not modified


def test_a_margin_equal_to_the_threshold_is_not_a_near_tie():
    graph = _hand_made_graph(_scores(s2_to_top=1.0, s2_to_bottom=1.125))  # S2's margin is exactly 0.125
    labels, report = relabel_near_tie_slivers(graph, WATERSHED, 4, [], margin=0.125)
    assert labels.tolist() == WATERSHED.tolist() and report.below_margin == 0
    labels, _ = relabel_near_tie_slivers(graph, WATERSHED, 4, [], margin=np.nextafter(0.125, 1))
    assert labels[S2] == 1


def test_a_seed_or_zero_node_triangle_is_never_relabelled():
    labels, report = relabel_near_tie_slivers(_hand_made_graph(_scores()), WATERSHED, 4, [S2])
    assert labels.tolist() == WATERSHED.tolist()
    assert report.skipped_protected == 1 and report.triangles_changed == 0


def test_when_both_are_near_ties_the_first_visited_triangle_is_relabelled_and_the_z_is_gone():
    labels, report = relabel_near_tie_slivers(_hand_made_graph(_scores(s1_to_top=1.04)), WATERSHED, 4, [])
    assert labels.tolist() == [2, 2, 1, 1, 2, 2]  # S1 (the first side of the dual edge) takes S2's label 2
    assert report.relabelled_triangles == [S1] and report.flips == 1


def test_triangles_that_are_not_flat_change_nothing():
    points = POINTS.copy()
    points[M1] = [2.0, 1.5]  # S1's largest angle is now about 106 degrees
    graph = _hand_made_graph(_scores())
    graph.Vertices = points
    labels, report = relabel_near_tie_slivers(graph, WATERSHED, 4, [])
    assert labels.tolist() == WATERSHED.tolist()
    assert report.alternating_pairs_before == 0 and report.triangles_changed == 0


def test_the_default_margin_is_a_tenth_of_a_pixel():
    assert NEAR_TIE_SCORE_MARGIN == 0.1


# ---------------------------------------------------------------- a real image where it fires
def test_on_an_equilibrium_foam_the_rule_relabels_what_the_research_prototype_relabelled():
    """n008_slot01, cells of 40 px radius, min_dist 3: the prototype relabelled triangles 784 and 821."""
    labels = np.load(FIXTURE)["labels"].astype(np.int64)
    dw = _build(labels, min_dist=3, relocate_junctions=False)
    rep = dw.near_tie_sliver_report
    assert rep.relabelled_triangles == [784, 821]
    assert (rep.valley_slivers, rep.alternating_pairs_before, rep.passes, rep.flips) == (69, 15, 2, 2)
    assert (rep.candidates_examined, rep.below_margin, rep.refused_no_gain, rep.skipped_protected) == (54, 48, 46, 0)
    assert rep.converged
    off = _build(labels, min_dist=3, relocate_junctions=False, relabel_near_tie_slivers=False)
    differ = np.flatnonzero(_triangle_labels(dw) != _triangle_labels(off))
    assert differ.tolist() == [784, 821]


def test_after_the_rule_the_projection_has_no_move_to_refuse_on_that_image():
    """Without the rule the projection refuses one crossing and one contact there (the Z-fold); with it, none."""
    labels = np.load(FIXTURE)["labels"].astype(np.int64)
    kw = dict(min_dist=3, relocate_junctions=True, project_interface_vertices=True)
    without = _build(labels, **kw)
    with_rule = _build(labels, relabel_near_tie_slivers=True, **kw)
    assert without.locus_projection_report["refused"] == {"crossing": 1, "contact": 1}
    assert with_rule.locus_projection_report["refused"] == {}
    assert with_rule.locus_projection_report["contacts_after"] == 0


# ---------------------------------------------------------------- the default
def test_the_rule_is_on_for_the_plain_reconstruction_and_off_after_relocation_unless_asked():
    """Since 0.8.0 the relocation is on by default, so the rule is off by default and on for relocate_junctions=False."""
    labels = np.load(FIXTURE)["labels"].astype(np.int64)
    assert _build(labels, min_dist=3, relocate_junctions=False).relabel_near_tie_slivers
    assert _build(labels, min_dist=3, relocate_junctions=False, deterministic=False).relabel_near_tie_slivers
    default = _build(labels, min_dist=3)
    assert default.relocate_junctions and not default.relabel_near_tie_slivers and default.near_tie_sliver_report is None
    relocated = _build(labels, min_dist=3, relocate_junctions=True)
    assert not relocated.relabel_near_tie_slivers and relocated.near_tie_sliver_report is None
    projected = _build(labels, min_dist=3, relocate_junctions=True, project_interface_vertices=True)
    assert not projected.relabel_near_tie_slivers and projected.near_tie_sliver_report is None
    assert _build(labels, min_dist=3, relocate_junctions=True, relabel_near_tie_slivers=True).near_tie_sliver_report is not None
    off = _build(labels, min_dist=3, relabel_near_tie_slivers=False)
    assert not off.relabel_near_tie_slivers and off.near_tie_sliver_report is None


# ---------------------------------------------------------------- golden cases: nothing to relabel
@pytest.fixture(scope="session")
def golden_triangulations():
    return np.load(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_triangulations.npz")


def _assert_same_reconstruction(a, b):
    assert np.array_equal(_triangle_labels(a), _triangle_labels(b))
    va, ea = a.return_mesh()
    vb, eb = b.return_mesh()
    assert np.array_equal(va, vb) and np.array_equal(ea, eb)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_rule_relabels_nothing_on_every_golden_case_of_the_plain_route(name, labels, min_dist, expansion):
    on = _build(labels, min_dist=min_dist, expansion_labels=expansion, relocate_junctions=False)
    off = _build(labels, min_dist=min_dist, expansion_labels=expansion, relocate_junctions=False,
                 relabel_near_tie_slivers=False)
    assert on.near_tie_sliver_report.triangles_changed == 0 and on.near_tie_sliver_report.converged
    _assert_same_reconstruction(on, off)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=IDS)
def test_the_rule_relabels_nothing_on_every_golden_case_of_the_legacy_route(golden_triangulations, monkeypatch,
                                                                            name, labels, min_dist, expansion):
    import dw2d.Geometric_utilities as geometric_utilities

    qhull = geometric_utilities.Delaunay
    monkeypatch.setattr(geometric_utilities, "Delaunay",
                        lambda points: types.SimpleNamespace(points=qhull(points).points,
                                                             simplices=golden_triangulations[name]))
    kw = dict(min_dist=min_dist, expansion_labels=expansion, deterministic=False, relocate_junctions=False,
              repair_pinch_vertices=False)
    on = _build(labels, relabel_near_tie_slivers=True, **kw)
    off = _build(labels, relabel_near_tie_slivers=False, **kw)
    assert on.near_tie_sliver_report.triangles_changed == 0
    _assert_same_reconstruction(on, off)
