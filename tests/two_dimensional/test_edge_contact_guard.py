"""The guard that refuses a vertex move creating an edge contact (``dw2d.junction_relocation``, "Edge contacts").

Pinned: (1) every contact class is recognised on hand-made segments, exactly, including point triples
so close to collinear that the floating-point orientation cannot decide; (2) the whole-mesh contact
count counts each class and skips zero-length edges; (3) the mover refuses a move that creates a
contact, leaves the vertex where it was, and accepts a move that keeps an existing contact as it was;
(4) the locus projection refuses the move that flattens a Z-shaped turn onto the edge before it, on a
hand-made chain and on an equilibrium-foam image (cell radius 40 px, ``min_dist`` 3) where the
projection without the guard made an edge touch another; (5) where no move creates a contact (the
golden cases), the relocation and the projection give bitwise the vertices they gave without the guard.
"""
import contextlib
import io
from fractions import Fraction

import numpy as np
import pytest

from .conftest import DATA, golden_cases
from dw2d import geometry_reconstruction_2d
from dw2d import junction_relocation as JR
from dw2d.locus_projection import _project_on_edges as project_interface_vertices

FIXTURE = DATA / "equilibrium_foam_with_a_flattened_z_fold.npz"


def _quiet(labels, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(labels, **kw)
        verts, edges = dw.return_mesh()
    return dw, verts, edges


def _pt(*x):
    return np.array(x, float)


@pytest.mark.parametrize("p1, p2, q1, q2, expected", [
    ((0, 0), (2, 2), (0, 2), (2, 0), "proper_crossing"),
    ((0, 0), (4, 0), (2, 0), (3, 5), "t_touch"),          # an endpoint inside the other edge
    ((0, 0), (4, 0), (2, -1), (2, 0), "t_touch"),
    ((0, 0), (4, 0), (4, 0), (5, 3), "coincident_vertices"),  # two vertices at the same point
    ((0, 0), (4, 0), (4, 0), (6, 0), "collinear_point"),      # end to end on one line
    ((0, 0), (4, 0), (3, 0), (6, 0), "collinear_overlap"),
    ((0, 0), (4, 0), (1, 0), (2, 0), "collinear_overlap"),    # one inside the other
    ((0, 0), (4, 0), (5, 0), (6, 0), None),                   # collinear, apart
    ((0, 0), (4, 0), (0, 1), (4, 1), None),                   # parallel
    ((0, 0), (4, 0), (2, 0.5), (3, 5), None),                 # near, not touching
])
def test_each_contact_class_between_edges_sharing_no_vertex(p1, p2, q1, q2, expected):
    assert JR._contact_class(_pt(*p1), _pt(*p2), _pt(*q1), _pt(*q2)) == expected
    assert JR._contact_class(_pt(*q1), _pt(*q2), _pt(*p1), _pt(*p2)) == expected  # symmetric


@pytest.mark.parametrize("s, x, y, expected", [
    ((0, 0), (4, 0), (2, 0), "adjacent_overlap"),   # (s, y) runs back along (s, x): a turn of 180 degrees
    ((0, 0), (4, 4), (8, 8), "adjacent_overlap"),
    ((0, 0), (4, 0), (-2, 0), None),                # straight continuation through s
    ((0, 0), (4, 0), (2, 1e-9), None),              # almost folded back, not exactly
])
def test_adjacent_overlap_between_edges_sharing_a_vertex(s, x, y, expected):
    assert JR._adjacent_contact_class(_pt(*s), _pt(*x), _pt(*y)) == expected


def _orient_fraction(a, b, c):
    ax, ay, bx, by, cx, cy = (Fraction(float(t)) for t in (*a, *b, *c))
    d = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (d > 0) - (d < 0)


def test_orientation_is_exact_where_floating_point_cannot_decide():
    rng = np.random.default_rng(0)
    undecided = 0
    for _ in range(2000):
        a = rng.uniform(0, 300, 2)
        b = rng.uniform(0, 300, 2)
        t = rng.uniform(-1, 2)
        c = a + t * (b - a)                        # on the line up to rounding
        c = c + rng.integers(-2, 3, 2) * np.spacing(c)
        detleft = (a[0] - c[0]) * (b[1] - c[1]); detright = (a[1] - c[1]) * (b[0] - c[0])
        undecided += abs(detleft - detright) <= JR._CCW_ERRBOUND_A * (abs(detleft) + abs(detright))
        assert JR._orient_sign(a, b, c) == _orient_fraction(a, b, c)
    assert undecided > 100  # the exact fallback was exercised, not only the filter


def test_a_point_on_a_float_segment_is_a_touch_only_if_it_is_exactly_on_it():
    p1, p2 = _pt(0.1, 0.1), _pt(0.7, 0.7)
    on = _pt(0.3, 0.3)                              # exactly on the line y = x
    assert JR._contact_class(p1, p2, on, _pt(0.3, 5.0)) == "t_touch"
    near = _pt(0.3, np.nextafter(0.3, 1.0))         # one unit in the last place above it
    assert JR._contact_class(p1, p2, near, _pt(0.3, 5.0)) is None


def test_whole_mesh_contacts_counts_every_class_and_skips_zero_length_edges():
    verts = np.array([[0, 0], [4, 0], [2, 0], [3, 5], [10, 0], [12, 0], [11, 0], [20, 0], [20, 0], [20, 3]], float)
    edges = np.array([[0, 1, 1, 2],     # 0
                      [2, 3, 1, 2],     # 1: T-touch on edge 0
                      [4, 5, 1, 2],     # 2
                      [5, 6, 1, 2],     # 3: runs back along edge 2 (adjacent overlap)
                      [7, 8, 1, 2],     # 4: zero length
                      [8, 9, 1, 2]])    # 5
    assert JR.whole_mesh_contacts(verts, edges) == 2
    assert JR.whole_mesh_contacts(verts[:, ::-1], edges) == 2  # the frame does not matter
    assert JR.whole_mesh_contacts(verts, edges[[0, 2, 4, 5]]) == 0


def _straight_interface():
    """A chain a-b-c-d with a Z turn at b and c, between labels 1 (rows < 5) and 2 (rows >= 5).

    a and d have degree 1 and are never projected; b already sits on the locus (row 4.5); the
    projection would move c from (4, 4) to (4.5, 4), inside edge a-b: a T-touch of c-d on a-b and
    an exact fold-back of b-c along b-a.
    """
    L = np.ones((10, 12), int); L[5:] = 2
    verts = np.array([[4.5, 1.0], [4.5, 7.0], [4.0, 4.0], [4.4, 10.0]])
    edges = np.array([[0, 1, 1, 2], [1, 2, 1, 2], [2, 3, 1, 2]])
    return L, verts, edges


def test_mover_refuses_a_move_that_creates_a_contact_and_the_vertex_stays():
    _, verts, edges = _straight_interface()
    mover = JR.GuardedVertexMover(verts, edges)
    assert sorted(mover._local_contacts(2, np.array([4.5, 4.0]))) == [("adjacent_overlap", 0, 1), ("t_touch", 0, 2)]
    assert mover.try_move(2, np.array([4.5, 4.0])) == "contact"
    assert np.array_equal(mover.v, verts)
    assert mover.try_move(2, np.array([4.2, 4.0])) == ""          # closer, no contact: accepted
    assert JR.whole_mesh_contacts(mover.v, edges) == 0


def test_mover_accepts_a_move_that_keeps_an_existing_contact():
    verts = np.array([[0, 0], [4, 0], [2, 0], [3, 5], [3, 6]], float)
    edges = np.array([[0, 1, 1, 2], [2, 3, 1, 2], [3, 4, 1, 2]])
    mover = JR.GuardedVertexMover(verts, edges)
    assert JR.whole_mesh_contacts(verts, edges) == 1
    assert mover.try_move(3, np.array([3.2, 5.0])) == ""          # edge 1 still touches edge 0 at (2, 0)
    assert JR.whole_mesh_contacts(mover.v, edges) == 1


def test_projection_refuses_to_flatten_a_z_turn_onto_the_edge_before_it():
    L, verts, edges = _straight_interface()
    moved, report = project_interface_vertices(verts, edges, L)
    assert report["refused"] == {"contact": 1}
    assert report["contacts_before"] == report["contacts_after"] == 0
    assert np.array_equal(moved, verts)
    flattened = verts.copy(); flattened[2] = [4.5, 4.0]
    assert JR.whole_mesh_contacts(flattened, edges) == 2


def _without_the_contact_guard(monkeypatch):
    monkeypatch.setattr(JR.GuardedVertexMover, "_local_contacts", lambda self, vid, pos: set())


def test_projection_on_an_equilibrium_foam_never_creates_a_contact(monkeypatch):
    """Cell radius 40 px, eight cells (benchmark tissue n008_slot01): the projection without the guard made a T-touch."""
    labels = np.load(FIXTURE)["labels"].astype(np.int64)
    dw, verts, edges = _quiet(labels, min_dist=3, relocate_junctions=True, project_interface_vertices=True)
    rep = dw.locus_projection_report
    assert rep["refused"].get("contact") == 1
    assert rep["contacts_before"] == rep["contacts_after"] == 0
    assert JR.whole_mesh_contacts(verts, edges) == 0
    # the plain mesh of the relocated route, which is built without the near-tie sliver relabelling
    relocated, _ = JR._relocate_on_edges(*_quiet(labels, min_dist=3, relocate_junctions=False,
                                                                  relabel_near_tie_slivers=False)[1:], dw.labels)
    _without_the_contact_guard(monkeypatch)
    with pytest.raises(RuntimeError, match="contact count rose 0 -> 2"):
        project_interface_vertices(relocated, edges, dw.labels)


@pytest.mark.parametrize("case", golden_cases(), ids=lambda c: c[0])
def test_meshes_without_contacts_are_bitwise_unchanged(case, monkeypatch):
    name, labels, min_dist, expansion = case
    kw = dict(min_dist=min_dist, expansion_labels=expansion, relocate_junctions=True)
    dw_r, v_r, e_r = _quiet(labels, **kw)
    dw_p, v_p, e_p = _quiet(labels, project_interface_vertices=True, **kw)
    assert "contact" not in dw_r.junction_relocation_report["refused"]
    assert "contact" not in dw_p.locus_projection_report["refused"]
    assert dw_p.locus_projection_report["contacts_after"] == 0
    _without_the_contact_guard(monkeypatch)
    _, v_r0, e_r0 = _quiet(labels, **kw)
    _, v_p0, e_p0 = _quiet(labels, project_interface_vertices=True, **kw)
    assert np.array_equal(v_r, v_r0) and np.array_equal(e_r, e_r0)
    assert np.array_equal(v_p, v_p0) and np.array_equal(e_p, e_p0)
