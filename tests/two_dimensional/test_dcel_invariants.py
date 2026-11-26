"""Invariants of the half-edge structure built by ``DCEL_Data`` on the golden cases.

``test_half_edge_order_matches_endpoints`` pins a defect found by the 2D audit of 17 Oct 2026 and fixed in 0.8.0:
the orientation repair (``make_closed_oriented_closed_line``) reversed the order of a loop's half-edges without
swapping their endpoints, so on reversed loops ``he.next.origin`` was not ``he.destination``. Angle values survived
(the angle formula is symmetric); the interface lengths and curvatures read along the faces did not
(``test_dcel_lengths.py``). ``reverse_closed_line`` now reverses a loop.
"""
import pytest
from .conftest import golden_cases

CASES = golden_cases()


def build(labels, md, expansion):
    from dw2d import geometry_reconstruction_2d

    return geometry_reconstruction_2d(labels, min_dist=md, expansion_labels=expansion).return_dcel()


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=[c[0] for c in CASES])
def test_every_edge_has_two_twin_half_edges_on_different_faces(name, labels, min_dist, expansion):
    mesh = build(labels, min_dist, expansion)
    assert len(mesh.half_edges) == 2 * len(mesh.e)
    for he in mesh.half_edges:
        assert he.twin is not None and he.twin.twin is he
        assert he.incident_face is not he.twin.incident_face
        assert {id(he.origin), id(he.destination)} == {id(he.twin.origin), id(he.twin.destination)}


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=[c[0] for c in CASES])
def test_every_face_is_one_closed_loop(name, labels, min_dist, expansion):
    mesh = build(labels, min_dist, expansion)
    for face in mesh.faces:
        assert face.closed()
        loop, he = [], face.outer_component
        while True:
            loop.append(he)
            he = he.next
            if he is face.outer_component:
                break
        assert all(h.incident_face is face for h in loop)


@pytest.mark.parametrize("name, labels, min_dist, expansion", CASES, ids=[c[0] for c in CASES])
def test_half_edge_order_matches_endpoints(name, labels, min_dist, expansion):
    mesh = build(labels, min_dist, expansion)
    assert all(he.next.origin is he.destination for he in mesh.half_edges)
    assert all(he.prev.destination is he.origin for he in mesh.half_edges)
