"""Why `post_process_mesh_surgery` leaves a handful of abnormal non-manifold edges.

The first diagnostic of the junction-protection work. Boundary layer + junction protection leaves **7**
abnormal non-manifold edges over the 51-case set (the deterministic configuration leaves 5,
the boundary layer alone leaves 327). The acceptance bar was <= 5, so the question is what
the excess is made of. `benchmarks/diagnose_abnormal_edges.py` produced the measurement;
this module pins the two mechanisms it found, so neither can be silently forgotten and
neither can be re-discovered from scratch later.

An abnormal edge is, by `_find_abnormal_non_manifold_edges`'s definition, an edge with
`k >= 4` incident triangles across fewer than `k` distinct materials. Walking the ring of
tetrahedra around such an edge, the labels form `k` maximal blocks and some material owns
two or more of them -- the mesh crosses the same cell twice on the way round. Both
mechanisms below are that shape; they differ in what surgery can do about it.

**Mechanism 1 -- outside the original vocabulary, now inside it (4 of the 7).**
`[A|B|A|C]`: material `A` occupies two blocks, one of them a singleton, flanked by two
*different* materials `B != C`. Before the AABB-widening fix (`dw3d/mesh_surgery.py`,
`_find_candidate_for_label_switching`), the single-switch rule only fired when a
singleton's two flanking blocks carried the *same* label, and the double-switch rule only
on a length-2 block flanked by two length-2 blocks of the same label -- neither pattern
matches `B != C`, so surgery declined to act on all four. This exact shape (the "AABB"
pattern, `A`'s other occurrence recurring elsewhere in the cycle) was identified in
production data and specified as a widening, which this fix implements. All four of these
survivors match the widened precondition and now resolve via a single label switch --
verified below. `_find_candidate_for_two_labels_switching` is untouched by this fix and
still declines all four, unaffected.

**Mechanism 2 -- surgery oscillates (3 of the 7).** A repair *is* available and it does
remove the edge -- but it recreates the same defect on a neighbouring edge, whose repair
recreates the first. Traced out to 8 iterations on `benchmarking-dataset` cases 030 and
032, the count sits at exactly 1 forever, alternating between two edges with period 2;
`max_iter=3` simply stops the ping-pong at an arbitrary point. Raising `max_iter` is
therefore **not** a fix (measured at 3/6/12/30: unchanged on 030 and 032; case 044 is the
one that does converge, at 6). It is also not free -- it would move `get_dithered_algorithm`'s
output and invalidate the published-comparability fixtures -- so `max_iter` is left at 3
and the oscillation is recorded instead.
"""

import numpy as np
import pytest

from dw3d.mesh_surgery import (
    _find_abnormal_non_manifold_edges,
    _find_candidate_for_label_switching,
    _find_candidate_for_two_labels_switching,
)

# The seven label cycles measured on the offset-included (boundary layer + junction
# protection) survivors, verbatim from `benchmarks/baseline/abnormal_edges_default.json`.
# Case -> (cycle, n_materials).
OFFSET_INCLUDED_SURVIVOR_CYCLES = {
    "004": ([3, 2, 2, 2, 6, 6, 2, 3], 3),
    "008": ([1, 0, 0, 1, 1, 1, 2, 2, 2], 3),
    "010": ([4, 4, 4, 1, 1, 1, 4, 0, 0], 3),
    "030": ([6, 9, 9, 9, 6, 9, 6, 6], 2),
    "032": ([7, 7, 2, 7, 2, 2, 7, 7, 7], 2),
    "044": ([4, 3, 3, 4, 4, 3, 3, 4], 2),
    "045": ([7, 0, 0, 7, 2, 2], 3),
}

# The five deterministic-configuration survivors, same source
# (`benchmarks/baseline/abnormal_edges_deterministic.json`).
DETERMINISTIC_SURVIVOR_CYCLES = {
    "002": ([2, 2, 0, 1, 1, 0, 2, 2], 3),
    "019": ([0, 1, 1, 0, 2, 2, 2], 3),
    "030": ([8, 6, 1, 1, 9, 9, 9, 1], 4),
    "043": ([7, 7, 0, 3, 3, 0], 3),
    "045": ([7, 7, 7, 5, 0, 0, 5, 7, 7], 3),
}

SURGERY_UNREPAIRABLE = ("004", "008", "010", "045")
SURGERY_OSCILLATES = ("030", "032", "044")


def _blocks(cycle: list[int]) -> list[int]:
    """The maximal runs of equal labels around the cyclic sequence, as a list of labels."""
    rotated = list(cycle)
    while len(set(rotated)) > 1 and rotated[0] == rotated[-1]:
        rotated.append(rotated.pop(0))
    blocks = [rotated[0]]
    for label in rotated[1:]:
        if label != blocks[-1]:
            blocks.append(label)
    return blocks


@pytest.mark.parametrize("case", sorted(OFFSET_INCLUDED_SURVIVOR_CYCLES))
def test_every_survivor_revisits_one_material(case):
    """The shared signature: the ring of tetrahedra crosses some cell twice.

    This is what `_find_abnormal_non_manifold_edges` detects, restated on the object
    surgery actually manipulates. Recorded so a later investigation of "which local
    topology fails" has the answer in one place.
    """
    cycle, n_materials = OFFSET_INCLUDED_SURVIVOR_CYCLES[case]
    blocks = _blocks(cycle)
    assert len(blocks) > len(set(blocks)), f"{case}: no material is revisited"
    assert len(set(blocks)) == n_materials
    # Valence 4 on every one of the seven -- the minimum an abnormal edge can have.
    assert len(blocks) == 4


def test_the_excess_over_deterministic_is_a_new_two_material_category():
    """Junction protection's 7 contain three *2-material* defects; deterministic's 5 contain none.

    A 2-material valence-4 edge is a single film pinched against itself -- all four
    incident triangles carry the same label pair. It is not a mis-resolved quadruple point
    and junction protection can neither cause nor cure it: two of the three (032, 044) sit
    more than 10 voxels from any 1-stratum, with a maximum label multiplicity of 2 in their
    neighbourhood, i.e. in the middle of a plain interface.
    """
    offset_included_two_material = [c for c, (_, n) in OFFSET_INCLUDED_SURVIVOR_CYCLES.items() if n == 2]
    deterministic_two_material = [c for c, (_, n) in DETERMINISTIC_SURVIVOR_CYCLES.items() if n == 2]
    assert sorted(offset_included_two_material) == ["030", "032", "044"]
    assert deterministic_two_material == []


@pytest.mark.parametrize("case", SURGERY_UNREPAIRABLE)
def test_mechanism_1_now_has_a_single_switch_repair(case):
    """Four of the seven: the widened single-switch rule now finds a candidate.

    Before the widening, neither the single- nor double-switch search found anything (see
    git history for the assertions this replaced). The double-switch search is untouched
    and still declines all four -- its precondition (a length-2 singleton flanked by two
    length-2 same-label blocks) is a different shape from the AABB pattern targeted here.

    The exhaustive, non-tautological statement is kept alongside, reinterpreted rather
    than weakened: of every single-tetrahedron relabelling tried, none produces a *valid
    new quadjunction* (4 blocks, 4 distinct materials -- `len(blocks) > len(set(blocks))`
    false and `len(blocks) > 3`). The repair `_find_candidate_for_label_switching` finds
    is always the other kind: collapsing the ring to an ordinary trijunction
    (`len(blocks) <= 3`), the same lightweight single-tetrahedron relabel
    `_apply_one_label_switch` already performs everywhere else -- not the whole-block,
    cell-volume-moving merge this module's docstring describes as unbuilt machinery.
    """
    cycle, _ = OFFSET_INCLUDED_SURVIVOR_CYCLES[case]
    assert _find_candidate_for_label_switching(cycle) != []
    assert _find_candidate_for_two_labels_switching(cycle) == []

    materials = set(cycle)
    for position in range(len(cycle)):
        for new_label in materials:
            candidate = list(cycle)
            candidate[position] = new_label
            blocks = _blocks(candidate)
            assert len(blocks) > len(set(blocks)) or len(blocks) <= 3, (
                f"{case}: relabelling tetrahedron {position} to {new_label} "
                "would produce a valid new quadjunction, not just collapse to a trijunction"
            )


@pytest.mark.parametrize("case", SURGERY_OSCILLATES)
def test_mechanism_2_does_have_a_repair_that_locally_succeeds(case):
    """Three of the seven: surgery has a candidate, and applying it *does* fix the ring.

    So the survivor is not a failure of the repair rule but of the loop that applies it:
    the repaired defect reappears on a neighbouring edge and the two trade places with
    period 2 (see the module docstring for the traced runs).
    """
    cycle, _ = OFFSET_INCLUDED_SURVIVOR_CYCLES[case]
    single = _find_candidate_for_label_switching(cycle)
    double = _find_candidate_for_two_labels_switching(cycle)
    assert single or double, f"{case}: expected a repair candidate"

    from dw3d.mesh_surgery import _double_label_switching, _label_switching

    switched = _label_switching(cycle, single) if single else _double_label_switching(cycle, double[:1])
    assert len(_blocks(switched)) <= len(set(_blocks(switched)))


def test_abnormal_edge_definition_is_valence_versus_material_count():
    """Pin the detector itself on a hand-built pinched film, with no image involved.

    Two triangle fans sharing edge (0, 1): four triangles, all labelled (1, 2). Four
    incident triangles, two materials -> abnormal. Adding a third material to two of them
    would make it a legitimate quadjunction-like edge and it would not be flagged.
    """
    points = np.array(
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]])
    pinched = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])
    assert len(_find_abnormal_non_manifold_edges(points, triangles, pinched)) == 1

    quad = np.array([[1, 2], [2, 3], [3, 4], [4, 1]])
    assert len(_find_abnormal_non_manifold_edges(points, triangles, quad)) == 0
