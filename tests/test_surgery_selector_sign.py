"""The surgery selector's comparator sign, and the case-031 2-cycle it was causing.

Every test here is **dataset-independent** -- it runs from plain integer label cycles,
never from a reconstruction -- so it always runs rather than skipping when
`benchmarking-dataset` is not beside the repository. That is deliberate:
`test_aabb_surgery_rule.py` skips in a home-directory worktree, which is exactly where this
work was done, and a regression test that skips where the work happens pins nothing.

Carried across from an earlier lookahead-based fixture that measured this same reproducer
but pinned it against a one-step lookahead later measurement showed to be unnecessary (arm
C matches arm B at every rung; see the project's research history for the full derivation).
The data below is unchanged from that earlier fixture and none of it is invented: the label
cycles, the scores and the per-switch abnormal-edge outcomes were all traced off the real
surgery loop on case 031 at `min_distance=3`. What changed is the claim being pinned: **the
correctly-signed comparator alone is enough.**
"""

import inspect

import pytest

import dw3d.mesh_surgery as mesh_surgery
from dw3d.mesh_surgery import (
    _evaluate_cycle,
    _find_candidate_for_label_switching,
    _label_switching,
    _powerset,
    _search_and_make_double_switching,
    post_process_mesh_surgery,
)

# --------------------------------------------------------------------------------------
# Part 1 -- the defect: the 2-cycle the shipped `argmin` selector fell into on case 031 at md=3.
#
# Two edges sharing vertex 3561. Repairing (3561, 3562) created (3561, 3564); repairing that
# recreated the first. Confirmed non-convergent over five full periods at max_iter=10; shipped
# max_iter=3 stopped it mid-cycle, which is why case 031 read 1 abnormal edge instead of 0.
# --------------------------------------------------------------------------------------
CYCLE_STATE_A = [3, 2, 3, 0, 0, 0]  # edge (3561, 3562)
CYCLE_STATE_B = [3, 0, 2, 0, 3, 3, 3]  # edge (3561, 3564)

TWO_CYCLE_STATES = pytest.mark.parametrize(
    ("label_cycle", "widened_candidates", "original_forced"),
    [(CYCLE_STATE_A, [0, 1, 2], 1), (CYCLE_STATE_B, [1, 2, 3], 2)],
    ids=["state_A", "state_B"],
)


def _select(label_cycle: list[int], candidates: list[int], *, maximise: bool) -> tuple[int, ...]:
    """The production selection expression, in pure Python, over `_powerset(candidates)`.

    `np.argmax`/`np.argmin` return the *first* index of the extremum and `_powerset` enumerates
    by ascending subset size then input order, so `max`/`min` over `range(len(members))` keyed by
    score reproduces each of them exactly -- ties included. `test_the_source_selects_by_argmax_at
    _both_call_sites` is what keeps this helper honest about which one production uses.
    """
    members = _powerset(candidates)
    scores = [_evaluate_cycle(_label_switching(list(label_cycle), list(m))) for m in members]
    pick = max if maximise else min
    return members[pick(range(len(members)), key=lambda i: scores[i])]


@TWO_CYCLE_STATES
def test_the_widened_rule_turns_one_forced_candidate_into_three(label_cycle, widened_candidates, original_forced):
    """The precondition under which the selector is consulted at all.

    With a single candidate production forces it and never enumerates a powerset; the
    AABB-widening fix is what created the competition, and therefore what exposed the
    selection. This is also why the defect stayed latent for as long as it did.
    """
    assert _find_candidate_for_label_switching(label_cycle) == widened_candidates
    assert original_forced in widened_candidates


@TWO_CYCLE_STATES
def test_the_corrected_selector_takes_the_switch_the_original_rule_forced(
    label_cycle, widened_candidates, original_forced,
):
    """The fix, on the two states of the 2-cycle, as an assertion rather than a description.

    `_evaluate_cycle` documents "the higher the better", and on both states its maximum is
    exactly the candidate the rule forced before the AABB widening -- the choice under which
    the shipped algorithm never had this 2-cycle. `argmin` takes a different one. Maximising
    is therefore not merely the documented direction; on this case it is the direction that
    terminates.
    """
    corrected = _select(label_cycle, widened_candidates, maximise=True)
    shipped = _select(label_cycle, widened_candidates, maximise=False)
    assert corrected == (original_forced,)
    assert shipped != corrected, "argmin and argmax must disagree here, or this case pins nothing"


def test_evaluate_cycle_penalises_the_degenerate_collapse_below_every_other_score():
    """Why `argmin` was a defect and not a stale docstring.

    `_evaluate_cycle` returns 0 for the all-one-label cycle to mean "avoid at all cost" -- that
    outcome deletes the edge -- and 0 is strictly below every non-degenerate score, so a
    *minimising* selector prefers precisely what the guard exists to prevent. Not hypothetical:
    on case 031 at md=3 the shipped selector's fourth decision picked a switch scoring exactly 0.
    """
    assert _evaluate_cycle([4, 4, 4, 4]) == 0
    for cycle in ([1, 2, 1, 2], [1, 1, 2, 2], CYCLE_STATE_A, CYCLE_STATE_B):
        assert _evaluate_cycle(cycle) >= 1


# --------------------------------------------------------------------------------------
# Part 2 -- the one decision whose per-switch consequences were measured end to end.
#
# The second surgery decision on case 031 at md=3. Switch (2,) scores 1, is first in powerset
# order, and is therefore what the shipped `argmin` selected -- and it is the switch that created
# edge (3561, 3564), the second half of the 2-cycle.
# --------------------------------------------------------------------------------------
DECISION_CYCLE = [0, 0, 3, 2, 3]
DECISION_CANDIDATES = [2, 3, 4]
TARGET_EDGE = (3561, 4206)
CREATED_EDGE = (3561, 3564)
#: switch -> (creates, removes), exactly as the earlier lookahead fixture's production
#: computed them.
DECISION_OUTCOMES = {
    (2,): ({CREATED_EDGE}, {TARGET_EDGE}),
    (3,): (set(), {TARGET_EDGE}),
    (4,): (set(), {TARGET_EDGE}),
    (2, 3): (set(), {TARGET_EDGE}),
    (2, 4): ({CREATED_EDGE}, {TARGET_EDGE}),
    (3, 4): (set(), {TARGET_EDGE}),
    (2, 3, 4): (set(), {TARGET_EDGE}),
}


def test_scores_of_the_measured_decision_are_pinned():
    """If `_evaluate_cycle` is ever changed, this fails before anything downstream does."""
    expected = {(2,): 1, (3,): 2, (4,): 1, (2, 3): 2, (2, 4): 2, (3, 4): 1, (2, 3, 4): 1}
    for switch, score in expected.items():
        assert _evaluate_cycle(_label_switching(list(DECISION_CYCLE), list(switch))) == score, switch


def test_the_shipped_selector_chose_the_edge_creating_switch():
    """`argmin` over powerset order lands on (2,), which is what created the 2-cycle's other half."""
    assert _select(DECISION_CYCLE, DECISION_CANDIDATES, maximise=False) == (2,)
    assert DECISION_OUTCOMES[(2,)][0] == {CREATED_EDGE}


def test_the_corrected_selector_takes_a_switch_that_creates_nothing():
    """The 2-cycle broken with no lookahead, in isolation and without the dataset.

    The corrected comparator picks the highest-scored member, which here is `(3,)`. By the
    measured outcome table that switch removes the target edge and creates nothing -- so the
    edge whose repair recreated the first is never made, and the cycle does not close.

    The earlier lookahead fixture reached the same *kind* of outcome by a different route,
    selecting `(4,)`: it kept `argmin`'s score-1 tier and refused `(2,)` on the lookahead.
    Both terminate. The difference measured here is that this one costs nothing (a
    lookahead pass over the candidate set).
    """
    selected = _select(DECISION_CYCLE, DECISION_CANDIDATES, maximise=True)
    assert selected == (3,)
    assert DECISION_OUTCOMES[selected][0] == set(), "the selected switch must create nothing"
    assert DECISION_OUTCOMES[selected][1] == {TARGET_EDGE}, "and must still remove the target"


@pytest.mark.parametrize(
    ("label_cycle", "candidates"),
    [
        (DECISION_CYCLE, DECISION_CANDIDATES),
        (CYCLE_STATE_A, [0, 1, 2]),
        (CYCLE_STATE_B, [1, 2, 3]),
        ([3, 0, 0, 0, 3, 3, 3, 0], [0, 7]),
        ([3, 0, 3, 0, 0, 0], [0, 1, 2]),
    ],
)
def test_the_corrected_selector_never_takes_a_zero_scored_switch_when_a_better_one_exists(
    label_cycle, candidates,
):
    """The property that makes the sign the *right* fix rather than a lucky one.

    A score of 0 means "this switch collapses the cycle to one label and deletes the edge".
    Maximising cannot select it while any other member scores above 0; minimising selects it
    whenever it is offered. The five cycles here are the four competitive decisions case 031
    presents at md=3 plus state B -- every competitive decision that case actually has.
    """
    members = _powerset(candidates)
    scores = [_evaluate_cycle(_label_switching(list(label_cycle), list(m))) for m in members]
    chosen = _select(label_cycle, candidates, maximise=True)
    assert _evaluate_cycle(_label_switching(list(label_cycle), list(chosen))) == max(scores)
    if max(scores) > 0:
        assert _evaluate_cycle(_label_switching(list(label_cycle), list(chosen))) > 0


def test_the_selection_is_deterministic():
    """Repeated evaluation on identical input gives one repeatable answer.

    `np.argmax` returns the first index of the maximum and `_powerset` enumerates by ascending
    subset size then input order, so the winner is (highest score, fewest positions,
    lexicographically smallest) -- a strict total order, with no residual choice left to
    enumeration accident.
    """
    assert {_select(DECISION_CYCLE, DECISION_CANDIDATES, maximise=True) for _ in range(5)} == {(3,)}


# --------------------------------------------------------------------------------------
# Part 3 -- the sign itself, at both call sites, so it cannot be re-inverted silently.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "function",
    [post_process_mesh_surgery, _search_and_make_double_switching],
    ids=["post_process_mesh_surgery", "_search_and_make_double_switching"],
)
def test_the_source_selects_by_argmax_at_both_call_sites(function):
    """Both selection sites must maximise `_evaluate_cycle`, which is what its docstring asks.

    Asserted on the source text rather than on behaviour because behaviour needs a mesh and this
    file is dataset-independent by design. Crude, and deliberately so: the failure this guards
    against is someone reading `argmax` as a typo and "fixing" it back, and a source assertion
    with this docstring attached is what stops that from being a silent one-character change.
    `_search_and_make_double_switching` is included because it carries the *same* inversion and
    was corrected in the same commit -- it is not covered by any other test here.
    """
    source = inspect.getsource(function)
    assert "np.argmax(evaluation_of_switch)" in source
    assert "np.argmin(evaluation_of_switch)" not in source


def test_evaluate_cycle_is_reached_only_through_those_two_selection_sites():
    """The scope claim the whole fix rests on, checked instead of asserted in a commit message.

    If `_evaluate_cycle` were consulted anywhere else, correcting the comparator at the two
    selection sites would be a partial fix. It is not: the module's source contains exactly two
    calls to it, and both are the selection expressions above.
    """
    source = inspect.getsource(mesh_surgery)
    calls = source.count("_evaluate_cycle(cycle)")
    assert calls == 2, f"expected exactly 2 selection-site calls to _evaluate_cycle, found {calls}"
