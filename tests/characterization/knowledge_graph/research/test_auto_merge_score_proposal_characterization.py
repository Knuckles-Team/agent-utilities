"""Characterization tests for ``GovernedAutoMerger.score_proposal`` (CX-AU-09).

CCN 18 at time of writing
(``agent_utilities/knowledge_graph/research/auto_merge.py``). These tests
pin the OBSERVED, black-box behaviour before any decomposition: the exact
per-attribute score weights (name 0.25, goal 0.25, members 0.25 + a further
0.15 for >= 2 members, lead 0.10), the [0,1] clamp, the explicit
``quality_score`` short-circuit (including its own clamp), the fallback to
structural scoring when ``quality_score`` is present but not convertible to
float, and support for both attribute-style and dict-style specs.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``auto_merge.py`` before any refactor commit, and
must not change during the refactor commit that follows.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.auto_merge import GovernedAutoMerger


class _Spec:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_empty_spec_scores_zero() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec()) == 0.0


def test_name_alone_scores_quarter() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec(name="x")) == pytest.approx(0.25)


def test_goal_alone_scores_quarter() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec(goal="y")) == pytest.approx(0.25)


def test_lead_alone_scores_tenth() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec(lead="L")) == pytest.approx(0.10)


def test_single_member_scores_quarter_only() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec(members=["a"])) == pytest.approx(
        0.25
    )


def test_two_or_more_members_gets_the_extra_bonus() -> None:
    assert GovernedAutoMerger.score_proposal(
        _Spec(members=["a", "b"])
    ) == pytest.approx(0.40)
    # a 3rd member does not add further -- the bonus is a flat >=2 threshold.
    assert GovernedAutoMerger.score_proposal(
        _Spec(members=["a", "b", "c"])
    ) == pytest.approx(0.40)


def test_full_strong_proposal_sums_to_one() -> None:
    spec = _Spec(name="Team", goal="Goal", members=["a", "b"], lead="Lead")
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(1.0)


def test_empty_string_attributes_do_not_score() -> None:
    spec = _Spec(name="", goal="", members=[], lead="")
    assert GovernedAutoMerger.score_proposal(spec) == 0.0


def test_dict_spec_uses_the_same_weights() -> None:
    spec = {"name": "n", "goal": "g", "members": ["a", "b"], "lead": "L"}
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(1.0)


def test_explicit_quality_score_wins_over_structural() -> None:
    spec = _Spec(name="x", goal="y", members=["a", "b"], lead="L", quality_score=0.42)
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(0.42)


def test_explicit_quality_score_is_clamped_to_unit_interval() -> None:
    assert GovernedAutoMerger.score_proposal(_Spec(quality_score=5.0)) == 1.0
    assert GovernedAutoMerger.score_proposal(_Spec(quality_score=-3.0)) == 0.0


def test_dict_spec_explicit_quality_score_wins() -> None:
    spec = {"quality_score": 0.77, "name": "n"}
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(0.77)


def test_non_numeric_quality_score_falls_back_to_structural_scoring() -> None:
    # OBSERVED: a quality_score present but not float()-convertible does NOT
    # raise and does NOT count as "no explicit score" for None-checks -- it
    # falls through to the structural heuristic computed from scratch.
    spec = _Spec(quality_score="not-a-number", name="x", goal="y")
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(0.50)


def test_none_quality_score_falls_back_to_structural_scoring() -> None:
    spec = _Spec(quality_score=None, name="x")
    assert GovernedAutoMerger.score_proposal(spec) == pytest.approx(0.25)
