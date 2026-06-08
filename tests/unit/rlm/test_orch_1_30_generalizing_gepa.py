"""CONCEPT:ORCH-1.30 — Generalizing GEPA (held-out split, AgentSpec, held-out selection)."""

from __future__ import annotations

import pytest

from agent_utilities.rlm.gepa import (
    AgentSpec,
    Candidate,
    GEPAInstance,
    select_best_on_heldout,
    split_dataset,
)


def _ds(n: int) -> list[GEPAInstance]:
    return [
        GEPAInstance(
            id=str(i), input_data={"input_text": str(i)}, reference_output=str(i)
        )
        for i in range(n)
    ]


@pytest.mark.concept(id="ORCH-1.30")
def test_split_dataset_disjoint_and_sized():
    fb, dev = split_dataset(_ds(10), dev_fraction=0.3, seed=1)
    assert len(dev) == 3 and len(fb) == 7
    fb_ids = {d.id for d in fb}
    dev_ids = {d.id for d in dev}
    assert fb_ids.isdisjoint(dev_ids)  # held-out is truly held out
    assert fb_ids | dev_ids == {str(i) for i in range(10)}


@pytest.mark.concept(id="ORCH-1.30")
def test_split_dataset_deterministic():
    a = split_dataset(_ds(10), 0.3, seed=42)
    b = split_dataset(_ds(10), 0.3, seed=42)
    assert [d.id for d in a[1]] == [d.id for d in b[1]]


@pytest.mark.concept(id="ORCH-1.30")
def test_split_dataset_zero_fraction_is_noop():
    fb, dev = split_dataset(_ds(5), dev_fraction=0.0)
    assert len(fb) == 5 and dev == []


@pytest.mark.concept(id="ORCH-1.30")
def test_select_best_on_heldout_picks_max_score():
    cands = [
        Candidate(id="a", prompt_text="A", generation=1, scores={}),
        Candidate(id="b", prompt_text="B", generation=2, scores={}),
        Candidate(id="c", prompt_text="C", generation=1, scores={}),
    ]
    # b wins on held-out even if it overfit the feedback minibatch.
    best = select_best_on_heldout(cands, {"a": 0.4, "b": 0.9, "c": 0.4})
    assert best.id == "b"
    # Tie → earlier generation (simpler) wins: a (gen 1) over c (gen 1) — first max; both gen1, both 0.4.
    best_tie = select_best_on_heldout(cands, {"a": 0.5, "c": 0.5, "b": 0.1})
    assert best_tie.id in ("a", "c")


@pytest.mark.concept(id="ORCH-1.30")
def test_agent_spec_as_prompt_grounds_and_enforces_generalization():
    spec = AgentSpec(
        use_cases=["book a meeting", "pay a friend"],
        runtime_grounding=["call_api(app, api, kwargs)", "SUBMIT(answer)"],
        scoring_rule="exact task completion",
        counterfactual_axis="a different app/API set than training",
    )
    text = spec.as_prompt()
    assert "book a meeting" in text and "call_api" in text
    assert "exact task completion" in text
    assert "MUST generalize across" in text and "different app/API set" in text


@pytest.mark.concept(id="ORCH-1.30")
def test_agent_spec_empty_is_blankish():
    assert "Agent Specification" in AgentSpec().as_prompt()
