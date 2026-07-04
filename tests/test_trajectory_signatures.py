#!/usr/bin/python
"""Tests for realized-difficulty trajectory signatures (CONCEPT:AU-AHE.reward.search-task-corpus)."""

from __future__ import annotations

from agent_utilities.graph.training_signals import (
    answer_hit_time,
    mean_answer_hit_time,
    prior_shortcut_rate,
    realized_difficulty,
    solving_cost,
)

ANS = ["Ada Botanist"]


def _search(observation: str = "") -> dict:
    return {"kind": "search", "observation": observation, "model_text": ""}


def _model(text: str = "") -> dict:
    return {"kind": "model", "observation": "", "model_text": text}


def test_solving_cost_counts_retrieval_only():
    traj = {
        "steps": [_search(), _model("thinking"), _search(), _search()],
        "answer_aliases": ANS,
    }
    assert solving_cost([traj]) == 3.0  # 3 retrievals, model turn not counted


def test_answer_hit_time_late_tool_hit():
    traj = {
        "steps": [
            _search("nothing"),
            _search("nothing"),
            _search("found Ada Botanist"),
        ],
        "answer_aliases": ANS,
    }
    assert answer_hit_time(traj) == 3


def test_answer_hit_time_sentinel_when_never_found():
    traj = {"steps": [_search("a"), _search("b")], "answer_aliases": ANS}
    assert answer_hit_time(traj) == 3  # len(steps) + 1


def test_prior_shortcut_rate_flags_answer_before_evidence():
    bound = {
        "steps": [_model("the answer is Ada Botanist"), _search("Ada Botanist record")],
        "answer_aliases": ANS,
    }
    grounded = {
        "steps": [_search("Ada Botanist record"), _model("so it is Ada Botanist")],
        "answer_aliases": ANS,
    }
    assert prior_shortcut_rate([bound]) == 1.0
    assert prior_shortcut_rate([grounded]) == 0.0
    assert prior_shortcut_rate([bound, grounded]) == 0.5


def test_realized_difficulty_gate():
    easy = {  # 1 retrieval, answer in first observation, no prior binding
        "steps": [_search("Ada Botanist")],
        "answer_aliases": ANS,
    }
    hard = {  # many retrievals, late hit, no prior binding
        "steps": [_search(), _search(), _search(), _search("Ada Botanist")],
        "answer_aliases": ANS,
    }
    assert realized_difficulty([easy])["search_heavy"] is False
    metrics = realized_difficulty([hard])
    assert metrics["search_heavy"] is True
    assert metrics["solving_cost"] == 4.0
    assert metrics["prior_shortcut_rate"] == 0.0
    assert mean_answer_hit_time([hard]) == 4.0
