#!/usr/bin/python
from __future__ import annotations

"""Unit tests for adaptive_stopping (CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval, TASR)."""

from agent_utilities.knowledge_graph.retrieval.adaptive_stopping import (
    IterativeStopper,
    StopDecision,
    answer_repeats,
    normalized_answer,
)


def test_normalized_answer_lowercase_punct_whitespace() -> None:
    assert normalized_answer("  The   ANSWER, is:  42! ") == "the answer is 42"
    assert normalized_answer("") == ""
    assert normalized_answer("   ") == ""
    # idempotent
    once = normalized_answer("Hello,  World!!")
    assert normalized_answer(once) == once == "hello world"


def test_answer_repeats_exact_after_normalization() -> None:
    assert answer_repeats("Paris.", "  paris ") is True
    # first round (no prior answer) never repeats
    assert answer_repeats(None, "anything") is False
    # clearly different answers do not repeat
    assert answer_repeats("Paris", "London") is False


def test_answer_repeats_near_via_jaccard() -> None:
    prev = "the capital of france is paris"
    cur = "the capital of france is paris indeed"  # 6/7 token overlap
    assert answer_repeats(prev, cur, similarity_threshold=0.8) is True
    # raise the bar above the actual similarity -> not a repeat
    assert answer_repeats(prev, cur, similarity_threshold=0.99) is False


def test_stop_on_answer_repeat() -> None:
    s = IterativeStopper(max_rounds=10)
    d1 = s.update(answer="Paris", evidence_ids=["a"])
    assert d1 == StopDecision(stop=False, reason="")
    d2 = s.update(answer="paris.", evidence_ids=["b"])
    assert d2.stop is True
    assert d2.reason == "answer_repeat"
    assert s.rounds == 2


def test_coverage_saturation_after_patience() -> None:
    # patience=2: two consecutive rounds with no new evidence -> stop
    s = IterativeStopper(max_rounds=10, min_new_evidence=1, patience=2)
    assert s.update(answer="a1", evidence_ids=["x"]).stop is False  # novel
    # round 2: no new ids (streak=1) but answer differs -> keep going
    assert s.update(answer="a2", evidence_ids=["x"]).stop is False
    # round 3: still no new ids (streak=2 >= patience) -> saturate
    d = s.update(answer="a3", evidence_ids=["x"])
    assert d.stop is True
    assert d.reason == "coverage_saturation"


def test_coverage_saturation_patience_one_immediate() -> None:
    s = IterativeStopper(max_rounds=10, min_new_evidence=1, patience=1)
    assert s.update(answer="a1", evidence_ids=["x"]).stop is False
    d = s.update(answer="a2", evidence_ids=["x"])  # no novel id, streak=1
    assert d.stop is True
    assert d.reason == "coverage_saturation"


def test_saturation_streak_resets_on_new_evidence() -> None:
    s = IterativeStopper(max_rounds=10, min_new_evidence=1, patience=2)
    assert s.update(answer="a1", evidence_ids=["x"]).stop is False
    assert s.update(answer="a2", evidence_ids=["x"]).stop is False  # streak=1
    assert s.update(answer="a3", evidence_ids=["y"]).stop is False  # novel -> reset
    assert s.update(answer="a4", evidence_ids=["y"]).stop is False  # streak=1
    # still not saturated because the streak reset
    assert s.rounds == 4


def test_max_rounds() -> None:
    # productive loop (new answer + new ids each round) only stops at the cap
    s = IterativeStopper(max_rounds=3, min_new_evidence=1, patience=10)
    assert s.update(answer="a1", evidence_ids=["1"]).stop is False
    assert s.update(answer="a2", evidence_ids=["2"]).stop is False
    d = s.update(answer="a3", evidence_ids=["3"])
    assert d.stop is True
    assert d.reason == "max_rounds"
    assert s.rounds == 3


def test_productive_loop_does_not_stop_early() -> None:
    s = IterativeStopper(max_rounds=100, min_new_evidence=1, patience=2)
    for i in range(20):
        d = s.update(answer=f"distinct answer {i}", evidence_ids=[f"ev{i}"])
        assert d.stop is False
    assert s.rounds == 20


def test_answer_repeat_takes_priority_over_saturation() -> None:
    # both an answer repeat AND no new evidence on the same round -> repeat wins
    s = IterativeStopper(max_rounds=10, min_new_evidence=1, patience=1)
    assert s.update(answer="same", evidence_ids=["x"]).stop is False
    d = s.update(answer="same", evidence_ids=["x"])
    assert d.stop is True
    assert d.reason == "answer_repeat"


def test_determinism() -> None:
    def run() -> list[StopDecision]:
        s = IterativeStopper(max_rounds=4, min_new_evidence=1, patience=2)
        return [
            s.update(answer="x", evidence_ids=["a"]),
            s.update(answer="y", evidence_ids=["a"]),
            s.update(answer="z", evidence_ids=["a"]),
        ]

    assert run() == run()


def test_none_prior_never_repeats() -> None:
    # A None prior answer never counts as a repeat (first round can't halt on it).
    assert answer_repeats(None, "") is False
    s = IterativeStopper(max_rounds=5)
    # round 1 stores prev=None; round 2 compares against None -> no repeat
    assert s.update(answer=None, evidence_ids=["a"]).stop is False
    assert s.update(answer="", evidence_ids=["b"]).reason == ""


def test_empty_then_empty_repeats() -> None:
    s = IterativeStopper(max_rounds=5)
    # a real first answer, then two empty answers that normalize equal -> repeat
    assert s.update(answer="something", evidence_ids=["a"]).stop is False
    assert s.update(answer="", evidence_ids=["b"]).stop is False
    d = s.update(answer="   ", evidence_ids=["c"])
    assert d.stop is True
    assert d.reason == "answer_repeat"
