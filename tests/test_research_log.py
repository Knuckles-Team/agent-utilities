from __future__ import annotations

"""Tests for the research-craft discipline primitives (CONCEPT:AU-AHE.evaluation.disconfirming-evidence-log)."""

from dataclasses import dataclass, field

from agent_utilities.harness.research_log import (
    BeliefEntry,
    FailureCase,
    FailureTriage,
    ResearchLog,
)


# ---------------------------------------------------------------------------
# Fake corpus shapes (no LLM / network)
# ---------------------------------------------------------------------------


@dataclass
class StubCluster:
    label: str
    root_cause_summary: str = ""
    task_ids: list[str] = field(default_factory=list)


@dataclass
class StubCorpus:
    failure_clusters: list[StubCluster] = field(default_factory=list)
    entries: list[object] = field(default_factory=list)


@dataclass
class StubEntry:
    task_id: str
    pass_fail: bool
    root_cause: str | None = None
    content: str = ""


# ---------------------------------------------------------------------------
# FailureTriage
# ---------------------------------------------------------------------------


def test_add_failure_returns_case() -> None:
    triage = FailureTriage()
    case = triage.add_failure("c1", "boom", "timeout", transcript="t")
    assert isinstance(case, FailureCase)
    assert case.case_id == "c1"
    assert case.pile == "timeout"
    assert case.transcript == "t"


def test_piles_ordered_biggest_first() -> None:
    triage = FailureTriage()
    for i in range(3):
        triage.add_failure(f"a{i}", "x", "timeout")
    for i in range(2):
        triage.add_failure(f"b{i}", "x", "auth")
    triage.add_failure("c0", "x", "parse")

    piles = triage.piles()
    assert piles == {"timeout": 3, "auth": 2, "parse": 1}
    # biggest first
    assert list(piles)[0] == "timeout"


def test_piles_tie_break_deterministic() -> None:
    triage = FailureTriage()
    triage.add_failure("1", "x", "zeta")
    triage.add_failure("2", "x", "alpha")
    # equal counts -> alphabetical label order
    assert list(triage.piles()) == ["alpha", "zeta"]


def test_biggest_pile() -> None:
    triage = FailureTriage()
    triage.add_failure("a0", "x", "timeout")
    triage.add_failure("a1", "x", "timeout")
    triage.add_failure("b0", "x", "auth")

    result = triage.biggest_pile()
    assert result is not None
    label, cases = result
    assert label == "timeout"
    assert {c.case_id for c in cases} == {"a0", "a1"}


def test_biggest_pile_empty() -> None:
    assert FailureTriage().biggest_pile() is None


def test_sample_k() -> None:
    triage = FailureTriage()
    for i in range(5):
        triage.add_failure(f"t{i}", "x", "timeout", transcript=f"log{i}")
    triage.add_failure("other", "x", "auth")

    sampled = triage.sample("timeout", k=3)
    assert len(sampled) == 3
    # insertion order preserved
    assert [c.case_id for c in sampled] == ["t0", "t1", "t2"]
    assert all(c.pile == "timeout" for c in sampled)


def test_sample_k_zero_and_missing_pile() -> None:
    triage = FailureTriage()
    triage.add_failure("t0", "x", "timeout")
    assert triage.sample("timeout", k=0) == []
    assert triage.sample("nonexistent") == []


def test_from_evidence_corpus_clusters() -> None:
    corpus = StubCorpus(
        failure_clusters=[
            StubCluster(
                label="timeout", root_cause_summary="hung", task_ids=["a", "b"]
            ),
            StubCluster(label="auth", root_cause_summary="401", task_ids=["c"]),
        ]
    )
    triage = FailureTriage()
    count = triage.from_evidence_corpus(corpus)

    assert count == 3
    assert triage.piles() == {"timeout": 2, "auth": 1}
    biggest = triage.biggest_pile()
    assert biggest is not None
    assert biggest[0] == "timeout"
    # summary carried from cluster
    assert all(c.summary == "hung" for c in biggest[1])


def test_from_evidence_corpus_entries_fallback() -> None:
    corpus = StubCorpus(
        entries=[
            StubEntry(task_id="t1", pass_fail=True, root_cause=None),
            StubEntry(
                task_id="t2", pass_fail=False, root_cause="parse error", content="trace"
            ),
            StubEntry(task_id="t3", pass_fail=False, root_cause="parse error"),
        ]
    )
    triage = FailureTriage()
    count = triage.from_evidence_corpus(corpus)

    # only the two failures ingested
    assert count == 2
    assert triage.piles() == {"parse error": 2}
    sampled = triage.sample("parse error")
    assert sampled[0].transcript == "trace"


def test_from_evidence_corpus_dict_shape() -> None:
    corpus = {
        "failure_clusters": [
            {"label": "io", "root_cause_summary": "disk full", "task_ids": ["x", "y"]}
        ]
    }
    triage = FailureTriage()
    assert triage.from_evidence_corpus(corpus) == 2
    assert triage.piles() == {"io": 2}


def test_from_evidence_corpus_empty_graceful() -> None:
    assert FailureTriage().from_evidence_corpus(StubCorpus()) == 0
    assert FailureTriage().from_evidence_corpus(None) == 0


# ---------------------------------------------------------------------------
# ResearchLog
# ---------------------------------------------------------------------------


def test_record_supports() -> None:
    log = ResearchLog()
    entry = log.record("H1", "observed X", supports=True, timestamp="t0")
    assert isinstance(entry, BeliefEntry)
    assert entry.supports is True
    assert entry.timestamp == "t0"


def test_disconfirming_collects_refuting() -> None:
    log = ResearchLog()
    log.record("H1", "for", supports=True)
    log.record("H1", "against", supports=False)
    log.record("H2", "against2", supports=False)

    disc = log.disconfirming()
    assert len(disc) == 2
    assert all(not e.supports for e in disc)


def test_disconfirming_filtered_by_hypothesis() -> None:
    log = ResearchLog()
    log.record("H1", "against1", supports=False)
    log.record("H2", "against2", supports=False)
    log.record("H1", "for", supports=True)

    disc = log.disconfirming("H1")
    assert len(disc) == 1
    assert disc[0].evidence == "against1"


def test_balance_tally() -> None:
    log = ResearchLog()
    log.record("H1", "a", supports=True)
    log.record("H1", "b", supports=True)
    log.record("H1", "c", supports=False)
    log.record("H2", "d", supports=True)

    assert log.balance("H1") == {"supports": 2, "refutes": 1}
    assert log.balance("H2") == {"supports": 1, "refutes": 0}
    assert log.balance("missing") == {"supports": 0, "refutes": 0}


def test_contested_only_dual_evidence() -> None:
    log = ResearchLog()
    log.record("contested", "for", supports=True)
    log.record("contested", "against", supports=False)
    log.record("only_for", "for", supports=True)
    log.record("only_against", "against", supports=False)

    assert log.contested() == ["contested"]


def test_contested_sorted_deterministic() -> None:
    log = ResearchLog()
    for h in ("zeta", "alpha"):
        log.record(h, "for", supports=True)
        log.record(h, "against", supports=False)
    assert log.contested() == ["alpha", "zeta"]


def test_determinism_repeated_calls() -> None:
    triage = FailureTriage()
    triage.add_failure("a", "x", "p1")
    triage.add_failure("b", "x", "p2")
    triage.add_failure("c", "x", "p1")
    assert triage.piles() == triage.piles()

    log = ResearchLog()
    log.record("H", "a", supports=True)
    log.record("H", "b", supports=False)
    assert log.balance("H") == log.balance("H")
    assert log.contested() == log.contested()
