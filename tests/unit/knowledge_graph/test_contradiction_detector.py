"""Explicit node-vs-node contradiction / friction detector (CONCEPT:AU-KG.research.explicit-node-node-contradiction).

Covers the night-shift Critic contract: opposition detection (antonym + negation
flips), topical gating, friction emission for similar-and-opposing pairs only,
similarity-scaled severity, symmetric-pair dedup in all-pairs scans, injected
similarity_fn usage, and full determinism. No LLM, no network.

@pytest.mark.concept("AU-KG.research.explicit-node-node-contradiction")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (
    Claim,
    ContradictionDetector,
    FrictionFinding,
    lexical_similarity,
    opposes,
)

pytestmark = pytest.mark.concept("AU-KG.research.explicit-node-node-contradiction")


# ── opposes() ────────────────────────────────────────────────────────────


def test_opposes_antonym_flip() -> None:
    assert opposes("Caching improves performance", "Caching degrades performance")


def test_opposes_negation_flip() -> None:
    assert opposes("the sky is blue", "the sky is not blue")


def test_opposes_contraction_negation_flip() -> None:
    assert opposes("the build passes", "the build doesn't pass")


def test_opposes_numeric_contradiction() -> None:
    assert opposes(
        "the cache hit rate is 90 percent",
        "the cache hit rate is 40 percent",
    )


def test_opposes_false_for_unrelated() -> None:
    # Different subjects entirely — not a contradiction, just unrelated.
    assert not opposes("the sky is blue", "lithium powers electric vehicles")


def test_opposes_false_for_merely_different() -> None:
    # Same topic, no opposing polarity — these simply add information.
    assert not opposes(
        "Caching improves performance",
        "Caching reduces database load",
    )


def test_opposes_false_when_both_negated() -> None:
    # Both negated = same polarity, not a flip.
    assert not opposes("the sky is not blue", "the sky is never blue")


def test_opposes_false_for_empty() -> None:
    assert not opposes("", "the sky is blue")


# ── lexical_similarity() ─────────────────────────────────────────────────


def test_lexical_similarity_bounds_and_self() -> None:
    s = lexical_similarity(
        "Caching improves performance", "Caching degrades performance"
    )
    assert 0.0 < s <= 1.0
    assert lexical_similarity("a", "a") <= 1.0
    assert lexical_similarity("", "anything") == 0.0


def test_lexical_similarity_unrelated_is_low() -> None:
    assert lexical_similarity("the sky is blue", "lithium powers vehicles") == 0.0


# ── check() ──────────────────────────────────────────────────────────────


def test_check_emits_for_topical_and_opposing() -> None:
    det = ContradictionDetector()
    new = Claim("c2", "Caching degrades performance")
    existing = [Claim("c1", "Caching improves performance")]
    findings = det.check(new, existing)
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, FrictionFinding)
    assert f.new_id == "c2"
    assert f.conflict_id == "c1"
    assert "[FRICTION]" in f.reason
    assert f.severity in {"high", "medium", "low"}


def test_check_none_for_topical_but_agreeing() -> None:
    det = ContradictionDetector()
    new = Claim("c2", "Caching reduces database load and improves performance")
    existing = [Claim("c1", "Caching improves performance")]
    assert det.check(new, existing) == []


def test_check_none_for_non_topical() -> None:
    det = ContradictionDetector()
    new = Claim("c2", "lithium powers electric vehicles")
    existing = [Claim("c1", "the sky is not blue")]
    assert det.check(new, existing) == []


def test_check_ignores_self_id() -> None:
    det = ContradictionDetector()
    new = Claim("same", "the sky is blue")
    existing = [Claim("same", "the sky is not blue")]
    assert det.check(new, existing) == []


def test_check_severity_scales_with_similarity() -> None:
    det = ContradictionDetector(min_similarity=0.05)
    # High: antonym flip on a heavily-shared subject → high topical similarity.
    high = Claim("h", "the new caching layer clearly improves database performance")
    high_existing = [
        Claim("he", "the new caching layer clearly degrades database performance")
    ]
    # Low: the same antonym flip on the shared subject ("caching performance"),
    # but each side padded with distinct words so token Jaccard (and thus
    # similarity) drops below the high case. Antonym opposition needs only a
    # shared topic token, so it still fires.
    low = Claim("l", "caching improves performance across web mobile desktop tiers")
    low_existing = [
        Claim("le", "caching degrades performance under heavy concurrent batch load")
    ]

    f_high = det.check(high, high_existing)[0]
    f_low = det.check(low, low_existing)[0]
    assert f_high.similarity > f_low.similarity
    bands = {"low": 0, "medium": 1, "high": 2}
    assert bands[f_high.severity] >= bands[f_low.severity]


# ── scan() ───────────────────────────────────────────────────────────────


def test_scan_dedups_symmetric_pairs() -> None:
    det = ContradictionDetector()
    claims = [
        Claim("a", "Caching improves performance"),
        Claim("b", "Caching degrades performance"),
    ]
    findings = det.scan(claims)
    # Exactly one finding for the symmetric pair, oriented to the smaller id.
    assert len(findings) == 1
    assert findings[0].new_id == "a"
    assert findings[0].conflict_id == "b"


def test_scan_finds_multiple_and_skips_agreeing() -> None:
    det = ContradictionDetector()
    claims = [
        Claim("a", "Caching improves performance"),
        Claim("b", "Caching degrades performance"),
        Claim("c", "the sky is blue"),
        Claim("d", "the sky is not blue"),
        Claim("e", "lithium powers electric vehicles"),
    ]
    findings = det.scan(claims)
    pairs = {(f.new_id, f.conflict_id) for f in findings}
    assert ("a", "b") in pairs
    assert ("c", "d") in pairs
    # 'e' is unrelated to everything else — no friction involving it.
    assert all("e" not in (f.new_id, f.conflict_id) for f in findings)


# ── injected similarity_fn ───────────────────────────────────────────────


def test_injected_similarity_fn_is_used() -> None:
    calls: list[tuple[str, str]] = []

    def fake_sim(a: str, b: str) -> float:
        calls.append((a, b))
        return 0.95  # force topical gate open regardless of lexical overlap

    det = ContradictionDetector(similarity_fn=fake_sim)
    # Texts that are opposing but lexically sparse; the injected fn supplies topicality.
    new = Claim("c2", "revenue fell")
    existing = [Claim("c1", "revenue rose")]
    findings = det.check(new, existing)
    assert calls, "injected similarity_fn was not called"
    assert len(findings) == 1
    assert findings[0].similarity == pytest.approx(0.95)


def test_injected_similarity_fn_can_close_topical_gate() -> None:
    det = ContradictionDetector(similarity_fn=lambda a, b: 0.0)
    new = Claim("c2", "Caching degrades performance")
    existing = [Claim("c1", "Caching improves performance")]
    # Opposing, but injected similarity says non-topical → gated out.
    assert det.check(new, existing) == []


# ── concrete battery example ─────────────────────────────────────────────


def test_battery_cost_friction_surfaces() -> None:
    det = ContradictionDetector()
    new = Claim("n", "sodium-ion batteries undercut lithium on cost")
    existing = [Claim("e", "lithium cost is the binding constraint on EV adoption")]
    findings = det.check(new, existing)
    assert len(findings) == 1
    assert findings[0].conflict_id == "e"
    assert "[FRICTION]" in findings[0].reason


# ── determinism ──────────────────────────────────────────────────────────


def test_check_is_deterministic() -> None:
    det = ContradictionDetector()
    new = Claim("c3", "Caching degrades performance")
    existing = [
        Claim("c1", "Caching improves performance"),
        Claim("c2", "Caching boosts performance and improves throughput"),
        Claim("z9", "Caching enhances performance"),
    ]
    runs = [
        [
            (f.new_id, f.conflict_id, f.similarity, f.severity)
            for f in det.check(new, existing)
        ]
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs)


def test_scan_is_deterministic_and_sorted() -> None:
    det = ContradictionDetector()
    claims = [
        Claim("a", "Caching improves performance"),
        Claim("b", "Caching degrades performance"),
        Claim("c", "the sky is blue"),
        Claim("d", "the sky is not blue"),
    ]
    runs = [
        [(f.new_id, f.conflict_id, f.similarity) for f in det.scan(claims)]
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs)
    sims = [f.similarity for f in det.scan(claims)]
    assert sims == sorted(sims, reverse=True)
