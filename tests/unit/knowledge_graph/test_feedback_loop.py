"""Human-correction feedback loop tests (CONCEPT:KG-2.8).

Covers FeedbackService (outcome/rule/eval), governance-rule application at
retrieval time, and the KG-backed eval corpus — all with fakes, no live engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService
from agent_utilities.knowledge_graph.retrieval.governance_rules import (
    apply_governance_rules,
)


@dataclass
class FakeDesignation:
    id: str
    score: float
    capabilities: set = field(default_factory=set)


class FakeIndex:
    def __init__(self):
        self.rewards = {}

    def record_outcome(self, id, success=None, reward=None, alpha=0.3):
        self.rewards[id] = reward if reward is not None else (1.0 if success else 0.0)
        return self.rewards[id]


from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


# ── governance rules (the "rules change behaviour" link) ──────────────────────
def test_forbid_rule_drops_designation():
    desigs = [FakeDesignation("tool:bad", 0.9), FakeDesignation("tool:good", 0.5)]
    rules = [{"kind": "forbid", "target": "tool:bad"}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["tool:good"]


def test_prefer_rule_reranks():
    desigs = [FakeDesignation("tool:a", 0.5), FakeDesignation("tool:b", 0.6)]
    rules = [{"kind": "prefer", "target": "tool:a", "weight": 0.5}]
    out = apply_governance_rules(desigs, rules)
    assert out[0].id == "tool:a"  # boosted above b


def test_no_rules_is_passthrough():
    desigs = [FakeDesignation("x", 0.1)]
    assert apply_governance_rules(desigs, None) == desigs


# ── FeedbackService: outcome ──────────────────────────────────────────────────
def test_outcome_correction_updates_reward():
    idx = FakeIndex()
    svc = FeedbackService(capability_index=idx)
    res = svc.record_correction("outcome", "tool:x", reward=0.1, reason="bad call")
    assert res.applied
    assert idx.rewards["tool:x"] == 0.1


# ── FeedbackService: rule → persisted + then enforced at retrieval ────────────
def test_rule_correction_persists_and_then_enforced():
    backend = FakeBackend()
    svc = FeedbackService(backend=backend)
    res = svc.record_correction(
        "rule",
        "tool:bad",
        reason="never use in public",
        rule_scope="source",
        rule_kind="forbid",
    )
    assert res.applied
    # A correction node + a source_rule node were written, linked by 'corrects'.
    types = {p["type"] for p in backend.nodes.values()}
    assert "correction" in types and "source_rule" in types
    assert any(rel == "corrects" for _, _, rel in backend.edges)

    # Build the rule dict the way load_active_rules would, and prove it bites.
    rule = next(p for p in backend.nodes.values() if p["type"] == "source_rule")
    rules = [{"kind": rule["kind"], "target": rule["target"]}]
    desigs = [FakeDesignation("tool:bad", 0.99), FakeDesignation("tool:ok", 0.4)]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["tool:ok"]


# ── FeedbackService: eval → corpus case ───────────────────────────────────────
def test_eval_correction_adds_corpus_case():
    corpus = EvalCorpus()
    svc = FeedbackService(eval_corpus=corpus)
    res = svc.record_correction(
        "eval",
        "what is our refund policy?",
        corrected_value="30-day refund",
        reason="agent said 14",
    )
    assert res.applied
    assert corpus.size == 1


def test_unknown_correction_type_is_rejected():
    svc = FeedbackService()
    res = svc.record_correction("nonsense", "x")
    assert not res.applied


# ── FeedbackService: selective_erasure live path (CONCEPT:KG-2.276) ───────────
def test_selective_erasure_live_path_forgets_superseded_rewards():
    """LIVE-PATH: a `selective_erasure` correction dispatched through
    `record_correction` reaches the real `CapabilityIndex` and forgets only the
    named superseded designations (RQGM selective erasure)."""
    from agent_utilities.knowledge_graph.retrieval.capability_index import (
        CapabilityIndex,
    )

    idx = CapabilityIndex(dim=4, prefer_backend="numpy")
    idx.add("tool:old_a", [1.0, 0.0, 0.0, 0.0], ["web"])
    idx.add("tool:old_b", [0.0, 1.0, 0.0, 0.0], ["web"])
    idx.add("tool:keep", [0.0, 0.0, 1.0, 0.0], ["web"])
    for nid in ("tool:old_a", "tool:old_b", "tool:keep"):
        idx.record_outcome(nid, reward=1.0)

    svc = FeedbackService(capability_index=idx)
    res = svc.record_correction(
        "selective_erasure",
        "tool:old_a",
        corrected_value=["tool:old_b"],
        reason="capability redeployed",
    )
    assert res.applied
    assert "erased 2" in res.detail
    # Exactly the superseded ids were forgotten; the unrelated one is preserved.
    assert idx.reward_of("tool:old_a") == 0.5
    assert idx.reward_of("tool:old_b") == 0.5
    assert idx.reward_of("tool:keep") > 0.5


# ── FeedbackService: reads_avoided loop (CONCEPT:AHE-3.61) ────────────────────
def test_reads_avoided_full_replacement_rewards_high_and_grades():
    idx = FakeIndex()
    corpus = EvalCorpus()
    svc = FeedbackService(capability_index=idx, eval_corpus=corpus)
    res = svc.record_correction(
        "reads_avoided",
        "code_context:how:run_agent",
        corrected_value={
            "reads_avoided": True,
            "files_read": 0,
            "correct": True,
            "query": "how does run_agent work",
        },
    )
    assert res.applied
    assert idx.rewards["code_context:how:run_agent"] == 1.0  # replaced the read
    assert corpus.size == 1  # graded case persisted for regression


def test_reads_avoided_partial_and_wrong_rewards():
    idx = FakeIndex()
    svc = FeedbackService(capability_index=idx)
    # helped but files still read -> 0.7
    svc.record_reads_avoided("cap:a", reads_avoided=True, files_read=2, correct=True)
    assert idx.rewards["cap:a"] == 0.7
    # read anyway -> 0.3
    svc.record_reads_avoided("cap:b", reads_avoided=False, files_read=3, correct=True)
    assert idx.rewards["cap:b"] == 0.3
    # wrong answer -> 0.0 regardless
    svc.record_reads_avoided("cap:c", reads_avoided=True, files_read=0, correct=False)
    assert idx.rewards["cap:c"] == 0.0


def test_reads_avoided_accepts_json_string_payload():
    idx = FakeIndex()
    svc = FeedbackService(capability_index=idx)
    res = svc.record_correction(
        "reads_avoided",
        "cap:json",
        corrected_value='{"reads_avoided": true, "files_read": 0, "correct": true}',
    )
    assert res.applied
    assert idx.rewards["cap:json"] == 1.0


# ── FeedbackService: universal action_outcome (CONCEPT:AHE-3.62) ──────────────
def test_action_outcome_success_rewards_high_and_grades():
    idx = FakeIndex()
    corpus = EvalCorpus()
    svc = FeedbackService(capability_index=idx, eval_corpus=corpus)
    res = svc.record_correction(
        "action_outcome",
        "deploy:graph-os",
        corrected_value={
            "success": True,
            "expected": "healthy",
            "query": "deploy graph-os",
        },
    )
    assert res.applied
    assert idx.rewards["deploy:graph-os"] == 1.0
    assert corpus.size == 1  # expected + query -> graded case


def test_action_outcome_failure_and_explicit_reward():
    idx = FakeIndex()
    svc = FeedbackService(capability_index=idx)
    svc.record_action_outcome("a", success=False)
    assert idx.rewards["a"] == 0.0
    svc.record_action_outcome("b", reward=0.6)  # explicit reward overrides success
    assert idx.rewards["b"] == 0.6
    svc.record_action_outcome("c", reward=5.0)  # clamped to [0,1]
    assert idx.rewards["c"] == 1.0


def test_action_outcome_accepts_json_string_payload():
    idx = FakeIndex()
    svc = FeedbackService(capability_index=idx)
    res = svc.record_correction(
        "action_outcome",
        "route:qwen-local",
        corrected_value='{"success": true, "reward": 0.8}',
    )
    assert res.applied
    assert idx.rewards["route:qwen-local"] == 0.8


# ── EvalCorpus run ────────────────────────────────────────────────────────────
def test_eval_corpus_runs_cases():
    corpus = EvalCorpus()
    corpus.add_case("ping", "pong")
    # actual matches expected -> should pass
    results = corpus.run_corpus(actual_output_fn=lambda case: "pong")
    assert len(results) == 1
    assert results[0].passed
