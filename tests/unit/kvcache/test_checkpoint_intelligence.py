"""Wiring tests for the KV-checkpoint intelligence layer.

CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring +
CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility +
CONCEPT:AU-ORCH.optimization.checkpoint-recommendation-surface.

These are deliberately **wiring** tests rather than existence tests. The three things
that had to be proven, and are:

1. A checkpoint is genuinely **taken** when the scorers cross the threshold on the
   autonomous path — and genuinely **not** taken below it, with the payload callable
   never even invoked (:func:`test_autonomous_path_takes_a_ram_checkpoint_when_scored_worthy`
   / :func:`test_autonomous_path_declines_and_never_materializes_the_payload`).
2. The disk gate genuinely **refuses** without eligibility — the durable store is never
   called for an agent- or system-initiated persistence, and a checkpoint that has been
   sitting in RAM does not thereby acquire consent
   (:func:`test_agent_initiated_persistence_never_reaches_the_durable_store`,
   :func:`test_ram_residency_is_not_disk_consent`).
3. A user-invoked checkpoint works **end to end over the live MCP action**, including
   the promotion and the "why was this persisted?" explanation
   (:func:`test_user_invoked_checkpoint_end_to_end_over_the_mcp_action`).

Plus the fourth wiring claim that is easy to get wrong: the advisory actually reaches
the model. :func:`test_advisory_reaches_the_model_through_create_agent` runs a real
agent built by the standard factory against ``TestModel`` and asserts the advisory text
lands in the outgoing ``ModelRequest.instructions``.

Everything here is offline and deterministic — no engine, no network, no LLM.
"""

from __future__ import annotations

import base64
import json
import logging

import pytest

from agent_utilities.kvcache.checkpoint import (
    CrossTenantCheckpointError,
    KVCheckpointError,
    KVCheckpointKey,
    KVCheckpointRecord,
)
from agent_utilities.kvcache.eligibility import (
    AlwaysDenyEligibility,
    EligibilityDecision,
    OperatorGrantEligibility,
    PersistenceRequest,
    get_persistence_eligibility_gate,
    set_persistence_eligibility_gate,
)
from agent_utilities.kvcache.rebuild_cost import (
    RebuildCostInputs,
    estimate_rebuild_cost,
)
from agent_utilities.kvcache.tiering import (
    RAMCheckpointStore,
    TieredCheckpointManager,
    prefix_digest,
)
from agent_utilities.kvcache.worthiness import (
    CheckpointAdvisor,
    CheckpointObservation,
    CheckpointScorerRegistry,
    CheckpointSignal,
    CheckpointTier,
    ContradictionScorer,
    DiskPromotionRule,
    ModelSelfReport,
    PredictedReuseScorer,
    RebuildCostScorer,
    build_default_scorers,
    clear_checkpoint_advisory,
    publish_checkpoint_advisory,
    render_checkpoint_advisory_instructions,
)

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _key(tenant: str = "t1", prefix: str = "ctx") -> KVCheckpointKey:
    return KVCheckpointKey(
        model_identity="qwen3.6-27b",
        quantization="fp16",
        serving_engine="vllm",
        engine_version="0.9.0",
        prefix_digest=prefix_digest(prefix),
        tenant=tenant,
        policy_version="v1",
    )


def _strong_observation(**overrides) -> CheckpointObservation:
    """An observation every default scorer likes — expensive, reused, converged, clean."""
    fields = {
        "run_id": "run-1",
        "tenant": "t1",
        "point": "post-plan",
        "rebuild": RebuildCostInputs(
            prompt_tokens=55_000, completion_tokens=3_000, tool_calls=19,
            retrievals=14, wall_time_s=110.0,
        ),
        "sibling_task_count": 5,
        "queued_task_count": 3,
        "retrieved_items": 20,
        "novel_items": 1,
        "claim_count": 12,
        "evidence_span_count": 15,
        "unresolved_contradictions": 0,
        "high_severity_contradictions": 0,
        "context_rewrites": 0,
        "evicted_items": 0,
        "turns_since_context_change": 6,
        "phase": "plan",
        "phase_completed": True,
    }
    fields.update(overrides)
    return CheckpointObservation(**fields)


def _weak_observation() -> CheckpointObservation:
    """A cheap, still-exploring, churning context — nothing worth freezing."""
    return CheckpointObservation(
        run_id="run-2",
        tenant="t1",
        rebuild=RebuildCostInputs(prompt_tokens=400, tool_calls=0, retrievals=1,
                                  wall_time_s=1.5),
        sibling_task_count=0,
        queued_task_count=0,
        retrieved_items=10,
        novel_items=10,
        claim_count=4,
        evidence_span_count=0,
        unresolved_contradictions=2,
        high_severity_contradictions=0,
        context_rewrites=4,
        evicted_items=3,
        turns_since_context_change=0,
        phase="execute",
        phase_completed=False,
    )


class _FakeDiskStore:
    """Stands in for :class:`KVCheckpointStore` — records every durable write."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.records: dict[str, KVCheckpointRecord] = {}

    def create_checkpoint(self, data, *, key, run_id, point, session=None,
                          provenance=None):
        self.created.append(
            {
                "data": data, "key": key, "run_id": run_id, "point": point,
                "provenance": provenance or {},
            }
        )
        record = KVCheckpointRecord(
            checkpoint_id=key.checkpoint_id, key=key,
            blob_id=f"kvblob:{key.tenant}:d", digest="d", run_id=run_id, point=point,
            size_bytes=len(data), created_at="2026-07-31T00:00:00Z",
            provenance=provenance or {},
        )
        self.records[record.checkpoint_id] = record
        return record

    def get_checkpoint(self, checkpoint_id, *, requesting_tenant,
                       current_policy_version=None):
        record = self.records.get(checkpoint_id)
        if record is None:
            raise KVCheckpointError(f"no checkpoint {checkpoint_id}")
        if record.key.tenant != requesting_tenant:
            raise CrossTenantCheckpointError("cross tenant")
        return record


@pytest.fixture
def manager() -> TieredCheckpointManager:
    return TieredCheckpointManager(
        ram_store=RAMCheckpointStore(), disk_store=_FakeDiskStore()
    )


@pytest.fixture(autouse=True)
def _isolate_advisory():
    """Never leak a published advisory between tests."""
    clear_checkpoint_advisory()
    yield
    clear_checkpoint_advisory()


@pytest.fixture(autouse=True)
def _restore_default_gate():
    """A test that installs a custom eligibility gate must not affect its neighbours."""
    original = get_persistence_eligibility_gate()
    yield
    set_persistence_eligibility_gate(original)


# ---------------------------------------------------------------------------
# 1. The scorer framework is genuinely pluggable
# ---------------------------------------------------------------------------


class _AlwaysHigh:
    name = "always_high"
    weight = 1.0

    def score(self, observation):
        return CheckpointSignal(
            name=self.name, value=1.0, weight=self.weight, rationale="always"
        )


class _Explodes:
    name = "explodes"
    weight = 1.0

    def score(self, observation):
        raise RuntimeError("scorer blew up")


def test_a_custom_scorer_participates_in_the_aggregate():
    """Registering a scorer changes the verdict without touching the advisor."""
    registry = CheckpointScorerRegistry([])
    advisor = CheckpointAdvisor(registry=registry)
    assert advisor.evaluate(_weak_observation()).recommended_tier is CheckpointTier.NONE

    registry.register(_AlwaysHigh())
    result = advisor.evaluate(_weak_observation())
    assert result.score == pytest.approx(1.0)
    assert result.recommended_tier is CheckpointTier.RAM


def test_unregistering_a_default_removes_its_influence():
    registry = CheckpointScorerRegistry(build_default_scorers())
    assert "contradictions" in registry
    removed = registry.unregister("contradictions")
    assert removed is not None and "contradictions" not in registry
    # With the veto gone, the same contradicted observation no longer vetoes.
    contradicted = _strong_observation(high_severity_contradictions=3)
    result = CheckpointAdvisor(registry=registry).evaluate(contradicted)
    assert result.recommended_tier is not CheckpointTier.NONE


def test_duplicate_registration_is_refused_unless_replacing():
    registry = CheckpointScorerRegistry([_AlwaysHigh()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_AlwaysHigh())
    registry.register(_AlwaysHigh(), replace=True)
    assert len(registry) == 1


def test_a_non_conforming_object_is_refused_at_registration():
    with pytest.raises(TypeError, match="name/weight/score"):
        CheckpointScorerRegistry().register(object())  # type: ignore[arg-type]


def test_a_raising_scorer_becomes_a_loud_abstention(caplog):
    """A third-party scorer must not break the decision path — but must not be silent."""
    registry = CheckpointScorerRegistry([_Explodes(), _AlwaysHigh()])
    with caplog.at_level(logging.WARNING):
        result = CheckpointAdvisor(registry=registry).evaluate(_weak_observation())
    assert "explodes" in result.abstained
    assert result.score == pytest.approx(1.0)  # the healthy scorer still counts
    # The cause is preserved in the message text. It cannot be asserted via
    # ``record.exc_info``: ``core/log_privacy.install_log_privacy_boundary`` deliberately
    # strips exc_info from every ``agent_utilities.*`` record (tracebacks carry host
    # filesystem paths), so the exception TYPE and MESSAGE in the log line are the
    # cause-preservation contract for this package.
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "explodes" in m and "RuntimeError" in m and "scorer blew up" in m
        for m in messages
    )


def test_abstention_does_not_dilute_the_aggregate():
    """An abstaining scorer contributes nothing — it does not drag the score to zero."""
    registry = CheckpointScorerRegistry([_AlwaysHigh(), PredictedReuseScorer()])
    result = CheckpointAdvisor(registry=registry).evaluate(
        CheckpointObservation()  # no reuse counts -> PredictedReuseScorer abstains
    )
    assert "predicted_reuse" in result.abstained
    assert result.score == pytest.approx(1.0)


def test_all_scorers_abstaining_yields_no_recommendation():
    result = CheckpointAdvisor().evaluate(CheckpointObservation())
    assert result.recommended_tier is CheckpointTier.NONE
    assert any("every scorer abstained" in b for b in result.blockers)


# ---------------------------------------------------------------------------
# 2. The default signal set behaves as specified
# ---------------------------------------------------------------------------


def test_unmeasured_is_not_zero_for_rebuild_cost():
    """`None` (not measured) abstains; `0` (measured zero) scores. The distinction is
    the whole reason the estimator refuses to guess."""
    assert estimate_rebuild_cost(RebuildCostInputs()).known is False
    measured_zero = estimate_rebuild_cost(
        RebuildCostInputs(prompt_tokens=0, tool_calls=0)
    )
    assert measured_zero.known is True
    assert measured_zero.normalized == pytest.approx(0.0)

    scorer = RebuildCostScorer()
    assert scorer.score(CheckpointObservation()).abstained is True
    assert (
        scorer.score(
            CheckpointObservation(rebuild=RebuildCostInputs(prompt_tokens=60_000))
        ).value
        == pytest.approx(1.0)
    )


def test_rebuild_cost_leaves_usd_none_for_an_unpriced_model():
    """An unknown model is a clean abstain on price, never a fabricated 0.0."""
    estimate = estimate_rebuild_cost(
        RebuildCostInputs(prompt_tokens=1000, model="definitely-not-a-real-model")
    )
    assert estimate.known is True
    assert estimate.usd is None


def test_predicted_reuse_abstains_rather_than_inventing_a_number():
    """This platform has no reuse model; the scorer must say so, not guess."""
    signal = PredictedReuseScorer().score(CheckpointObservation())
    assert signal.abstained is True
    assert "no predicted-reuse model" in signal.rationale


def test_high_severity_contradiction_vetoes_an_otherwise_perfect_moment():
    """The strongest negative gate: no amount of rebuild cost overrides it."""
    result = CheckpointAdvisor().evaluate(
        _strong_observation(unresolved_contradictions=1, high_severity_contradictions=1)
    )
    assert result.recommended_tier is CheckpointTier.NONE
    assert any("VETO" in b for b in result.blockers)


def test_contradiction_scorer_abstains_when_nothing_was_measured():
    assert ContradictionScorer().score(CheckpointObservation()).abstained is True


def test_rising_novelty_reads_as_still_exploring():
    """All-new retrievals => saturation 0 => a bad moment to freeze."""
    signal = next(
        s
        for s in CheckpointAdvisor()
        .evaluate(CheckpointObservation(retrieved_items=10, novel_items=10))
        .signals
        if s.name == "retrieval_saturation"
    )
    assert signal.value == pytest.approx(0.0)


def test_model_self_report_alone_cannot_carry_a_recommendation():
    """A model claiming understanding is evidence, not proof."""
    result = CheckpointAdvisor().evaluate(
        CheckpointObservation(
            model_self_report=ModelSelfReport(
                subject="the auth flow", confidence=1.0, rationale="I read it all"
            )
        )
    )
    assert result.recommended_tier is CheckpointTier.NONE
    assert any("only non-abstaining signal" in b for b in result.blockers)


def test_model_self_report_counts_when_corroborated():
    result = CheckpointAdvisor().evaluate(
        _strong_observation(
            model_self_report=ModelSelfReport(subject="the auth flow", confidence=0.9)
        )
    )
    assert "model_self_report" not in result.abstained
    assert result.recommended_tier is not CheckpointTier.NONE


# ---------------------------------------------------------------------------
# 3. The disk-promotion rule is a materially higher, inspectable bar
# ---------------------------------------------------------------------------


def test_disk_rule_fails_on_an_abstention_not_just_a_low_value():
    """'We don't know' must not satisfy 'it's high'."""
    # Everything strong EXCEPT predicted reuse, which abstains for lack of counts.
    observation = _strong_observation(sibling_task_count=None, queued_task_count=None)
    result = CheckpointAdvisor().evaluate(observation)
    assert result.recommended_tier is CheckpointTier.RAM
    assert result.disk_verdict is not None
    assert not result.disk_verdict.satisfied
    assert any(
        "predicted_reuse" in f and "abstained" in f for f in result.disk_verdict.failures
    )
    assert any("disk not recommended" in b for b in result.blockers)


def test_disk_rule_names_exactly_which_requirement_failed():
    rule = DiskPromotionRule(
        min_aggregate=0.0, required_signals={"context_stability": 0.99}
    )
    result = CheckpointAdvisor(disk_rule=rule).evaluate(
        _strong_observation(context_rewrites=5, evicted_items=5,
                            turns_since_context_change=0)
    )
    assert result.disk_verdict is not None and not result.disk_verdict.satisfied
    assert any("context_stability" in f for f in result.disk_verdict.failures)


def test_a_fully_satisfying_moment_recommends_disk():
    result = CheckpointAdvisor().evaluate(_strong_observation())
    assert result.recommended_tier is CheckpointTier.DISK
    assert result.disk_verdict is not None and result.disk_verdict.satisfied


# ---------------------------------------------------------------------------
# 4. WIRING — the system-autonomous path actually takes a checkpoint
# ---------------------------------------------------------------------------


def test_autonomous_path_takes_a_ram_checkpoint_when_scored_worthy(manager):
    """Crossing the threshold must actually STORE bytes, not merely return a verdict."""
    key = _key()
    outcome = manager.observe(
        _strong_observation(), key=key, payload=b"kv-bytes", run_id="run-1",
        point="post-plan",
    )
    assert outcome.taken is True
    assert outcome.trigger == "system"
    # The proof: the bytes are retrievable from the RAM tier afterwards.
    record, data = manager.ram_store.fetch(
        outcome.checkpoint_id, requesting_tenant="t1"
    )
    assert data == b"kv-bytes"
    assert record.trigger == "system"
    assert record.recommendation is not None


def test_autonomous_path_declines_and_never_materializes_the_payload(manager):
    """Below threshold: nothing stored AND the payload callable is never invoked, which
    is what makes this path cheap enough to call often."""
    calls: list[int] = []

    def _payload() -> bytes:
        calls.append(1)
        return b"expensive-serialization"

    outcome = manager.observe(_weak_observation(), key=_key(), payload=_payload)
    assert outcome.taken is False
    assert calls == []
    assert manager.ram_store.stats().entries == 0
    assert "not checkpoint-worthy" in outcome.reason


def test_autonomous_path_does_not_advise_the_model_it_already_acted(manager):
    """`observe` acts; it must not ALSO tell the model to checkpoint, or a model reading
    the advisory would take a duplicate of what the system just took."""
    outcome = manager.observe(_strong_observation(), key=_key(), payload=b"kv")
    assert outcome.taken is True
    assert render_checkpoint_advisory_instructions() == ""


def test_autonomous_path_cannot_reach_disk_on_its_own(manager):
    """A system-scored DISK verdict still produces only a RAM checkpoint plus a
    recorded refusal — visible evidence the system wanted to persist and could not."""
    outcome = manager.observe(_strong_observation(), key=_key(), payload=b"kv")
    assert outcome.recommendation is not None
    assert outcome.recommendation.recommended_tier is CheckpointTier.DISK
    assert outcome.tier is CheckpointTier.RAM
    assert outcome.eligibility is not None and outcome.eligibility.permitted is False
    assert manager.disk_store.created == []


# ---------------------------------------------------------------------------
# 5. WIRING — the disk gate genuinely refuses
# ---------------------------------------------------------------------------


def test_agent_initiated_persistence_never_reaches_the_durable_store(manager):
    outcome = manager.checkpoint_now(
        b"kv", key=_key(), trigger="agent", persist=True, operator_grant=True
    )
    # Even claiming an operator grant, an agent initiator is refused by the default gate.
    assert outcome.tier is CheckpointTier.RAM
    assert outcome.eligibility is not None and not outcome.eligibility.permitted
    assert manager.disk_store.created == []


def test_user_grant_persists_and_records_why(manager):
    outcome = manager.checkpoint_now(
        b"kv", key=_key(), run_id="run-1", point="post-plan", trigger="user",
        persist=True, operator_grant=True, observation=_strong_observation(),
    )
    assert outcome.tier is CheckpointTier.DISK
    assert len(manager.disk_store.created) == 1
    provenance = manager.disk_store.created[0]["provenance"]
    # The "why was this persisted?" record travels with the durable checkpoint.
    assert provenance["trigger"] == "user"
    assert provenance["eligibility_gate"] == "operator-grant-default"
    assert provenance["worthiness_score"] > 0
    assert provenance["worthiness_drivers"]
    assert "data_residency_region" in provenance["eligibility_unresolved"]


def test_persist_without_a_grant_is_refused_even_for_a_user(manager):
    outcome = manager.checkpoint_now(
        b"kv", key=_key(), trigger="user", persist=True, operator_grant=False
    )
    assert outcome.tier is CheckpointTier.RAM
    assert manager.disk_store.created == []
    assert "no explicit operator grant" in (outcome.eligibility.reason if outcome.eligibility else "")


def test_ram_residency_is_not_disk_consent(manager):
    """A checkpoint sitting in RAM does not acquire consent by having been kept."""
    taken = manager.checkpoint_now(b"kv", key=_key(), trigger="user", persist=False)
    assert taken.tier is CheckpointTier.RAM
    promoted = manager.promote(
        taken.checkpoint_id, requesting_tenant="t1", trigger="user",
        operator_grant=False,
    )
    assert promoted.tier is CheckpointTier.RAM
    assert manager.disk_store.created == []
    assert promoted.eligibility is not None and not promoted.eligibility.permitted


def test_always_deny_gate_refuses_even_an_explicit_operator_grant(manager):
    set_persistence_eligibility_gate(AlwaysDenyEligibility())
    outcome = manager.checkpoint_now(
        b"kv", key=_key(), trigger="user", persist=True, operator_grant=True
    )
    assert outcome.tier is CheckpointTier.RAM
    assert manager.disk_store.created == []
    assert outcome.eligibility is not None
    assert outcome.eligibility.gate == "always-deny"


def test_a_deployment_gate_can_widen_what_is_persistable(manager):
    """The extension point: the real policy plugs in here and nothing else changes."""

    class _PolicyGate:
        name = "deployment-policy"

        def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
            return EligibilityDecision(
                permitted=request.worthiness_score >= 0.8,
                gate=self.name,
                reason="deployment policy permits high-worthiness agent checkpoints",
                policy_ref="policy://kv-retention/v1",
            )

    set_persistence_eligibility_gate(_PolicyGate())
    outcome = manager.observe(_strong_observation(), key=_key(), payload=b"kv")
    assert outcome.tier is CheckpointTier.DISK
    assert len(manager.disk_store.created) == 1
    assert (
        manager.disk_store.created[0]["provenance"]["eligibility_policy_ref"]
        == "policy://kv-retention/v1"
    )


def test_the_default_gate_reports_every_unanswerable_policy_question():
    """The unresolved privacy/residency half of D-5.1-3 is visible at runtime, not only
    in a report."""
    decision = OperatorGrantEligibility().evaluate(
        PersistenceRequest(tenant="t1", initiator="user", operator_grant=True)
    )
    assert decision.permitted is True
    assert set(decision.unresolved) == {
        "data_residency_region", "data_classification", "retention_period",
    }


def test_a_manager_without_a_durable_store_refuses_promotion_cleanly():
    ram_only = TieredCheckpointManager(disk_store=None)
    outcome = ram_only.checkpoint_now(
        b"kv", key=_key(), trigger="user", persist=True, operator_grant=True
    )
    assert outcome.tier is CheckpointTier.RAM
    assert "no durable store is bound" in outcome.reason


# ---------------------------------------------------------------------------
# 6. The RAM tier is a real, bounded, tenant-isolated store
# ---------------------------------------------------------------------------


def test_ram_tier_refuses_a_cross_tenant_load():
    store = RAMCheckpointStore()
    record = store.put(b"kv", key=_key(tenant="tenant-a"))
    with pytest.raises(CrossTenantCheckpointError):
        store.fetch(record.checkpoint_id, requesting_tenant="tenant-b")
    assert store.fetch(record.checkpoint_id, requesting_tenant="tenant-a")[1] == b"kv"


def test_ram_tier_requires_a_requesting_tenant():
    store = RAMCheckpointStore()
    record = store.put(b"kv", key=_key())
    with pytest.raises(KVCheckpointError, match="requesting_tenant is mandatory"):
        store.fetch(record.checkpoint_id, requesting_tenant="")


def test_ram_tier_evicts_lru_on_the_entry_bound():
    store = RAMCheckpointStore(max_entries=2)
    first = store.put(b"a", key=_key(prefix="a"))
    store.put(b"b", key=_key(prefix="b"))
    store.put(b"c", key=_key(prefix="c"))
    assert store.stats().entries == 2
    assert store.evictions == 1
    with pytest.raises(KVCheckpointError):
        store.fetch(first.checkpoint_id, requesting_tenant="t1")


def test_ram_tier_evicts_on_the_byte_bound():
    store = RAMCheckpointStore(max_entries=100, max_bytes=10)
    store.put(b"x" * 8, key=_key(prefix="a"))
    store.put(b"y" * 8, key=_key(prefix="b"))
    assert store.stats().resident_bytes <= 10
    assert store.evictions == 1


def test_ram_tier_refuses_an_empty_payload():
    with pytest.raises(KVCheckpointError, match="empty KV payload"):
        RAMCheckpointStore().put(b"", key=_key())


# ---------------------------------------------------------------------------
# 7. Inspectability — a user can ask WHY
# ---------------------------------------------------------------------------


def test_explain_returns_the_full_reasoning_for_a_persisted_checkpoint(manager):
    outcome = manager.checkpoint_now(
        b"kv", key=_key(), run_id="run-1", point="post-plan", trigger="user",
        persist=True, operator_grant=True, observation=_strong_observation(),
    )
    explanation = manager.explain(outcome.checkpoint_id, requesting_tenant="t1")
    assert explanation.tier is CheckpointTier.DISK
    assert explanation.trigger == "user"
    assert explanation.recommendation is not None
    assert explanation.recommendation.score > 0
    assert explanation.eligibility is not None and explanation.eligibility.permitted
    assert explanation.eligibility_gate_in_force == "operator-grant-default"


def test_explain_refuses_a_cross_tenant_question(manager):
    outcome = manager.checkpoint_now(b"kv", key=_key(tenant="tenant-a"), trigger="user")
    with pytest.raises(CrossTenantCheckpointError):
        manager.explain(outcome.checkpoint_id, requesting_tenant="tenant-b")


# ---------------------------------------------------------------------------
# 8. WIRING — the recommendation surface reaches the model
# ---------------------------------------------------------------------------


def test_nothing_published_renders_no_instructions():
    assert render_checkpoint_advisory_instructions() == ""


def test_a_not_worth_it_verdict_renders_nothing(manager):
    manager.recommend(_weak_observation())
    assert render_checkpoint_advisory_instructions() == ""


def test_recommend_publishes_the_advisory_for_the_next_model_call(manager):
    """`recommend` is what makes this a recommendation TO THE MODEL, not a return value."""
    recommendation = manager.recommend(_strong_observation())
    rendered = render_checkpoint_advisory_instructions()
    assert "checkpoint-worthy: score" in rendered
    assert "drivers:" in rendered
    assert f"{recommendation.score:.2f}" in rendered
    assert "graph_kv_checkpoint" in rendered  # tells the model how to act on it


def test_advisory_names_its_blockers_and_missing_evidence(manager):
    manager.recommend(_strong_observation(sibling_task_count=None,
                                          queued_task_count=None))
    rendered = render_checkpoint_advisory_instructions()
    assert "blockers:" in rendered
    assert "no evidence from:" in rendered
    assert "predicted_reuse" in rendered


def test_advisory_reaches_the_model_through_create_agent(manager):
    """LIVE PATH: an agent built by the standard factory carries the advisory into the
    outgoing request's instructions — the proof that the recommendation actually reaches
    the LLM rather than merely being renderable."""
    pytest.importorskip(
        "pydantic_ai_harness",
        reason="agent.factory needs the optional pydantic-ai-harness extra",
    )
    from agent_utilities.agent.factory import create_agent

    agent, _ = create_agent(
        name="kv-advisory-probe",
        system_prompt="You are a probe.",
        mcp_toolsets=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    manager.recommend(_strong_observation())
    result = agent.run_sync("hello")
    instructions = result.all_messages()[0].instructions or ""
    assert "KV-CHECKPOINT ADVISORY" in instructions
    assert "checkpoint-worthy: score" in instructions


def test_an_untouched_run_gets_a_byte_identical_prompt():
    """No advisory published => the instructions are exactly what they were before this
    layer existed. The injection is free for every run that never scores a moment."""
    pytest.importorskip(
        "pydantic_ai_harness",
        reason="agent.factory needs the optional pydantic-ai-harness extra",
    )
    from agent_utilities.agent.factory import create_agent

    agent, _ = create_agent(
        name="kv-advisory-null-probe",
        system_prompt="You are a probe.",
        mcp_toolsets=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    result = agent.run_sync("hello")
    assert "KV-CHECKPOINT ADVISORY" not in (result.all_messages()[0].instructions or "")


def test_acting_on_the_advisory_clears_it(manager):
    """Once someone checkpoints, the advisory is spent — leaving it published would
    keep inviting the model to duplicate the checkpoint just taken."""
    manager.recommend(_strong_observation())
    assert render_checkpoint_advisory_instructions() != ""
    manager.checkpoint_now(b"kv", key=_key(), trigger="agent")
    assert render_checkpoint_advisory_instructions() == ""


def test_publish_is_context_local_not_global(manager):
    """Two interleaved runs must never read each other's verdict."""
    import contextvars

    manager.recommend(_strong_observation())
    assert render_checkpoint_advisory_instructions() != ""

    def _other_context() -> str:
        publish_checkpoint_advisory(None)
        return render_checkpoint_advisory_instructions()

    assert contextvars.copy_context().run(_other_context) == ""
    # The outer context's advisory survives the inner context's reset.
    assert render_checkpoint_advisory_instructions() != ""


# ---------------------------------------------------------------------------
# 9. WIRING — the user-invoked path end to end over the live MCP action
# ---------------------------------------------------------------------------


def test_user_invoked_checkpoint_end_to_end_over_the_mcp_action(monkeypatch):
    """The full operator story through the real MCP action surface: checkpoint now →
    it is resident → promote it with a grant → ask why it was persisted."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    disk = _FakeDiskStore()
    ram = RAMCheckpointStore()
    monkeypatch.setattr(est, "_RAM_CHECKPOINT_STORE", ram, raising=False)
    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(ram_store=ram, disk_store=disk),
    )

    common = dict(
        graph="", model_identity="qwen3.6-27b", quantization="fp16",
        serving_engine="vllm", engine_version="0.9.0",
        prefix_digest=prefix_digest("ctx"), tenant="t1", policy_version="v1",
        run_id="run-1", point="post-plan", requesting_tenant="t1",
        observation_json="{}", evidence_bundle_json="{}",
        context_bundle_json="{}", checkpoint_id="", data_b64="",
        initiator="user", persist=False, operator_grant=False,
    )

    taken = json.loads(
        est._kv_checkpoint_intelligence(
            "checkpoint_now",
            **{**common, "data_b64": base64.b64encode(b"kv-bytes").decode()},
        )
    )
    assert taken["result"]["taken"] is True
    assert taken["result"]["tier"] == "ram"
    checkpoint_id = taken["result"]["checkpoint_id"]

    stats = json.loads(est._kv_checkpoint_intelligence("ram_stats", **common))
    assert stats["result"]["entries"] == 1
    assert stats["result"]["eligibility_gate"] == "operator-grant-default"
    # A score is uninterpretable without knowing which signals produced it.
    assert {s["name"] for s in stats["result"]["scorers"]} >= {
        "rebuild_cost", "contradictions", "model_self_report",
    }

    refused = json.loads(
        est._kv_checkpoint_intelligence(
            "promote", **{**common, "checkpoint_id": checkpoint_id}
        )
    )
    assert refused["result"]["tier"] == "ram"
    assert disk.created == []

    promoted = json.loads(
        est._kv_checkpoint_intelligence(
            "promote",
            **{**common, "checkpoint_id": checkpoint_id, "operator_grant": True},
        )
    )
    assert promoted["result"]["tier"] == "disk"
    assert len(disk.created) == 1

    explained = json.loads(
        est._kv_checkpoint_intelligence(
            "explain", **{**common, "checkpoint_id": checkpoint_id}
        )
    )
    assert explained["result"]["tier"] == "disk"
    assert explained["result"]["eligibility"]["permitted"] is True


def test_mcp_recommend_action_returns_a_rendered_advisory(monkeypatch):
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(
            ram_store=RAMCheckpointStore(), disk_store=_FakeDiskStore()
        ),
    )
    observation = json.dumps(
        {
            "rebuild": {"prompt_tokens": 55000, "tool_calls": 19, "retrievals": 14,
                        "wall_time_s": 110.0},
            "sibling_task_count": 5,
            "retrieved_items": 20,
            "novel_items": 1,
            "claim_count": 12,
            "evidence_span_count": 15,
            "unresolved_contradictions": 0,
            "high_severity_contradictions": 0,
            "turns_since_context_change": 6,
            "phase": "plan",
            "phase_completed": True,
        }
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "recommend",
            graph="", data_b64="", model_identity="", quantization="",
            serving_engine="", engine_version="", prefix_digest="", tenant="",
            policy_version="", run_id="", point="", checkpoint_id="",
            requesting_tenant="", observation_json=observation,
            evidence_bundle_json="{}", context_bundle_json="{}", initiator="agent",
            persist=False, operator_grant=False,
        )
    )
    assert "checkpoint-worthy: score" in payload["result"]["advisory"]
    assert payload["result"]["recommended_tier"] in {"ram", "disk"}
    assert payload["result"]["drivers"]


def test_mcp_recommend_rejects_a_malformed_observation(monkeypatch):
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est, "_checkpoint_manager", lambda graph: TieredCheckpointManager()
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "recommend",
            graph="", data_b64="", model_identity="", quantization="",
            serving_engine="", engine_version="", prefix_digest="", tenant="",
            policy_version="", run_id="", point="", checkpoint_id="",
            requesting_tenant="", observation_json="[1,2,3]",
            evidence_bundle_json="{}", context_bundle_json="{}", initiator="agent",
            persist=False, operator_grant=False,
        )
    )
    assert payload["error"]["code"] == "invalid_request"


def test_mcp_recommend_derives_signals_from_handed_in_bundles(monkeypatch):
    """LIVE PATH for the bundle adapters: an agent hands over what graph_ask already
    gave it, and the grounding/contradiction/novelty axes populate themselves."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est, "_checkpoint_manager", lambda graph: TieredCheckpointManager()
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "recommend",
            graph="", data_b64="", model_identity="", quantization="",
            serving_engine="", engine_version="", prefix_digest="", tenant="",
            policy_version="", run_id="", point="", checkpoint_id="",
            requesting_tenant="", observation_json="{}",
            evidence_bundle_json=json.dumps(
                {
                    "claims": [{"id": "c1"}, {"id": "c2"}],
                    "evidence_spans": [{"id": "e1"}, {"id": "e2"}],
                    "contradictions": [{"severity": "high"}],
                }
            ),
            context_bundle_json="{}", initiator="agent", persist=False,
            operator_grant=False,
        )
    )
    signals = {s["name"]: s for s in payload["result"]["signals"]}
    assert signals["grounding_density"]["value"] is not None
    # The high-severity contradiction rode in from the bundle and vetoed.
    assert payload["result"]["recommended_tier"] == "none"
    assert any("VETO" in b for b in payload["result"]["blockers"])


def test_mcp_explicit_observation_fields_win_over_a_bundle(monkeypatch):
    """A caller's direct measurement is more authoritative than an inference."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est, "_checkpoint_manager", lambda graph: TieredCheckpointManager()
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "recommend",
            graph="", data_b64="", model_identity="", quantization="",
            serving_engine="", engine_version="", prefix_digest="", tenant="",
            policy_version="", run_id="", point="", checkpoint_id="",
            requesting_tenant="",
            observation_json=json.dumps({"claim_count": 99}),
            evidence_bundle_json=json.dumps({"claims": [{"id": "c1"}]}),
            context_bundle_json="{}", initiator="agent", persist=False,
            operator_grant=False,
        )
    )
    grounding = next(
        s for s in payload["result"]["signals"] if s["name"] == "grounding_density"
    )
    assert grounding["detail"]["claim_count"] == 99


def test_mcp_rejects_a_malformed_bundle(monkeypatch):
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est, "_checkpoint_manager", lambda graph: TieredCheckpointManager()
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "recommend",
            graph="", data_b64="", model_identity="", quantization="",
            serving_engine="", engine_version="", prefix_digest="", tenant="",
            policy_version="", run_id="", point="", checkpoint_id="",
            requesting_tenant="", observation_json="{}",
            evidence_bundle_json="[1,2,3]", context_bundle_json="{}",
            initiator="agent", persist=False, operator_grant=False,
        )
    )
    assert payload["error"]["code"] == "invalid_request"


def test_mcp_rejects_an_unrecognized_initiator(monkeypatch):
    """The initiator is what the whole eligibility decision turns on — an unrecognized
    value must be refused at the boundary, never coerced or guessed."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est, "_checkpoint_manager", lambda graph: TieredCheckpointManager()
    )
    payload = json.loads(
        est._kv_checkpoint_intelligence(
            "checkpoint_now",
            graph="", data_b64=base64.b64encode(b"kv").decode(),
            model_identity="m", quantization="fp16", serving_engine="vllm",
            engine_version="1", prefix_digest="abc", tenant="t1", policy_version="v1",
            run_id="", point="", checkpoint_id="", requesting_tenant="t1",
            observation_json="{}", evidence_bundle_json="{}",
            context_bundle_json="{}", initiator="root", persist=True,
            operator_grant=True,
        )
    )
    assert payload["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 10. Observation adapters over the existing evidence/context machinery
# ---------------------------------------------------------------------------


def test_observation_reads_grounding_and_contradictions_from_an_evidence_bundle():
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    bundle = EvidenceBundle(
        claims=[{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}],
        evidence_spans=[{"id": "e1"}, {"id": "e2"}, {"id": "e3"}],
        contradictions=[{"new_id": "c1", "conflict_id": "c2", "severity": "high"}],
    )
    observation = CheckpointObservation.from_evidence_bundle(bundle)
    assert observation.claim_count == 2
    assert observation.evidence_span_count == 3
    assert observation.unresolved_contradictions == 1
    assert observation.high_severity_contradictions == 1
    # And that veto flows straight through the advisor.
    assert CheckpointAdvisor().evaluate(observation).recommended_tier is (
        CheckpointTier.NONE
    )


def test_observation_reads_novelty_from_a_context_bundle():
    class _Bundle:
        items = [object(), object(), object()]
        dropped_redundant = 7
        citations = [object(), object(), object()]

    observation = CheckpointObservation.from_context_bundle(_Bundle())
    assert observation.novel_items == 3
    assert observation.retrieved_items == 10
    assert observation.evidence_span_count == 3
