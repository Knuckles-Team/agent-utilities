"""Operational ActionPolicy decision point (CONCEPT:AU-OS.deployment.fleet-lifecycle-control).

Covers: shipped-default conservatism (mutating ⇒ approval, diagnostics ⇒
auto), file-rule tiers (auto / auto_notify / forbidden), durable rate-limit
and blast-radius accounting via the ActionDecision ledger, maintenance-window
downgrade, KG governance_rule overrides beating file rules, approval-queue
dedup, fail-closed decisions, and shipped-YAML ⇄ embedded-default parity.

@pytest.mark.concept("AU-OS.deployment.fleet-lifecycle-control")
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from agent_utilities.orchestration import action_policy as ap
from agent_utilities.orchestration.action_policy import (
    DEFAULT_POLICY,
    ActionPolicy,
    ActionRequest,
    in_maintenance_window,
)

from .fleet_autonomy_fakes import CaptureNotifier, FakeEngine, write_policy

pytestmark = pytest.mark.concept("AU-OS.deployment.fleet-lifecycle-control")


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def notifier(monkeypatch):
    sink = CaptureNotifier()
    from agent_utilities.knowledge_graph.actions import dispatch

    monkeypatch.setattr(dispatch, "_DEFAULT_NOTIFIER", sink)
    return sink


# ---------------------------------------------------------------------------
# Shipped conservative default
# ---------------------------------------------------------------------------


def test_default_policy_queues_mutating_actions(engine):
    policy = ActionPolicy(engine=engine)
    decision = policy.decide(
        ActionRequest(kind="restart_service", target="caddy-mcp", source="reconciler")
    )
    assert decision.decision == "queue_approval"
    assert not decision.allowed
    approvals = engine.by_type("ActionApproval")
    assert len(approvals) == 1
    assert approvals[0]["kind"] == "restart_service"
    assert approvals[0]["status"] == "pending"


def test_default_policy_allows_diagnostics(engine):
    policy = ActionPolicy(engine=engine)
    decision = policy.decide(ActionRequest(kind="diagnose", target="anything"))
    assert decision.decision == "allow"
    assert decision.allowed


def test_unknown_kind_falls_to_approval_default(engine):
    policy = ActionPolicy(engine=engine)
    decision = policy.decide(ActionRequest(kind="format_disk", target="r820"))
    assert decision.decision == "queue_approval"


def test_every_decision_is_audited(engine):
    policy = ActionPolicy(engine=engine)
    policy.decide(ActionRequest(kind="diagnose", target="a"))
    policy.decide(ActionRequest(kind="restart_service", target="b"))
    audited = engine.by_type("ActionDecision")
    assert len(audited) == 2
    assert {a["decision"] for a in audited} == {"allow", "queue_approval"}
    assert all(a.get("decided_unix") for a in audited)


def test_shipped_yaml_matches_embedded_default():
    shipped = (
        Path(__file__).resolve().parents[2] / "deploy" / "action-policy.default.yml"
    )
    assert yaml.safe_load(shipped.read_text(encoding="utf-8")) == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# File rules: tiers
# ---------------------------------------------------------------------------


def test_auto_tier_allows(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n  - {kind: restart_service, target: '*', tier: auto}\n",
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert decision.decision == "allow"


def test_auto_notify_tier_allows_and_notifies(engine, tmp_path, notifier):
    path = write_policy(
        tmp_path,
        "rules:\n  - {kind: restart_service, target: '*', tier: auto_notify}\n",
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert decision.decision == "allow_notify"
    assert decision.allowed
    assert any("restart_service(caddy-mcp)" in m for m in notifier.messages)


def test_forbidden_tier_denies(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n  - {kind: restart_service, target: 'prod-*', tier: forbidden}\n",
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="prod-db")
    )
    assert decision.decision == "deny"
    assert engine.by_type("ActionApproval") == []


def test_target_selector_glob_scoping(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n"
        "  - {kind: restart_service, target: 'caddy-*', tier: auto}\n"
        "  - {kind: restart_service, target: '*', tier: forbidden}\n",
    )
    policy = ActionPolicy(engine=engine, policy_path=path)
    assert policy.decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    ).allowed
    assert (
        policy.decide(ActionRequest(kind="restart_service", target="other")).decision
        == "deny"
    )


# ---------------------------------------------------------------------------
# Rate limit / blast radius (durable, ledger-backed)
# ---------------------------------------------------------------------------


def test_rate_limit_denies_looping_action(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n"
        "  - kind: restart_service\n"
        "    target: '*'\n"
        "    tier: auto\n"
        "    rate_limit: {max: 1, window_s: 3600}\n",
    )
    policy = ActionPolicy(engine=engine, policy_path=path)
    first = policy.decide(ActionRequest(kind="restart_service", target="caddy-mcp"))
    second = policy.decide(ActionRequest(kind="restart_service", target="caddy-mcp"))
    assert first.decision == "allow"
    assert second.decision == "deny"
    assert "rate limit" in second.reason


def test_rate_limit_is_per_target(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n"
        "  - kind: restart_service\n"
        "    target: '*'\n"
        "    tier: auto\n"
        "    rate_limit: {max: 1, window_s: 3600}\n"
        "    blast_radius: {max_targets: 10, window_s: 3600}\n",
    )
    policy = ActionPolicy(engine=engine, policy_path=path)
    assert policy.decide(ActionRequest(kind="restart_service", target="svc-a")).allowed
    assert policy.decide(ActionRequest(kind="restart_service", target="svc-b")).allowed


def test_blast_radius_queues_wide_wave(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n"
        "  - kind: restart_service\n"
        "    target: '*'\n"
        "    tier: auto\n"
        "    rate_limit: {max: 10, window_s: 3600}\n"
        "    blast_radius: {max_targets: 1, window_s: 3600}\n",
    )
    policy = ActionPolicy(engine=engine, policy_path=path)
    first = policy.decide(ActionRequest(kind="restart_service", target="svc-a"))
    second = policy.decide(ActionRequest(kind="restart_service", target="svc-b"))
    assert first.decision == "allow"
    assert second.decision == "queue_approval"
    assert "blast-radius" in second.reason


# ---------------------------------------------------------------------------
# Maintenance window
# ---------------------------------------------------------------------------


def test_in_maintenance_window_parsing():
    assert in_maintenance_window(None, 0)
    # 00:00 UTC epoch: inside 22:00-04:00 (wraps), outside 09:00-17:00.
    assert in_maintenance_window("22:00-04:00", 0)
    assert not in_maintenance_window("09:00-17:00", 0)
    assert in_maintenance_window("garbage", 0)  # unparseable fails open


def test_outside_maintenance_window_downgrades_auto_to_queue(engine, tmp_path):
    now = time.gmtime()
    minute = now.tm_hour * 60 + now.tm_min
    # A 2-minute window guaranteed NOT to contain 'now'.
    start = (minute + 120) % 1440
    window = f"{start // 60:02d}:{start % 60:02d}-{(start + 2) // 60 % 24:02d}:{(start + 2) % 60:02d}"
    path = write_policy(
        tmp_path,
        "rules:\n"
        "  - kind: restart_service\n"
        "    target: '*'\n"
        "    tier: auto\n"
        f"    maintenance_window: '{window}'\n",
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert decision.decision == "queue_approval"
    assert "maintenance window" in decision.reason


# ---------------------------------------------------------------------------
# KG governance_rule overrides win
# ---------------------------------------------------------------------------


def test_kg_override_beats_file_rule(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n  - {kind: restart_service, target: '*', tier: auto}\n",
    )
    engine.add_node(
        "governance_rule:1",
        "governance_rule",
        properties={
            "scope": "action_policy",
            "kind": "restart_service",
            "target": "*",
            "tier": "forbidden",
        },
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert decision.decision == "deny"
    assert decision.rule_origin == "kg"


def test_inactive_kg_override_is_skipped(engine, tmp_path):
    path = write_policy(
        tmp_path,
        "rules:\n  - {kind: restart_service, target: '*', tier: auto}\n",
    )
    engine.add_node(
        "governance_rule:1",
        "governance_rule",
        properties={
            "scope": "action_policy",
            "kind": "restart_service",
            "target": "*",
            "tier": "forbidden",
            "active": "false",
        },
    )
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert decision.decision == "allow"


# ---------------------------------------------------------------------------
# Approval queue mechanics
# ---------------------------------------------------------------------------


def test_pending_approval_is_deduped(engine):
    policy = ActionPolicy(engine=engine)
    first = policy.decide(ActionRequest(kind="restart_service", target="caddy-mcp"))
    second = policy.decide(ActionRequest(kind="restart_service", target="caddy-mcp"))
    assert first.approval_id == second.approval_id
    assert len(engine.by_type("ActionApproval")) == 1


def test_approval_queue_notifies_operators(engine, notifier):
    ActionPolicy(engine=engine).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    assert any("approval required" in m for m in notifier.messages)


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_internal_error_fails_closed(engine, monkeypatch):
    policy = ActionPolicy(engine=engine)
    monkeypatch.setattr(
        policy, "_decide_inner", lambda req: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    decision = policy.decide(ActionRequest(kind="diagnose", target="x"))
    assert decision.decision == "deny"
    assert "fail closed" in decision.reason


def test_broken_policy_file_falls_back_to_default(engine, tmp_path):
    path = write_policy(tmp_path, "rules: 'not-a-list'\n")
    decision = ActionPolicy(engine=engine, policy_path=path).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp")
    )
    # Shipped conservative default applies: mutating ⇒ approval.
    assert decision.decision == "queue_approval"


def test_module_default_policy_is_conservative():
    assert DEFAULT_POLICY["defaults"]["tier"] == ap.TIER_APPROVAL
    mutating = {
        r["kind"]: r["tier"]
        for r in DEFAULT_POLICY["rules"]
        if r["kind"]
        in ("restart_service", "scale_service", "deploy_service", "merge_promotion")
    }
    assert set(mutating.values()) == {ap.TIER_APPROVAL}


def test_promote_mined_claim_default_never_auto():
    """SAFETY-CRITICAL (workstream C4, Insight Engine closed loop): a mined KG
    finding (:AssociationRule/:Anomaly/:PredictedEdge) promoted to a verified
    Claim goes through ``action_policy.decide(kind="promote_mined_claim")`` on
    EVERY path (autonomy tier on or off — see ``loop_controller.
    _run_insight_validation``). This test must FAIL THE BUILD the moment
    someone (accidentally or otherwise) relaxes the shipped default for that
    kind to ``auto``/``auto_notify`` — the only acceptable shipped default is
    ``approval_required``, matching every other lifecycle-flipping kind
    (``merge_promotion``, ``spec_promotion``) in this policy.
    """
    rules_by_kind = {r["kind"]: r["tier"] for r in DEFAULT_POLICY["rules"]}
    assert "promote_mined_claim" in rules_by_kind, (
        "promote_mined_claim rule missing from the shipped ActionPolicy — a mined "
        "claim would fall through to the policy default tier instead of an "
        "explicit, auditable rule"
    )
    assert rules_by_kind["promote_mined_claim"] == ap.TIER_APPROVAL
    assert rules_by_kind["promote_mined_claim"] not in (
        ap.TIER_AUTO,
        ap.TIER_AUTO_NOTIFY,
    )

    # Behavioral form of the same guarantee: an actual decide() call, with no KG
    # overrides present, must queue for approval rather than allow.
    engine = FakeEngine()
    decision = ActionPolicy(engine=engine).decide(
        ActionRequest(
            kind="promote_mined_claim",
            target="claim:insight:deadbeef",
            source="loop_engine",
        )
    )
    assert decision.tier == ap.TIER_APPROVAL
    assert decision.decision == "queue_approval"
    assert not decision.allowed


# ---------------------------------------------------------------------------
# Assurance gate wiring (CONCEPT:AU-OS.governance.assurance-state-machine-verifier):
# the deterministic verifier runs BEFORE the tier pipeline in decide()/classify(),
# and evaluate() offers the same check with no side effects.
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.governance.assurance-state-machine-verifier")
class TestAssuranceGateWiring:
    def test_valid_payload_is_governed_normally(self, engine):
        # Well-formed scale_service from the reconciler role — the verifier
        # passes and the existing tier (approval_required by default) applies.
        decision = ActionPolicy(engine=engine).decide(
            ActionRequest(
                kind="scale_service",
                target="caddy-mcp",
                params={"replicas": 3},
                source="reconciler",
            )
        )
        assert decision.invariant == ""
        assert decision.decision == "queue_approval"  # default tier, unaffected

    def test_out_of_role_kind_is_denied_before_tier_check(self, engine, tmp_path):
        # Even an auto-tier rule for this kind must not save an out-of-role request —
        # the assurance check runs BEFORE the rule/tier pipeline.
        path = write_policy(
            tmp_path,
            "rules:\n  - {kind: secret.delete, target: '*', tier: auto}\n",
        )
        decision = ActionPolicy(engine=engine, policy_path=path).decide(
            ActionRequest(
                kind="secret.delete",
                target="db-creds",
                params={"path": "apps/db-creds"},
                source="reconciler",
            )
        )
        assert decision.decision == "deny"
        assert not decision.allowed
        assert decision.invariant == "role"
        assert "reconciler" in decision.reason
        # No approval was filed either — the assurance deny short-circuits everything.
        assert engine.by_type("ActionApproval") == []

    def test_missing_argument_is_denied_with_schema_invariant(self, engine):
        decision = ActionPolicy(engine=engine).decide(
            ActionRequest(kind="scale_service", target="caddy-mcp", source="reconciler")
        )
        assert decision.decision == "deny"
        assert decision.invariant == "schema"
        assert "replicas" in decision.reason

    def test_illegal_state_transition_is_denied_with_precondition_invariant(
        self, engine
    ):
        decision = ActionPolicy(engine=engine).decide(
            ActionRequest(
                kind="merge_promotion",
                target="proposal:1",
                params={"proposal_id": "proposal:1", "state": "active"},
            )
        )
        assert decision.decision == "deny"
        assert decision.invariant == "precondition"

    def test_hallucinated_reference_is_denied_when_registry_configured(
        self, engine, tmp_path
    ):
        # options.known_tools wires the reference-existence check on.
        path = write_policy(
            tmp_path,
            "rules: []\noptions:\n  known_tools: [observe_screen, click]\n",
        )
        decision = ActionPolicy(engine=engine, policy_path=path).decide(
            ActionRequest(
                kind="workspace.computer_use",
                target="sandbox-1",
                params={"tool": "not_a_real_tool"},
            )
        )
        assert decision.decision == "deny"
        assert decision.invariant == "reference"
        assert "not_a_real_tool" in decision.reason

    def test_evaluate_is_side_effect_free(self, engine):
        """evaluate() gives the same verdict as decide() but writes no ledger node."""
        request = ActionRequest(
            kind="scale_service", target="caddy-mcp", source="reconciler"
        )
        verdict = ActionPolicy(engine=engine).evaluate(request)
        assert verdict.decision == "deny"
        assert verdict.invariant == "schema"
        assert engine.by_type("ActionDecision") == []
        assert engine.by_type("ActionApproval") == []

    def test_evaluate_allows_a_valid_payload(self, engine):
        verdict = ActionPolicy(engine=engine).evaluate(
            ActionRequest(kind="diagnose", target="anything")
        )
        assert verdict.decision == "allow"
        assert verdict.allowed

    def test_classify_forbids_an_invariant_violation(self, engine):
        # classify() is the side-effect-free tier read the PreToolUse gate uses;
        # the assurance check must forbid there too, before the rule tier.
        tier = ActionPolicy(engine=engine).classify(
            ActionRequest(
                kind="secret.delete",
                target="x",
                params={"path": "apps/x"},
                source="reconciler",
            )
        )
        assert tier == ap.TIER_FORBIDDEN

    def test_classify_unaffected_for_a_valid_payload(self, engine):
        tier = ActionPolicy(engine=engine).classify(
            ActionRequest(kind="diagnose", target="anything")
        )
        assert tier == ap.TIER_AUTO


# ---------------------------------------------------------------------------
# Live-path integration: the deny path actually blocks execution, exercised
# through the REAL fleet_reconciler convergence step + a real actuator (not
# just the ActionPolicy helper in isolation).
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.governance.assurance-state-machine-verifier")
class TestAssuranceGateBlocksLiveExecution:
    def test_valid_convergence_step_executes_via_dry_run_actuator(self, engine):
        from agent_utilities.orchestration.fleet_actuation import DryRunActuator
        from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

        actuator = DryRunActuator()
        reconciler = FleetReconciler(engine=engine, actuator=actuator, policy=None)
        # auto tier via file override so we isolate the assurance check.
        reconciler.policy = ActionPolicy(engine=engine)
        request = ActionRequest(
            kind="scale_service",
            target="caddy-mcp",
            params={"replicas": 3},
            source="reconciler",
        )
        entry = reconciler._converge_one(request)
        # default tier is approval_required, so nothing executes yet — but the
        # verifier passed (no invariant recorded on the queued decision).
        assert entry["decision"] == "queue_approval"
        assert "execution" not in entry
        assert actuator.applied == []

    def test_assurance_denial_blocks_actuator_even_when_the_kind_would_otherwise_run(
        self, engine, tmp_path
    ):
        """A misconfigured/loosened tier rule (auto) must NOT let an out-of-role
        or hallucinated action reach the actuator — this is the whole point of
        running the verifier BEFORE the tier pipeline, not just before allow.
        """
        from agent_utilities.orchestration.fleet_actuation import DryRunActuator
        from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

        path = write_policy(
            tmp_path,
            "rules:\n  - {kind: secret.delete, target: '*', tier: auto}\n",
        )
        actuator = DryRunActuator()
        policy = ActionPolicy(engine=engine, policy_path=path)
        reconciler = FleetReconciler(engine=engine, actuator=actuator, policy=policy)
        request = ActionRequest(
            kind="secret.delete",
            target="db-creds",
            params={"path": "apps/db-creds"},
            source="reconciler",  # reconciler is NOT allowed secret.delete
        )
        entry = reconciler._converge_one(request)
        assert entry["decision"] == "deny"
        # The actuator's apply() was never invoked — the live path genuinely
        # did not execute the action.
        assert actuator.applied == []
        assert "execution" not in entry


# ---------------------------------------------------------------------------
# ``get_action_policy`` process-wide cache (SCALE-P2-1 perf fix,
# CONCEPT:AU-OS.governance.action-policy-decision-point).
#
# Before this fix, ``get_action_policy(engine)`` built a brand-new
# ``ActionPolicy()`` on EVERY call — throwing away the mtime-guarded YAML
# parse cache described in ``ActionPolicy``'s own docstring ("stateless apart
# from an mtime-cached parse of the policy file") every single time. A fresh
# instance never gets to reuse that cache, so every ``decide()`` call re-read
# and re-parsed ``deploy/action-policy.default.yml`` from disk — measured at
# ~300ms/call under load in the SCALE-P2-1 soak harness (see
# ``scripts/scale/loadgen.py``'s ``_MAX_MESSAGE_EVENTS`` comment), two-plus
# orders of magnitude above the AddNode anchor. These tests pin the fix's
# contract: O(1) ``ActionPolicy`` construction + O(1) real YAML parses across
# N ``decide()`` calls for the same engine, AND byte-identical governance
# decisions to a freshly-constructed (pre-fix-shaped) instance — this is a
# perf fix, not a policy change.
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.governance.action-policy-decision-point")
class TestGetActionPolicyCache:
    def setup_method(self, _method):
        ap.reset_action_policy_cache_for_tests()

    def teardown_method(self, _method):
        ap.reset_action_policy_cache_for_tests()

    def test_same_engine_returns_the_same_cached_instance(self, engine):
        first = ap.get_action_policy(engine)
        second = ap.get_action_policy(engine)
        assert first is second

    def test_different_engines_get_independent_policy_instances(self):
        e1, e2 = FakeEngine(), FakeEngine()
        p1 = ap.get_action_policy(e1)
        p2 = ap.get_action_policy(e2)
        assert p1 is not p2
        assert p1.engine is e1
        assert p2.engine is e2

    def test_no_engine_returns_a_shared_singleton(self):
        assert ap.get_action_policy() is ap.get_action_policy()
        assert ap.get_action_policy(None).engine is None

    def test_reset_forces_reconstruction(self, engine):
        first = ap.get_action_policy(engine)
        ap.reset_action_policy_cache_for_tests()
        second = ap.get_action_policy(engine)
        assert first is not second

    def test_decide_constructs_the_policy_and_parses_the_yaml_at_most_once(
        self, engine, monkeypatch
    ):
        """The perf regression test: N ``decide()`` calls through the factory
        must build exactly ONE ``ActionPolicy`` and parse the on-disk policy
        YAML at most once — not once per call (the SCALE-P2-1 bug)."""
        construct_calls = 0
        real_init = ActionPolicy.__init__

        def counting_init(self, *a, **kw):
            nonlocal construct_calls
            construct_calls += 1
            real_init(self, *a, **kw)

        monkeypatch.setattr(ActionPolicy, "__init__", counting_init)

        real_safe_load = yaml.safe_load
        parse_calls = 0

        def counting_safe_load(*a, **kw):
            nonlocal parse_calls
            parse_calls += 1
            return real_safe_load(*a, **kw)

        monkeypatch.setattr(yaml, "safe_load", counting_safe_load)

        # Same (kind, target) every call so the default rate-limit/blast-radius
        # budget (max=3/window) doesn't itself change the *decision* partway
        # through — irrelevant to what this test is pinning (construction/parse
        # counts), but keeping the decision itself well-defined throughout.
        num_calls = 50
        decisions = [
            ap.get_action_policy(engine)
            .decide(ActionRequest(kind="bus.send", target="peer-1", source="bus"))
            .decision
            for _ in range(num_calls)
        ]
        assert all(d in ap._ALLOWING | {ap.DECISION_QUEUE, ap.DECISION_DENY} for d in decisions)

        assert construct_calls == 1, (
            f"expected ActionPolicy constructed once across {num_calls} "
            f"decide() calls, constructed {construct_calls} times"
        )
        assert parse_calls <= 1, (
            f"expected the policy YAML parsed at most once across "
            f"{num_calls} decide() calls (mtime cache should absorb the "
            f"rest), parsed {parse_calls} times"
        )

    @pytest.mark.parametrize(
        "kind,target,source",
        [
            ("bus.send", "peer-1", "bus"),
            ("restart_service", "caddy-mcp", "reconciler"),
            ("diagnose", "anything", "manual"),
            ("secret.delete", "db-creds", "reconciler"),
        ],
    )
    def test_cached_decision_matches_a_freshly_constructed_instance(
        self, kind, target, source
    ):
        """Same governance verdict whether ``ActionPolicy`` is cached (new) or
        rebuilt from scratch (old) — the fix changes allocation, not
        semantics. Each side gets its OWN fresh engine so ledger/rate-limit
        accounting can't leak between them and skew the comparison."""
        request = ActionRequest(kind=kind, target=target, source=source)
        cached_decision = ap.get_action_policy(FakeEngine()).decide(request)

        fresh_request = ActionRequest(kind=kind, target=target, source=source)
        fresh_decision = ActionPolicy(engine=FakeEngine()).decide(fresh_request)

        assert cached_decision.decision == fresh_decision.decision
        assert cached_decision.tier == fresh_decision.tier
        assert cached_decision.rule_origin == fresh_decision.rule_origin
        assert cached_decision.reason == fresh_decision.reason
