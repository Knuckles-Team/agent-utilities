"""Operational ActionPolicy decision point (CONCEPT:OS-5.24).

Covers: shipped-default conservatism (mutating ⇒ approval, diagnostics ⇒
auto), file-rule tiers (auto / auto_notify / forbidden), durable rate-limit
and blast-radius accounting via the ActionDecision ledger, maintenance-window
downgrade, KG governance_rule overrides beating file rules, approval-queue
dedup, fail-closed decisions, and shipped-YAML ⇄ embedded-default parity.

@pytest.mark.concept("OS-5.24")
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

pytestmark = pytest.mark.concept("OS-5.24")


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
