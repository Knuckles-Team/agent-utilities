"""Tests for the sync conflict / backfeed preflight module (CA-22, DEC-CA-07/P11).

Covers (CONCEPT:AU-KG.ingest.backfeed-preflight):

  * the 4-policy conflict matrix (``resolve_field_conflict``),
  * fail-closed :class:`ConflictPolicySpec` defaults,
  * :func:`evaluate_backfeed_preflight`'s three outcomes (``None`` / ``PreflightRejection``
    / ``BackfeedProposal``) and every named rejection reason
    (``undeclared_capability`` / ``backfeed_disabled`` / ``stale_expected_version``),
  * :class:`BackfeedProposal` stamping.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.ontology.sync_conflict import (
    BackfeedCapabilitySpec,
    BackfeedProposal,
    ConflictFieldPolicy,
    ConflictPolicySpec,
    PreflightRejection,
    SyncConflict,
    evaluate_backfeed_preflight,
    resolve_field_conflict,
)

# ── ConflictFieldPolicy / ConflictPolicySpec ─────────────────────────────────


def test_conflict_field_policy_rejects_unknown_policy():
    with pytest.raises(ValidationError):
        ConflictFieldPolicy(field="status", policy="last_write_wins")


def test_conflict_policy_spec_default_is_fail_closed():
    spec = ConflictPolicySpec()
    assert spec.default_policy == "manual_review"
    assert spec.policy_for("anything") == "manual_review"
    assert spec.declares("anything") is False


def test_conflict_policy_spec_policy_for_declared_field():
    spec = ConflictPolicySpec(
        fields=[ConflictFieldPolicy(field="status", policy="source_wins")],
        default_policy="reject",
    )
    assert spec.policy_for("status") == "source_wins"
    assert spec.policy_for("owner") == "reject"  # falls back to default
    assert spec.declares("status") is True
    assert spec.declares("owner") is False


def test_backfeed_capability_spec_enabled_requires_approval_class():
    assert BackfeedCapabilitySpec().enabled is False
    assert (
        BackfeedCapabilitySpec(capabilities=["servicenow.incident.update"]).enabled
        is False
    )
    assert (
        BackfeedCapabilitySpec(
            capabilities=["servicenow.incident.update"], approval_class="change"
        ).enabled
        is True
    )


# ── resolve_field_conflict (the 4-policy matrix) ─────────────────────────────


def test_resolve_field_conflict_source_wins_returns_source_value():
    assert resolve_field_conflict("source_wins", "new", "old") == "new"


def test_resolve_field_conflict_graph_derived_returns_graph_value():
    assert resolve_field_conflict("graph_derived", "new", "old") == "old"


@pytest.mark.parametrize("policy", ["manual_review", "reject"])
def test_resolve_field_conflict_review_policies_never_auto_resolve(policy):
    result = resolve_field_conflict(
        policy,
        "new",
        "old",
        connector="leanix",
        node_id="leanix:app-1",
        field_name="status",
    )
    assert isinstance(result, SyncConflict)
    assert result.policy == policy
    assert result.source_value == "new"
    assert result.graph_value == "old"
    assert result.connector == "leanix"
    assert result.node_id == "leanix:app-1"
    assert result.field == "status"


def test_resolve_field_conflict_unknown_policy_raises():
    with pytest.raises(ValueError):
        resolve_field_conflict("last_write_wins", "new", "old")


# ── evaluate_backfeed_preflight ──────────────────────────────────────────────


def test_preflight_no_conflict_no_version_check_is_a_noop():
    """Additive migration: no signal supplied -> None, byte-identical to pre-CA-22."""
    assert evaluate_backfeed_preflight(connector="leanix") is None


def test_preflight_stale_expected_version_blocks_even_without_a_conflict():
    outcome = evaluate_backfeed_preflight(
        connector="servicenow",
        node_id="servicenow:INC001",
        expected_source_version="v1",
        current_source_version="v2",
    )
    assert isinstance(outcome, PreflightRejection)
    assert outcome.reason == "stale_expected_version"


def test_preflight_matching_expected_version_is_not_stale():
    outcome = evaluate_backfeed_preflight(
        connector="servicenow",
        expected_source_version="v2",
        current_source_version="v2",
    )
    assert outcome is None


def test_preflight_conflict_with_no_backfeed_declared_is_undeclared_capability():
    conflict = SyncConflict(
        connector="leanix",
        node_id="leanix:app-1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="manual_review",
    )
    outcome = evaluate_backfeed_preflight(connector="leanix", conflict=conflict)
    assert isinstance(outcome, PreflightRejection)
    assert outcome.reason == "undeclared_capability"


def test_preflight_conflict_with_capabilities_but_no_approval_class_is_backfeed_disabled():
    conflict = SyncConflict(
        connector="leanix",
        node_id="leanix:app-1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="manual_review",
    )
    outcome = evaluate_backfeed_preflight(
        connector="leanix",
        conflict=conflict,
        backfeed=BackfeedCapabilitySpec(capabilities=["leanix.factsheet.update"]),
    )
    assert isinstance(outcome, PreflightRejection)
    assert outcome.reason == "backfeed_disabled"


def test_preflight_conflict_with_enabled_backfeed_raises_a_stamped_proposal():
    conflict = SyncConflict(
        connector="leanix",
        node_id="leanix:app-1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="manual_review",
    )
    outcome = evaluate_backfeed_preflight(
        connector="leanix",
        node_id="leanix:app-1",
        conflict=conflict,
        backfeed=BackfeedCapabilitySpec(
            capabilities=["leanix.factsheet.update"], approval_class="change"
        ),
    )
    assert isinstance(outcome, BackfeedProposal)
    assert outcome.connector == "leanix"
    assert outcome.node_id == "leanix:app-1"
    assert outcome.field == "status"
    assert outcome.approval_class == "change"
    assert outcome.conflict is conflict
    # Never auto-applied: the proposal carries the conflicting values, not a decision.
    assert outcome.source_value == "new"
    assert outcome.graph_value == "old"
    d = outcome.as_dict()
    assert d["field"] == "status"
    assert d["approval_class"] == "change"


def test_backfeed_proposal_stamping_never_raises_even_if_tenant_sharing_unavailable(
    monkeypatch,
):
    """Stamping is best-effort (mirrors approval.py::_stamp) -- a stamping failure
    must never prevent the proposal itself from being returned."""

    def _boom(*_a, **_kw):
        raise RuntimeError("no active engine authority")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.tenant_sharing.stamp_ownership", _boom
    )
    conflict = SyncConflict(
        connector="leanix",
        node_id="leanix:app-1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="reject",
    )
    outcome = evaluate_backfeed_preflight(
        connector="leanix",
        conflict=conflict,
        backfeed=BackfeedCapabilitySpec(
            capabilities=["leanix.factsheet.update"], approval_class="change"
        ),
    )
    assert isinstance(outcome, BackfeedProposal)
    assert outcome.stamped == {}
