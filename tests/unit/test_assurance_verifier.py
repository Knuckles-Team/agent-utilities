"""Deterministic pre-execution assurance verifier (CONCEPT:AU-OS.governance.assurance-state-machine-verifier).

Covers the four invariants in isolation (role, schema, precondition, reference)
via :func:`verify_action`, plus a latency assertion proving the check is
sub-few-ms as required. The wiring into ``ActionPolicy.decide``/``classify``/
``evaluate`` (including the live-path enforcement through the fleet
reconciler) is covered in ``tests/unit/test_action_policy.py``.

@pytest.mark.concept("AU-OS.governance.assurance-state-machine-verifier")
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.orchestration.action_policy import ActionRequest
from agent_utilities.orchestration.assurance_verifier import verify_action

pytestmark = pytest.mark.concept("AU-OS.governance.assurance-state-machine-verifier")


# ---------------------------------------------------------------------------
# A valid payload passes every invariant.
# ---------------------------------------------------------------------------


def test_valid_payload_passes():
    result = verify_action(
        ActionRequest(kind="restart_service", target="caddy-mcp", source="reconciler")
    )
    assert result.ok
    assert result.invariant == ""
    assert result.reason == ""


def test_valid_payload_with_full_schema_and_state_passes():
    result = verify_action(
        ActionRequest(
            kind="scale_service",
            target="caddy-mcp",
            params={"replicas": 3},
            source="reconciler",
        )
    )
    assert result.ok


def test_unregistered_kind_has_no_declared_invariant_and_passes():
    """A kind with no declared invariant isn't structurally denied (only role/
    reference checks — themselves opt-in — can deny it)."""
    result = verify_action(ActionRequest(kind="diagnose", target="anything"))
    assert result.ok


# ---------------------------------------------------------------------------
# (a) Role / allowed-set (RLS).
# ---------------------------------------------------------------------------


def test_out_of_role_tool_is_denied():
    # "reconciler" may restart/stop/scale services — NOT touch the secret store.
    result = verify_action(
        ActionRequest(kind="secret.delete", target="db-creds", source="reconciler")
    )
    assert not result.ok
    assert result.invariant == "role"
    assert "reconciler" in result.reason
    assert "secret.delete" in result.reason


def test_registered_role_within_its_allowed_set_passes():
    result = verify_action(
        ActionRequest(kind="restart_service", target="caddy-mcp", source="reconciler")
    )
    assert result.ok


def test_unrecognized_role_is_not_restricted():
    """A role we've never seen isn't denied by RBAC — the tier gate still applies;
    this is what keeps enabling the verifier by default non-regressing."""
    result = verify_action(
        ActionRequest(
            kind="secret.delete",
            target="x",
            params={"path": "apps/foo"},
            source="some-future-service",
        )
    )
    assert result.ok


def test_default_manual_source_is_unrestricted():
    result = verify_action(
        ActionRequest(kind="secret.delete", target="x", params={"path": "apps/foo"})
    )
    assert result.ok


# ---------------------------------------------------------------------------
# (b) Argument shape/types (schema).
# ---------------------------------------------------------------------------


def test_missing_required_argument_is_denied():
    result = verify_action(
        ActionRequest(kind="scale_service", target="caddy-mcp", source="reconciler")
    )
    assert not result.ok
    assert result.invariant == "schema"
    assert "replicas" in result.reason


def test_mistyped_argument_is_denied():
    result = verify_action(
        ActionRequest(
            kind="scale_service",
            target="caddy-mcp",
            params={"replicas": "three"},
            source="reconciler",
        )
    )
    assert not result.ok
    assert result.invariant == "schema"
    assert "replicas" in result.reason
    assert "int" in result.reason


def test_bool_is_rejected_where_int_declared():
    """bool is a subclass of int in Python — must not silently satisfy an int schema."""
    result = verify_action(
        ActionRequest(
            kind="scale_service",
            target="caddy-mcp",
            params={"replicas": True},
            source="reconciler",
        )
    )
    assert not result.ok
    assert result.invariant == "schema"


# ---------------------------------------------------------------------------
# (c) Preconditions — declared state-machine transitions.
# ---------------------------------------------------------------------------


def test_illegal_state_transition_is_denied():
    result = verify_action(
        ActionRequest(
            kind="merge_promotion",
            target="proposal:1",
            params={"proposal_id": "proposal:1", "state": "active"},
        )
    )
    assert not result.ok
    assert result.invariant == "precondition"
    assert "active" in result.reason


def test_no_claimed_state_at_all_is_not_a_violation():
    """Opt-in precondition: most production callers of merge_promotion predate
    this gate and never claim a state — that absence must not deny them."""
    result = verify_action(ActionRequest(kind="merge_promotion", target="proposal:1"))
    assert result.ok


def test_legal_state_transition_passes():
    result = verify_action(
        ActionRequest(
            kind="merge_promotion",
            target="proposal:1",
            params={"proposal_id": "proposal:1", "state": "proposed"},
        )
    )
    assert result.ok


def test_run_select_precondition():
    assert verify_action(
        ActionRequest(
            kind="run.select",
            target="run:1",
            params={"run_id": "run:1", "state": "held"},
        )
    ).ok
    denied = verify_action(
        ActionRequest(
            kind="run.select",
            target="run:1",
            params={"run_id": "run:1", "state": "materialized"},
        )
    )
    assert not denied.ok
    assert denied.invariant == "precondition"


# ---------------------------------------------------------------------------
# (d) Reference existence — anti-hallucination, best-effort.
# ---------------------------------------------------------------------------


def test_reference_to_nonexistent_tool_is_denied_when_registry_supplied():
    result = verify_action(
        ActionRequest(
            kind="workspace.computer_use",
            target="sandbox-1",
            params={"tool": "totally_made_up_tool"},
        ),
        known_tools={"observe_screen", "click", "type_text"},
    )
    assert not result.ok
    assert result.invariant == "reference"
    assert "totally_made_up_tool" in result.reason


def test_reference_to_existing_tool_passes():
    result = verify_action(
        ActionRequest(
            kind="workspace.computer_use",
            target="sandbox-1",
            params={"tool": "observe_screen"},
        ),
        known_tools={"observe_screen", "click", "type_text"},
    )
    assert result.ok


def test_reference_check_skipped_without_a_registry():
    """Fail-closed applies only when a registry IS supplied — an unresolvable
    reference (no registry available) is never a false deny."""
    result = verify_action(
        ActionRequest(
            kind="workspace.computer_use",
            target="sandbox-1",
            params={"tool": "anything_at_all"},
        )
    )
    assert result.ok


# ---------------------------------------------------------------------------
# Latency — sub-few-ms by construction (pure Python, no I/O).
# ---------------------------------------------------------------------------


def test_verify_action_is_sub_few_ms():
    request = ActionRequest(
        kind="scale_service",
        target="caddy-mcp",
        params={"replicas": 3},
        source="reconciler",
    )
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        result = verify_action(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    mean_ms = elapsed_ms / n
    assert result.ok
    assert result.latency_ms < 1.0  # single-call self-measured latency
    assert mean_ms < 1.0  # amortized mean across many calls
