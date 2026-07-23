"""Tests for the graph_claims MCP tool + REST twin (CONCEPT:AU-KG.evolution.mining-flywheel).

Exercises the live two-surface path: the shared ``_execute_tool`` action core
both the MCP tool and the auto-mounted ``/graph/claims`` REST route dispatch
into, the REAL ``ClaimFlywheel`` state machine (never mocked/stubbed out),
and the fail-closed ActionPolicy gate on every state-changing action.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.mcp import kg_server

pytestmark = pytest.mark.concept("AU-KG.evolution.mining-flywheel")


class _ClaimStubEngine:
    """A competent-enough double: ``ClaimLifecycleEvent`` nodes round-trip
    through ``query_cypher`` (mirrors
    ``tests/unit/knowledge_graph/test_claim_flywheel.py``'s
    ``_FlywheelStubEngine``), so cross-call state reads (``graph_claims``
    builds a FRESH ``ClaimFlywheel`` per dispatch) are exercised for real,
    not just an in-process cache."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if "ClaimLifecycleEvent" not in query:
            return []
        rows = [
            n for n in self.nodes.values() if n.get("type") == "ClaimLifecycleEvent"
        ]
        cid = (params or {}).get("id")
        if cid is not None:  # ClaimFlywheel.history()'s per-claim shape
            rows = [r for r in rows if r.get("claim_id") == cid]
        # else: list_claims()'s unfiltered-all shape (no `id` param at all).
        return [
            {
                "claim_id": r.get("claim_id"),
                "from_state": r.get("from_state"),
                "to_state": r.get("to_state"),
                "reason": r.get("reason"),
                "actor": r.get("actor"),
                "governance_valid": r.get("governance_valid"),
                "action_decision": r.get("action_decision"),
                "timestamp": r.get("timestamp"),
            }
            for r in rows
        ]


@pytest.fixture
def registered():
    kg_server.ensure_tools_registered()
    return kg_server


@pytest.fixture
def stub_engine(monkeypatch):
    engine = _ClaimStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return engine


@pytest.fixture
def allow_policy(monkeypatch):
    """Force every claim.* ActionPolicy gate to allow, for tests of the
    lifecycle dispatch itself (the denial path gets its own dedicated test)."""
    import agent_utilities.mcp.tools.claim_tools as ct

    monkeypatch.setattr(
        ct,
        "_gate",
        lambda kind, target, reason: (
            True,
            {"decision": "allow", "tier": "auto", "reason": "test"},
        ),
    )


def test_graph_claims_registered_on_both_surfaces(registered):
    assert "graph_claims" in registered.REGISTERED_TOOLS
    assert registered.ACTION_TOOL_ROUTES.get("graph_claims") == "/graph/claims"


@pytest.mark.asyncio
async def test_full_lifecycle_through_the_shared_action_core(
    registered, stub_engine, allow_policy
):
    """propose -> validate -> accept -> deprecate -> retract, each dispatched
    through kg_server._execute_tool — the SAME core both MCP and REST use."""
    propose = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="propose", claim_id="claim:mcp-1", reason="mined"
        )
    )
    assert propose["transition"]["to_state"] == "proposed"
    assert propose["transition"]["from_state"] == ""

    validate = json.loads(
        await kg_server._execute_tool(
            "graph_claims",
            action="validate",
            claim_id="claim:mcp-1",
            valid=True,
            reason="governance passed",
        )
    )
    assert validate["transition"]["to_state"] == "validated"

    accept = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="accept", claim_id="claim:mcp-1"
        )
    )
    assert accept["transition"]["to_state"] == "accepted"
    # The ActionPolicy decision (from the allowed gate) rides along as the
    # accepted transition's recorded `action_decision`.
    assert accept["transition"]["action_decision"] == "allow"

    deprecate = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="deprecate", claim_id="claim:mcp-1"
        )
    )
    assert deprecate["transition"]["to_state"] == "deprecated"

    retract = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="retract", claim_id="claim:mcp-1"
        )
    )
    assert retract["transition"]["to_state"] == "retracted"

    get_res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="get", claim_id="claim:mcp-1"
        )
    )
    assert get_res["current_state"] == "retracted"
    assert [h["to_state"] for h in get_res["history"]] == [
        "proposed",
        "validated",
        "accepted",
        "deprecated",
        "retracted",
    ]


@pytest.mark.asyncio
async def test_validate_false_holds_without_advancing(
    registered, stub_engine, allow_policy
):
    await kg_server._execute_tool(
        "graph_claims", action="propose", claim_id="claim:hold-1"
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_claims",
            action="validate",
            claim_id="claim:hold-1",
            valid=False,
            reason="shacl: conforms=false",
        )
    )
    assert res["transition"] is None  # a hold, not a state-advancing transition

    get_res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="get", claim_id="claim:hold-1"
        )
    )
    assert get_res["current_state"] == "proposed"
    assert any(h["reason"] == "shacl: conforms=false" for h in get_res["history"])


@pytest.mark.asyncio
async def test_illegal_transition_returns_a_typed_error_not_a_crash(
    registered, stub_engine, allow_policy
):
    await kg_server._execute_tool(
        "graph_claims", action="propose", claim_id="claim:illegal-1"
    )
    # PROPOSED cannot jump straight to ACCEPTED — must pass through VALIDATED.
    res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="accept", claim_id="claim:illegal-1"
        )
    )
    assert res["error"] == "illegal_transition"

    # The illegal attempt did not silently change the recorded state.
    get_res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="get", claim_id="claim:illegal-1"
        )
    )
    assert get_res["current_state"] == "proposed"


@pytest.mark.asyncio
async def test_denied_policy_blocks_every_state_changing_action(
    registered, stub_engine, monkeypatch
):
    """A denied ActionPolicy verdict must block the mutation for EVERY
    state-changing action — the flywheel is never even called."""
    import agent_utilities.mcp.tools.claim_tools as ct

    monkeypatch.setattr(
        ct,
        "_gate",
        lambda kind, target, reason: (
            False,
            {"decision": "queue_approval", "tier": "approval_required"},
        ),
    )

    for action in ("propose", "validate", "accept", "deprecate", "retract"):
        res = json.loads(
            await kg_server._execute_tool(
                "graph_claims", action=action, claim_id="claim:denied-1"
            )
        )
        assert res["error"] == "policy_denied", f"{action} was not blocked"
        assert res["policy"]["decision"] == "queue_approval"

    # Nothing was ever written — the claim never even reached PROPOSED.
    get_res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="get", claim_id="claim:denied-1"
        )
    )
    assert get_res["current_state"] == "proposed"  # ClaimFlywheel's no-history default
    assert get_res["history"] == []


@pytest.mark.asyncio
async def test_list_enumerates_and_filters_by_current_state(
    registered, stub_engine, allow_policy
):
    for cid in ("claim:list-1", "claim:list-2", "claim:list-3"):
        await kg_server._execute_tool("graph_claims", action="propose", claim_id=cid)
    await kg_server._execute_tool(
        "graph_claims", action="validate", claim_id="claim:list-2", valid=True
    )

    all_claims = json.loads(
        await kg_server._execute_tool("graph_claims", action="list")
    )
    ids = {c["claim_id"] for c in all_claims["claims"]}
    assert {"claim:list-1", "claim:list-2", "claim:list-3"} <= ids

    proposed_only = json.loads(
        await kg_server._execute_tool("graph_claims", action="list", state="proposed")
    )
    proposed_ids = {c["claim_id"] for c in proposed_only["claims"]}
    assert "claim:list-1" in proposed_ids
    assert "claim:list-2" not in proposed_ids  # advanced past PROPOSED

    validated_only = json.loads(
        await kg_server._execute_tool("graph_claims", action="list", state="validated")
    )
    assert {c["claim_id"] for c in validated_only["claims"]} == {"claim:list-2"}


@pytest.mark.asyncio
async def test_real_unstubbed_action_policy_denies_by_default(registered, stub_engine):
    """No ``_gate`` stub here: the REAL ``get_action_policy(...).decide(...)``
    is consulted. ``claim.*`` has no explicit rule in the shipped default
    policy, so it falls to the conservative default tier
    (``approval_required`` — the SAME posture ``secret.set``/``secret.delete``
    rely on with no override either), which is not an allowing decision —
    proving the gate is genuinely wired, not merely correctly branched on."""
    res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="accept", claim_id="claim:real-gate-1"
        )
    )
    assert res["error"] == "policy_denied"
    assert res["policy"]["decision"] in ("queue_approval", "deny")

    get_res = json.loads(
        await kg_server._execute_tool(
            "graph_claims", action="get", claim_id="claim:real-gate-1"
        )
    )
    assert get_res["current_state"] == "proposed"
    assert get_res["history"] == []


@pytest.mark.asyncio
async def test_missing_claim_id_and_unknown_action(registered, stub_engine):
    res = json.loads(await kg_server._execute_tool("graph_claims", action="accept"))
    assert "requires claim_id" in res["error"]

    res = json.loads(await kg_server._execute_tool("graph_claims", action="get"))
    assert "requires claim_id" in res["error"]

    res = json.loads(await kg_server._execute_tool("graph_claims", action="bogus"))
    assert "unknown action" in res["error"]
