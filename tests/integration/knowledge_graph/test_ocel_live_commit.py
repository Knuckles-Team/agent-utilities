"""Live-engine proof that OCEL 'mine' mode actually reaches the graph (D-61-4).

CONCEPT:AU-KG.mining.ocel-lossless-roundtrip — ``tests/unit/test_engine_surface_tools.py``
proves the OCEL commit WIRING against a mocked engine boundary (right function
called, right envelope shape). That is necessary but not sufficient: a mock can
certify the call was made without certifying the call actually persists anything a
reader can see. This module closes that gap with a REAL, ephemeral, redb-backed
``epistemic-graph`` engine (CONCEPT:AU-KG.memory.provides-real-ephemeral-one, the
same ``engine_graph``/``tiny_engine`` fixtures every other live-engine test in this
suite uses) — it commits an OCEL document through the production ``graph_mine``
MCP tool, then queries the SAME engine with Cypher and asserts the committed
``ProcessEvent``/``BusinessObject``/``ProcessPerspective`` nodes are there.

Only the trace-discovery sub-call (``mining.process`` — an unrelated
directly-follows discovery step, not the commit this test is about) is stubbed,
matching the existing mocked-boundary test's scope. The commit path itself —
``kg_server._get_engine()`` → ``ingest_envelope`` → the real engine's native
``ApplyChangeEnvelope`` — is never mocked.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_surface_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

_OCEL_MINE_FIXTURE = {
    "eventTypes": [{"name": "create", "attributes": []}],
    "objectTypes": [{"name": "Order", "attributes": []}],
    "events": [
        {
            "id": "e1",
            "type": "create",
            "time": "2026-01-01T00:00:00Z",
            "attributes": [],
            "relationships": [{"objectId": "order-1", "qualifier": "order"}],
        }
    ],
    "objects": [
        {
            "id": "order-1",
            "type": "Order",
            "attributes": [],
            "relationships": [],
        }
    ],
}


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


@pytest.fixture
def tools() -> dict[str, object]:
    mcp = _CollectingMCP()
    engine_surface_tools.register_engine_surface_tools(mcp)
    return mcp.tools


@pytest.mark.engine
def test_ocel_mine_mode_commits_to_a_real_engine_and_is_queryable(
    engine_graph, monkeypatch, tools
) -> None:
    """``graph_mine(action="process", ocel_mode="mine")`` against a REAL engine:
    the committed nodes must be readable back via Cypher on that same engine —
    the actual claim D-61-4 asks for, not just that a commit function was called."""
    from types import SimpleNamespace

    from _test_engine import TEST_TENANT

    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # A REAL engine backend, bound to the SAME tenant graph the active
    # GraphSession (from ``engine_graph``) already authorizes — never a mock.
    real_engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: real_engine)

    # Only the unrelated trace-discovery sub-call is stubbed (matches the
    # mocked-boundary unit test's scope) — the commit itself is never touched.
    mining = SimpleNamespace(process=lambda **kwargs: {"traces": kwargs.get("traces")})
    monkeypatch.setattr(
        engine_surface_tools,
        "_client",
        lambda graph: SimpleNamespace(mining=mining),
    )

    # The ChangeEnvelope's tenant must match the active GraphSession's tenant
    # (``_native_session``'s cross-tenant guard) — ``engine_graph`` already
    # minted a session for ``TEST_TENANT``, so the OCEL commit declares the
    # same tenant rather than an arbitrary one.
    with use_actor(
        ActorContext(
            actor_id="ocel-live-test",
            actor_type=ActorType.SYSTEM,
            tenant_id=TEST_TENANT,
            authenticated=True,
        )
    ):
        out = json.loads(
            tools["graph_mine"](
                action="process",
                params_json=json.dumps(
                    {
                        "ocel_json": _OCEL_MINE_FIXTURE,
                        "tenant": TEST_TENANT,
                        "object_type": "Order",
                        "perspective_id": "case:order-view",
                        "derivation_version": "v1",
                    }
                ),
                graph="",
            )
        )

    assert out["tekg"]["commit_status"] == "success"

    # The proof: query the REAL engine — no mock in the read path either — and
    # find the exact committed nodes. Reads go through
    # ``EpistemicGraphBackend.execute_read`` directly (the same method
    # ``IntelligenceGraphEngine.query_cypher`` itself calls for the actual
    # Cypher round-trip) rather than through the higher ``query_cypher`` ACL
    # wrapper: freshly-committed content nodes carry no explicit per-node ACL
    # grant (no connector stamps one for plain content), so a non-privileged
    # actor's row-level permission filter denies them by default regardless of
    # tenant match — a separate, pre-existing content-governance gap tracked
    # as D-W6F-1, not a defect in the OCEL commit path this test verifies.
    def _read(query: str) -> list[dict]:
        return real_engine.backend.execute_read(query, {"_clearance_level": 999})

    events = _read(
        "MATCH (n:ProcessEvent) RETURN n.id AS id, "
        "n.source_record_id AS source_record_id, n.activity AS activity"
    )
    assert any(
        (row or {}).get("source_record_id") == "e1"
        and (row or {}).get("activity") == "create"
        for row in events or []
    ), f"committed ProcessEvent not queryable via Cypher: {events!r}"

    objects = _read(
        "MATCH (n:BusinessObject) RETURN n.id AS id, "
        "n.source_record_id AS source_record_id"
    )
    assert any(
        (row or {}).get("source_record_id") == "order-1" for row in objects or []
    ), f"committed BusinessObject not queryable via Cypher: {objects!r}"

    perspectives = _read(
        "MATCH (n:ProcessPerspective) RETURN n.id AS id, "
        "n.perspective_id AS perspective_id"
    )
    assert any(
        (row or {}).get("perspective_id") == "case:order-view"
        for row in perspectives or []
    ), f"committed ProcessPerspective not queryable via Cypher: {perspectives!r}"
