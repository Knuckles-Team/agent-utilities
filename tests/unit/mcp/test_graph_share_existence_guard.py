"""BUG-6 (kg-exhaustive-smoke.md): ``graph_share(action="org", node_id=...)``
reported success (``{"node_id": ..., "shared_scope": "org"}``) for a node id
that does not exist — ``tenant_sharing.share_with_org`` ran a bare Cypher
``MATCH (n {id: $id}) SET ...`` that silently matched zero rows, with no
signal back to the caller. Exercises the REAL ``graph_share`` MCP tool
dispatch end to end (``kg_server.REGISTERED_TOOLS["graph_share"]``).
"""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.core import tenant_sharing as ts
from agent_utilities.mcp import kg_server


class _FakeStore:
    """Existence-check-aware fake: only ``known_ids`` "exist"."""

    def __init__(self, known_ids: set[str]) -> None:
        self.known_ids = known_ids
        self.calls: list[tuple[str, dict]] = []

    def execute(self, cypher, params=None):
        params = params or {}
        self.calls.append((cypher, params))
        if params.get("id") in self.known_ids:
            return [{"id": params["id"]}]
        return []


def _get_tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["graph_share"]


def test_org_share_on_nonexistent_node_returns_clean_error(monkeypatch):
    store = _FakeStore(known_ids=set())
    monkeypatch.setattr(ts, "_store", lambda store_arg=None: store)

    tool = _get_tool()
    out = json.loads(tool(action="org", node_id="smoke-test-node-doesnotexist"))

    assert "error" in out
    assert "not found" in out["error"].lower()
    assert "shared_scope" not in out  # never reports a fake success


def test_org_share_on_real_node_succeeds(monkeypatch):
    store = _FakeStore(known_ids={"smoke:test-node-001"})
    monkeypatch.setattr(ts, "_store", lambda store_arg=None: store)

    tool = _get_tool()
    out = json.loads(tool(action="org", node_id="smoke:test-node-001"))

    assert out == {"node_id": "smoke:test-node-001", "shared_scope": "org"}


def test_private_share_on_nonexistent_node_returns_clean_error(monkeypatch):
    store = _FakeStore(known_ids=set())
    monkeypatch.setattr(ts, "_store", lambda store_arg=None: store)

    tool = _get_tool()
    out = json.loads(tool(action="private", node_id="smoke-test-node-doesnotexist"))

    assert "error" in out
    assert "not found" in out["error"].lower()
