"""Wire-First: the real graph_* MCP tools route through the connection registry
when a ``target`` is supplied (CONCEPT:KG-2.63).

These call the tools via ``kg_server._execute_tool`` (the same dispatch the MCP
and REST surfaces use, which resolves ``Field`` defaults) and assert the registry
was actually consulted — without standing up real backends or the daemon.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import ConnectionRegistry
from agent_utilities.mcp import kg_server


class _FakeBackend:
    cypher_support = "full"
    supports_sparql = False

    def close(self):
        pass


class _FakeEngine:
    def __init__(self, label):
        self.label = label
        self.backend = _FakeBackend()

    def query_cypher(self, cypher, params=None, as_of=None):
        return [{"engine": self.label}]


@pytest.fixture(autouse=True)
def _reset_registry():
    saved = kg_server._CONNECTION_REGISTRY
    kg_server._CONNECTION_REGISTRY = None
    yield
    kg_server._CONNECTION_REGISTRY = saved


async def test_unknown_target_is_reported_via_registry():
    # A single unknown named target must surface a registry KeyError as a tool
    # error — proving the tool consulted the registry (no daemon needed; the
    # unknown name raises before any engine is built).
    kg_server.ensure_tools_registered()
    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (n) RETURN n AS n",
        target="does-not-exist",
    )
    payload = json.loads(out)
    assert "error" in payload
    assert "does-not-exist" in payload["error"]


async def test_fanout_returns_labeled_per_connection_results():
    # Install a registry whose engines are fakes, then fan out via target="all".
    default_engine = _FakeEngine("default")
    registry = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    registry._build_engine = lambda spec: _FakeEngine(spec.get("_label", "named"))  # type: ignore[method-assign]
    registry.register("other", {"backend": "memory", "_label": "other"})
    kg_server._CONNECTION_REGISTRY = registry

    kg_server.ensure_tools_registered()
    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (n) RETURN n AS n",
        target="all",
    )
    payload = json.loads(out)
    assert set(payload["targets"]) == {"default", "other"}
    assert payload["targets"]["default"] == [{"engine": "default"}]
    assert payload["targets"]["other"] == [{"engine": "other"}]
    assert payload["errors"] == {}


async def test_write_does_not_fanout_on_default(monkeypatch):
    # graph_write with no target must hit exactly the default engine (single
    # write), never fan out.
    calls = []

    class _WriteEngine(_FakeEngine):
        def add_node(self, node_id, node_type, props=None):
            calls.append((self.label, node_id))

    default_engine = _WriteEngine("default")
    registry = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    registry._build_engine = lambda spec: _WriteEngine("other")  # type: ignore[method-assign]
    registry.register("other", {"backend": "memory"})
    kg_server._CONNECTION_REGISTRY = registry

    kg_server.ensure_tools_registered()
    out = await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id="n1",
        node_type="Thing",
    )
    assert "added" in out
    assert calls == [("default", "n1")]  # only the default engine was written
