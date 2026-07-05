"""Wire-First: the real graph_* MCP tools route through the connection registry
when a ``target`` is supplied (CONCEPT:AU-KG.backend.multi-connection-registry).

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


async def test_store_memory_accepts_raw_content_string(monkeypatch):
    """Regression: graph_write(store_memory) passes ``properties`` as the RAW memory
    content, not a JSON dict. The dispatcher must not eagerly ``json.loads`` it — that
    raised "Write error: Expecting value: line 1 column 1 (char 0)" on plain content
    text and broke graph_memory store/recall via the served surface.
    CONCEPT:AU-KG.memory.unified-memory-crud-core
    """
    recorded: dict = {}

    class _MemoryEngine(_FakeEngine):
        # Canonical store lives on the engine facade (MemoryMixin.store_memory).
        def store_memory(self, *, content, memory_type, tags, agent_id):
            recorded.update(
                content=content, memory_type=memory_type, tags=tags, agent_id=agent_id
            )

    default_engine = _MemoryEngine("default")
    registry = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    kg_server._CONNECTION_REGISTRY = registry

    kg_server.ensure_tools_registered()
    out = await kg_server._execute_tool(
        "graph_write",
        action="store_memory",
        node_type="episodic",
        properties="phase2 benchmark warm-read probe",  # RAW content, not JSON
        nodes='["bench"]',
        agent_id="bench-probe",
    )

    assert "Memory stored." in out
    assert "Expecting value" not in out
    assert "not available" not in out
    assert recorded["content"] == "phase2 benchmark warm-read probe"
    assert recorded["memory_type"] == "episodic"
    assert recorded["tags"] == ["bench"]
    assert recorded["agent_id"] == "bench-probe"


class _CASBackend(_FakeBackend):
    """Records the compare_and_set call and returns a configurable result."""

    def __init__(self, result: bool):
        self.result = result
        self.calls: list[tuple[str, dict, dict]] = []

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        self.calls.append((node_id, conditions, updates))
        return self.result


class _CASEngine(_FakeEngine):
    def __init__(self, label, result: bool):
        super().__init__(label)
        self.backend = _CASBackend(result)


def _install_cas_registry(result: bool) -> _CASEngine:
    engine = _CASEngine("default", result)
    registry = ConnectionRegistry(default_engine_provider=lambda: engine)
    kg_server._CONNECTION_REGISTRY = registry
    kg_server.ensure_tools_registered()
    return engine


async def test_compare_and_set_calls_backend_and_returns_applied_true():
    # The compare_and_set action must call the backend's
    # compare_and_set_node_fields with the EXACT node_id/conditions/updates and
    # surface a True result as applied=True (CONCEPT:AU-KG.compute.user-override-prompt-library).
    engine = _install_cas_registry(result=True)
    out = await kg_server._execute_tool(
        "graph_write",
        action="compare_and_set",
        node_id="task-1",
        conditions={"status": "pending"},
        updates={"status": "claimed", "owner": "agent-7"},
    )
    payload = json.loads(out)
    assert payload == {
        "action": "compare_and_set",
        "node_id": "task-1",
        "applied": True,
    }
    assert engine.backend.calls == [
        ("task-1", {"status": "pending"}, {"status": "claimed", "owner": "agent-7"})
    ]


async def test_compare_and_set_surfaces_false_result():
    # A lost race / failed precondition (backend returns False) must be surfaced
    # as applied=False, never swallowed.
    engine = _install_cas_registry(result=False)
    out = await kg_server._execute_tool(
        "graph_write",
        action="compare_and_set",
        node_id="task-1",
        conditions={"status": "pending"},
        updates={"status": "claimed"},
    )
    payload = json.loads(out)
    assert payload["applied"] is False
    assert payload["node_id"] == "task-1"
    assert engine.backend.calls == [
        ("task-1", {"status": "pending"}, {"status": "claimed"})
    ]


async def test_compare_and_set_coerces_omitted_dicts():
    # When conditions/updates are omitted (the REST body / default_factory case,
    # which the dispatcher leaves as an unresolved FieldInfo, not {}), the handler
    # must coerce them to empty dicts rather than pass a FieldInfo to the backend.
    engine = _install_cas_registry(result=True)
    out = await kg_server._execute_tool(
        "graph_write",
        action="compare_and_set",
        node_id="task-1",
    )
    payload = json.loads(out)
    assert payload["applied"] is True
    assert engine.backend.calls == [("task-1", {}, {})]


async def test_compare_and_set_accepts_json_string_dicts():
    # Some MCP clients send dict params as JSON strings; the handler parses them.
    engine = _install_cas_registry(result=True)
    out = await kg_server._execute_tool(
        "graph_write",
        action="compare_and_set",
        node_id="task-1",
        conditions='{"status": "pending"}',
        updates='{"status": "claimed"}',
    )
    payload = json.loads(out)
    assert payload["applied"] is True
    assert engine.backend.calls == [
        ("task-1", {"status": "pending"}, {"status": "claimed"})
    ]


async def test_compare_and_set_requires_node_id():
    # Missing node_id is a clear error, and the backend is never called.
    engine = _install_cas_registry(result=True)
    out = await kg_server._execute_tool(
        "graph_write",
        action="compare_and_set",
        conditions={"status": "pending"},
        updates={"status": "claimed"},
    )
    assert "node_id required" in out
    assert engine.backend.calls == []
