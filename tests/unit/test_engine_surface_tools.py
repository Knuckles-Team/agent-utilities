"""Unit tests for the KG-2.310 engine-surface MCP tools.

CONCEPT:AU-KG.coordination.engine-message-broker — MCP surface for the new epistemic-graph v2.2.0 engine ops
(broker / kvcache / federated-search / promql / traces / gis). These tests assert
that each new tool (a) registers on both the MCP tool table and the REST route
table, (b) dispatches its action + params into a MOCK engine client / KV backend
and returns the expected JSON shape, and (c) DEGRADES CLEANLY (never raises) when
the connected engine build lacks the surface. No live engine is required.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from agent_utilities.kvcache.remote_backend import KvCacheStats
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_surface_tools


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
    """Register the KG-2.310 tools onto a collecting MCP and return them by name."""
    mcp = _CollectingMCP()
    engine_surface_tools.register_engine_surface_tools(mcp)
    return mcp.tools


# ── helpers ──────────────────────────────────────────────────────────────────
def _fake_client(**subs) -> SimpleNamespace:
    """A fake engine client exposing only the given sub-clients."""
    return SimpleNamespace(**subs)


def _recording_method(recorder: list, name: str):
    def _call(**kwargs):
        recorder.append((name, kwargs))
        return {"echoed": name, "kwargs": kwargs}

    return _call


# ── registration / parity ────────────────────────────────────────────────────
_EXPECTED_ROUTES = {
    "graph_broker": "/graph/broker",
    "graph_kvcache": "/graph/kvcache",
    "graph_federated_search": "/graph/federated-search",
    "graph_promql": "/graph/promql",
    "graph_traces": "/graph/traces",
    "graph_gis": "/graph/gis",
    "graph_memory": "/graph/memory",
}


def test_kg_2_310_all_tools_registered_with_rest_twins(tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — every new tool is on the MCP table AND has a REST twin."""
    for name, route in _EXPECTED_ROUTES.items():
        assert name in tools, f"{name} not registered as MCP tool"
        assert kg_server.REGISTERED_TOOLS.get(name) is not None
        assert kg_server.ACTION_TOOL_ROUTES.get(name) == route


# ── graph_broker ─────────────────────────────────────────────────────────────
def test_kg_2_310_broker_dispatches_to_client(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_broker publish routes into client.broker.publish."""
    calls: list = []
    broker = SimpleNamespace(publish=_recording_method(calls, "publish"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(broker=broker)
    )
    out = json.loads(
        tools["graph_broker"](
            action="publish",
            exchange="ex1",
            queue="",
            routing_key="rk",
            payload="hello",
            exchange_type="",
            params_json='{"durable": true}',
            graph="",
        )
    )
    assert out["surface"] == "broker"
    assert out["action"] == "publish"
    assert out["result"]["echoed"] == "publish"
    assert calls == [
        (
            "publish",
            {
                "exchange": "ex1",
                "routing_key": "rk",
                "payload": "hello",
                "durable": True,
            },
        )
    ]


def test_kg_2_310_broker_degrades_when_surface_absent(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_broker degrades cleanly when no broker surface."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_broker"](
            action="publish",
            exchange="ex",
            queue="",
            routing_key="rk",
            payload="hi",
            exchange_type="",
            params_json="{}",
            graph="",
        )
    )
    assert out["degraded"] is True
    assert out["surface"] == "broker"
    assert "not available" in out["error"]


# ── graph_kvcache ────────────────────────────────────────────────────────────
class _FakeKV:
    def __init__(self, store=None) -> None:
        self.store = dict(store or {})
        self.closed = False

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value) -> bool:
        self.store[key] = value
        return True

    def contains(self, key) -> bool:
        return key in self.store

    def exists(self, key) -> bool:
        return key in self.store

    def stats(self) -> KvCacheStats:
        return KvCacheStats(unique_blocks=len(self.store), total_refs=len(self.store))

    def close(self) -> None:
        self.closed = True


def test_kg_2_310_kvcache_get_hit_and_miss(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_kvcache get returns base64 hit / clean miss."""
    kv = _FakeKV({"k1": b"abc"})
    monkeypatch.setattr(engine_surface_tools, "_kv_backend", lambda: kv)

    hit = json.loads(tools["graph_kvcache"](action="get", key="k1", value_b64=""))
    assert hit["hit"] is True
    assert base64.b64decode(hit["value_b64"]) == b"abc"

    miss = json.loads(tools["graph_kvcache"](action="get", key="nope", value_b64=""))
    assert miss["hit"] is False
    assert miss["value_b64"] is None
    assert kv.closed is True  # connector closed after use


def test_kg_2_310_kvcache_put_and_stats(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_kvcache put stores bytes; stats reports counters."""
    kv = _FakeKV()
    monkeypatch.setattr(engine_surface_tools, "_kv_backend", lambda: kv)

    stored = json.loads(
        tools["graph_kvcache"](
            action="put", key="k2", value_b64=base64.b64encode(b"xyz").decode()
        )
    )
    assert stored["stored"] is True
    assert kv.store["k2"] == b"xyz"

    stats = json.loads(tools["graph_kvcache"](action="stats", key="", value_b64=""))
    assert stats["result"]["unique_blocks"] == 1


def test_kg_2_310_kvcache_contains(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_kvcache contains/exists probe the backend."""
    kv = _FakeKV({"k": b"v"})
    monkeypatch.setattr(engine_surface_tools, "_kv_backend", lambda: kv)
    out = json.loads(tools["graph_kvcache"](action="contains", key="k", value_b64=""))
    assert out["present"] is True


# ── graph_federated_search ───────────────────────────────────────────────────
def test_kg_2_310_federated_search_dispatches(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_federated_search routes into a search sub-client."""
    calls: list = []
    search = SimpleNamespace(
        federated_search=_recording_method(calls, "federated_search")
    )
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(search=search)
    )
    out = json.loads(
        tools["graph_federated_search"](
            query="who calls foo",
            references="ref1, ref2",
            top_k=5,
            params_json="{}",
            graph="",
        )
    )
    assert out["surface"] == "federated_search"
    assert calls[0][1] == {
        "query": "who calls foo",
        "top_k": 5,
        "references": ["ref1", "ref2"],
    }


def test_kg_2_310_federated_search_degrades(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_federated_search degrades when absent."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_federated_search"](
            query="q", references="", top_k=10, params_json="{}", graph=""
        )
    )
    assert out["degraded"] is True


# ── graph_promql ─────────────────────────────────────────────────────────────
def test_kg_2_310_promql_instant_and_range(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_promql routes instant/range to the right method."""
    calls: list = []
    obs = SimpleNamespace(
        promql=_recording_method(calls, "promql"),
        promql_range=_recording_method(calls, "promql_range"),
    )
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(observability=obs)
    )
    inst = json.loads(
        tools["graph_promql"](
            query="up",
            action="instant",
            time="",
            start="",
            end="",
            step="",
            params_json="{}",
            graph="",
        )
    )
    assert inst["action"] == "instant"
    rng = json.loads(
        tools["graph_promql"](
            query="up",
            action="range",
            time="",
            start="0",
            end="10",
            step="30s",
            params_json="{}",
            graph="",
        )
    )
    assert rng["action"] == "range"
    assert [c[0] for c in calls] == ["promql", "promql_range"]
    assert calls[1][1] == {"query": "up", "start": "0", "end": "10", "step": "30s"}


def test_kg_2_310_promql_degrades(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_promql degrades when no metrics surface."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_promql"](
            query="up",
            action="instant",
            time="",
            start="",
            end="",
            step="",
            params_json="{}",
            graph="",
        )
    )
    assert out["degraded"] is True


# ── graph_traces ─────────────────────────────────────────────────────────────
def test_kg_2_310_traces_search_and_get(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_traces routes search/get to the trace surface."""
    calls: list = []
    obs = SimpleNamespace(
        search_traces=_recording_method(calls, "search_traces"),
        get_trace=_recording_method(calls, "get_trace"),
    )
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(observability=obs)
    )
    s = json.loads(
        tools["graph_traces"](
            action="search",
            trace_id="",
            service="svc",
            operation="",
            query="",
            limit=7,
            params_json="{}",
            graph="",
        )
    )
    assert s["action"] == "search"
    assert calls[0][1] == {"service": "svc", "limit": 7}
    g = json.loads(
        tools["graph_traces"](
            action="get",
            trace_id="t123",
            service="",
            operation="",
            query="",
            limit=20,
            params_json="{}",
            graph="",
        )
    )
    assert g["action"] == "get"
    assert calls[1] == ("get_trace", {"trace_id": "t123"})


def test_kg_2_310_traces_degrades(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_traces degrades when no trace surface."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_traces"](
            action="search",
            trace_id="",
            service="",
            operation="",
            query="",
            limit=20,
            params_json="{}",
            graph="",
        )
    )
    assert out["degraded"] is True


# ── graph_gis ────────────────────────────────────────────────────────────────
def test_kg_2_310_gis_dispatches(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_gis routes an action into client.gis.<action>."""
    calls: list = []
    gis = SimpleNamespace(route=_recording_method(calls, "route"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(gis=gis)
    )
    out = json.loads(
        tools["graph_gis"](
            action="route",
            params_json='{"from": [1, 2], "to": [3, 4]}',
            graph="",
        )
    )
    assert out["surface"] == "gis"
    assert calls == [("route", {"from": [1, 2], "to": [3, 4]})]


def test_kg_2_310_gis_degrades(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_gis degrades when no GIS surface."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(tools["graph_gis"](action="route", params_json="{}", graph=""))
    assert out["degraded"] is True


# ── graph_memory (EG-318) ────────────────────────────────────────────────────
def test_kg_2_310_memory_trajectory_dispatches(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_memory append_step routes into client.trajectory."""
    calls: list = []
    trajectory = SimpleNamespace(append_step=_recording_method(calls, "append_step"))
    monkeypatch.setattr(
        engine_surface_tools,
        "_client",
        lambda graph: _fake_client(trajectory=trajectory),
    )
    out = json.loads(
        tools["graph_memory"](
            action="append-step",  # dash form normalizes to append_step
            params_json='{"trajectory_id": "t1", "step": {"reward": 1.0}}',
            graph="",
        )
    )
    assert out["surface"] == "memory"
    assert out["action"] == "append_step"
    assert calls == [("append_step", {"trajectory_id": "t1", "step": {"reward": 1.0}})]


def test_kg_2_310_memory_create_summary_dispatches(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_memory create_summary routes into client.memory."""
    calls: list = []
    memory = SimpleNamespace(create_summary=_recording_method(calls, "create_summary"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(memory=memory)
    )
    out = json.loads(
        tools["graph_memory"](
            action="create_summary",
            params_json='{"node_ids": ["n1", "n2"]}',
            graph="",
        )
    )
    assert out["result"]["echoed"] == "create_summary"
    assert calls == [("create_summary", {"node_ids": ["n1", "n2"]})]


def test_kg_2_310_memory_read_by_action_name(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — an unlisted read action probes the three sub-clients."""
    calls: list = []
    scene = SimpleNamespace(get_scene=_recording_method(calls, "get_scene"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(scene=scene)
    )
    out = json.loads(
        tools["graph_memory"](action="get_scene", params_json="{}", graph="")
    )
    assert out["action"] == "get_scene"
    assert calls == [("get_scene", {})]


def test_kg_2_310_memory_degrades(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — graph_memory degrades when no memory surface."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_memory"](action="consolidate", params_json="{}", graph="")
    )
    assert out["degraded"] is True
    assert out["surface"] == "memory"


# ── transport failure ────────────────────────────────────────────────────────
def test_kg_2_310_engine_unavailable_is_reported(monkeypatch, tools):
    """CONCEPT:AU-KG.coordination.engine-message-broker — an unreachable engine is surfaced as data, not raised."""

    def _boom(graph):
        raise ConnectionError("engine down")

    monkeypatch.setattr(engine_surface_tools, "_client", _boom)
    out = json.loads(tools["graph_gis"](action="route", params_json="{}", graph=""))
    assert "engine unavailable" in out["error"]
