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

from agent_utilities.kvcache import KVCheckpointStore
from agent_utilities.kvcache.remote_backend import KvCacheStats
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_surface_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}
        self.descriptions: dict[str, str] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            self.descriptions[name] = description
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
    "graph_kv_checkpoint": "/graph/kv_checkpoint",
    "graph_federated_search": "/graph/federated-search",
    "graph_promql": "/graph/promql",
    "graph_traces": "/graph/traces",
    "graph_gis": "/graph/gis",
    "graph_memory": "/graph/memory",
}

# The full 18-action graph_mine surface (CONCEPT:EG-KG.mining.frequent-itemset-mining
# / W0.9) — matches the client-introspected MiningClient method set 1:1.
_MINING_ACTIONS = (
    "associate",
    "cluster",
    "anomaly",
    "classify_fit",
    "classify_predict",
    "reduce",
    "sequence",
    "forecast",
    "text",
    "subgraph",
    "entity_resolve",
    "causal_impact",
    "process",
    "root_cause",
    "risk_propagation",
    "ontology_gap",
    "retrieval_quality",
    "community",
)


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


def test_engine_surface_failure_redacts_transport_exception(monkeypatch, caplog):
    sensitive = "https://identity:credential@private.invalid/local/path"

    def _fail(_graph):
        raise RuntimeError(sensitive)

    monkeypatch.setattr(engine_surface_tools, "_client", _fail)
    out = json.loads(
        engine_surface_tools._invoke(
            surface="broker",
            action="publish",
            graph="",
            candidates=(("broker", "publish"),),
            params={},
        )
    )

    assert out["surface"] == "broker"
    assert out["action"] == "publish"
    assert out["error"]["code"] == "dependency_unavailable"
    assert sensitive not in json.dumps(out)
    assert sensitive not in caplog.text
    assert "RuntimeError" in caplog.text


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


# ── graph_kv_checkpoint (CONCEPT:AU-KG.memory.kv-checkpoint-resource) ───────
class _FakeCheckpointBlob:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def store(self, data: bytes) -> str:
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        self.data[digest] = data
        return digest

    def incref(self, digest: str) -> int:
        return 1

    def unref(self, digest: str) -> int:
        return 0

    def fetch(self, digest: str) -> bytes:
        return self.data[digest]


class _FakeCheckpointTxn:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self._n = 0

    def begin(self, graph=None):
        self._n += 1
        return f"txn{self._n}"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = dict(props)

    def blob_ref(self, txn, node_id, digest):
        return True

    def commit(self, txn) -> bool:
        return True


class _FakeCheckpointNodes:
    def __init__(self, backing: dict[str, dict]) -> None:
        self._backing = backing

    def has(self, node_id: str) -> bool:
        return node_id in self._backing

    def properties(self, node_id: str):
        return self._backing.get(node_id)


class _FakeCheckpointEdges:
    def __init__(self) -> None:
        self.edges: list[tuple] = []

    def add(self, source, dest, props):
        self.edges.append((source, dest, props))


def _fake_checkpoint_store():
    """A real :class:`KVCheckpointStore` bound to a fake engine client, matching
    ``tests/unit/kvcache/test_checkpoint.py``'s pattern — proves the MCP tool wires
    through to the real store logic, not a hand-rolled tool-level stand-in."""
    client = SimpleNamespace(
        blob=_FakeCheckpointBlob(),
        txn=_FakeCheckpointTxn(),
        nodes=None,
        edges=_FakeCheckpointEdges(),
    )
    client.nodes = _FakeCheckpointNodes(client.txn.nodes)
    return KVCheckpointStore(SimpleNamespace(_client=client, graph_name="__commons__"))


def _kv_checkpoint_call(tools, **overrides):
    """Call ``graph_kv_checkpoint`` with every parameter explicitly supplied.

    The ``_CollectingMCP`` fixture captures the RAW undecorated tool function, so a
    parameter left unspecified keeps its literal ``pydantic.Field(...)`` default
    object (a ``FieldInfo``, not the resolved default value) — the real
    ``mcp.tool()`` decorator resolves that, this fake doesn't. Every call site must
    therefore pass the full parameter set, like every other tool test in this file.
    """
    defaults = dict(
        action="",
        data_b64="",
        model_identity="",
        quantization="",
        serving_engine="",
        engine_version="",
        prefix_digest="",
        tenant="",
        policy_version="",
        run_id="",
        point="",
        provenance_json="{}",
        checkpoint_id="",
        requesting_tenant="",
        current_policy_version="",
        new_run_id="",
        conversation_id="",
        allow_cold_start=False,
        graph="",
    )
    defaults.update(overrides)
    return json.loads(tools["graph_kv_checkpoint"](**defaults))


def test_kg_2_310_kv_checkpoint_create_then_instantiate_and_restore(monkeypatch, tools):
    """CONCEPT:AU-KG.memory.kv-checkpoint-resource — end-to-end through the MCP tool:
    create → instantiate_agent (lineage edge) → restore_conversation (bytes back)."""
    store = _fake_checkpoint_store()
    monkeypatch.setattr(engine_surface_tools, "_checkpoint_store", lambda graph: store)

    created = _kv_checkpoint_call(
        tools,
        action="create",
        data_b64=base64.b64encode(b"kv-bytes").decode(),
        model_identity="m",
        quantization="q",
        serving_engine="vllm",
        engine_version="1",
        prefix_digest="d",
        tenant="tenant-a",
        policy_version="p1",
        run_id="run-1",
        point="checkpoint-a",
    )
    checkpoint_id = created["result"]["checkpoint_id"]
    assert checkpoint_id

    instantiated = _kv_checkpoint_call(
        tools,
        action="instantiate_agent",
        checkpoint_id=checkpoint_id,
        requesting_tenant="tenant-a",
        new_run_id="run-2",
    )
    assert instantiated["result"]["checkpoint_id"] == checkpoint_id

    restored = _kv_checkpoint_call(
        tools,
        action="restore_conversation",
        checkpoint_id=checkpoint_id,
        conversation_id="conv-1",
        requesting_tenant="tenant-a",
    )
    assert restored["result"]["restored"] is True
    assert restored["result"]["cold_start"] is False
    # The heavy blob bytes are NEVER inlined over the MCP JSON surface.
    assert "data" not in restored["result"]
    assert restored["result"]["size_bytes"] == len(b"kv-bytes")


def test_kg_2_310_kv_checkpoint_cross_tenant_refused(monkeypatch, tools):
    """CONCEPT:AU-KG.memory.kv-checkpoint-resource — a cross-tenant load through the
    MCP tool surfaces a typed error, never a silent degrade."""
    store = _fake_checkpoint_store()
    monkeypatch.setattr(engine_surface_tools, "_checkpoint_store", lambda graph: store)

    created = _kv_checkpoint_call(
        tools,
        action="create",
        data_b64=base64.b64encode(b"kv-bytes").decode(),
        model_identity="m",
        quantization="q",
        serving_engine="vllm",
        engine_version="1",
        prefix_digest="d",
        tenant="tenant-a",
        run_id="run-1",
        point="checkpoint-a",
    )
    checkpoint_id = created["result"]["checkpoint_id"]

    denied = _kv_checkpoint_call(
        tools,
        action="restore_conversation",
        checkpoint_id=checkpoint_id,
        conversation_id="conv-1",
        requesting_tenant="tenant-b",
    )
    assert denied["error"]["code"] == "permission_denied"

    # allow_cold_start makes the fallback explicit instead of an error payload.
    cold = _kv_checkpoint_call(
        tools,
        action="restore_conversation",
        checkpoint_id=checkpoint_id,
        conversation_id="conv-1",
        requesting_tenant="tenant-b",
        allow_cold_start=True,
    )
    assert cold["result"]["cold_start"] is True
    assert cold["result"]["fallback_reason"] == "CrossTenantCheckpointError"


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
    assert out["error"]["code"] == "dependency_unavailable"


# ── graph_mine (CONCEPT:EG-KG.mining.frequent-itemset-mining — W0.9 full 18-action surface) ──
def test_graph_mine_registered_with_rest_twin(tools):
    assert "graph_mine" in tools
    assert kg_server.REGISTERED_TOOLS.get("graph_mine") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_mine") == "/mining/associate"


def test_graph_mine_description_enumerates_all_18_actions():
    """The tool description documents the full 18-action mining surface — not
    just the 10 it originally shipped with."""
    mcp = _CollectingMCP()
    engine_surface_tools.register_engine_surface_tools(mcp)
    description = mcp.descriptions["graph_mine"]
    missing = [name for name in _MINING_ACTIONS if f"'{name}'" not in description]
    assert not missing, f"graph_mine description is missing actions: {missing}"


def test_graph_mine_manifest_matches_introspected_mining_client():
    """CONCEPT:AU-KG.compute.engine-surface-manifest — the SAME manifest
    engine_mining is built from (client-introspected, not hand-maintained) has
    exactly the 18 real MiningClient methods."""
    assert engine_surface_tools._mining_actions() == frozenset(_MINING_ACTIONS)


def test_graph_mine_dispatches_a_newly_exposed_family(monkeypatch, tools):
    """One of the 8 families that were previously undocumented/unreachable by
    name (root_cause) now dispatches like any other action."""
    calls: list = []
    mining = SimpleNamespace(root_cause=_recording_method(calls, "root_cause"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    out = json.loads(
        tools["graph_mine"](
            action="root_cause",
            params_json=json.dumps(
                {
                    "nodes": ["a", "b"],
                    "scores": [0.1, 0.9],
                    "edges": [["a", "b", 1.0]],
                    "symptom": "b",
                }
            ),
            graph="",
        )
    )
    assert out["surface"] == "mining"
    assert out["action"] == "root_cause"
    assert calls == [
        (
            "root_cause",
            {
                "nodes": ["a", "b"],
                "scores": [0.1, 0.9],
                "edges": [["a", "b", 1.0]],
                "symptom": "b",
            },
        )
    ]


def test_graph_mine_alias_entity_resolution_hits_entity_resolve(monkeypatch, tools):
    """The guessable name 'entity_resolution' used to silently degrade (a
    MiningClient miss on a name that isn't its real 'entity_resolve' attr) —
    it must now resolve and dispatch for real."""
    calls: list = []
    mining = SimpleNamespace(entity_resolve=_recording_method(calls, "entity_resolve"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    out = json.loads(
        tools["graph_mine"](
            action="entity_resolution",
            params_json=json.dumps({"records": [["a"], ["a"]], "threshold": 0.5}),
            graph="",
        )
    )
    assert out.get("degraded") is not True
    assert out["surface"] == "mining"
    assert out["action"] == "entity_resolve"
    assert calls == [("entity_resolve", {"records": [["a"], ["a"]], "threshold": 0.5})]


def test_graph_mine_alias_process_mining_hits_process(monkeypatch, tools):
    """The guessable name 'process_mining' used to silently degrade (the real
    attr is 'process') — it must now resolve and dispatch for real."""
    calls: list = []
    mining = SimpleNamespace(process=_recording_method(calls, "process"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    out = json.loads(
        tools["graph_mine"](
            action="process_mining",
            params_json=json.dumps({"traces": [["a", "b"]]}),
            graph="",
        )
    )
    assert out.get("degraded") is not True
    assert out["action"] == "process"
    assert calls == [("process", {"traces": [["a", "b"]]})]


def test_graph_mine_projects_object_events_in_one_native_call(monkeypatch, tools):
    calls: list = []
    mining = SimpleNamespace(process=_recording_method(calls, "process"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    events = [
        {
            "event_id": "e2",
            "activity": "ship",
            "occurred_at": "2026-01-02T00:00:00Z",
            "source_ref": "source:2",
            "objects": [{"id": "o1", "type": "Order"}],
        },
        {
            "event_id": "e1",
            "activity": "create",
            "occurred_at": "2026-01-01T00:00:00Z",
            "source_ref": "source:1",
            "objects": [{"id": "o1", "type": "Order"}],
        },
    ]

    out = json.loads(
        tools["graph_mine"](
            action="process",
            params_json=json.dumps(
                {
                    "events": events,
                    "object_type": "Order",
                    "perspective_id": "case:order-view",
                    "derivation_version": "v1",
                }
            ),
            graph="",
        )
    )

    assert calls == [("process", {"traces": [["create", "ship"]]})]
    assert out["projection"] == {
        "mode": "object_type_projection",
        "object_type": "Order",
        "perspective_id": "case:order-view",
        "derivation_version": "v1",
        "trace_count": 1,
        "event_count": 2,
        "source_count": 2,
        "lineage_digest": out["projection"]["lineage_digest"],
    }
    assert len(out["projection"]["lineage_digest"]) == 64


def test_graph_mine_event_projection_without_a_declared_perspective_is_refused(
    tools,
) -> None:
    """CONCEPT:AU-KG.mining.governed-perspective-flattening — a bare 'object_type'
    with no 'perspective_id'/'derivation_version' is an undisclosed flattening
    attempt and is refused, not silently defaulted."""
    out = json.loads(
        tools["graph_mine"](
            action="process",
            params_json=json.dumps({"events": [], "object_type": "Order"}),
            graph="",
        )
    )
    assert out["error"]["code"] == "invalid_request"


def test_graph_mine_event_projection_writeback_fails_closed(tools):
    out = json.loads(
        tools["graph_mine"](
            action="process",
            params_json=json.dumps(
                {"events": [], "object_type": "Order", "writeback": True}
            ),
            graph="",
        )
    )
    assert out["code"] == "lineage_required"
    assert "writeback is disabled" in out["error"]


def test_graph_mine_ocel_validate_live_path_uses_the_registered_process_tool(tools):
    """The existing MCP/REST process surface reaches the governed OCEL seam."""
    ocel = {
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

    with use_actor(
        ActorContext(
            actor_id="ocel-test",
            actor_type=ActorType.SYSTEM,
            tenant_id="tenant-a",
            authenticated=True,
        )
    ):
        out = json.loads(
            tools["graph_mine"](
                action="process",
                params_json=json.dumps(
                    {
                        "ocel_json": ocel,
                        "tenant": "tenant-a",
                        "ocel_mode": "validate",
                    }
                ),
                graph="",
            )
        )

    assert set(out["ocel"]) == {"eventTypes", "objectTypes", "events", "objects"}
    assert out["tekg"]["tenant"] == "tenant-a"
    assert len(out["tekg"]["idempotency_key"]) == 64
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_mine") == "/mining/associate"


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


def test_graph_mine_ocel_mine_mode_requires_a_declared_perspective(tools) -> None:
    """CONCEPT:AU-KG.mining.governed-perspective-flattening — 'mine' mode
    always derives traces, so it always requires the same disclosed
    perspective triple 'events' mode does; there is no separate silent path."""
    with use_actor(
        ActorContext(
            actor_id="ocel-test",
            actor_type=ActorType.SYSTEM,
            tenant_id="tenant-a",
            authenticated=True,
        )
    ):
        out = json.loads(
            tools["graph_mine"](
                action="process",
                params_json=json.dumps(
                    {"ocel_json": _OCEL_MINE_FIXTURE, "tenant": "tenant-a"}
                ),
                graph="",
            )
        )
    assert out["error"]["code"] == "invalid_request"


def test_graph_mine_ocel_mine_mode_commits_a_real_change_envelope(
    monkeypatch, tools
) -> None:
    """CONCEPT:AU-KG.mining.ocel-lossless-roundtrip — 'mine' mode (the
    default) does not just plan a ChangeEnvelope, it COMMITS it via the same
    ``ingest_envelope`` idiom every other connector uses, folding the
    disclosed ProcessPerspective into the same committed slice."""
    calls: list = []
    mining = SimpleNamespace(process=_recording_method(calls, "process"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: "fake-engine")

    committed: list = []

    def _fake_ingest_envelope(engine, envelope):
        assert engine == "fake-engine"
        committed.append(envelope)
        return {"status": "success"}

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module

    monkeypatch.setattr(
        envelope_ingest_module, "ingest_envelope", _fake_ingest_envelope
    )

    with use_actor(
        ActorContext(
            actor_id="ocel-test",
            actor_type=ActorType.SYSTEM,
            tenant_id="tenant-a",
            authenticated=True,
        )
    ):
        out = json.loads(
            tools["graph_mine"](
                action="process",
                params_json=json.dumps(
                    {
                        "ocel_json": _OCEL_MINE_FIXTURE,
                        "tenant": "tenant-a",
                        "object_type": "Order",
                        "perspective_id": "case:order-view",
                        "derivation_version": "v1",
                    }
                ),
                graph="",
            )
        )

    assert len(committed) == 1
    envelope = committed[0]
    assert envelope.connector == "ocel"
    node_types = {node["node_type"] for node in envelope.typed_payload["entities"]}
    assert "ProcessPerspective" in node_types
    assert "ProcessEvent" in node_types
    assert "BusinessObject" in node_types
    assert out["tekg"]["commit_status"] == "success"
    assert out["projection"]["perspective_id"] == "case:order-view"
    assert calls == [("process", {"traces": [["create"]]})]
    assert out["tekg"]["write_status"] == "success"


def test_graph_mine_ocel_mine_mode_surfaces_a_failed_commit(monkeypatch, tools) -> None:
    """A rejected ``ingest_envelope`` is reported, never silently discarded.

    Two lanes independently fixed the discarded-ChangeEnvelope bug here
    (feat/ocel-roundtrip-and-derivation and feat/wire-first-reachability-gate).
    The reconciliation kept ONE commit -- the ocel form, which also commits the
    disclosed ProcessPerspective -- and adopted wire-first's failure handling:
    a structured ``write_failed`` short-circuit instead of a bare RuntimeError,
    so the caller learns the write outcome and mining does NOT proceed past a
    failed write. This test is wire-first's coverage for that behaviour, kept.
    """
    calls: list = []
    mining = SimpleNamespace(process=_recording_method(calls, "process"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: "fake-engine")

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module

    monkeypatch.setattr(
        envelope_ingest_module,
        "ingest_envelope",
        lambda engine, envelope: {"status": "rejected", "error": "synthetic rejection"},
    )

    with use_actor(
        ActorContext(
            actor_id="ocel-reject-test",
            actor_type=ActorType.SYSTEM,
            tenant_id="tenant-a",
            authenticated=True,
        )
    ):
        out = json.loads(
            tools["graph_mine"](
                action="process",
                params_json=json.dumps(
                    {
                        "ocel_json": _OCEL_MINE_FIXTURE,
                        "tenant": "tenant-a",
                        "object_type": "Order",
                        "perspective_id": "case:order-view",
                        "derivation_version": "v1",
                    }
                ),
                graph="",
            )
        )

    assert out["code"] == "write_failed"
    assert "synthetic rejection" in out["error"]
    # mining must NOT have run past the failed write
    assert calls == []
    assert "projection" not in out


def test_graph_mine_alias_hyphenated_variant_resolves(monkeypatch, tools):
    """'entity-resolution' normalizes to 'entity_resolution' (the existing
    hyphen->underscore fold) THEN resolves via the alias map to
    'entity_resolve'."""
    calls: list = []
    mining = SimpleNamespace(entity_resolve=_recording_method(calls, "entity_resolve"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(mining=mining)
    )
    out = json.loads(
        tools["graph_mine"](action="entity-resolution", params_json="{}", graph="")
    )
    assert out["action"] == "entity_resolve"
    assert calls == [("entity_resolve", {})]


def test_graph_mine_unknown_action_lists_valid_actions_from_the_real_manifest(tools):
    """A genuinely bogus action gets a proper 'unknown action' error listing the
    introspected valid actions — NOT the old silent {"degraded": true} path that
    made a typo indistinguishable from the whole mining surface being absent."""
    out = json.loads(
        tools["graph_mine"](
            action="not_a_real_mining_action", params_json="{}", graph=""
        )
    )
    assert out.get("degraded") is not True
    assert "unknown action" in out["error"]
    assert set(out["actions"]) == set(_MINING_ACTIONS)


def test_graph_mine_unknown_action_with_mocked_manifest(monkeypatch, tools):
    """Same behavior, isolated from the real installed epistemic_graph package."""
    monkeypatch.setattr(
        engine_surface_tools,
        "_mining_actions",
        lambda: frozenset({"associate", "cluster"}),
    )
    out = json.loads(
        tools["graph_mine"](action="bogus_action", params_json="{}", graph="")
    )
    assert out.get("degraded") is not True
    assert out["error"] == (
        "unknown action 'bogus_action' for graph_mine; choose one of "
        "['associate', 'cluster']"
    )
    assert out["actions"] == ["associate", "cluster"]


def test_graph_mine_still_degrades_for_a_valid_action_the_client_lacks(
    monkeypatch, tools
):
    """Distinguishes the two failure modes: a VALID action name the connected
    client genuinely doesn't expose (e.g. a slim/mining-less engine build)
    still degrades cleanly — only a NAME miss stopped silently degrading."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_mine"](action="associate", params_json="{}", graph="")
    )
    assert out["degraded"] is True
    assert out["surface"] == "mining"


# ── graph_mine_deep (CONCEPT:AU-KG.mining.dsm-forecast-delegation — Phase 6) ─────────────────
# The engine core stays pure-Rust; graph_mine_deep dispatches the deep-learning /
# heavy-Python mining family to agents/data-science-mcp over the fleet
# call_tool_once connector and folds the decoded result back into the KG as
# typed nodes. These tests mock BOTH call_tool_once (the delegated call) and
# kg_server._execute_tool (the KG read/write) — no torch, no live engine, no
# live data-science-mcp required.


def test_kg_2_310_graph_mine_deep_registered_with_rest_twin(tools):
    assert "graph_mine_deep" in tools
    assert kg_server.REGISTERED_TOOLS.get("graph_mine_deep") is not None
    assert (
        kg_server.ACTION_TOOL_ROUTES.get("graph_mine_deep")
        == "/mining/deep/deep_forecast"
    )
    assert kg_server.DEEP_MINING_ACTIONS == (
        "deep_forecast",
        "deep_classify",
        "autoencoder_anomaly",
        "xgboost",
        "embed",
    )


def test_graph_mine_deep_dispatches_raw_rows_to_data_science_mcp(monkeypatch, tools):
    """deep_classify with raw x/y ships algo=mlp_classify to data-science-mcp, args-style."""
    captured: dict = {}

    async def _fake_call_tool_once(**kwargs):
        captured.update(kwargs)
        return {
            "algo": "mlp_classify",
            "available": True,
            "result": {
                "rows": [{"id": 0, "label": 1, "proba": 0.9}],
                "classes": [0, 1],
            },
        }

    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)
    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_classify",
            params_json=json.dumps({"x": [[0.0, 0.0], [1.0, 1.0]], "y": [0, 1]}),
            graph="",
        )
    )
    assert out["available"] is True
    assert out["provider"] == "data-science-mcp"
    assert out["result"]["classes"] == [0, 1]
    assert captured["server"] == "data-science-mcp"
    assert captured["tool"] == "deep_train_predict"
    assert captured["params_style"] == "args"
    assert captured["params"]["algo"] == "mlp_classify"
    assert json.loads(captured["params"]["x_json"]) == [[0.0, 0.0], [1.0, 1.0]]
    assert json.loads(captured["params"]["y_json"]) == [0, 1]
    # no writeback requested ⇒ no KG write attempted
    assert out["written_node_ids"] == []


def test_graph_mine_deep_gathers_source_rows_and_writes_back(monkeypatch, tools):
    """A 'source' spec is gathered via graph_query, and writeback=true folds one
    :Classification node per row back, linked DEEP_RESULT_OF its source node."""
    written_nodes: list[dict] = []
    written_edges: list[dict] = []

    async def _fake_execute_tool(tool_name, **kwargs):
        if tool_name == "graph_query":
            return json.dumps(
                [
                    {"id": "doc:1", "f0": 1.0},
                    {"id": "doc:2", "f0": 2.0},
                ]
            )
        if tool_name == "graph_write" and kwargs.get("action") == "add_node":
            written_nodes.append(kwargs)
            return json.dumps({"status": "ok"})
        if tool_name == "graph_write" and kwargs.get("action") == "add_edge":
            written_edges.append(kwargs)
            return json.dumps({"status": "ok"})
        raise AssertionError(f"unexpected _execute_tool call: {tool_name} {kwargs}")

    async def _fake_call_tool_once(**kwargs):
        return {
            "algo": "histgbm_classify",
            "available": True,
            "result": {
                "rows": [
                    {"id": 0, "label": 0, "proba": 0.8},
                    {"id": 1, "label": 1, "proba": 0.7},
                ],
                "classes": [0, 1],
            },
        }

    monkeypatch.setattr(kg_server, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)

    out = json.loads(
        tools["graph_mine_deep"](
            action="xgboost",
            params_json=json.dumps(
                {
                    "source": {"node_label": "Doc", "fields": ["f0"], "limit": 200},
                    "y": [0, 1],
                    "writeback": True,
                }
            ),
            graph="",
        )
    )
    assert out["available"] is True
    assert len(out["written_node_ids"]) == 2
    assert len(written_nodes) == 2
    assert all(n["node_type"] == "Classification" for n in written_nodes)
    assert all(
        json.loads(n["properties"])["provider"] == "data-science-mcp"
        for n in written_nodes
    )
    assert len(written_edges) == 2
    assert {e["rel_type"] for e in written_edges} == {"DEEP_RESULT_OF"}
    assert {e["target_id"] for e in written_edges} == {"doc:1", "doc:2"}


def test_graph_mine_deep_forecast_writeback_links_series(monkeypatch, tools):
    """deep_forecast with series_id + writeback=true creates one :Forecast node
    linked FORECAST_OF the named series node."""
    written_nodes: list[dict] = []
    written_edges: list[dict] = []

    async def _fake_execute_tool(tool_name, **kwargs):
        if tool_name == "graph_write" and kwargs.get("action") == "add_node":
            written_nodes.append(kwargs)
            return json.dumps({"status": "ok"})
        if tool_name == "graph_write" and kwargs.get("action") == "add_edge":
            written_edges.append(kwargs)
            return json.dumps({"status": "ok"})
        raise AssertionError(f"unexpected _execute_tool call: {tool_name} {kwargs}")

    async def _fake_call_tool_once(**kwargs):
        return {
            "algo": "lstm_forecast",
            "available": True,
            "result": {
                "forecast": [1.0, 2.0],
                "lower": [0.5, 1.5],
                "upper": [1.5, 2.5],
                "horizon": 2,
            },
        }

    monkeypatch.setattr(kg_server, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)

    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_forecast",
            params_json=json.dumps(
                {
                    "values": [1.0, 2.0, 3.0],
                    "horizon": 2,
                    "series_id": "metric:cpu",
                    "writeback": True,
                }
            ),
            graph="",
        )
    )
    assert out["available"] is True
    assert len(written_nodes) == 1
    assert written_nodes[0]["node_type"] == "Forecast"
    assert written_edges == [
        {
            "action": "add_edge",
            "source_id": written_nodes[0]["node_id"],
            "target_id": "metric:cpu",
            "rel_type": "FORECAST_OF",
            "target": "",
        }
    ]


def test_graph_mine_deep_degrades_when_data_science_mcp_unreachable(monkeypatch, tools):
    """The fleet server being unreachable degrades cleanly — never raises."""

    async def _boom(**kwargs):
        raise ConnectionError("no route to data-science-mcp")

    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _boom)
    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_forecast",
            params_json=json.dumps({"values": [1.0, 2.0, 3.0]}),
            graph="",
        )
    )
    assert out["available"] is False
    assert out["delegated"] is True
    assert out["error"]["code"] == "dependency_unavailable"


def test_graph_mine_deep_passes_through_torch_unavailable(monkeypatch, tools):
    """data-science-mcp itself reporting available=false (e.g. torch missing) passes through cleanly."""

    async def _fake_call_tool_once(**kwargs):
        return {
            "algo": "mlp_classify",
            "available": False,
            "error": "torch not installed",
        }

    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)
    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_classify",
            params_json=json.dumps({"x": [[0.0]], "y": [0]}),
            graph="",
        )
    )
    assert out["available"] is False
    assert out["error"] == "torch not installed"


# ── BUG-7 (kg-exhaustive-smoke.md): a JSON-encoded-STRING response from the
# delegate must still be parsed, not reported as an "unexpected response
# shape". ``call_tool_once``'s decoder prefers a FastMCP result's structured
# ``.data`` verbatim — when the delegate's own tool function returns an
# already-``json.dumps``-encoded string (this repo's own tool convention),
# ``.data`` IS that raw string, so ``raw`` used to arrive as ``str`` even
# though the delegate answered normally.


def test_graph_mine_deep_parses_json_string_response(monkeypatch, tools):
    async def _fake_call_tool_once(**kwargs):
        # The delegate answered normally, but as an ALREADY-JSON-ENCODED
        # STRING (the shape ``call_tool_once`` can hand back verbatim).
        return json.dumps(
            {
                "algo": "lstm_forecast",
                "available": True,
                "result": {"forecast": [1.0, 2.0], "horizon": 2},
            }
        )

    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)
    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_forecast",
            params_json=json.dumps({"values": [1, 2, 3, 4, 5], "horizon": 2}),
            graph="",
        )
    )
    assert out["available"] is True
    assert "error" not in out
    assert out["result"]["forecast"] == [1.0, 2.0]


def test_graph_mine_deep_genuinely_non_json_string_still_reports_shape_error(
    monkeypatch, tools
):
    async def _fake_call_tool_once(**kwargs):
        return "not json at all"

    monkeypatch.setattr(engine_surface_tools, "call_tool_once", _fake_call_tool_once)
    out = json.loads(
        tools["graph_mine_deep"](
            action="deep_forecast",
            params_json=json.dumps({"values": [1, 2, 3], "horizon": 1}),
            graph="",
        )
    )
    assert out["available"] is False
    assert "unexpected data-science-mcp response shape" in out["error"]


def test_graph_mine_deep_unknown_action_is_reported(tools):
    out = json.loads(
        tools["graph_mine_deep"](action="bogus", params_json="{}", graph="")
    )
    assert "unknown action" in out["error"]


def test_graph_mine_deep_missing_input_is_reported(tools):
    """No x/values/source at all ⇒ a clear error, not a crash."""
    out = json.loads(
        tools["graph_mine_deep"](
            action="autoencoder_anomaly", params_json="{}", graph=""
        )
    )
    assert out["error"]["code"] == "operation_failed"


# ── graph_pipeline (CONCEPT:EG-KG.mining.ml-pipeline — W4.4 composable ML pipeline) ──
def test_graph_pipeline_registered_with_rest_twin(tools):
    assert "graph_pipeline" in tools
    assert kg_server.REGISTERED_TOOLS.get("graph_pipeline") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_pipeline") == "/pipeline/train"


def test_graph_pipeline_train_dispatches(monkeypatch, tools):
    """train forwards name/spec/source verbatim to client.pipeline.train."""
    calls: list = []
    pipeline = SimpleNamespace(train=_recording_method(calls, "train"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(pipeline=pipeline)
    )
    spec = {
        "features": [{"step": "embedding", "method": "fastrp", "dim": 16}],
        "label_property": "label",
        "model": {"family": "classify", "algorithm": "logistic"},
    }
    out = json.loads(
        tools["graph_pipeline"](
            action="train",
            params_json=json.dumps(
                {"name": "community", "spec": spec, "source": {"node_label": "Person"}}
            ),
            graph="",
        )
    )
    assert out["surface"] == "pipeline"
    assert out["action"] == "train"
    assert calls == [
        (
            "train",
            {"name": "community", "spec": spec, "source": {"node_label": "Person"}},
        )
    ]


def test_graph_pipeline_eval_alias_hits_evaluate(monkeypatch, tools):
    """The MCP 'eval' verb maps to the client's evaluate() method."""
    calls: list = []
    pipeline = SimpleNamespace(evaluate=_recording_method(calls, "evaluate"))
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(pipeline=pipeline)
    )
    out = json.loads(
        tools["graph_pipeline"](
            action="eval",
            params_json=json.dumps({"name": "community", "version": 1}),
            graph="",
        )
    )
    assert out["surface"] == "pipeline"
    assert out["action"] == "evaluate"
    assert calls == [("evaluate", {"name": "community", "version": 1})]


def test_graph_pipeline_serve_predict_compare_dispatch(monkeypatch, tools):
    calls: list = []
    pipeline = SimpleNamespace(
        serve=_recording_method(calls, "serve"),
        predict=_recording_method(calls, "predict"),
        compare=_recording_method(calls, "compare"),
    )
    monkeypatch.setattr(
        engine_surface_tools, "_client", lambda graph: _fake_client(pipeline=pipeline)
    )
    tools["graph_pipeline"](
        action="serve", params_json=json.dumps({"name": "c", "version": 1}), graph=""
    )
    tools["graph_pipeline"](
        action="predict",
        params_json=json.dumps({"name": "c", "version": 0, "writeback": True}),
        graph="",
    )
    tools["graph_pipeline"](
        action="compare",
        params_json=json.dumps({"name": "c", "version_a": 1, "version_b": 2}),
        graph="",
    )
    assert [c[0] for c in calls] == ["serve", "predict", "compare"]
    assert calls[0][1] == {"name": "c", "version": 1}
    assert calls[1][1] == {"name": "c", "version": 0, "writeback": True}
    assert calls[2][1] == {"name": "c", "version_a": 1, "version_b": 2}


def test_graph_pipeline_unknown_action_is_reported(tools):
    out = json.loads(
        tools["graph_pipeline"](action="bogus", params_json="{}", graph="")
    )
    assert out["surface"] == "pipeline"
    assert "unknown action" in out["error"]


def test_graph_pipeline_degrades_when_surface_absent(monkeypatch, tools):
    """A no-ml-pipeline engine build (client has no .pipeline) degrades cleanly."""
    monkeypatch.setattr(engine_surface_tools, "_client", lambda graph: _fake_client())
    out = json.loads(
        tools["graph_pipeline"](
            action="train",
            params_json=json.dumps({"name": "x", "spec": {}}),
            graph="",
        )
    )
    assert out["degraded"] is True
