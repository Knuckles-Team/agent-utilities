"""Tenant-partitioned engine sharding tests (CONCEPT:KG-2.58 / OS-5.28).

Covers HRW routing determinism (and parity with ``epistemic_graph.pool``'s
``ShardRouter``), single-endpoint back-compat, the tenant→graph naming
discipline, the fail-loud unreachable-shard contract (autostart never applies
to remote shards), and the status/metrics visibility surfaces. No live
engines: clients are injected fakes.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core import shard_topology
from agent_utilities.knowledge_graph.core.shard_topology import (
    DEFAULT_LOCAL_ENDPOINT,
    is_local_endpoint,
    resolve_endpoints,
    resolve_routing_graph,
    shard_endpoint_for,
    shard_topology_status,
    tenant_graph_name,
)
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.concept("KG-2.58")

THREE_SHARDS = ["tcp://shard-a:9100", "tcp://shard-b:9100", "tcp://shard-c:9100"]

# Captured at collection time, BEFORE the autouse isolate_graph_compute_engine
# fixture wraps __init__ with per-test graph renaming: the default-graph /
# ambient-tenant resolution tests need the engine's real constructor.
from agent_utilities.knowledge_graph.core.graph_compute import (  # noqa: E402
    GraphComputeEngine as _Engine,
)

_REAL_ENGINE_INIT = _Engine.__init__


def _build_engine_unwrapped(graph_name=None):
    engine = _Engine.__new__(_Engine)
    _REAL_ENGINE_INIT(engine, graph_name=graph_name)
    return engine


class _FakeConfig:
    def __init__(self, **overrides):
        self.graph_service_endpoints = overrides.get("graph_service_endpoints")
        self.graph_service_tcp_addr = overrides.get("graph_service_tcp_addr")
        self.graph_service_socket = overrides.get("graph_service_socket")
        self.kg_default_graph = overrides.get("kg_default_graph", "__commons__")


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_resolve_endpoints_default_single():
    cfg = _FakeConfig()
    assert resolve_endpoints(cfg) == [DEFAULT_LOCAL_ENDPOINT]


def test_resolve_endpoints_socket_and_tcp_precedence():
    assert resolve_endpoints(_FakeConfig(graph_service_socket="/tmp/x.sock")) == [
        "unix:///tmp/x.sock"
    ]
    assert resolve_endpoints(_FakeConfig(graph_service_tcp_addr="h:9100")) == [
        "tcp://h:9100"
    ]
    # The endpoint list overrides socket/tcp_addr.
    cfg = _FakeConfig(
        graph_service_endpoints=THREE_SHARDS, graph_service_socket="/tmp/x.sock"
    )
    assert resolve_endpoints(cfg) == THREE_SHARDS


def test_endpoints_env_accepts_comma_and_json(monkeypatch):
    from agent_utilities.core.config import AgentConfig

    monkeypatch.setenv(
        "GRAPH_SERVICE_ENDPOINTS", "tcp://a:1, tcp://b:2 ,unix:///tmp/c.sock"
    )
    assert AgentConfig().graph_service_endpoints == [
        "tcp://a:1",
        "tcp://b:2",
        "unix:///tmp/c.sock",
    ]
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", '["tcp://a:1", "tcp://b:2"]')
    assert AgentConfig().graph_service_endpoints == ["tcp://a:1", "tcp://b:2"]
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "")
    assert AgentConfig().graph_service_endpoints is None


# ---------------------------------------------------------------------------
# HRW routing
# ---------------------------------------------------------------------------


def test_shard_selection_deterministic_and_order_independent():
    reordered = [THREE_SHARDS[2], THREE_SHARDS[0], THREE_SHARDS[1]]
    for key in ("__commons__", "tenant__acme____commons__", "kg_enrich_comm", "alpha"):
        first = shard_endpoint_for(key, THREE_SHARDS)
        assert first in THREE_SHARDS
        assert first == shard_endpoint_for(key, THREE_SHARDS)  # stable
        assert first == shard_endpoint_for(key, reordered)  # HRW order-free


def test_shard_selection_matches_shard_router():
    """Sync placement must agree with epistemic_graph.pool.ShardRouter."""
    from epistemic_graph.pool import ShardRouter

    router = ShardRouter(list(THREE_SHARDS))
    for key in ("a", "b", "c", "__commons__", "tenant__t1____commons__", "graph-42"):
        assert shard_endpoint_for(key, THREE_SHARDS) == router._get_shard_endpoint(key)


def test_shard_selection_spreads_keys():
    chosen = {
        shard_endpoint_for(f"graph_{i}", THREE_SHARDS) for i in range(64)
    }
    assert chosen == set(THREE_SHARDS)


def test_single_endpoint_is_identity():
    assert shard_endpoint_for("anything", ["unix:///tmp/only.sock"]) == (
        "unix:///tmp/only.sock"
    )


def test_shard_selection_requires_endpoints():
    with pytest.raises(ValueError):
        shard_endpoint_for("g", [])


# ---------------------------------------------------------------------------
# Tenant → graph naming discipline
# ---------------------------------------------------------------------------


def test_tenant_graph_name_single_tenant_unchanged():
    assert tenant_graph_name(None, "__commons__") == "__commons__"
    assert tenant_graph_name("", "__commons__") == "__commons__"
    assert tenant_graph_name("   ", "__commons__") == "__commons__"


def test_tenant_graph_name_sanitizes_and_is_deterministic():
    assert tenant_graph_name("acme", "__commons__") == "tenant__acme____commons__"
    assert tenant_graph_name("Acme Corp/EU", "kg") == "tenant__acme_corp_eu__kg"
    assert tenant_graph_name("acme", "kg") == tenant_graph_name("acme", "kg")


def test_facade_and_package_expose_tenant_naming():
    import agent_utilities.knowledge_graph as kg_pkg
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    assert kg_pkg.tenant_graph_name("t1", "base") == "tenant__t1__base"
    facade = KnowledgeGraph.__new__(KnowledgeGraph)  # no layers needed
    assert facade.tenant_graph(tenant="t1") == "tenant__t1____commons__"
    with use_actor(ActorContext(actor_id="u", tenant_id="ambient")):
        assert facade.tenant_graph() == "tenant__ambient____commons__"
    assert facade.tenant_graph() == "__commons__"  # no ambient tenant


def test_resolve_routing_graph_precedence():
    cfg = _FakeConfig()
    # 1. explicit non-default graph wins, tenant or not
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph("named", cfg) == "named"
        # 2. ambient tenant maps the default graph
        assert resolve_routing_graph(None, cfg) == "tenant__acme____commons__"
        assert resolve_routing_graph("__commons__", cfg) == "tenant__acme____commons__"
    # 3. otherwise the configured default
    assert resolve_routing_graph(None, cfg) == "__commons__"
    custom = _FakeConfig(kg_default_graph="knowledge")
    assert resolve_routing_graph(None, custom) == "knowledge"
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph("knowledge", custom) == (
            "tenant__acme__knowledge"
        )


# ---------------------------------------------------------------------------
# Engine client path (fake clients — no live engines)
# ---------------------------------------------------------------------------


def _fake_connect_recorder(connects: list):
    def _connect(**kwargs):
        connects.append(kwargs)
        return MagicMock(name="fake_engine_client")

    return _connect


@pytest.fixture
def quiet_engine_env(monkeypatch):
    """Keep engine construction hermetic: no event bridge, no autostart."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "test-suite:9092")
    monkeypatch.delenv("EPISTEMIC_GRAPH_AUTOSTART", raising=False)


def test_engine_routes_to_hrw_shard(monkeypatch, quiet_engine_env):
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    connects: list = []
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_fake_connect_recorder(connects)),
    )

    engine = GraphComputeEngine(graph_name="shard_routing_probe")
    expected = shard_endpoint_for("shard_routing_probe", THREE_SHARDS)
    assert connects, "engine never connected"
    assert connects[-1]["tcp_addr"] == expected.removeprefix("tcp://")
    assert engine.graph_name == "shard_routing_probe"


def test_engine_sharded_maps_ambient_tenant_to_tenant_graph(
    monkeypatch, quiet_engine_env
):
    from epistemic_graph.client import SyncEpistemicGraphClient

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    connects: list = []
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_fake_connect_recorder(connects)),
    )
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        engine = _build_engine_unwrapped(graph_name="__commons__")
    assert engine.graph_name == "tenant__acme____commons__"
    expected = shard_endpoint_for("tenant__acme____commons__", THREE_SHARDS)
    assert connects[-1]["tcp_addr"] == expected.removeprefix("tcp://")


def test_engine_single_endpoint_backcompat(monkeypatch, quiet_engine_env):
    """One endpoint: no tenant mapping, no routing surprises (zero-infra)."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    monkeypatch.setenv("GRAPH_SERVICE_SOCKET", "/tmp/backcompat.sock")
    connects: list = []
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_fake_connect_recorder(connects)),
    )
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        engine = _build_engine_unwrapped(graph_name="__commons__")
    # Single-endpoint mode must NOT remap the graph for the ambient tenant.
    assert engine.graph_name == "__commons__"
    assert connects[-1]["socket_path"] == "/tmp/backcompat.sock"

    # And the no-argument default resolves to the configured default graph.
    engine_default = _build_engine_unwrapped()
    assert engine_default.graph_name == "__commons__"


def test_unreachable_remote_shard_fails_loud_without_autostart(
    monkeypatch, quiet_engine_env
):
    """A configured remote shard that is down is a hard error naming the shard;
    EPISTEMIC_GRAPH_AUTOSTART must never spawn a stand-in for it."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    monkeypatch.setenv("EPISTEMIC_GRAPH_AUTOSTART", "1")

    def _refuse(**kwargs):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(
        SyncEpistemicGraphClient, "connect", staticmethod(_refuse)
    )
    spawned: list = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: spawned.append(a) or MagicMock()
    )

    graph = "fail_loud_probe"
    expected = shard_endpoint_for(graph, THREE_SHARDS)
    with pytest.raises(ConnectionError) as excinfo:
        graph_compute.GraphComputeEngine(graph_name=graph)
    assert expected in str(excinfo.value)
    assert graph in str(excinfo.value)
    assert not spawned, "autostart must not spawn engines for remote shards"


def test_local_endpoint_detection():
    assert is_local_endpoint("unix:///tmp/a.sock")
    assert is_local_endpoint("/tmp/a.sock")
    assert not is_local_endpoint("tcp://127.0.0.1:9100")
    assert not is_local_endpoint("tcp://shard-a:9100")


# ---------------------------------------------------------------------------
# Topology visibility (CONCEPT:OS-5.28)
# ---------------------------------------------------------------------------


def test_shard_topology_status_reports_per_shard_reachability(monkeypatch):
    cfg = _FakeConfig(graph_service_endpoints=THREE_SHARDS)
    monkeypatch.setattr(
        shard_topology,
        "probe_endpoint",
        lambda ep, timeout=0.5: ep != "tcp://shard-b:9100",
    )
    status = shard_topology_status(cfg)
    assert status["mode"] == "sharded"
    assert status["default_graph"] == "__commons__"
    by_ep = {e["endpoint"]: e for e in status["endpoints"]}
    assert set(by_ep) == set(THREE_SHARDS)
    assert by_ep["tcp://shard-a:9100"]["reachable"] is True
    assert by_ep["tcp://shard-b:9100"]["reachable"] is False
    assert all(e["local"] is False for e in status["endpoints"])
    assert all("breaker" in e for e in status["endpoints"])


def test_shard_topology_status_single_mode(monkeypatch):
    cfg = _FakeConfig(graph_service_socket="/tmp/solo.sock")
    monkeypatch.setattr(shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True)
    status = shard_topology_status(cfg)
    assert status["mode"] == "single"
    assert status["endpoints"][0]["endpoint"] == "unix:///tmp/solo.sock"
    assert status["endpoints"][0]["local"] is True


def test_probe_endpoint_unreachable_targets():
    assert probe_unreachable_tcp() is False
    assert shard_topology.probe_endpoint("unix:///nonexistent/path.sock") is False
    assert shard_topology.probe_endpoint("tcp://bad", timeout=0.1) is False


def probe_unreachable_tcp() -> bool:
    # TEST-NET-1 address (RFC 5737) — guaranteed non-routable, short timeout.
    return shard_topology.probe_endpoint("tcp://192.0.2.1:9", timeout=0.2)


def test_shard_metrics_registered_and_exported(monkeypatch):
    from agent_utilities.observability import gateway_metrics as gm

    cfg = _FakeConfig(graph_service_endpoints=THREE_SHARDS)
    monkeypatch.setattr(shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True)
    shard_topology_status(cfg)  # refreshes the gauge
    if not gm.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client not installed (metrics extra)")
    payload, _ = gm.render_metrics()
    assert b"agent_utilities_engine_shard_up" in payload
    assert b"tcp://shard-a:9100" in payload


def test_unified_daemon_status_includes_shards(monkeypatch):
    from agent_utilities.knowledge_graph.core import engine_tasks

    class _Daemonish(engine_tasks.TaskManagerMixin):
        def _maintenance_jobs(self):
            return []

    monkeypatch.setattr(
        shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True
    )
    # __new__: unified_daemon_status reads its attributes defensively, and the
    # full mixin __init__ would build a real task queue this test doesn't need.
    status = _Daemonish.__new__(_Daemonish).unified_daemon_status()
    assert "shards" in status
    assert status["shards"]["mode"] in {"single", "sharded"}
    assert status["shards"]["endpoints"]


def test_gateway_daemon_shards_route(monkeypatch):
    """The dashboard router exposes the shard topology (CONCEPT:OS-5.28)."""
    import asyncio

    from agent_utilities.gateway import api as gateway_api

    monkeypatch.setattr(
        shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True
    )
    result = asyncio.run(gateway_api.daemon_shards())
    assert result["mode"] in {"single", "sharded"}
    assert result["endpoints"]
