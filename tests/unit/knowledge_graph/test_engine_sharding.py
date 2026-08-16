"""Engine-authoritative placement topology tests.

Covers ordered coordinator contacts, tenant→graph naming, fail-loud remote
contacts, and topology visibility. No live engines: clients are injected fakes.
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
    shard_topology_status,
    tenant_graph_name,
)
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.concept("AU-KG.sharding.tenant-partitioned-sharding-hrw")

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
        self.kg_default_graph = overrides.get("kg_default_graph", "__commons__")


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_resolve_endpoints_default_single():
    cfg = _FakeConfig()
    assert resolve_endpoints(cfg) == [DEFAULT_LOCAL_ENDPOINT]


def test_resolve_endpoints_explicit_topology():
    cfg = _FakeConfig(graph_service_endpoints=THREE_SHARDS)
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


def test_group_endpoint_map_is_typed_and_fails_closed(monkeypatch):
    from agent_utilities.core.config import AgentConfig

    monkeypatch.setenv(
        "GRAPH_RAFT_GROUP_ENDPOINTS",
        '{"01": "tls://group-one.invalid:9443", "2": "tcp://group-two.invalid:9100"}',
    )
    assert AgentConfig().graph_raft_group_endpoints == {
        "1": "tls://group-one.invalid:9443",
        "2": "tcp://group-two.invalid:9100",
    }

    monkeypatch.setenv("GRAPH_RAFT_GROUP_ENDPOINTS", '{"group": "https://invalid"}')
    with pytest.raises(ValueError):
        AgentConfig()


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
    assert facade.tenant_graph(tenant="") == "__commons__"


def test_resolve_routing_graph_precedence():
    cfg = _FakeConfig()
    # 1. explicit non-default graph wins, tenant or not
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph("named", cfg) == "named"
        # 2. ambient tenant maps the default graph
        assert resolve_routing_graph(None, cfg) == "tenant__acme____commons__"
        # BUG-020 (GOC-61 phase 1) — FIXED: this line used to assert the
        # DEFECT itself (an explicit "__commons__" request silently
        # misrouted to the tenant graph, because shard_topology.py:198
        # conflated "explicit request for the literal default name" with "no
        # name given"). It now asserts the correct, post-fix value — an
        # explicit request for `__commons__` is honored verbatim, exactly
        # like any other explicit graph name. See
        # test_resolve_routing_graph_explicit_commons_should_be_honored_verbatim
        # below for the dedicated regression test.
        assert resolve_routing_graph("__commons__", cfg) == "__commons__"
    # 3. otherwise the configured default
    with use_actor(ActorContext(actor_id="u", authenticated=True)):
        assert resolve_routing_graph(None, cfg) == "__commons__"
        custom = _FakeConfig(kg_default_graph="knowledge")
        assert resolve_routing_graph(None, custom) == "knowledge"
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        # BUG-020 (GOC-61 phase 1) — FIXED: same defect pattern as the
        # `__commons__` case above, generalized to a NON-default-named
        # config: `custom.kg_default_graph == "knowledge"`, so an explicit
        # request for the literal string "knowledge" here is ALSO "the
        # caller explicitly asked for the graph literally equal to the
        # configured default" — pre-fix this collapsed into tenant-mapping
        # exactly like `__commons__` did; post-fix it is honored verbatim.
        # This line is what proves the fix is general (any configured
        # default), not hardcoded to the literal string "__commons__".
        assert resolve_routing_graph("knowledge", custom) == "knowledge"


# ---------------------------------------------------------------------------
# BUG-020 / GOC-61 phase 1 — regression tests. FIX APPLIED (shard_topology.py
# resolve_routing_graph): the four tests below were originally written
# failing-first (three already passed pre-fix as regression guards; the
# fourth, explicit-commons-honored-verbatim, was intentionally red) and now
# all pass post-fix. The bug-pinning test that asserted the defect itself
# (`test_resolve_routing_graph_explicit_commons_is_currently_misrouted`) has
# been DELETED in this same change, per its own docstring's instruction —
# its assertion WAS the bug, so keeping it would assert broken behavior
# forever. shard_topology.py:198's old `graph_name != default` collapsed two
# different intents into one branch: "the caller explicitly asked for the
# graph literally named `__commons__`" and "the caller passed nothing and
# wants their ambient default" both satisfied `graph_name == default`, so an
# EXPLICIT request for `__commons__` was (incorrectly) tenant-mapped exactly
# like an implicit `None` request. `__control__`/other explicit names never
# collided with the default sentinel, so they were never affected — only the
# literal commons name was. The fix (GOC-61 design doc §6): drop the
# `!= default` comparison so any explicit, non-empty graph_name is honored
# verbatim; only `graph_name is None` triggers tenant mapping.
# ---------------------------------------------------------------------------


def test_resolve_routing_graph_explicit_commons_should_be_honored_verbatim():
    """BUG-020 case (a): an explicit request for `__commons__` must resolve to
    `__commons__` itself, not the caller's tenant graph. FIXED — was red
    pre-fix, passes now."""
    cfg = _FakeConfig()
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph("__commons__", cfg) == "__commons__"


def test_resolve_routing_graph_none_still_tenant_maps():
    """BUG-020 case (b): `graph_name is None` must still tenant-map correctly
    — the fix narrows the tenant-mapping trigger to exactly this case, so it
    must be unaffected. PASSED both pre- and post-fix (regression guard)."""
    cfg = _FakeConfig()
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph(None, cfg) == "tenant__acme____commons__"


def test_resolve_routing_graph_explicit_non_default_name_honored_verbatim():
    """BUG-020 case (c): an explicit request for a non-default name (a
    content graph, e.g. ``code_agent_utilities``) is honored verbatim — this
    already worked pre-fix (it never collided with the default sentinel) and
    keeps working post-fix. PASSED both pre- and post-fix."""
    cfg = _FakeConfig()
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert (
            resolve_routing_graph("code_agent_utilities", cfg) == "code_agent_utilities"
        )


def test_resolve_routing_graph_control_graph_behaviour_unchanged():
    """BUG-020 case (d): ``__control__`` was never affected by this defect
    (its name never equals the default sentinel) — the fix leaves its
    behaviour byte-for-byte identical. PASSED both pre- and post-fix."""
    cfg = _FakeConfig()
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        assert resolve_routing_graph("__control__", cfg) == "__control__"
    with use_actor(ActorContext(actor_id="u", authenticated=True)):
        assert resolve_routing_graph("__control__", cfg) == "__control__"


# ---------------------------------------------------------------------------
# is_system_graph — PHASE-1 STOPGAP predicate (GOC-61 §2.5/§6, W03)
# ---------------------------------------------------------------------------


def test_is_system_graph_matches_the_known_dunder_family():
    from agent_utilities.knowledge_graph.core.shard_topology import is_system_graph

    for name in ("__commons__", "__control__", "__secrets__"):
        assert is_system_graph(name) is True


def test_is_system_graph_rejects_tenant_and_content_graphs():
    from agent_utilities.knowledge_graph.core.shard_topology import is_system_graph

    for name in (
        "tenant__homelab____commons__",
        "tenant__acme__knowledge",
        "code:agent-utilities",
        "code_agent_utilities",
        "src:servicenow",
        "chat:planner",
        "research:arxiv",
        "knowledge",
        "",
        None,
        "__",
        "____",
    ):
        assert is_system_graph(name) is False


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


def test_engine_connects_to_first_coordinator_contact(monkeypatch, quiet_engine_env):
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import engine_transport

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    monkeypatch.setattr(
        engine_transport, "engine_client_transport_kwargs", lambda *a, **k: {}
    )
    connects: list = []
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_fake_connect_recorder(connects)),
    )

    engine = _build_engine_unwrapped(graph_name="shard_routing_probe")
    assert connects, "engine never connected"
    assert connects[-1]["tcp_addr"] == THREE_SHARDS[0].removeprefix("tcp://")
    assert engine.graph_name == "shard_routing_probe"
    engine.close()


def test_engine_sharded_maps_ambient_tenant_to_tenant_graph(
    monkeypatch, quiet_engine_env
):
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import engine_transport

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    monkeypatch.setattr(
        engine_transport, "engine_client_transport_kwargs", lambda *a, **k: {}
    )
    connects: list = []
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_fake_connect_recorder(connects)),
    )
    # BUG-020/GOC-61 phase 1 (see shard_topology.resolve_routing_graph's
    # docstring): an EXPLICIT graph_name — including the literal
    # "__commons__" — is now honored verbatim, never tenant-mapped. Only
    # graph_name=None ("caller asked for nothing specific") maps to the
    # ambient tenant's graph, which is what this test actually exercises.
    with use_actor(ActorContext(actor_id="u", tenant_id="acme")):
        engine = _build_engine_unwrapped(graph_name=None)
    assert engine.graph_name == "tenant__acme____commons__"
    assert connects[-1]["tcp_addr"] == THREE_SHARDS[0].removeprefix("tcp://")
    engine.close()


def test_engine_packaged_local_endpoint_identity(monkeypatch, quiet_engine_env):
    """One endpoint: no tenant mapping, no routing surprises (zero-infra)."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "unix:///tmp/packaged-local.sock")
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
    assert connects[-1]["socket_path"] == "/tmp/packaged-local.sock"

    # And the no-argument default resolves to the configured default graph.
    engine.close()
    with use_actor(ActorContext(actor_id="u", authenticated=True)):
        engine_default = _build_engine_unwrapped()
    assert engine_default.graph_name == "__commons__"
    engine_default.close()


def test_unreachable_remote_shard_fails_loud_without_autostart(
    monkeypatch, quiet_engine_env
):
    """A configured remote shard that is down is a hard error naming the shard;
    the configured external topology must never spawn a stand-in for it."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import engine_transport, graph_compute

    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", ",".join(THREE_SHARDS))
    monkeypatch.setattr(
        engine_transport, "engine_client_transport_kwargs", lambda *a, **k: {}
    )

    def _refuse(**kwargs):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(SyncEpistemicGraphClient, "connect", staticmethod(_refuse))
    spawned: list = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: spawned.append(a) or MagicMock()
    )

    graph = "fail_loud_probe"
    with pytest.raises(ConnectionError) as excinfo:
        graph_compute.GraphComputeEngine(graph_name=graph)
    assert "configured engine shard" in str(excinfo.value).lower()
    assert not spawned, "autostart must not spawn engines for remote shards"


def test_local_endpoint_detection():
    assert is_local_endpoint("unix:///tmp/a.sock")
    assert is_local_endpoint("/tmp/a.sock")
    assert is_local_endpoint("tcp://127.0.0.1:9100")
    assert is_local_endpoint("tcp://localhost:9100")
    assert is_local_endpoint("tcp://[::1]:9100")
    assert not is_local_endpoint("tcp://shard-a:9100")
    assert not is_local_endpoint("tcp://0.0.0.0:9100")
    assert not is_local_endpoint("tls://127.0.0.1:9100")


# ---------------------------------------------------------------------------
# Topology visibility (CONCEPT:AU-OS.scaling.shard-topology-visibility-per)
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
    cfg = _FakeConfig(graph_service_endpoints=["unix:///tmp/solo.sock"])
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

    monkeypatch.setattr(shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True)
    # __new__: unified_daemon_status reads its attributes defensively, and the
    # full mixin __init__ would build a real task queue this test doesn't need.
    status = _Daemonish.__new__(_Daemonish).unified_daemon_status()
    assert "shards" in status
    assert status["shards"]["mode"] in {"single", "sharded"}
    assert status["shards"]["endpoints"]


def test_gateway_daemon_shards_route(monkeypatch):
    """The dashboard router exposes the shard topology (CONCEPT:AU-OS.scaling.shard-topology-visibility-per)."""
    import asyncio

    from agent_utilities.gateway import api as gateway_api

    monkeypatch.setattr(shard_topology, "probe_endpoint", lambda ep, timeout=0.5: True)
    result = asyncio.run(gateway_api.daemon_shards())
    assert result["mode"] in {"single", "sharded"}
    assert result["endpoints"]
