"""Tests for the multi-backend deployment planner (in_process/container/kubernetes/native_shell).

Scope note (see the module docstring in ``agent_utilities/deployment/backends.py``
for the full honesty statement): only :class:`InProcessBackend` is genuinely
live-tested end to end. The other three are PLAN-ONLY by design — these tests
assert the plans are correct/renderable and that ``apply(dry_run=False)``
refuses rather than silently no-op-ing (no masking).
"""

from __future__ import annotations

import threading

import pytest
import yaml

from agent_utilities.deployment import backends
from agent_utilities.deployment.backends import PlanOnlyBackendError
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.mcp import co_service_supervisor as cosvc
from agent_utilities.messaging import daemon as messaging_daemon
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _verified_session(actor_id: str = "deploy-test") -> GraphSession:
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("system",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        policy_version="current",
        audience="graph-runtime",
    )


@pytest.fixture(autouse=True)
def _detect_no_composition(monkeypatch):
    """Keep detection deterministic for these planner tests (covered separately
    in test_co_service_supervisor.py)."""
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: []
    )
    monkeypatch.setattr(cosvc, "host_daemon_needed", lambda: False)

    class _Cfg:
        enable_web_ui = False

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    yield


def test_registry_knows_all_four_backends():
    for name in ("in_process", "container", "kubernetes", "native_shell"):
        b = backends.get_backend(name)
        assert b.name == name


def test_registry_rejects_unknown_backend():
    with pytest.raises(ValueError):
        backends.get_backend("teleport")  # type: ignore[arg-type]


# ── in_process — REAL ───────────────────────────────────────────────────────


def test_in_process_plan_is_live_capable():
    plan = backends.InProcessBackend().plan()
    assert plan.backend == "in_process"
    assert plan.live_capable is True
    assert plan.steps  # at least one step


def test_in_process_apply_dry_run_does_not_start_anything():
    b = backends.InProcessBackend()
    plan = b.plan()
    out = b.apply(plan, dry_run=True)
    assert out["applied"] is False


def test_in_process_apply_requires_a_session():
    b = backends.InProcessBackend()
    plan = b.plan()
    with pytest.raises(ValueError):
        b.apply(plan, dry_run=False)


def test_in_process_apply_live_actually_starts_messaging(monkeypatch):
    """LIVE-PATH: InProcessBackend.apply(dry_run=False) really starts the
    composed co-services — through the SAME start_co_services the graph-os
    entrypoint calls — not merely that the method exists."""
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: ["fake"]
    )

    from agent_utilities.messaging.service import MessagingService

    MessagingService._instance = None
    reached = threading.Event()

    class _FakeBackend:
        id = "fake"
        is_connected = True

        async def register_commands(self, specs):
            return None

        async def listen(self):
            import asyncio

            while True:  # pragma: no branch
                await asyncio.sleep(3600)
                yield {}

    async def _fake_get_backend(self, platform):
        reached.set()
        return _FakeBackend()

    async def _fake_planner_handler(engine):
        async def _handler(event):
            return None

        return _handler

    monkeypatch.setattr(MessagingService, "get_backend", _fake_get_backend)
    monkeypatch.setattr(
        "agent_utilities.messaging.router.create_planner_handler",
        _fake_planner_handler,
    )

    b = backends.InProcessBackend()
    plan = b.plan()
    session = _verified_session()
    out = b.apply(plan, dry_run=False, session=session, engine=object())
    supervisor = out["supervisor"]
    try:
        assert reached.wait(timeout=10.0)
        assert "messaging" in supervisor.running()
    finally:
        supervisor.stop_all(timeout=10.0)
    MessagingService._instance = None


# ── container — PLAN-ONLY ───────────────────────────────────────────────────


def test_container_plan_renders_valid_compose_yaml():
    plan = backends.ContainerBackend().plan(
        target="r510", image="agent-utilities:1.2.3"
    )
    assert plan.backend == "container"
    assert plan.live_capable is False
    compose = yaml.safe_load(plan.artifacts["compose.yml"])
    assert compose["services"]["graph-os"]["image"] == "agent-utilities:1.2.3"
    assert "agent-webui" not in compose["services"]  # not configured in this test

    fleet_calls = [s.fleet_call for s in plan.steps if s.fleet_call]
    assert len(fleet_calls) == 1
    call = fleet_calls[0]
    assert call.server == "container-manager-mcp"
    assert call.tool == "cm_compose_operations"
    assert call.args["action"] == "up"
    assert call.args["host"] == "r510"


def test_container_plan_includes_webui_when_configured(monkeypatch):
    class _Cfg:
        enable_web_ui = True

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    plan = backends.ContainerBackend().plan(target="r510")
    compose = yaml.safe_load(plan.artifacts["compose.yml"])
    assert "agent-webui" in compose["services"]


def test_container_apply_never_executes_only_plans():
    b = backends.ContainerBackend()
    plan = b.plan(target="r510")
    out = b.apply(plan, dry_run=True)
    assert out["applied"] is False
    with pytest.raises(PlanOnlyBackendError):
        b.apply(plan, dry_run=False)


# ── kubernetes — PLAN-ONLY, and honestly further limited ───────────────────


def test_kubernetes_plan_renders_manifest_and_never_claims_a_live_apply_tool():
    plan = backends.KubernetesBackend().plan(target="prod-cluster", namespace="graphos")
    assert plan.live_capable is False
    docs = list(yaml.safe_load_all(plan.artifacts["manifest.yaml"]))
    kinds = {d["kind"] for d in docs}
    assert kinds == {"Deployment", "Service"}
    assert any("no generic manifest-apply" in w for w in plan.warnings)
    assert any("READ-ONLY" in w for w in plan.warnings)
    # The only fleet call offered is one that genuinely exists.
    fleet_calls = [s.fleet_call for s in plan.steps if s.fleet_call]
    assert all(c.server == "container-manager-mcp" for c in fleet_calls)
    assert all(c.tool == "cm_k8s_config" for c in fleet_calls)


def test_kubernetes_apply_always_refuses():
    b = backends.KubernetesBackend()
    plan = b.plan(target="prod-cluster")
    with pytest.raises(PlanOnlyBackendError):
        b.apply(plan, dry_run=False)


# ── native_shell — PLAN-ONLY (real unit-file rendering) ─────────────────────


def test_native_shell_renders_a_valid_systemd_unit_and_plans_via_tunnel_manager():
    plan = backends.NativeShellBackend().plan(target="edge-node")
    unit = plan.artifacts["graph-os.service"]
    assert "[Unit]" in unit and "[Service]" in unit and "[Install]" in unit
    assert "ExecStart=" in unit
    assert "agent_utilities.cli graph-os" in unit

    fleet_calls = [s.fleet_call for s in plan.steps if s.fleet_call]
    assert len(fleet_calls) == 2
    assert all(c.server == "tunnel-manager" for c in fleet_calls)
    assert all(c.tool == "tm_remote" for c in fleet_calls)
    assert all(c.args["host"] == "edge-node" for c in fleet_calls)


def test_native_shell_apply_always_refuses():
    b = backends.NativeShellBackend()
    plan = b.plan(target="edge-node")
    with pytest.raises(PlanOnlyBackendError):
        b.apply(plan, dry_run=False)
