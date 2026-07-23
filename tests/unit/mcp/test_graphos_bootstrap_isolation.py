"""Focused proofs for GraphOS bootstrap authority and lazy fleet discovery."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _capture_verified_background_session,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionExpiredError,
    SessionRequiredError,
    current_session,
    suspend_session,
    use_session,
)
from agent_utilities.mcp.multiplexer import MCPMultiplexer
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import (
    ActorContext,
    CredentialLease,
    current_actor,
    use_actor,
)


def _verified_session(actor_id: str = "runtime-agent") -> GraphSession:
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.SYSTEM,
        roles=("system",),
        tenant_id="runtime-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        policy_version="current",
        audience="graph-runtime",
    )


def _catalog_path(root: Path) -> Path:
    path = root / "mcp_config.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "target-mcp": {"command": "target-server"},
                    "unrelated-a": {"command": "unrelated-a"},
                    "unrelated-b": {"command": "unrelated-b"},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_background_task_worker_binds_captured_session_and_actor() -> None:
    session = _verified_session()
    seen: list[tuple[object, object]] = []
    finished = threading.Event()
    engine = TaskManagerMixin.__new__(TaskManagerMixin)
    engine.backend = object()
    engine._worker_lock = threading.Lock()
    engine._workers_running = False
    engine._task_queue_backend_name = "sqlite"

    def one_shot_worker() -> None:
        seen.append((current_session(), current_actor()))
        finished.set()

    engine._task_worker_loop = one_shot_worker
    with (
        patch(
            "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
            return_value="host",
        ),
        patch(
            "agent_utilities.core.config.DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND",
            True,
        ),
        use_actor(session.actor),
        use_session(session),
    ):
        engine.start_task_workers(worker_count=1)

    assert finished.wait(2.0)
    assert seen == [(session, session.actor)]
    assert engine._background_worker_session is session


def test_background_authority_fails_closed_when_actor_and_session_differ() -> None:
    session = _verified_session("session-agent")
    other = _verified_session("different-agent").actor
    try:
        with use_actor(other), use_session(session):
            _capture_verified_background_session()
    except SessionRequiredError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("mismatched background authority was accepted")


def test_background_authority_fails_closed_when_session_is_absent() -> None:
    errors: list[BaseException] = []

    def capture_without_context() -> None:
        try:
            _capture_verified_background_session()
        except BaseException as exc:  # noqa: BLE001 - relayed to the test thread
            errors.append(exc)

    thread = threading.Thread(target=capture_without_context)
    thread.start()
    thread.join(2.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SessionRequiredError)


def test_targeted_catalog_probe_does_not_probe_unrelated_servers() -> None:
    with TemporaryDirectory() as tmp:
        mux = MCPMultiplexer(_catalog_path(Path(tmp)))
        calls: list[str] = []

        async def probe_one(server: str, **_kwargs: object) -> dict:
            calls.append(server)
            return {
                "tools": [
                    {
                        "name": "inspect",
                        "description": "Inspect runtime state.",
                        "inputSchema": {},
                    }
                ],
                "error": None,
            }

        async def probe_all(**_kwargs: object) -> dict:
            raise AssertionError("targeted catalog lookup probed the whole fleet")

        mux.probe_server = probe_one  # type: ignore[method-assign]
        mux.probe_catalog = probe_all  # type: ignore[method-assign]
        with patch(
            "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
            return_value=None,
        ):
            result = asyncio.run(mux.list_catalog(server="target-mcp"))

    assert calls == ["target-mcp"]
    assert result["server"] == "target-mcp"
    assert result["tools"][0]["tool"] == "inspect"


def test_metadata_catalog_listing_never_probes_children() -> None:
    with TemporaryDirectory() as tmp:
        mux = MCPMultiplexer(_catalog_path(Path(tmp)))

        async def forbidden_probe(*_args: object, **_kwargs: object) -> dict:
            raise AssertionError("metadata-only catalog listing started a child")

        mux.probe_server = forbidden_probe  # type: ignore[method-assign]
        mux.probe_catalog = forbidden_probe  # type: ignore[method-assign]
        with patch(
            "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
            return_value=None,
        ):
            result = asyncio.run(mux.list_catalog(include_tools=False))

    assert result["total_servers"] == 3
    assert result["unavailable"] == []
    assert all(item["probed"] is False for item in result["servers"])
    assert all(item["available"] is None for item in result["servers"])


def test_startup_capability_ingest_does_not_probe_fleet() -> None:
    from agent_utilities.mcp import kg_server

    class RecordingEngine:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def add_node(
            self,
            node_id: str,
            node_type: str,
            properties: dict | None = None,
            **kwargs: object,
        ) -> None:
            self.nodes[node_id] = {"type": node_type, **(properties or kwargs)}

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _catalog_path(root)
        engine = RecordingEngine()
        full_probe = MagicMock(
            side_effect=AssertionError("startup attempted a full fleet probe")
        )
        with (
            patch("platformdirs.user_config_path", return_value=root),
            patch("pkgutil.iter_modules", return_value=[]),
            patch(
                "agent_utilities.core.providers.resolve_skill_provider_dirs",
                return_value=[],
            ),
            patch.object(kg_server, "_ingest_skill_capabilities", return_value=0),
            patch.object(kg_server, "get_existing_disabled", return_value=False),
            patch(
                "agent_utilities.knowledge_graph.core.source_sync.sync_source",
                full_probe,
            ),
        ):
            kg_server._ingest_capabilities(engine)

    assert full_probe.call_count == 0
    assert {
        node["name"]
        for node in engine.nodes.values()
        if node.get("type") == "MCPServer"
    } == {"target-mcp", "unrelated-a", "unrelated-b"}


def test_graphos_entrypoint_activates_configured_otel_live_path() -> None:
    from agent_utilities.mcp import kg_server

    with (
        patch.object(kg_server, "setting", return_value=True),
        patch("agent_utilities.observability.custom_observability.setup_otel") as setup,
    ):
        kg_server._configure_graphos_otel()

    setup.assert_called_once_with(service_name="graph-os")
    source = inspect.getsource(kg_server.mcp_server)
    assert source.index("load_config()") < source.index("_configure_graphos_otel()")


def test_packaged_skill_readiness_blocks_background_bootstrap_and_preserves_authority(
    monkeypatch,
) -> None:
    from agent_utilities.mcp import kg_server

    session = _verified_session()
    critical_entered = threading.Event()
    release_critical = threading.Event()
    background_started = threading.Event()
    errors: list[BaseException] = []

    class Engine:
        backend = object()

    class DeferredBackground:
        def start(self) -> None:
            background_started.set()

    def ensure_ready(_engine):
        assert current_session() is session
        assert current_actor() is session.actor
        critical_entered.set()
        assert release_critical.wait(2.0)
        return {"required": 10, "already_ready": 0, "ingested": 10, "ready": 10}

    def launch() -> None:
        try:
            kg_server._start_engine_bootstrap(session)
        except BaseException as exc:  # noqa: BLE001 - relayed to the test thread
            errors.append(exc)

    with (
        patch.object(kg_server, "_get_engine", return_value=Engine()),
        patch.object(
            kg_server, "_ensure_bundled_skills_ready", side_effect=ensure_ready
        ),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            return_value=DeferredBackground(),
        ),
    ):
        caller = threading.Thread(target=launch)
        caller.start()
        assert critical_entered.wait(2.0)
        assert not background_started.is_set()
        assert caller.is_alive()
        release_critical.set()
        caller.join(2.0)

    assert not caller.is_alive()
    assert errors == []
    assert background_started.is_set()


def test_packaged_skill_readiness_failure_is_controlled_and_serves_degraded(
    caplog,
) -> None:
    """Packaged-skill readiness is a capability concern, not a correctness or
    security one: a failure must not raise and must not take the process
    down. It is recorded (via ``bundled_skill_readiness()``, surfaced on
    ``/health``) and logged loudly, but ``_start_engine_bootstrap`` returns
    normally — the same "log loudly, keep serving" contract as the
    ``DuplicateSkillIdentity`` sweep resilience in ``core/providers.py``.
    Background/noncritical service startup still does not run this cycle
    (it is gated behind readiness having been established at all).

    "Logged loudly" means the actual cause, not just its class: this is the
    same critical-startup-path diagnosability fix as
    ``fix(graphos): surface startup failures instead of logging only their
    type`` — swallowing the message here is exactly what turned one real
    boot failure into an hours-long "graphos_bundled_skills_unready" dead
    end (HANDOFF-2026-07-22). So the failure detail MUST appear in the log,
    not be redacted.
    """
    from agent_utilities.mcp import kg_server

    session = _verified_session()
    background = MagicMock()
    with (
        patch.object(kg_server, "_get_engine", return_value=object()),
        patch.object(
            kg_server,
            "_ensure_bundled_skills_ready",
            side_effect=RuntimeError("environment-specific failure detail"),
        ),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            background,
        ),
        caplog.at_level("ERROR", logger="agent_utilities.mcp.kg_server"),
    ):
        kg_server._start_engine_bootstrap(session)

    assert "SERVING DEGRADED" in caplog.text
    assert "environment-specific failure detail" in caplog.text
    background.assert_not_called()
    from agent_utilities.skills import BUNDLED_SKILLS

    report = kg_server.bundled_skill_readiness()
    assert report["ready"] == 0
    assert sorted(report["not_ready"]) == sorted(BUNDLED_SKILLS)


def test_noncritical_bootstrap_skips_packaged_skill_reingestion() -> None:
    from agent_utilities.mcp import kg_server
    from agent_utilities.skills import BUNDLED_SKILLS

    session = _verified_session()
    ingest = MagicMock()

    class Engine:
        backend = object()
        start_task_workers = MagicMock()

    class ImmediateAuthorizedBackground:
        def __init__(self, target) -> None:
            self.target = target

        def start(self) -> None:
            with use_actor(session.actor), use_session(session):
                self.target()

    def authorized(_session, target, **_kwargs):
        return ImmediateAuthorizedBackground(target)

    with (
        patch.object(kg_server, "_get_engine", return_value=Engine()),
        patch.object(
            kg_server,
            "_ensure_bundled_skills_ready",
            return_value={
                "required": 10,
                "already_ready": 10,
                "ingested": 0,
                "ready": 10,
            },
        ),
        patch.object(kg_server, "_ingest_capabilities", ingest),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            side_effect=authorized,
        ),
        patch(
            "agent_utilities.mcp.tools.ontology_tools._sync_package_ontologies",
            return_value={},
        ),
    ):
        kg_server._start_engine_bootstrap(session)

    ingest.assert_called_once_with(
        ANY,
        skip_skill_names=frozenset(BUNDLED_SKILLS),
    )


def test_graphos_listener_starts_only_after_readiness_barrier() -> None:
    from agent_utilities.mcp import kg_server

    source = inspect.getsource(kg_server.mcp_server)
    assert source.index("_start_engine_bootstrap(bootstrap_session)") < source.index(
        'mcp.run(transport="stdio")'
    )


def test_graphos_stdio_uses_private_local_process_authority() -> None:
    from agent_utilities.mcp import kg_server

    session = MagicMock()
    with (
        patch(
            "agent_utilities.security.request_identity.local_process_authority_enabled",
            return_value=True,
        ),
        patch(
            "agent_utilities.security.request_identity.mint_local_process_session",
            return_value=session,
        ) as mint_local,
        patch(
            "agent_utilities.security.request_identity.acquire_process_identity_token"
        ) as acquire_external,
    ):
        assert kg_server._mint_process_session("stdio") is session

    mint_local.assert_called_once_with()
    acquire_external.assert_not_called()
    session.engine_verified_context.assert_called_once_with()


def test_graphos_network_transport_never_uses_private_local_authority() -> None:
    from agent_utilities.mcp import kg_server

    actor = replace(
        _verified_session("external-runtime").actor,
        credential_expires_at=int(time.time()) + 300,
    )
    session = MagicMock()
    with (
        patch(
            "agent_utilities.security.request_identity.local_process_authority_enabled",
            return_value=True,
        ),
        patch(
            "agent_utilities.security.request_identity.mint_local_process_session"
        ) as mint_local,
        patch(
            "agent_utilities.security.request_identity.acquire_process_identity_token",
            return_value="header.payload.signature",
        ),
        patch(
            "agent_utilities.security.request_identity.mint_actor_from_token_sync",
            return_value=actor,
        ),
        patch(
            "agent_utilities.security.request_identity.mint_graph_session",
            return_value=session,
        ) as mint_session,
    ):
        assert kg_server._mint_process_session("streamable-http") is session

    mint_local.assert_not_called()
    session.engine_verified_context.assert_called_once_with()
    minted_actor = mint_session.call_args.args[0]
    assert minted_actor.credential_lease is not None
    assert minted_actor.credential_lease.expires_at == actor.credential_expires_at


@pytest.mark.asyncio
async def test_stdio_process_authority_renews_off_the_active_event_loop() -> None:
    from agent_utilities.mcp import kg_server

    current = _verified_session("expiring-runtime")
    lease = CredentialLease(int(time.time()) - 1)
    current = replace(
        current,
        actor=replace(
            current.actor,
            credential_expires_at=int(time.time()) - 1,
            credential_lease=lease,
        ),
    )
    event_loop_thread = threading.get_ident()
    renewal_threads: list[int] = []

    def renew(session: GraphSession) -> GraphSession:
        renewal_threads.append(threading.get_ident())
        # This would raise if renewal still ran inside the active MCP loop.
        asyncio.run(asyncio.sleep(0))
        lease.renew(int(time.time()) + 300)
        return session

    with (
        patch.object(kg_server, "_PROCESS_SESSION", current),
        patch.object(kg_server, "_refresh_process_authority", side_effect=renew) as mint,
        suspend_session(),
        use_actor(current.actor),
    ):
        selected = await kg_server._ensure_process_authority_current()
        with kg_server.verified_tool_session_scope() as scoped:
            assert scoped is current

    assert selected is current
    mint.assert_called_once_with(current)
    assert renewal_threads and renewal_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_stdio_expired_static_authority_fails_closed_when_renewal_fails() -> None:
    from agent_utilities.mcp import kg_server

    expired = _verified_session("expired-runtime")
    expired = replace(
        expired,
        actor=replace(
            expired.actor,
            credential_expires_at=int(time.time()) - 1,
            credential_lease=CredentialLease(int(time.time()) - 1),
        ),
    )
    with (
        patch.object(kg_server, "_PROCESS_SESSION", expired),
        patch.object(
            kg_server,
            "_refresh_process_authority",
            side_effect=RuntimeError("renewal rejected"),
        ),
        suspend_session(),
        pytest.raises(RuntimeError, match="renewal rejected"),
    ):
        await kg_server._ensure_process_authority_current()


def test_process_authority_refresh_updates_only_the_shared_expiry() -> None:
    from agent_utilities.mcp import kg_server

    lease = CredentialLease(int(time.time()) - 1)
    session = _verified_session("stable-runtime")
    session = replace(
        session,
        actor=replace(
            session.actor,
            credential_expires_at=lease.expires_at,
            credential_lease=lease,
        ),
    )
    renewed_actor = replace(
        session.actor,
        credential_expires_at=int(time.time()) + 300,
        credential_lease=None,
    )
    with (
        patch(
            "agent_utilities.security.request_identity.acquire_process_identity_token",
            return_value="synthetic.jwt.material",
        ),
        patch(
            "agent_utilities.security.request_identity.mint_actor_from_token_sync",
            return_value=renewed_actor,
        ),
    ):
        assert kg_server._refresh_process_authority(session) is session

    assert lease.expires_at == renewed_actor.credential_expires_at
    assert session.actor.actor_id == "stable-runtime"


def test_process_authority_refresh_rejects_identity_drift() -> None:
    from agent_utilities.mcp import kg_server

    lease = CredentialLease(int(time.time()) - 1)
    session = _verified_session("stable-runtime")
    session = replace(
        session,
        actor=replace(
            session.actor,
            credential_expires_at=lease.expires_at,
            credential_lease=lease,
        ),
    )
    changed_actor = replace(
        session.actor,
        actor_id="different-runtime",
        credential_expires_at=int(time.time()) + 300,
        credential_lease=None,
    )
    with (
        patch(
            "agent_utilities.security.request_identity.acquire_process_identity_token",
            return_value="synthetic.jwt.material",
        ),
        patch(
            "agent_utilities.security.request_identity.mint_actor_from_token_sync",
            return_value=changed_actor,
        ),
        pytest.raises(RuntimeError, match="changed during renewal"),
    ):
        kg_server._refresh_process_authority(session)

    assert lease.expires_at < int(time.time())


def test_background_worker_observes_shared_process_lease_rollover() -> None:
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _authorized_background_thread,
    )

    lease = CredentialLease(int(time.time()) + 300)
    session = _verified_session("renewable-worker")
    session = replace(
        session,
        actor=replace(
            session.actor,
            credential_expires_at=lease.expires_at,
            credential_lease=lease,
        ),
    )
    ready = threading.Event()
    check_expired = threading.Event()
    expired_seen = threading.Event()
    check_renewed = threading.Event()
    finished = threading.Event()
    observations: list[str] = []

    def worker() -> None:
        ready.set()
        assert check_expired.wait(2.0)
        try:
            current_session().ensure_authority_current()  # type: ignore[union-attr]
        except SessionExpiredError:
            observations.append("expired")
        expired_seen.set()
        assert check_renewed.wait(2.0)
        current_session().ensure_authority_current()  # type: ignore[union-attr]
        observations.append("renewed")
        finished.set()

    thread = _authorized_background_thread(session, worker, name="LeaseRolloverTest")
    thread.start()
    assert ready.wait(2.0)
    lease.renew(int(time.time()) - 1)
    check_expired.set()
    assert expired_seen.wait(2.0)
    lease.renew(int(time.time()) + 300)
    check_renewed.set()
    assert finished.wait(2.0)
    thread.join(2.0)
    assert observations == ["expired", "renewed"]


@pytest.mark.parametrize("failure_point", ["security", "readiness"])
def test_graphos_startup_failure_releases_process_authority(failure_point: str) -> None:
    """Every post-mint startup failure tears down renewable authority state."""
    from agent_utilities.mcp import kg_server

    args = MagicMock()
    args.transport = "stdio"
    args.host = "127.0.0.1"
    args.port = 8000
    args.auth_type = "none"
    mcp = MagicMock()
    session = MagicMock()
    fleet = MagicMock()
    fleet.aclose = AsyncMock()
    security = MagicMock()
    readiness = MagicMock()
    failure = RuntimeError(f"{failure_point} failed")
    (security if failure_point == "security" else readiness).side_effect = failure

    with (
        patch("agent_utilities.core.config.load_config"),
        patch.object(kg_server, "_configure_graphos_otel"),
        patch.object(kg_server, "_build_server", return_value=(args, mcp, [])),
        patch(
            "agent_utilities.mcp.multiplexer.attach_fleet_loader",
            return_value=fleet,
        ),
        patch.object(kg_server, "_mint_process_session", return_value=session),
        patch.object(kg_server, "_start_process_authority_supervisor") as start,
        patch.object(kg_server, "_stop_process_authority_supervisor") as stop,
        patch(
            "agent_utilities.security.request_identity.apply_served_security_profile",
            security,
        ),
        patch.object(kg_server, "_start_engine_bootstrap", readiness),
        patch.object(kg_server, "_PROCESS_SESSION", None),
    ):
        with pytest.raises(RuntimeError) as captured:
            kg_server.mcp_server()

        assert captured.value is failure
        start.assert_called_once_with(session)
        stop.assert_called_once_with()
        fleet.aclose.assert_awaited_once_with()
        mcp.run.assert_not_called()
        assert kg_server._PROCESS_SESSION is None
