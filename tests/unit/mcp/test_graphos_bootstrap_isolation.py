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
from types import SimpleNamespace
from typing import Any
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


def test_materialization_gate_waits_for_complete_valid_manifest() -> None:
    from agent_utilities.mcp import kg_server

    manifests = iter(
        [
            [
                {
                    "name": "__commons__",
                    "materialization": "partial",
                    "valid": False,
                    "completeness_cursor": {
                        "node_offset": 4096,
                        "edge_offset": 0,
                    },
                }
            ],
            [
                {
                    "name": "__commons__",
                    "materialization": "complete",
                    "valid": True,
                    "completeness_cursor": None,
                }
            ],
        ]
    )
    client = SimpleNamespace(tenants=SimpleNamespace(list=lambda: next(manifests)))
    engine = SimpleNamespace(
        graph_name="__commons__",
        client=client,
        query_cypher=MagicMock(
            side_effect=RuntimeError(
                '{"code":"PARTIAL_MATERIALIZATION","retryable":true}'
            )
        ),
    )

    report = kg_server._wait_for_engine_materialization(
        engine,
        timeout_seconds=1.0,
        poll_seconds=0.0,
    )

    assert report["materialization"] == "complete"
    assert report["valid"] is True
    engine.query_cypher.assert_called_once()


def test_materialization_gate_resolves_high_level_engine_authority() -> None:
    from agent_utilities.mcp import kg_server

    native = SimpleNamespace(
        graph_name="__commons__",
        client=SimpleNamespace(tenants=SimpleNamespace(list=MagicMock())),
        query_cypher=MagicMock(return_value=[]),
    )

    report = kg_server._wait_for_engine_materialization(
        SimpleNamespace(graph_compute=native)
    )

    assert report == {
        "graph": "__commons__",
        "materialization": "complete",
        "valid": True,
    }
    native.query_cypher.assert_called_once()
    native.client.tenants.list.assert_not_called()


def test_materialization_gate_probes_when_rls_hides_manifest() -> None:
    from agent_utilities.mcp import kg_server

    engine = SimpleNamespace(
        graph_name="__secrets__",
        client=SimpleNamespace(tenants=SimpleNamespace(list=MagicMock(return_value=[]))),
        query_cypher=MagicMock(
            side_effect=[
                RuntimeError('{"code":"PARTIAL_MATERIALIZATION"}'),
                RuntimeError('{"code":"PARTIAL_MATERIALIZATION"}'),
                [],
            ]
        ),
    )

    report = kg_server._wait_for_engine_materialization(
        engine,
        timeout_seconds=1.0,
        poll_seconds=0.0,
    )

    assert report == {
        "graph": "__secrets__",
        "materialization": "complete",
        "valid": True,
        "manifest_visible": False,
    }
    assert engine.client.tenants.list.call_count == 1
    assert engine.query_cypher.call_count == 3


def test_materialization_gate_preserves_nonpartial_engine_error() -> None:
    from agent_utilities.mcp import kg_server

    engine = SimpleNamespace(
        graph_name="__commons__",
        client=SimpleNamespace(tenants=SimpleNamespace(list=MagicMock())),
        query_cypher=MagicMock(side_effect=PermissionError("access denied")),
    )

    with pytest.raises(PermissionError, match="access denied"):
        kg_server._wait_for_engine_materialization(engine)
    engine.client.tenants.list.assert_not_called()


def test_materialization_gate_precedes_skill_and_background_bootstrap() -> None:
    from agent_utilities.mcp import kg_server

    session = _verified_session()
    calls: list[str] = []

    class Engine:
        backend = object()

        def start_background_daemons(self) -> None:
            calls.append("daemons")

    class DeferredBackground:
        def start(self) -> None:
            calls.append("background")

    with (
        patch.object(kg_server, "_get_engine", return_value=Engine()),
        patch.object(
            kg_server,
            "_wait_for_engine_materialization",
            side_effect=lambda _engine: calls.append("materialization"),
        ),
        patch.object(
            kg_server,
            "_ensure_bundled_skills_ready",
            side_effect=lambda _engine: (
                calls.append("skills")
                or {
                    "required": 10,
                    "already_ready": 10,
                    "ingested": 0,
                    "ready": 10,
                }
            ),
        ),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            return_value=DeferredBackground(),
        ),
    ):
        kg_server._start_engine_bootstrap(session)

    assert calls == ["materialization", "skills", "daemons", "background"]


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
    # Deliberate contract update: the failure log now PRESERVES the exception
    # message (this is a diagnosability fix — an operator needs to see WHICH
    # packaged skill failed and why, not just "graphos_bundled_skills_unready"
    # for every distinct cause) instead of collapsing it to the class name.
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
    plan = MagicMock()

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
        patch.object(kg_server, "_run_boot_hydration_plan", plan),
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

    plan.assert_called_once_with(
        ANY,
        skip_skill_names=frozenset(BUNDLED_SKILLS),
    )


def test_noncritical_bootstrap_runs_phase_f_hydration_legs() -> None:
    """Phase F (ingestion-hydration-program.md §3): the background bootstrap
    thread drives the prompt (C) and self-tool-surface (E) boot-hydration legs,
    in addition to the pre-existing capability ingest, on every boot."""
    from agent_utilities.mcp import kg_server

    session = _verified_session()
    plan = MagicMock()

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

    engine_instance = Engine()

    with (
        patch.object(kg_server, "_get_engine", return_value=engine_instance),
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
        patch.object(kg_server, "_run_boot_hydration_plan", plan),
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

    plan.assert_called_once_with(engine_instance, skip_skill_names=ANY)


def test_boot_hydration_plan_uses_fixed_priority_and_queues_only_configured_work() -> (
    None
):
    """The plan is stable: runnable metadata first, then prompts, ontology,
    and only then the incremental code/connector owners."""
    from agent_utilities.mcp import kg_server

    calls: list[str] = []
    records: list[tuple[str, int, str]] = []
    engine = object()
    with (
        patch.object(
            kg_server,
            "_ingest_capabilities",
            side_effect=lambda *_a, **_k: calls.append("capabilities"),
        ),
        patch.object(
            kg_server,
            "_ingest_self_tool_surface_at_boot",
            side_effect=lambda *_a: calls.append("self"),
        ),
        patch.object(
            kg_server,
            "_ingest_prompts_at_boot",
            side_effect=lambda: calls.append("prompts"),
        ),
        patch.object(
            kg_server,
            "_enqueue_fleet_tool_schema_hydration",
            side_effect=lambda *_a: calls.append("fleet"),
        ),
        patch.object(
            kg_server,
            "_sync_ontologies_at_boot",
            side_effect=lambda *_a: calls.append("ontologies"),
        ),
        patch.object(
            kg_server,
            "_hydrate_code_and_configured_connectors",
            side_effect=lambda *_a: calls.append("sources"),
        ),
        patch.object(
            kg_server,
            "_record_boot_hydration_step",
            side_effect=lambda _e, n, p, s: records.append((n, p, s)),
        ),
    ):
        kg_server._run_boot_hydration_plan(
            engine, skip_skill_names=frozenset({"bundled"})
        )

    assert calls == [
        "self",
        "fleet",
        "capabilities",
        "prompts",
        "ontologies",
        "sources",
    ]
    assert [
        (name, priority) for name, priority, status in records if status == "running"
    ] == [
        ("graphos_tool_surface", 1),
        ("fleet_tool_schemas", 1),
        ("capabilities", 1),
        ("prompts", 2),
        ("ontologies", 3),
        ("code_and_connectors", 4),
    ]


def test_fleet_tool_schema_boot_hydration_is_durable_priority_one() -> None:
    from agent_utilities.mcp import kg_server

    engine = MagicMock()
    engine.submit_task.return_value = "job-fleet"
    kg_server._enqueue_fleet_tool_schema_hydration(engine)
    engine.submit_task.assert_called_once_with(
        target_path="fleet",
        is_codebase=False,
        provenance={"sync_mode": "delta", "boot_hydration": True},
        task_type="connector_sync",
        priority=1,
    )


def test_boot_code_hydration_falls_back_to_workspace_manifest(monkeypatch) -> None:
    from agent_utilities.mcp import kg_server

    breadth = MagicMock()
    sweep = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.core.config.config.kg_breadth_library_roots", ""
    )
    monkeypatch.setattr("agent_utilities.core.config.config.kg_breadth_repo_roots", "")
    with (
        patch(
            "agent_utilities.core.workspace_config.workspace_project_roots",
            return_value=["/workspace/repo-a", "/workspace/repo-b"],
        ),
        patch(
            "agent_utilities.knowledge_graph.assimilation.breadth_ingest.run_breadth_ingest",
            breadth,
        ),
        patch(
            "agent_utilities.knowledge_graph.core.source_sync.sweep_all_sources",
            sweep,
        ),
    ):
        kg_server._hydrate_code_and_configured_connectors(object())

    breadth.assert_called_once_with(
        ANY,
        library_roots=[],
        repo_roots=["/workspace/repo-a", "/workspace/repo-b"],
    )
    sweep.assert_called_once_with(ANY, mode="delta", enqueue=True, priority=3)


def test_prompt_boot_hydration_leg_is_isolated_from_failure(caplog) -> None:
    """A failure in the Phase C prompt leg is caught, logged, and never raised —
    it must not prevent the self-tool-surface leg (or ontology sync) from
    running afterward."""
    from agent_utilities.mcp import kg_server

    with (
        patch(
            "agent_utilities.agent.registry_builder.ingest_prompts_to_graph",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level("ERROR", logger="agent_utilities.mcp.kg_server"),
    ):
        kg_server._ingest_prompts_at_boot()  # must not raise

    assert "Prompt-base boot ingestion failed" in caplog.text


def test_self_tool_surface_boot_hydration_leg_is_isolated_from_failure(caplog) -> None:
    """A failure in the Phase E self-tool-surface leg is caught and logged, not
    raised — same best-effort isolation contract as every other boot leg.

    ``IngestionEngine._ingest_self_tools`` already isolates a *provider*
    exception internally (a buggy provider must not break ingest — see its own
    docstring), so a raising provider alone would never reach this wrapper's
    ``except``. Fail the registration call itself instead, to exercise this
    wrapper's OWN isolation (e.g. a broken import or a bad ``engine`` handle).
    """
    from agent_utilities.mcp import kg_server

    with (
        patch(
            "agent_utilities.knowledge_graph.ingestion.engine.register_self_tool_surface_provider",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level("ERROR", logger="agent_utilities.mcp.kg_server"),
    ):
        kg_server._ingest_self_tool_surface_at_boot(object())  # must not raise

    assert "Self tool-surface boot ingestion failed" in caplog.text


def test_self_tool_surface_provider_failure_is_isolated_inside_ingest(caplog) -> None:
    """A separate, deeper isolation layer: even if the REGISTERED provider
    itself raises, ``_ingest_self_tools`` catches it internally and returns a
    ``failed`` :class:`IngestionResult` — the boot wrapper still completes
    normally (logging the failed status at INFO, not ERROR) because nothing
    propagates out of ``_ingest_self_tools`` for it to catch."""
    from agent_utilities.mcp import kg_server

    with (
        patch.object(
            kg_server,
            "_graphos_self_tool_surface",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level("INFO", logger="agent_utilities.mcp.kg_server"),
    ):
        kg_server._ingest_self_tool_surface_at_boot(object())  # must not raise

    assert "Self tool-surface boot ingestion failed" not in caplog.text
    assert "status=failed" in caplog.text


def test_self_tool_surface_boot_hydration_registers_provider_and_ingests() -> None:
    """The Phase E leg registers the in-process provider and drives
    ``IngestionEngine._ingest_self_tools`` exactly once per boot."""
    from agent_utilities.knowledge_graph.ingestion import engine as ingestion_engine
    from agent_utilities.mcp import kg_server

    fake_kg_engine = MagicMock()
    fake_kg_engine.backend = MagicMock()
    fake_kg_engine.graph_compute = MagicMock()

    try:
        kg_server._ingest_self_tool_surface_at_boot(fake_kg_engine)
        # The registered provider must be exactly the module's own closure —
        # never rebuilt/rewrapped — so re-registration stays idempotent.
        assert (
            ingestion_engine._SELF_TOOL_SURFACE_PROVIDER
            is kg_server._graphos_self_tool_surface
        )
    finally:
        ingestion_engine.register_self_tool_surface_provider(None)


def test_graphos_self_tool_surface_reads_registered_tools_in_process() -> None:
    """The provider closure is a pure, synchronous read of ``REGISTERED_TOOLS`` —
    no network, no self-probe (CONCEPT:AU-KG.ingest.self-tool-surface)."""
    from agent_utilities.mcp import kg_server

    def _documented() -> None:
        """A documented tool."""

    def _undocumented() -> None:
        pass

    registered_before = dict(kg_server.REGISTERED_TOOLS)
    try:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS["graph_documented"] = _documented
        kg_server.REGISTERED_TOOLS["graph_undocumented"] = _undocumented

        surface = kg_server._graphos_self_tool_surface()
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(registered_before)

    assert surface == [
        {"name": "graph_documented", "description": "A documented tool."},
        {"name": "graph_undocumented", "description": ""},
    ]


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
        patch.object(
            kg_server, "_refresh_process_authority", side_effect=renew
        ) as mint,
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
    session = _verified_session("graphos-bootstrap")
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


@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
def test_mcp_server_selects_local_engine_path_for_both_transports(
    monkeypatch, tmp_path, transport: str
) -> None:
    """The unified/self-contained engine path is transport-agnostic (unified-
    binary-program W-D). ``--transport`` selects only how ``graph-os`` itself
    is served (stdio process-identity bootstrap vs. served-profile identity
    enforcement); whether the ENGINE it uses is local or remote is decided
    solely by ``GRAPH_SERVICE_ENDPOINTS`` via the one engine resolver
    (``engine_resolver.resolve_engine``). Nothing in ``mcp_server()`` branches
    on transport between minting the process session and calling
    ``_get_engine()`` — this proves it by letting the REAL resolver run (with
    only the deepest socket-probe location patched to a private tmp_path so
    the test never depends on ambient engine state) and asserting a
    non-"remote" resolution for BOTH transports when
    ``GRAPH_SERVICE_ENDPOINTS`` is unset.
    """
    from agent_utilities.knowledge_graph.core import engine_resolver as er
    from agent_utilities.mcp import kg_server

    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    sock_path = str(tmp_path / "unified-local-engine.sock")
    monkeypatch.setattr(er, "resolve_endpoints", lambda _cfg: [f"unix://{sock_path}"])

    real_resolve_engine = er.resolve_engine
    resolved_calls: list[Any] = []
    seen_endpoints_configured: list[bool] = []

    def _spy_get_engine():
        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
        seen_endpoints_configured.append(bool(cfg.graph_service_endpoints))
        resolved_calls.append(real_resolve_engine(cfg, "__commons__"))
        return SimpleNamespace(backend=None)

    args = MagicMock()
    args.transport = transport
    args.host = "127.0.0.1"
    args.port = 8000
    args.auth_type = "none"
    mcp = MagicMock()
    fleet = MagicMock()
    fleet.aclose = AsyncMock()
    session = _verified_session("unified-bootstrap")

    with (
        patch("agent_utilities.core.config.load_config"),
        patch.object(kg_server, "_configure_graphos_otel"),
        patch.object(kg_server, "_build_server", return_value=(args, mcp, [])),
        patch(
            "agent_utilities.mcp.multiplexer.attach_fleet_loader",
            return_value=fleet,
        ),
        patch.object(kg_server, "_mint_process_session", return_value=session),
        patch.object(kg_server, "_start_process_authority_supervisor"),
        patch.object(kg_server, "_stop_process_authority_supervisor"),
        patch(
            "agent_utilities.security.request_identity.apply_served_security_profile"
        ),
        patch(
            "agent_utilities.mcp.server_factory.mcp_network_run_kwargs",
            return_value={},
        ),
        patch.object(kg_server, "_get_engine", side_effect=_spy_get_engine),
        patch.object(
            kg_server,
            "_ensure_bundled_skills_ready",
            return_value={
                "required": 0,
                "already_ready": 0,
                "ingested": 0,
                "ready": 0,
            },
        ),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            return_value=MagicMock(),
        ),
        patch.object(kg_server, "_PROCESS_SESSION", None),
    ):
        kg_server.mcp_server()

    # _get_engine() is reached at least once (the readiness barrier) — for
    # streamable-http it is reached a second time (start_co_services) too;
    # every reachable call must see the SAME unset-endpoints, non-remote
    # resolution, proving the selection is transport-symmetric.
    assert resolved_calls, "mcp_server() never reached the engine resolver"
    assert seen_endpoints_configured == [False] * len(seen_endpoints_configured)
    assert all(resolved.mode != "remote" for resolved in resolved_calls)
    assert all(
        resolved.endpoint == f"unix://{sock_path}" for resolved in resolved_calls
    )

    if transport == "stdio":
        mcp.run.assert_called_once_with(transport="stdio")
    else:
        mcp.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000
        )
