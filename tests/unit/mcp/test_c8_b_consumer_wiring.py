"""MCP composition regressions for the AU C8-B registry seam.

These tests drive the real ``graph_query`` and ``graph_catalog`` registrations
through ``kg_server._execute_tool``.  The process registry is pre-bound to a
sentinel, while the real registry factory and source-catalog projection are
observed as pass-through seams.  This proves that composition supplies the
registry after the verified session check, rather than relying on a lower KG
layer to import ``kg_server`` and discover one itself.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Callable
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core import session as session_module
from agent_utilities.knowledge_graph.core import source_catalog
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    use_session,
)
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import query_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from tests.wiring import observe


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="c8b-wiring-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="c8b-wiring-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph="c8b-wiring-graph",
        policy_version="c8b-wiring-policy",
        audience="agent-services",
    )


def _register_query_tools() -> None:
    from fastmcp import FastMCP

    query_tools.register_query_tools(FastMCP("c8b-wiring-test"))


class _FederatedEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute_federated_query(
        self,
        reference_id: str,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        registry: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "reference_id": reference_id,
                "query": query,
                "parameters": parameters,
                "registry": registry,
                "session": session_module.current_session(),
            }
        )
        return [{"reference_id": reference_id}]


@pytest.mark.asyncio
async def test_graph_query_mcp_composition_forwards_registry_after_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real MCP graph-query entrypoint forwards the exact bound registry.

    ``_run_graph_query_federated`` is synchronous, so ``_execute_tool`` sends
    it through the real dispatch/to-thread path.  The fake is only the
    downstream engine: the registry factory and session resolver remain real
    functions under ``observe``.
    """
    _register_query_tools()
    registry = object()
    engine = _FederatedEngine()
    events: list[tuple[str, bool, bool, object | None]] = []
    session = _session()

    monkeypatch.setattr(kg_server, "_CONNECTION_REGISTRY", registry)

    def _get_engine() -> _FederatedEngine:
        events.append(
            (
                "engine",
                resolved.called,
                acquired.called,
                session_module.current_session(),
            )
        )
        return engine

    monkeypatch.setattr(kg_server, "_get_engine", _get_engine)

    with (
        observe(session_module, "resolve_session") as resolved,
        observe(kg_server, "get_connection_registry") as acquired,
        use_actor(session.actor),
        use_session(session),
    ):
        result = await kg_server._execute_tool(
            "graph_query",
            cypher="SELECT * WHERE { ?s ?p ?o }",
            scope="federated",
            reference_id="external-graph",
        )

    assert result.model_dump()["error"] is None
    resolved.assert_called(
        times=1,
        why="federated graph_query must resolve the verified carrier before dependencies",
    )
    acquired_call = acquired.assert_called(
        times=1,
        why="the MCP composition root must supply the process-owned registry",
    )
    assert acquired_call.result is registry
    assert events == [("engine", True, False, session)]
    assert len(engine.calls) == 1
    assert engine.calls[0]["registry"] is registry
    assert engine.calls[0]["session"] is session


def test_graph_query_composition_does_not_touch_dependencies_without_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing carrier fails before engine or registry acquisition."""
    registry = object()
    engine_calls: list[str] = []

    def _unexpected_engine() -> object:
        engine_calls.append("engine")
        raise AssertionError("engine acquisition preceded the session guard")

    monkeypatch.setattr(kg_server, "_PROCESS_SESSION", None)
    monkeypatch.setattr(kg_server, "_CONNECTION_REGISTRY", registry)
    monkeypatch.setattr(kg_server, "_get_engine", _unexpected_engine)

    with (
        observe(session_module, "resolve_session") as resolved,
        observe(kg_server, "get_connection_registry") as acquired,
    ):
        raw = contextvars.Context().run(
            query_tools._run_graph_query_federated,
            "SELECT * WHERE { ?s ?p ?o }",
            "external-graph",
            {},
        )

    payload = json.loads(raw)
    assert payload["error"]["code"] == "operation_failed"
    resolved.assert_called(
        times=1,
        why="the federation composition helper must require a verified carrier",
    )
    acquired.assert_not_called(
        why="an unauthenticated federation request must not initialize the registry",
    )
    assert engine_calls == []


class _SourceRegistry:
    def __init__(
        self,
        events: list[tuple[str, int, bool]],
        state: Callable[[], tuple[int, bool]],
    ) -> None:
        self._events = events
        self._state = state

    def status(self) -> dict[str, Any]:
        resolved_count, acquired_called = self._state()
        self._events.append(("registry.status", resolved_count, acquired_called))
        return {"connections": []}

    def export_specs(self) -> list[dict[str, Any]]:
        resolved_count, acquired_called = self._state()
        self._events.append(("registry.export_specs", resolved_count, acquired_called))
        return []


@pytest.mark.asyncio
async def test_graph_catalog_mcp_composition_forwards_registry_after_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered ``graph_catalog`` source leg uses its bound registry."""
    _register_query_tools()
    events: list[tuple[str, int, bool]] = []
    session = _session()

    # The state callback is evaluated only after both observers have been
    # entered, when the registry methods run inside the live catalog path.
    registry = _SourceRegistry(events, lambda: (resolved.count, acquired.called))

    monkeypatch.setattr(kg_server, "_CONNECTION_REGISTRY", registry)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())

    async def _catalog_stub() -> dict[str, Any]:
        return {"available": False, "reason": "test fixture"}

    monkeypatch.setattr(query_tools, "_catalog_graphs", _catalog_stub)
    monkeypatch.setattr(query_tools, "_catalog_sql", _catalog_stub)
    monkeypatch.setattr(query_tools, "_engine_domain_methods", lambda _domain: None)

    with (
        observe(session_module, "resolve_session") as resolved,
        observe(kg_server, "get_connection_registry") as acquired,
        observe(source_catalog, "build_source_catalog") as built,
        use_actor(session.actor),
        use_session(session),
    ):
        raw = await kg_server._execute_tool("graph_catalog", action="list")

    payload = json.loads(raw)
    assert "sources" in payload
    assert resolved.count == 2, (
        "graph_catalog and its source leg each guard the carrier"
    )
    acquired_call = acquired.assert_called(
        times=1,
        why="the catalog source adapter must receive the process-owned registry",
    )
    assert acquired_call.result is registry
    built_call = built.assert_called(
        times=1,
        why="graph_catalog's live source leg must invoke source-catalog composition",
    )
    assert built_call.arg("registry") is registry
    assert events == [
        ("registry.status", 2, True),
        ("registry.export_specs", 2, True),
    ]


def test_graph_catalog_composition_does_not_touch_registry_without_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source catalog helper denies before registry lookup without a carrier."""
    registry = object()
    monkeypatch.setattr(kg_server, "_PROCESS_SESSION", None)
    monkeypatch.setattr(kg_server, "_CONNECTION_REGISTRY", registry)

    async def _invoke() -> dict[str, Any]:
        return await query_tools._catalog_sources()

    with (
        observe(kg_server, "get_connection_registry") as acquired,
        observe(source_catalog, "build_source_catalog") as built,
        pytest.raises(SessionRequiredError),
    ):
        contextvars.Context().run(lambda: asyncio.run(_invoke()))

    acquired.assert_not_called(
        why="an unauthenticated catalog request must not initialize the registry",
    )
    built.assert_not_called(
        why="the source projection must not run without a verified carrier",
    )
