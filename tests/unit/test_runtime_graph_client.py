"""The AU runtime binds non-owning EG views to verified graph authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.api.runtime import AgentRuntime
from agent_utilities.api.session import GraphSession, SessionRequiredError, use_session
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session() -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="agent:runtime-test",
            actor_type=ActorType.AI_AGENT,
            tenant_id="tenant:test",
            authenticated=True,
        ),
        tenant="tenant:test",
        graph="tenant-test",
        scopes=frozenset({"kg:read"}),
    )


def test_graph_client_reuses_the_process_transport() -> None:
    graphs: list[str] = []

    def for_graph(graph: str) -> SimpleNamespace:
        graphs.append(graph)
        return SimpleNamespace(async_client=object())

    runtime = AgentRuntime(
        SimpleNamespace(graph_compute=SimpleNamespace(for_graph=for_graph)), "host"
    )
    with use_session(_session()):
        client = runtime.graph_client("tenant-test")
    assert client is not None
    assert graphs == ["tenant-test"]


def test_graph_client_rejects_unverified_or_retargeted_views() -> None:
    graphs: list[str] = []

    def for_graph(graph: str) -> SimpleNamespace:
        graphs.append(graph)
        return SimpleNamespace(async_client=object())

    runtime = AgentRuntime(
        SimpleNamespace(graph_compute=SimpleNamespace(for_graph=for_graph)), "host"
    )
    with pytest.raises(SessionRequiredError):
        runtime.graph_client("tenant-test")
    with use_session(_session()), pytest.raises(SessionRequiredError):
        runtime.graph_client("other-graph")
    assert graphs == []
