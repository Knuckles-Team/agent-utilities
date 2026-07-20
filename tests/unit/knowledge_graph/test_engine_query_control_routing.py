"""Current-contract tests for control-plane query routing."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import (
    QueryMixin,
    _is_control_plane_query,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


@dataclass
class _Backend:
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def execute_read(self, query: str, params: dict) -> list[dict]:
        self.calls.append((query, params))
        return [{"id": "record-1"}]


class _Harness(QueryMixin):
    def __init__(self, *, content: _Backend, control: _Backend | None) -> None:
        self.backend = content
        self.control_backend = control


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="principal:test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-test",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant-test",
        scopes=frozenset({"kg:read"}),
        policy_version="policy-test",
        audience="agent-services",
    )


@pytest.fixture(autouse=True)
def _identity_read_policy(monkeypatch):
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "scope", lambda query, _actor: query)
    monkeypatch.setattr(secured_reads, "filter_rows", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "visible", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "audit_read", lambda *_args, **_kwargs: None)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (w:WorkItem) RETURN w",
        "MATCH (:Schedule) RETURN count(*)",
        "MATCH (p:ProfileSpan) RETURN p",
        "MATCH (w:WorkItem)-[:CREATED_BY]->(s:Schedule) RETURN w, s",
    ],
)
def test_current_control_labels_route_to_control_authority(query):
    assert _is_control_plane_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (t:Task) RETURN t",
        "MATCH (d:DeadLetterTask) RETURN d",
        "MATCH (l:Loop) RETURN l",
        "MATCH (n:Document) RETURN n",
        "MATCH (w:WorkItem)-[:ABOUT]->(n:Document) RETURN w, n",
        "MATCH (n) RETURN n",
    ],
)
def test_removed_or_content_labels_never_route_to_control_authority(query):
    assert _is_control_plane_query(query) is False


def test_control_read_uses_only_control_backend():
    content = _Backend()
    control = _Backend()
    engine = _Harness(content=content, control=control)
    session = _session()

    with use_session(session):
        rows = engine.query_cypher("MATCH (w:WorkItem) RETURN w", session=session)

    assert rows == [{"id": "record-1"}]
    assert len(control.calls) == 1
    assert content.calls == []


def test_content_read_uses_only_content_backend():
    content = _Backend()
    control = _Backend()
    engine = _Harness(content=content, control=control)
    session = _session()

    with use_session(session):
        rows = engine.query_cypher("MATCH (l:Loop) RETURN l", session=session)

    assert rows == [{"id": "record-1"}]
    assert len(content.calls) == 1
    assert control.calls == []


def test_control_read_fails_closed_without_control_authority():
    content = _Backend()
    engine = _Harness(content=content, control=None)
    session = _session()

    with (
        use_session(session),
        pytest.raises(RuntimeError, match="configured WorkItem authority"),
    ):
        engine.query_cypher("MATCH (w:WorkItem) RETURN w", session=session)

    assert content.calls == []
