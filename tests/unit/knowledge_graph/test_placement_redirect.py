"""Placement redirects refresh the catalog fence and retry exactly once."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars

import pytest
from epistemic_graph.client import ChangeEnvelopeClient, StaleRouteError

from agent_utilities.knowledge_graph.core import placement_catalog
from agent_utilities.knowledge_graph.core.graph_compute import (
    _CLIENT_NAMESPACES,
    _SessionRoutedAsyncClient,
)
from agent_utilities.knowledge_graph.core.placement_catalog import PlacementResult
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    use_session,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


class _Namespace:
    def __init__(self, client):
        self._client = client


class _Base:
    def __init__(self) -> None:
        self._graph_name = "tenant:graph"
        self._auth_secret = "test-" + "auth-secret"
        self.calls: list[dict] = []
        for name in _CLIENT_NAMESPACES:
            setattr(self, name, _Namespace(self))

    @contextlib.contextmanager
    def use_verified_context(self, _context):
        yield self

    async def _send(self, _method, params, **_kwargs):
        self.calls.append(params)
        if len(self.calls) == 1:
            raise StaleRouteError(
                "route moved", {"graph": "tenant:graph", "group": 2, "epoch": 2}
            )
        return {"status": "committed"}


class _ChangeReadBase:
    def __init__(self) -> None:
        self._graph_name = "tenant:graph"
        self._auth_secret = "test-" + "auth-secret"
        self.calls: list[tuple[str, dict, dict]] = []
        self.contexts: list[dict] = []
        for name in _CLIENT_NAMESPACES:
            setattr(self, name, _Namespace(self))
        self.changes = ChangeEnvelopeClient(self)

    @staticmethod
    def _verified_tenant() -> str:
        return "tenant:transport-only"

    @contextlib.contextmanager
    def use_verified_context(self, context):
        self.contexts.append(context)
        try:
            yield self
        finally:
            self.contexts.pop()

    async def _send(self, method, params, **_kwargs):
        self.calls.append((method, params, dict(self.contexts[-1])))
        return None


def _verified_session() -> GraphSession:
    actor = ActorContext(
        actor_id="service:test-suite",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="tenant:test",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant:graph",
        policy_version="policy:test",
        audience="epistemic-graph-test",
    )


def test_stale_route_refreshes_fence_and_retries_once(monkeypatch):
    routes = [
        PlacementResult("tcp://coordinator:9100", 1, 7, 11, True),
        PlacementResult("tcp://coordinator:9100", 2, 8, 12, True),
    ]

    def _resolve(*_args, **kwargs):
        return routes[1] if kwargs.get("force_refresh") else routes[0]

    monkeypatch.setattr(placement_catalog, "resolve_placement", _resolve)
    base = _Base()
    view = _SessionRoutedAsyncClient(
        base,
        route_config=object(),
        route_endpoints=("tcp://coordinator:9100",),
        transport_endpoint="tcp://coordinator:9100",
    )
    params = {"envelope": {"mutation": {"placement_epoch": 0}}}
    with use_session(_verified_session()):
        result = asyncio.run(
            view._send("ApplyChangeEnvelope", params, graph="tenant:graph")
        )

    assert result == {"status": "committed"}
    assert len(base.calls) == 2
    assert base.calls[0]["envelope"]["mutation"] == {
        "placement_epoch": 1,
        "fencing_token": 11,
    }
    assert base.calls[1]["envelope"]["mutation"] == {
        "placement_epoch": 2,
        "fencing_token": 12,
    }


def test_missing_task_local_session_is_denied_before_base_dispatch() -> None:
    base = _Base()
    view = _SessionRoutedAsyncClient(
        base,
        route_config=None,
        transport_endpoint="tcp://coordinator:9100",
    )

    with pytest.raises(SessionRequiredError, match="task-local verified"):
        contextvars.Context().run(
            lambda: asyncio.run(view._send("Health", {}, graph="tenant:graph"))
        )

    assert base.calls == []


def test_unplaced_route_omits_optional_fencing_sentinel() -> None:
    params = {"envelope": {"mutation": {"placement_epoch": 7, "fencing_token": 9}}}

    bound = _SessionRoutedAsyncClient._route_bound_params(
        "ApplyChangeEnvelope",
        params,
        PlacementResult("tcp://coordinator:9100", 0, 0, 0, True),
    )

    assert bound is not None
    assert bound["envelope"]["mutation"] == {"placement_epoch": 0}
    assert params["envelope"]["mutation"] == {
        "placement_epoch": 7,
        "fencing_token": 9,
    }


def test_placed_route_requires_positive_fencing_token() -> None:
    with pytest.raises(RuntimeError, match="missing a fencing token"):
        _SessionRoutedAsyncClient._route_bound_params(
            "ApplyChangeEnvelope",
            {"envelope": {"mutation": {"placement_epoch": 0}}},
            PlacementResult("tcp://coordinator:9100", 4, 1, 0, True),
        )


def test_change_envelope_reads_bind_verified_session_tenant() -> None:
    base = _ChangeReadBase()
    view = _SessionRoutedAsyncClient(
        base,
        route_config=None,
        transport_endpoint="tcp://coordinator:9100",
    )

    async def _read_prerequisites() -> None:
        await view.changes.get("envelope:synthetic")
        await view.changes.content_version("object:synthetic")
        await view.changes.cursor("source:synthetic", "partition:synthetic")

    with use_session(_verified_session()):
        asyncio.run(_read_prerequisites())

    assert [method for method, _params, _context in base.calls] == [
        "GetChangeEnvelope",
        "GetContentVersion",
        "GetChangeCursor",
    ]
    assert all(
        params["tenant"] == "tenant:test" for _method, params, _context in base.calls
    )
    assert all(
        context["tenant"] == "tenant:test" for _method, _params, context in base.calls
    )
