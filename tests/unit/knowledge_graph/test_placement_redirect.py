"""Placement redirects refresh the catalog fence and retry exactly once."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars

import pytest

# The compiled epistemic_graph client package must be present for these tests;
# skip the whole module cleanly when it isn't, rather than erroring out collection
# (BUG-026 un-blinding: the lean CI `gates` env deliberately excludes epistemic-graph).
_epistemic_graph_client = pytest.importorskip("epistemic_graph.client")
ChangeEnvelopeClient = _epistemic_graph_client.ChangeEnvelopeClient
StaleRouteError = _epistemic_graph_client.StaleRouteError

from agent_utilities.knowledge_graph.core import placement_catalog
from agent_utilities.knowledge_graph.core.graph_compute import (
    _CLIENT_NAMESPACES,
    _SessionRoutedAsyncClient,
    _sync_client_view,
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


class _ContextBase:
    def __init__(self) -> None:
        self._graph_name = "tenant:graph"
        self._auth_secret = "test-" + "auth-secret"
        self.calls: list[tuple[str, dict]] = []
        self.contexts: list[dict] = []
        for name in _CLIENT_NAMESPACES:
            setattr(self, name, _Namespace(self))

    @contextlib.contextmanager
    def use_verified_context(self, context):
        self.contexts.append(dict(context))
        try:
            yield self
        finally:
            self.contexts.pop()

    async def _send(self, method, _params, **_kwargs):
        self.calls.append((method, dict(self.contexts[-1])))
        return {"status": "ok"}


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


def test_single_endpoint_view_reuses_managed_client_for_placement(monkeypatch) -> None:
    """One contact reuses its non-owning session-routed transport."""
    import epistemic_graph.client as client_module

    class _SyncView:
        def __init__(self, client, loop, thread) -> None:
            self._client = client
            self._loop = loop
            self._thread = thread

    owner = type(
        "Owner",
        (),
        {
            "_client": _Base(),
            "_loop": object(),
            "_thread": object(),
            "_au_route_config": object(),
            "_au_route_endpoints": ("tcp://coordinator:9100",),
            "_au_route_endpoint": "tcp://coordinator:9100",
        },
    )()
    monkeypatch.setattr(client_module, "SyncEpistemicGraphClient", _SyncView)

    view = _sync_client_view(owner)
    factory = view._client._placement_client_factory

    assert factory is not None
    assert factory("tcp://coordinator:9100") is view
    with pytest.raises(ConnectionError, match="not the managed endpoint"):
        factory("tcp://another:9100")


def test_multi_endpoint_view_preserves_independent_catalog_failover(
    monkeypatch,
) -> None:
    """Multiple contacts retain the resolver's open-each-contact behavior."""
    import epistemic_graph.client as client_module

    class _SyncView:
        def __init__(self, client, loop, thread) -> None:
            self._client = client
            self._loop = loop
            self._thread = thread

    owner = type(
        "Owner",
        (),
        {
            "_client": _Base(),
            "_loop": object(),
            "_thread": object(),
            "_au_route_config": object(),
            "_au_route_endpoints": (
                "tcp://coordinator-a:9100",
                "tcp://coordinator-b:9100",
            ),
            "_au_route_endpoint": "tcp://coordinator-a:9100",
        },
    )()
    monkeypatch.setattr(client_module, "SyncEpistemicGraphClient", _SyncView)

    view = _sync_client_view(owner)

    assert view._client._placement_client_factory is None


def test_placement_route_bypasses_routing_and_binds_verified_session(
    monkeypatch,
) -> None:
    """The reused catalog RPC cannot recurse into placement resolution."""
    base = _ContextBase()
    view = _SessionRoutedAsyncClient(
        base,
        route_config=object(),
        route_endpoints=("tcp://coordinator:9100",),
        transport_endpoint="tcp://coordinator:9100",
    )

    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("PlacementRoute must not resolve placement")

    monkeypatch.setattr(
        placement_catalog,
        "resolve_placement",
        unexpected_resolution,
    )
    session = _verified_session()
    with use_session(session):
        result = asyncio.run(
            view._send(
                "PlacementRoute",
                {"tenant": "tenant", "sub_key": "graph", "client_epoch": 0},
            )
        )

    assert result == {"status": "ok"}
    assert base.calls == [
        ("PlacementRoute", session.engine_verified_context()),
    ]


def test_routed_request_forwards_managed_catalog_factory_and_session(
    monkeypatch,
) -> None:
    """Data routing resolves through the injected authority under one session."""
    base = _ContextBase()
    catalog_client = object()

    def client_factory(_contact: str):
        return catalog_client

    def resolve(*_args, **kwargs):
        assert kwargs["client_factory"]("tcp://coordinator:9100") is catalog_client
        return PlacementResult("tcp://coordinator:9100", 0, 0, 0, True)

    monkeypatch.setattr(placement_catalog, "resolve_placement", resolve)
    view = _SessionRoutedAsyncClient(
        base,
        route_config=object(),
        route_endpoints=("tcp://coordinator:9100",),
        transport_endpoint="tcp://coordinator:9100",
        placement_client_factory=client_factory,
    )
    session = _verified_session()
    with use_session(session):
        result = asyncio.run(view._send("NodesCount"))

    assert result == {"status": "ok"}
    assert base.calls == [("NodesCount", session.engine_verified_context())]


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
