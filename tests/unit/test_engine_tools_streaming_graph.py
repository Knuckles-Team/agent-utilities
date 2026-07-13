"""BUG-4 (kg-exhaustive-smoke.md): ``engine_streaming(action="list_triggers")`` (and
sibling ``StreamingClient`` methods — ``cdc_read``/``fired_triggers``/``watch``/
``register_trigger``/``register_continuous_query``) failed with
``missing 1 required positional argument: 'graph'`` even though the tool's own
top-level ``graph`` field was filled in — the field only selected/pooled the
*connection* (``engine_tools._client_for``), it was never threaded into the
call itself.

Exercises the REAL ``_dispatch``/``engine_streaming`` tool path (the same
pattern ``tests/unit/test_engine_tools_uql_embed.py`` uses for ``engine_query``)
against a fake sub-client whose methods declare an explicit ``graph`` parameter
— exactly like the real ``epistemic_graph.client.StreamingClient`` — so a
regression surfaces as the same ``TypeError`` the live bug did.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

NON_ADMIN_ACTOR = ActorContext(
    actor_id="agent:marketing",
    actor_type=ActorType.AI_AGENT,
    roles=("marketing",),
    tenant_id="acme",
)


class _FakeStreamingSub:
    """Mirrors the real ``StreamingClient``'s signatures — ``graph`` is a real
    required positional/keyword parameter, so calling without it raises
    ``TypeError`` exactly like the live engine client does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def list_triggers(self, graph: str):
        self.calls.append(("list_triggers", (graph,), {}))
        return []

    def cdc_read(self, graph: str, from_seq: int = 0, *, limit: int = 0):
        self.calls.append(("cdc_read", (graph, from_seq), {"limit": limit}))
        return []


class _FakeClient:
    def __init__(self) -> None:
        self.streaming = _FakeStreamingSub()


class _SyncWrapperLikeStreamingSub:
    """Faithfully mirrors ``epistemic_graph.client.SyncEpistemicGraphClient.
    _SyncWrapper`` — the REAL shape of ``client.streaming`` in production.

    ``_SyncWrapper.__getattr__`` returns a fresh generic
    ``sync_wrapper(*args, **kwargs)`` closure per attribute access, discarding
    the wrapped async method's real parameter names. The original BUG-4 "fix"
    introspected ``fn`` (that closure) directly via ``inspect.signature``,
    which always resolves to ``(*args, **kwargs)`` — so ``"graph" in
    sig.parameters`` never fired and the graph-threading fix was silent dead
    code for every real ``_SyncWrapper``-mediated domain. The fixture above
    (``_FakeStreamingSub``) doesn't reproduce this — its methods are plain
    functions with real signatures, so it couldn't have caught the bug. THIS
    fixture reproduces the real wrapper shape (private ``_namespace`` holding
    the real signature, ``__getattr__`` returning an opaque closure) so a
    regression here fails exactly like production did.
    """

    def __init__(self) -> None:
        self._namespace = _FakeStreamingSub()

    def __getattr__(self, name: str):
        target = getattr(self._namespace, name)

        def sync_wrapper(*args, **kwargs):
            return target(*args, **kwargs)

        return sync_wrapper


class _SyncWrapperLikeClient:
    def __init__(self) -> None:
        self.streaming = _SyncWrapperLikeStreamingSub()


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


def _tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["engine_streaming"]


def test_list_triggers_threads_resolved_default_graph(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    monkeypatch.setattr(
        engine_tools, "_resolve_graph_name", lambda graph: graph or "commons-default"
    )

    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                _tool()(action="list_triggers", params_json="{}", graph="")
            )
        )
    assert out == []
    assert client.streaming.calls == [("list_triggers", ("commons-default",), {})]


def test_list_triggers_threads_explicit_top_level_graph(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                _tool()(action="list_triggers", params_json="{}", graph="tenant-x")
            )
        )
    assert out == []
    assert client.streaming.calls == [("list_triggers", ("tenant-x",), {})]


def test_caller_supplied_graph_in_params_json_wins(monkeypatch):
    # If the caller already put `graph` in params_json, don't overwrite it.
    client = _FakeClient()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            _tool()(
                action="list_triggers",
                params_json=json.dumps({"graph": "explicit-in-params"}),
                graph="",
            )
        )
    assert client.streaming.calls == [
        ("list_triggers", ("explicit-in-params",), {})
    ]


def test_cdc_read_also_gets_graph_threaded(monkeypatch):
    # Not just list_triggers — any streaming method declaring `graph`.
    client = _FakeClient()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    monkeypatch.setattr(engine_tools, "_resolve_graph_name", lambda graph: graph or "d")

    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            _tool()(
                action="cdc_read",
                params_json=json.dumps({"from_seq": 5, "limit": 10}),
                graph="",
            )
        )
    assert client.streaming.calls == [("cdc_read", ("d", 5), {"limit": 10})]


def test_list_triggers_threads_graph_through_real_sync_wrapper_shape(monkeypatch):
    """Regression for the follow-up to BUG-4: reproduces the REAL
    ``SyncEpistemicGraphClient._SyncWrapper`` shape (see
    ``_SyncWrapperLikeClient`` above) where ``inspect.signature(fn)`` on the
    dispatched method itself is USELESS (always ``(*args, **kwargs)``).
    Before the follow-up fix, this raised
    ``TypeError: list_triggers() missing 1 required positional argument:
    'graph'`` — identical to the live production failure — because the
    graph-threading check never saw ``graph`` in the closure's fake
    signature. Confirmed live in-pod against the real engine client too
    (kubectl exec smoke, 2026-07-13): ``engine_streaming(list_triggers)``
    went from a missing-arg ``TypeError`` to a clean ``[]``.
    """
    client = _SyncWrapperLikeClient()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    monkeypatch.setattr(
        engine_tools, "_resolve_graph_name", lambda graph: graph or "commons-default"
    )

    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                _tool()(action="list_triggers", params_json="{}", graph="")
            )
        )
    assert out == []
    assert client.streaming._namespace.calls == [
        ("list_triggers", ("commons-default",), {})
    ]
