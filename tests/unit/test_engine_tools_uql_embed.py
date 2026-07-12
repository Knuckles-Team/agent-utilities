"""Tests for F2 (uql-crossseam-findings.md): ``RANK BY ~"text"`` inside a UQL /
``unified`` query dispatched through ``engine_query`` (``mcp/tools/engine_tools.py``)
must be pre-embedded client-side — mirroring how ``graph_search`` embeds its query
via ``create_embedding_model`` — before the text reaches the engine. The engine
itself has no server-side embedder bound on its ``PlanCtx``
(CONCEPT:EG-KG.compute.no-embedder-bound-op), so a raw ``~"text"`` leg always fails
there with "no server-side text embedder is bound on this query".

Exercises the REAL ``_dispatch``/``engine_query`` tool path (not just the bare
helper) with a mocked embedder + a fake wire client, per the same pattern
``tests/unit/test_engine_tools_scope_policy.py`` uses.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.core import embedding_utilities
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


def _fake_client_factory():
    """A fake ``SyncEpistemicGraphClient`` recording every call made to it."""
    calls: list[tuple[str, str, dict]] = []

    def _sub(domain: str):
        def _make(name):
            def _call(**kwargs):
                calls.append((domain, name, kwargs))
                return {"ok": True, "domain": domain, "method": name}

            return _call

        class _Sub:
            def __getattr__(self, name):
                return _make(name)

        return _Sub()

    class _Client:
        def __getattr__(self, name):
            return _sub(name)

    return _Client(), calls


class _FakeEmbedModel:
    """Minimal ``BaseEmbedding``-shaped stub: batch-embeds to a fixed-length
    deterministic vector per text (distinct texts get distinct vectors)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_text_embedding_batch(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), float(i), 0.5] for i, t in enumerate(texts)]


NON_ADMIN_ACTOR = ActorContext(
    actor_id="agent:marketing",
    actor_type=ActorType.AI_AGENT,
    roles=("marketing",),
    tenant_id="acme",
)


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


@pytest.fixture
def fake_embed_model(monkeypatch):
    model = _FakeEmbedModel()
    monkeypatch.setattr(
        embedding_utilities, "create_embedding_model", lambda: model
    )
    return model


# ── wire-first + live-path: RANK BY ~"text" gets pre-embedded ────────────────
def test_uql_rank_by_quoted_text_is_pre_embedded(monkeypatch, fake_embed_model):
    """The real ``engine_query`` tool path: a ``RANK BY ~"..."`` leg is rewritten
    to an inline literal vector before ``client.query.uql`` ever sees it."""
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]
    uql_text = 'MATCH (:Concept) |> RANK BY ~"some text" |> LIMIT 10'
    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                tool(
                    action="uql",
                    params_json=json.dumps({"text": uql_text}),
                    graph="",
                )
            )
        )
    assert out.get("ok") is True, out
    assert len(calls) == 1
    (domain, method, kwargs) = calls[0]
    assert (domain, method) == ("query", "uql")
    sent_text = kwargs["text"]
    assert "~[" in sent_text
    assert '~"' not in sent_text
    # the surrounding pipeline stages are untouched
    assert sent_text.startswith("MATCH (:Concept) |> RANK BY ~[")
    assert sent_text.endswith("|> LIMIT 10")
    # the embedder was actually invoked with the quoted text (unescaped)
    assert fake_embed_model.calls == [["some text"]]


def test_uql_rank_text_inside_fuse_branches_is_pre_embedded(
    monkeypatch, fake_embed_model
):
    """A ``~"text"`` RANK leg nested inside ``FUSE [...] [...]`` is also embedded —
    the fix isn't limited to a single top-level RANK stage."""
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]
    uql_text = 'MATCH (:Concept) |> FUSE [RANK BY ~"alpha query"] [TEXT "beta query"]'
    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            tool(
                action="uql",
                params_json=json.dumps({"text": uql_text}),
                graph="",
            )
        )
    sent_text = calls[0][2]["text"]
    assert "~[" in sent_text
    assert '~"' not in sent_text
    # TEXT's own quoted string is a DIFFERENT op (BM25 lexical) — untouched.
    assert 'TEXT "beta query"' in sent_text
    assert fake_embed_model.calls == [["alpha query"]]


# ── inline vector / reserved handle pass through untouched, no embed call ────
def test_uql_inline_vector_and_handle_untouched(monkeypatch, fake_embed_model):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]

    inline_text = "MATCH (:Concept) |> RANK BY ~[1.0, -0.5, 0.25] |> LIMIT 5"
    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            tool(action="uql", params_json=json.dumps({"text": inline_text}), graph="")
        )
    assert calls[0][2]["text"] == inline_text  # byte-for-byte unchanged
    assert fake_embed_model.calls == []  # embedder never invoked

    calls.clear()
    handle_text = "MATCH (:Concept) |> RANK BY ~myhandle |> LIMIT 5"
    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            tool(action="uql", params_json=json.dumps({"text": handle_text}), graph="")
        )
    assert calls[0][2]["text"] == handle_text  # byte-for-byte unchanged
    assert fake_embed_model.calls == []  # embedder never invoked


# ── the structured `unified` plan surface gets the same fix ──────────────────
def test_unified_plan_rank_text_query_is_pre_embedded(monkeypatch, fake_embed_model):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]
    plan = [
        {"Scan": {"label": "Concept"}},
        {"Rank": {"query": "some text"}},
        {"Limit": {"k": 10}},
    ]
    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                tool(
                    action="unified",
                    params_json=json.dumps({"plan": plan}),
                    graph="",
                )
            )
        )
    assert out.get("ok") is True, out
    sent_plan = calls[0][2]["plan"]
    rank_query = sent_plan[1]["Rank"]["query"]
    assert isinstance(rank_query, list)
    assert all(isinstance(c, float) for c in rank_query)
    assert fake_embed_model.calls == [["some text"]]


def test_unified_plan_inline_vector_rank_untouched(monkeypatch, fake_embed_model):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]
    plan = [
        {"Scan": {"label": "Concept"}},
        {"Rank": {"query": [1.0, 0.0, 0.0]}},
    ]
    with use_actor(NON_ADMIN_ACTOR):
        asyncio.run(
            tool(action="unified", params_json=json.dumps({"plan": plan}), graph="")
        )
    sent_plan = calls[0][2]["plan"]
    assert sent_plan[1]["Rank"]["query"] == [1.0, 0.0, 0.0]
    assert fake_embed_model.calls == []


# ── fails loud, never silently drops the RANK leg ─────────────────────────────
def test_uql_rank_text_embedder_unavailable_fails_loud(monkeypatch):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    def _boom():
        raise RuntimeError("no embedder configured")

    monkeypatch.setattr(embedding_utilities, "create_embedding_model", _boom)

    tool = kg_server.REGISTERED_TOOLS["engine_query"]
    uql_text = 'MATCH (:Concept) |> RANK BY ~"some text"'
    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(
                tool(action="uql", params_json=json.dumps({"text": uql_text}), graph="")
            )
        )
    assert "error" in out
    assert "embedder" in out["error"].lower()
    assert calls == []  # never reached the engine client with the raw ~"text"


# ── the bare-helper unit surface (span-finder + rewrite) ──────────────────────
def test_uql_rank_text_spans_helper_escaping():
    spans = engine_tools._uql_rank_text_spans(
        'RANK BY ~"he said \\"hi\\"" |> LIMIT 1'
    )
    assert len(spans) == 1
    _start, _end, literal = spans[0]
    assert literal == 'he said "hi"'


def test_uql_rank_text_spans_helper_no_match_for_inline_or_handle():
    assert engine_tools._uql_rank_text_spans("RANK BY ~[1.0, 2.0]") == []
    assert engine_tools._uql_rank_text_spans("RANK BY ~handle") == []
