"""CONCEPT:EG-KG.storage.nonblocking-checkpoint — card-generation cost controls (batching, persistent cache,
trivial-skip, per-language prompt)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.cards import (
    CardStore,
    build_symbol_prompt,
    generate_symbol_cards,
)
from agent_utilities.knowledge_graph.enrichment.models import CodeEntity


@pytest.fixture()
def card_store(engine_graph):
    """A :class:`CardStore` on the REAL ephemeral engine tenant (CONCEPT:AU-KG.backend.cache-lives-as).

    The card cache is engine-only (no SQLite), so its persistent-store behaviour is
    proven against the actual shipped database via the conftest ``engine_graph``
    fixture (CONCEPT:AU-KG.memory.provides-real-ephemeral-one) bound through an ``EpistemicGraphBackend``.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    return CardStore(backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name))


def _sym(
    name: str, h: str, *, kind: str = "function", language: str = "java"
) -> CodeEntity:
    return CodeEntity(
        id=f"code:X.java::{name}",
        name=name,
        qualname=name,
        kind=kind,
        language=language,
        file_path="X.java",
        line=1,
        ast_hash=h,
    )


def test_per_language_prompt_uses_language_not_python():
    p = build_symbol_prompt(_sym("compute", "h", language="rust"))
    assert "rust function" in p
    assert "Python" not in p


def test_multiple_symbols_share_one_batched_llm_call():
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        # Batch prompt → JSON array, one card per symbol in order.
        return (
            '[{"name":"a","summary":"does a","responsibilities":["x"]},'
            ' {"name":"b","summary":"does b","responsibilities":["y"]},'
            ' {"name":"c","summary":"does c","responsibilities":["z"]}]'
        )

    syms = [_sym("a", "h1"), _sym("b", "h2"), _sym("c", "h3")]
    cards = generate_symbol_cards(syms, fake_llm, batch_size=12, max_workers=1)
    assert calls["n"] == 1  # ONE call for three symbols
    by = {c.name: c for c in cards}
    assert by["a"].summary == "does a"
    assert by["c"].responsibilities == ["z"]


def test_trivial_symbols_skip_the_llm():
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        return '[{"name":"real","summary":"real work","responsibilities":[]}]'

    syms = [
        _sym("getValue", "h1", kind="method"),  # trivial getter
        _sym("setName", "h2", kind="method"),  # trivial setter
        _sym("Widget", "h3", kind="constructor"),  # trivial ctor
        _sym("real", "h4", kind="function"),  # real → 1 call
    ]
    cards = generate_symbol_cards(syms, fake_llm, batch_size=12, max_workers=1)
    by = {c.name: c for c in cards}
    assert by["getValue"].summary == ""
    assert by["setName"].summary == ""
    assert by["Widget"].summary == ""
    assert by["real"].summary == "real work"
    assert calls["n"] == 1  # only the non-trivial symbol hit the LLM


def test_persistent_store_avoids_relll_on_fresh_cache(card_store):
    store = card_store
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        return '{"summary":"cached body","responsibilities":["r"]}'

    # First run populates the persistent store.
    c1 = generate_symbol_cards([_sym("f", "hZ")], fake_llm, store=store, max_workers=1)
    assert c1[0].summary == "cached body"
    assert calls["n"] == 1

    # Second run with a FRESH in-memory cache hits the persistent store → no LLM.
    c2 = generate_symbol_cards(
        [_sym("f", "hZ")], fake_llm, cache={}, store=store, max_workers=1
    )
    assert c2[0].summary == "cached body"
    assert c2[0].responsibilities == ["r"]
    assert calls["n"] == 1  # still one — served from the persistent store


def test_card_store_round_trip(card_store):
    store = card_store
    store.put_many([("h1", "sum1", ["a", "b"]), ("h2", "sum2", [])])
    got = store.get_many(["h1", "h2", "missing"])
    assert got["h1"] == ("sum1", ["a", "b"])
    assert got["h2"] == ("sum2", [])
    assert "missing" not in got
