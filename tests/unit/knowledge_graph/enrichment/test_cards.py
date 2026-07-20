"""Capability cards: LLM summaries cached by ast_hash (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 2)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.cards import generate_symbol_cards
from agent_utilities.knowledge_graph.enrichment.models import CodeEntity


def _fn(name, ast_hash):
    return CodeEntity(
        id=f"code:m.py::{name}",
        name=name,
        qualname=name,
        kind="function",
        file_path="m.py",
        line=1,
        ast_hash=ast_hash,
        patterns=["Factory"],
    )


def _ctor(name, ast_hash):
    e = _fn(name, ast_hash)
    e.kind = "constructor"  # trivial → no LLM, permanent-empty 'skip'
    return e


def test_card_status_ok_skip_failed():
    """CONCEPT:AU-KG.enrichment.card-attempt-status — the backfill must tell a PERMANENT empty (trivial)
    from a TRANSIENT LLM failure, so it stops re-trying trivials but keeps retrying failures."""

    def good_llm(prompt: str) -> str:
        return '{"summary": "Does a thing.", "responsibilities": ["x"]}'

    def boom_llm(prompt: str) -> str:
        raise RuntimeError("vLLM 502")

    # ok: real summary landed.
    ok = generate_symbol_cards([_fn("do_thing", "hok")], good_llm, {})
    assert ok[0].status == "ok" and ok[0].summary

    # skip: a trivial symbol gets an empty card with NO LLM call → permanent, never retry.
    calls = {"n": 0}

    def counting_llm(prompt: str) -> str:
        calls["n"] += 1
        return '{"summary": "", "responsibilities": []}'

    skipped = generate_symbol_cards([_ctor("__init__", "hsk")], counting_llm, {})
    assert skipped[0].status == "skip" and skipped[0].summary == ""
    assert calls["n"] == 0  # trivial → no LLM round-trip

    # failed: an LLM transport error is transient → status 'failed' (retry + trip breaker).
    failed = generate_symbol_cards([_fn("hard_one", "hfa")], boom_llm, {})
    assert failed[0].status == "failed" and failed[0].summary == ""


def test_cards_generated_and_cached_by_ast_hash():
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        return '{"summary": "Builds a widget.", "responsibilities": ["construct", "validate"]}'

    cache: dict = {}
    cards = generate_symbol_cards([_fn("create_widget", "h1")], fake_llm, cache)
    assert cards[0].summary == "Builds a widget."
    assert cards[0].responsibilities == ["construct", "validate"]
    assert calls["n"] == 1

    # Same ast_hash (even different id) → served from cache, no new LLM call.
    again = generate_symbol_cards([_fn("create_widget", "h1")], fake_llm, cache)
    assert calls["n"] == 1
    assert again[0].summary == "Builds a widget."


def test_card_parse_degrades_to_raw_text():
    def noisy_llm(_prompt: str) -> str:
        return "no json here, just prose about the function"

    cards = generate_symbol_cards([_fn("f", "h2")], noisy_llm, {})
    assert "prose about the function" in cards[0].summary
    assert cards[0].responsibilities == []
