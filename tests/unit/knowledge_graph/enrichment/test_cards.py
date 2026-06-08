"""Capability cards: LLM summaries cached by ast_hash (CONCEPT:KG-2.8 Phase 2)."""

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
