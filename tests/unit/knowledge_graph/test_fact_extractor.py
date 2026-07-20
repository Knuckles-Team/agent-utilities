"""Unit tests for the document fact-extraction core (CONCEPT:AU-KG.enrichment.atomic-triple-extraction).

No live LLM/embedder: the stream and embed fns are injected fakes, so these
exercise the prompt-independent machinery — incremental parse, multi-round
recall, semantic dedup, persistence, and JSONL parity.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.extraction import (
    ExtractedFact,
    FactDeduper,
    extract_facts,
    facts_to_jsonl,
    parse_facts_incremental,
    persist_facts,
)


def _fact_json(subject: str, predicate: str, obj: str, conf: int = 90) -> str:
    return json.dumps(
        {
            "title": f"{subject} {predicate} {obj}",
            "description": "desc",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "evidence_span": "quote",
            "confidence": conf,
            "tags": ["t"],
        }
    )


# --------------------------------------------------------------------------- #
# normalize_key
# --------------------------------------------------------------------------- #


def test_normalize_key_merges_surface_variants() -> None:
    assert ExtractedFact.normalize_key("The Jina AI ") == "the jina ai"
    assert ExtractedFact.normalize_key('"Jina AI".') == "jina ai"
    assert ExtractedFact.normalize_key("jina  ai") == "jina ai"


# --------------------------------------------------------------------------- #
# incremental parser
# --------------------------------------------------------------------------- #


def test_parse_facts_incremental_emits_completed_objects_only() -> None:
    seen: set[int] = set()
    partial = '{"facts": [' + _fact_json("A", "rel", "B") + ', {"title": "x", "sub'
    got = parse_facts_incremental(partial, seen)
    assert len(got) == 1
    assert got[0]["subject"] == "A"
    # feeding the same buffer again yields nothing (hash dedup)
    assert parse_facts_incremental(partial, seen) == []


def test_parse_facts_incremental_completes_on_more_tokens() -> None:
    seen: set[int] = set()
    buf = "[" + _fact_json("A", "rel", "B")
    assert len(parse_facts_incremental(buf, seen)) == 1
    buf += ", " + _fact_json("C", "rel", "D")
    out = parse_facts_incremental(buf, seen)
    assert len(out) == 1 and out[0]["subject"] == "C"


# --------------------------------------------------------------------------- #
# deduper
# --------------------------------------------------------------------------- #


def _orthogonal_embedder() -> object:
    """Embed each unique dedup-text to its own basis vector → identical text is a
    perfect (1.0) cosine, different text is 0.0."""
    vocab: dict[str, int] = {}

    def _embed(text: str) -> list[float]:
        idx = vocab.setdefault(text, len(vocab))
        vec = [0.0] * 64
        vec[idx % 64] = 1.0
        return vec

    return _embed


def test_deduper_flags_identical_triples() -> None:
    d = FactDeduper(_orthogonal_embedder(), field="triple", threshold=0.9)
    f1 = ExtractedFact(subject="A", predicate="rel", object="B")
    f2 = ExtractedFact(subject="A", predicate="rel", object="B")
    f3 = ExtractedFact(subject="X", predicate="rel", object="Y")
    assert d.check(f1) == (False, 0.0)
    is_dup, sim = d.check(f2)
    assert is_dup and sim == pytest.approx(1.0)
    assert d.check(f3)[0] is False


def test_deduper_rehydrate_seeds_corpus() -> None:
    d = FactDeduper(_orthogonal_embedder(), field="triple", threshold=0.9)
    prior = [ExtractedFact(subject="A", predicate="rel", object="B")]
    d.rehydrate(prior)
    assert len(d) == 1
    again = ExtractedFact(subject="A", predicate="rel", object="B")
    assert d.check(again)[0] is True


# --------------------------------------------------------------------------- #
# extract_facts (multi-round + dedup) with an injected stream
# --------------------------------------------------------------------------- #


def _fake_stream(payloads: list[str]):
    """Return a StreamFn that yields ``payloads[round-1]`` token-by-token."""

    async def _stream(prompt: str, seed: int):
        # round index inferred from seed schedule isn't needed; emit per-call.
        idx = _stream.calls  # type: ignore[attr-defined]
        _stream.calls += 1  # type: ignore[attr-defined]
        text = payloads[min(idx, len(payloads) - 1)]
        for ch in text:
            yield ch

    _stream.calls = 0  # type: ignore[attr-defined]
    return _stream


@pytest.mark.asyncio
async def test_extract_facts_single_round_streams_facts() -> None:
    body = '{"facts": [' + _fact_json("A", "built_by", "B") + "]}"
    events = [
        e
        async for e in extract_facts(
            "doc",
            rounds=1,
            stream_fn=_fake_stream([body]),
            deduper=FactDeduper(_orthogonal_embedder()),
        )
    ]
    types = [e["type"] for e in events]
    assert types[0] == "round_start"
    assert types[-1] == "done"
    facts = [e for e in events if e["type"] == "fact"]
    assert len(facts) == 1
    assert facts[0]["fact"]["subject"] == "A"
    done = events[-1]
    assert done["unique_facts"] == 1 and done["duplicate_facts"] == 0


@pytest.mark.asyncio
async def test_extract_facts_multiround_dedups_repeats() -> None:
    r1 = '{"facts": [' + _fact_json("A", "rel", "B") + "]}"
    r2 = '{"facts": [' + _fact_json("A", "rel", "B") + "]}"  # same fact again
    deduper = FactDeduper(_orthogonal_embedder(), threshold=0.9)
    events = [
        e
        async for e in extract_facts(
            "doc", rounds=2, stream_fn=_fake_stream([r1, r2]), deduper=deduper
        )
    ]
    done = events[-1]
    assert done["total_facts"] == 2
    assert done["duplicate_facts"] == 1
    assert done["unique_facts"] == 1


# --------------------------------------------------------------------------- #
# persistence + JSONL
# --------------------------------------------------------------------------- #


class _FakeStore:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id: str, label: str = "", **props) -> None:
        self.nodes[node_id] = {"label": label, **props}

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        self.edges.append((source, target, rel_type, props))


def test_persist_facts_writes_merged_edges() -> None:
    store = _FakeStore()
    facts = [
        ExtractedFact(subject="Jina AI", predicate="built", object="v5", confidence=95),
        ExtractedFact(
            subject="jina ai", predicate="released", object="v5", confidence=80
        ),
        ExtractedFact(subject="X", predicate="rel", object="Y", is_duplicate=True),
    ]
    report = persist_facts(store, facts)
    # "Jina AI" and "jina ai" merge to one node key; v5 shared → 2 nodes total
    assert report["nodes"] == 2
    assert report["edges"] == 2  # duplicate fact skipped
    src, tgt, rel, props = store.edges[0]
    assert src == "jina ai" and rel == "built"
    assert props["confidence"] == pytest.approx(0.95)
    assert props["evidence_span"] == ""


def test_facts_to_jsonl_roundtrips() -> None:
    facts = [ExtractedFact(subject="A", predicate="rel", object="B", confidence=70)]
    line = facts_to_jsonl(facts)
    rec = json.loads(line)
    assert rec["subject"] == "A" and rec["confidence"] == 70
    assert rec["is_duplicate"] is False
