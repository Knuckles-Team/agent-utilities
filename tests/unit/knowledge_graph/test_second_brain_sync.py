"""Tests for CONCEPT:AU-KG.enrichment.second-brain-note-sync — the one-call
second-brain sync composing fact_extractor + EntityClaimExtractor +
ContradictionDetector over a personal note corpus.

No live LLM/embedder/engine: the fact-extraction stream and embedder are
injected fakes (mirroring ``test_fact_extractor.py``'s own convention), and
the KG engine is a purpose-built stub (mirroring ``test_claim_flywheel.py``'s
``_FlywheelStubEngine`` / ``test_loop_controller.py``'s ``_BeliefStubEngine``)
that backs ``add_node``/``add_edge``/``query_cypher`` with one in-memory node
store, plus a minimal ``.graph`` double and ``.backend = None`` for
``EntityClaimExtractor`` compatibility (a real ``GraphComputeEngine`` needs a
live epistemic-graph engine this sandbox has none built for — see
``tests/conftest.py``'s engine-unreachable skip hook — so this stays
hermetic), plus a keyword-overlap ``search_hybrid`` fake seeded with an
"existing KG belief" for the contradiction scenario.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.extraction.fact_extractor import FactDeduper
from agent_utilities.knowledge_graph.extraction.second_brain_sync import (
    iter_note_files,
    sync_second_brain,
)
from agent_utilities.knowledge_graph.research.claim_flywheel import (
    ClaimFlywheel,
    ClaimLifecycleState,
)

pytestmark = pytest.mark.concept("AU-KG.enrichment.second-brain-note-sync")


# --------------------------------------------------------------------------- #
# Fakes — mirrors the fact_extractor / claim_flywheel test conventions.
# --------------------------------------------------------------------------- #


def _orthogonal_embedder() -> object:
    """Embed each unique dedup-text to its own basis vector (see
    ``test_fact_extractor.py``) so identical text is a perfect cosine match and
    different text is orthogonal — deterministic, no real model."""
    vocab: dict[str, int] = {}

    def _embed(text: str) -> list[float]:
        idx = vocab.setdefault(text, len(vocab))
        vec = [0.0] * 64
        vec[idx % 64] = 1.0
        return vec

    return _embed


def _fake_stream_for(fact_by_marker: dict[str, dict[str, Any]]):
    """A fake fact-extraction ``StreamFn``: yields the mapped fact once its
    marker substring appears in the prompt (the prompt always ends with the
    note's own text — see ``fact_extractor.extract_facts``), else no facts."""

    async def _stream(prompt: str, seed: int):
        for marker, fact in fact_by_marker.items():
            if marker in prompt:
                yield json.dumps({"facts": [fact]})
                return
        yield '{"facts": []}'

    return _stream


class _FakeGraph:
    """Minimal in-memory double for ``EntityClaimExtractor``'s ``.graph`` needs
    (``add_node``/``__contains__``/``nodes(data=True)``) — a real
    ``GraphComputeEngine`` requires a live epistemic-graph engine this sandbox
    has none built for (see ``tests/conftest.py``'s engine-unreachable skip
    hook, which likewise skips ``test_entity_claim_extractor.py``'s own
    engine-backed tests), so this stays hermetic and dependency-free."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, **properties: Any) -> None:
        self._nodes[node_id] = dict(properties)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self, data: bool = False) -> list[Any]:
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())


class _SecondBrainStubEngine:
    """Minimal KG engine double satisfying every interface
    ``sync_second_brain`` composes: the canonical ``add_node(id, type,
    properties=)``/``add_edge(source, target, rel_type, **props)``/
    ``query_cypher(query, params)`` triad (real ``IntelligenceGraphEngine``
    signature — see ``core/engine.py``/``EngineStoreAdapter``), PLUS
    ``.graph``/``.backend``/``.link_nodes`` for ``EntityClaimExtractor``
    compatibility, PLUS a ``search_hybrid`` fake seeded with "existing KG"
    content for the contradiction scenario.
    """

    def __init__(self, existing_claims: list[tuple[str, str]] | None = None) -> None:
        self.graph = _FakeGraph()
        self.backend = None
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []
        self.existing_claims = list(existing_claims or [])

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> dict[str, Any]:
        props = dict(properties or {})
        self.nodes[node_id] = {"id": node_id, "type": node_type, **props}
        if node_id not in self.graph:
            self.graph.add_node(node_id, type=node_type, **props)
        return self.nodes[node_id]

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        ephemeral: bool = False,
        **properties: Any,
    ) -> None:
        self.edges.append((source, target, rel_type, properties))

    def link_nodes(
        self, source: str, target: str, edge_type: Any, metadata: dict | None = None
    ) -> None:
        rel = edge_type.value if hasattr(edge_type, "value") else edge_type
        self.edges.append((source, target, rel, dict(metadata or {})))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "ClaimLifecycleEvent" in query:
            rows = [
                n for n in self.nodes.values() if n.get("type") == "ClaimLifecycleEvent"
            ]
            cid = params.get("id")
            if cid is not None:
                rows = [r for r in rows if r.get("claim_id") == cid]
            rows.sort(key=lambda r: str(r.get("timestamp") or ""))
            return rows
        node_id = params.get("id")
        if node_id is not None:
            return [{"id": node_id}] if node_id in self.nodes else []
        return []

    def search_hybrid(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, str, str]] = []
        for cid, text in self.existing_claims:
            overlap = len(q_tokens & set(re.findall(r"[a-z0-9]+", text.lower())))
            if overlap:
                scored.append((overlap, cid, text))
        scored.sort(key=lambda t: -t[0])
        return [{"id": cid, "description": text} for _, cid, text in scored[:top_k]]


# --------------------------------------------------------------------------- #
# Fixture note corpus — 4 markdown notes, one contradicting a stubbed
# EXISTING KG claim seeded into ``search_hybrid``.
# --------------------------------------------------------------------------- #

_EXISTING_BELIEF_ID = "claim:existing-caching-belief"
_EXISTING_BELIEF_TEXT = "the caching layer clearly improves database performance"

_NOTE_TEXT = {
    "note-caching.md": (
        "MARKER-CACHING-OK\n"
        "The team must validate that the new caching layer clearly improves "
        "database performance under sustained load."
    ),
    "note-conflict.md": (
        "MARKER-CACHING-CONFLICT\n"
        "We recommend documenting that the caching layer clearly degrades "
        "database performance under peak load."
    ),
    "note-unrelated.md": (
        "MARKER-UNRELATED\n"
        "We suggest scheduling the backup job every night at midnight."
    ),
    "note-citation.md": (
        "MARKER-CITATION\n"
        "See [[Vector Search]] for details. We propose adopting vector "
        "search for semantic retrieval across the corpus."
    ),
}

_FACT_MAP = {
    "MARKER-CACHING-OK": {
        "title": "Caching layer improves DB performance",
        "description": "desc",
        "subject": "caching layer",
        "predicate": "improves",
        "object": "database performance",
        "evidence_span": (
            "the new caching layer clearly improves database performance "
            "under sustained load"
        ),
        "confidence": 85,
        "tags": ["perf"],
    },
    "MARKER-CACHING-CONFLICT": {
        "title": "Caching layer degrades DB performance",
        "description": "desc",
        "subject": "caching layer",
        "predicate": "degrades",
        "object": "database performance",
        "evidence_span": (
            "the caching layer clearly degrades database performance under peak load"
        ),
        "confidence": 80,
        "tags": ["perf"],
    },
    "MARKER-UNRELATED": {
        "title": "Backup job scheduled nightly",
        "description": "desc",
        "subject": "backup job",
        "predicate": "scheduled_at",
        "object": "midnight",
        "evidence_span": "scheduling the backup job every night at midnight",
        "confidence": 70,
        "tags": ["ops"],
    },
    "MARKER-CITATION": {
        "title": "Vector search enables semantic retrieval",
        "description": "desc",
        "subject": "vector search",
        "predicate": "enables",
        "object": "semantic retrieval",
        "evidence_span": "adopting vector search for semantic retrieval across the corpus",
        "confidence": 75,
        "tags": ["retrieval"],
    },
}


def _write_notes(root: Path) -> Path:
    notes_dir = root / "notes"
    notes_dir.mkdir()
    for name, text in _NOTE_TEXT.items():
        (notes_dir / name).write_text(text, encoding="utf-8")
    return notes_dir


def _new_engine() -> _SecondBrainStubEngine:
    return _SecondBrainStubEngine(
        existing_claims=[(_EXISTING_BELIEF_ID, _EXISTING_BELIEF_TEXT)]
    )


async def _run_sync(engine: _SecondBrainStubEngine, notes_dir: Path):
    return await sync_second_brain(
        engine,
        str(notes_dir),
        corpus_name="test-second-brain",
        fact_stream_fn=_fake_stream_for(_FACT_MAP),
        fact_deduper=FactDeduper(_orthogonal_embedder()),
    )


# --------------------------------------------------------------------------- #
# iter_note_files
# --------------------------------------------------------------------------- #


def test_iter_note_files_finds_markdown_notes(tmp_path: Path) -> None:
    notes_dir = _write_notes(tmp_path)
    found = iter_note_files(str(notes_dir))
    assert [p.name for p in found] == sorted(_NOTE_TEXT)


def test_iter_note_files_since_filters_unchanged(tmp_path: Path) -> None:
    notes_dir = _write_notes(tmp_path)
    future = max(p.stat().st_mtime for p in notes_dir.iterdir()) + 3600
    assert iter_note_files(str(notes_dir), since=future) == []


def test_iter_note_files_missing_path_returns_empty(tmp_path: Path) -> None:
    assert iter_note_files(str(tmp_path / "nope")) == []


# --------------------------------------------------------------------------- #
# sync_second_brain — facts with evidence spans
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sync_extracts_facts_with_evidence_citing_the_source_note(
    tmp_path: Path,
) -> None:
    notes_dir = _write_notes(tmp_path)
    engine = _new_engine()
    result = await _run_sync(engine, notes_dir)

    assert result.notes_seen == 4
    assert result.notes_synced == 4
    assert result.facts == len(_FACT_MAP)  # one fact per marker note, none dropped

    # Every persisted fact edge carries an evidence_span that is a verbatim
    # substring of ITS OWN source note (not a different note's text).
    fact_edges = [e for e in engine.edges if "evidence_span" in e[3]]
    assert len(fact_edges) == len(_FACT_MAP)
    for _source, _target, _rel, props in fact_edges:
        span = props["evidence_span"]
        source_file = Path(props["source_file"])
        note_text = _NOTE_TEXT[source_file.name]
        assert span in note_text
        assert props["confidence"] > 0  # 0..1 aggregate, not the raw 0..100 input


# --------------------------------------------------------------------------- #
# sync_second_brain — claims proposed into the governed ClaimFlywheel
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sync_proposes_claims_into_governed_flywheel_not_silently_accepted(
    tmp_path: Path,
) -> None:
    notes_dir = _write_notes(tmp_path)
    engine = _new_engine()
    result = await _run_sync(engine, notes_dir)

    assert result.claims >= 4
    assert len(result.claims_proposed) == result.claims

    flywheel = ClaimFlywheel(engine)
    for claim_id in result.claims_proposed:
        assert flywheel.current_state(claim_id) == ClaimLifecycleState.PROPOSED
        history = flywheel.history(claim_id)
        assert len(history) == 1
        assert history[0]["to_state"] == "proposed"

    # Claims are real, persisted ClaimNodes — not just flywheel audit rows.
    claim_nodes = [
        n
        for _, n in engine.graph.nodes(data=True)
        if str(n.get("type", "")).lower() == "claim"
    ]
    assert len(claim_nodes) == result.claims


# --------------------------------------------------------------------------- #
# sync_second_brain — contradiction surfaces as exactly one BeliefRevisionProposal
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sync_contradiction_produces_exactly_one_reviewable_proposal(
    tmp_path: Path,
) -> None:
    notes_dir = _write_notes(tmp_path)
    engine = _new_engine()
    result = await _run_sync(engine, notes_dir)

    assert len(result.contradictions) == 1
    finding = result.contradictions[0]
    assert finding["conflict_id"] == _EXISTING_BELIEF_ID
    assert finding["severity"] in {"high", "medium", "low"}

    proposals = [
        n for n in engine.nodes.values() if n.get("type") == "BeliefRevisionProposal"
    ]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["status"] == "proposal"
    assert proposal["new_contradicted_by_node_ids"] == [_EXISTING_BELIEF_ID]
    assert "reasoning_trace" in proposal and proposal["reasoning_trace"]
    # Propose-only: nothing else in the flywheel/graph was mutated by the scan.
    assert not hasattr(engine, "update_node")


# --------------------------------------------------------------------------- #
# sync_second_brain — idempotent re-sync (content-hash short-circuit)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resync_is_idempotent_no_duplicate_claims_facts_or_proposals(
    tmp_path: Path,
) -> None:
    notes_dir = _write_notes(tmp_path)
    engine = _new_engine()

    first = await _run_sync(engine, notes_dir)
    assert first.notes_synced == 4
    facts_after_first = len(engine.edges)
    claims_after_first = sum(
        1
        for _, n in engine.graph.nodes(data=True)
        if str(n.get("type", "")).lower() == "claim"
    )
    proposals_after_first = sum(
        1 for n in engine.nodes.values() if n.get("type") == "BeliefRevisionProposal"
    )
    assert claims_after_first == first.claims
    assert proposals_after_first == 1

    second = await _run_sync(engine, notes_dir)

    assert second.notes_seen == 4
    assert second.notes_synced == 0
    assert second.notes_skipped_unchanged == 4
    assert second.facts == 0
    assert second.claims == 0
    assert second.contradictions == []

    # Nothing grew: same fact-edge count, same claim-node count, same single
    # proposal — a full re-sync of an unchanged corpus is a true no-op.
    assert len(engine.edges) == facts_after_first
    claims_after_second = sum(
        1
        for _, n in engine.graph.nodes(data=True)
        if str(n.get("type", "")).lower() == "claim"
    )
    proposals_after_second = sum(
        1 for n in engine.nodes.values() if n.get("type") == "BeliefRevisionProposal"
    )
    assert claims_after_second == claims_after_first
    assert proposals_after_second == 1


@pytest.mark.asyncio
async def test_sync_links_notes_to_one_personal_corpus_node(tmp_path: Path) -> None:
    notes_dir = _write_notes(tmp_path)
    engine = _new_engine()
    result = await _run_sync(engine, notes_dir)

    corpus_edges = [e for e in engine.edges if e[2] == "PART_OF"]
    assert len(corpus_edges) == 4
    assert all(e[1] == result.corpus_id for e in corpus_edges)
    assert engine.nodes[result.corpus_id]["type"] == "PersonalCorpus"


# --------------------------------------------------------------------------- #
# Wire-First live path — the real graph_ingest MCP action actually invokes
# sync_second_brain (an empty notes dir so this never touches the real
# fact-extraction model/embedder: notes_seen == 0 short-circuits before either
# is reached — see sync_second_brain's own early return).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_graph_ingest_sync_second_brain_action_live_path(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_utilities.mcp import kg_server

    empty_notes_dir = tmp_path / "empty-notes"
    empty_notes_dir.mkdir()
    engine = _new_engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool(
        "graph_ingest",
        action="sync_second_brain",
        target_path=str(empty_notes_dir),
        corpus_name="live-path-corpus",
    )
    payload = json.loads(res)
    assert payload["notes_seen"] == 0
    assert payload["corpus_id"] == "corpus:live-path-corpus"
    assert engine.nodes[payload["corpus_id"]]["type"] == "PersonalCorpus"


@pytest.mark.asyncio
async def test_graph_ingest_sync_second_brain_requires_target_path(
    monkeypatch,
) -> None:
    from agent_utilities.mcp import kg_server

    engine = _new_engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool(
        "graph_ingest", action="sync_second_brain", target_path=""
    )
    payload = json.loads(res)
    assert "error" in payload
