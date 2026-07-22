"""Live-path proof for the ``DocumentSpan`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

The seam audit (``reports/seam-closure-audit-2026-07-22.md``) and
``docs/architecture/evidence_spine_convergence.md`` both named
``ExtractedFact.evidence_span`` as a verbatim substring of the source text that
was kept only as a loose narrative property on the persisted fact edge, never
located back into the document as a ``DocumentSpan`` evidence locus. This test
proves the gap is closed: the EXISTING fact-persistence entry point
(``IngestionEngine._extract_facts_into_graph``, invoked from ``_enrich_text``
on every text-bearing ingestion) now also writes a ``DocumentSpan`` evidence
locus per persisted fact, as a side effect — not a new opt-in call site.

Offline: the LLM extraction call and the media store are both faked; no real
model, no engine round-trip. Ontology grounding/direction-repair run for real
(pure, local, no network) so the persisted-edge shape is unchanged.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.extraction import fact_extractor
from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
from agent_utilities.knowledge_graph.memory import native_ingest


class _FakeBackend:
    """Fakes the ``add_node``/``add_edge`` graph-write protocol ``persist_facts`` uses."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict]] = []
        self.edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id: str, label: str = "", **props) -> None:
        self.nodes.append((node_id, {"label": label, **props}))

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **props}))


class _FakeStore:
    def __init__(self) -> None:
        self.span_calls: list[tuple[bytes, dict]] = []

    def store_document_span_evidence(self, data: bytes, **kwargs):
        self.span_calls.append((data, kwargs))
        return object()


async def _fake_extract_facts(text, *, rounds, dedup, source_file, schema, **_kw):
    yield {
        "type": "fact",
        "is_duplicate": False,
        "fact": {
            "subject": "Acme Corp",
            "predicate": "acquired",
            "object": "Widget Inc",
            "evidence_span": "Acme Corp acquired Widget Inc in 2024",
            "confidence": 90,
            "source_file": source_file,
        },
    }


@pytest.mark.asyncio
async def test_extract_facts_into_graph_writes_document_span_evidence(monkeypatch):
    monkeypatch.setattr(fact_extractor, "extract_facts", _fake_extract_facts)

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    engine = IngestionEngine(kg_engine=None, backend=_FakeBackend())
    write_target = _FakeBackend()
    text = "Intro paragraph.\nAcme Corp acquired Widget Inc in 2024.\nOutro."

    # The EXISTING fact-persistence entry point — no new call site.
    written = await engine._extract_facts_into_graph(
        "doc:1", text, "document", writer=write_target
    )

    # Fact persistence itself is unchanged.
    assert written == 1
    assert len(write_target.edges) == 1

    # AND a DocumentSpan evidence locus was written, located back into the
    # EXACT window text the fact was extracted from via the real substring
    # offset (no fabricated coordinates).
    assert len(store.span_calls) == 1
    data, kw = store.span_calls[0]
    span = "Acme Corp acquired Widget Inc in 2024"
    assert data == span.encode("utf-8")
    assert kw["document_id"] == "doc:1"
    assert kw["start"] == text.find(span)
    assert kw["end"] == kw["start"] + len(span)
    assert kw["source"] == "document"


@pytest.mark.asyncio
async def test_extract_facts_into_graph_skips_evidence_for_nonverbatim_span(
    monkeypatch,
):
    """A quoted span that doesn't literally appear in the window → no write, no raise."""

    async def _fake_extract_facts_paraphrased(
        text, *, rounds, dedup, source_file, schema, **_kw
    ):
        yield {
            "type": "fact",
            "is_duplicate": False,
            "fact": {
                "subject": "Acme Corp",
                "predicate": "acquired",
                "object": "Widget Inc",
                "evidence_span": "this text never appears verbatim in the window",
                "confidence": 90,
                "source_file": source_file,
            },
        }

    monkeypatch.setattr(
        fact_extractor, "extract_facts", _fake_extract_facts_paraphrased
    )

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    engine = IngestionEngine(kg_engine=None, backend=_FakeBackend())
    write_target = _FakeBackend()
    written = await engine._extract_facts_into_graph(
        "doc:1", "Acme Corp acquired Widget Inc.", "document", writer=write_target
    )

    assert written == 1
    assert store.span_calls == []
