"""Tests for contextual-retrieval enrichment (CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment).

Deterministic + offline: the heuristic path (no LLM) is exercised, and the
DocumentProcessor integration is checked with the default (no embedding model).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.contextual_enrichment import (
    ContextualEnricher,
)
from agent_utilities.knowledge_graph.ontology.document_processing import (
    ChunkingConfig,
    DocumentProcessor,
)

_DOC = (
    "# Introduction\nAlpha beta gamma. This is the opening line.\n\n"
    "# Methods\nWe did several things in the methods section. " * 8
)


@pytest.mark.concept("AU-KG.enrichment.contextual-retrieval-enrichment")
def test_heuristic_context_is_deterministic_offline():
    enr = ContextualEnricher(llm_fn=None)
    chunks = ["chunk one text", "chunk two text", "chunk three text"]
    a = enr.enrich(_DOC, chunks, title="My Paper")
    b = ContextualEnricher(llm_fn=None).enrich(_DOC, chunks, title="My Paper")
    assert a == b  # deterministic
    assert all(ctx for ctx in a)  # non-empty
    assert "My Paper" in a[0]
    assert "part 1 of 3" in a[0] and "part 2 of 3" in a[1]


@pytest.mark.concept("AU-KG.enrichment.contextual-retrieval-enrichment")
def test_enricher_llm_path_used_when_available():
    calls = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return "Situated context from the model."

    enr = ContextualEnricher(llm_fn=llm)
    out = enr.enrich("doc text here", ["c1", "c2"], title="T")
    assert out == [
        "Situated context from the model.",
        "Situated context from the model.",
    ]
    assert calls  # the llm was actually invoked


@pytest.mark.concept("AU-KG.enrichment.contextual-retrieval-enrichment")
def test_enricher_degrades_when_llm_raises():
    def bad_llm(prompt: str) -> str:
        raise RuntimeError("model down")

    enr = ContextualEnricher(llm_fn=bad_llm)
    out = enr.enrich(_DOC, ["chunk a", "chunk b"], title="Paper")
    assert all(ctx for ctx in out)  # fell back to deterministic context


@pytest.mark.concept("AU-KG.enrichment.contextual-retrieval-enrichment")
def test_document_processor_off_by_default():
    proc = DocumentProcessor(None, chunking=ChunkingConfig(chunk_size=120, overlap=20))
    result = proc.process(_DOC, document_id="doc:x", title="Paper")
    assert result.chunk_count > 0
    assert all("context" not in cn for cn in result.chunk_nodes)


@pytest.mark.concept("AU-KG.enrichment.contextual-retrieval-enrichment")
def test_document_processor_contextual_on_stores_context():
    proc = DocumentProcessor(
        None,
        chunking=ChunkingConfig(chunk_size=120, overlap=20),
        contextual=True,
        enricher=ContextualEnricher(None),
    )
    result = proc.process(_DOC, document_id="doc:x", title="Paper")
    assert result.chunk_count > 0
    assert all(cn.get("context") for cn in result.chunk_nodes)
    assert all(cn.get("contextual_summary") for cn in result.chunk_nodes)
