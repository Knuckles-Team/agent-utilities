"""Cross-ingestion + research→spec distillation (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 3)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.distill import (
    distill_specs,
    gather_enhancement_candidates,
)
from agent_utilities.knowledge_graph.enrichment.extractors.document import (
    detect_doc_type,
    extract_document,
)
from agent_utilities.knowledge_graph.enrichment.models import Concept, EnrichmentEdge
from agent_utilities.knowledge_graph.enrichment.semantic import (
    find_related,
    link_concepts_to_code,
)


# ── document + concept extraction ────────────────────────────────────────────
def test_detect_doc_type():
    assert detect_doc_type("x.eml", "From: a@b\nTo: c@d\nSubject: hi") == "email"
    assert detect_doc_type("sow_acme.md", "Statement of Work for Acme") == "sow"
    assert detect_doc_type("reqs.md", "Business Requirement Document\n...") == "brd"


def test_extract_document_with_concepts():
    def fake_llm(prompt):
        return (
            '[{"name": "Retrieval Augmented Generation", "kind": "technique", '
            '"summary": "Ground LLM output in retrieved context."},'
            '{"name": "HNSW Index", "kind": "method", "summary": "ANN vector search."}]'
        )

    doc, concepts, edges = extract_document(
        "/papers/rag.pdf", "Abstract: a paper about RAG and vector search.", fake_llm
    )
    assert doc.doc_type == "paper"
    assert {c.name for c in concepts} == {
        "Retrieval Augmented Generation",
        "HNSW Index",
    }
    assert all(e.rel_type == "MENTIONS" and e.source == doc.id for e in edges)
    assert concepts[0].id.startswith("concept:")


# ── semantic cross-linking ───────────────────────────────────────────────────
def test_link_concepts_to_code_by_similarity():
    concepts = [Concept(id="concept:hnsw", name="HNSW Index", summary="ANN search")]

    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def fake_search(vec, k):
        return [
            {"id": "code:a.py::HnswIndex", "type": "Code", "_similarity": 0.81},
            {"id": "code:b.py::helper", "type": "Code", "_similarity": 0.60},
            {"id": "doc:zzz", "type": "Document", "_similarity": 0.95},  # ignored
        ]

    edges = link_concepts_to_code(
        concepts,
        fake_embed,
        fake_search,
        relates_threshold=0.55,
        realizes_threshold=0.78,
    )
    rels = {(e.target.split("::")[-1], e.rel_type) for e in edges}
    assert ("HnswIndex", "REALIZES") in rels  # 0.81 >= 0.78
    assert ("helper", "RELATES_TO") in rels  # 0.60 in [0.55, 0.78)
    assert not any(e.target == "doc:zzz" for e in edges)  # non-code filtered


def test_find_related_ranks_cross_ingestion():
    def fake_embed(texts):
        return [[1.0, 0.0]]

    def fake_search(vec, k):
        return [
            {"id": "code:x", "type": "Code", "name": "router", "_similarity": 0.9},
            {"id": "concept:y", "type": "Concept", "name": "RAG", "_similarity": 0.8},
        ]

    res = find_related("how is routing done", fake_embed, fake_search)
    assert [r["type"] for r in res] == ["Code", "Concept"]
    assert res[0]["similarity"] == 0.9


# ── distillation: research → spec ────────────────────────────────────────────
def test_gather_and_distill_specs():
    concepts = [
        Concept(
            id="concept:rag",
            name="RAG",
            summary="retrieval-augmented gen",
            source_ids=["doc:1"],
        ),
        Concept(
            id="concept:hnsw", name="HNSW", summary="ANN index", source_ids=["doc:1"]
        ),
    ]
    edges = [
        EnrichmentEdge(
            source="concept:rag",
            target="code:agents/bot/x.py::retrieve",
            rel_type="RELATES_TO",
        ),
        EnrichmentEdge(
            source="concept:rag",
            target="code:agents/bot/y.py::answer",
            rel_type="RELATES_TO",
        ),
        EnrichmentEdge(
            source="concept:hnsw", target="code:other/z.py::idx", rel_type="RELATES_TO"
        ),  # outside target codebase
    ]
    code_files = {
        "code:agents/bot/x.py::retrieve": "agents/bot/x.py",
        "code:agents/bot/y.py::answer": "agents/bot/y.py",
        "code:other/z.py::idx": "other/z.py",
    }
    cands = gather_enhancement_candidates(concepts, edges, code_files, "agents/bot")
    # RAG relates to 2 components in agents/bot; HNSW relates to 0 in-codebase.
    assert cands[0].concept_name == "RAG"
    assert cands[0].value_score == 2.0
    assert all(c.concept_name != "HNSW" for c in cands)

    def fake_llm(prompt):
        return (
            '[{"title": "Add RAG retrieval", "problem": "no grounding", '
            '"approach": "add retriever + HNSW", "value": "accuracy", '
            '"concept_names": ["RAG"]}]'
        )

    specs = distill_specs("agents/bot", cands, fake_llm, limit=3)
    assert specs[0].title == "Add RAG retrieval"
    assert specs[0].target_codebase == "agents/bot"
    assert "concept:rag" in specs[0].concept_ids
