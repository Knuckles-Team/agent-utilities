"""Tests for research-paper link extraction + roundup detection (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.ingestion.paper_links import (
    extract_paper_links,
    is_research_roundup,
)

ROUNDUP = """
# This week's papers

- [HarnessBridge](https://arxiv.org/abs/2603.09022) is neat
- FORT-Searcher: see https://arxiv.org/pdf/2604.12345v2
- MaxProof (arxiv 2605.00001) scales proofs
- A bio paper https://doi.org/10.1101/2026.01.02.123456
- Direct PDF: https://example.org/report.pdf
"""


def test_extracts_arxiv_doi_pdf():
    refs = extract_paper_links(ROUNDUP)
    kinds = {r.kind for r in refs}
    idents = {r.ident for r in refs}
    assert "arxiv" in kinds and "doi" in kinds and "pdf" in kinds
    assert "2603.09022" in idents  # from /abs/ url
    assert "2604.12345v2" in idents  # from /pdf/ url
    assert "2605.00001" in idents  # from bare "arxiv 2605.00001"
    assert any(i.startswith("10.1101/") for i in idents)
    assert "https://example.org/report.pdf" in idents


def test_arxiv_pdf_not_double_counted():
    # an arxiv /pdf/ link must not also appear as a generic .pdf ref
    refs = extract_paper_links("https://arxiv.org/pdf/2601.00001")
    assert len(refs) == 1 and refs[0].kind == "arxiv"


def test_roundup_threshold():
    refs = extract_paper_links(ROUNDUP)
    assert is_research_roundup(refs)  # 3 arxiv + 1 doi ≥ 3
    # a page with one incidental pdf is not a roundup
    few = extract_paper_links("see https://example.org/x.pdf and nothing else")
    assert not is_research_roundup(few)


def test_dedup_same_paper():
    text = "https://arxiv.org/abs/2603.09022 and again https://arxiv.org/abs/2603.09022"
    refs = extract_paper_links(text)
    assert len([r for r in refs if r.ident == "2603.09022"]) == 1
