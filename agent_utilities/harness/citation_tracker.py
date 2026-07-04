#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-AHE.harness.citation-quality-tracking — Citation Quality Tracking.

Measures citation quality in agent responses by extracting and validating
references to KG nodes and external URLs. Inspired by BrowseComp-Plus
(arXiv:2508.06600), which reports citation precision/recall as separate
metrics proving agents with better retrievers cite more accurately.

Tracks both KG-sourced citations and external URLs.

See docs/pillars/3_agentic_harness_engineering.md
"""

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Patterns for extracting citations from agent responses
_KG_REF_PATTERN = re.compile(r"\[(?:KG|source|ref|node):\s*([^\]]+)\]", re.IGNORECASE)
_CONCEPT_PATTERN = re.compile(r"CONCEPT:([A-Z]+-\d+(?:\.\d+)*)", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'<>]+", re.IGNORECASE)
_FILE_REF_PATTERN = re.compile(r"file:///[^\s\)\]\"'<>]+", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"(?:arXiv:\s*)?(\d{4}\.\d{4,5})", re.IGNORECASE)


class Citation(BaseModel):
    """A single citation extracted from agent output."""

    source_id: str = Field(description="Normalized source identifier")
    citation_type: str = Field(
        description="Type: 'kg_node', 'concept', 'url', 'file', 'arxiv'"
    )
    raw_text: str = Field(description="Original citation text as found")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence",
    )


class CitationReport(BaseModel):
    """Quality metrics for citations in an agent response.

    CONCEPT:AU-AHE.harness.citation-quality-tracking — Citation Quality Tracking (BrowseComp-Plus)
    """

    total_citations: int = Field(default=0)
    precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of citations that reference actual retrieved documents",
    )
    recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of retrieved documents that were cited",
    )
    f1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of precision and recall",
    )
    hallucinated_citations: list[str] = Field(
        default_factory=list,
        description="Citations that don't match any retrieved document",
    )
    uncited_evidence: list[str] = Field(
        default_factory=list,
        description="Retrieved documents that were not cited",
    )
    citation_types: dict[str, int] = Field(
        default_factory=dict,
        description="Count of citations by type",
    )


class CitationTracker:
    """Extracts and evaluates citations in agent responses.

    CONCEPT:AU-AHE.harness.citation-quality-tracking — Citation Quality Tracking

    Supports both KG-sourced references (``[KG:node-id]``, ``CONCEPT:X``)
    and external URLs (``https://...``, ``file:///...``, arXiv IDs).

    Usage::

        tracker = CitationTracker()
        citations = tracker.extract_citations(agent_response_text)
        report = tracker.evaluate_citations(
            citations,
            retrieved_doc_ids={"doc-1", "doc-2"},
            gold_doc_ids={"doc-1"},
        )
        print(f"Precision: {report.precision:.2f}")
    """

    def extract_citations(self, response_text: str) -> list[Citation]:
        """Extract all citations from agent response text.

        Identifies:
        - KG node references: ``[KG:node-id]``, ``[source:node-id]``
        - Concept references: ``CONCEPT:AU-KG.memory.auto-similarity-memory-graph``
        - External URLs: ``https://...``
        - File references: ``file:///...``
        - arXiv IDs: ``arXiv:2508.06600`` or ``2508.06600``

        Args:
            response_text: The full agent response text.

        Returns:
            List of extracted Citation objects.
        """
        citations: list[Citation] = []
        seen: set[str] = set()

        # KG node references
        for match in _KG_REF_PATTERN.finditer(response_text):
            source_id = match.group(1).strip()
            if source_id not in seen:
                seen.add(source_id)
                citations.append(
                    Citation(
                        source_id=source_id,
                        citation_type="kg_node",
                        raw_text=match.group(0),
                    )
                )

        # Concept references
        for match in _CONCEPT_PATTERN.finditer(response_text):
            source_id = match.group(1).strip()
            if source_id not in seen:
                seen.add(source_id)
                citations.append(
                    Citation(
                        source_id=source_id,
                        citation_type="concept",
                        raw_text=match.group(0),
                    )
                )

        # URLs
        for match in _URL_PATTERN.finditer(response_text):
            url = match.group(0).rstrip(".,;:)")
            if url not in seen:
                seen.add(url)
                citations.append(
                    Citation(
                        source_id=url,
                        citation_type="url",
                        raw_text=url,
                    )
                )

        # File references
        for match in _FILE_REF_PATTERN.finditer(response_text):
            path = match.group(0).rstrip(".,;:)")
            if path not in seen:
                seen.add(path)
                citations.append(
                    Citation(
                        source_id=path,
                        citation_type="file",
                        raw_text=path,
                    )
                )

        # arXiv IDs
        for match in _ARXIV_PATTERN.finditer(response_text):
            arxiv_id = match.group(1)
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                citations.append(
                    Citation(
                        source_id=arxiv_id,
                        citation_type="arxiv",
                        raw_text=match.group(0),
                    )
                )

        return citations

    def evaluate_citations(
        self,
        citations: list[Citation],
        retrieved_doc_ids: set[str] | None = None,
        gold_doc_ids: set[str] | None = None,
    ) -> CitationReport:
        """Evaluate citation quality against retrieval and gold sets.

        Args:
            citations: Extracted citations from agent response.
            retrieved_doc_ids: IDs of documents that were actually retrieved.
            gold_doc_ids: IDs of documents known to contain the answer.

        Returns:
            CitationReport with precision, recall, F1, and diagnostics.
        """
        retrieved = retrieved_doc_ids or set()
        gold = gold_doc_ids or set()
        reference_set = retrieved | gold

        cited_ids = {c.source_id for c in citations}

        # Type distribution
        type_counts: dict[str, int] = {}
        for c in citations:
            type_counts[c.citation_type] = type_counts.get(c.citation_type, 0) + 1

        if not citations:
            return CitationReport(
                total_citations=0,
                uncited_evidence=sorted(reference_set),
                citation_types=type_counts,
            )

        # Precision: fraction of citations that match retrieved/gold docs
        if reference_set:
            matched = cited_ids & reference_set
            precision = len(matched) / len(cited_ids) if cited_ids else 0.0
        else:
            # No reference set — can't compute precision meaningfully
            precision = 1.0

        # Recall: fraction of retrieved/gold docs that were cited
        if reference_set:
            matched = cited_ids & reference_set
            recall = len(matched) / len(reference_set)
        else:
            recall = 0.0

        # F1
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Diagnostics
        hallucinated = sorted(cited_ids - reference_set) if reference_set else []
        uncited = sorted(reference_set - cited_ids)

        return CitationReport(
            total_citations=len(citations),
            precision=precision,
            recall=recall,
            f1=f1,
            hallucinated_citations=hallucinated,
            uncited_evidence=uncited,
            citation_types=type_counts,
        )
