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
from collections.abc import Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Patterns for extracting citations from agent responses
_KG_REF_PATTERN = re.compile(r"\[(?:KG|source|ref|node):\s*([^\]]+)\]", re.IGNORECASE)
# CONCEPT:AU-OS.governance.okf-cis-standard — matches both the legacy numeric id
# scheme (``KG-2.63``) and the current OKF-CIS ``<SLUG>-<PILLAR>.<domain>.<concept>``
# scheme (``AU-KG.memory.auto-similarity-memory-graph``); the pattern predated the
# OKF-CIS migration and only ever matched the legacy form.
_CONCEPT_PATTERN = re.compile(
    r"CONCEPT:("
    r"[A-Z]+-\d+(?:\.\d+)*"
    r"|[A-Za-z]+-[A-Za-z]+(?:\.[A-Za-z0-9][\w-]*)+"
    r")",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'<>]+", re.IGNORECASE)
_FILE_REF_PATTERN = re.compile(r"file:///[^\s\)\]\"'<>]+", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"(?:arXiv:\s*)?(\d{4}\.\d{4,5})", re.IGNORECASE)


def _group1_stripped(match: re.Match[str]) -> tuple[str, str]:
    """``(source_id, raw_text)`` for a pattern whose id is a stripped group 1."""
    return match.group(1).strip(), match.group(0)


def _group1_raw(match: re.Match[str]) -> tuple[str, str]:
    """``(source_id, raw_text)`` for a pattern whose id is an unstripped group 1."""
    return match.group(1), match.group(0)


def _full_match_trimmed(match: re.Match[str]) -> tuple[str, str]:
    """``(source_id, raw_text)`` for a pattern where the whole match IS the id.

    Trailing punctuation likely to be sentence structure, not part of the
    reference, is stripped; ``source_id`` and ``raw_text`` are identical.
    """
    value = match.group(0).rstrip(".,;:)")
    return value, value


# One (pattern, citation_type, row-extractor) entry per citation form. Order
# matches the original sequential scan: KG refs, concepts, URLs, files, arXiv.
_CITATION_EXTRACTORS: tuple[
    tuple[re.Pattern[str], str, Callable[[re.Match[str]], tuple[str, str]]], ...
] = (
    (_KG_REF_PATTERN, "kg_node", _group1_stripped),
    (_CONCEPT_PATTERN, "concept", _group1_stripped),
    (_URL_PATTERN, "url", _full_match_trimmed),
    (_FILE_REF_PATTERN, "file", _full_match_trimmed),
    (_ARXIV_PATTERN, "arxiv", _group1_raw),
)


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
        for pattern, citation_type, row in _CITATION_EXTRACTORS:
            for match in pattern.finditer(response_text):
                source_id, raw_text = row(match)
                if source_id in seen:
                    continue
                seen.add(source_id)
                citations.append(
                    Citation(
                        source_id=source_id,
                        citation_type=citation_type,
                        raw_text=raw_text,
                    )
                )
        return citations

    @staticmethod
    def _citation_type_counts(citations: list[Citation]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in citations:
            counts[c.citation_type] = counts.get(c.citation_type, 0) + 1
        return counts

    @staticmethod
    def _precision_recall_f1(
        cited_ids: set[str], reference_set: set[str]
    ) -> tuple[float, float, float]:
        if not reference_set:
            # No reference set — precision can't be computed meaningfully, and
            # recall is vacuously 0 (nothing to have recalled).
            return 1.0, 0.0, 0.0
        matched = cited_ids & reference_set
        precision = len(matched) / len(cited_ids) if cited_ids else 0.0
        recall = len(matched) / len(reference_set)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

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
        type_counts = self._citation_type_counts(citations)

        if not citations:
            return CitationReport(
                total_citations=0,
                uncited_evidence=sorted(reference_set),
                citation_types=type_counts,
            )

        precision, recall, f1 = self._precision_recall_f1(cited_ids, reference_set)

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
