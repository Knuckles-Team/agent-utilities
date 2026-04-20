#!/usr/bin/python
# coding: utf-8
"""Knowledge Base Pydantic Models.

Defines type-safe Pydantic models used throughout the KB system:
- Structured extraction results (used by Pydantic AI agents)
- KB metadata and status models
- Document chunk models for pipeline ingestion
- Health report models for KB linting
"""

from __future__ import annotations

from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Document Parsing Models
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A single parsed chunk from a raw source document."""

    content: str
    source_path: str
    source_type: str  # md, pdf, docx, epub, txt, html, url
    chunk_index: int
    content_hash: str
    word_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedSource(BaseModel):
    """A fully parsed raw source with all its chunks."""

    name: str
    file_path: str
    source_type: str
    content_hash: str
    file_size: int
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured Extraction Models (used as Pydantic AI result_type)
# ---------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """An atomic fact extracted from source documents."""

    content: str = Field(
        description="The factual statement, written as a complete sentence"
    )
    certainty: float = Field(
        ge=0.0, le=1.0, description="Confidence that this fact is accurate (0.0–1.0)"
    )
    source_snippet: str = Field(
        description="Direct quote or close paraphrase from the source that supports this fact"
    )


class ExtractedArticle(BaseModel):
    """A compiled wiki article produced by LLM structured extraction.

    This is the result_type for the Pydantic AI extraction agent, ensuring
    all output is validated before it enters the knowledge graph.
    """

    title: str = Field(description="Clear, concise title for this article")
    summary: str = Field(description="2-3 sentence summary covering the core concept")
    content: str = Field(
        description="Full compiled markdown article with sections, examples, and links"
    )
    concepts: List[str] = Field(
        description="Key concept names this article covers (max 10)", max_length=10
    )
    facts: List[ExtractedFact] = Field(
        description="Atomic facts extracted from the source material",
        default_factory=list,
    )
    backlinks: List[str] = Field(
        description="Titles of other articles in this knowledge base that are closely related",
        default_factory=list,
    )
    tags: List[str] = Field(
        description="Keyword tags for discovery (lowercase, hyphen-separated)",
        default_factory=list,
    )


class ExtractedKBIndex(BaseModel):
    """A generated index document summarizing all articles in a KB."""

    overview: str = Field(description="2-3 paragraph overview of the knowledge base")
    article_summaries: List[Dict[str, str]] = Field(
        description="List of {title, one_liner} for each article"
    )
    key_concepts: List[str] = Field(
        description="Top-level concepts across all articles"
    )
    suggested_queries: List[str] = Field(
        description="Example questions this KB can answer (for agent discoverability)"
    )


# ---------------------------------------------------------------------------
# KB Metadata & Status Models
# ---------------------------------------------------------------------------


class KnowledgeBaseMetadata(BaseModel):
    """Metadata about a knowledge base instance."""

    id: str
    name: str
    topic: str
    description: str
    source_type: Literal["skill_graph", "directory", "url", "mixed"]
    article_count: int = 0
    source_count: int = 0
    status: Literal["ingesting", "ready", "updating", "error", "archived"] = "ingesting"
    importance_score: float = 0.5
    timestamp: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBSearchResult(BaseModel):
    """A single result from a KB hybrid search."""

    article_id: str
    article_title: str
    kb_id: str
    kb_name: str
    excerpt: str
    score: float
    result_type: Literal["article", "fact", "concept"]


class KBSummary(BaseModel):
    """Lightweight summary of a KB for agent discovery (list_knowledge_bases)."""

    id: str
    name: str
    topic: str
    description: str
    article_count: int
    source_count: int
    status: str
    suggested_queries: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health Check Models
# ---------------------------------------------------------------------------


class KBIssue(BaseModel):
    """A single issue found during KB health check linting."""

    issue_type: Literal[
        "contradiction", "missing_data", "stale", "orphan", "inconsistency"
    ]
    description: str
    affected_node_id: str
    severity: Literal["low", "medium", "high"]
    suggested_action: str = ""


class KBHealthReport(BaseModel):
    """Full health check report for a knowledge base.

    This is the result_type for the health-check Pydantic AI agent.
    """

    kb_id: str
    kb_name: str
    issues: List[KBIssue] = Field(default_factory=list)
    suggested_articles: List[str] = Field(
        description="New article titles the LLM suggests to fill gaps",
        default_factory=list,
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0, description="Overall KB consistency score (1.0 = no issues)"
    )
    summary: str = Field(description="Human-readable summary of findings")


# ---------------------------------------------------------------------------
# Archive Models
# ---------------------------------------------------------------------------


class KBArchiveResult(BaseModel):
    """Result of a KB archive/compression operation."""

    kb_id: str
    articles_compressed: int  # full content → summary only
    nodes_pruned: int
    bytes_saved: int
    archive_timestamp: str
