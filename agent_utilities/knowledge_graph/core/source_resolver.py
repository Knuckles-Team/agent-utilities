#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.12 — KG Source Resolver for Comparative Analysis.

Bridges the Knowledge Graph (indexing/discovery layer) to the
comparative-analysis skill (analysis layer) by materializing stored
documents to filesystem paths with metadata enrichment.

Architecture:
    - **KGSourceResolver**: Queries the KG for matching ArticleNodes,
      KnowledgeBaseNodes, and codemap entries, then materializes their
      content to a persistent directory for analysis script consumption.
    - **ResolvedSource**: Unified representation of a KG-backed source
      with filesystem path and enrichment metadata.

This is OPTIONAL — the comparative-analysis skill works without a KG.
When a KG is available, it provides an additional discovery mechanism.

Integrates with:
    - CONCEPT:KG-2.0 (IntelligenceGraphEngine): Source discovery
    - CONCEPT:KG-2.11 (ResearchPipeline): Paper ingestion
    - comparative-analysis skill: discover_projects.py --kg-query

See docs/research-pipeline.md §CONCEPT:KG-2.12.
"""


import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Default materialization directory
DEFAULT_RESOLVE_DIR = str(Path.home() / ".scholarx" / "analysis")


class ResolvedSource(BaseModel):
    """A KG-backed source materialized to the filesystem.

    Attributes:
        source_id: KG node ID of the source.
        name: Human-readable name.
        source_type: 'research' or 'codebase'.
        file_path: Filesystem path where content was materialized.
        kg_metadata: Enrichment metadata from the KG.
        relevance_score: How relevant this source is to the query.
        authors: Author names if available.
        domains: Taxonomy domain tags.
        content_preview: First 200 chars of content.
    """

    source_id: str
    name: str
    source_type: str = "research"
    file_path: str = ""
    kg_metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    authors: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    content_preview: str = ""


class KGSourceResolver:
    """CONCEPT:KG-2.12 — Resolves KG-stored documents into filesystem-ready sources.

    Bridges the KG (indexing layer) to the comparative-analysis (analysis layer)
    by materializing stored documents to persistent paths with metadata enrichment.

    This class is designed to be OPTIONAL. If no KG engine is provided,
    all methods return empty lists gracefully.

    Args:
        engine: The IntelligenceGraphEngine to query. Optional.
        resolve_dir: Directory to materialize files into.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        resolve_dir: str = DEFAULT_RESOLVE_DIR,
    ):
        self.engine = engine
        self.resolve_dir = Path(resolve_dir)

    def is_available(self) -> bool:
        """Check if the KG is available for resolution."""
        return self.engine is not None

    def resolve_papers(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ResolvedSource]:
        """Find and materialize research papers from the KG.

        Searches ArticleNodes in the graph using keyword matching,
        then writes their content to the resolve directory.

        Args:
            query: Search query string.
            top_k: Maximum number of sources to return.

        Returns:
            List of ResolvedSource objects with materialized file paths.
        """
        if not self.engine:
            return []

        from ...models.knowledge_graph import RegistryNodeType

        candidates = self._search_nodes(
            query=query,
            node_type=RegistryNodeType.ARTICLE,
            top_k=top_k,
        )
        return self._materialize_sources(candidates, source_type="research")

    def resolve_codebases(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ResolvedSource]:
        """Find and materialize codebases indexed in the KG.

        Searches KnowledgeBaseNodes with source_type containing 'codebase'
        or 'directory'.

        Args:
            query: Search query string.
            top_k: Maximum number of sources to return.

        Returns:
            List of ResolvedSource objects.
        """
        if not self.engine:
            return []

        from ...models.knowledge_graph import RegistryNodeType

        candidates = self._search_nodes(
            query=query,
            node_type=RegistryNodeType.KNOWLEDGE_BASE,
            top_k=top_k,
            extra_filter=lambda data: (
                data.get("source_type", "") in ("directory", "codebase", "skill_graph")
            ),
        )
        return self._materialize_sources(candidates, source_type="codebase")

    def resolve_any(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[ResolvedSource]:
        """Universal resolver: papers, codebases, or knowledge bases.

        Searches across all node types and returns the most relevant.

        Args:
            query: Search query string.
            top_k: Maximum total sources to return.

        Returns:
            List of ResolvedSource objects sorted by relevance.
        """
        if not self.engine:
            return []

        papers = self.resolve_papers(query, top_k=top_k)
        codebases = self.resolve_codebases(query, top_k=top_k)

        # Merge and sort by relevance
        all_sources = papers + codebases
        all_sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return all_sources[:top_k]

    def _search_nodes(
        self,
        query: str,
        node_type: str,
        top_k: int = 5,
        extra_filter: Any = None,
    ) -> list[dict[str, Any]]:
        """Search graph nodes by keyword matching.

        Args:
            query: Search query.
            node_type: Node type to filter by.
            top_k: Max results.
            extra_filter: Optional callable(data) -> bool for additional filtering.

        Returns:
            List of node dicts with 'id', 'data', and 'score'.
        """
        if not self.engine:
            return []

        query_lower = query.lower()
        query_terms = query_lower.split()
        results: list[dict[str, Any]] = []

        for node_id, data in self.engine.graph.nodes(data=True):
            if data.get("type") != node_type:
                continue

            if extra_filter and not extra_filter(data):
                continue

            # Score based on keyword hits in name, description, content, tags
            searchable = " ".join(
                [
                    str(data.get("name", "")),
                    str(data.get("description", "")),
                    str(data.get("content", ""))[:2000],
                    " ".join(data.get("tags", []))
                    if isinstance(data.get("tags"), list)
                    else "",
                ]
            ).lower()

            score = sum(1 for term in query_terms if term in searchable)
            if score > 0:
                results.append(
                    {
                        "id": node_id,
                        "data": dict(data),
                        "score": score,
                    }
                )

        # Try hybrid search if available for better results
        if hasattr(self.engine, "hybrid_retriever") and self.engine.hybrid_retriever:
            try:
                hybrid_results = self.engine.hybrid_retriever.retrieve_hybrid(
                    query=query, context_window=top_k
                )
                for hr in hybrid_results:
                    hr_id = (
                        hr.get("id", "")
                        if isinstance(hr, dict)
                        else getattr(hr, "id", "")
                    )
                    if hr_id and hr_id not in {r["id"] for r in results}:
                        hr_data = self.engine.graph.nodes.get(hr_id, {})
                        if hr_data.get("type") == node_type:
                            hr_score = (
                                hr.get("score", 0.5) if isinstance(hr, dict) else 0.5
                            )
                            results.append(
                                {
                                    "id": hr_id,
                                    "data": dict(hr_data),
                                    "score": hr_score * 10,  # Normalize
                                }
                            )
            except Exception as e:
                logger.debug(f"Hybrid search unavailable: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _materialize_sources(
        self,
        candidates: list[dict[str, Any]],
        source_type: str,
    ) -> list[ResolvedSource]:
        """Write candidate node content to filesystem files.

        Args:
            candidates: Node search results.
            source_type: 'research' or 'codebase'.

        Returns:
            List of ResolvedSource with populated file_path.
        """
        self.resolve_dir.mkdir(parents=True, exist_ok=True)
        resolved: list[ResolvedSource] = []

        for candidate in candidates:
            node_id = candidate["id"]
            data = candidate["data"]
            score = candidate["score"]

            name = data.get("name", node_id)
            content = data.get("content", "") or data.get("description", "")

            if not content:
                continue

            # Create a safe filename
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in name[:60]
            ).strip("_")
            file_path = self.resolve_dir / f"{safe_name}.md"

            # Write content as markdown
            header = f"# {name}\n\n"
            metadata_block = ""

            authors = data.get("authors", [])
            if isinstance(authors, str):
                try:
                    import json

                    authors = json.loads(authors)
                except (json.JSONDecodeError, TypeError):
                    authors = [authors] if authors else []

            tags = data.get("tags", [])
            if isinstance(tags, str):
                try:
                    import json

                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []

            if authors or tags:
                metadata_block = "---\n"
                if authors:
                    metadata_block += f"authors: {', '.join(authors[:5])}\n"
                if tags:
                    metadata_block += f"tags: {', '.join(tags[:10])}\n"
                metadata_block += f"source_id: {node_id}\n"
                metadata_block += f"importance: {data.get('importance_score', 0.0)}\n"
                metadata_block += "---\n\n"

            full_content = metadata_block + header + content
            file_path.write_text(full_content, encoding="utf-8")

            resolved.append(
                ResolvedSource(
                    source_id=node_id,
                    name=name,
                    source_type=source_type,
                    file_path=str(file_path),
                    kg_metadata={
                        "importance_score": data.get("importance_score", 0.0),
                        "timestamp": data.get("timestamp", ""),
                        "source_type": data.get("source_type", ""),
                    },
                    relevance_score=score,
                    authors=authors if isinstance(authors, list) else [],
                    domains=tags if isinstance(tags, list) else [],
                    content_preview=content[:200],
                )
            )

        logger.info(
            "[CONCEPT:KG-2.12] Resolved %d sources for type '%s' to %s",
            len(resolved),
            source_type,
            self.resolve_dir,
        )
        return resolved

    def cleanup(self) -> int:
        """Remove all materialized files from the resolve directory.

        Returns:
            Number of files removed.
        """
        if not self.resolve_dir.exists():
            return 0

        count = 0
        for f in self.resolve_dir.glob("*.md"):
            f.unlink()
            count += 1

        logger.info("[CONCEPT:KG-2.12] Cleaned up %d materialized files", count)
        return count
