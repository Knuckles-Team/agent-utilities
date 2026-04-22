#!/usr/bin/python
"""KB Ingestion Engine.

Orchestrates the full ingestion pipeline:
  parse → extract (Pydantic AI) → embed → write to graph backend → update index

Designed for incremental operation: hash-based deduplication means only
changed/new files are re-processed, saving both I/O and LLM tokens.
"""

import hashlib
import logging
import time
import uuid
from pathlib import Path

import networkx as nx

from ...models.knowledge_base import (
    KBArchiveResult,
    KBHealthReport,
    KnowledgeBaseMetadata,
    ParsedSource,
)
from ...models.knowledge_graph import (
    ArticleNode,
    KBConceptNode,
    KBFactNode,
    KBIndexNode,
    KnowledgeBaseNode,
    RawSourceNode,
    RegistryEdgeType,
    RegistryNodeType,
)
from ..backends.base import GraphBackend
from .extractor import KBExtractor
from .parser import KBDocumentParser

logger = logging.getLogger(__name__)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _kb_id(name: str) -> str:
    return f"kb:{name.lower().replace(' ', '-').replace('_', '-')}"


def _article_id(kb_id: str, title: str) -> str:
    slug = title.lower().replace(" ", "-").replace("/", "-")[:60]
    return f"article:{kb_id}:{slug}"


def _source_id(file_path: str) -> str:
    h = hashlib.sha256(file_path.encode()).hexdigest()[:12]
    return f"raw:{h}"


class KBIngestionEngine:
    """Orchestrates KB document parsing, LLM extraction, and graph ingestion.

    Usage:
        engine = KBIngestionEngine(graph=nx_graph, backend=backend)
        await engine.ingest_skill_graph("/path/to/pydantic-ai-docs")
        await engine.ingest_directory("/path/to/papers", kb_name="research")
        await engine.ingest_url("https://example.com/article", kb_name="web")
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        backend: GraphBackend | None = None,
        extractor: KBExtractor | None = None,
        chunk_size: int = 1024,
    ):
        self.graph = graph
        self.backend = backend
        self.parser = KBDocumentParser(chunk_size=chunk_size)
        self.extractor = extractor or KBExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_skill_graph(
        self, graph_path: str | Path, force: bool = False
    ) -> KnowledgeBaseMetadata:
        """Ingest a skill-graph directory into the knowledge graph.

        Reads SKILL.md frontmatter for KB metadata, then ingests all files
        in the reference/ subdirectory.

        Args:
            graph_path: Path to the skill-graph directory (must contain SKILL.md).
            force: If True, re-ingest even if content hash is unchanged.

        Returns:
            KnowledgeBaseMetadata describing the resulting KB.
        """
        graph_path = Path(graph_path)
        meta = self.parser.read_skill_graph_metadata(graph_path)
        kb_name = meta.get("name", graph_path.name)
        topic = meta.get("description", kb_name)
        tags = meta.get("tags", [])

        logger.info(f"Ingesting skill-graph: {kb_name}")
        sources = self.parser.parse_skill_graph(graph_path)

        return await self._ingest_sources(
            kb_name=kb_name,
            topic=topic,
            source_type="skill_graph",
            sources=sources,
            description=topic,
            force=force,
            extra_metadata={"tags": tags, "source_url": meta.get("source_url", "")},
        )

    async def ingest_directory(
        self,
        path: str | Path,
        kb_name: str | None = None,
        topic: str | None = None,
        recursive: bool = True,
        force: bool = False,
    ) -> KnowledgeBaseMetadata:
        """Ingest a directory of documents into a knowledge base.

        Args:
            path: Directory path to ingest.
            kb_name: Name for the KB (defaults to directory name).
            topic: Topic description (defaults to kb_name).
            recursive: Whether to scan subdirectories.
            force: If True, re-ingest unchanged files.
        """
        path = Path(path)
        if not kb_name:
            kb_name = path.name
        if not topic:
            topic = kb_name

        sources = self.parser.parse_directory(path, recursive=recursive)
        return await self._ingest_sources(
            kb_name=kb_name,
            topic=topic,
            source_type="directory",
            sources=sources,
            description=f"Documents from {path}",
            force=force,
        )

    async def ingest_url(
        self,
        url: str,
        kb_name: str,
        topic: str | None = None,
        force: bool = False,
    ) -> KnowledgeBaseMetadata:
        """Ingest a web URL into a knowledge base.

        Args:
            url: URL to fetch and ingest.
            kb_name: Name for the KB.
            topic: Topic description.
            force: If True, re-fetch even if previously ingested.
        """
        source = self.parser.parse_url(url, kb_name)
        if not source:
            raise RuntimeError(f"Failed to fetch/parse URL: {url}")

        return await self._ingest_sources(
            kb_name=kb_name,
            topic=topic or kb_name,
            source_type="url",
            sources=[source],
            description=f"Web content from {url}",
            force=force,
        )

    async def update_kb(self, kb_id: str) -> KnowledgeBaseMetadata | None:
        """Re-ingest any changed source files for an existing KB.

        Uses stored content hashes to detect changes — only changed or new
        files are re-processed by the LLM, making updates cheap.

        Args:
            kb_id: The KB node ID (e.g., "kb:pydantic-ai-docs").
        """
        if kb_id not in self.graph.nodes:
            logger.error(f"KB not found in graph: {kb_id}")
            return None

        kb_data = self.graph.nodes[kb_id]
        logger.info(f"Updating KB: {kb_data.get('name', kb_id)}")

        # Find all RawSource nodes linked to this KB
        sources_to_check = [
            n
            for n in self.graph.predecessors(kb_id)
            if self.graph.nodes[n].get("type") == RegistryNodeType.RAW_SOURCE
        ]

        updated = 0
        for source_id in sources_to_check:
            source_data = self.graph.nodes[source_id]
            file_path = source_data.get("file_path", "")
            old_hash = source_data.get("content_hash", "")

            if not file_path or not Path(file_path).exists():
                continue

            try:
                new_source = self.parser.parse_file(file_path)
                if new_source and new_source.content_hash != old_hash:
                    logger.info(f"Re-ingesting changed source: {file_path}")
                    # Update the source node
                    self.graph.nodes[source_id]["content_hash"] = (
                        new_source.content_hash
                    )
                    # Re-extract any articles compiled from this source
                    await self._process_source(
                        new_source, kb_id, kb_data.get("topic", ""), force=True
                    )
                    updated += 1
            except Exception as e:
                logger.warning(f"Failed to update source {file_path}: {e}")

        # Refresh the KB index
        await self._refresh_kb_index(kb_id)
        logger.info(f"KB update complete: {updated} sources changed")

        return KnowledgeBaseMetadata(
            id=kb_id,
            name=kb_data.get("name", kb_id),
            topic=kb_data.get("topic", ""),
            description=kb_data.get("description", ""),
            source_type=kb_data.get("source_type", "directory"),
            article_count=kb_data.get("article_count", 0),
            source_count=kb_data.get("source_count", 0),
            status="ready",
        )

    async def run_health_check(self, kb_id: str) -> KBHealthReport | None:
        """Run LLM-backed health check: find contradictions, orphans, gaps.

        Args:
            kb_id: The KB node ID to audit.
        """
        if kb_id not in self.graph.nodes:
            return KBHealthReport(
                kb_id=kb_id,
                kb_name=kb_id,
                consistency_score=0.0,
                summary=f"KB not found: {kb_id}",
            )

        kb_data = self.graph.nodes[kb_id]
        # Collect all articles for this KB
        articles = [
            {
                "id": n,
                "title": self.graph.nodes[n].get("name", n),
                "summary": self.graph.nodes[n].get("description", ""),
                "tags": self.graph.nodes[n].get("tags", []),
            }
            for n in self.graph.predecessors(kb_id)
            if self.graph.nodes[n].get("type") == RegistryNodeType.ARTICLE
        ]

        return await self.extractor.run_health_check(
            kb_id=kb_id,
            kb_name=kb_data.get("name", kb_id),
            articles=articles,
        )

    async def archive_kb(self, kb_id: str, threshold: float = 0.3) -> KBArchiveResult:
        """Archive a KB: compress low-importance articles to summary-only.

        Articles below the importance threshold have their full `content`
        removed from the graph, keeping only `summary`. This dramatically
        reduces graph size while preserving discoverability.

        Args:
            kb_id: The KB to archive.
            threshold: Importance score below which articles are compressed.
        """
        if kb_id not in self.graph.nodes:
            return KBArchiveResult(
                kb_id=kb_id,
                articles_compressed=0,
                nodes_pruned=0,
                bytes_saved=0,
                archive_timestamp=_now(),
            )

        compressed = 0
        pruned = 0
        bytes_saved = 0

        for n in list(self.graph.predecessors(kb_id)):
            node_data = self.graph.nodes[n]
            if node_data.get("type") != RegistryNodeType.ARTICLE:
                continue

            importance = node_data.get("importance_score", 1.0)
            if importance < threshold:
                content = node_data.get("content", "")
                if content:
                    bytes_saved += len(content.encode("utf-8"))
                    self.graph.nodes[n]["content"] = ""  # Summary only
                    compressed += 1

        logger.info(
            f"Archived KB {kb_id}: {compressed} articles compressed, "
            f"{bytes_saved:,} bytes saved"
        )
        return KBArchiveResult(
            kb_id=kb_id,
            articles_compressed=compressed,
            nodes_pruned=pruned,
            bytes_saved=bytes_saved,
            archive_timestamp=_now(),
        )

    def list_knowledge_bases(self) -> list[dict]:
        """Return lightweight summaries of all KBs in the graph.

        This is the fast 'discovery' method for agents — reads from the graph
        without any backend query needed.
        """
        kbs = []
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == RegistryNodeType.KNOWLEDGE_BASE:
                kbs.append(
                    {
                        "id": n,
                        "name": data.get("name", n),
                        "topic": data.get("topic", ""),
                        "description": data.get("description", ""),
                        "article_count": data.get("article_count", 0),
                        "source_count": data.get("source_count", 0),
                        "status": data.get("status", "unknown"),
                        "suggested_queries": [],  # populated from KBIndex if available
                    }
                )
        return sorted(kbs, key=lambda x: x["name"])

    def search_knowledge_base(
        self, query: str, kb_id: str | None = None, top_k: int = 5
    ) -> list[dict]:
        """Simple keyword-based search across KB articles (offline fallback)."""
        query_lower = query.lower()
        results = []

        for n, data in self.graph.nodes(data=True):
            if data.get("type") != RegistryNodeType.ARTICLE:
                continue

            if kb_id:
                kb_neighbors = [t for t in self.graph.successors(n) if t == kb_id]
                if not kb_neighbors:
                    continue

            title = data.get("name", "")
            summary = data.get("description", "")
            content = data.get("content", "")
            text = f"{title} {summary} {content}".lower()

            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                parent_kb = next(
                    (
                        t
                        for t in self.graph.successors(n)
                        if self.graph.nodes.get(t, {}).get("type")
                        == RegistryNodeType.KNOWLEDGE_BASE
                    ),
                    None,
                )
                results.append(
                    {
                        "article_id": n,
                        "article_title": title,
                        "kb_id": parent_kb or "",
                        "kb_name": self.graph.nodes.get(parent_kb or "", {}).get(
                            "name", ""
                        ),
                        "excerpt": summary[:200],
                        "score": score,
                        "result_type": "article",
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search(
        self, query: str, kb_id: str | None = None, top_k: int = 5
    ) -> list[dict]:
        """Alias for search_knowledge_base."""
        return self.search_knowledge_base(query, kb_id=kb_id, top_k=top_k)

    async def health_check(self, kb_id: str) -> dict:
        """Alias for run_health_check."""
        report = await self.run_health_check(kb_id)
        return report.model_dump() if report else {}

    async def ingest(
        self, kb_id: str, source: str, name: str | None = None, **kwargs
    ) -> dict:
        """Legacy ingest alias for compatibility."""
        # result is not awaited in some places, so we return a sync-like dict if possible
        # but ingest_directory is async.
        await self.ingest_directory(source, kb_name=kb_id, topic=name, **kwargs)
        return {"status": "success", "job_id": "sync"}

    def list_bases(self) -> list[dict]:
        """Alias for list_knowledge_bases."""
        return self.list_knowledge_bases()

    def get_article(self, article_id: str) -> dict | None:
        """Retrieve a specific article by ID."""
        for n, data in self.graph.nodes(data=True):
            if n == article_id and data.get("type") == RegistryNodeType.ARTICLE:
                return {"id": n, **data}
        return None

    async def update(self, kb_id: str, **kwargs):
        """Alias for update_kb."""
        await self.update_kb(kb_id)

    # ------------------------------------------------------------------
    # Internal ingestion pipeline
    # ------------------------------------------------------------------

    async def _ingest_sources(
        self,
        kb_name: str,
        topic: str,
        source_type: str,
        sources: list[ParsedSource],
        description: str = "",
        force: bool = False,
        extra_metadata: dict | None = None,
    ) -> KnowledgeBaseMetadata:
        """Core ingestion pipeline: parse → extract → write graph → sync."""
        kb_id = _kb_id(kb_name)
        timestamp = _now()

        # Upsert the KnowledgeBase root node
        kb_node = KnowledgeBaseNode(
            id=kb_id,
            name=kb_name,
            description=description or topic,
            topic=topic,
            source_type=source_type,
            source_count=len(sources),
            article_count=0,
            status="ingesting",
            importance_score=0.8,
            timestamp=timestamp,
            metadata=extra_metadata or {},
        )
        self.graph.add_node(kb_id, **kb_node.model_dump())

        articles_written = 0
        for source in sources:
            try:
                articles_written += await self._process_source(
                    source, kb_id, topic, force=force
                )
            except Exception as e:
                logger.error(f"Failed to process source {source.name}: {e}")

        # Update counts
        self.graph.nodes[kb_id]["article_count"] = articles_written
        self.graph.nodes[kb_id]["status"] = "ready"

        # Generate/refresh KB index for agent discoverability
        await self._refresh_kb_index(kb_id)

        # Persist to backend if available
        if self.backend:
            self._persist_node(kb_id)

        logger.info(
            f"KB '{kb_name}' ready: {articles_written} articles from {len(sources)} sources"
        )

        return KnowledgeBaseMetadata(
            id=kb_id,
            name=kb_name,
            topic=topic,
            description=description or topic,
            source_type=source_type,  # type: ignore[arg-type]
            article_count=articles_written,
            source_count=len(sources),
            status="ready",
            timestamp=timestamp,
        )

    async def _process_source(
        self,
        source: ParsedSource,
        kb_id: str,
        topic: str,
        force: bool = False,
    ) -> int:
        """Process a single ParsedSource → RawSourceNode + ArticleNode."""
        source_id = _source_id(source.file_path)

        # Check deduplication (skip unchanged files)
        if not force and source_id in self.graph.nodes:
            existing_hash = self.graph.nodes[source_id].get("content_hash", "")
            if existing_hash == source.content_hash:
                logger.debug(f"Skipping unchanged source: {source.name}")
                return 0

        # Write RawSource node
        raw_node = RawSourceNode(
            id=source_id,
            name=source.name,
            description=f"Source document: {source.name}",
            file_path=source.file_path,
            source_type=source.source_type,
            content_hash=source.content_hash,
            file_size=source.file_size,
            status="processed",
            importance_score=0.5,
            timestamp=_now(),
        )
        self.graph.add_node(source_id, **raw_node.model_dump())
        self.graph.add_edge(source_id, kb_id, type=RegistryEdgeType.BELONGS_TO_KB)

        # Extract article with Pydantic AI (or fallback if LLM unavailable)
        kb_name = self.graph.nodes[kb_id].get("name", kb_id)
        article_topic = f"{kb_name} - {source.name}"

        # Check if article already exists for incremental update
        existing_article = None
        article_id = _article_id(kb_id, source.name)
        if article_id in self.graph.nodes:
            node_data = self.graph.nodes[article_id]
            from ...models.knowledge_base import ExtractedArticle

            existing_article = ExtractedArticle(
                title=node_data.get("name", source.name),
                summary=node_data.get("description", ""),
                content=node_data.get("content", ""),
                concepts=[],
                facts=[],
                backlinks=[],
                tags=node_data.get("tags", []),
            )

        extracted = await self.extractor.extract_article(
            chunks=source.chunks,
            topic=article_topic,
            existing_article=existing_article,
        )

        if not extracted:
            return 0

        # Write Article node
        article_node = ArticleNode(
            id=article_id,
            name=extracted.title,
            description=extracted.summary,
            summary=extracted.summary,
            content=extracted.content,
            word_count=len(extracted.content.split()),
            tags=extracted.tags,
            importance_score=0.6,
            timestamp=_now(),
        )
        self.graph.add_node(article_id, **article_node.model_dump())
        self.graph.add_edge(article_id, kb_id, type=RegistryEdgeType.BELONGS_TO_KB)
        self.graph.add_edge(article_id, source_id, type=RegistryEdgeType.COMPILED_FROM)
        self.graph.add_edge(article_id, source_id, type=RegistryEdgeType.CITES)

        # Write KBConcept nodes
        for concept_name in extracted.concepts:
            concept_id = f"kbc:{kb_id}:{concept_name.lower().replace(' ', '-')}"
            if concept_id not in self.graph.nodes:
                concept_node = KBConceptNode(
                    id=concept_id,
                    name=concept_name,
                    description=f"Concept: {concept_name} (from {kb_name})",
                    importance_score=0.5,
                    timestamp=_now(),
                )
                self.graph.add_node(concept_id, **concept_node.model_dump())
                self.graph.add_edge(
                    concept_id, kb_id, type=RegistryEdgeType.BELONGS_TO_KB
                )
            self.graph.add_edge(article_id, concept_id, type=RegistryEdgeType.ABOUT)

        # Write KBFact nodes
        for fact in extracted.facts:
            fact_id = f"kbf:{uuid.uuid4().hex[:12]}"
            fact_node = KBFactNode(
                id=fact_id,
                name=f"Fact from {extracted.title}",
                description=fact.content[:100],
                content=fact.content,
                certainty=fact.certainty,
                source_ids=[source_id],
                importance_score=fact.certainty * 0.6,
                timestamp=_now(),
            )
            self.graph.add_node(fact_id, **fact_node.model_dump())
            self.graph.add_edge(fact_id, kb_id, type=RegistryEdgeType.BELONGS_TO_KB)
            self.graph.add_edge(fact_id, source_id, type=RegistryEdgeType.CITES)

        # Write BACKLINKS edges (deferred — titles may not exist yet)
        # These are resolved in _refresh_kb_index after all articles are written

        if self.backend:
            self._persist_node(article_id)
            self._persist_node(source_id)

        return 1

    async def _refresh_kb_index(self, kb_id: str) -> None:
        """Generate/update the KBIndex node for agent discoverability."""
        # Collect articles
        articles_data = []
        for n in self.graph.predecessors(kb_id):
            if self.graph.nodes[n].get("type") == RegistryNodeType.ARTICLE:
                node_data = self.graph.nodes[n]
                from ...models.knowledge_base import ExtractedArticle

                articles_data.append(
                    ExtractedArticle(
                        title=node_data.get("name", n),
                        summary=node_data.get("description", ""),
                        content="",
                        concepts=[],
                        facts=[],
                        backlinks=[],
                        tags=node_data.get("tags", []),
                    )
                )

        if not articles_data:
            return

        # Generate index (may use LLM or fallback to simple list)
        extracted_index = await self.extractor.generate_index(kb_id, articles_data)

        kb_name = self.graph.nodes[kb_id].get("name", kb_id)
        if extracted_index:
            index_content = (
                f"# {kb_name} Knowledge Base Index\n\n"
                f"{extracted_index.overview}\n\n"
                f"## Articles\n"
                + "\n".join(
                    f"- **{s.get('title', '?')}**: {s.get('one_liner', '')}"
                    for s in extracted_index.article_summaries
                )
                + "\n\n## Key Concepts\n"
                + ", ".join(extracted_index.key_concepts)
                + "\n\n## Example Queries\n"
                + "\n".join(f"- {q}" for q in extracted_index.suggested_queries)
            )
        else:
            # Simple fallback index
            index_content = (
                f"# {kb_name} Knowledge Base Index\n\n"
                f"Contains {len(articles_data)} articles.\n\n## Articles\n"
                + "\n".join(f"- {a.title}: {a.summary}" for a in articles_data[:20])
            )

        index_id = f"kbi:{kb_id}"
        index_node = KBIndexNode(
            id=index_id,
            name=f"{kb_name} Index",
            description=f"Auto-maintained index for {kb_name}",
            content=index_content,
            kb_id=kb_id,
            article_count=len(articles_data),
            importance_score=0.9,
            timestamp=_now(),
        )
        self.graph.add_node(index_id, **index_node.model_dump())
        self.graph.add_edge(index_id, kb_id, type=RegistryEdgeType.BELONGS_TO_KB)

        # Add INDEXES_KB edges to all articles
        for n in self.graph.predecessors(kb_id):
            if self.graph.nodes[n].get("type") == RegistryNodeType.ARTICLE:
                if not self.graph.has_edge(index_id, n):
                    self.graph.add_edge(index_id, n, type=RegistryEdgeType.INDEXES_KB)

        if self.backend:
            self._persist_node(index_id)

    def _persist_node(self, node_id: str) -> None:
        """Persist a single node to the backend."""
        if not self.backend or node_id not in self.graph.nodes:
            return
        data = dict(self.graph.nodes[node_id])
        node_type = data.get("type", "")
        if isinstance(node_type, RegistryNodeType):
            node_type = node_type.value
        # Map to DDL table names
        _TYPE_TO_TABLE = {
            "knowledge_base": "KnowledgeBase",
            "article": "Article",
            "raw_source": "RawSource",
            "kb_concept": "KBConcept",
            "kb_fact": "KBFact",
            "kb_index": "KBIndex",
        }
        table = _TYPE_TO_TABLE.get(node_type)
        if not table:
            return
        try:
            # Build a Cypher MERGE with all scalar fields
            fields = {
                k: v
                for k, v in data.items()
                if isinstance(v, (str, int, float, bool)) and k != "id"
            }
            set_clause = ", ".join(f"n.{k} = ${k}" for k in fields)
            query = f"MERGE (n:{table} {{id: $id}}) SET {set_clause}"
            self.backend.execute(query, {"id": node_id, **fields})
        except Exception as e:
            logger.debug(f"Backend persist failed for {node_id}: {e}")
