"""Ingestion Engine — Single entrypoint for all data ingestion into the Knowledge Graph.

CONCEPT:KG-3.0 — Ingestion Engine

Type-safe ingestion pipeline with content-typed adaptors. Each ``ContentType``
maps 1:1 to an ``@adaptor``-decorated method on ``IngestionEngine``.

Consumers construct an ``IngestionManifest`` and call ``engine.ingest(manifest)``
or ``engine.ingest_batch([...])`` for concurrent multi-source ingestion.

Supported content types:

  ==================  ===========================================================
  ContentType         Description
  ==================  ===========================================================
  CODEBASE            Rust tree-sitter AST parse → Symbol nodes in KG
  DOCUMENT            KB extraction pipeline (chunking, LLM, embedding, graph)
  CONVERSATION        Episode nodes (chat messages, agent turns)
  SOCIAL              Social media posts (X/Twitter) → classifier → KG
  KNOWLEDGE_BASE      Skill-graph or document directory ingestion
  SPARQL              Federated entity pull from SPARQL endpoints
  SKILL               Agent skill directory (SKILL.md + frontmatter)
  MCP_SERVER          MCP config JSON or A2A agent card ingestion
  POLICY              Constitution, engineering rules, governance policies
  EVENT_STREAM        Webhook / Kafka / CDC event payloads
  PROMPT              Prompt template files → KG prompt nodes
  ==================  ===========================================================
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ContentType(StrEnum):
    """Content types supported by the Ingestion Engine.

    CONCEPT:KG-3.0

    Each value maps 1:1 to a registered ``@adaptor`` method on ``IngestionEngine``.
    """

    CODEBASE = "codebase"
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    SOCIAL = "social"
    KNOWLEDGE_BASE = "kb"
    SPARQL = "sparql"
    SKILL = "skill"
    MCP_SERVER = "mcp_server"
    POLICY = "policy"
    EVENT_STREAM = "event"
    PROMPT = "prompt"


class IngestionManifest(BaseModel):
    """Describes a single ingestion job.

    CONCEPT:KG-3.0

    Attributes:
        content_type: What kind of content is being ingested.
        source_uri: Path, URL, or identifier for the source material.
        metadata: Arbitrary key-value metadata passed to the adaptor.
        max_depth: Maximum directory traversal depth (for CODEBASE type).
        force: Re-ingest even if content hash is unchanged.
    """

    content_type: ContentType
    source_uri: str
    metadata: dict[str, Any] = {}
    max_depth: int = 3
    force: bool = False


class IngestionResult(BaseModel):
    """Standardized result from an ingestion run.

    CONCEPT:KG-3.0

    Attributes:
        manifest: The manifest that was ingested.
        status: ``"success"``, ``"failed"``, or ``"skipped"``.
        nodes_created: Number of graph nodes created.
        edges_created: Number of graph edges created.
        error: Error message if status is ``"failed"``.
        duration_ms: Wall-clock duration of the ingestion in milliseconds.
        details: Adaptor-specific result details.
    """

    manifest: IngestionManifest
    status: str
    nodes_created: int = 0
    edges_created: int = 0
    error: str | None = None
    duration_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


# ── Adaptor Registry ──────────────────────────────────────────────────────

_ADAPTORS: dict[ContentType, Callable] = {}


def adaptor(content_type: ContentType) -> Callable:
    """Register a method as the adaptor for a ``ContentType``."""

    def decorator(func: Callable) -> Callable:
        _ADAPTORS[content_type] = func
        return func

    return decorator


# ── IngestionEngine ───────────────────────────────────────────────────────


class IngestionEngine:
    """Single ingestion engine for the Knowledge Graph.

    CONCEPT:KG-3.0 — Ingestion Engine

    All content enters the KG through this engine. Each ``ContentType``
    is handled by an ``@adaptor``-decorated method that contains the
    domain-specific ingestion logic.

    Usage::

        engine = IngestionEngine(kg_engine=my_kg)

        # Ingest a codebase via Rust AST parser
        result = await engine.ingest(IngestionManifest(
            content_type=ContentType.CODEBASE,
            source_uri="/path/to/project",
        ))

        # Batch ingest multiple sources
        results = await engine.ingest_batch([
            IngestionManifest(content_type=ContentType.DOCUMENT, source_uri="doc.md"),
            IngestionManifest(content_type=ContentType.SOCIAL, source_uri='{"post_id": "123"}'),
        ])
    """

    def __init__(
        self,
        kg_engine: Any = None,
        backend: Any = None,
    ):
        """Initialize the Ingestion Engine.

        Args:
            kg_engine: ``IntelligenceGraphEngine`` instance.
                If ``None``, attempts to get the active singleton.
            backend: ``GraphBackend`` instance for persistence.
                If ``None``, uses ``kg_engine.backend``.
        """
        if kg_engine is None:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            kg_engine = IntelligenceGraphEngine.get_active()
        self.kg = kg_engine
        self.backend = backend or getattr(kg_engine, "backend", None)
        self._history: list[IngestionResult] = []

    @property
    def history(self) -> list[IngestionResult]:
        """Return the ingestion history for this engine instance."""
        return list(self._history)

    async def ingest(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a single manifest using the appropriate adaptor.

        Args:
            manifest: Describes what to ingest and how.

        Returns:
            ``IngestionResult`` with status, counts, and timing.
        """
        logger.info(
            "[KG-3.0] Ingesting %s from %s",
            manifest.content_type.value,
            manifest.source_uri[:80],
        )
        start = time.monotonic()

        if manifest.content_type not in _ADAPTORS:
            result = IngestionResult(
                manifest=manifest,
                status="failed",
                error=f"No adaptor registered for {manifest.content_type.value}",
            )
            self._history.append(result)
            return result

        try:
            handler = _ADAPTORS[manifest.content_type]
            result = await handler(self, manifest)
            result.duration_ms = (time.monotonic() - start) * 1000
            self._history.append(result)
            return result
        except Exception as e:
            logger.exception("[KG-3.0] Ingestion failed for %s", manifest.source_uri)
            result = IngestionResult(
                manifest=manifest,
                status="failed",
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )
            self._history.append(result)
            return result

    async def ingest_batch(
        self, manifests: list[IngestionManifest]
    ) -> list[IngestionResult]:
        """Ingest multiple manifests concurrently.

        Args:
            manifests: List of ingestion descriptors.

        Returns:
            List of ``IngestionResult``, one per manifest.
        """
        import asyncio

        results = await asyncio.gather(
            *[self.ingest(m) for m in manifests],
            return_exceptions=True,
        )

        processed: list[IngestionResult] = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                processed.append(
                    IngestionResult(
                        manifest=manifests[i],
                        status="failed",
                        error=str(res),
                    )
                )
            else:
                processed.append(res)
        return processed

    # ── Adaptors ───────────────────────────────────────────────────────

    @adaptor(ContentType.CODEBASE)
    async def _ingest_codebase(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a codebase using the Rust tree-sitter AST backend.

        CONCEPT:KG-3.0

        Walks the directory via ``GraphComputeEngine.parse_repository()``,
        parses files into Symbol nodes (functions, classes, imports), records
        all mutations in the reactive ledger, then flushes to the backend.
        """
        source_path = manifest.source_uri
        if not Path(source_path).exists():
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error=f"Path does not exist: {source_path}",
            )

        try:
            graph_compute = getattr(self.kg, "graph_compute", None)
            if graph_compute is None:
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error="GraphComputeEngine not available on KG engine",
                )

            graph_compute.parse_repository(source_path)
            flushed = 0
            if self.backend and hasattr(graph_compute, "flush_ledger_to_backend"):
                flushed = graph_compute.flush_ledger_to_backend(self.backend)

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=flushed,
                details={"source_path": source_path, "flushed": flushed},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.DOCUMENT)
    async def _ingest_document(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a document through the KB extraction pipeline.

        CONCEPT:KG-3.0

        Routes to ``KBIngestionEngine.ingest_directory()`` or
        ``ingest_url()`` depending on whether ``source_uri`` is a path or URL.
        Handles chunking, LLM extraction, embedding, and graph persistence.
        """
        source = manifest.source_uri
        kb_name = manifest.metadata.get("kb_name")
        topic = manifest.metadata.get("topic")
        force = manifest.force

        try:
            from ..kb.ingestion import KBIngestionEngine

            graph_compute = getattr(self.kg, "graph_compute", None)
            if graph_compute is None:
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error="GraphComputeEngine not available",
                )

            kb_engine = KBIngestionEngine(graph=graph_compute, backend=self.backend)

            if source.startswith(("http://", "https://")):
                meta = await kb_engine.ingest_url(
                    url=source,
                    kb_name=kb_name or "web",
                    topic=topic,
                    force=force,
                )
            else:
                path = Path(source)
                if path.is_file():
                    meta = await kb_engine.ingest_directory(
                        path=path.parent,
                        kb_name=kb_name or path.stem,
                        topic=topic,
                        force=force,
                    )
                else:
                    meta = await kb_engine.ingest_directory(
                        path=path,
                        kb_name=kb_name or path.name,
                        topic=topic,
                        force=force,
                    )

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=meta.article_count,
                details={"kb_id": meta.id, "article_count": meta.article_count},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.CONVERSATION)
    async def _ingest_conversation(
        self, manifest: IngestionManifest
    ) -> IngestionResult:
        """Ingest a conversation episode into the graph.

        CONCEPT:KG-3.0

        Creates an episode node representing a chat turn or conversation
        fragment. The ``source_uri`` is treated as the conversation content.
        """
        try:
            source = manifest.metadata.get("source", "chat")
            timestamp = manifest.metadata.get("timestamp")

            if hasattr(self.kg, "ingest_episode"):
                ep_id = self.kg.ingest_episode(
                    content=manifest.source_uri,
                    source=source,
                    timestamp=timestamp,
                )
                return IngestionResult(
                    manifest=manifest,
                    status="success",
                    nodes_created=1,
                    details={"episode_id": ep_id},
                )

            # Fallback: create a simple episode node
            import uuid

            ep_id = f"ep:{uuid.uuid4().hex[:8]}"
            graph = getattr(self.kg, "graph", None)
            if graph:
                graph.add_node(
                    ep_id,
                    type="episode",
                    description=manifest.source_uri,
                    source=source,
                    timestamp=timestamp or _now(),
                )
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1,
                details={"episode_id": ep_id},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.SOCIAL)
    async def _ingest_social(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest social media content (X/Twitter posts) into the graph.

        CONCEPT:KG-3.0

        Routes through ``XIngestionBridge.ingest_browse_result()`` which
        handles classification, tier scoring, and evolution candidate creation.
        """
        try:
            from ..kb.x_ingestion import XIngestionBridge

            graph_compute = getattr(self.kg, "graph_compute", None)
            bridge = XIngestionBridge(
                graph=graph_compute or getattr(self.kg, "graph", None),
                backend=self.backend,
            )

            result = await bridge.ingest_browse_result(
                browse_json=manifest.source_uri,
                kg_context=manifest.metadata.get("kg_context"),
            )

            action = result.get("action", "skip")
            node_id = result.get("node_id")
            return IngestionResult(
                manifest=manifest,
                status="success" if action != "skip" else "skipped",
                nodes_created=1 if node_id else 0,
                details=result,
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.KNOWLEDGE_BASE)
    async def _ingest_kb(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a knowledge base (skill-graph directory or document directory).

        CONCEPT:KG-3.0

        Auto-detects skill-graphs (directories containing ``SKILL.md``) and
        routes appropriately to ``KBIngestionEngine``.
        """
        try:
            from ..kb.ingestion import KBIngestionEngine

            graph_compute = getattr(self.kg, "graph_compute", None)
            if graph_compute is None:
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error="GraphComputeEngine not available",
                )

            kb_engine = KBIngestionEngine(graph=graph_compute, backend=self.backend)
            source_path = Path(manifest.source_uri)

            if (source_path / "SKILL.md").exists():
                meta = await kb_engine.ingest_skill_graph(
                    graph_path=source_path,
                    force=manifest.force,
                )
            else:
                meta = await kb_engine.ingest_directory(
                    path=source_path,
                    kb_name=manifest.metadata.get("kb_name", source_path.name),
                    topic=manifest.metadata.get("topic"),
                    force=manifest.force,
                )

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=meta.article_count + meta.source_count,
                details={
                    "kb_id": meta.id,
                    "article_count": meta.article_count,
                    "source_count": meta.source_count,
                },
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.SPARQL)
    async def _ingest_sparql(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest entities from a SPARQL endpoint.

        CONCEPT:KG-3.0

        Pulls entities from an external SPARQL endpoint and maps them to
        native ``RegistryNode`` schema using configurable ontology mappings.
        ``source_uri`` should be the SPARQL endpoint URL.
        """
        try:
            from ..integrations.sparql_ingestor import FederatedSparqlIngestor

            graph_compute = getattr(self.kg, "graph_compute", None)
            endpoints = [manifest.source_uri]
            limit = manifest.metadata.get("limit", 100)
            mapping = manifest.metadata.get("mapping")

            ingestor = FederatedSparqlIngestor(
                endpoints=endpoints,
                engine=graph_compute,
                mapping_config=mapping,
            )
            total = ingestor.ingest_entities(limit=limit)

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=total,
                details={
                    "endpoint": manifest.source_uri,
                    "entities_ingested": total,
                },
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.SKILL)
    async def _ingest_skill(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest an agent skill directory into the graph.

        CONCEPT:KG-3.0

        Parses YAML frontmatter from ``SKILL.md`` and creates a skill node
        in the KG. ``source_uri`` should be the directory containing ``SKILL.md``.
        """
        try:
            skill_path = Path(manifest.source_uri)
            skill_md = skill_path / "SKILL.md"

            if not skill_md.exists():
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error=f"No SKILL.md found in {manifest.source_uri}",
                )

            content = skill_md.read_text(encoding="utf-8")
            frontmatter = self._parse_skill_frontmatter(content)
            if not frontmatter.get("name"):
                frontmatter["name"] = skill_path.name

            if hasattr(self.kg, "ingest_agent_skill"):
                self.kg.ingest_agent_skill(
                    skill_file_path=str(skill_md),
                    frontmatter=frontmatter,
                    content=content,
                )
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1,
                details={"skill_name": frontmatter.get("name", "")},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.MCP_SERVER)
    async def _ingest_mcp_server(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest an MCP server configuration or A2A agent card.

        CONCEPT:KG-3.0

        ``source_uri`` should be a path to an ``mcp_config.json``, a URL
        for an A2A agent card, or a directory containing ``mcp_config.json``.
        """
        try:
            source = manifest.source_uri

            if source.startswith(("http://", "https://")):
                # A2A agent card from URL
                if hasattr(self.kg, "ingest_a2a_agent_card"):
                    import os

                    import httpx

                    verify_ssl = os.environ.get("AGENTS_INSECURE_SSL", "0") != "1"
                    async with httpx.AsyncClient(
                        timeout=15.0, verify=verify_ssl
                    ) as client:
                        resp = await client.get(source)
                        resp.raise_for_status()
                        card = resp.json()
                    self.kg.ingest_a2a_agent_card(url=source, card=card)
                    return IngestionResult(
                        manifest=manifest,
                        status="success",
                        nodes_created=1,
                        details={
                            "type": "a2a_agent",
                            "name": card.get("name", ""),
                        },
                    )
            else:
                # Local MCP config file
                import json as json_mod

                config_path = Path(source)
                if config_path.is_dir():
                    config_path = config_path / "mcp_config.json"

                if not config_path.exists():
                    return IngestionResult(
                        manifest=manifest,
                        status="failed",
                        error=f"Config file not found: {config_path}",
                    )

                config_data = json_mod.loads(config_path.read_text(encoding="utf-8"))
                servers = config_data.get("mcpServers", {})
                ingested = 0

                for name, srv in servers.items():
                    if hasattr(self.kg, "ingest_mcp_server"):
                        self.kg.ingest_mcp_server(
                            name=name,
                            url=f"stdio://{srv.get('command', '')}",
                            tools=[],
                            resources={"env": srv.get("env", {})},
                        )
                        ingested += 1

                return IngestionResult(
                    manifest=manifest,
                    status="success",
                    nodes_created=ingested,
                    details={
                        "type": "mcp_config",
                        "servers_ingested": ingested,
                    },
                )

            return IngestionResult(manifest=manifest, status="skipped")
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.POLICY)
    async def _ingest_policy(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest policy / constitution / engineering rules into the graph.

        CONCEPT:KG-3.0

        ``source_uri`` should be the workspace path. ``metadata`` may contain:

        - ``policy_type``: ``"constitution"``, ``"rules"``, or ``"all"``
        - ``version``: Semantic version string
        - ``rules_books_path``: Path to engineering rules
        """
        try:
            workspace_path = manifest.source_uri
            policy_type = manifest.metadata.get("policy_type", "all")
            version = manifest.metadata.get("version", "1.0.0")
            rules_path = manifest.metadata.get("rules_books_path")

            if policy_type == "constitution" and hasattr(
                self.kg, "ingest_constitution"
            ):
                stats = self.kg.ingest_constitution(
                    workspace_path=workspace_path,
                    version=version,
                )
            elif policy_type == "rules" and hasattr(
                self.kg, "ingest_engineering_rules"
            ):
                stats = self.kg.ingest_engineering_rules(
                    rules_books_path=rules_path,
                    version=version,
                )
            elif hasattr(self.kg, "ingest_all_policies"):
                stats = self.kg.ingest_all_policies(
                    workspace_path=workspace_path,
                    rules_books_path=rules_path,
                    version=version,
                )
            else:
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error="Policy ingestion methods not available on KG engine",
                )

            node_count = stats.get("policies_ingested", 0) + stats.get(
                "rules_ingested", 0
            )
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=node_count,
                details=stats,
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.EVENT_STREAM)
    async def _ingest_event_stream(
        self, manifest: IngestionManifest
    ) -> IngestionResult:
        """Ingest events from an event stream (webhook, Kafka, CDC).

        CONCEPT:KG-3.0

        Parses the ``source_uri`` as a JSON event payload and processes it
        through the event stream pipeline with automatic provenance tracking.
        """
        try:
            import json as json_mod

            from ..core.company_brain import EventStreamIngester, WebhookEvent

            ingester = EventStreamIngester()
            event_data = (
                json_mod.loads(manifest.source_uri)
                if isinstance(manifest.source_uri, str)
                else manifest.source_uri
            )

            event = WebhookEvent(
                event_id=event_data.get(
                    "event_id",
                    hashlib.sha256(manifest.source_uri.encode()).hexdigest()[:12],
                ),
                source_type=event_data.get("source_type", "webhook"),
                event_type=event_data.get("event_type", "generic"),
                payload=event_data.get("payload", event_data),
                timestamp=event_data.get("timestamp", _now()),
            )
            ingester.submit_event(event)
            result = ingester.process_batch()

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=result.nodes_created,
                edges_created=result.edges_created,
                details={
                    "events_ingested": result.events_ingested,
                    "events_failed": result.events_failed,
                },
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.PROMPT)
    async def _ingest_prompt(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a prompt template into the graph as a prompt node.

        CONCEPT:KG-3.0

        ``source_uri`` should be the path to a prompt markdown file.
        """
        try:
            prompt_path = Path(manifest.source_uri)
            if not prompt_path.exists():
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error=f"Prompt file not found: {manifest.source_uri}",
                )

            content = prompt_path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
            prompt_id = f"prompt:{prompt_path.stem}:{content_hash}"

            graph = getattr(self.kg, "graph", None)
            if graph:
                graph.add_node(
                    prompt_id,
                    type="prompt_template",
                    name=prompt_path.stem,
                    content=content,
                    file_path=str(prompt_path),
                    content_hash=content_hash,
                    timestamp=_now(),
                )

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1,
                details={"prompt_id": prompt_id},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
        """Parse YAML frontmatter from a SKILL.md file.

        Expects::

            ---
            name: my-skill
            description: Does things
            ---
            # Skill instructions...
        """
        import re

        frontmatter: dict[str, Any] = {}
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return frontmatter

        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key:
                    frontmatter[key] = value
        return frontmatter
