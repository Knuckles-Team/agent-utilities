"""Ingestion Engine — Single entrypoint for all data ingestion into the Knowledge Graph.

CONCEPT:KG-2.7 — Ingestion Engine

Type-safe ingestion pipeline with content-typed adaptors. Each ``ContentType``
maps 1:1 to an ``@adaptor``-decorated method on ``IngestionEngine``.

Consumers construct an ``IngestionManifest`` and call ``engine.ingest(manifest)``
or ``engine.ingest_batch([...])`` for concurrent multi-source ingestion.

Supported content types:

  ==================  ===========================================================
  ContentType         Description
  ==================  ===========================================================
  CODEBASE            Rust tree-sitter parse → Code/Test/Feature nodes (+ cards
                      backfilled by the background enrichment daemon)
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
import os
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

    CONCEPT:KG-2.7

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
    CONFIG = "config"

    @classmethod
    def classify(cls, source: str) -> ContentType:
        """Best-effort content-type inference from a source path/URL.

        Shared by the MCP ``graph_ingest`` wrapper and any caller so the
        path/URL → ContentType mapping lives in one place. (CONCEPT:KG-2.7)
        """
        s = (source or "").strip()
        low = s.lower()
        # Chat-log auto-discovery sentinel + Claude/IDE conversation dirs.
        if low in ("chats", "conversations") or "/.claude/projects" in low:
            return cls.CONVERSATION
        if low.startswith(("http://", "https://")):
            return cls.DOCUMENT
        p = Path(s)
        name = p.name.lower()
        if name.endswith("mcp_config.json"):
            return cls.MCP_SERVER
        # Agent-utilities config.json (model registry + tunings). Checked AFTER
        # mcp_config.json (which also ends with config.json).
        if name == "config.json":
            return cls.CONFIG
        if name == "skill.md" or (p.is_dir() and (p / "SKILL.md").exists()):
            return cls.SKILL
        doc_exts = {
            ".md",
            ".txt",
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".csv",
            ".epub",
            ".html",
            ".htm",
            ".rst",
            ".rtf",
            ".ipynb",
        }
        if p.suffix.lower() in doc_exts:
            return cls.DOCUMENT
        # Directories and Python files default to a codebase parse.
        return cls.CODEBASE


class IngestionManifest(BaseModel):
    """Describes a single ingestion job.

    CONCEPT:KG-2.7

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

    CONCEPT:KG-2.7

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


# ── Content-hash registry (durable delta-skip, CONCEPT:KG-2.8) ─────────────

_HASHERS: dict[ContentType, Callable] = {}

# Content types that participate in the durable manifest skip via the default
# source hasher — file/dir content where the hash genuinely reflects whether a
# re-ingest is needed. Deliberately EXCLUDED:
#   - CONVERSATION / SOCIAL / EVENT_STREAM: stream/episodic; each submission is new.
#   - SPARQL: source is an endpoint URL whose hash never changes even as the
#     remote data does → would skip every pull after the first.
#   - MCP_SERVER / SKILL: have their own freshness/refresh semantics (a refresh
#     re-discovers live capabilities even when the config is unchanged).
#   - POLICY: source is the whole workspace dir; hashing it is wrong + expensive.
_DEFAULT_TRACKED: set[ContentType] = {
    ContentType.CODEBASE,
    ContentType.DOCUMENT,
    ContentType.KNOWLEDGE_BASE,
    ContentType.PROMPT,
    ContentType.CONFIG,
}

# Vendored / build dirs excluded from directory content digests (mirrors
# ``enrichment.pipeline._SKIP_DIRS``).
_SKIP_DIRS = {
    ".venv",
    "venv",
    ".git",
    "node_modules",
    "__pycache__",
    "site-packages",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "target",
    "site",
}


def content_hasher(content_type: ContentType) -> Callable:
    """Register a custom content-identity hasher for a ``ContentType``.

    Registering also marks the type as manifest-tracked. Use this only when the
    default source hasher (file bytes / directory mtime+size / inline string)
    isn't the right identity for that content type.
    """

    def decorator(func: Callable) -> Callable:
        _HASHERS[content_type] = func
        return func

    return decorator


def _default_source_hash(source: str) -> str | None:
    """Content-identity hash for a path/URL/inline source.

    Returns ``None`` when the source can't be hashed (treated as untracked, so
    ingestion proceeds without a skip decision).
    """
    if source.startswith(("http://", "https://")):
        return hashlib.sha256(source.encode()).hexdigest()
    p = Path(source)
    if p.exists():
        if p.is_file():
            try:
                return hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                return None
        # Directory: cheap digest over (relpath, mtime_ns, size) of non-vendored
        # files — detects any add/remove/modify without reading file contents.
        # Uses ``os.walk`` with in-place pruning of ``_SKIP_DIRS`` so we never
        # *descend* into vendored/build trees (``.git``/``node_modules``/``.venv``
        # /``target`` …). ``rglob("*")`` would walk every file under those first
        # and only filter afterwards — pathological (minutes of CPU) on repos
        # with large vendored deps. (CONCEPT:KG-2.7)
        h = hashlib.sha256()
        entries: list[tuple[str, int, int]] = []
        for root, dirnames, filenames in os.walk(p):
            # Prune skip-dirs in place so os.walk does not recurse into them.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                fp = os.path.join(root, name)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, p)
                entries.append((rel, st.st_mtime_ns, st.st_size))
        for rel, mtime_ns, size in sorted(entries):
            h.update(rel.encode())
            h.update(str(mtime_ns).encode())
            h.update(str(size).encode())
        return h.hexdigest()
    # Inline content (JSON / text payload passed as the source string).
    return hashlib.sha256(source.encode()).hexdigest()


# ── IngestionEngine ───────────────────────────────────────────────────────


class IngestionEngine:
    """Single ingestion engine for the Knowledge Graph.

    CONCEPT:KG-2.7 — Ingestion Engine

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

        # Durable delta-skip manifest (graph-native when the backend is durable,
        # SQLite fallback otherwise), keyed by the tenant graph. (CONCEPT:KG-2.8)
        from .manifest import DeltaManifest

        self.manifest = DeltaManifest(backend=self.backend)
        gc = getattr(kg_engine, "graph_compute", None)
        self.graph_name = getattr(gc, "graph_name", "__bus__")

    @property
    def history(self) -> list[IngestionResult]:
        """Return the ingestion history for this engine instance."""
        return list(self._history)

    def _content_identity(self, manifest: IngestionManifest) -> tuple[str, str] | None:
        """Return ``(canonical_uri, content_hash)`` for a tracked manifest.

        ``None`` means the content type is not manifest-tracked (or the source
        can't be hashed), so ingestion runs without a delta-skip decision.
        """
        ct = manifest.content_type
        if ct not in _HASHERS and ct not in _DEFAULT_TRACKED:
            return None
        custom = _HASHERS.get(ct)
        h = (
            custom(self, manifest)
            if custom
            else _default_source_hash(manifest.source_uri)
        )
        if not h:
            return None
        p = Path(manifest.source_uri)
        canonical = str(p.resolve()) if p.exists() else manifest.source_uri
        return canonical, h

    async def ingest(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a single manifest using the appropriate adaptor.

        Args:
            manifest: Describes what to ingest and how.

        Returns:
            ``IngestionResult`` with status, counts, and timing.
        """
        logger.info(
            "[KG-2.7] Ingesting %s from %s",
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

        # Durable delta-skip: if this source's content is unchanged since the
        # last successful ingest, skip before dispatching the adaptor.
        identity: tuple[str, str] | None = None
        try:
            identity = self._content_identity(manifest)
        except Exception:  # noqa: BLE001 — never let hashing break ingestion
            logger.debug(
                "content identity failed for %s", manifest.source_uri, exc_info=True
            )
        if (
            identity
            and not manifest.force
            and self.manifest.seen(
                self.graph_name, manifest.content_type.value, identity[0], identity[1]
            )
        ):
            result = IngestionResult(
                manifest=manifest,
                status="skipped",
                duration_ms=(time.monotonic() - start) * 1000,
                details={"reason": "unchanged", "source": identity[0]},
            )
            self._history.append(result)
            return result

        try:
            handler = _ADAPTORS[manifest.content_type]
            result = await handler(self, manifest)
            result.duration_ms = (time.monotonic() - start) * 1000
            # Record the content hash only on a clean success so failures retry.
            if identity and result.status == "success":
                try:
                    self.manifest.record(
                        self.graph_name,
                        manifest.content_type.value,
                        identity[0],
                        identity[1],
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("manifest record failed", exc_info=True)
            self._history.append(result)
            return result
        except Exception as e:
            logger.exception("[KG-2.7] Ingestion failed for %s", manifest.source_uri)
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

    def _extract_and_link_concepts(
        self, source_id: str, text: str, source_type: str, title: str = ""
    ) -> int:
        """Extract Concept nodes + MENTIONS edges from text and persist them.

        Generic across categories (prompts, chats, docs) so they converge on
        shared Concept nodes (CONCEPT:KG-2.8). Best-effort + lazy LLM; the
        ``_concept_llm_fn`` is cached. Returns the number of concepts written.
        """
        import json as _json

        try:
            from ..enrichment.extractors.text import extract_text_concepts

            llm = getattr(self, "_concept_llm_fn", None)
            if llm is None:
                from ..enrichment.cards import make_lite_llm_fn

                llm = make_lite_llm_fn()
                self._concept_llm_fn = llm
            concepts, edges = extract_text_concepts(
                text, source_id, llm, source_type=source_type, title=title
            )
        except Exception:  # noqa: BLE001
            return 0

        add_node = getattr(self.backend, "add_node", None)
        if not callable(add_node):
            return 0
        for c in concepts:
            try:
                add_node(
                    c.id,
                    type="Concept",
                    name=c.name,
                    kind=c.kind,
                    summary=c.summary,
                    source_ids=_json.dumps(c.source_ids),
                )
            except Exception:  # noqa: BLE001
                pass
        add_edge = getattr(self.backend, "add_edge", None)
        if callable(add_edge):
            for e in edges:
                try:
                    add_edge(e.source, e.target, rel_type=e.rel_type)
                except Exception:  # noqa: BLE001
                    pass

        # Cross-link concepts → Code/Feature via vector similarity so the graph
        # interweaves (RELATES_TO / REALIZES). Best-effort: needs an embedding
        # model + a vector-searchable backend; degrades silently otherwise.
        # Gated by KG_CONCEPT_CODE_LINK (default on). (CONCEPT:KG-2.8)
        self._link_concepts_to_code(concepts)
        return len(concepts)

    def _link_concepts_to_code(self, concepts: list[Any]) -> int:
        """Write RELATES_TO/REALIZES edges from concepts to similar Code/Feature.

        Returns the number of edges written (0 if disabled/unavailable).
        """
        import os

        if not concepts or os.getenv("KG_CONCEPT_CODE_LINK", "1") == "0":
            return 0
        add_edge = getattr(self.backend, "add_edge", None)
        search = getattr(self.backend, "semantic_search", None)
        if not callable(add_edge) or not callable(search):
            return 0
        try:
            from ..enrichment.semantic import link_concepts_to_code, make_embed_fn

            embed_fn = getattr(self, "_concept_embed_fn", None)
            if embed_fn is None:
                embed_fn = make_embed_fn()
                self._concept_embed_fn = embed_fn

            def search_fn(vec: list[float], top_k: int) -> list[dict[str, Any]]:
                return search(vec, top_k) or []

            edges = link_concepts_to_code(concepts, embed_fn, search_fn)
        except Exception:  # noqa: BLE001 — enrichment must never break ingest
            return 0
        written = 0
        for e in edges:
            try:
                add_edge(e.source, e.target, rel_type=e.rel_type)
                written += 1
            except Exception:  # noqa: BLE001
                pass
        return written

    # ── Adaptors ───────────────────────────────────────────────────────

    @adaptor(ContentType.CODEBASE)
    async def _ingest_codebase(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest a codebase — **structural phase** (CONCEPT:KG-2.8).

        Runs the in-process ``EnrichmentPipeline`` over the Rust tree-sitter
        parser to produce ``Code``/``Test``/``Feature`` nodes, ``COVERS``/
        ``CALLS``/``PART_OF_FEATURE`` edges, design-pattern tags, and test
        classification — all immediately queryable. No LLM here: capability-card
        summaries are filled in later by the background enrichment daemon
        (``needs_card`` = ``Code`` nodes whose ``summary`` is empty).

        Two-level delta: the engine-level dir digest (category ``codebase``)
        skips the whole repo when nothing changed; per-file ``content_hash``
        (category ``codebase_file``) skips unchanged files within a changed repo.
        Writes upsert by stable id, so re-ingest never duplicates.
        """
        source_path = manifest.source_uri
        if not Path(source_path).exists():
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error=f"Path does not exist: {source_path}",
            )
        graph_compute = getattr(self.kg, "graph_compute", None)
        if graph_compute is None:
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error="GraphComputeEngine not available on KG engine",
            )
        import asyncio

        try:
            return await asyncio.to_thread(
                self._run_codebase_structural, manifest, graph_compute, source_path
            )
        except Exception as e:  # noqa: BLE001
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    def _run_codebase_structural(
        self, manifest: IngestionManifest, graph_compute: Any, source_path: str
    ) -> IngestionResult:
        """Blocking structural enrichment (runs in a worker thread)."""
        from ..enrichment.features import make_community_fn
        from ..enrichment.pipeline import EnrichmentPipeline, make_parse_fn

        # Enrichment writers need add_node/add_edge; wrap graph_compute if the
        # injected backend doesn't expose them (non-epistemic backends).
        backend = self.backend
        if not (hasattr(backend, "add_node") and hasattr(backend, "add_edge")):
            from ..backends.epistemic_graph_backend import EpistemicGraphBackend

            backend = EpistemicGraphBackend()
            backend._graph = graph_compute

        # Features (call-graph communities) are cheap + non-LLM, but use a
        # transient tenant for community detection so the main graph isn't
        # polluted. Best-effort: degrade to no-features on any failure.
        community_fn = None
        comm = None
        comm_name = f"{self.graph_name}__enrich_comm"
        if manifest.metadata.get("features", True):
            try:
                from ..core.graph_compute import GraphComputeEngine

                comm = GraphComputeEngine(graph_name=comm_name)
                community_fn = make_community_fn(comm)
            except Exception:  # noqa: BLE001
                community_fn = None

        # Seed per-file delta from the durable manifest (one bulk load).
        file_cat = "codebase_file"
        try:
            hash_seen = self.manifest.load_for_graph(self.graph_name, file_cat)
        except Exception:  # noqa: BLE001
            hash_seen = {}

        # CONCEPT:KG-2.8 — build the capability writeback callable (EA tools) for injection. Gated
        # by KG_EA_WRITEBACK; returns None (no-op) unless EA writeback is enabled + clients exist.
        from ..enrichment.capability_writeback import resolve_writeback_fn

        pipe = EnrichmentPipeline(
            backend,
            make_parse_fn(graph_compute),
            community_fn=community_fn,
            hash_seen=hash_seen,
            writeback_fn=resolve_writeback_fn(backend),
        )
        try:
            summary = pipe.enrich(source_path)
        finally:
            if comm is not None:
                try:
                    comm._client.tenants.delete(comm_name)
                except Exception:  # noqa: BLE001
                    pass

        # Persist per-file content hashes back to the durable manifest so the
        # per-file skip survives restarts.
        try:
            for fp, fh in hash_seen.items():
                self.manifest.record(self.graph_name, file_cat, fp, fh)
        except Exception:  # noqa: BLE001
            logger.debug("codebase per-file manifest persist failed", exc_info=True)

        # Auto-detect specs: the repo's own ``.specify/**/*.md`` → Spec nodes
        # (bounded to source_path/.specify so we don't walk the whole tree).
        specs = 0
        spec_root = Path(source_path) / ".specify"
        if spec_root.is_dir() and backend is not None:
            import hashlib as _hashlib

            for sp in sorted(spec_root.rglob("*.md")):
                try:
                    text = sp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                sid = "spec:" + _hashlib.sha256(str(sp).encode()).hexdigest()[:12]
                try:
                    backend.add_node(
                        sid,
                        type="Spec",
                        name=sp.name,
                        file_path=str(sp),
                        ast_hash=_hashlib.sha256(text.encode()).hexdigest(),
                        summary=text[:500],
                    )
                    specs += 1
                except Exception:  # noqa: BLE001
                    logger.debug("spec write failed for %s", sp, exc_info=True)

        nodes = summary.code + summary.tests + summary.features + specs
        cards_pending = max(0, summary.code - summary.cards_generated)
        details = summary.model_dump()
        details["cards_pending"] = cards_pending
        details["specs"] = specs
        details["source_path"] = source_path
        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=nodes,
            edges_created=summary.covers_edges + summary.calls_edges,
            details=details,
        )

    @adaptor(ContentType.DOCUMENT)
    async def _ingest_document(self, manifest: IngestionManifest) -> IngestionResult:
        """Standardized document ingestion (file / directory / URL → one shape).

        CONCEPT:KG-2.7 — A document yields the SAME ``Document{content}`` +
        ``IdeaBlock`` chunks + ``Concept`` shape regardless of submission form;
        ``_ingest_document_file`` is the canonical per-document unit:

          * file      → one unit.
          * directory → the unit per discovered document file (NOT KB curation).
          * URL       → fetch + convert to text, then one unit.

        LLM curation into ``Article`` nodes is the SEPARATE, explicit
        ``KNOWLEDGE_BASE`` / ``curate_wiki`` path, or opt-in here via
        ``manifest.metadata["curate"]=True`` as an enrichment layer on top.
        """
        source = manifest.source_uri

        if source.startswith(("http://", "https://")):
            result = await self._ingest_document_url(manifest, source)
        else:
            path = Path(source)
            if path.is_file():
                result = self._ingest_document_file(manifest, path)
            elif path.is_dir():
                result = self._ingest_document_dir(manifest, path)
            else:
                return IngestionResult(
                    manifest=manifest, status="failed", error=f"Not found: {source}"
                )

        # Opt-in curation enrichment layer (default off) — adds curated Article
        # nodes on top of the standardized contract without replacing it.
        if manifest.metadata.get("curate") and not source.startswith(
            ("http://", "https://")
        ):
            try:
                from ..kb.ingestion import KBIngestionEngine

                graph_compute = getattr(self.kg, "graph_compute", None)
                if graph_compute is not None:
                    kb_engine = KBIngestionEngine(
                        graph=graph_compute, backend=self.backend
                    )
                    p = Path(source)
                    await kb_engine.ingest_directory(
                        path=p if p.is_dir() else p.parent,
                        kb_name=manifest.metadata.get("kb_name")
                        or (p.name if p.is_dir() else p.stem),
                        topic=manifest.metadata.get("topic"),
                        force=manifest.force,
                    )
                    if result.details is not None:
                        result.details["curated"] = True
            except Exception as e:  # noqa: BLE001 — curation is best-effort
                logger.warning("Opt-in curation enrichment failed: %s", e)

        return result

    # Document file extensions the standardized unit can read verbatim.
    _DOC_EXTENSIONS = {
        ".md", ".markdown", ".txt", ".rst", ".text",
        ".pdf", ".html", ".htm", ".org", ".adoc",
    }

    def _ingest_document_dir(
        self, manifest: IngestionManifest, root: Path
    ) -> IngestionResult:
        """Ingest every document file under ``root`` via the canonical unit, so a
        directory yields exactly the same node shape as its files would alone."""
        import json as _json

        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in self._DOC_EXTENSIONS
            and not any(part in _SKIP_DIRS for part in p.parts)
        ]
        if not files:
            return IngestionResult(
                manifest=manifest, status="skipped",
                details={"reason": "no document files found", "root": str(root)},
            )

        nodes = edges = docs = 0
        for f in files:
            sub = IngestionManifest(
                content_type=manifest.content_type,
                source_uri=str(f),
                metadata={**manifest.metadata, "source_url": str(f)},
                force=manifest.force,
            )
            res = self._ingest_document_file(sub, f)
            if res.status == "success":
                docs += 1
                nodes += res.nodes_created or 0
                edges += res.edges_created or 0

        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=nodes,
            edges_created=edges,
            details={"documents": docs, "files_seen": len(files),
                     "summary": _json.dumps({"root": str(root)})},
        )

    async def _ingest_document_url(
        self, manifest: IngestionManifest, url: str
    ) -> IngestionResult:
        """Fetch a URL, convert to text, and ingest it as one canonical document.

        Best-effort HTML→markdown via ``markitdown`` (soft dep); falls back to a
        light tag strip. For bulk web ingestion, prefer crawling to markdown
        files first and ingesting the directory.
        """
        import re
        import tempfile

        try:
            import requests

            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.text
        except Exception as e:  # noqa: BLE001
            return IngestionResult(
                manifest=manifest, status="failed", error=f"fetch failed: {e}"
            )

        text = raw
        try:
            from markitdown import MarkItDown

            with tempfile.NamedTemporaryFile(
                "w", suffix=".html", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            text = MarkItDown().convert(tmp_path).text_content
            os.unlink(tmp_path)
        except Exception:  # noqa: BLE001 — fall back to a light tag strip
            text = re.sub(r"<[^>]+>", " ", raw)

        # Write the materialized text to a temp file so the canonical unit (which
        # reads from a path) can ingest it, recording the real URL as source.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            doc_path = Path(tmp.name)
        sub = IngestionManifest(
            content_type=manifest.content_type,
            source_uri=str(doc_path),
            metadata={**manifest.metadata, "source_url": url},
            force=manifest.force,
        )
        try:
            return self._ingest_document_file(sub, doc_path)
        finally:
            try:
                doc_path.unlink()
            except OSError:
                pass

    def _ingest_document_file(
        self, manifest: IngestionManifest, path_obj: Path
    ) -> IngestionResult:
        """Canonical single-document ingest → the standardized contract.

        CONCEPT:KG-2.7 — Standardized document ingestion. One document, however
        it is submitted (file / dir element / fetched URL), always yields the
        SAME shape so fidelity never depends on submission form:

            Document{content: <full verbatim>}
              ├─ IdeaBlock{trusted_answer: <chunk>}   (PART_OF Document)
              └─ Concept{summary}                      (MENTIONS, from Document)

        Verbatim body is retained on the ``Document`` node (re-materialisable,
        e.g. distilled back into a skill-graph). Chunks are the retrieval/dedup
        substrate. No body-rewriting LLM is involved (curation is the separate,
        explicit KNOWLEDGE_BASE/curate_wiki path). Gates (via manifest.metadata):
        ``extract_concepts`` (default on), ``chunk`` (default on).
        """
        import json as _json

        from ..enrichment.extractors.document import (
            extract_document,
            read_document_text,
        )

        text = read_document_text(str(path_obj))
        if not text.strip():
            return IngestionResult(
                manifest=manifest,
                status="skipped",
                details={"reason": "empty/unreadable"},
            )

        want_concepts = manifest.metadata.get("extract_concepts", True)
        if want_concepts:
            llm = getattr(self, "_concept_llm_fn", None)
            if llm is None:
                from ..enrichment.cards import make_lite_llm_fn

                llm = make_lite_llm_fn()
                self._concept_llm_fn = llm
        else:
            llm = lambda _p: ""  # noqa: E731 — Document node only, no concepts

        doc, concepts, edges = extract_document(str(path_obj), text, llm)
        add_node = getattr(self.backend, "add_node", None)
        if not callable(add_node):
            return IngestionResult(
                manifest=manifest, status="failed", error="backend.add_node unavailable"
            )
        source_url = manifest.metadata.get("source_url") or doc.file_path
        add_node(
            doc.id,
            type="Document",
            name=doc.title,
            doc_type=doc.doc_type,
            file_path=doc.file_path,
            ast_hash=doc.content_hash,
            content=doc.content,
            source_url=source_url,
            metadata=_json.dumps(doc.metadata)[:4000],
        )
        for c in concepts:
            add_node(
                c.id,
                type="Concept",
                name=c.name,
                kind=c.kind,
                summary=c.summary,
                source_ids=_json.dumps(c.source_ids),
            )
        add_edge = getattr(self.backend, "add_edge", None)
        if callable(add_edge):
            for e in edges:
                try:
                    add_edge(e.source, e.target, rel_type=e.rel_type)
                except Exception:  # noqa: BLE001
                    pass

        # Verbatim chunk substrate: deterministic ids keyed to the Document so a
        # re-ingest overwrites the same chunk nodes instead of duplicating them.
        chunks_created = 0
        if manifest.metadata.get("chunk", True):
            from ..distillation.distillation_engine import chunk_text

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                block_id = f"{doc.id}:chunk:{i}"
                add_node(
                    block_id,
                    type="idea_block",
                    name=f"{doc.title} §{i + 1}",
                    description=chunk[:200],
                    trusted_answer=chunk,
                    source_document_id=doc.id,
                    source=source_url,
                )
                if callable(add_edge):
                    try:
                        add_edge(block_id, doc.id, rel_type="PART_OF")
                    except Exception:  # noqa: BLE001
                        pass
                chunks_created += 1

        # KG-2.48 — opt-in materialization of first-class, embedded ``Chunk``
        # ontology objects linked to a ``Document`` via HAS_CHUNK / CHUNK_OF.
        # Default OFF so existing docs ingest unchanged; turned on per-manifest
        # via metadata["chunk_objects"]. Runs the document_processing pipeline on
        # the live backend write path.
        chunk_objects_created = 0
        chunk_object_edges = 0
        if manifest.metadata.get("chunk_objects"):
            try:
                from ..ontology.document_processing import (
                    ChunkingConfig,
                    DocumentProcessor,
                )

                processor = DocumentProcessor(
                    self.backend,
                    chunking=ChunkingConfig(
                        chunk_size=int(manifest.metadata.get("chunk_size", 800)),
                        overlap=int(manifest.metadata.get("overlap", 120)),
                    ),
                )
                processed = processor.process(
                    text,
                    document_id=doc.id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    source=source_url,
                )
                chunk_objects_created = processed.chunk_count
                chunk_object_edges = len(processed.edges)
            except Exception as e:  # noqa: BLE001 — opt-in enrichment never blocks
                logger.warning("[KG-2.48] chunk-object materialization failed: %s", e)

        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=1 + len(concepts) + chunks_created + chunk_objects_created,
            edges_created=len(edges) + chunks_created + chunk_object_edges,
            details={
                "doc_id": doc.id,
                "doc_type": doc.doc_type,
                "concepts": len(concepts),
                "chunks": chunks_created,
                "chunk_objects": chunk_objects_created,
            },
        )

    @adaptor(ContentType.CONVERSATION)
    async def _ingest_conversation(
        self, manifest: IngestionManifest
    ) -> IngestionResult:
        """Ingest a conversation episode into the graph.

        CONCEPT:KG-2.7

        Creates an episode node representing a chat turn or conversation
        fragment. The ``source_uri`` is treated as the conversation content.
        """
        try:
            # First-class multi-IDE chat-log ingestion: a "chats"/"conversations"
            # sentinel, a Claude/IDE log dir, or metadata.chats=True triggers
            # auto-discovery + bulk ingest of Thread/Message nodes (CONCEPT:KG-2.1).
            _low = manifest.source_uri.strip().lower()
            if (
                _low in ("chats", "conversations")
                or "/.claude/projects" in _low
                or manifest.metadata.get("chats")
            ):
                import asyncio as _asyncio

                from ..core.conversation_ingestion import ingest_conversations_to_kg

                ides = manifest.metadata.get("ides")  # None → all supported IDEs
                res = await _asyncio.to_thread(
                    ingest_conversations_to_kg,
                    ides=ides,
                    limit=manifest.metadata.get("limit"),
                    extract_concepts=manifest.metadata.get("extract_concepts", True),
                )
                if isinstance(res, dict) and res.get("error"):
                    return IngestionResult(
                        manifest=manifest, status="failed", error=str(res["error"])
                    )
                # Writer returns total_ingested (threads) + total_messages.
                created = 0
                if isinstance(res, dict):
                    created = int(res.get("total_ingested", 0) or 0)
                return IngestionResult(
                    manifest=manifest,
                    status="success",
                    nodes_created=created,
                    details=res if isinstance(res, dict) else {"result": str(res)},
                )

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

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

                import asyncio as _asyncio

                config_data = json_mod.loads(config_path.read_text(encoding="utf-8"))
                discover = manifest.metadata.get("discover", True)
                # Skip self (the KG server) — recursive + heavy to start.
                self_names = {"graph-os", "graph_os", "mcp-multiplexer"}

                parse = getattr(self.kg, "parse_mcp_config", None)
                if callable(parse):
                    entries = parse(config_data)
                else:
                    entries = [
                        {
                            "name": n,
                            "command": s.get("command", ""),
                            "args": s.get("args", []),
                            "env": s.get("env", {}),
                        }
                        for n, s in config_data.get("mcpServers", {}).items()
                        if not s.get("disabled")
                    ]

                discover_fn = getattr(self.kg, "discover_mcp_tools", None)
                sem = _asyncio.Semaphore(
                    int(manifest.metadata.get("discovery_concurrency", 6))
                )
                timeout = float(manifest.metadata.get("discovery_timeout", 15.0))

                async def _disc(entry):
                    if (
                        not discover
                        or entry["name"] in self_names
                        or not callable(discover_fn)
                    ):
                        return entry, []
                    async with sem:
                        try:
                            return entry, await discover_fn(entry, timeout=timeout)
                        except Exception:  # noqa: BLE001
                            return entry, []

                results = await _asyncio.gather(*[_disc(e) for e in entries])
                ingested = 0
                tools_total = 0
                for entry, tools in results:
                    if hasattr(self.kg, "ingest_mcp_server"):
                        self.kg.ingest_mcp_server(
                            name=entry["name"],
                            url=f"stdio://{entry.get('command', '')}",
                            tools=tools,
                            resources={"env": entry.get("env", {})},
                        )
                        ingested += 1
                        tools_total += len(tools)

                return IngestionResult(
                    manifest=manifest,
                    status="success",
                    nodes_created=ingested + tools_total,
                    edges_created=tools_total * 2,  # PROVIDES + HAS_METADATA per tool
                    details={
                        "type": "mcp_config",
                        "servers_ingested": ingested,
                        "tools_discovered": tools_total,
                        "discovery": discover,
                    },
                )

            return IngestionResult(manifest=manifest, status="skipped")
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.POLICY)
    async def _ingest_policy(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest policy / constitution / engineering rules into the graph.

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

        CONCEPT:KG-2.7

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

            # Concept extraction so prompts interweave with code/docs/chats
            # (CONCEPT:KG-2.8). Optional + lazy LLM; pass extract_concepts=False
            # for fast structural-only bulk runs.
            concepts = 0
            if manifest.metadata.get("extract_concepts", True):
                concepts = self._extract_and_link_concepts(
                    prompt_id, content, "prompt", prompt_path.stem
                )

            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1 + concepts,
                details={"prompt_id": prompt_id, "concepts": concepts},
            )
        except Exception as e:
            return IngestionResult(manifest=manifest, status="failed", error=str(e))

    @adaptor(ContentType.CONFIG)
    async def _ingest_config(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest the agent-utilities ``config.json`` model registry + tunings.

        CONCEPT:KG-2.7 — first-class config/LLM-model ingestion (supersedes the
        standalone ``scripts/ingest_config.py``). Creates ``LanguageModel`` /
        ``EmbeddingModel`` / ``SystemConfig`` nodes so models are queryable and
        OWL can link ``agent``→``USES_MODEL``→model. Secrets (base_url/api_key)
        are dropped before persistence.
        """
        import json as _json

        try:
            data = _json.loads(Path(manifest.source_uri).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return IngestionResult(
                manifest=manifest, status="failed", error=f"config read: {e}"
            )

        add_node = getattr(self.kg, "add_node", None)
        if not callable(add_node):
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error="engine.add_node unavailable",
            )

        drop = {"base_url", "api_key", "id"}
        models = 0
        for m in data.get("chat_models", []) or []:
            mid = m.get("id")
            if not mid:
                continue
            add_node(
                node_id=mid,
                node_type="LanguageModel",
                properties={k: v for k, v in m.items() if k not in drop},
            )
            models += 1
        for m in data.get("embedding_models", []) or []:
            mid = m.get("id")
            if not mid:
                continue
            add_node(
                node_id=mid,
                node_type="EmbeddingModel",
                properties={k: v for k, v in m.items() if k not in drop},
            )
            models += 1
        sys_keys = (
            "routing_strategy",
            "graph_router_timeout",
            "kg_llm_concurrency",
            "enable_otel",
            "a2a_broker",
            "a2a_storage",
            "max_concurrent_agents",
            "graph_persistence_type",
            "routing_strategy",
        )
        add_node(
            node_id="agent_system_config",
            node_type="SystemConfig",
            properties={k: data.get(k) for k in sys_keys if k in data},
        )
        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=models + 1,
            details={"source": manifest.source_uri, "models": models},
        )

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
