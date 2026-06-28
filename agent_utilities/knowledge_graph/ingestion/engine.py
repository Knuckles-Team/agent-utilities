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

from agent_utilities.core.config import setting

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
    CONNECTOR = "connector"

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
        if p.suffix.lower() in _DOC_EXTS:
            return cls.DOCUMENT
        # A directory is ambiguous: inspect its composition rather than
        # blindly assuming a codebase. A folder of PDFs/markdown is a
        # DOCUMENT corpus; a folder with packaging markers or source files
        # is a CODEBASE. A lone non-document file falls through to a
        # codebase parse. (CONCEPT:KG-2.7)
        if p.is_dir():
            return cls._classify_dir(p)
        return cls.CODEBASE

    @classmethod
    def _classify_dir(cls, root: Path) -> ContentType:
        """Infer a directory's content type from its file composition.

        A packaging/VCS marker (``pyproject.toml``, ``package.json``,
        ``.git``, …) is a definitive CODEBASE signal. Otherwise the
        (non-vendored) files are sampled: predominantly document formats →
        DOCUMENT; any meaningful amount of source code — or nothing
        recognizable — → CODEBASE. The latter is the safe default that keeps
        an empty/ambiguous directory a codebase parse, as before.

        Vendored/build/VCS subtrees (``.venv``, ``node_modules``, ``.git``,
        …) are pruned so a document corpus carrying a bundled virtualenv is
        not misread as code. Sampling is capped at ``_CLASSIFY_SCAN_BUDGET``
        files so classification stays cheap on huge trees — the decisive
        signals surface in the first few hundred entries. (CONCEPT:KG-2.7)
        """
        docs = 0
        code = 0
        scanned = 0
        for _dpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and d != ".git"]
            for fn in filenames:
                if fn.lower() in _CODE_MARKERS:
                    return cls.CODEBASE
                ext = Path(fn).suffix.lower()
                if ext in _DOC_EXTS:
                    docs += 1
                elif ext in _CODE_EXTS:
                    code += 1
                scanned += 1
            if scanned >= _CLASSIFY_SCAN_BUDGET:
                break
        # A ``.git`` dir (pruned from the walk above) is a definitive marker.
        if (root / ".git").exists():
            return cls.CODEBASE
        if docs > code:
            return cls.DOCUMENT
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
    # Text payloads the unified intelligence layer (``_enrich_text``) runs over,
    # drained centrally in ``ingest()`` so every content type is enriched in one
    # place (*Native by default*). Each entry: ``{source_id, text, source_type,
    # title, concepts_done}``. Aggregating adaptors (dir/codebase) collect their
    # units' payloads here so nested ingests are covered too.
    enrichable: list[dict[str, Any]] = Field(default_factory=list)


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

# Document formats handled by the document adaptor (file suffix or, for a
# directory, the dominant file type). Shared by ``ContentType.classify`` and
# its directory-composition heuristic ``_classify_dir``.
_DOC_EXTS = {
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

# Source-code extensions that mark a directory as a codebase.
_CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".c",
    ".h",
    ".hpp",
    ".cpp",
    ".cc",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".sql",
    ".lua",
}

# Filenames that definitively mark a directory as a codebase (packaging /
# build / VCS roots), matched case-insensitively.
_CODE_MARKERS = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "gemfile",
    "composer.json",
}

# Upper bound on files sampled when inferring a directory's type — keeps
# classification cheap on huge trees; the decisive markers/extensions surface
# in the first few hundred entries.
_CLASSIFY_SCAN_BUDGET = 600


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


def _git_head_sha(path: str) -> str | None:
    """Return the repo's HEAD commit sha, or ``None`` if ``path`` isn't a git
    work-tree (or git is unavailable). Used as the durable watermark for
    git-aware delta ingestion (CONCEPT:KG-2.8)."""
    import subprocess

    try:
        r = subprocess.run(  # nosec B607 B603
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _git_worktree_clean(path: str) -> bool:
    """True if the git work-tree has no uncommitted/untracked changes.

    Git-delta diffs ``since_sha..HEAD`` (commit-to-commit), which would miss
    uncommitted edits — so we only trust it on a clean tree; a dirty tree falls
    back to the full walk (still cheap via the pre-hash skip). (CONCEPT:KG-2.8)
    """
    import subprocess

    try:
        r = subprocess.run(  # nosec B607 B603
            ["git", "-C", path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and not r.stdout.strip()


def _changed_source_files(repo_path: str, since_sha: str) -> list[Path] | None:
    """Source files (any engine-supported language) changed since ``since_sha``
    (``git diff``), or ``None`` if git can't answer (caller falls back to a full
    walk).

    Only added/modified files whose extension is in :data:`SOURCE_EXTENSIONS`
    and outside vendored/build dirs are returned; deletions are dropped (the
    unchanged-everything-else case yields ``[]``, which correctly drives a
    near-empty re-ingest). Mirrors :func:`discover_source_files` so the git-delta
    fast path stays language-complete — a changed ``.java``/``.ts``/``.rs`` is
    caught, not just ``.py``. (CONCEPT:KG-2.8 / KG-2.3)
    """
    from ..core.fingerprint import detect_stale_files
    from ..enrichment.pipeline import SOURCE_EXTENSIONS

    try:
        changes = detect_stale_files(repo_path, since_commit=since_sha)
    except Exception:  # noqa: BLE001 — any git failure → full walk
        return None
    out: list[Path] = []
    for ch in changes:
        if ch.get("status") == "deleted":
            continue
        fp = ch.get("full_path") or ""
        if Path(fp).suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS for part in Path(fp).parts):
            continue
        out.append(Path(fp))
    return out


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
        self.graph_name = getattr(gc, "graph_name", "__commons__")
        # CONCEPT:KG-2.269 — cache of write backends bound to a routed content
        # graph (``code:<repo>`` / ``src:<connector>`` / …). Empty + unused when
        # KG_INGEST_GRAPH_ROUTING is off (every item routes to ``self.graph_name``).
        self._routed_backends: dict[str, Any] = {}

    def _routed_write(
        self,
        *,
        kind: str | None = None,
        source_type: str | None = None,
        repo: str | None = None,
    ) -> tuple[str, Any]:
        """Resolve ``(write_graph, backend)`` for an ingestion item (CONCEPT:KG-2.269).

        Maps the item to its destination graph via the routing policy and returns a
        backend bound to it, so structural/content writes land on a per-source graph
        that hashes to its own redb shard writer instead of all funnelling into
        ``__commons__``. When routing is off — or the item routes back to this
        engine's own graph — returns ``(self.graph_name, self.backend)`` unchanged
        (byte-for-byte legacy). The routed graph is registered so the read path can
        fan a unified query across it.
        """
        from ..core import ingest_routing

        routed = ingest_routing.route_graph(
            kind=kind, source_type=source_type, repo=repo
        )
        if routed == self.graph_name:
            return self.graph_name, self.backend
        backend = self._routed_backends.get(routed)
        if backend is None:
            from ..backends.epistemic_graph_backend import EpistemicGraphBackend

            backend = EpistemicGraphBackend(graph_name=routed)
            self._routed_backends[routed] = backend
        ingest_routing.register_content_graph(routed)
        return routed, backend

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

    async def ingest(
        self, manifest: IngestionManifest, *, defer_enrich: bool = False
    ) -> IngestionResult:
        """Ingest a single manifest using the appropriate adaptor.

        Args:
            manifest: Describes what to ingest and how.
            defer_enrich: When ``True``, run only the structural write (the adaptor
                handler) + delta-record, and return immediately WITHOUT the inline
                LLM enrichment — the ``enrichable`` payloads are left on the result
                for a downstream ENRICH stage to drain (CONCEPT:KG-2.267 staged
                pipeline). This is the seam that decouples ENRICH from WRITE so the
                writer never idles waiting on LLM work. Default ``False`` preserves
                the original inline behaviour exactly.

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
            # Unified always-on intelligence layer: drain every text payload the
            # adaptor surfaced through the one ``_enrich_text`` seam so concepts +
            # canonical facts are extracted for EVERY content type, not per-adaptor
            # (*Native by default*). ``enrich=False`` is the single opt-out for
            # fast structural-only bulk runs. Best-effort — never fails ingest.
            #
            # ``defer_enrich`` skips the inline pass entirely so the structural write
            # returns immediately and a downstream ENRICH stage (CONCEPT:KG-2.267)
            # drains ``result.enrichable`` independently — the writer never idles on
            # LLM work. The delta-record below still fires (structural success is the
            # durable unit; enrichment is best-effort and additive).
            if not defer_enrich:
                await self._run_inline_enrich(result, manifest)
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

    async def _run_inline_enrich(
        self, result: IngestionResult, manifest: IngestionManifest
    ) -> dict[str, int]:
        """Drain a result's ``enrichable`` payloads through ``_enrich_text``.

        Factored out of :meth:`ingest` so the SAME enrichment work can run either
        inline (legacy path) or in a decoupled ENRICH stage (CONCEPT:KG-2.267). The
        per-payload work is delegated to :meth:`_enrich_payload`. Best-effort —
        never raises into ingestion. Mutates ``result`` with the enrichment counts
        (matching the original inline behaviour) and returns them.
        """
        enriched = {"concepts": 0, "facts": 0}
        if not (
            result.status == "success"
            and result.enrichable
            and manifest.metadata.get("enrich", True)
        ):
            return enriched
        for payload in result.enrichable:
            counts = await self._enrich_payload(payload, manifest.content_type.value)
            enriched["concepts"] += counts["concepts"]
            enriched["facts"] += counts["facts"]
        result.nodes_created += enriched["concepts"]
        result.edges_created += enriched["facts"]
        result.details.setdefault("enrichment", enriched)
        return enriched

    async def _enrich_payload(
        self, payload: dict[str, Any], default_source_type: str
    ) -> dict[str, int]:
        """Enrich one ``enrichable`` payload — the unit a downstream ENRICH worker runs.

        Best-effort: a failed payload counts zero and never propagates, so a slow or
        broken enrichment can never stall the WRITE side (CONCEPT:KG-2.267).
        """
        try:
            return await self._enrich_text(
                payload["source_id"],
                payload.get("text", ""),
                payload.get("source_type", default_source_type),
                payload.get("title", ""),
                enrich_concepts=not payload.get("concepts_done", False),
            )
        except Exception:  # noqa: BLE001 — enrichment never breaks ingest
            logger.debug("enrich payload failed", exc_info=True)
            return {"concepts": 0, "facts": 0}

    async def ingest_batch(
        self, manifests: list[IngestionManifest]
    ) -> list[IngestionResult]:
        """Ingest multiple manifests concurrently.

        Native-by-default this runs the **staged pipeline** (CONCEPT:KG-2.267): the
        structural WRITE of each manifest and the LLM-bound ENRICH of each payload
        are separate worker pools joined by a bounded queue, so a repo enriching
        never blocks another repo's write. ``KG_STAGED_PIPELINE=0`` falls back to the
        original ``asyncio.gather`` over the inline path (the two are behaviourally
        equivalent; the staged path only changes *when* the enrichment runs).

        Args:
            manifests: List of ingestion descriptors.

        Returns:
            List of ``IngestionResult``, one per manifest (input order).
        """
        if setting("KG_STAGED_PIPELINE", "1") != "0" and len(manifests) > 1:
            try:
                return await self.ingest_batch_staged(manifests)
            except Exception:  # noqa: BLE001 — never let the pipeline break ingestion
                logger.warning(
                    "staged pipeline failed; falling back to inline batch",
                    exc_info=True,
                )

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

    async def ingest_batch_staged(
        self, manifests: list[IngestionManifest]
    ) -> list[IngestionResult]:
        """Staged, non-blocking batch ingest (CONCEPT:KG-2.267).

        Two separated, interconnected, never-dependent-locked stages:

        * **WRITE** — pool sized by ``compute_ingest_worker_count`` (cpu/engine-batch
          bound). Each worker runs ``ingest(m, defer_enrich=True)`` — the structural
          write + delta-record only — then emits one ENRICH item per ``enrichable``
          payload. A full ENRICH queue backpressures *this* producer alone.
        * **ENRICH** — pool sized by the LLM/GPU slot count (minus the reserved
          interactive slot). Each worker drains one payload through ``_enrich_text``.

        While repo A's payloads enrich, repo B's structural write proceeds on the
        independent WRITE pool — the writer never idles on LLM work. Results return
        in input order; per-stage queue-depth + throughput counters are attached to
        ``details['pipeline']`` of the first result for observability.
        """
        from .staged_pipeline import Stage, StagedPipeline, compute_stage_workers

        results: list[IngestionResult | None] = [None] * len(manifests)

        async def _write(item: tuple[int, IngestionManifest]) -> list[dict[str, Any]]:
            idx, manifest = item
            res = await self.ingest(manifest, defer_enrich=True)
            results[idx] = res
            out: list[dict[str, Any]] = []
            if (
                res.status == "success"
                and res.enrichable
                and manifest.metadata.get("enrich", True)
            ):
                ctype = manifest.content_type.value
                for payload in res.enrichable:
                    out.append({"payload": payload, "ctype": ctype, "result": res})
            return out

        async def _enrich(item: dict[str, Any]) -> None:
            counts = await self._enrich_payload(item["payload"], item["ctype"])
            res: IngestionResult = item["result"]
            # Additive, best-effort accounting back onto the owning result. Plain
            # ``+=`` is safe: each ENRICH worker is a coroutine on the one event loop,
            # so there is no concurrent-thread race — and no cross-stage lock.
            res.nodes_created += counts["concepts"]
            res.edges_created += counts["facts"]
            agg = res.details.setdefault("enrichment", {"concepts": 0, "facts": 0})
            agg["concepts"] += counts["concepts"]
            agg["facts"] += counts["facts"]

        write_stage = Stage(
            "write",
            _write,
            workers=compute_stage_workers("write"),
            capacity=max(8, len(manifests)),
        )
        enrich_stage = Stage(
            "enrich",
            _enrich,
            workers=compute_stage_workers("enrich"),
            # Bound the in-flight enrich backlog so a slow LLM caps memory and
            # backpressures WRITE — never an unbounded queue, never a stall.
            capacity=max(16, compute_stage_workers("enrich") * 8),
        )
        pipeline = StagedPipeline([write_stage, enrich_stage])
        await pipeline.run(list(enumerate(manifests)))

        processed: list[IngestionResult] = []
        for i, res in enumerate(results):
            processed.append(
                res
                if res is not None
                else IngestionResult(
                    manifest=manifests[i], status="failed", error="not processed"
                )
            )
        if processed:
            processed[0].details.setdefault("pipeline", pipeline.metrics())
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

        if not concepts or setting("KG_CONCEPT_CODE_LINK", "1") == "0":
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

    def _fact_store(self) -> Any:
        """A ``persist_facts``-compatible store over the ingest backend.

        Maps the ``add_node(key, label=)`` / ``add_edge(s, o, rel_type=, **props)``
        protocol ``persist_facts`` writes against onto the backend's own
        signatures, so canonical-entity facts land as ``Entity`` nodes + typed
        edges that interlink with the Concept/Code graph (KG-2.64 + KG-2.8).
        """
        backend: Any = self.backend

        class _Store:
            def add_node(self, node_id: str, label: str = "", **props: Any) -> None:
                backend.add_node(node_id, type="Entity", name=label, **props)

            def add_edge(
                self, source: str, target: str, rel_type: str = "", **props: Any
            ) -> None:
                # ``tags`` is a list; flatten so every backend persists it.
                tags = props.get("tags")
                if isinstance(tags, list):
                    props["tags"] = ",".join(str(t) for t in tags)
                backend.add_edge(source, target, rel_type=rel_type, **props)

        return _Store()

    async def _extract_facts_into_graph(
        self, source_id: str, text: str, source_type: str
    ) -> int:
        """Canonical-entity fact extraction → persisted graph edges (KG-2.64).

        Single seed-stable round, dedup-on; reuses the shared extraction core +
        embedder. Best-effort — degrades to 0 facts if the LLM/engine is down.
        Returns the number of fact edges written.
        """
        if not text or not text.strip():
            return 0
        if not callable(getattr(self.backend, "add_edge", None)):
            return 0
        try:
            from ..extraction.fact_extractor import (
                ExtractedFact,
                extract_facts,
                persist_facts,
            )
        except Exception:  # noqa: BLE001
            return 0
        # Ontology-guided extraction (KG-2.255): resolve the OWL TBox schema for
        # this content type and inject it into the extractor prompt so subjects/
        # objects come out as canonical ontology types and predicates respect
        # rdfs:domain/range direction. None (non-prose, or rdflib absent on the
        # lean serving plane) → unchanged free-vocab extraction (no regression).
        schema = None
        try:
            from ..extraction.extraction_schema import load_extraction_schema

            schema = load_extraction_schema(source_type)
        except Exception:  # noqa: BLE001 — schema load never breaks ingest
            schema = None
        facts: list[Any] = []
        try:
            async for ev in extract_facts(
                text, rounds=1, dedup=True, source_file=source_id, schema=schema
            ):
                if ev.get("type") == "fact" and not ev.get("is_duplicate"):
                    facts.append(ExtractedFact(**ev["fact"]))
        except Exception:  # noqa: BLE001 — enrichment must never break ingest
            return 0
        if not facts:
            return 0
        # Ground + direction-repair (KG-2.66 / KG-2.256) BEFORE persist, so edges
        # are written with subject/object oriented per rdfs:domain/range.
        # ground_facts computes the OWL types once; repair_directions swaps
        # reversed edges and flags domain/range violations (kept, not dropped).
        grounded: list[Any]
        try:
            from ..extraction.direction_repair import repair_directions
            from ..extraction.ontology_grounding import ground_facts

            grounded, _repair_tally = repair_directions(ground_facts(facts), schema)
        except Exception:  # noqa: BLE001 — grounding/repair never breaks ingest
            grounded = [(f, {}) for f in facts]
        repaired = [f for f, _ in grounded]
        try:
            written = int(persist_facts(self._fact_store(), repaired).get("edges", 0))
        except Exception:  # noqa: BLE001
            return 0
        # Annotate the canonical Entity nodes with the grounded OWL class
        # (KG-2.66) so cross-modal entities converge on a shared type. Reuses the
        # post-repair grounding so node types match the persisted orientation.
        try:
            from ..extraction.fact_extractor import ExtractedFact as _EF

            add_node = self.backend.add_node  # type: ignore[union-attr]  # best-effort; backend present during ingest (try-guarded)
            for fact, grounding in grounded:
                for surface, type_key in (
                    (fact.subject, "subject_type"),
                    (fact.object, "object_type"),
                ):
                    otype = grounding.get(type_key)
                    if otype:
                        add_node(
                            _EF.normalize_key(surface),
                            type="Entity",
                            ontology_type=otype,
                        )
        except Exception:  # noqa: BLE001 — grounding never breaks ingest
            pass
        return written

    async def _enrich_text(
        self,
        source_id: str,
        text: str,
        source_type: str,
        title: str = "",
        *,
        enrich_concepts: bool = True,
        enrich_facts: bool = True,
    ) -> dict[str, int]:
        """Unified always-on intelligence layer for any text-bearing ingestion.

        The single seam every content type funnels text through so enrichment is
        global, not per-adaptor bespoke (*Native by default*). Runs BOTH layers:

        * **Concepts** — ``Concept`` nodes + ``MENTIONS`` + concept↔code links so
          docs/chats/prompts/skills converge on shared concepts (KG-2.8).
        * **Facts** — canonical-entity ``subject -[predicate]-> object`` triples
          persisted as typed edges that interlink the graph (KG-2.64).

        Default-ON and woven into the flow; both layers share one lite LLM client
        + embedder. Pure best-effort — never raises into ingestion. Returns
        ``{'concepts': n, 'facts': m}``.

        Large text (a whole book is millions of tokens) is split into bounded
        windows so the LLM always sees a tractable prompt, and the number of
        enriched windows is capped (``KG_ENRICH_MAX_CHUNKS``, default 64) so cost
        stays controlled for big documents — raise it to enrich more deeply.
        """
        # Structure routing (KG-2.66): classify prose vs structured records so the
        # open LLM extraction below is reserved for prose/mixed; purely-structured
        # text (tables/forms/records) is cheap and deterministic to map and need
        # not pay for an LLM fact pass on every window.
        structure = "prose"
        try:
            from ..extraction.structure_router import classify_text

            structure = classify_text(text, doc_type=source_type)
        except Exception:  # noqa: BLE001 — routing never breaks ingest
            structure = "prose"

        windows = self._enrichment_windows(text)

        # Both enrichment passes fan their windows out with bounded concurrency
        # (CONCEPT:KG-2.174 — generalized cross-lane parallelization). vLLM batches
        # server-side, so an N-window document costs ~N/concurrency instead of N
        # sequential round-trips. CONCEPT:ORCH-1.59 — the ceiling is local-inference
        # capacity (KG_LLM_CONCURRENCY) MINUS the reserved interactive slot, so this
        # background sweep can never starve the slot the messaging responder /
        # graph-os-spawned agents need to answer. ONE semaphore is shared across both
        # passes (they run sequentially: concepts, then facts) to bound total in-flight
        # LLM work. Concurrent graph writes are already the production reality — the
        # durable task-worker pool runs `compute_ingest_worker_count()` ingest threads
        # at once — so off-threading the sync concept extractor is safe.
        import asyncio

        from agent_utilities.core.config import RESERVED_INTERACTIVE_INSTANCES

        _capacity = setting("KG_LLM_CONCURRENCY", 4)
        sem = asyncio.Semaphore(max(1, _capacity - RESERVED_INTERACTIVE_INSTANCES))

        concepts = 0
        if enrich_concepts and windows:
            # Concept extraction uses the SYNC lite-LLM client; off-thread each window
            # so they extract concurrently (was a serial per-window loop).
            async def _concepts_for(window: str) -> int:
                async with sem:
                    return await asyncio.to_thread(
                        self._extract_and_link_concepts,
                        source_id,
                        window,
                        source_type,
                        title,
                    )

            c_results = await asyncio.gather(
                *(_concepts_for(w) for w in windows), return_exceptions=True
            )
            concepts = sum(r for r in c_results if isinstance(r, int))

        facts = 0
        # Skip the open LLM fact pass ONLY for a genuinely small structured record
        # (a single window of CSV/form/table). Multi-window content is a real
        # document (a book's PDF text can look tabular to the classifier) and must
        # always be mined — never skip it, or we silently lose its facts.
        skip_facts = structure == "structured" and len(windows) <= 1
        if enrich_facts and windows and not skip_facts:
            # Fact extraction is the measured bottleneck (~tens of seconds/window on
            # the chat model). It is async, so fan the windows out concurrently under
            # the shared semaphore.
            async def _facts_for(window: str) -> int:
                async with sem:
                    return await self._extract_facts_into_graph(
                        source_id, window, source_type
                    )

            results = await asyncio.gather(
                *(_facts_for(w) for w in windows), return_exceptions=True
            )
            facts = sum(r for r in results if isinstance(r, int))
        summary: dict[str, Any] = {
            "concepts": concepts,
            "facts": facts,
            "structure": structure,
        }
        return summary

    def _enrichment_windows(self, text: str) -> list[str]:
        """Bounded LLM windows over ``text``.

        Small text → one window (unchanged). Large text → chunked via the shared
        document chunker, capped at ``KG_ENRICH_MAX_CHUNKS`` so a book yields
        robust-but-bounded coverage instead of one infeasible mega-prompt.
        """
        if not text:
            return []
        threshold = int(setting("KG_ENRICH_CHUNK_THRESHOLD", "12000"))
        if len(text) <= threshold:
            return [text]
        try:
            from ..distillation.distillation_engine import chunk_text

            chunks = chunk_text(text)
        except Exception:  # noqa: BLE001 — fall back to a single capped window
            return [text[:threshold]]
        cap = int(setting("KG_ENRICH_MAX_CHUNKS", "64"))
        return [c for c in chunks[:cap] if c and c.strip()]

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
        from ..enrichment.pipeline import (
            EnrichmentPipeline,
            make_batch_parse_fn,
            make_index_fn,
            make_parse_fn,
        )

        # CONCEPT:KG-2.269 — route this repo's durable structural writes to a
        # per-repo graph (``code:<repo>``) when graph routing is on, so they hash
        # to their own redb shard writer instead of all landing on ``__commons__``.
        # ``write_graph`` keys this repo's delta-skip manifest + community tenant
        # too, keeping them co-resident with the content they guard. Routing off →
        # ``(self.graph_name, self.backend)`` unchanged.
        write_graph, backend = self._routed_write(
            kind="code", repo=Path(source_path).name
        )

        # Enrichment writers need add_node/add_edge; wrap graph_compute if the
        # injected backend doesn't expose them (non-epistemic backends).
        if not (hasattr(backend, "add_node") and hasattr(backend, "add_edge")):
            from ..backends.epistemic_graph_backend import EpistemicGraphBackend

            backend = EpistemicGraphBackend()
            backend._graph = graph_compute

        # Dedicated ingest engine (CONCEPT:KG-2.58, Phase D): route the heavy,
        # contention-prone compute — stateless PARSE and the throwaway COMMUNITY
        # tenant — to a private ephemeral engine, isolated from the query engine and
        # the background daemons (embedding backfill / reconcile / task poll) that
        # profiling showed dominate a daemon ingest's wall-clock. The durable code/
        # feature WRITES still go to ``backend`` (the query engine, where queries
        # read them). Health-gated: if the ingest engine is unset or unreachable,
        # ``ingest_ep`` is None and everything runs on the query engine as before.
        from ..core.graph_compute import GraphComputeEngine, resolve_engine_auth

        ingest_ep = None
        parse_gc = graph_compute
        try:
            from agent_utilities.core.config import AgentConfig

            from ..core.ingest_engine import ensure_ingest_engine

            _cfg = AgentConfig()
            _auth, _insecure = resolve_engine_auth(_cfg)
            ingest_ep = ensure_ingest_engine(
                _cfg.kg_ingest_engine_endpoint, _auth, insecure=_insecure
            )
            if ingest_ep:
                parse_gc = GraphComputeEngine(endpoint=ingest_ep)
                logger.info("[KG-2.58] ingest compute routed to %s", ingest_ep)
        except Exception:  # noqa: BLE001 — any failure → query engine
            ingest_ep = None
            parse_gc = graph_compute

        # Features (call-graph communities) are cheap + non-LLM, but use a
        # transient tenant for community detection so the main graph isn't
        # polluted. Best-effort: degrade to no-features on any failure.
        community_fn = None
        comm = None
        # Unique per-job transient tenant (CONCEPT:KG-2.8): parallel codebase
        # ingests previously shared ``{graph}__enrich_comm`` and deleted each
        # other's tenant mid-run (→ "Graph not found"). A uuid suffix isolates
        # each job's community-detection tenant so multi-repo ingest is safe.
        import uuid as _uuid

        comm_name = f"{write_graph}__enrich_comm_{_uuid.uuid4().hex[:8]}"
        if manifest.metadata.get("features", True):
            try:
                # Community tenant lives on the ingest engine when one is active.
                comm = (
                    GraphComputeEngine(graph_name=comm_name, endpoint=ingest_ep)
                    if ingest_ep
                    else GraphComputeEngine(graph_name=comm_name)
                )
                community_fn = make_community_fn(comm)
            except Exception:  # noqa: BLE001
                community_fn = None

        # Seed per-file delta from the durable manifest (one bulk load).
        file_cat = "codebase_file"
        try:
            hash_seen = self.manifest.load_for_graph(write_graph, file_cat)
        except Exception:  # noqa: BLE001
            hash_seen = {}

        # CONCEPT:KG-2.8 — build the capability writeback callable (EA tools) for injection. Gated
        # by KG_EA_WRITEBACK; returns None (no-op) unless EA writeback is enabled + clients exist.
        from ..enrichment.writeback import resolve_writeback_fn

        pipe = EnrichmentPipeline(
            backend,
            # Parse on the ingest engine (parse_gc) when one is active, else the
            # query engine. Writes still flow through ``backend`` (query engine).
            make_parse_fn(parse_gc),
            community_fn=community_fn,
            hash_seen=hash_seen,
            writeback_fn=resolve_writeback_fn(backend),
            # Batched parse (one RPC for N files) when the engine advertises it;
            # falls back to per-file parse otherwise. (CONCEPT:KG-2.16)
            batch_parse_fn=make_batch_parse_fn(parse_gc),
            # Cross-file type/scope resolver (one RPC = parse + resolution) when the
            # engine advertises IndexRepository; the PRIMARY code path. (CONCEPT:KG-2.100)
            index_fn=make_index_fn(parse_gc),
        )

        # Git-aware delta (CONCEPT:KG-2.8): when the source is a git work-tree we
        # already ingested at a prior HEAD, ask ``git diff`` for the changed *.py
        # files and enrich only those — instead of walking + hashing the whole tree.
        # On a large repo with a small diff this turns thousands of stat/read/hash
        # ops into a single ``git diff`` plus a handful of parses. First ingest
        # (no prior sha), a non-git path, or any git failure falls back to the full
        # walk; the per-file content_hash skip still guards correctness either way.
        git_cat = "codebase_git"
        repo_key = str(Path(source_path).resolve())
        head_sha = _git_head_sha(source_path)
        prior_sha: str | None = None
        if head_sha:
            try:
                prior_sha = self.manifest.get(write_graph, git_cat, repo_key)
            except Exception:  # noqa: BLE001
                prior_sha = None
        changed_files: list[Path] | None = None
        # Caller-scoped file subset (CONCEPT:KG-2.150): when the manifest declares
        # an explicit ``only_files`` list (e.g. the agent-utilities self-ingest
        # scoping a DIRTY tree to its git-status-modified files), honor it verbatim
        # — parse/chunk only those files instead of falling back to the full walk a
        # dirty tree would otherwise force. Takes precedence over the git-diff path.
        explicit = manifest.metadata.get("only_files") if manifest.metadata else None
        if explicit:
            subset = [Path(p) for p in explicit if Path(p).exists()]
            if subset:
                changed_files = subset
        if (
            changed_files is None
            and head_sha
            and prior_sha
            and prior_sha != head_sha
            and _git_worktree_clean(source_path)
        ):
            changed_files = _changed_source_files(source_path, prior_sha)

        # Structural ingest of a whole repo is the heaviest single KG task (parse +
        # community + thousands of writes). Hold the BULK-INGEST gate for its whole
        # run so every background drain (embedding-backfill, reconcile_durable,
        # relevance-sweep, evolution, hygiene) yields instead of contending for the
        # single-writer engine. The bulk-ingest gate (not just the interactive
        # foreground flag, and not the submission-queue depth which drops to 0 the
        # moment this task is claimed) is what keeps a post-restart background
        # backlog from stretching a ~60s ingest into many minutes. (CONCEPT:KG-2.7)
        from agent_utilities.core.background_throttle import get_throttle

        try:
            with get_throttle().bulk_ingest(), get_throttle().foreground():
                if changed_files is not None:
                    logger.info(
                        "[KG-2.8] git-delta ingest: %d changed source file(s) since %s",
                        len(changed_files),
                        prior_sha[:8] if prior_sha else "?",
                    )
                    summary = pipe.enrich_files(changed_files)
                else:
                    summary = pipe.enrich(source_path)
        finally:
            if comm is not None:
                try:
                    comm._client.tenants.delete(comm_name)
                except Exception:  # noqa: BLE001
                    pass

        # Record the new HEAD as the delta watermark on success (best-effort).
        # NOT after an explicit ``only_files`` subset (CONCEPT:KG-2.150): a scoped
        # partial ingest did not cover the whole tree at this HEAD, so advancing the
        # whole-repo watermark would falsely mark untouched files as up-to-date —
        # the per-file content_hash skip remains the correctness backstop instead.
        if head_sha and not explicit:
            try:
                self.manifest.record(write_graph, git_cat, repo_key, head_sha)
            except Exception:  # noqa: BLE001
                logger.debug("codebase git-sha manifest persist failed", exc_info=True)

        # Persist per-file content hashes back to the durable manifest so the
        # per-file skip survives restarts.
        try:
            for fp, fh in hash_seen.items():
                self.manifest.record(write_graph, file_cat, fp, fh)
        except Exception:  # noqa: BLE001
            logger.debug("codebase per-file manifest persist failed", exc_info=True)

        # Auto-detect specs: the repo's own ``.specify/**/*.md`` → Spec nodes
        # (bounded to source_path/.specify so we don't walk the whole tree).
        specs = 0
        spec_enrichable: list[dict[str, Any]] = []
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
                    # Specs are bounded NL requirement docs — the right grain to
                    # enrich (concepts + facts), unlike per-symbol code cards whose
                    # per-item LLM cost would violate native-by-default discipline.
                    if text.strip():
                        spec_enrichable.append(
                            {
                                "source_id": sid,
                                "text": text,
                                "source_type": "spec",
                                "title": sp.name,
                            }
                        )
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
            enrichable=spec_enrichable,
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
                result = await self._ingest_document_dir(manifest, path)
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

        # Content-aware research acquisition (CONCEPT:KG-2.7) — a web page that
        # points at many papers (a "latest research" roundup) auto-downloads them
        # and ingests each, linking the page to the papers it cites. Default ON for
        # detected roundups; ``metadata["extract_papers"]`` forces on/off.
        if (
            source.startswith(("http://", "https://"))
            and result.status == "success"
            and manifest.metadata.get("extract_papers") is not False
        ):
            page_text = next(
                (e.get("text", "") for e in result.enrichable if e.get("text")), ""
            )
            if page_text:
                try:
                    await self._acquire_referenced_papers(
                        result,
                        page_text,
                        source,
                        force=bool(manifest.metadata.get("extract_papers")),
                    )
                except Exception as e:  # noqa: BLE001 — acquisition is best-effort
                    logger.warning("Research-paper acquisition failed: %s", e)

        return result

    async def _acquire_referenced_papers(
        self,
        page_result: IngestionResult,
        page_text: str,
        page_url: str,
        *,
        force: bool,
    ) -> None:
        """Extract paper links from a page, download + ingest them, link the page.

        Downloaded PDFs ingest as ordinary ``DOCUMENT`` units (so they become
        ``Document`` + ``Concept`` nodes and get ConceptMatcher cross-links for
        free); a ``MENTIONS`` edge records that the page cited each paper.
        """
        from .paper_links import extract_paper_links, is_research_roundup
        from .research_acquisition import acquire_papers

        refs = extract_paper_links(page_text)
        if not refs or (not force and not is_research_roundup(refs)):
            return
        pdf_paths = acquire_papers(refs)
        if not pdf_paths:
            return

        sub = [
            IngestionManifest(
                content_type=ContentType.DOCUMENT,
                source_uri=str(p),
                metadata={"source_kind": "research_paper", "referenced_by": page_url},
            )
            for p in pdf_paths
        ]
        results = await self.ingest_batch(sub)

        # Link the roundup page → each ingested paper (MENTIONS).
        page_id = (page_result.details or {}).get("doc_id")
        add_edge = getattr(self.backend, "add_edge", None)
        if page_id and callable(add_edge):
            for r in results:
                paper_id = (r.details or {}).get("doc_id") if r else None
                if paper_id:
                    try:
                        add_edge(page_id, paper_id, rel_type="MENTIONS")
                    except Exception:  # noqa: BLE001 — link is best-effort
                        pass

        ingested = 0
        for r in results:
            if r is not None and r.status == "success":
                ingested += 1
        if page_result.details is not None:
            page_result.details["papers_acquired"] = len(pdf_paths)
            page_result.details["papers_ingested"] = ingested

    # Document file extensions the standardized unit can read verbatim. Covers the
    # text/doc family plus every modality the reader registry handles (KG-2.66), so
    # a directory ingest picks up slides/sheets/email/audio/scanned images too.
    _DOC_EXTENSIONS = {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".text",
        ".pdf",
        ".html",
        ".htm",
        ".org",
        ".adoc",
        # reader-registry modalities (CONCEPT:KG-2.66)
        ".eml",
        ".msg",
        ".pptx",
        ".xlsx",
        ".csv",
        ".tsv",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
    }

    async def _ingest_document_dir(
        self, manifest: IngestionManifest, root: Path
    ) -> IngestionResult:
        """Ingest every document file under ``root`` via the canonical unit, so a
        directory yields exactly the same node shape as its files would alone.

        Files ingest CONCURRENTLY (CONCEPT:KG-2.174 — generalized cross-lane
        parallelization): each unit's sync ``_ingest_document_file`` is off-threaded
        and fanned out under a bounded semaphore (sized to the ingest worker count),
        so a directory of N docs costs ~N/concurrency instead of N sequential
        read+LLM passes. Covers every lane that ingests a directory — downloaded
        papers, crawled web pages, a repo's docs.
        """
        import asyncio
        import json as _json

        from ..core.engine_tasks import compute_ingest_worker_count

        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in self._DOC_EXTENSIONS
            and not any(part in _SKIP_DIRS for part in p.parts)
        ]
        if not files:
            return IngestionResult(
                manifest=manifest,
                status="skipped",
                details={"reason": "no document files found", "root": str(root)},
            )

        sem = asyncio.Semaphore(max(1, compute_ingest_worker_count()))

        async def _one(f: Path) -> IngestionResult:
            sub = IngestionManifest(
                content_type=manifest.content_type,
                source_uri=str(f),
                metadata={**manifest.metadata, "source_url": str(f)},
                force=manifest.force,
            )
            async with sem:
                return await asyncio.to_thread(self._ingest_document_file, sub, f)

        results = await asyncio.gather(
            *(_one(f) for f in files), return_exceptions=True
        )

        nodes = edges = docs = 0
        enrichable: list[dict[str, Any]] = []
        for res in results:
            if isinstance(res, IngestionResult) and res.status == "success":
                docs += 1
                nodes += res.nodes_created or 0
                edges += res.edges_created or 0
                # Bubble each unit's text up so the central seam enriches the
                # whole directory (the unit ran directly, bypassing ``ingest()``).
                enrichable.extend(res.enrichable)

        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=nodes,
            edges_created=edges,
            details={
                "documents": docs,
                "files_seen": len(files),
                "summary": _json.dumps({"root": str(root)}),
            },
            enrichable=enrichable,
        )

    async def _ingest_document_url(
        self, manifest: IngestionManifest, url: str
    ) -> IngestionResult:
        """Fetch a URL as markdown and ingest it as one canonical document.

        Fetching goes through the unified resolver (CONCEPT:KG-2.7,
        ``ingestion/web_fetch.py``): ArchiveBox's preserved snapshot when
        configured, else crawl4ai, else requests+markitdown. For bulk web
        ingestion, prefer crawling to markdown files first and ingesting the dir.

        A binary-document URL (``.pdf``/``.docx``/``.pptx``/``.xlsx`` — e.g. the
        ServiceNow release-notes PDF) is downloaded as bytes and read by the
        verbatim reader stack (``read_any``/PyMuPDF), not coerced through the
        text/markdown fetch path.
        """
        import tempfile

        from .web_fetch import _UA, resolve_web_fetch

        # Binary document URLs: download bytes → temp file with the right suffix →
        # the canonical unit reads them with the modality reader (CONCEPT:KG-2.66).
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix in {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".epub"}:
            try:
                import requests

                resp = requests.get(url, timeout=180, headers={"User-Agent": _UA})
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    "wb", suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(resp.content)
                    bin_path = Path(tmp.name)
            except Exception as e:  # noqa: BLE001
                return IngestionResult(
                    manifest=manifest, status="failed", error=f"fetch failed: {e}"
                )
            sub = IngestionManifest(
                content_type=manifest.content_type,
                source_uri=str(bin_path),
                metadata={
                    **manifest.metadata,
                    "source_url": url,
                    "fetch_backend": "requests",
                },
                force=manifest.force,
            )
            try:
                return self._ingest_document_file(sub, bin_path)
            finally:
                try:
                    bin_path.unlink()
                except OSError:
                    pass

        page = resolve_web_fetch(url)
        if page is None:
            return IngestionResult(
                manifest=manifest, status="failed", error=f"fetch failed: {url}"
            )

        # Write the materialized text to a temp file so the canonical unit (which
        # reads from a path) can ingest it, recording the real URL as source.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(page.markdown)
            doc_path = Path(tmp.name)
        sub = IngestionManifest(
            content_type=manifest.content_type,
            source_uri=str(doc_path),
            metadata={
                **manifest.metadata,
                "source_url": url,
                "fetch_backend": page.backend,
            },
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

        from ..core.ingest_profile import stage as _pstage
        from ..enrichment.extractors.document import extract_document
        from ..extraction.readers import read_any

        # read_any routes the document family to the PyMuPDF-fast read_document_text
        # and every other modality (email/pptx/xlsx/audio/OCR/...) to its registered
        # reader — one universal front door (CONCEPT:KG-2.66).
        with _pstage("read"):  # OS-5.70 per-stage timing
            text = read_any(str(path_obj))
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

        with _pstage("extract"):  # OS-5.70 — the LLM concept-extraction stage
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
                    # CONCEPT:KG-2.50 — opt-in contextual-retrieval enrichment.
                    contextual=bool(manifest.metadata.get("contextual", False)),
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
            # ``extract_document`` already did concepts → central seam adds the
            # canonical-fact layer over the same verbatim text (KG-2.64).
            enrichable=[
                {
                    "source_id": doc.id,
                    "text": text,
                    "source_type": "document",
                    "title": doc.title,
                    "concepts_done": want_concepts,
                }
            ],
        )

    @adaptor(ContentType.CONNECTOR)
    async def _ingest_connector(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest documents from a document-source connector (CONCEPT:KG-2.7).

        Bridges the ECO-4.25 connector framework to the KG: builds a connector
        from the registry/factory, drains it (incrementally via its checkpoint,
        ECO-4.26), and runs every :class:`SourceDocument` through the KG-2.48
        ``DocumentProcessor`` so it becomes first-class ``Document`` + ``Chunk``
        ontology objects — with contextual-retrieval enrichment (KG-2.50, on by
        default for connector ingest) and external-permission sync (ECO-4.28).

        Manifest contract:
          * ``source_uri`` — the connector ``source_type`` (e.g. ``filesystem``,
            ``web``, ``rest``, ``database``, or ``mcp:<package>``).
          * ``metadata["connector_config"]`` — the connector's config dict.
          * ``metadata["connector_id"]`` — stable id for checkpoint storage
            (defaults to ``source_type`` + a hash of the config).
          * ``metadata["contextual"]`` — enrichment toggle (default True here).
          * ``metadata["incremental"]`` — use ``poll`` (default True) vs ``load``.

        Incrementality: the connector's :class:`ConnectorCheckpoint` is persisted
        in the ``DeltaManifest`` (KG-2.8) under the ``connector_checkpoint``
        category, so a re-run resumes from the watermark rather than re-fetching.
        """
        import json as _json

        from ...protocols.source_connectors import (
            ConnectorCheckpoint,
            LoadConnector,
            PollConnector,
            build_connector,
            sync_access,
        )
        from ..ontology.document_processing import ChunkingConfig, DocumentProcessor

        source_type = manifest.source_uri
        config = dict(manifest.metadata.get("connector_config") or {})
        connector_id = manifest.metadata.get("connector_id") or (
            f"{source_type}:{hashlib.sha256(_json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
        )
        contextual = bool(manifest.metadata.get("contextual", True))
        incremental = bool(manifest.metadata.get("incremental", True))

        try:
            connector = build_connector(source_type, config)
        except (KeyError, ValueError, TypeError) as exc:
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error=f"connector build failed: {exc}",
            )

        # CONCEPT:KG-2.269 — route this connector's docs/chunks to a per-source
        # graph (``src:<source_type>``) when routing is on, so its writes hash to
        # their own redb shard writer instead of all landing on ``__commons__``.
        # ``write_graph`` co-keys the checkpoint manifest with the content it tracks.
        write_graph, backend = self._routed_write(
            kind="connector", source_type=source_type
        )

        # Resume from the stored checkpoint (ECO-4.26 + KG-2.8).
        prior_raw = self.manifest.get(write_graph, "connector_checkpoint", connector_id)
        prior_cp = ConnectorCheckpoint.from_json(prior_raw)

        processor = DocumentProcessor(
            backend,
            chunking=ChunkingConfig(
                chunk_size=int(manifest.metadata.get("chunk_size", 800)),
                overlap=int(manifest.metadata.get("overlap", 120)),
            ),
            contextual=contextual,
        )

        # Drain the connector → documents + the checkpoint to persist next.
        documents: list[Any] = []
        new_cp: ConnectorCheckpoint | None = None
        try:
            if incremental and isinstance(connector, PollConnector):
                for doc in connector.poll_all(prior_cp):
                    documents.append(doc)
                new_cp = connector.last_checkpoint
            elif isinstance(connector, LoadConnector):
                documents = list(connector.load())
            elif isinstance(connector, PollConnector):
                for doc in connector.poll_all(prior_cp):
                    documents.append(doc)
                new_cp = connector.last_checkpoint
            else:
                return IngestionResult(
                    manifest=manifest,
                    status="failed",
                    error=f"connector {source_type!r} supports neither load nor poll",
                )
        except Exception as exc:  # noqa: BLE001 — a failed source is a failed ingest
            return IngestionResult(
                manifest=manifest,
                status="failed",
                error=f"connector drain failed: {exc}",
            )

        docs_ok = nodes = edges = acl_synced = 0
        enrichable: list[dict[str, Any]] = []
        for doc in documents:
            if not getattr(doc, "text", "").strip():
                continue
            try:
                processed = processor.process(
                    doc.text,
                    document_id=f"doc:{source_type}:{doc.id}",
                    title=doc.title or doc.id,
                    doc_type=doc.doc_type,
                    source=doc.source_uri,
                    metadata={
                        **doc.metadata,
                        "connector": source_type,
                        "connector_id": connector_id,
                        "source_url": doc.source_uri,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — one bad doc must not abort the batch
                logger.warning("[KG-2.7] connector doc %s failed: %s", doc.id, exc)
                continue
            docs_ok += 1
            nodes += 1 + processed.chunk_count
            edges += len(processed.edges)
            enrichable.append(
                {
                    "source_id": processed.document_id,
                    "text": doc.text,
                    "source_type": "connector",
                    "title": doc.title or doc.id,
                }
            )
            # ECO-4.28 — mirror the source's ACL onto the doc + its chunks.
            if doc.external_access is not None:
                chunk_edges = [
                    (e["source"], e["target"])
                    for e in processed.edges
                    if e.get("type") == "HAS_CHUNK"
                ]
                if (
                    sync_access(processed.document_id, doc.external_access, chunk_edges)
                    is not None
                ):
                    acl_synced += 1

        # Persist the advanced checkpoint so the next run is incremental.
        if new_cp is not None:
            try:
                self.manifest.record(
                    write_graph,
                    "connector_checkpoint",
                    connector_id,
                    new_cp.to_json(),
                )
            except Exception:  # noqa: BLE001
                logger.debug("connector checkpoint record failed", exc_info=True)

        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=nodes,
            edges_created=edges,
            details={
                "connector": source_type,
                "connector_id": connector_id,
                "documents": docs_ok,
                "acl_synced": acl_synced,
                "contextual": contextual,
                "checkpoint_advanced": new_cp is not None,
            },
            enrichable=enrichable,
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
                    enrichable=[
                        {
                            "source_id": ep_id,
                            "text": manifest.source_uri,
                            "source_type": "conversation",
                        }
                    ],
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
            enrichable: list[dict[str, Any]] = []
            if node_id and result.get("content_text"):
                enrichable.append(
                    {
                        "source_id": node_id,
                        "text": result["content_text"],
                        "source_type": "social",
                        "title": result.get("title", ""),
                    }
                )
            return IngestionResult(
                manifest=manifest,
                status="success" if action != "skip" else "skipped",
                nodes_created=1 if node_id else 0,
                details=result,
                enrichable=enrichable,
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

    def _materialize_body_chunks(
        self,
        node_id: str,
        text: str,
        *,
        title: str,
        doc_type: str,
        source: str = "",
    ) -> tuple[int, int]:
        """Chunk + embed a prose body and link Chunk objects to ``node_id`` (KG-2.48).

        The shared enrichment substrate that gives EVERY skill-type object — atomic
        Skill, SkillGraph reference docs, WorkflowDefinition — the same semantic-search
        + provenance layer documents/books get: ``HAS_CHUNK`` Chunk objects with bounded
        embeddings. Returns ``(chunks_created, chunk_edges)``; never raises (enrichment
        must never block ingestion).
        """
        if not (text or "").strip():
            return 0, 0
        try:
            from ..ontology.document_processing import ChunkingConfig, DocumentProcessor

            # Chunk under a distinct body-doc id so we never CLOBBER the parent
            # Skill/SkillGraph/WorkflowDefinition node (process() writes a Document at
            # its document_id); link the body-doc back to the parent with HAS_BODY.
            body_id = f"{node_id}::body"
            processor = DocumentProcessor(self.backend, chunking=ChunkingConfig())
            processed = processor.process(
                text,
                document_id=body_id,
                title=title,
                doc_type=doc_type,
                source=source,
            )
            if self.backend is not None and hasattr(self.backend, "add_edge"):
                try:
                    self.backend.add_edge(node_id, body_id, "HAS_BODY")
                except Exception:  # noqa: BLE001 — linking is best-effort
                    pass
            return processed.chunk_count, len(processed.edges)
        except Exception as e:  # noqa: BLE001 — opt-in enrichment never blocks ingest
            logger.warning("[KG-2.48] body chunking failed for %s: %s", node_id, e)
            return 0, 0

    @adaptor(ContentType.SKILL)
    async def _ingest_skill(self, manifest: IngestionManifest) -> IngestionResult:
        """Ingest an agent skill directory into the graph (full enrichment).

        CONCEPT:KG-2.7 — standardized so an atomic skill gets the SAME enrichment level
        as documents/books and skill-graphs: a Skill node carrying its skill-type-unique
        attributes (``skill_type``/triggers/tags/``has_scripts``), the instruction body
        chunked + embedded into ``Chunk`` objects (semantic search), and concepts + facts
        extracted via the central seam (the ``enrichable`` payload). ``source_uri`` is
        the directory containing ``SKILL.md``.
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
            # Skill-type-unique attributes so the skill family is queryable + each
            # type is distinguishable (atomic skill vs graph vs workflow).
            frontmatter.setdefault("skill_type", "atomic")
            frontmatter.setdefault("has_scripts", (skill_path / "scripts").is_dir())
            body = self._strip_frontmatter(content)

            skill_id = ""
            if hasattr(self.kg, "ingest_agent_skill"):
                skill_id = (
                    self.kg.ingest_agent_skill(
                        skill_file_path=str(skill_md),
                        frontmatter=frontmatter,
                        content=content,
                    )
                    or ""
                )
            skill_id = skill_id or f"skill:{frontmatter.get('name', skill_path.name)}"

            # Same enrichment level as documents/books: chunk + embed the instructions
            # (semantic search), then concepts + facts via the central seam (enrichable).
            chunks, chunk_edges = self._materialize_body_chunks(
                skill_id,
                body,
                title=str(frontmatter.get("name", "")),
                doc_type="skill",
                source=str(skill_md),
            )
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1 + chunks,
                edges_created=chunk_edges,
                details={
                    "skill_name": frontmatter.get("name", ""),
                    "skill_type": frontmatter.get("skill_type"),
                    "chunks": chunks,
                },
                enrichable=[
                    {
                        "source_id": skill_id,
                        "text": body or content,
                        "source_type": "skill",
                        "title": frontmatter.get("name", ""),
                    }
                ],
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
                    import httpx

                    verify_ssl = setting("AGENTS_INSECURE_SSL", "0") != "1"
                    async with httpx.AsyncClient(
                        timeout=15.0, verify=verify_ssl
                    ) as client:
                        resp = await client.get(source)
                        resp.raise_for_status()
                        card = resp.json()
                    self.kg.ingest_a2a_agent_card(url=source, card=card)
                    card_name = card.get("name", "")
                    card_text = "\n".join(
                        s for s in (card_name, card.get("description", "")) if s
                    )
                    return IngestionResult(
                        manifest=manifest,
                        status="success",
                        nodes_created=1,
                        details={
                            "type": "a2a_agent",
                            "name": card_name,
                        },
                        enrichable=(
                            [
                                {
                                    "source_id": f"a2a:{card_name or source}",
                                    "text": card_text,
                                    "source_type": "mcp_server",
                                    "title": card_name,
                                }
                            ]
                            if card_text.strip()
                            else []
                        ),
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
                enrichable: list[dict[str, Any]] = []
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
                        # The server's name + its tools' descriptions are the NL
                        # surface to mine for concepts + facts (what the server does).
                        tool_text = "\n".join(
                            f"{t.get('name', '')}: {t.get('description', '')}"
                            for t in tools
                            if isinstance(t, dict)
                        )
                        server_text = "\n".join(
                            s for s in (entry["name"], tool_text) if s
                        )
                        if server_text.strip():
                            enrichable.append(
                                {
                                    "source_id": f"mcp:{entry['name']}",
                                    "text": server_text,
                                    "source_type": "mcp_server",
                                    "title": entry["name"],
                                }
                            )

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
                    enrichable=enrichable,
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
                enrichable=stats.get("enrichable", []),
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

            # Unified intelligence layer (concepts + canonical facts) runs
            # centrally in ``ingest()`` over this payload (KG-2.8 + KG-2.64).
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=1,
                details={"prompt_id": prompt_id},
                enrichable=[
                    {
                        "source_id": prompt_id,
                        "text": content,
                        "source_type": "prompt",
                        "title": prompt_path.stem,
                    }
                ],
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

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Return a SKILL.md body with its leading YAML frontmatter removed."""
        import re

        m = re.match(r"^---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
        return content[m.end() :] if m else content
