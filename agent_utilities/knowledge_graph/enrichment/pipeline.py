"""In-process KG enrichment pipeline (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 1).

Clean, in-process ingestion that uses the **epistemic-graph Rust engine** as the
compute layer (AST + native test metrics) and writes typed entities through the
single ``GraphBackend`` interface. No per-repo subprocess, no shared-graph
staging feedback loop — discovery → Rust parse → classify → upsert, gated by
``content_hash`` so re-ingest of unchanged files is ~free.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_utilities.core.config import setting
from agent_utilities.models.knowledge_graph import (
    RETIRED_EDGE_RELATIONSHIP_PROPERTIES,
    retired_edge_relationship_property_error,
    retired_node_type_property_error,
)

from .cards import CapabilityCard, LLMFn, generate_symbol_cards
from .classify import TestThresholds, classify_test
from .extractors.code_test import (
    BatchParseFn,
    IncompleteParse,
    IndexFn,
    ParseFn,
    entities_from_index_result,
    extract_source,
    extract_source_files,
    resolve_covers,
)
from .extractors.document import (
    extract_document,
    extract_intelligence,
    read_document_text,
)
from .features import CommunityFn, cluster_features, resolve_call_edges
from .iac import discover_iac_files, extract_iac, link_resources_to_service
from .models import Concept, EnrichmentEdge, ExtractionResult, GraphNode
from .patterns import detect_patterns
from .realizes import EmbedFn, resolve_realizes
from .routes import extract_routes, link_routes_to_service, resolve_service_id

logger = logging.getLogger(__name__)

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
    "target",  # Rust/Java build output
    "site",
    "vendor",  # Go/PHP vendored deps
    ".gradle",
    "bin",  # C#/Java/general build output
    "obj",  # C#/MSBuild
    ".next",
    "out",
    "third_party",
    "Pods",
}

# Source extensions the Rust engine can parse — kept in sync with
# ``parser::tree_sitter::SUPPORTED_EXTENSIONS``. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".mts",
        ".cts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hxx",
        ".hh",
        ".cs",
        # Extended-language tier (CONCEPT:AU-KG.compute.built-ast-extended; engine built with ast-extended).
        ".rb",
        ".php",
        ".sh",
        ".bash",
        ".scala",
        ".sc",
        ".lua",
    }
)


class EnrichmentSummary(BaseModel):
    files_seen: int = 0
    files_parsed: int = 0
    files_skipped_unchanged: int = 0
    tests: int = 0
    code: int = 0
    covers_edges: int = 0
    calls_edges: int = 0
    inherits_edges: int = 0
    realizes_struct_edges: int = 0
    similar_edges: int = 0
    routes: int = 0
    serves_edges: int = 0
    served_by_edges: int = 0
    resources: int = 0
    provisions_edges: int = 0
    tests_needing_work: int = 0
    patterns_tagged: int = 0
    features: int = 0
    cards_generated: int = 0
    documents: int = 0
    concepts: int = 0
    mentions_edges: int = 0
    realizes_edges: int = 0
    capabilities_minted: int = 0
    capabilities_pushed: int = 0
    intelligence_nodes: int = 0


def discover_source_files(root: str | Path) -> list[Path]:
    """Find source files of any engine-supported language under root.

    Covers Python/JS/TS/Go/Rust/Java/C/C++/C# (see :data:`SOURCE_EXTENSIONS`),
    skipping vendored/build dirs. The Rust parser dispatches on extension, so a
    repo in any of these languages produces ``Code`` nodes. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
    """
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SOURCE_EXTENSIONS else []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def logical_file_identity(real: Path, root_real: Path) -> str:
    """Repository-relative POSIX identity for ``real`` (CONCEPT:AU-KG.ingest.logical-identity).

    This is the persisted artifact identity: the per-file content-hash key,
    the id embedded in every ``code:``/``test:`` entity id, and what a native
    ``index_repository``/parse response's ``file_path`` is matched against.
    It is deliberately **not** the caller's absolute path — the same commit
    ingested from a worktree, a container mount, or the canonical checkout
    must land on the identical identity, and a checkout-directory rename must
    not perturb it.

    Both ``real`` and ``root_real`` MUST already be resolved, symlink-free
    real paths (:meth:`Path.resolve`) — the caller's containment check
    (``real == root_real or root_real in real.parents``) has already proven
    ``real`` is ``root_real`` itself or one of its descendants; this function
    does not re-check containment, only computes the relative identity.

    When ``root_real`` names a single file rather than a directory (a
    single-file ``enrich_files`` call), there is no containing tree to be
    relative to, so the identity is just that file's basename.

    Raises :class:`IncompleteParse` for a logical name that would be empty or
    ``"."`` (``real`` resolves to ``root_real`` itself, e.g. a directory
    handed to ``source_root`` and to the file list) — an empty identity can
    never be a safe hash-map key or entity id.
    """
    if root_real.is_file():
        return root_real.name
    try:
        rel = real.relative_to(root_real)
    except ValueError as exc:  # pragma: no cover — containment already proven by caller
        raise IncompleteParse(
            f"file is not under source root {root_real}: {real}"
        ) from exc
    posix = rel.as_posix()
    if not posix or posix == ".":
        raise IncompleteParse(
            f"file resolves to an empty logical identity under source root "
            f"{root_real}: {real}"
        )
    return posix


class _BatchedBackend:
    """Buffers ``add_node``/``add_edge`` and flushes them through the engine's
    bulk ``batch_update`` (one RPC per ``batch_size``) instead of one RPC per
    write. For a big-repo ingest (tens of thousands of symbols) this is the
    dominant cost — the engine is a socket round-trip per call, so N per-node
    writes = N round-trips. Nodes are flushed before edges so every edge endpoint
    already exists; reads delegate to the wrapped backend. Falls back to per-item
    writes if the engine has no bulk path or a batch fails. (CONCEPT:EG-KG.storage.nonblocking-checkpoint/2.16, #1)
    """

    def __init__(
        self, backend: Any, batch_size: int = 1000, source_system: str | None = None
    ) -> None:
        self._backend = backend
        self._batch_size = batch_size
        # Provenance source id stamped on every buffered node (CONCEPT:AU-KG.ingest.code-source-partition)
        # so code records land in their ``urn:source:code:<repo>`` named graph instead of the
        # SPARQL default graph. ``None`` ⇒ no stamping (non-code enrichment is unchanged).
        self._source_system = source_system
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[dict[str, Any]] = []
        graph = getattr(backend, "_graph", None)
        self._bulk = getattr(graph, "bulk_mutate", None) or getattr(
            graph, "batch_update", None
        )
        # Multiplexed pool fan-out: when the engine exposes the
        # pooled concurrent submitter, a large flush is split into independent
        # sub-batches that ride SEPARATE pooled connections, so the engine services
        # them as parallel per-connection tasks (and coalesces their durable commits)
        # instead of one serial BatchUpdate on the single shared client. The flush is
        # a set of independent node (or, after nodes land, edge) writes, so splitting
        # it preserves correctness; ``flush()`` still drains nodes fully before edges.
        self._bulk_concurrent = getattr(graph, "batch_update_concurrent", None)
        # Sub-batch grain: ~a quarter of the flush so a full batch fans across ~4
        # connections — enough to overlap submission without shredding the payload.
        self._sub_batch = max(1, batch_size // 4)

    @property
    def bulk_available(self) -> bool:
        """True when the wrapped backend exposes an engine bulk path, so callers
        can choose batching only when it actually collapses round-trips (and keep
        their own robust per-item path otherwise)."""
        return self._bulk is not None

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        if "type" in properties:
            raise retired_node_type_property_error()
        props: dict[str, Any] = {"node_type": label, **properties}
        if not props.get("node_type"):
            raise ValueError("node_type is required")
        # Stamp the source-provenance contract (source_system + domain) on every code
        # node at the one write chokepoint, so partition routing sends it to the right
        # named graph. A caller-set source_system still wins (stamp_source setdefaults).
        if self._source_system:
            from .provenance import stamp_source

            stamp_source(props, self._source_system)
        # Defence-in-depth ACL registration (CONCEPT:AU-KG.backend.company-brain-write-guard):
        # this buffered batch path flushes straight through the engine's bulk
        # RPC (``graph.bulk_mutate``/``batch_update``), bypassing
        # IntelligenceGraphEngine._upsert_node/GraphComputeEngine.add_node and
        # BrainGuardedBackend.add_node entirely — the other three chokepoints
        # this same fix stamps. Without this, every code-symbol node ingested
        # through this batching seam (the dominant KG-2.9g repo-ingest path)
        # shared the identical "written but permanently unreadable" gap.
        #
        # BUG-033/BUG-039: fail closed, same as the other four chokepoints —
        # a write reaching this seam with NO bound actor at all must raise,
        # not silently land unowned. A genuinely privileged/system actor
        # still lands intentionally unowned (``stamp_ownership``'s own,
        # unchanged policy for platform/code-symbol data).
        from ..core.tenant_sharing import stamp_classification, stamp_ownership

        stamp_ownership(props)
        stamp_classification(props, props.get("node_type"))
        self._nodes.append(
            {
                "op": "add_node",
                "id": node_id,
                "properties": props,
            }
        )
        if len(self._nodes) >= self._batch_size:
            self._flush_nodes()

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        aliases = RETIRED_EDGE_RELATIONSHIP_PROPERTIES.intersection(properties)
        if aliases:
            raise retired_edge_relationship_property_error(aliases)
        relationship = str(properties.get("relationship") or rel_type).strip()
        if not relationship:
            raise ValueError("relationship is required")
        props: dict[str, Any] = {**properties, "relationship": relationship}
        # Defence-in-depth ACL registration (CONCEPT:AU-KG.backend.company-brain-write-guard):
        # this buffered batch path flushes straight through the engine's bulk
        # RPC (``graph.bulk_mutate``/``batch_update``), bypassing
        # ``GraphComputeEngine.add_edge``/``IntelligenceGraphEngine`` entirely —
        # the same reason ``add_node`` above stamps directly instead of relying
        # on the wrapped backend's own gate. Without this, every edge ingested
        # through this batching seam (the dominant KG-2.9g repo-ingest path)
        # shared BUG-058's "unconditionally ungoverned" gap.
        #
        # BUG-058 (fail closed, same contract as ``add_node`` above): a write
        # reaching this seam with NO bound actor at all must raise, not
        # silently land unowned. A genuinely privileged/system actor still
        # lands intentionally unowned (``stamp_ownership``'s own, unchanged
        # policy for platform/code-symbol data).
        from ..core.tenant_sharing import stamp_classification, stamp_ownership

        stamp_ownership(props)
        stamp_classification(props, props.get("relationship"))
        self._edges.append(
            {
                "op": "add_edge",
                "source": source,
                "target": target,
                "properties": props,
            }
        )

    def _submit_bulk(self, ops: list[dict[str, Any]]) -> bool:
        """Send one independent ``ops`` flush through the fastest available path.

        Order of preference (KG-2.16): the pooled concurrent
        submitter (sub-batches fanned across separate connections) → the single bulk
        ``batch_update`` → ``False`` so the caller degrades to per-item writes.
        """
        if self._bulk_concurrent is not None and len(ops) > self._sub_batch:
            chunks = [
                ops[i : i + self._sub_batch]
                for i in range(0, len(ops), self._sub_batch)
            ]
            try:
                self._bulk_concurrent(chunks)
                return True
            except Exception as e:  # noqa: BLE001 - degrade to a single bulk RPC
                logger.debug("concurrent flush failed (%s); single-batch fallback", e)
        if self._bulk is not None:
            try:
                self._bulk(ops)
                return True
            except Exception as e:  # noqa: BLE001 - degrade to per-item writes
                logger.debug("batched flush failed (%s); per-item fallback", e)
        return False

    def _flush_nodes(self) -> None:
        if not self._nodes:
            return
        ops, self._nodes = self._nodes, []
        if self._submit_bulk(ops):
            return
        # Per-item fallback: skip a poison node instead of aborting the whole repo's
        # structural pass (CONCEPT:AU-KG.enrichment.card-attempt-status — one bad symbol must not wedge ingest).
        for op in ops:
            try:
                self._backend.add_node(op["id"], **op["properties"])
            except Exception:  # noqa: BLE001 - skip the bad node, keep ingesting
                logger.debug(
                    "per-item add_node failed for %s", op.get("id"), exc_info=True
                )

    def _flush_edges(self) -> None:
        if not self._edges:
            return
        ops, self._edges = self._edges, []
        if self._submit_bulk(ops):
            return
        for op in ops:
            try:
                self._backend.add_edge(op["source"], op["target"], **op["properties"])
            except Exception:  # noqa: BLE001 - skip the bad edge, keep ingesting
                logger.debug(
                    "per-item add_edge failed for %s->%s",
                    op.get("source"),
                    op.get("target"),
                    exc_info=True,
                )

    def flush(self) -> None:
        """Flush nodes first (so endpoints exist), then edges."""
        self._flush_nodes()
        self._flush_edges()

    def __getattr__(self, name: str) -> Any:  # delegate reads / other ops
        return getattr(self._backend, name)


class EnrichmentPipeline:
    """Enriches a target path into typed Test/Code entities + COVERS edges.

    ``parse_fn`` is the Rust ``ParseFile`` entry point (e.g.
    ``GraphComputeEngine.parse_file``). ``backend`` is any ``GraphBackend`` (must
    expose ``add_node``/``add_edge``). ``hash_seen`` lets the caller persist the
    incremental manifest across runs; pass a dict to dedupe within one run.
    """

    def __init__(
        self,
        backend: Any,
        parse_fn: ParseFn,
        thresholds: TestThresholds | None = None,
        hash_seen: dict[str, str] | None = None,
        llm_fn: LLMFn | None = None,
        community_fn: CommunityFn | None = None,
        card_cache: dict[str, CapabilityCard] | None = None,
        min_feature_size: int = 3,
        capability_provider: Callable[[], list[Any]] | None = None,
        capability_registry: list[Any] | None = None,
        mint_capabilities: bool = True,
        realizes_embed_fn: EmbedFn | None = None,
        writeback_fn: Callable[[list[GraphNode]], Any] | None = None,
        batch_parse_fn: BatchParseFn | None = None,
        index_fn: IndexFn | None = None,
        source_system: str | None = None,
    ) -> None:
        self.backend = backend
        # Canonical source id (e.g. ``code:<repo>``) stamped on every node this pipeline
        # writes, so code lands in its own ``urn:source:code:<repo>`` named graph rather
        # than the SPARQL default graph (CONCEPT:AU-KG.ingest.code-source-partition). ``None`` ⇒ unstamped.
        self.source_system = source_system
        self.parse_fn = parse_fn
        # Optional batched parse (one RPC for N files). When set, changed files
        # are parsed in a single round-trip instead of per-file. (CONCEPT:EG-KG.compute.graph-compute-engine)
        self.batch_parse_fn = batch_parse_fn
        # Optional cross-file resolver (one RPC = parse + type/scope resolution).
        # When set, it is the PRIMARY code path: symbols and already-resolved
        # CALLS/INHERITS/REALIZES come from one engine round-trip, replacing the
        # per-file parse + Python name-only call resolution. (CONCEPT:EG-KG.compute.type-scope-resolved-call)
        self.index_fn = index_fn
        self.thresholds = thresholds or TestThresholds()
        self._hash_seen = hash_seen if hash_seen is not None else {}
        self.llm_fn = llm_fn
        self.community_fn = community_fn
        self.card_cache = card_cache if card_cache is not None else {}
        self.min_feature_size = min_feature_size
        # Code → capability (REALIZES) resolution (CONCEPT:EG-KG.storage.nonblocking-checkpoint).
        self.capability_provider = capability_provider
        self.capability_registry = capability_registry
        self.mint_capabilities = mint_capabilities
        self.realizes_embed_fn = realizes_embed_fn
        self.writeback_fn = writeback_fn

    def enrich(self, target_path: str | Path) -> EnrichmentSummary:
        files = discover_source_files(target_path)
        # IaC files alongside the code (CONCEPT:AU-KG.enrichment.read-them-here-so): read them here so the
        # pipeline writes Resource nodes in the same batched pass.
        iac: list[tuple[str, str]] = []
        for p in discover_iac_files(target_path):
            try:
                iac.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
        # The ingest root's name is the best-effort hint for the deployed service a
        # route is servedBy (CONCEPT:AU-KG.enrichment.http-route-extraction).
        return self.enrich_files(
            files,
            service_hint=Path(target_path).name,
            iac_files=iac,
            source_root=target_path,
        )

    def enrich_files(
        self,
        files: Iterable[Path],
        service_hint: str = "",
        iac_files: list[tuple[str, str]] | None = None,
        source_root: str | Path | None = None,
    ) -> EnrichmentSummary:
        """Enrich an explicit file set.

        ``source_root``, when supplied, is the repository-relative containment
        root (CONCEPT:AU-KG.ingest.exact-parser-acknowledgement) AND the base
        every input's **persisted identity** is normalized against
        (CONCEPT:AU-KG.ingest.logical-identity): every input's resolved real
        path must stay inside it, or the whole batch is rejected
        (:class:`IncompleteParse`) rather than silently ingesting content a
        symlink smuggled in from outside the repository, and its identity —
        the ``self._hash_seen``/``pending_hashes`` key and the id embedded in
        every ``code:``/``test:`` entity — becomes the repository-relative
        POSIX logical path (:func:`logical_file_identity`), NOT the caller's
        absolute path. This is deliberate: the same commit ingested from a
        worktree, a container mount, or the canonical checkout must land on
        the identical identity, and a watermark recorded under one checkout
        root must remain valid after that checkout is renamed.

        ``source_root=None`` is a **deprecated legacy mode**, kept only for a
        caller that has not yet been updated to pass it. It uses the file's
        own resolved absolute path as the identity instead — always
        self-describing and never ambiguous with a logical identity, because
        a resolved absolute path always starts with ``/`` (POSIX) while a
        logical identity, being relative, never does. Every production caller
        this pipeline ships with (``enrich``, and the ingestion engine's
        structural-ingest path) already passes ``source_root``; prefer that
        over the legacy mode in any new caller.

        Any raised :class:`IncompleteParse` — from root containment, an
        unreadable file, duplicate request identity, an empty logical
        identity, or a downstream parser acknowledgement failure — leaves
        ``self._hash_seen`` and this batch's writes entirely unpersisted, so a
        caller MUST NOT advance a per-file hash or repository watermark for
        it.

        Migration note: upgrading a deployment that already has a persisted
        ``self._hash_seen``/watermark keyed by absolute path is fail-safe by
        construction, not by special-cased migration code. A logical identity
        never starts with ``/`` and a legacy identity always does (see above),
        so the two key spaces cannot collide; old absolute-path keys simply
        never match a lookup by the new logical identity, so every file looks
        "changed" once and is re-parsed and re-hashed under its logical key on
        the first run after upgrade. No content is skipped as unchanged when
        it was never actually verified under the new identity scheme — the
        transition can only cost an extra parse, never a missed one.
        """
        summary = EnrichmentSummary()
        root_real: Path | None = None
        if source_root is not None:
            try:
                root_real = Path(source_root).resolve(strict=False)
            except OSError:
                root_real = Path(source_root)

        # Phase 1 — pre-hash filter (CONCEPT:EG-KG.storage.nonblocking-checkpoint): hash the raw bytes BEFORE
        # parsing so an unchanged file costs one local sha256, not a Rust-engine
        # parse round-trip. The hash is byte-identical to ``ExtractionResult.
        # content_hash`` (same ``surrogatepass`` encoding), so the skip is exact.
        pending: list[tuple[str, str]] = []  # (identity, source_text)
        pending_hashes: dict[str, str] = {}  # identity -> content_hash
        unreadable: list[str] = []
        for fp in files:
            summary.files_seen += 1
            p = Path(fp)
            try:
                real = p.resolve(strict=False)
            except OSError:
                real = p
            if root_real is not None:
                if real != root_real and root_real not in real.parents:
                    raise IncompleteParse(f"file escapes source root {root_real}: {fp}")
                identity = logical_file_identity(real, root_real)
            else:
                # Legacy identity (deprecated — see docstring): the resolved
                # absolute path. Always starts with "/", so it can never be
                # mistaken for a logical (always-relative) identity.
                identity = str(real)
            try:
                source = p.read_text(encoding="utf-8", errors="surrogatepass")
            except (OSError, UnicodeDecodeError):
                unreadable.append(identity)
                continue
            content_hash = hashlib.sha256(
                source.encode("utf-8", "surrogatepass")
            ).hexdigest()
            if self._hash_seen.get(identity) == content_hash:
                summary.files_skipped_unchanged += 1
                continue
            pending.append((identity, source))
            pending_hashes[identity] = content_hash

        if unreadable:
            raise IncompleteParse(
                f"{len(unreadable)} of {summary.files_seen} requested file(s) "
                f"could not be read: {unreadable[:5]}"
                + (" ..." if len(unreadable) > 5 else "")
            )
        if len(pending) != len(pending_hashes):
            raise IncompleteParse(
                "duplicate identity in the requested input set: "
                f"{len(pending)} input(s) resolved to only "
                f"{len(pending_hashes)} unique file(s)"
            )

        # Phase 2 — parse + resolve the changed files. PRIMARY path (CONCEPT:EG-KG.compute.type-scope-resolved-call):
        # one ``index_repository`` round-trip both parses every file and resolves
        # cross-file calls type/scope-aware in Rust, yielding the symbols AND the
        # already-bound CALLS/INHERITS/REALIZES edges. Fallback (engine without the
        # resolver): per-file parse + Python name-only call resolution.
        struct_edges: list[EnrichmentEdge] = []
        call_edges: list[EnrichmentEdge] | None = None
        results: list[ExtractionResult] = []
        if self.index_fn is not None and pending:
            try:
                raw = [
                    (fp, src.encode("utf-8", "surrogatepass")) for fp, src in pending
                ]
                index = self.index_fn(raw)
            except Exception as exc:  # noqa: BLE001 — the RPC itself failed or is
                # unsupported by this engine build: degrade to the per-file parse
                # path, which independently re-verifies every file rather than
                # trusting anything from the failed call.
                logger.debug("index_repository call failed (%s); parse fallback", exc)
                results = []
            else:
                # A response WAS received: exact acknowledgement validation
                # (entities_from_index_result) is authoritative from here.
                # IncompleteParse — partial/unknown-identity/miscounted — MUST
                # propagate rather than be silently smoothed over by the parse
                # fallback below: a native response that answered but cannot be
                # trusted is a defect to surface, not mask.
                results, resolved = entities_from_index_result(index, pending_hashes)
                call_edges = [e for e in resolved if e.rel_type == "CALLS"]
                struct_edges = [e for e in resolved if e.rel_type != "CALLS"]
        if not results:
            if self.batch_parse_fn is not None and pending:
                results = extract_source_files(pending, self.batch_parse_fn)
            else:
                results = [
                    extract_source(fp, source, self.parse_fn) for fp, source in pending
                ]
        for res in results:
            self._hash_seen[res.file_path] = res.content_hash
            summary.files_parsed += 1

        all_code = [c for r in results for c in r.code]
        all_tests = [t for r in results for t in r.tests]

        # L0/structural: design-pattern tags (deterministic, no LLM).
        for c in all_code:
            c.patterns = detect_patterns(c)
            if c.patterns:
                summary.patterns_tagged += 1

        # Resolve the code→code CALLS edges ONCE: community detection clusters on
        # them and the write section below persists the same set. The resolver path
        # already produced them in Rust; only the fallback resolves names here.
        if call_edges is None:
            call_edges = resolve_call_edges(all_code)

        # Features: cluster the call graph via the engine's community detection.
        features = []
        if self.community_fn is not None:
            features = cluster_features(
                all_code,
                self.community_fn,
                self.min_feature_size,
                call_edges=call_edges,
            )

        # L2 semantic: capability cards (LLM, cached by ast_hash).
        cards_by_id: dict[str, CapabilityCard] = {}
        if self.llm_fn is not None:
            calls_by_id = {c.id: c.calls for c in all_code}
            for card in generate_symbol_cards(
                all_code, self.llm_fn, self.card_cache, calls_by_id
            ):
                cards_by_id[card.id] = card
                summary.cards_generated += 1

        # Batch all writes for this repo through one buffered backend: a big repo
        # is tens of thousands of nodes, and each per-node write is a socket
        # round-trip. The buffer flushes via the engine's bulk op (nodes before
        # edges). Reads (e.g. capability_provider) still hit the real backend. (#1)
        real_backend = self.backend
        self.backend = _BatchedBackend(real_backend, source_system=self.source_system)
        try:
            for c in all_code:
                self._write_code(c, cards_by_id.get(c.id))
                summary.code += 1
            for t in all_tests:
                if self._write_test(t):
                    summary.tests_needing_work += 1
                summary.tests += 1

            for e in resolve_covers(results):
                self._write_edge(e.source, e.target, e.rel_type)
                summary.covers_edges += 1
            for e in call_edges:
                self._write_edge(e.source, e.target, e.rel_type, e.props)
                summary.calls_edges += 1
            # Structural + similarity edges (INHERITS/REALIZES/SIMILAR_TO) from the
            # Rust resolver (CONCEPT:EG-KG.compute.type-scope-resolved-call/2.101).
            for e in struct_edges:
                self._write_edge(e.source, e.target, e.rel_type, e.props)
                if e.rel_type == "INHERITS":
                    summary.inherits_edges += 1
                elif e.rel_type == "REALIZES":
                    summary.realizes_struct_edges += 1
                elif e.rel_type == "SIMILAR_TO":
                    summary.similar_edges += 1

            for f in features:
                self._write_feature(f)
                for mid in f.member_ids:
                    self._write_edge(mid, f.id, "PART_OF_FEATURE")
                summary.features += 1

            # CONCEPT:AU-KG.enrichment.http-route-extraction — HTTP routes from handler decorators: Route nodes +
            # SERVES (handler→route), and the code↔ecosystem SERVED_BY link to the
            # deployed Service (best-effort name match), so OWL reasoning can chain
            # Code –serves→ Route –servedBy→ Service –deployedOn→ Node.
            route_nodes, serves_edges = extract_routes(all_code)
            for rn in route_nodes:
                self.backend.add_node(rn.id, label="Route", **rn.props)
                summary.routes += 1
            for e in serves_edges:
                self._write_edge(e.source, e.target, e.rel_type)
                summary.serves_edges += 1
            service_id = (
                resolve_service_id(service_hint, self._ecosystem_service_ids())
                if service_hint
                else ""
            )
            if route_nodes and service_id:
                for e in link_routes_to_service(route_nodes, service_id):
                    self._write_edge(e.source, e.target, e.rel_type)
                    summary.served_by_edges += 1

            # CONCEPT:AU-KG.enrichment.read-them-here-so — IaC Resources (Dockerfile/K8s/Terraform) + the
            # PROVISIONS link to the deployed Service, spanning code → infra.
            if iac_files:
                resource_nodes, _ = extract_iac(iac_files)
                for rn in resource_nodes:
                    self.backend.add_node(rn.id, label="Resource", **rn.props)
                    summary.resources += 1
                if service_id:
                    for e in link_resources_to_service(resource_nodes, service_id):
                        self._write_edge(e.source, e.target, e.rel_type)
                        summary.provisions_edges += 1

            # Code → capability: match features to BusinessCapability nodes
            # (LeanIX/Archi), mint provisional ones bottom-up, emit REALIZES edges,
            # and optionally push the minted capabilities back to EA tools (KG-2.8).
            if features and (
                self.capability_provider is not None
                or self.capability_registry is not None
                or self.mint_capabilities
            ):
                capabilities = (
                    self.capability_provider() if self.capability_provider else []
                )
                minted, realizes_edges = resolve_realizes(
                    features,
                    capabilities,
                    registry=self.capability_registry,
                    mint_missing=self.mint_capabilities,
                    embed_fn=self.realizes_embed_fn,
                )
                for cap in minted:
                    self._write_capability(cap)
                    summary.capabilities_minted += 1
                for e in realizes_edges:
                    self._write_edge(e.source, e.target, e.rel_type)
                    summary.realizes_edges += 1
                if minted and self.writeback_fn is not None:
                    result = self.writeback_fn(minted)
                    summary.capabilities_pushed = _writeback_count(result)
        finally:
            self.backend.flush()
            self.backend = real_backend

        return summary

    # ── writers (GraphBackend single interface) ──────────────────────────
    def _write_code(self, c: Any, card: CapabilityCard | None = None) -> None:
        self.backend.add_node(
            c.id,
            label="Code",
            name=c.name,
            qualname=c.qualname,
            kind=c.kind,
            language=c.language,
            file_path=c.file_path,
            line=c.line,
            ast_hash=c.ast_hash,
            patterns=",".join(c.patterns),
            is_abstract=c.is_abstract,
            summary=(card.summary if card else ""),
            responsibilities=(json.dumps(card.responsibilities) if card else "[]"),
        )

    def enrich_documents(
        self, paths: Iterable[Path | str]
    ) -> tuple[list[Concept], list[EnrichmentEdge], EnrichmentSummary]:
        """Extract Document + Concept nodes (+ MENTIONS) from documents.

        Requires ``llm_fn`` (concept extraction). Returns the concepts + edges so
        the caller can cross-link and distil. Hash-incremental by content_hash.

        BUG-059: this sibling of :meth:`enrich_files` used to write straight
        through ``self.backend`` (the real backend), never through the
        governed ``_BatchedBackend`` wrapper — so every Document/Concept/
        Insight/Fact/Framework/Playbook node from this path skipped
        ``stamp_ownership``/``stamp_classification`` entirely, unlike
        ``enrich_files``, which already swaps ``self.backend`` for a
        ``_BatchedBackend`` around its whole write section. There was no
        reason for the two paths to disagree, and ``enrich_files`` was the
        one that was right — apply the SAME swap here.
        """
        summary = EnrichmentSummary()
        all_concepts: dict[str, Concept] = {}
        all_edges: list[EnrichmentEdge] = []
        if self.llm_fn is None:
            logger.warning("enrich_documents needs llm_fn; skipping concept extraction")
            return [], [], summary

        real_backend = self.backend
        self.backend = _BatchedBackend(real_backend, source_system=self.source_system)
        try:
            for p in paths:
                p = str(p)
                summary.files_seen += 1
                text = read_document_text(p)
                if not text.strip():
                    continue
                doc, concepts, edges = extract_document(p, text, self.llm_fn)
                if self._hash_seen.get(p) == doc.content_hash:
                    summary.files_skipped_unchanged += 1
                    continue
                self._hash_seen[p] = doc.content_hash
                summary.files_parsed += 1
                self.backend.add_node(
                    doc.id,
                    label="Document",
                    name=doc.title,
                    doc_type=doc.doc_type,
                    file_path=doc.file_path,
                    ast_hash=doc.content_hash,
                    metadata=json.dumps(doc.metadata)[:4000],
                )
                summary.documents += 1
                # Distil reusable operating intelligence (CONCEPT:EG-KG.storage.nonblocking-checkpoint): turn the
                # document/call into Insight/Fact/Framework/Playbook nodes.
                try:
                    intel_nodes, intel_edges = extract_intelligence(
                        text,
                        doc.id,
                        self.llm_fn,
                        source_type=doc.doc_type,
                        title=doc.title,
                    )
                    for node in intel_nodes:
                        self._write_intelligence(node)
                        summary.intelligence_nodes += 1
                    all_edges.extend(intel_edges)
                except Exception as exc:  # pragma: no cover - enrichment best-effort  # noqa: BLE001 — the Document node itself was already committed via self.backend.add_node above; a failed intelligence-extraction pass just means fewer Insight/Fact/Framework/Playbook nodes for this document, not a lost or falsely-marked-processed document
                    logger.debug("intelligence extraction skipped for %s: %s", p, exc)
                for c in concepts:
                    # Concepts are canonical by id; merge source_ids across docs.
                    existing = all_concepts.get(c.id)
                    if existing:
                        existing.source_ids = sorted(
                            set(existing.source_ids) | set(c.source_ids)
                        )
                    else:
                        all_concepts[c.id] = c
                all_edges.extend(edges)

            for c in all_concepts.values():
                self.backend.add_node(
                    c.id,
                    label="Concept",
                    name=c.name,
                    kind=c.kind,
                    summary=c.summary,
                    source_ids=json.dumps(c.source_ids),
                )
                summary.concepts += 1
            for e in all_edges:
                self._write_edge(e.source, e.target, e.rel_type)
                summary.mentions_edges += 1
        finally:
            self.backend.flush()
            self.backend = real_backend

        return list(all_concepts.values()), all_edges, summary

    def _write_intelligence(self, node: Any) -> None:
        """Persist an Insight/Fact/Framework/Playbook node (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

        The node type label is the model class name (``Insight``/...); list
        fields are JSON-serialised so they survive scalar property storage.
        """
        data = node.model_dump()
        node_id = data.pop("id")
        props = {
            k: (json.dumps(v) if isinstance(v, list) else v)
            for k, v in data.items()
            if v is not None
        }
        self.backend.add_node(node_id, label=type(node).__name__, **props)

    def _write_feature(self, f: Any) -> None:
        self.backend.add_node(
            f.id,
            label="Feature",
            name=f.name,
            summary=f.summary,
            size=f.size,
            patterns=",".join(f.patterns),
            member_ids=json.dumps(f.member_ids),
        )

    def _write_capability(self, cap: GraphNode) -> None:
        """Persist a (provisional, code-derived) BusinessCapability node."""
        props = {k: v for k, v in cap.props.items() if v is not None}
        self.backend.add_node(cap.id, label=cap.type, **props)

    def _write_test(self, t: Any) -> bool:
        issues = classify_test(t, self.thresholds)
        needs_work = bool(issues)
        self.backend.add_node(
            t.id,
            label="Test",
            name=t.name,
            file_path=t.file_path,
            line=t.line,
            ast_hash=t.ast_hash,
            assert_count=t.assert_count,
            raises_count=t.raises_count,
            mock_count=t.mock_count,
            fixture_count=t.fixture_count,
            marks=",".join(t.marks),
            is_skipped=t.is_skipped,
            needs_work=needs_work,
            issues=json.dumps([i.model_dump() for i in issues]),
        )
        return needs_work

    def _write_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        add_edge = getattr(self.backend, "add_edge", None)
        if callable(add_edge):
            add_edge(source, target, rel_type=rel_type, **(props or {}))

    def _ecosystem_service_ids(self) -> set[str]:
        """Deployed ecosystem ``Service`` node ids, for code↔service `servedBy`
        linking (CONCEPT:AU-KG.enrichment.http-route-extraction). Best-effort: empty when the backend has no
        query path or no services — we never invent a topology link."""
        execute = getattr(self.backend, "execute", None)
        if not callable(execute):
            return set()
        try:
            rows = execute("MATCH (s:Service) RETURN s.id AS id", {})
        except Exception:  # noqa: BLE001 — best-effort; no services -> no servedBy
            return set()
        return {str(r["id"]) for r in (rows or []) if r.get("id")}


def _writeback_count(result: Any) -> int:
    """Total capabilities pushed by a writeback result (tolerant of shape)."""
    if result is None:
        return 0
    pushed = getattr(result, "archi_pushed", 0) + getattr(result, "leanix_pushed", 0)
    return int(pushed)


def make_parse_fn(graph_compute: Any) -> ParseFn:
    """Adapt a GraphComputeEngine into the extractor's ParseFn."""
    return lambda file_path, source: graph_compute.parse_file(file_path, source)


def make_batch_parse_fn(graph_compute: Any) -> BatchParseFn:
    """Adapt a GraphComputeEngine into the mandatory batched ParseFn.

    Files are sent in chunks of ``KG_PARSE_BATCH`` (default 512) so a first ingest
    of a large repo makes few round-trips: the engine's ``parse_files`` parses a
    whole chunk in parallel across cores (rayon), and request/response is serialized
    on one connection, so a bigger chunk = bigger parallel batch + fewer round-trips
    (the dominant parse cost). (CONCEPT:EG-KG.compute.graph-compute-engine)
    """

    try:
        chunk = max(1, setting("KG_PARSE_BATCH", 512))
    except ValueError:
        chunk = 512

    def _fn(files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(0, len(files), chunk):
            out.extend(graph_compute.parse_files(files[i : i + chunk]))
        return out

    return _fn


def make_index_fn(graph_compute: Any) -> IndexFn:
    """Adapt a GraphComputeEngine into the mandatory cross-file resolver entry point.

    The whole batch is one resolution scope, so it ships in a SINGLE round-trip:
    the engine parses (rayon) and resolves cross-file calls type/scope-aware over
    the whole set, returning one merged ``IndexResult``. (CONCEPT:EG-KG.compute.type-scope-resolved-call)
    """
    if not callable(getattr(graph_compute, "index_repository", None)):
        raise RuntimeError("current engine is missing mandatory IndexRepository")
    return lambda files: graph_compute.index_repository(files)
