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

from .cards import CapabilityCard, LLMFn, generate_symbol_cards
from .classify import TestThresholds, classify_test
from .extractors.code_test import (
    BatchParseFn,
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
        # Multiplexed pool fan-out (CONCEPT:AU-KG.compute.when-exposes): when the engine exposes the
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
        props: dict[str, Any] = {"label": label, **properties}
        # Stamp the source-provenance contract (source_system + domain) on every code
        # node at the one write chokepoint, so partition routing sends it to the right
        # named graph. A caller-set source_system still wins (stamp_source setdefaults).
        if self._source_system:
            from .provenance import stamp_source

            stamp_source(props, self._source_system)
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
        self._edges.append(
            {
                "op": "add_edge",
                "source": source,
                "target": target,
                "properties": {"rel_type": rel_type, **properties},
            }
        )

    def _submit_bulk(self, ops: list[dict[str, Any]]) -> bool:
        """Send one independent ``ops`` flush through the fastest available path.

        Order of preference (CONCEPT:AU-KG.compute.when-exposes → KG-2.16): the pooled concurrent
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
            files, service_hint=Path(target_path).name, iac_files=iac
        )

    def enrich_files(
        self,
        files: Iterable[Path],
        service_hint: str = "",
        iac_files: list[tuple[str, str]] | None = None,
    ) -> EnrichmentSummary:
        summary = EnrichmentSummary()

        # Phase 1 — pre-hash filter (CONCEPT:EG-KG.storage.nonblocking-checkpoint): hash the raw bytes BEFORE
        # parsing so an unchanged file costs one local sha256, not a Rust-engine
        # parse round-trip. The hash is byte-identical to ``ExtractionResult.
        # content_hash`` (same ``surrogatepass`` encoding), so the skip is exact.
        pending: list[tuple[str, str]] = []  # (file_path, source_text)
        pending_hashes: dict[str, str] = {}  # file_path -> content_hash
        for fp in files:
            summary.files_seen += 1
            try:
                source = Path(fp).read_text(encoding="utf-8", errors="surrogatepass")
            except (OSError, UnicodeDecodeError):
                continue
            content_hash = hashlib.sha256(
                source.encode("utf-8", "surrogatepass")
            ).hexdigest()
            if self._hash_seen.get(str(fp)) == content_hash:
                summary.files_skipped_unchanged += 1
                continue
            pending.append((str(fp), source))
            pending_hashes[str(fp)] = content_hash

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
                results, resolved = entities_from_index_result(index, pending_hashes)
                call_edges = [e for e in resolved if e.rel_type == "CALLS"]
                struct_edges = [e for e in resolved if e.rel_type != "CALLS"]
            except Exception as exc:  # noqa: BLE001 — degrade to the parse path
                logger.debug(
                    "index_repository resolve failed (%s); parse fallback", exc
                )
                results = []
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
                self.backend.add_node(rn.id, type="Route", **rn.props)
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
                    self.backend.add_node(rn.id, type="Resource", **rn.props)
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
            type="Code",
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
        """
        summary = EnrichmentSummary()
        all_concepts: dict[str, Concept] = {}
        all_edges: list[EnrichmentEdge] = []
        if self.llm_fn is None:
            logger.warning("enrich_documents needs llm_fn; skipping concept extraction")
            return [], [], summary

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
                type="Document",
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
            except Exception as exc:  # pragma: no cover - enrichment best-effort
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
                type="Concept",
                name=c.name,
                kind=c.kind,
                summary=c.summary,
                source_ids=json.dumps(c.source_ids),
            )
            summary.concepts += 1
        for e in all_edges:
            self._write_edge(e.source, e.target, e.rel_type)
            summary.mentions_edges += 1

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
        self.backend.add_node(node_id, type=type(node).__name__, **props)

    def _write_feature(self, f: Any) -> None:
        self.backend.add_node(
            f.id,
            type="Feature",
            name=f.name,
            summary=f.summary,
            size=f.size,
            patterns=",".join(f.patterns),
            member_ids=json.dumps(f.member_ids),
        )

    def _write_capability(self, cap: GraphNode) -> None:
        """Persist a (provisional, code-derived) BusinessCapability node."""
        props = {k: v for k, v in cap.props.items() if v is not None}
        self.backend.add_node(cap.id, type=cap.type, **props)

    def _write_test(self, t: Any) -> bool:
        issues = classify_test(t, self.thresholds)
        needs_work = bool(issues)
        self.backend.add_node(
            t.id,
            type="Test",
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


def make_batch_parse_fn(graph_compute: Any) -> BatchParseFn | None:
    """Adapt a GraphComputeEngine into a batched ParseFn — or ``None`` if the
    engine doesn't support the ``ParseFiles`` op (caller falls back to per-file).

    Files are sent in chunks of ``KG_PARSE_BATCH`` (default 512) so a first ingest
    of a large repo makes few round-trips: the engine's ``parse_files`` parses a
    whole chunk in parallel across cores (rayon), and request/response is serialized
    on one connection, so a bigger chunk = bigger parallel batch + fewer round-trips
    (the dominant parse cost). (CONCEPT:EG-KG.compute.graph-compute-engine)
    """

    try:
        if not getattr(graph_compute, "supports_batch_parse", False):
            return None
    except Exception:  # noqa: BLE001
        return None
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


def make_index_fn(graph_compute: Any) -> IndexFn | None:
    """Adapt a GraphComputeEngine into the cross-file resolver entry point — or
    ``None`` if the engine doesn't advertise ``IndexRepository`` (caller falls
    back to parse + Python name-only call resolution).

    The whole batch is one resolution scope, so it ships in a SINGLE round-trip:
    the engine parses (rayon) and resolves cross-file calls type/scope-aware over
    the whole set, returning one merged ``IndexResult``. (CONCEPT:EG-KG.compute.type-scope-resolved-call)
    """
    try:
        if not getattr(graph_compute, "supports_index_repository", False):
            return None
    except Exception:  # noqa: BLE001
        return None
    return lambda files: graph_compute.index_repository(files)
