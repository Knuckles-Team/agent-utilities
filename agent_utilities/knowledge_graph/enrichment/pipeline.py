"""In-process KG enrichment pipeline (CONCEPT:KG-2.8 Phase 1).

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

from .cards import CapabilityCard, LLMFn, generate_symbol_cards
from .classify import TestThresholds, classify_test
from .extractors.code_test import (
    BatchParseFn,
    ParseFn,
    extract_python,
    extract_python_files,
    resolve_covers,
)
from .extractors.document import (
    extract_document,
    extract_intelligence,
    read_document_text,
)
from .features import CommunityFn, cluster_features, resolve_call_edges
from .models import Concept, EnrichmentEdge, GraphNode
from .patterns import detect_patterns
from .realizes import EmbedFn, resolve_realizes

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
    "target",
    "site",
}


class EnrichmentSummary(BaseModel):
    files_seen: int = 0
    files_parsed: int = 0
    files_skipped_unchanged: int = 0
    tests: int = 0
    code: int = 0
    covers_edges: int = 0
    calls_edges: int = 0
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


def discover_python_files(root: str | Path) -> list[Path]:
    """Find Python files under root, skipping vendored/build dirs."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


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
    ) -> None:
        self.backend = backend
        self.parse_fn = parse_fn
        # Optional batched parse (one RPC for N files). When set, changed files
        # are parsed in a single round-trip instead of per-file. (CONCEPT:KG-2.16)
        self.batch_parse_fn = batch_parse_fn
        self.thresholds = thresholds or TestThresholds()
        self._hash_seen = hash_seen if hash_seen is not None else {}
        self.llm_fn = llm_fn
        self.community_fn = community_fn
        self.card_cache = card_cache if card_cache is not None else {}
        self.min_feature_size = min_feature_size
        # Code → capability (REALIZES) resolution (CONCEPT:KG-2.8).
        self.capability_provider = capability_provider
        self.capability_registry = capability_registry
        self.mint_capabilities = mint_capabilities
        self.realizes_embed_fn = realizes_embed_fn
        self.writeback_fn = writeback_fn

    def enrich(self, target_path: str | Path) -> EnrichmentSummary:
        files = discover_python_files(target_path)
        return self.enrich_files(files)

    def enrich_files(self, files: Iterable[Path]) -> EnrichmentSummary:
        summary = EnrichmentSummary()

        # Phase 1 — pre-hash filter (CONCEPT:KG-2.8): hash the raw bytes BEFORE
        # parsing so an unchanged file costs one local sha256, not a Rust-engine
        # parse round-trip. The hash is byte-identical to ``ExtractionResult.
        # content_hash`` (same ``surrogatepass`` encoding), so the skip is exact.
        pending: list[tuple[str, str]] = []  # (file_path, source_text)
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

        # Phase 2 — parse the changed files. One batched RPC when the engine
        # supports it (CONCEPT:KG-2.16), else the per-file path. Both yield the
        # same per-file ``ExtractionResult`` list.
        if self.batch_parse_fn is not None and pending:
            results = extract_python_files(pending, self.batch_parse_fn)
        else:
            results = [
                extract_python(fp, source, self.parse_fn) for fp, source in pending
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

        # Features: cluster the call graph via the engine's community detection.
        features = []
        if self.community_fn is not None:
            features = cluster_features(
                all_code, self.community_fn, self.min_feature_size
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
        for e in resolve_call_edges(all_code):
            self._write_edge(e.source, e.target, e.rel_type)
            summary.calls_edges += 1

        for f in features:
            self._write_feature(f)
            for mid in f.member_ids:
                self._write_edge(mid, f.id, "PART_OF_FEATURE")
            summary.features += 1

        # Code → capability: match features to BusinessCapability nodes (LeanIX/
        # Archi), mint provisional ones bottom-up, emit REALIZES edges, and
        # optionally push the minted capabilities back to EA tools (KG-2.8).
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

        return summary

    # ── writers (GraphBackend single interface) ──────────────────────────
    def _write_code(self, c: Any, card: CapabilityCard | None = None) -> None:
        self.backend.add_node(
            c.id,
            type="Code",
            name=c.name,
            qualname=c.qualname,
            kind=c.kind,
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
            # Distil reusable operating intelligence (CONCEPT:KG-2.8): turn the
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
        """Persist an Insight/Fact/Framework/Playbook node (CONCEPT:KG-2.8).

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

    def _write_edge(self, source: str, target: str, rel_type: str) -> None:
        add_edge = getattr(self.backend, "add_edge", None)
        if callable(add_edge):
            add_edge(source, target, rel_type=rel_type)


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

    Files are sent in chunks of ``KG_PARSE_BATCH`` (default 128) so a first ingest
    of a large repo never builds one giant request. (CONCEPT:KG-2.16)
    """
    import os

    try:
        if not getattr(graph_compute, "supports_batch_parse", False):
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        chunk = max(1, int(os.environ.get("KG_PARSE_BATCH", "128")))
    except ValueError:
        chunk = 128

    def _fn(files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(0, len(files), chunk):
            out.extend(graph_compute.parse_files(files[i : i + chunk]))
        return out

    return _fn
