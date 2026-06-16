"""Code/test entity extraction from the epistemic-graph Rust AST (CONCEPT:KG-2.8).

The AST + test-quality metrics are computed in the **Rust compute layer**
(`epistemic-graph` ``ParseFile`` RPC → ``parser::tree_sitter``), not in Python.
This module only *maps* that native output into typed entities and resolves
COVERS edges. No Python AST walking — the Rust engine is the compute layer.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

from ..models import CodeEntity, EnrichmentEdge, ExtractionResult, TestEntity

# A parse function: (file_path, source_bytes) -> Rust ParseResult dict
ParseFn = Callable[[str, bytes], dict[str, Any]]
# A batched parse function: [(file_path, source_bytes), ...] -> [ParseResult dict, ...]
# (one result per input file, in input order). (CONCEPT:KG-2.16)
BatchParseFn = Callable[[list[tuple[str, bytes]]], list[dict[str, Any]]]
# An index function: [(file_path, source_bytes), ...] -> one merged IndexResult dict
# (parse + cross-file type/scope resolution in a SINGLE round-trip). (CONCEPT:KG-2.100)
IndexFn = Callable[[list[tuple[str, bytes]]], dict[str, Any]]

# Engine resolved edge types → enrichment rel types (CONCEPT:KG-2.100/2.101).
_RESOLVED_EDGE_RELS = {
    "calls": "CALLS",
    "inherits": "INHERITS",
    "realizes": "REALIZES",
    "similar_to": "SIMILAR_TO",
}


def _is_test_file(file_path: str) -> bool:
    """A pytest test lives in a test file — not just any ``test_*`` function.

    Avoids false positives like a production helper named ``test_connection`` or
    ``tests_needing_work``.
    """
    base = os.path.basename(file_path)
    if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
        return True
    norm = file_path.replace("\\", "/")
    return "/tests/" in norm or norm.startswith("tests/")


def _int(props: dict[str, Any], key: str) -> int:
    try:
        return int(props.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _bool(props: dict[str, Any], key: str) -> bool:
    return str(props.get(key, "")).lower() == "true"


def _split_decorators(raw: str) -> list[str]:
    """Split the parser's ``decorators`` property. Function route decorators embed
    commas (``app.route("/x", methods=[...])``) so they are US-separated (\\x1f);
    class decorators are comma-joined. Detect which (CONCEPT:KG-2.102)."""
    raw = raw or ""
    sep = "\x1f" if "\x1f" in raw else ","
    return [d for d in raw.split(sep) if d]


def entities_from_parse_result(
    file_path: str, content_hash: str, parsed: dict[str, Any]
) -> ExtractionResult:
    """Map a Rust ``ParseFile`` result into code/test entities."""
    result = ExtractionResult(file_path=file_path, content_hash=content_hash)
    for node in parsed.get("nodes", []):
        props = node.get("properties", {}) or {}
        sym_type = props.get("symbol_type")
        name = props.get("name", "")
        line = _int(props, "line")
        ast_hash = props.get("ast_hash", "")
        # Stable identity per (file, symbol) — NOT the Rust content-hash id, which
        # collides for identically-bodied symbols (e.g. ``def test_x(): pass``).
        # ast_hash is kept as a property for incremental change detection.

        if (
            sym_type == "Function"
            and _bool(props, "is_test")
            and _is_test_file(file_path)
        ):
            marks = [m for m in (props.get("marks", "") or "").split(",") if m]
            calls = [c for c in (props.get("calls", "") or "").split(",") if c]
            # ``model_construct`` skips Pydantic validation — the values already
            # come typed from our own Rust parser (+ ``_int``/``_bool`` coercion),
            # so validating tens of thousands of entities per big repo is pure
            # overhead on the ingest hot path. (CONCEPT:KG-2.8, #3)
            result.tests.append(
                TestEntity.model_construct(
                    id=f"test:{file_path}::{name}",
                    name=name,
                    qualname=name,
                    file_path=file_path,
                    line=line,
                    ast_hash=ast_hash,
                    assert_count=_int(props, "assert_count"),
                    raises_count=_int(props, "raises_count"),
                    mock_count=_int(props, "mock_count"),
                    fixture_count=_int(props, "fixture_count"),
                    marks=marks,
                    is_skipped=_bool(props, "is_skipped"),
                    calls=calls,
                )
            )
        elif sym_type in ("Function", "Class"):
            is_class = sym_type == "Class"
            # ``kind_detail`` carries the precise kind from the Rust parser
            # (interface/struct/enum/trait/method/constructor/...); fall back to
            # the coarse class/function bucket for older engine builds.
            kind = props.get("kind_detail") or ("class" if is_class else "function")
            result.code.append(
                CodeEntity.model_construct(
                    id=f"code:{file_path}::{name}",
                    name=name,
                    qualname=name,
                    kind=kind,
                    language=props.get("language", ""),
                    file_path=file_path,
                    line=line,
                    ast_hash=ast_hash,
                    is_test=False,
                    calls=[c for c in (props.get("calls", "") or "").split(",") if c],
                    bases=[b for b in (props.get("bases", "") or "").split(",") if b],
                    methods=[
                        m for m in (props.get("methods", "") or "").split(",") if m
                    ],
                    decorators=_split_decorators(props.get("decorators", "")),
                    is_abstract=_bool(props, "is_abstract"),
                )
            )
    return result


def _entity_id_for(props: dict[str, Any], file_path: str) -> tuple[str, str] | None:
    """The (entity_id, kind) a parsed SYMBOL maps to, mirroring
    :func:`entities_from_parse_result`. ``kind`` is ``"code"`` or ``"test"``;
    ``None`` for a node that yields no entity."""
    sym_type = props.get("symbol_type")
    name = props.get("name", "")
    if sym_type == "Function" and _bool(props, "is_test") and _is_test_file(file_path):
        return f"test:{file_path}::{name}", "test"
    if sym_type in ("Function", "Class"):
        return f"code:{file_path}::{name}", "code"
    return None


def entities_from_index_result(
    index: dict[str, Any], content_hashes: dict[str, str]
) -> tuple[list[ExtractionResult], list[EnrichmentEdge]]:
    """Map one engine ``IndexResult`` into per-file entities AND already-resolved
    ``CALLS``/``INHERITS``/``REALIZES`` edges (CONCEPT:KG-2.100).

    A single ``IndexRepository`` round-trip both parses every file and resolves
    cross-file calls type/scope-aware in Rust, so the symbols come from the merged
    ``nodes`` (grouped by file) and the call graph from the merged ``edges`` —
    bound to definitions, not name-matched in Python. ``CALLS`` stays code→code
    (test coverage is the separate name-resolved ``COVERS`` edge); resolved-edge
    properties (``strategy``/``confidence``) ride on each edge.
    """
    nodes = index.get("nodes", []) or []
    by_file: dict[str, list[dict[str, Any]]] = {}
    engine_to_entity: dict[str, str] = {}
    entity_kind: dict[str, str] = {}
    for node in nodes:
        if node.get("node_type") != "SYMBOL":
            continue
        props = node.get("properties", {}) or {}
        fp = props.get("file_path", "")
        by_file.setdefault(fp, []).append(node)
        mapped = _entity_id_for(props, fp)
        if mapped is not None:
            eid, kind = mapped
            engine_to_entity[str(node.get("node_id", ""))] = eid
            entity_kind[eid] = kind

    results = [
        entities_from_parse_result(
            fp, content_hashes.get(fp, ""), {"nodes": file_nodes}
        )
        for fp, file_nodes in by_file.items()
    ]

    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in index.get("edges", []) or []:
        rel = _RESOLVED_EDGE_RELS.get(edge.get("edge_type", ""))
        if rel is None:
            continue
        src = engine_to_entity.get(str(edge.get("source", "")))
        tgt = engine_to_entity.get(str(edge.get("target", "")))
        if not src or not tgt or src == tgt:
            continue
        if rel == "CALLS" and (
            entity_kind.get(src) != "code" or entity_kind.get(tgt) != "code"
        ):
            continue
        key = (src, tgt, rel)
        if key in seen:
            continue
        seen.add(key)
        props = {
            k: v
            for k, v in (edge.get("properties") or {}).items()
            if k in ("strategy", "confidence", "score")
        }
        edges.append(EnrichmentEdge(source=src, target=tgt, rel_type=rel, props=props))
    return results, edges


def extract_source(file_path: str, source: str, parse_fn: ParseFn) -> ExtractionResult:
    """Parse one source file (any engine-supported language) and map to entities.

    The Rust engine dispatches on file extension, so Python/JS/TS/Go/Rust/Java/
    C/C++/C# all flow through here; the ``language`` is carried on each entity.
    """
    raw = source.encode("utf-8", "surrogatepass")
    content_hash = hashlib.sha256(raw).hexdigest()
    try:
        parsed = parse_fn(file_path, raw)
    except Exception:
        return ExtractionResult(file_path=file_path, content_hash=content_hash)
    return entities_from_parse_result(file_path, content_hash, parsed or {})


def extract_source_files(
    files: list[tuple[str, str]], batch_parse_fn: BatchParseFn
) -> list[ExtractionResult]:
    """Batch variant of :func:`extract_source` — parse N files in ONE RPC.

    ``files`` is ``[(file_path, source_text), ...]``; ``batch_parse_fn`` takes
    ``[(file_path, source_bytes), ...]`` and returns one ParseResult dict per file
    in order. Returns one :class:`ExtractionResult` per input file, in input
    order. A file whose parse failed or is missing from the response degrades to
    an empty result (its ``content_hash`` is still recorded), mirroring the
    per-file fault tolerance of :func:`extract_source`. (CONCEPT:KG-2.16)
    """
    raw = [(fp, src.encode("utf-8", "surrogatepass")) for fp, src in files]
    hashes = [hashlib.sha256(b).hexdigest() for _, b in raw]
    try:
        parsed_list = batch_parse_fn(raw)
    except Exception:
        parsed_list = []
    out: list[ExtractionResult] = []
    for i, (fp, _src) in enumerate(files):
        parsed = parsed_list[i] if i < len(parsed_list) else None
        out.append(entities_from_parse_result(fp, hashes[i], parsed or {}))
    return out


def resolve_covers(results: list[ExtractionResult]) -> list[EnrichmentEdge]:
    """Resolve TESTS/COVERS edges by matching test call names to code entities.

    A test ``COVERS`` an application function/class when it calls something with
    that name. Name-based resolution across the whole ingest set (Phase 1); a
    later phase can tighten this with import/scope resolution.
    """
    by_name: dict[str, list[str]] = {}
    for r in results:
        for c in r.code:
            if not c.is_test:
                by_name.setdefault(c.name, []).append(c.id)

    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str]] = set()
    for r in results:
        for t in r.tests:
            for callee in set(t.calls):
                for code_id in by_name.get(callee, []):
                    key = (t.id, code_id)
                    if key not in seen:
                        seen.add(key)
                        edges.append(
                            EnrichmentEdge(
                                source=t.id, target=code_id, rel_type="COVERS"
                            )
                        )
    return edges
