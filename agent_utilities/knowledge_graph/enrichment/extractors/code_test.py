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
            result.tests.append(
                TestEntity(
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
                CodeEntity(
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
                    decorators=[
                        d for d in (props.get("decorators", "") or "").split(",") if d
                    ],
                    is_abstract=_bool(props, "is_abstract"),
                )
            )
    return result


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
