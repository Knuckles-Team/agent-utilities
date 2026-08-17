"""Code/test entity extraction from the epistemic-graph Rust AST (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

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


class IncompleteParse(RuntimeError):
    """A native parse/index response did not exactly acknowledge every
    requested input, or a per-file parse could not be verified.

    Raised instead of ever converting a partial, malformed, unknown-identity,
    duplicate-identity, or genuinely failed parse into an indistinguishable
    empty *successful* :class:`~agent_utilities.knowledge_graph.enrichment.models.ExtractionResult`.
    A caller (:class:`~agent_utilities.knowledge_graph.enrichment.pipeline.EnrichmentPipeline`)
    MUST NOT persist a per-file content hash or advance a repository watermark
    for the batch this was raised from — "no entities found" (a verified,
    count-covered empty parse) is a distinct outcome from "omitted"/"failed"/
    "malformed", and only the former is a safe basis for a watermark advance.
    """


# A parse function: (file_path, source_bytes) -> Rust ParseResult dict
ParseFn = Callable[[str, bytes], dict[str, Any]]
# A batched parse function: [(file_path, source_bytes), ...] -> [ParseResult dict, ...]
# (one result per input file, in input order). (CONCEPT:EG-KG.compute.graph-compute-engine)
BatchParseFn = Callable[[list[tuple[str, bytes]]], list[dict[str, Any]]]
# An index function: [(file_path, source_bytes), ...] -> one merged IndexResult dict
# (parse + cross-file type/scope resolution in a SINGLE round-trip). (CONCEPT:EG-KG.compute.type-scope-resolved-call)
IndexFn = Callable[[list[tuple[str, bytes]]], dict[str, Any]]

# Engine resolved edge types → enrichment rel types (CONCEPT:EG-KG.compute.type-scope-resolved-call/2.101).
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
    class decorators are comma-joined. Detect which (CONCEPT:AU-KG.compute.http-route-graph)."""
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
            # overhead on the ingest hot path. (CONCEPT:EG-KG.storage.nonblocking-checkpoint, #3)
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
    ``CALLS``/``INHERITS``/``REALIZES`` edges (CONCEPT:EG-KG.compute.type-scope-resolved-call).

    A single ``IndexRepository`` round-trip both parses every file and resolves
    cross-file calls type/scope-aware in Rust, so the symbols come from the merged
    ``nodes`` (grouped by file) and the call graph from the merged ``edges`` —
    bound to definitions, not name-matched in Python. ``CALLS`` stays code→code
    (test coverage is the separate name-resolved ``COVERS`` edge); resolved-edge
    properties (``strategy``/``confidence``) ride on each edge.

    Exact acknowledgement: ``content_hashes`` is also the REQUEST set (its keys
    are every input file's logical identity). This function raises
    :class:`IncompleteParse` — leaving the caller nothing to persist — when the
    response cannot be trusted to cover exactly that set:

    * ``index["files_parsed"]`` does not equal the number of requested files
      (a truncated/miscounted/stale response).
    * a ``SYMBOL`` node names a file that was never requested (unknown
      identity).

    A requested file with **no** ``SYMBOL`` node in the response is *not*
    silently dropped: once ``files_parsed`` has confirmed the engine actually
    processed the full requested set, that file's absence can only mean a
    verified, genuinely empty parse (CONCEPT:AU-KG.ingest.exact-parser-acknowledgement)
    — it is emitted as an explicit zero-entity :class:`ExtractionResult`
    (its ``content_hash`` locally computed, never trusted from the wire), so
    the caller's hash/watermark bookkeeping accounts for every requested file
    exactly once, distinguishing "parsed, found nothing" from "never
    acknowledged".
    """
    requested = set(content_hashes)
    files_parsed: Any = index.get("files_parsed")
    try:
        files_parsed_int = int(files_parsed)
    except (TypeError, ValueError) as exc:
        raise IncompleteParse(
            f"index result files_parsed={files_parsed!r} is not an integer"
        ) from exc
    if files_parsed_int != len(requested):
        raise IncompleteParse(
            f"index result files_parsed={files_parsed_int} does not match the "
            f"{len(requested)} requested input(s)"
        )

    nodes = index.get("nodes", []) or []
    by_file: dict[str, list[dict[str, Any]]] = {}
    engine_to_entity: dict[str, str] = {}
    entity_kind: dict[str, str] = {}
    for node in nodes:
        if node.get("node_type") != "SYMBOL":
            continue
        props = node.get("properties", {}) or {}
        fp = props.get("file_path", "")
        if fp not in requested:
            raise IncompleteParse(f"index result references unrequested file {fp!r}")
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
    # Verified-empty files: requested, count-covered, but zero SYMBOL nodes.
    for fp in requested - set(by_file):
        results.append(
            ExtractionResult(file_path=fp, content_hash=content_hashes.get(fp, ""))
        )

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

    A ``parse_fn`` failure is REJECTED (raises :class:`IncompleteParse`), never
    silently converted into an empty *successful* result — a genuinely empty
    file and a failed parse must stay distinguishable to the caller's
    hash/watermark bookkeeping (CONCEPT:AU-KG.ingest.exact-parser-acknowledgement).
    """
    raw = source.encode("utf-8", "surrogatepass")
    content_hash = hashlib.sha256(raw).hexdigest()
    try:
        parsed = parse_fn(file_path, raw)
    except Exception as exc:
        raise IncompleteParse(f"parse failed for {file_path!r}: {exc}") from exc
    return entities_from_parse_result(file_path, content_hash, parsed or {})


def extract_source_files(
    files: list[tuple[str, str]], batch_parse_fn: BatchParseFn
) -> list[ExtractionResult]:
    """Batch variant of :func:`extract_source` — parse N files in ONE RPC.

    ``files`` is ``[(file_path, source_text), ...]``; ``batch_parse_fn`` takes
    ``[(file_path, source_bytes), ...]`` and returns one ParseResult dict per file
    in order. Returns one :class:`ExtractionResult` per input file, in input
    order. Raises :class:`IncompleteParse` — rather than degrading a
    failed/missing slot into an indistinguishable empty successful result — when
    the batch call itself fails, or when the response does not contain exactly
    one result per requested input. (CONCEPT:EG-KG.compute.graph-compute-engine,
    CONCEPT:AU-KG.ingest.exact-parser-acknowledgement)
    """
    raw = [(fp, src.encode("utf-8", "surrogatepass")) for fp, src in files]
    hashes = [hashlib.sha256(b).hexdigest() for _, b in raw]
    try:
        parsed_list = batch_parse_fn(raw)
    except Exception as exc:
        raise IncompleteParse(
            f"batch parse failed for {len(files)} file(s): {exc}"
        ) from exc
    if len(parsed_list) != len(files):
        raise IncompleteParse(
            f"batch parse returned {len(parsed_list)} result(s) for "
            f"{len(files)} requested file(s)"
        )
    return [
        entities_from_parse_result(fp, hashes[i], parsed_list[i] or {})
        for i, (fp, _src) in enumerate(files)
    ]


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
