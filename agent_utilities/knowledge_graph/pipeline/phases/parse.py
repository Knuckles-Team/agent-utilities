"""CONCEPT:AU-KG.query.object-graph-mapper / KG-2.106 — code AST parsing delegated to the epistemic-graph engine.

The Rust engine (``eg-compute``, native tree-sitter, 14 grammars) extracts the SAME
registry-graph schema this phase used to build by hand with Python tree-sitter —
``file:<path>`` + ``symbol:<sha256>`` nodes joined by ``IMPLEMENTS``, plus ``calls_raw`` /
``depends_on_raw`` edges (and richer call-graph + MinHash similarity signals across many
more languages). Delegating here is what lets agent-utilities drop the Python
``tree-sitter*`` wheels entirely: ``epistemic_graph.parser.RustASTParser`` (GOC-73:
shipped via the ``agent-utilities[graphos]`` extra, not a base dependency) is the ONE
code-parsing implementation. When the engine socket is unavailable, ``RustASTParser``
transparently falls back to Python's stdlib ``ast`` (Python sources only) — but the
``epistemic_graph`` package itself, and therefore this phase, still requires
``[graphos]`` to be installed. Markdown CONCEPT/SDD extraction is regex-based (it
never used tree-sitter) and stays here.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)

# Code extensions the engine parser ingests (mirrors eg-compute SUPPORTED_EXTENSIONS,
# incl. the ast-extended grammar tier). Markdown is handled separately below.
_CODE_EXTENSIONS = {
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
    # SQL DDL → database ontology (CONCEPT:AU-KG.ontology.emits-database-ontology-entities)
    ".sql",
    ".ddl",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".scala",
    ".sc",
    ".lua",
}

_CONCEPT_PATTERN = re.compile(r"CONCEPT:([A-Z]+-[\d\.]+)(?:[:\s\-—]+([^<*\n]+))?")


def _ingest_markdown(
    file_path: str,
    file_node_id: str,
    graph: Any,
    RegistryNodeType: Any,
    RegistryEdgeType: Any,
) -> int:
    """Extract SDD nodes + CONCEPT tags from a markdown file (regex, no tree-sitter)."""
    extracted = 0
    stem = Path(file_path).stem
    lower_path = file_path.lower()

    if lower_path.endswith("constitution.md"):
        node_id = "policy:constitution"
        graph.add_node(
            node_id,
            node_type=RegistryNodeType.POLICY,
            policy_id="constitution",
            condition="all operations",
            action="Adhere to core project governance and rules defined in constitution",
        )
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1
    elif ".specify/tasks" in lower_path:
        node_id = f"task:{stem}"
        graph.add_node(
            node_id,
            node_type=RegistryNodeType.WORK_ITEM_PRIORITY,
            task_id=stem,
            status="pending",
        )
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1
    elif ".specify/specs" in lower_path:
        node_id = f"goal:{stem}"
        graph.add_node(
            node_id,
            node_type=RegistryNodeType.GOAL,
            goal_text=stem,
            status="active",
        )
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1
    elif ".specify/design" in lower_path:
        node_id = f"doc:design:{stem}"
        graph.add_node(node_id, node_type=RegistryNodeType.DOCUMENT, title=stem)
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1
    elif ".specify/memory" in lower_path:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        node_id = f"memory:{stem}"
        graph.add_node(
            node_id,
            node_type=RegistryNodeType.MEMORY,
            category="sdd_memory",
            content=content[:200],
        )
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1
    elif ".specify/reports" in lower_path:
        node_id = f"doc:report:{stem}"
        graph.add_node(node_id, node_type=RegistryNodeType.DOCUMENT, title=stem)
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1

    # Explicit CONCEPT tags
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    for match in _CONCEPT_PATTERN.finditer(content):
        concept_id = match.group(1).strip()
        desc = match.group(2).strip() if match.group(2) else ""
        node_id = f"concept:{concept_id}"
        graph.add_node(
            node_id,
            node_type=RegistryNodeType.CONCEPT,
            concept_id=concept_id,
            definition=desc,
            name=concept_id,
        )
        graph.add_edge(
            node_id, file_node_id, relationship=RegistryEdgeType.MENTIONED_IN
        )
        extracted += 1

    return extracted


def _replay_parse_result(
    result: dict[str, Any], graph: Any, RegistryNodeType: Any
) -> int:
    """Replay an engine ParseResult (nodes/edges) into the registry graph.

    Faithful to the previous hand-rolled extraction: SYMBOL nodes keyed by
    ``symbol:<sha256>`` (occurrence identity since 2026-09-04, not a content hash --
    ``ast_hash`` carries content) with name/symbol_type/line/ast_hash/file_path, joined to their
    ``file:<path>`` by IMPLEMENTS, plus calls_raw / depends_on_raw edges. The engine adds
    language/kind_detail/minhash (resolution inputs) which we carry through untouched.
    """
    # SQL DDL extraction (CONCEPT:AU-KG.ontology.emits-database-ontology-entities) emits database-ontology entities
    # alongside the code SYMBOL path; map each engine node_type to its registry type.
    db_types = {
        "DatabaseTable": RegistryNodeType.DATABASE_TABLE,
        "DatabaseColumn": RegistryNodeType.DATABASE_COLUMN,
        "DatabaseView": RegistryNodeType.DATABASE_VIEW,
    }
    for node in result.get("nodes", []) or []:
        node_type = node.get("node_type")
        props = dict(node.get("properties", {}) or {})
        # The engine serializes everything as strings; coerce line for numeric consumers.
        if "line" in props:
            try:
                props["line"] = int(props["line"])
            except (TypeError, ValueError):
                pass
        if node_type == "SYMBOL":
            symbol_type = props.pop("symbol_type", "Symbol")
            graph.add_node(
                node["node_id"],
                node_type=RegistryNodeType.SYMBOL,
                symbol_type=symbol_type,
                **props,
            )
        elif node_type in db_types:
            graph.add_node(node["node_id"], node_type=db_types[node_type], **props)
        else:
            # FILE nodes are created by the scan phase; nothing else is expected here.
            continue

    for edge in result.get("edges", []) or []:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge.get("edge_type", "RELATED_TO"),
            **(edge.get("properties", {}) or {}),
        )

    extracted = result.get("symbols_extracted")
    if extracted is None:
        extracted = sum(
            1 for n in (result.get("nodes") or []) if n.get("node_type") == "SYMBOL"
        )
    return int(extracted)


async def _parse_files_batch(
    parser: Any, files: list[tuple[str, bytes]]
) -> list[dict[str, Any]] | None:
    """One ``ParseFiles`` round-trip for ``files`` (EH-273).

    Reuses the SAME batched mechanism ``core/graph_compute.py``'s
    ``GraphComputeEngine.parse_files``/``index_repository`` already use --
    ``EpistemicGraphClient.graph.parse_files`` -- rather than inventing a
    second batching scheme. ``parser`` is the ``RustASTParser`` the caller
    already built; its public ``socket_path``/``auth_secret``/
    ``verified_context`` attributes are reused verbatim so this opens the
    identical connection ``RustASTParser`` itself would, just once for the
    whole batch instead of once per file.

    Returns ``None`` (never raises) when the engine connection is
    unavailable, so the caller can fall back to ``parser.parse_file()``'s
    own per-file local-``ast`` degradation -- the exact fallback this phase
    already had, just reached once per batch instead of once per file.
    """
    from epistemic_graph.client import EpistemicGraphClient

    try:
        client = await EpistemicGraphClient.connect(
            socket_path=parser.socket_path,
            auth_secret=parser.auth_secret,
            graph_name="__commons__",
            verified_context=parser.verified_context,
        )
    except (
        FileNotFoundError,
        ConnectionRefusedError,
        ConnectionResetError,
        OSError,
        asyncio.IncompleteReadError,
    ) as exc:
        logger.warning(
            "AST batch service unavailable (%s); per-file fallback for this batch",
            exc,
        )
        return None
    try:
        return await client.graph.parse_files(files)
    except Exception as exc:
        # Deliberately broad: ANY batch-RPC failure degrades to the per-file
        # fallback rather than raising into the pipeline phase.
        logger.warning(
            "Batched parse failed (%s); per-file fallback for this batch", exc
        )
        return None
    finally:
        await client.close()


async def _parse_per_file_fallback(
    parser: Any,
    files: list[tuple[str, bytes]],
    graph: Any,
    RegistryNodeType: Any,
) -> int:
    """The ORIGINAL per-file loop body, kept verbatim as the fallback path
    when the batch RPC is unavailable or returns a malformed response —
    ``parser.parse_file()`` still degrades to local Python ``ast`` per file
    on its own when the engine socket is down, so this preserves the exact
    pre-EH-273 behavior for that degraded case."""
    extracted = 0
    for rel_path, source in files:
        try:
            result = await parser.parse_file(rel_path, source)
            extracted += _replay_parse_result(result, graph, RegistryNodeType)
        except Exception as e:
            logger.error("Pipeline source parse failed for %s: %s", rel_path, e)
    return extracted


async def _parse_batch_and_replay(
    parser: Any,
    batch: list[tuple[str, bytes]],
    graph: Any,
    RegistryNodeType: Any,
) -> int:
    """Batch-parse ``batch`` (EH-273), chunked like ``enrichment/pipeline.py``'s
    ``make_batch_parse_fn`` (same ``KG_PARSE_BATCH`` setting, default 512) so a
    big scan makes few round-trips instead of one per file. Falls back to the
    exact previous per-file loop for any chunk whose batch RPC fails or whose
    response doesn't have exactly one result per requested file — a
    partial/malformed batch response is never silently mis-mapped to the
    wrong files (CONCEPT:AU-KG.ingest.exact-parser-acknowledgement)."""
    from agent_utilities.core.config import setting

    try:
        chunk_size = max(1, int(setting("KG_PARSE_BATCH", 512)))
    except (TypeError, ValueError):
        chunk_size = 512

    extracted = 0
    for i in range(0, len(batch), chunk_size):
        chunk = batch[i : i + chunk_size]
        results = await _parse_files_batch(parser, chunk)
        if results is None:
            extracted += await _parse_per_file_fallback(
                parser, chunk, graph, RegistryNodeType
            )
            continue
        if len(results) != len(chunk):
            logger.error(
                "parse_files returned %d result(s) for %d requested file(s); "
                "falling back to per-file parse for this chunk",
                len(results),
                len(chunk),
            )
            extracted += await _parse_per_file_fallback(
                parser, chunk, graph, RegistryNodeType
            )
            continue
        for (rel_path, _source), result in zip(chunk, results, strict=True):
            try:
                extracted += _replay_parse_result(result, graph, RegistryNodeType)
            except Exception as e:
                logger.error("Pipeline source parse failed for %s: %s", rel_path, e)
    return extracted


async def execute_parse(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Extract symbols: markdown via regex (here), code via the epistemic-graph engine.

    EH-273: code files are parsed in ONE batched ``ParseFiles`` round-trip
    (chunked at ``KG_PARSE_BATCH``) instead of one engine RPC per file — the
    same mechanism the ``IndexRepository`` path already uses. Markdown files
    are still handled inline, individually, exactly as before (no engine
    call involved). NOTE — semantic change: markdown files are now all
    processed before the code-file batch, rather than interleaved in the
    scanner's original per-file order; both are purely additive graph
    writes with no shared ids between the two categories, so this is not
    expected to be observable, but it IS an ordering change from the
    literal per-file loop this replaces.
    """

    from ....models.knowledge_graph import (
        RegistryEdgeType,
        RegistryNodeType,
    )

    files = deps["scan"].output
    graph = ctx.graph
    symbols_extracted = 0

    # GOC-73: ``epistemic-graph[full]`` is the ``agent-utilities[graphos]`` extra, not
    # a base dependency, but THIS pipeline phase (code parsing) genuinely requires the
    # engine's native parser bindings — there is no fallback parser. A caller that
    # reaches this phase without `[graphos]` installed gets Python's own
    # ``ModuleNotFoundError: No module named 'epistemic_graph'`` here, which is an
    # invalid installation for a deployment that runs ingestion, not a supported
    # degradation path (install with ``pip install agent-utilities[graphos]``).
    from epistemic_graph.parser import RustASTParser

    from ...core.session import GraphSession

    parser: Any = RustASTParser(
        verified_context=GraphSession.from_ambient().engine_verified_context()
    )

    pending: list[tuple[str, bytes]] = []
    for file_path in files:
        try:
            rel_path = os.path.relpath(file_path, ctx.config.workspace_path)

            if file_path.endswith(".md"):
                file_node_id = f"file:{rel_path}"
                symbols_extracted += _ingest_markdown(
                    file_path, file_node_id, graph, RegistryNodeType, RegistryEdgeType
                )
                continue

            if parser is None:
                continue
            if os.path.splitext(file_path)[1].lower() not in _CODE_EXTENSIONS:
                continue

            with open(file_path, "rb") as rb_f:
                source = rb_f.read()
            pending.append((rel_path, source))

        except Exception as e:
            logger.error("Pipeline source parse failed: %s", e)

    # `_parse_batch_and_replay` is a safe no-op (returns 0) on an empty
    # `pending`, so no guard is needed here.
    symbols_extracted += await _parse_batch_and_replay(
        parser, pending, graph, RegistryNodeType
    )

    return {"symbols_extracted": symbols_extracted}


parse_phase = PipelinePhase(name="parse", deps=["scan"], execute_fn=execute_parse)
