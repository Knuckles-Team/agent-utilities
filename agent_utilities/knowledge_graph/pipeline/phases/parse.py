"""CONCEPT:KG-2.0 / KG-2.106 — code AST parsing delegated to the epistemic-graph engine.

The Rust engine (``eg-compute``, native tree-sitter, 14 grammars) extracts the SAME
registry-graph schema this phase used to build by hand with Python tree-sitter —
``file:<path>`` + ``symbol:<sha256>`` nodes joined by ``IMPLEMENTS``, plus ``calls_raw`` /
``depends_on_raw`` edges (and richer call-graph + MinHash similarity signals across many
more languages). Delegating here is what lets agent-utilities drop the Python
``tree-sitter*`` wheels entirely: the engine is the ONE code-parsing implementation
(``epistemic_graph.parser.RustASTParser``, in the optional ``[engine]`` extra). When the
engine socket is unavailable, ``RustASTParser`` transparently falls back to Python's
stdlib ``ast`` (Python sources only). Markdown CONCEPT/SDD extraction is regex-based (it
never used tree-sitter) and stays here.
"""

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
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".hh", ".cs", ".rb", ".php", ".sh", ".bash", ".scala", ".sc", ".lua",
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
            type=RegistryNodeType.POLICY,
            policy_id="constitution",
            condition="all operations",
            action="Adhere to core project governance and rules defined in constitution",
        )
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1
    elif ".specify/tasks" in lower_path:
        node_id = f"task:{stem}"
        graph.add_node(
            node_id,
            type=RegistryNodeType.PRIORITIZED_TASK,
            task_id=stem,
            status="pending",
        )
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1
    elif ".specify/specs" in lower_path:
        node_id = f"goal:{stem}"
        graph.add_node(
            node_id,
            type=RegistryNodeType.GOAL,
            goal_text=stem,
            status="active",
        )
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1
    elif ".specify/design" in lower_path:
        node_id = f"doc:design:{stem}"
        graph.add_node(node_id, type=RegistryNodeType.DOCUMENT, title=stem)
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1
    elif ".specify/memory" in lower_path:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        node_id = f"memory:{stem}"
        graph.add_node(
            node_id,
            type=RegistryNodeType.MEMORY,
            category="sdd_memory",
            content=content[:200],
        )
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1
    elif ".specify/reports" in lower_path:
        node_id = f"doc:report:{stem}"
        graph.add_node(node_id, type=RegistryNodeType.DOCUMENT, title=stem)
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
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
            type=RegistryNodeType.CONCEPT,
            concept_id=concept_id,
            definition=desc,
            name=concept_id,
        )
        graph.add_edge(node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN)
        extracted += 1

    return extracted


def _replay_parse_result(result: dict[str, Any], graph: Any, RegistryNodeType: Any) -> int:
    """Replay an engine ParseResult (nodes/edges) into the registry graph.

    Faithful to the previous hand-rolled extraction: SYMBOL nodes keyed by
    ``symbol:<sha256>`` with name/symbol_type/line/ast_hash/file_path, joined to their
    ``file:<path>`` by IMPLEMENTS, plus calls_raw / depends_on_raw edges. The engine adds
    language/kind_detail/minhash (resolution inputs) which we carry through untouched.
    """
    for node in result.get("nodes", []) or []:
        if node.get("node_type") != "SYMBOL":
            # FILE nodes are created by the scan phase; nothing else is expected here.
            continue
        props = dict(node.get("properties", {}) or {})
        # The engine serializes everything as strings; coerce line for numeric consumers.
        if "line" in props:
            try:
                props["line"] = int(props["line"])
            except (TypeError, ValueError):
                pass
        symbol_type = props.pop("symbol_type", "Symbol")
        graph.add_node(
            node["node_id"],
            type=RegistryNodeType.SYMBOL,
            symbol_type=symbol_type,
            **props,
        )

    for edge in result.get("edges", []) or []:
        graph.add_edge(
            edge["source"],
            edge["target"],
            type=edge.get("edge_type", "RELATED_TO"),
            **(edge.get("properties", {}) or {}),
        )

    extracted = result.get("symbols_extracted")
    if extracted is None:
        extracted = sum(
            1 for n in (result.get("nodes") or []) if n.get("node_type") == "SYMBOL"
        )
    return int(extracted)


async def execute_parse(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Extract symbols: markdown via regex (here), code via the epistemic-graph engine."""

    from ....models.knowledge_graph import (
        RegistryEdgeType,
        RegistryNodeType,
    )

    files = deps["scan"].output
    graph = ctx.graph
    symbols_extracted = 0

    # The engine client lives in the optional [engine] extra. A bare/[mcp] install never
    # runs ingestion so this phase is never reached, but guard the import so the module
    # stays importable without epistemic-graph installed.
    try:
        from epistemic_graph.parser import RustASTParser

        parser: Any = RustASTParser()
    except ImportError:
        parser = None
        logger.info(
            "epistemic-graph not installed; code AST parse skipped "
            "(install agent-utilities[engine]). Markdown extraction still runs."
        )

    for file_path in files:
        try:
            rel_path = os.path.relpath(file_path, ctx.config.workspace_path)
            file_node_id = f"file:{rel_path}"

            if file_path.endswith(".md"):
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

            # Delegate to the engine (native tree-sitter; Python `ast` fallback when the
            # engine socket is down). The returned file_node_id is `file:<rel_path>` since
            # we pass rel_path — identical to the previous behavior.
            result = await parser.parse_file(rel_path, source)
            symbols_extracted += _replay_parse_result(result, graph, RegistryNodeType)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    return {"symbols_extracted": symbols_extracted}


parse_phase = PipelinePhase(name="parse", deps=["scan"], execute_fn=execute_parse)
