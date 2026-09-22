"""CONCEPT:AU-KG.query.object-graph-mapper -- markdown CONCEPT/SDD extraction (Phase 4).

RF-031 (2026-09-22): this phase used to ALSO parse code files into
``file:<path>`` + ``symbol:<sha256>`` SYMBOL nodes (delegating to
``epistemic_graph.parser.RustASTParser``) and SQL DDL into
``DATABASE_TABLE``/``DATABASE_COLUMN``/``DATABASE_VIEW`` nodes. That branch is
deleted: nothing outside this pipeline's own closed loop
(``resolve``/``mro``/``reference``, deleted in the same change) ever read a
SYMBOL node, a ``calls_raw``/``depends_on_raw``/``inherits_from`` edge, or a
DATABASE_* node this phase wrote -- verified with ``git grep`` across the whole
production tree, not just this package (see the ``au-deletion`` lane's
``WRAPUP.md`` for the full evidence). The live code-graph the MCP
``graph_code action=code_context`` tool actually reads (the
``:Code``/``:Test``/``:Feature`` schema) is written by
``agent_utilities.knowledge_graph.enrichment.pipeline.EnrichmentPipeline`` via
the ``_bg_codebase`` async task handler -- a separate, already-engine-native
path this deletion does not touch. Markdown CONCEPT/SDD extraction (regex,
never used tree-sitter) is the one piece of this phase with a real, distinct
purpose (feeding the registry graph's POLICY/GOAL/MEMORY/DOCUMENT/CONCEPT
nodes) and is all that remains here.
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


async def execute_parse(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Extract registry content (POLICY/GOAL/MEMORY/DOCUMENT/CONCEPT) from markdown.

    RF-031: this phase no longer parses code -- see the module docstring. Only
    ``.md`` files from the scan output are inspected; every other file is
    skipped (code parsing/symbol writing is the live
    ``EnrichmentPipeline``/``_bg_codebase`` path's job, unaffected by this
    change).
    """

    from ....models.knowledge_graph import (
        RegistryEdgeType,
        RegistryNodeType,
    )

    files = deps["scan"].output
    graph = ctx.graph
    symbols_extracted = 0

    for file_path in files:
        if not file_path.endswith(".md"):
            continue
        try:
            rel_path = os.path.relpath(file_path, ctx.config.workspace_path)
            file_node_id = f"file:{rel_path}"
            symbols_extracted += _ingest_markdown(
                file_path, file_node_id, graph, RegistryNodeType, RegistryEdgeType
            )
        except Exception as e:
            logger.error("Pipeline markdown parse failed: %s", e)

    return {"symbols_extracted": symbols_extracted}


parse_phase = PipelinePhase(name="parse", deps=["scan"], execute_fn=execute_parse)
