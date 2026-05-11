"""CONCEPT:KG-2.0"""

import logging
import os
from typing import Any

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)

# Load languages
PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())


def get_parser(file_path: str) -> Parser | None:
    ext = os.path.splitext(file_path)[1]
    lang = None
    if ext == ".py":
        lang = PY_LANGUAGE
    elif ext == ".js":
        lang = JS_LANGUAGE
    elif ext in (".ts", ".mts", ".cts"):
        lang = TS_LANGUAGE
    elif ext == ".tsx":
        lang = TSX_LANGUAGE

    if lang:
        return Parser(lang)
    return None


def get_node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


async def execute_parse(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Parse files using tree-sitter and extract symbols."""

    from ....models.knowledge_graph import (
        RegistryEdgeType,
        RegistryNodeType,
        SymbolMetadata,
    )

    files = deps["scan"].output
    graph = ctx.nx_graph

    symbols_extracted = 0
    for file_path in files:
        try:
            rel_path = os.path.relpath(file_path, ctx.config.workspace_path)
            file_node_id = f"file:{rel_path}"

            if file_path.endswith(".md"):
                import re
                from pathlib import Path

                # Check for SDD files first
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
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
                elif ".specify/tasks" in lower_path:
                    node_id = f"task:{stem}"
                    graph.add_node(
                        node_id,
                        type=RegistryNodeType.PRIORITIZED_TASK,
                        task_id=stem,
                        status="pending",
                    )
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
                elif ".specify/specs" in lower_path:
                    node_id = f"goal:{stem}"
                    graph.add_node(
                        node_id,
                        type=RegistryNodeType.GOAL,
                        goal_text=stem,
                        status="active",
                    )
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
                elif ".specify/design" in lower_path:
                    node_id = f"doc:design:{stem}"
                    graph.add_node(node_id, type=RegistryNodeType.DOCUMENT, title=stem)
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
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
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
                elif ".specify/reports" in lower_path:
                    node_id = f"doc:report:{stem}"
                    graph.add_node(node_id, type=RegistryNodeType.DOCUMENT, title=stem)
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1

                # Extract explicit CONCEPT tags
                concept_pattern = re.compile(
                    r"CONCEPT:([A-Z]+-[\d\.]+)(?:[:\s\-—]+([^<*\n]+))?"
                )
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                for match in concept_pattern.finditer(content):
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
                    graph.add_edge(
                        node_id, file_node_id, type=RegistryEdgeType.MENTIONED_IN
                    )
                    symbols_extracted += 1
                continue

            parser = get_parser(file_path)
            if not parser:
                continue

            with open(file_path, "rb") as rb_f:
                source = rb_f.read()
            tree = parser.parse(source)

            def walk(node, source, file_path, file_node_id):
                nonlocal symbols_extracted

                # Python
                if node.type == "class_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = get_node_text(name_node, source)
                        symbol_id = f"symbol:{name}"

                        # Extract docstring if present
                        docstring = None
                        body = node.child_by_field_name("body")
                        if body and body.children:
                            first = body.children[0]
                            if (
                                first.type == "expression_statement"
                                and first.children[0].type == "string"
                            ):
                                docstring = get_node_text(
                                    first.children[0], source
                                ).strip("'\"")

                        # Extract superclasses
                        bases = []
                        superclasses = node.child_by_field_name("superclasses")
                        if superclasses:
                            for arg in superclasses.children:
                                if arg.type == "identifier":
                                    bases.append(get_node_text(arg, source))

                        meta = SymbolMetadata(
                            name=name,
                            type="Class",
                            line=node.start_point[0] + 1,
                            docstring=docstring,
                            args=bases,
                        )
                        meta_data = meta.model_dump()
                        s_type = meta_data.pop("type")
                        graph.add_node(
                            symbol_id,
                            **meta_data,
                            symbol_type=s_type,
                            type=RegistryNodeType.SYMBOL,
                            file_path=file_path,
                        )
                        graph.add_edge(
                            symbol_id, file_node_id, type=RegistryEdgeType.CONTAINS
                        )
                        symbols_extracted += 1

                elif node.type == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = get_node_text(name_node, source)
                        symbol_id = f"symbol:{name}"
                        meta = SymbolMetadata(
                            name=name, type="Function", line=node.start_point[0] + 1
                        )
                        meta_data = meta.model_dump()
                        s_type = meta_data.pop("type")
                        graph.add_node(
                            symbol_id,
                            **meta_data,
                            symbol_type=s_type,
                            type=RegistryNodeType.SYMBOL,
                            file_path=file_path,
                        )
                        graph.add_edge(
                            symbol_id, file_node_id, type=RegistryEdgeType.CONTAINS
                        )
                        symbols_extracted += 1

                elif node.type == "call":
                    function_node = node.child_by_field_name("function")
                    if function_node:
                        callee = get_node_text(function_node, source)
                        # Mark raw dependency
                        graph.add_edge(
                            file_node_id, callee, type="calls_raw", raw=callee
                        )

                # JS/TS
                elif node.type in (
                    "class_declaration",
                    "function_declaration",
                    "method_definition",
                ):
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = get_node_text(name_node, source)
                        symbol_id = f"symbol:{name}"
                        graph.add_node(
                            symbol_id,
                            type=RegistryNodeType.SYMBOL,
                            name=name,
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                        )
                        graph.add_edge(
                            symbol_id, file_node_id, type=RegistryEdgeType.CONTAINS
                        )
                        symbols_extracted += 1

                # Imports (Python)
                elif node.type == "import_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            module_name = get_node_text(child, source)
                            graph.add_edge(
                                file_node_id,
                                module_name,
                                type="depends_on_raw",
                                raw=module_name,
                            )
                elif node.type == "import_from_statement":
                    module_node = node.child_by_field_name("module_name")
                    if module_node:
                        module_name = get_node_text(module_node, source)
                        graph.add_edge(
                            file_node_id,
                            module_name,
                            type="depends_on_raw",
                            raw=module_name,
                        )

                # Imports (JS/TS)
                elif node.type == "import_declaration":
                    source_node = node.child_by_field_name("source")
                    if source_node:
                        module_name = get_node_text(source_node, source).strip("'\"")
                        graph.add_edge(
                            file_node_id,
                            module_name,
                            type="depends_on_raw",
                            raw=module_name,
                        )

                for child in node.children:
                    walk(child, source, file_path, file_node_id)

            walk(tree.root_node, source, file_path, file_node_id)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    return {"symbols_extracted": symbols_extracted}


parse_phase = PipelinePhase(name="parse", deps=["scan"], execute_fn=execute_parse)
