from __future__ import annotations

"""CONCEPT:AU-KG.query.object-graph-mapper"""

import uuid
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.core.model_factory import create_model
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

from ...models.codemap import (
    CodemapArtifact,
    CodemapEdge,
    CodemapNode,
    HierarchicalSection,
)
from .engine import IntelligenceGraphEngine


def _safe_file_reference(value: object, *, namespace: str) -> str:
    """Keep repository-relative paths readable and replace host paths opaquely."""
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or PureWindowsPath(str(value or "")).is_absolute()
        or ".." in path.parts
    ):
        return persistence_reference("codemap_file", raw, namespace=namespace)
    clean, _ = PersistencePrivacyGuard().sanitize_text(raw)
    return clean


def _sanitize_node(
    value: CodemapNode | dict,
    *,
    node_refs: dict[str, str],
    namespace: str,
) -> CodemapNode:
    row = value.model_dump() if isinstance(value, CodemapNode) else dict(value)
    raw_id = str(row.get("id") or "")
    opaque_id = node_refs.setdefault(
        raw_id,
        persistence_reference("codemap_node", raw_id, namespace=namespace),
    )
    guard = PersistencePrivacyGuard()
    label, _ = guard.sanitize_text(str(row.get("label") or opaque_id))
    description = row.get("description")
    clean_description = None
    if description is not None:
        clean_description, _ = guard.sanitize_text(str(description))
    return CodemapNode(
        id=opaque_id,
        label=label,
        type=str(row.get("type") or "symbol"),
        file=_safe_file_reference(row.get("file"), namespace=namespace),
        line=row.get("line"),
        end_line=row.get("end_line"),
        description=clean_description,
        importance=float(row.get("importance", row.get("centrality", 0.0)) or 0.0),
    )


def _sanitize_hierarchy(
    section: HierarchicalSection,
    *,
    node_refs: dict[str, str],
    namespace: str,
) -> HierarchicalSection:
    guard = PersistencePrivacyGuard()
    title, _ = guard.sanitize_text(section.title)
    trace = section.trace_guide
    if trace is not None:
        explanation, _ = guard.sanitize_text(trace.explanation)
        trace = trace.model_copy(
            update={
                "title": guard.sanitize_text(trace.title)[0],
                "explanation": explanation,
                "key_insights": [
                    guard.sanitize_text(item)[0] for item in trace.key_insights
                ],
                "related_nodes": [
                    node_refs.setdefault(
                        node_id,
                        persistence_reference(
                            "codemap_node", node_id, namespace=namespace
                        ),
                    )
                    for node_id in trace.related_nodes
                ],
            }
        )
    return HierarchicalSection(
        title=title,
        nodes=[
            _sanitize_node(node, node_refs=node_refs, namespace=namespace)
            for node in section.nodes
        ],
        children=[
            _sanitize_hierarchy(child, node_refs=node_refs, namespace=namespace)
            for child in section.children
        ],
        trace_guide=trace,
    )


class CodemapGenerator:
    """Just-in-Time Codemap generator — based on the Windsurf Codemaps pattern."""

    def __init__(self, kg: IntelligenceGraphEngine):
        self.kg = kg
        # Use existing model factory to create models for pydantic-ai
        self.fast_model = create_model(model_id="gpt-3.5-turbo")  # Fast model
        self.smart_model = create_model()  # Default/Smart model (usually gpt-4o)

    async def create(
        self,
        prompt: str,
        mode: Literal["fast", "smart"] = "smart",
        max_nodes: int = 150,
    ) -> CodemapArtifact:
        """Generate a task-specific codemap."""

        prompt_ref = persistence_reference(
            "codemap_prompt", prompt, namespace="codemap"
        )
        workspace = getattr(self.kg, "repo_root", None)
        workspace_ref = (
            persistence_reference("codemap_workspace", workspace, namespace="codemap")
            if workspace
            else None
        )
        node_refs: dict[str, str] = {}

        # 1. Focused subgraph extraction (reuses existing engine machinery)
        subgraph = await self.kg.extract_focused_subgraph(
            query=prompt,
            max_nodes=max_nodes,
        )

        # 2. Build raw nodes/edges for the visual layer
        nodes: list[CodemapNode] = []
        edges: list[CodemapEdge] = []
        for node_data in subgraph.nodes:
            nodes.append(
                _sanitize_node(
                    node_data,
                    node_refs=node_refs,
                    namespace=prompt_ref,
                )
            )
        for edge_data in subgraph.edges:
            source = str(edge_data["source"])
            target = str(edge_data["target"])
            edges.append(
                CodemapEdge(
                    source=node_refs.setdefault(
                        source,
                        persistence_reference(
                            "codemap_node", source, namespace=prompt_ref
                        ),
                    ),
                    target=node_refs.setdefault(
                        target,
                        persistence_reference(
                            "codemap_node", target, namespace=prompt_ref
                        ),
                    ),
                    type=str(edge_data["type"]),
                    weight=float(edge_data.get("weight", 1.0)),
                )
            )

        # 3. LLM pass → hierarchical outline + trace guides
        model = self.smart_model if mode == "smart" else self.fast_model

        # We use pydantic-ai Agent for structured output
        # In a real implementation, we might want to use a specific system prompt
        hierarchy_agent = create_context_agent(
            model=model,
            output_type=list[HierarchicalSection],
            system_prompt=(
                "You are a senior architect creating a concise, hierarchical codemap of a codebase. "
                "Given a user task and a list of relevant files and functions (subgraph), "
                "produce a clean, logical hierarchy that explains how the code flows to solve the task. "
                "Include trace guides for complex sections."
            ),
        )

        subgraph_context = f"User task: {prompt}\n\nSubgraph nodes:\n"
        for n in nodes:
            subgraph_context += f"- {n.label} ({n.type}) in {n.file}\n"

        result = await hierarchy_agent.run(subgraph_context)
        hierarchy = [
            _sanitize_hierarchy(
                section,
                node_refs=node_refs,
                namespace=prompt_ref,
            )
            for section in result.output
        ]

        # 4. Assemble final artifact
        artifact = CodemapArtifact(
            id=str(uuid.uuid4()),
            prompt_ref=prompt_ref,
            workspace_ref=workspace_ref,
            mode=mode,
            hierarchy=hierarchy,
            nodes=nodes,
            edges=edges,
            evidence_refs=sorted(node_refs.values()),
            subgraph_node_count=len(nodes),
        )

        # 5. Persist the codemap to the Knowledge Graph
        await self.kg.store_codemap(artifact)

        return artifact

    async def skeleton(
        self,
        prompt: str,
        max_nodes: int = 150,
        max_tokens: int = 1024,
    ) -> str:
        """Render a token-budgeted, importance-ranked code skeleton (ORCH-1.48).

        The aider-style "repo map": it does the focused-subgraph extraction and
        importance ranking but **skips the expensive LLM hierarchy pass**, then
        renders :meth:`CodemapArtifact.to_skeleton`. Cheap enough to inject into a
        model's context on every turn.
        """
        subgraph = await self.kg.extract_focused_subgraph(
            query=prompt, max_nodes=max_nodes
        )
        prompt_ref = persistence_reference(
            "codemap_prompt", prompt, namespace="codemap"
        )
        node_refs: dict[str, str] = {}
        nodes = [
            _sanitize_node(
                nd,
                node_refs=node_refs,
                namespace=prompt_ref,
            )
            for nd in subgraph.nodes
        ]
        artifact = CodemapArtifact(
            id=str(uuid.uuid4()),
            prompt_ref=prompt_ref,
            workspace_ref=(
                persistence_reference(
                    "codemap_workspace",
                    getattr(self.kg, "repo_root", None),
                    namespace="codemap",
                )
                if getattr(self.kg, "repo_root", None)
                else None
            ),
            mode="fast",
            nodes=nodes,
            evidence_refs=sorted(node_refs.values()),
            subgraph_node_count=len(nodes),
        )
        return artifact.to_skeleton(max_tokens=max_tokens)
