from __future__ import annotations

import uuid
from typing import Literal

from pydantic_ai import Agent

from agent_utilities.core.model_factory import create_model

from ..models.codemap import (
    CodemapArtifact,
    CodemapEdge,
    CodemapNode,
    HierarchicalSection,
)
from .engine import IntelligenceGraphEngine


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
                CodemapNode(
                    id=node_data["id"],
                    label=node_data["label"],
                    type=node_data["type"],
                    file=node_data["file"],
                    line=node_data.get("line"),
                    end_line=node_data.get("end_line"),
                    description=node_data.get("description"),
                    importance=node_data.get("centrality", 0.0),
                )
            )
        for edge_data in subgraph.edges:
            edges.append(CodemapEdge(**edge_data))

        # 3. LLM pass → hierarchical outline + trace guides
        model = self.smart_model if mode == "smart" else self.fast_model

        # We use pydantic-ai Agent for structured output
        # In a real implementation, we might want to use a specific system prompt
        hierarchy_agent = Agent(
            model=model,
            result_type=list[HierarchicalSection],
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
        hierarchy = result.data

        # 4. Assemble final artifact
        artifact = CodemapArtifact(
            id=str(uuid.uuid4()),
            prompt=prompt,
            mode=mode,
            repo_root=getattr(self.kg, "repo_root", None),
            hierarchy=hierarchy,
            nodes=nodes,
            edges=edges,
            metadata={
                "subgraph_node_count": len(nodes),
                "generated_by": "agent-utilities-codemap-v1",
            },
        )

        # 5. Persist the codemap to the Knowledge Graph
        await self.kg.store_codemap(artifact)

        return artifact
