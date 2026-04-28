from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CodemapNode(BaseModel):
    """A single node in the codemap (file, function, class, etc.)."""

    id: str = Field(..., description="Unique node ID (usually symbol_id or file:line)")
    label: str = Field(..., description="Human-readable name")
    type: str = Field(..., description="Node type: file, class, function, module, etc.")
    file: str = Field(..., description="Relative path from repo root")
    line: int | None = Field(None, description="Starting line number")
    end_line: int | None = Field(None)
    description: str | None = Field(None, description="Short one-line summary")
    importance: float = Field(
        0.0, ge=0.0, le=1.0, description="PageRank / centrality score"
    )


class CodemapEdge(BaseModel):
    """Relationship between nodes."""

    source: str
    target: str
    type: str = Field(..., description="Edge type: calls, imports, inherits, etc.")
    weight: float = 1.0


class TraceGuideSection(BaseModel):
    """Descriptive explanation of why a group of nodes belongs together."""

    title: str
    explanation: str
    key_insights: list[str] = Field(default_factory=list)
    related_nodes: list[str] = Field(default_factory=list)


class HierarchicalSection(BaseModel):
    """Collapsible hierarchical outline (the main "map" structure)."""

    title: str
    nodes: list[CodemapNode] = Field(default_factory=list)
    children: list[HierarchicalSection] = Field(default_factory=list)
    trace_guide: TraceGuideSection | None = None


class CodemapArtifact(BaseModel):
    """The complete shareable codemap artifact."""

    id: str = Field(..., description="Unique codemap UUID or slug")
    prompt: str = Field(..., description="The user prompt that generated this map")
    mode: Literal["fast", "smart"] = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    repo_root: str | None = None

    # Hierarchical outline (what humans read first)
    hierarchy: list[HierarchicalSection] = Field(default_factory=list)

    # Full graph for visual rendering
    nodes: list[CodemapNode] = Field(default_factory=list)
    edges: list[CodemapEdge] = Field(default_factory=list)

    # Metadata for agents & sharing
    version: str = "1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "populate_by_name": True,
    }

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart for the codemap graph."""
        from ..mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=f"Codemap: {self.prompt}")

        # Add nodes
        for node in self.nodes:
            shape = "box"
            if node.type == "file":
                shape = "cylinder"
            elif node.type == "class":
                shape = "round"
            elif node.type == "function":
                shape = "box"

            builder.add_node(node.id, label=f"{node.label}\n({node.type})", shape=shape)

        # Add edges
        for edge in self.edges:
            edge_type = "-->"
            if edge.type == "inherits":
                edge_type = "==>"
            elif edge.type == "imports":
                edge_type = "-.->"

            builder.add_edge(
                edge.source, edge.target, label=edge.type, edge_type=edge_type
            )

        return builder.render()

    @classmethod
    def from_json(cls, data: str) -> CodemapArtifact:
        return cls.model_validate_json(data)
