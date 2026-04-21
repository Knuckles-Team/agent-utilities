# agent_utilities/graph/models.py
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GraphNode(BaseModel):
    """Base class for ALL KG nodes — enforces common metadata and validation."""

    node_id: str = Field(
        ..., alias="id", description="Unique identifier (use UUID or slug)"
    )
    labels: list[str] = Field(..., min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="1.0")
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_permanent: bool = Field(
        default=False, description="Hub node protection: do not prune if True"
    )
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_labels_and_temporal(self) -> "GraphNode":
        if not self.labels:
            raise ValueError("Every node must have at least one label")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from cannot be after valid_to")
        return self

    def to_cypher_props(self) -> dict[str, Any]:
        """Ready for Cypher parameter binding."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        # Flatten properties for Cypher
        props = data.pop("properties", {})
        return {**data, **props}


class Policy(GraphNode):
    """Declarative business/policy/governance logic (formerly Rule)."""

    labels: list[str] = Field(default_factory=lambda: ["Policy"])
    policy_id: str
    name: str
    description: str
    condition: str  # Cypher fragment, JSONLogic, or Python expr
    action: str  # tool name or function reference
    priority: int = Field(default=50, ge=0, le=100)
    applies_to: list[str] = Field(default_factory=list)  # entity types


class ProcessFlow(GraphNode):
    """Full workflow/process definition (natively integrable with SDD)."""

    labels: list[str] = Field(default_factory=lambda: ["ProcessFlow"])
    flow_id: str
    name: str
    goal: str
    start_step: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessStep(GraphNode):
    """Individual step inside a ProcessFlow (linked via :NEXT)."""

    labels: list[str] = Field(default_factory=lambda: ["ProcessStep"])
    step_id: str
    name: str
    step_type: Literal["user_input", "auto", "human_in_loop", "tool_call", "llm_call"]
    tool: str | None = None
    condition: str | None = None  # guard expression for edges


class KnowledgeBaseTopic(GraphNode):
    """Your recently added topic KBs — now fully typed."""

    labels: list[str] = Field(default_factory=lambda: ["KnowledgeBaseTopic"])
    topic_id: str
    name: str
    description: str
    source: str | None = None
    embedding: list[float] | None = None


class Concept(GraphNode):
    """Atomic knowledge unit (e.g. 'p53 gene', 'SN2 reaction')."""

    labels: list[str] = Field(default_factory=lambda: ["Concept"])
    concept_id: str
    name: str
    definition: str
    topic_id: str | None = None
    dynamic_properties: dict[str, Any] = Field(
        default_factory=dict
    )  # key-value for specific domain fields
    embedding: list[float] | None = None


class Source(GraphNode):
    """Reference material (e.g. papers, journals, datasets)."""

    labels: list[str] = Field(default_factory=lambda: ["Source"])
    source_id: str
    title: str
    doi: str | None = None
    url: str | None = None
    publication_date: datetime | None = None
    authors: list[str] = Field(default_factory=list)


class Evidence(GraphNode):
    """Claims or findings extracted from sources."""

    labels: list[str] = Field(default_factory=lambda: ["Evidence"])
    evidence_id: str
    claim: str
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_id: str | None = None


class Person(GraphNode):
    """Researchers, authors, or agents."""

    labels: list[str] = Field(default_factory=lambda: ["Person"])
    person_id: str
    name: str
    expertise: list[str] = Field(default_factory=list)
    affiliation: str | None = None
