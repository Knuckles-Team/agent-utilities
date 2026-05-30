from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskStatus(StrEnum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UserStory(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class Spec(BaseModel):
    feature_id: str
    title: str
    user_stories: list[UserStory]
    non_functional_requirements: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = ""
    title: str = ""
    description: Any = ""
    input_data: Any | None = Field(default=None)
    file_paths: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    parallel: bool = False
    status: str = "pending"
    assigned_to: str | None = None
    result: str | None = None
    git_commit: str | None = None
    subtasks: list["Task"] = Field(default_factory=list)
    """CONCEPT:ORCH-1.1 — HTN subtasks allowing recursive goal decomposition."""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Unified fields from ExecutionStep
    refined_subtask: str | None = None
    timeout: float = 3600.0
    access_list: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: dict[str, Any] | Any) -> dict[str, Any] | Any:
        if isinstance(data, dict):
            if "node_id" in data and "id" not in data:
                data["id"] = data["node_id"]
            if "id" in data and "node_id" not in data:
                data["node_id"] = data["id"]

            if "is_parallel" in data and "parallel" not in data:
                data["parallel"] = data["is_parallel"]
            if "parallel" in data and "is_parallel" not in data:
                data["is_parallel"] = data["parallel"]

        return data

    @property
    def node_id(self) -> str:
        return self.id

    @node_id.setter
    def node_id(self, val: str) -> None:
        self.id = val

    @property
    def is_parallel(self) -> bool:
        return self.parallel

    @is_parallel.setter
    def is_parallel(self, val: bool) -> None:
        self.parallel = val


class Tasks(BaseModel):
    feature_id: str = "default"
    tasks: list[Task] = Field(default_factory=list)
    parallel_waves: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_mermaid(self, title: str = "Task Dependencies") -> str:
        """Generate a Mermaid flowchart for the tasks."""
        from agent_utilities.observability.mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=title)

        for task in self.tasks:
            shape = "box" if not task.parallel else "round"
            # Highlight by status
            css_class = (
                task.status.value if hasattr(task.status, "value") else str(task.status)
            )

            builder.add_node(
                task.id,
                label=f"{task.title}\n({task.id})",
                shape=shape,
                css_class=css_class,
            )

            for dep in task.depends_on:
                builder.add_edge(dep, task.id)

        # Add styling
        builder.lines.append(
            "  classDef completed fill:#2e7d32,stroke:#1b5e20,color:#fff"
        )
        builder.lines.append("  classDef failed fill:#c62828,stroke:#b71c1c,color:#fff")
        builder.lines.append(
            "  classDef in_progress fill:#1565c0,stroke:#0d47a1,color:#fff,stroke-width:2px"
        )
        builder.lines.append(
            "  classDef pending fill:#455a64,stroke:#263238,color:#fff"
        )

        return builder.render()


class ImplementationPlan(BaseModel):
    feature_id: str
    title: str
    approach: str = ""
    technical_context: str = ""
    proposed_changes: list[str] = Field(default_factory=list)
    tradeoffs: dict[str, str] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    risk_assessment: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_mermaid(self) -> str:
        """Generate a Mermaid class diagram for the implementation plan."""
        from agent_utilities.observability.mermaid import ClassDiagramBuilder

        builder = ClassDiagramBuilder(title=f"Implementation: {self.title}")

        # Add main plan entity
        builder.add_class(
            "ImplementationPlan",
            attributes=[f"feature_id: {self.feature_id}", f"title: {self.title}"],
            methods=["get_approach()", "get_risks()"],
        )

        # Add proposed changes as related components
        for change in self.proposed_changes:
            # Simple heuristic to extract component name
            comp_name = change.split(":")[0].strip() if ":" in change else "Component"
            safe_comp = comp_name.replace(" ", "_").replace("/", "_").replace(".", "_")
            builder.add_class(safe_comp, annotation="Modified")
            builder.add_relationship("ImplementationPlan", safe_comp, "-->", "modifies")

        return builder.render()


class ProjectConstitution(BaseModel):
    vision: str = ""
    mission: str = ""
    core_principles: list[str] = Field(default_factory=list)
    tech_stack: dict[str, Any] = Field(default_factory=dict)
    quality_gates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- DSTDD (Design-Spec-Test Driven Development) Models ---


class ExtensionStrategy(StrEnum):
    """How a new feature relates to existing pillar concepts."""

    AUGMENT = "augment"
    """Adds functionality to an existing concept without changing its interface."""

    COMPOSE = "compose"
    """Combines multiple existing concepts into a higher-level abstraction."""

    SPECIALIZE = "specialize"
    """Creates a domain-specific variant of an existing concept."""

    NEW = "new"
    """Introduces an entirely new concept (requires justification)."""


class NearestConcept(BaseModel):
    """A KG node that is semantically similar to the proposed feature."""

    concept_id: str = Field(description="CONCEPT:ID of the nearest match")
    name: str = Field(description="Human-readable name of the concept")
    similarity: float = Field(
        description="Semantic similarity score (0.0 to 1.0)", ge=0.0, le=1.0
    )
    pillar: str = Field(
        description="Pillar this concept belongs to (ORCH/KG/AHE/ECO/OS)"
    )


class NewConceptProposal(BaseModel):
    """Justification for introducing an entirely new concept.

    Required only when extension_strategy is 'new' and no existing concept
    can be extended to accommodate the feature.
    """

    proposed_id: str = Field(
        description="Proposed CONCEPT:ID (e.g., 'KG-2.54', 'ECO-4.14')"
    )
    target_pillar: str = Field(
        description="Which pillar this new concept augments (ORCH/KG/AHE/ECO/OS)"
    )
    pipeline_phase: str = Field(
        default="",
        description="Which of the 15-phase pipeline phases this wires into",
    )
    justification: str = Field(
        description="Why this cannot be expressed as an extension of existing concepts"
    )


class KGAnalysis(BaseModel):
    """Knowledge Graph analysis results for a proposed feature.

    This is the mandatory first step of the DSTDD pipeline. Every new feature
    must demonstrate that it has been analyzed against the existing KG to find
    the nearest concepts and determine the optimal extension strategy.
    """

    nearest_concepts: list[NearestConcept] = Field(
        default_factory=list,
        description="Top-5 nearest existing concepts from KG semantic search",
    )
    extension_point: str | None = Field(
        default=None,
        description="CONCEPT:ID of the concept being extended (if applicable)",
    )
    extension_strategy: ExtensionStrategy = Field(
        default=ExtensionStrategy.AUGMENT,
        description="How this feature relates to existing concepts",
    )
    new_concept_proposal: NewConceptProposal | None = Field(
        default=None,
        description="Required when extension_strategy is 'new'",
    )


class RiskAssessment(BaseModel):
    """Risk analysis for a proposed feature's impact on the system."""

    blast_radius: list[str] = Field(
        default_factory=list,
        description="List of existing modules affected by this change",
    )
    backward_compatible: bool = Field(
        default=True,
        description="Whether this change maintains backward compatibility",
    )
    breaking_changes: list[str] = Field(
        default_factory=list,
        description="List of breaking changes (if any)",
    )


class DesignDocument(BaseModel):
    """Design Document — First phase of the DSTDD pipeline.

    CONCEPT:ORCH-1.3 Extension — Design-Spec-Test Driven Development

    Every feature begins with a design document that gates creation through
    the Knowledge Graph. The design must demonstrate:
    1. KG analysis was performed (nearest concepts identified)
    2. Extension strategy is justified
    3. C4 integration diagram shows where this fits in the 5-pillar topology
    4. Risk assessment covers blast radius and backward compatibility

    The design document is stored in ``.specify/design/<feature_id>.md``
    and must be validated before a Spec can be created.
    """

    feature_id: str = Field(description="Unique feature identifier")
    title: str = Field(description="Human-readable feature title")
    kg_analysis: KGAnalysis = Field(
        description="KG search and extension analysis results"
    )
    c4_diagram: str = Field(
        default="",
        description="Mermaid C4 diagram source showing integration point",
    )
    data_flow: str = Field(
        default="",
        description="Description of how data flows through the 5 pillars",
    )
    risk_assessment: RiskAssessment = Field(
        default_factory=RiskAssessment,
        description="Impact and compatibility analysis",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
