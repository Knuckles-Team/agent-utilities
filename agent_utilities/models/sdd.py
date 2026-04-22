from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
    id: str
    title: str
    description: str
    file_paths: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    parallel: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    result: str | None = None
    git_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tasks(BaseModel):
    feature_id: str = "default"
    tasks: list[Task] = Field(default_factory=list)
    parallel_waves: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_mermaid(self, title: str = "Task Dependencies") -> str:
        """Generate a Mermaid flowchart for the tasks."""
        from ..mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=title)
        
        for task in self.tasks:
            shape = "box" if not task.parallel else "round"
            # Highlight by status
            css_class = task.status.value if hasattr(task.status, "value") else str(task.status)
            
            builder.add_node(
                task.id, 
                label=f"{task.title}\n({task.id})", 
                shape=shape,
                css_class=css_class
            )
            
            for dep in task.depends_on:
                builder.add_edge(dep, task.id)
        
        # Add styling
        builder.lines.append("  classDef completed fill:#2e7d32,stroke:#1b5e20,color:#fff")
        builder.lines.append("  classDef failed fill:#c62828,stroke:#b71c1c,color:#fff")
        builder.lines.append("  classDef in_progress fill:#1565c0,stroke:#0d47a1,color:#fff,stroke-width:2px")
        builder.lines.append("  classDef pending fill:#455a64,stroke:#263238,color:#fff")
        
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
        from ..mermaid import ClassDiagramBuilder

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
