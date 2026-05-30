from typing import Any

from pydantic import BaseModel, Field


class WideSearchWorkboard(BaseModel):
    """CONCEPT:ORCH-1.1 — Pydantic-Native Shared Workboard.

    A thread-safe/merge-safe shared memory scratchpad for parallel workers
    during wide-search extraction tasks.
    """

    schema_definition: dict[str, Any] = Field(default_factory=dict)
    expected_row_count: int = 0
    row_slots: dict[str, dict[str, Any]] = Field(default_factory=dict)
    conflict_log: list[dict[str, Any]] = Field(default_factory=list)


class GraphResponse(BaseModel):
    status: str = "pending"
    results: dict[str, Any] = Field(default_factory=dict)
    mermaid: str | None = None
    usage: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


from agent_utilities.models.sdd import Task

ExecutionStep = Task


class ParallelBatch(BaseModel):
    tasks: list[Task] = Field(default_factory=list)


class GraphPlan(BaseModel):
    steps: list[ExecutionStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_acp_plan_entries(self) -> list[dict[str, str]]:
        _STATUS_MAP = {
            "pending": "pending",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "pending",
        }
        entries: list[dict[str, str]] = []
        for step in self.steps:
            entries.append(
                {
                    "content": (
                        f"{step.id}: {step.description}"
                        if step.description
                        else step.id
                    ),
                    "status": _STATUS_MAP.get(str(step.status), "pending"),
                    "priority": "high" if not step.parallel else "medium",
                }
            )
        return entries

    def to_mermaid(self, title: str = "Execution Plan") -> str:
        """Generate a Mermaid flowchart for the graph plan."""
        from agent_utilities.observability.mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=title)

        # Add nodes
        for step in self.steps:
            shape = "box" if not step.parallel else "round"
            # Highlight by status
            css_class = None
            status_str = str(step.status)
            if status_str == "completed":
                css_class = "success"
            elif status_str == "failed":
                css_class = "error"
            elif status_str == "in_progress":
                css_class = "active"

            builder.add_node(
                step.id,
                label=f"{step.id}\n{step.description or ''}",
                shape=shape,
                css_class=css_class,
            )

            # Add dependencies
            for dep in step.depends_on:
                builder.add_edge(dep, step.id)

        # Add styling
        builder.lines.append(
            "  classDef success fill:#2e7d32,stroke:#1b5e20,color:#fff"
        )
        builder.lines.append("  classDef error fill:#c62828,stroke:#b71c1c,color:#fff")
        builder.lines.append(
            "  classDef active fill:#1565c0,stroke:#0d47a1,color:#fff,stroke-width:2px"
        )

        return builder.render()
