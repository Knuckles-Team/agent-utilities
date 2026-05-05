from typing import Any

from pydantic import BaseModel, Field


class WideSearchWorkboard(BaseModel):
    """CONCEPT:AU-055 — Pydantic-Native Shared Workboard.

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


class ExecutionStep(BaseModel):
    node_id: str = Field(description="The ID of the functional step to execute")
    input_data: Any | None = Field(
        default=None, description="Input data passed to the step"
    )
    refined_subtask: str | None = Field(
        default=None,
        description=(
            "CONCEPT:ORCH-1.1 — A focused natural-language instruction crafted "
            "by the router/planner for this specific specialist. When present, "
            "the executor uses this instead of the raw user query as the "
            "specialist's primary task description. Inspired by the RL "
            "Conductor's per-step subtask specification (Nielsen et al., ICLR 2026)."
        ),
    )
    is_parallel: bool = Field(
        default=False, description="Whether this step starts a parallel batch"
    )
    status: str = Field(default="pending", description="Current execution status")
    timeout: float = Field(default=120.0, description="Per-node timeout in seconds")
    depends_on: list[str] = Field(
        default_factory=list, description="Node IDs that must complete before this step"
    )
    access_list: list[str] = Field(
        default_factory=list,
        description=(
            "CONCEPT:ORCH-1.3 — Execution Visibility Graph. Controls which prior "
            "step results are injected into this specialist's context. An empty "
            "list means no prior results are shared. Use ['all'] to inject the "
            "entire results_registry. Specific step node_ids (e.g., ['researcher', "
            "'architect']) inject only those results. Inspired by the RL Conductor's "
            "access_list per workflow step (Nielsen et al., ICLR 2026)."
        ),
    )


class ParallelBatch(BaseModel):
    tasks: list[ExecutionStep] = Field(default_factory=list)


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
                        f"{step.node_id}: {step.input_data}"
                        if step.input_data
                        else step.node_id
                    ),
                    "status": _STATUS_MAP.get(step.status, "pending"),
                    "priority": "high" if not step.is_parallel else "medium",
                }
            )
        return entries

    def to_mermaid(self, title: str = "Execution Plan") -> str:
        """Generate a Mermaid flowchart for the graph plan."""
        from agent_utilities.observability.mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=title)

        # Add nodes
        for step in self.steps:
            shape = "box" if not step.is_parallel else "round"
            # Highlight by status
            css_class = None
            if step.status == "completed":
                css_class = "success"
            elif step.status == "failed":
                css_class = "error"
            elif step.status == "in_progress":
                css_class = "active"

            builder.add_node(
                step.node_id,
                label=f"{step.node_id}\n{step.input_data or ''}",
                shape=shape,
                css_class=css_class,
            )

            # Add dependencies
            for dep in step.depends_on:
                builder.add_edge(dep, step.node_id)

        # Add styling
        builder.lines.append(
            "  classDef success fill:#2e7d32,stroke:#1b5e20,color:#fff"
        )
        builder.lines.append("  classDef error fill:#c62828,stroke:#b71c1c,color:#fff")
        builder.lines.append(
            "  classDef active fill:#1565c0,stroke:#0d47a1,color:#fff,stroke-width:2px"
        )

        return builder.render()
