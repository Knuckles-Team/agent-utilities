from typing import Any

from pydantic import BaseModel, Field


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
    is_parallel: bool = Field(
        default=False, description="Whether this step starts a parallel batch"
    )
    status: str = Field(default="pending", description="Current execution status")
    timeout: float = Field(default=120.0, description="Per-node timeout in seconds")
    depends_on: list[str] = Field(
        default_factory=list, description="Node IDs that must complete before this step"
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
