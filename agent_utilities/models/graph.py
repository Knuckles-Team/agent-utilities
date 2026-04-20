from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GraphResponse(BaseModel):
    status: str = "pending"
    results: Dict[str, Any] = Field(default_factory=dict)
    mermaid: Optional[str] = None
    usage: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionStep(BaseModel):
    node_id: str = Field(description="The ID of the functional step to execute")
    input_data: Optional[Any] = Field(None, description="Input data passed to the step")
    is_parallel: bool = Field(
        False, description="Whether this step starts a parallel batch"
    )
    status: str = Field("pending", description="Current execution status")
    timeout: float = Field(120.0, description="Per-node timeout in seconds")
    depends_on: List[str] = Field(
        default_factory=list, description="Node IDs that must complete before this step"
    )


class ParallelBatch(BaseModel):
    tasks: List[ExecutionStep] = Field(default_factory=list)


class GraphPlan(BaseModel):
    steps: List[ExecutionStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
