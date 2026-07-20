from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

from ...models.knowledge_graph import PhaseResult, PipelineConfig
from ..backends.base import GraphBackend


class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: PipelineConfig
    graph: GraphComputeEngine = Field(default_factory=GraphComputeEngine.get_or_create)
    results: dict[str, PhaseResult] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend: GraphBackend | None = Field(
        default=None, description="Shared graph backend instance from the engine"
    )


class PipelinePhase(BaseModel):
    name: str
    deps: list[str] = Field(default_factory=list)
    execute_fn: Callable[[PipelineContext, dict[str, PhaseResult]], Awaitable[Any]]

    async def execute(self, ctx: PipelineContext, deps: dict[str, PhaseResult]) -> Any:
        return await self.execute_fn(ctx, deps)
