from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

from ...models.knowledge_graph import PhaseResult, PipelineConfig
from ..backends.base import GraphBackend


class PipelineContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    config: PipelineConfig
    graph: GraphComputeEngine = Field(default_factory=GraphComputeEngine)
    results: dict[str, PhaseResult] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend: GraphBackend | None = Field(
        default=None, description="Shared graph backend instance from the engine"
    )

    def __init__(self, **data: Any) -> None:
        if "nx_graph" in data and "graph" not in data:
            data["graph"] = data.pop("nx_graph")
        super().__init__(**data)

    @property
    def nx_graph(self) -> GraphComputeEngine:
        return self.graph

    @nx_graph.setter
    def nx_graph(self, val: GraphComputeEngine) -> None:
        self.graph = val


class PipelinePhase(BaseModel):
    name: str
    deps: list[str] = Field(default_factory=list)
    execute_fn: Callable[[PipelineContext, dict[str, PhaseResult]], Awaitable[Any]]

    async def execute(self, ctx: PipelineContext, deps: dict[str, PhaseResult]) -> Any:
        return await self.execute_fn(ctx, deps)
