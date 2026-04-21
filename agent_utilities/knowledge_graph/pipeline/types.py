from collections.abc import Awaitable, Callable
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from ...models.knowledge_graph import PhaseResult, PipelineConfig
from ..backends.base import GraphBackend


class PipelineContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    config: PipelineConfig
    nx_graph: nx.MultiDiGraph = Field(default_factory=nx.MultiDiGraph)
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
