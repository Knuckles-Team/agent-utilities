from typing import Any, Dict, List, Callable, Awaitable, Optional
from pydantic import BaseModel, Field
import networkx as nx
from ...models.knowledge_graph import PipelineConfig, PhaseResult
from ..backends.base import GraphBackend


class PipelineContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    config: PipelineConfig
    nx_graph: nx.MultiDiGraph = Field(default_factory=nx.MultiDiGraph)
    results: Dict[str, PhaseResult] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    backend: Optional[GraphBackend] = Field(
        default=None, description="Shared graph backend instance from the engine"
    )


class PipelinePhase(BaseModel):
    name: str
    deps: List[str] = Field(default_factory=list)
    execute_fn: Callable[[PipelineContext, Dict[str, PhaseResult]], Awaitable[Any]]

    async def execute(self, ctx: PipelineContext, deps: Dict[str, PhaseResult]) -> Any:
        return await self.execute_fn(ctx, deps)
