from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage backends (e.g., Ladybug, Neo4j)."""

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query against the backend."""
        ...

    def create_schema(self) -> None:
        """Initialize required database schema."""
        ...

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add an embedding vector to a specific node."""
        ...

    def prune(self, criteria: dict[str, Any]) -> None:
        """Run pruning logic based on criteria."""
        ...


@runtime_checkable
class AgentInterface(Protocol):
    """Protocol for specialist agents."""

    name: str
    description: str

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        """Execute the agent with a given prompt."""
        ...


@runtime_checkable
class ToolInterface(Protocol):
    """Protocol for tools."""

    name: str
    description: str

    async def call(self, **kwargs: Any) -> Any:
        """Call the tool with provided arguments."""
        ...
