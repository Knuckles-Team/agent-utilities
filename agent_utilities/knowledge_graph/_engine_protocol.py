from __future__ import annotations

"""Protocol defining the engine interface expected by all mixins.

This module is used under ``TYPE_CHECKING`` to give mypy visibility
into the attributes that the composed ``IntelligenceGraphEngine`` provides
to each mixin.  At runtime this is never imported, avoiding circularity.
"""


from typing import Any, Protocol

from .backends.base import GraphBackend
from .core.graph_compute import GraphComputeEngine


class _EngineProtocol(Protocol):
    """Structural typing contract for IntelligenceGraphEngine.

    All mixin classes should declare ``if TYPE_CHECKING`` blocks that
    reference this protocol so mypy understands the available attributes.
    """

    graph: GraphComputeEngine
    backend: GraphBackend | None

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        ...

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]) -> None:
        ...

    def _get_set_clause(
        self, data: dict[str, Any], alias: str = "n", label: str | None = None
    ) -> str:
        ...

    def _get_allowed_columns(self, label: str) -> list[str]:
        ...

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        ...

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> Any:
        ...

    # Query mixin methods (used by AHE and Registry)
    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        ...

    def search_hybrid(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        ...

    # Hybrid retriever (set in __init__)
    hybrid_retriever: Any
    inference_engine: Any
