"""Live source adapters for operation-shaped sources.

API, MCP, A2A and GraphQL sources expose entities through operations. An
:class:`OperationAdapter` calls the operation that discovery recorded for
the mapped entity. It pushes a key filter only when the connection declares
``filter:in``. The adapter always re-applies the key filter locally, so a
source that ignores or widens the filter cannot change the answer.

SQL, Iceberg and Teradata-style sources bind through EG OBDA named virtual
graphs instead (EG-UNIFIED-DATA-PLANE-R005); they need no AU adapter here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities.knowledge_graph.virtual_graph.contracts import (
    MetadataContract,
    SourceConnection,
    VirtualMapping,
)

Row = Mapping[str, Any]
#: ``(operation, arguments) -> rows``: an MCP tool call, A2A skill call,
#: GraphQL query or REST request bound by the caller.
OperationCall = Callable[[str, Mapping[str, Any]], Awaitable[Sequence[Row]]]


@dataclass
class OperationAdapter:
    """Reads mapped entities through their discovered operations."""

    connection: SourceConnection
    contract: MetadataContract
    call: OperationCall

    async def fetch(
        self,
        mapping: VirtualMapping,
        field: str | None = None,
        values: Sequence[Any] | None = None,
    ) -> list[Row]:
        entity = self.contract.entity(mapping.entity)
        if entity is None:
            raise LookupError(f"undiscovered entity {mapping.entity!r}")
        args: dict[str, Any] = {}
        if field is not None and self.connection.supports("filter:in"):
            args = {"field": field, "in": list(values or ())}
        rows = list(await self.call(entity.operation, args))
        if field is None:
            return rows
        wanted = set(values or ())
        return [row for row in rows if row.get(field) in wanted]


__all__ = ["OperationAdapter", "OperationCall"]
