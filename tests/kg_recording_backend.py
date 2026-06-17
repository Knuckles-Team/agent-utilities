"""Shared conformant test backend for the one materialization writer (KG-2.9).

``GraphBackend`` requires ``execute``/``execute_batch`` (both ``@abstractmethod``),
which is the single write path ``write_entities`` uses. This fake records those
UNWIND MERGE writes into ``.nodes`` / ``.edges`` so extractor/materialize tests can
assert on persisted content exactly as the old per-test ``add_node`` fakes did —
without each test re-mocking a non-conformant backend.
"""

from __future__ import annotations

import re
from typing import Any

from agent_utilities.knowledge_graph.backends.base import GraphBackend

_NODE_MERGE = re.compile(r"MERGE \(n:")
_EDGE_MERGE = re.compile(r"MERGE \(s\)-\[r:")


class RecordingGraphBackend(GraphBackend):
    """Records UNWIND node/edge MERGE writes into ``.nodes`` (id → props, id
    excluded) and ``.edges`` (list of ``(source, target, rel_type)``)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[Any, Any, Any]] = []
        self.edge_props: list[dict[str, Any]] = []

    def execute(self, query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        # content-hash delta prefetch → report stored hashes for known nodes.
        if "RETURN n.id AS id, n.content_hash AS h" in query:
            return [
                {"id": i, "h": self.nodes[i].get("content_hash")}
                for i in params.get("ids", [])
                if i in self.nodes and self.nodes[i].get("content_hash")
            ]
        # per-row (Ladybug-style) MERGE — record like the batch path.
        if _NODE_MERGE.search(query) and "id" in params:
            self.nodes[params["id"]] = {k: v for k, v in params.items() if k != "id"}
        elif _EDGE_MERGE.search(query) and "source" in params:
            self.edges.append(
                (params.get("source"), params.get("target"), params.get("type"))
            )
            self.edge_props.append({k: v for k, v in params.items()})
        return []

    def execute_batch(self, query: str, batch: list[dict]) -> list[dict[str, Any]]:
        if _NODE_MERGE.search(query):
            for row in batch:
                self.nodes[row["id"]] = {k: v for k, v in row.items() if k != "id"}
        elif _EDGE_MERGE.search(query):
            for row in batch:
                self.edges.append(
                    (row.get("source"), row.get("target"), row.get("type"))
                )
                self.edge_props.append({k: v for k, v in row.items()})
        return []

    # Some code paths (e.g. pipeline.py) write through add_node/add_edge directly
    # rather than the UNWIND writer; record those identically so one fake serves
    # both interfaces.
    def add_node(self, node_id: Any, **props: Any) -> None:
        self.nodes[node_id] = {k: v for k, v in props.items()}

    def add_edge(self, source: Any, target: Any, **props: Any) -> None:
        self.edges.append((source, target, props.get("rel_type")))
        self.edge_props.append({k: v for k, v in props.items()})

    def add_embedding(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        ...

    def create_schema(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        ...

    def prune(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        ...

    def semantic_search(
        self, *args: Any, **kwargs: Any
    ) -> list[Any]:  # pragma: no cover
        return []

    def close(self) -> None:  # pragma: no cover
        ...
