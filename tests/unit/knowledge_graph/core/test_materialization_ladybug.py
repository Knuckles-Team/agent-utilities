"""Regression coverage for the Ladybug materialization write path."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.materialization import write_entities
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from tests.kg_recording_backend import RecordingGraphBackend


class LadybugBackend(RecordingGraphBackend):
    """Recording Ladybug-shaped backend that rejects raw write execution."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_queries: list[str] = []

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        del include_epistemic
        if "MERGE" in query or "CREATE" in query:
            raise AssertionError("Ladybug writes must use execute_batch")
        return super().execute(query, params)

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.batch_queries.append(query)
        return super().execute_batch(query, batch)


def test_write_entities_uses_ladybug_batch_path_for_relationships() -> None:
    backend = LadybugBackend()
    actor = ActorContext(
        actor_id="writer:alice",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-a",
        authenticated=True,
    )

    with use_actor(actor):
        result = write_entities(
            backend,
            "materialization-test",
            [
                {"id": "source", "node_type": "Code", "name": "source"},
                {"id": "target", "node_type": "Code", "name": "target"},
            ],
            [
                {
                    "source": "source",
                    "target": "target",
                    "relationship": "calls",
                }
            ],
            delta=False,
        )

    assert result["nodes"] == 2
    assert result["edges"] == 1
    assert backend.edges == [("source", "target", "calls")]
    assert len(backend.batch_queries) == 3
    assert (
        "MATCH (s {id: $source}) MATCH (t {id: $target})" in backend.batch_queries[-1]
    )


def test_write_entities_persists_ladybug_relationship(tmp_path) -> None:
    pytest.importorskip("ladybug")
    from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
        LadybugBackend,
    )

    backend = LadybugBackend(db_path=str(tmp_path / "materialization.db"))
    actor = ActorContext(
        actor_id="writer:alice",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-a",
        authenticated=True,
    )
    try:
        with use_actor(actor):
            result = write_entities(
                backend,
                "materialization-test",
                [
                    {"id": "source", "node_type": "Code", "name": "source"},
                    {"id": "target", "node_type": "Code", "name": "target"},
                ],
                [
                    {
                        "source": "source",
                        "target": "target",
                        "relationship": "calls",
                    }
                ],
                delta=False,
            )
        rows = backend.execute(
            "MATCH (s:Code)-[:calls]->(t:Code) RETURN s.id AS source, t.id AS target"
        )
    finally:
        backend.close()

    assert result["edges"] == 1
    assert rows == [{"source": "source", "target": "target"}]
