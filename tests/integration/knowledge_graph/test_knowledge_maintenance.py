"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer
from agent_utilities.knowledge_graph.enrichment.semantic import (
    configured_embedding_dimension,
)

TEST_EMBEDDING_DIMENSION = configured_embedding_dimension()


def _embedding(value: float = 0.1) -> list[float]:
    return [value] * TEST_EMBEDDING_DIMENSION


class DummyBackend:
    def __init__(self, execute_results=None):
        self.queries = []
        self.execute_results = execute_results or []
        self.idx = 0
        self._props_by_id = {}
        self._graph = self

    def execute(self, query: str, props: dict | None = None):
        self.queries.append({"query": query, "props": props})
        if self.idx < len(self.execute_results):
            res = self.execute_results[self.idx]
            self.idx += 1
            self._props_by_id.update(
                {
                    str(row["id"]): dict(row.get("props") or {})
                    for row in res
                    if row.get("id")
                }
            )
            return res
        return []

    def _get_node_properties_batch(self, node_ids):
        return {node_id: self._props_by_id.get(node_id, {}) for node_id in node_ids}

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        self.queries.append(
            {
                "action": "compare_and_set_node_fields",
                "id": node_id,
                "conditions": conditions,
                "updates": updates,
            }
        )
        self._props_by_id.setdefault(node_id, {}).update(updates)
        return True

    def add_embedding(self, node_id, embedding):
        self.queries.append({"action": "add_embedding", "id": node_id})

    def compare_and_set_node_embedding(self, node_id, conditions, updates, embedding):
        if not self.compare_and_set_node_fields(node_id, conditions, updates):
            return False
        self.add_embedding(node_id, embedding)
        return True


def test_prune_cron_logs():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend

    maintainer = GraphMaintainer(engine)
    maintainer.prune_cron_logs(keep_days=30)

    assert len(backend.queries) == 1
    assert "DELETE l" in backend.queries[0]["query"]


def test_summarize_old_chats():
    # Return one thread, then two messages for that thread
    backend = DummyBackend(
        execute_results=[
            [{"id": "thread_1", "title": "Test Thread"}],
            [{"content": "hello"}, {"content": "world"}],
        ]
    )
    engine = MagicMock()
    engine.backend = backend

    maintainer = GraphMaintainer(engine)
    maintainer.summarize_old_chats(keep_days=30)

    # 1 query for threads, 1 for messages, 1 to create summary, 1 to link summary, 1 to delete old msgs
    assert len(backend.queries) == 5
    assert "ChatSummary" in backend.queries[2]["query"]


@patch(
    "agent_utilities.knowledge_graph.core.maintainer.generate_embedding",
    return_value=[0.1, 0.2, 0.3],
)
def test_enrich_embeddings(mock_generate_embedding):

    backend = DummyBackend(
        execute_results=[[{"id": "msg_1", "content": "hello", "embedding": None}]]
    )
    engine = MagicMock()
    engine.backend = backend

    maintainer = GraphMaintainer(engine)
    count = maintainer.enrich_embeddings()

    assert count == 1
    assert mock_generate_embedding.called
    assert any(q.get("action") == "add_embedding" for q in backend.queries)


def test_backfill_entity_embeddings_embeds_arbitrary_entity_types():
    """D-EMB: unlike enrich_embeddings (Message-only), this covers ANY node
    type -- the actual shape of the 26,680-node/136-embedded gap."""
    backend = DummyBackend(
        execute_results=[
            [
                {
                    "id": "incident-1",
                    "props": {
                        "id": "incident-1",
                        "type": "Incident",
                        "short_description": "disk full on host-3",
                        "description": "root partition is 98% full",
                    },
                },
                {
                    "id": "factsheet-1",
                    "props": {
                        "id": "factsheet-1",
                        "type": "Application",
                        "name": "billing-service",
                    },
                },
            ]
        ]
    )
    engine = MagicMock()
    engine.backend = backend

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn"
    ) as mock_make_embed_fn:
        mock_make_embed_fn.return_value = lambda texts: [_embedding() for _ in texts]
        maintainer = GraphMaintainer(engine)
        report = maintainer.backfill_entity_embeddings(limit=500, batch_size=64)

    assert report["scanned"] == 2
    assert report["embedded"] == 2
    assert report["skipped_no_text"] == 0
    add_embedding_ids = {
        q["id"] for q in backend.queries if q.get("action") == "add_embedding"
    }
    assert add_embedding_ids == {"incident-1", "factsheet-1"}


def test_backfill_entity_embeddings_skips_nodes_with_no_extractable_text():
    backend = DummyBackend(
        execute_results=[
            [{"id": "sensor-1", "props": {"id": "sensor-1", "reading": 42.0}}]
        ]
    )
    engine = MagicMock()
    engine.backend = backend

    maintainer = GraphMaintainer(engine)
    report = maintainer.backfill_entity_embeddings(limit=500)

    assert report["scanned"] == 1
    assert report["embedded"] == 0
    assert report["skipped_no_text"] == 1
    assert not any(q.get("action") == "add_embedding" for q in backend.queries)


def test_backfill_entity_embeddings_no_backend_returns_zeros():
    engine = MagicMock()
    engine.backend = None

    maintainer = GraphMaintainer(engine)
    report = maintainer.backfill_entity_embeddings()

    assert report == {
        "scanned": 0,
        "embedded": 0,
        "indexed": 0,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }


class _NativeBackfillBackend:
    """Engine-shaped store that rejects the unsupported properties(n) query."""

    def __init__(self):
        self._graph = self
        self.nodes = {
            f"node-{index}": {
                "id": f"node-{index}",
                "name": f"service {index}",
                "classification": "INTERNAL",
            }
            for index in range(4)
        }
        self.indexed = []

    def execute(self, query, props=None):
        assert "properties(n)" not in query
        limit = int((props or {})["limit"])
        return [
            {"id": node_id}
            for node_id, node_props in sorted(self.nodes.items())
            if node_props.get("embedding") is None
            and node_props.get("_embedding_backfill_state") is None
        ][:limit]

    def _get_node_properties_batch(self, node_ids):
        return {node_id: dict(self.nodes[node_id]) for node_id in node_ids}

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        node = self.nodes[node_id]
        if any(node.get(field) != expected for field, expected in conditions.items()):
            return False
        node.update(updates)
        return True

    def add_embedding(self, node_id, embedding):
        self.indexed.append((node_id, embedding))

    def compare_and_set_node_embedding(self, node_id, conditions, updates, embedding):
        if not self.compare_and_set_node_fields(node_id, conditions, updates):
            return False
        self.add_embedding(node_id, embedding)
        return True


def test_backfill_native_query_avoids_properties_function_and_persists_progress():
    backend = _NativeBackfillBackend()
    engine = MagicMock(backend=backend)

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding(float(len(text))) for text in texts],
    ):
        first = GraphMaintainer(engine).backfill_entity_embeddings(
            limit=2, batch_size=2
        )
        second = GraphMaintainer(engine).backfill_entity_embeddings(
            limit=2, batch_size=2
        )

    assert first == {
        "scanned": 2,
        "embedded": 2,
        "indexed": 2,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }
    assert second == {
        "scanned": 2,
        "embedded": 2,
        "indexed": 2,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }
    assert [node_id for node_id, _ in backend.indexed] == [
        "node-0",
        "node-1",
        "node-2",
        "node-3",
    ]
    assert all(node["classification"] == "INTERNAL" for node in backend.nodes.values())


def test_backfill_textless_first_node_does_not_starve_next_invocation():
    backend = _NativeBackfillBackend()
    backend.nodes["node-0"] = {
        "id": "node-0",
        "classification": "INTERNAL",
        "reading": 42.0,
    }
    engine = MagicMock(backend=backend)

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding(float(len(text))) for text in texts],
    ):
        first = GraphMaintainer(engine).backfill_entity_embeddings(limit=1)
        second = GraphMaintainer(engine).backfill_entity_embeddings(limit=1)

    assert first["scanned"] == 1
    assert first["skipped_no_text"] == 1
    assert first["deferred_no_text"] == 1
    assert backend.nodes["node-0"].get("embedding") is None
    assert backend.nodes["node-0"]["_embedding_backfill_state"] == "no_text"
    assert second["embedded"] == 1
    assert backend.indexed[0][0] == "node-1"


def test_backfill_rejects_concurrent_text_mutation_before_embedding_write():
    class _MutatingBackend(_NativeBackfillBackend):
        def compare_and_set_node_embedding(
            self, node_id, conditions, updates, embedding
        ):
            self.nodes[node_id]["name"] = "service changed concurrently"
            return super().compare_and_set_node_embedding(
                node_id, conditions, updates, embedding
            )

    backend = _MutatingBackend()
    backend.nodes = {"node-0": backend.nodes["node-0"]}
    engine = MagicMock(backend=backend)

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding(1.0) for _ in texts],
    ):
        report = GraphMaintainer(engine).backfill_entity_embeddings(limit=1)

    assert report["embedded"] == 0
    assert report["conflicted"] == 1
    assert backend.nodes["node-0"].get("embedding") is None
    assert backend.indexed == []


@pytest.mark.parametrize(
    "vectors",
    [
        [[]],
        [[float("nan")] * TEST_EMBEDDING_DIMENSION],
        [[1.0] * (TEST_EMBEDDING_DIMENSION - 1)],
    ],
    ids=["empty", "non-finite", "wrong-dimension"],
)
def test_backfill_rejects_invalid_vectors_before_any_property_write(vectors):
    backend = _NativeBackfillBackend()
    item_count = len(vectors)
    backend.nodes = dict(list(backend.nodes.items())[:item_count])
    engine = MagicMock(backend=backend)

    with (
        patch(
            "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
            return_value=lambda texts: vectors,
        ),
        pytest.raises(RuntimeError, match="embedding endpoint returned"),
    ):
        GraphMaintainer(engine).backfill_entity_embeddings(limit=item_count)

    assert all(node.get("embedding") is None for node in backend.nodes.values())
    assert backend.indexed == []
