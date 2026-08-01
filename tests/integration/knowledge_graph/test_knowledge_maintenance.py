"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock, patch

from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer


class DummyBackend:
    def __init__(self, execute_results=None):
        self.queries = []
        self.execute_results = execute_results or []
        self.idx = 0

    def execute(self, query: str, props: dict | None = None):
        self.queries.append({"query": query, "props": props})
        if self.idx < len(self.execute_results):
            res = self.execute_results[self.idx]
            self.idx += 1
            return res
        return []

    def add_embedding(self, node_id, embedding):
        self.queries.append({"action": "add_embedding", "id": node_id})


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
        mock_make_embed_fn.return_value = lambda texts: [[0.1, 0.2] for _ in texts]
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

    assert report == {"scanned": 0, "embedded": 0, "skipped_no_text": 0}
