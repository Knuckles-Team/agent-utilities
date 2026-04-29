from unittest.mock import MagicMock, patch

from agent_utilities.knowledge_graph.maintainer import GraphMaintainer


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


@patch("agent_utilities.knowledge_graph.maintainer.requests.post")
def test_enrich_embeddings(mock_post):
    # Mock LM Studio response
    mock_post.return_value.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    mock_post.return_value.raise_for_status = MagicMock()

    backend = DummyBackend(
        execute_results=[[{"id": "msg_1", "content": "hello", "embedding": None}]]
    )
    engine = MagicMock()
    engine.backend = backend

    maintainer = GraphMaintainer(engine)
    count = maintainer.enrich_embeddings()

    assert count == 1
    assert mock_post.called
    assert any(q.get("action") == "add_embedding" for q in backend.queries)
