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

    # 1 query for threads, 1 for messages, 1 to create summary, 1 to delete old msgs.
    # Linking the summary to its thread goes through engine.link_nodes(), not a raw
    # backend.execute() Cypher query (a comma-pattern MATCH + edge MERGE exceeds the
    # engine's native Cypher write subset — see maintainer.py's summarize_old_chats),
    # so it is asserted separately below rather than counted in backend.queries.
    assert len(backend.queries) == 4
    assert "ChatSummary" in backend.queries[2]["query"]
    engine.link_nodes.assert_called_once()
    assert engine.link_nodes.call_args.args[1] == "thread_1"
    assert engine.link_nodes.call_args.args[2] == "PART_OF"


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
        "errored": 0,
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
        "errored": 0,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
        "aborted_early": False,
    }
    assert second == {
        "scanned": 2,
        "embedded": 2,
        "indexed": 2,
        "errored": 0,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
        "aborted_early": False,
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


# ---------------------------------------------------------------------------
# D-CDX-101: never swallow the atomic-commit failure cause.
# ---------------------------------------------------------------------------


def test_backfill_captures_full_cause_chain_on_atomic_commit_failure(caplog):
    """Reproduces the LIVE production shape: the backend's atomic
    property+ANN commit raises (the exact ``RuntimeError`` the live engine
    raised when ``EPISTEMIC_GRAPH_ENCRYPTION_KEY`` was unset). The prior
    version of this code caught the exception, logged ONLY
    ``type(exc).__name__`` ("RuntimeError") -- discarding the actual message
    -- and silently ``continue``d with no distinct counter, so the final
    report was indistinguishable from an ordinary run that embedded nothing
    because there was nothing to embed. This test fails against that
    restored bug on BOTH assertions: ``report["errored"]`` does not exist on
    the old result dict (KeyError), and the real message never reaches the
    log record.
    """

    class _RaisingBackend(_NativeBackfillBackend):
        def compare_and_set_node_embedding(
            self, node_id, conditions, updates, embedding
        ):
            raise RuntimeError(
                "transaction durability requires EPISTEMIC_GRAPH_ENCRYPTION_KEY "
                "to be configured"
            )

    backend = _RaisingBackend()
    backend.nodes = {"node-0": backend.nodes["node-0"]}
    engine = MagicMock(backend=backend)

    with (
        patch(
            "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
            return_value=lambda texts: [_embedding(1.0) for _ in texts],
        ),
        caplog.at_level("WARNING"),
    ):
        report = GraphMaintainer(engine).backfill_entity_embeddings(limit=1)

    assert report["embedded"] == 0
    assert report["indexed"] == 0
    assert report["errored"] == 1
    assert report["conflicted"] == 0  # NOT folded into the OCC-conflict bucket
    assert backend.nodes["node-0"].get("embedding") is None
    assert backend.indexed == []

    warning_text = " ".join(record.getMessage() for record in caplog.records)
    assert "EPISTEMIC_GRAPH_ENCRYPTION_KEY" in warning_text, (
        "the real exception message must reach the log, not just the "
        "exception's class name"
    )
    assert "RuntimeError" in warning_text
    assert "node-0" in warning_text


def test_backfill_isolates_one_failing_node_from_the_rest_of_the_batch():
    """A per-node atomic-commit failure must not abort the other N-1 rows
    already staged in this batch -- the documented per-node retry design.
    Partial success is a real, expected outcome (task point 4 / D-CDX-9) and
    must stay visible, not collapse into an all-or-nothing batch."""

    class _OneNodeRaisesBackend(_NativeBackfillBackend):
        def compare_and_set_node_embedding(
            self, node_id, conditions, updates, embedding
        ):
            if node_id == "node-1":
                raise RuntimeError("simulated ANN staging failure for node-1 only")
            return super().compare_and_set_node_embedding(
                node_id, conditions, updates, embedding
            )

    backend = _OneNodeRaisesBackend()
    engine = MagicMock(backend=backend)

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding(float(len(text))) for text in texts],
    ):
        report = GraphMaintainer(engine).backfill_entity_embeddings(limit=4)

    assert report["scanned"] == 4
    assert report["embedded"] == 3
    assert report["indexed"] == 3
    assert report["errored"] == 1
    assert [node_id for node_id, _ in backend.indexed] == [
        "node-0",
        "node-2",
        "node-3",
    ]
    assert backend.nodes["node-1"].get("embedding") is None


# ---------------------------------------------------------------------------
# D-CDX-102: exclude secret-bearing nodes from embedding candidacy BY
# CONSTRUCTION (sourced from secrets_client's own constants), never by a
# maintained denylist.
# ---------------------------------------------------------------------------


class _SecretsAwareBackend:
    """Engine-shaped store that actually HONORS the D-CDX-102 exclusion
    params the caller must pass, unlike ``_NativeBackfillBackend`` (which
    ignores everything but ``limit``) -- this is what proves the exclusion
    clause is real, not merely constructed and discarded."""

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes
        self._graph = self
        self.indexed: list[str] = []

    def execute(self, query, props=None):
        params = props or {}
        assert "embedding_backfill_excluded_graph" in params, (
            "D-CDX-102 regression: the candidate query must always pass the "
            "secrets-exclusion params, not merely construct them"
        )
        assert "embedding_backfill_excluded_label" in params
        excluded_graph = params["embedding_backfill_excluded_graph"]
        excluded_label = params["embedding_backfill_excluded_label"]
        limit = int(params["limit"])
        rows = [
            {"id": node_id}
            for node_id, props_ in sorted(self.nodes.items())
            if props_.get("embedding") is None
            and props_.get("_embedding_backfill_state") is None
            and props_.get("graph_name") != excluded_graph
            and props_.get("node_type") != excluded_label
        ]
        return rows[:limit]

    def _get_node_properties_batch(self, node_ids):
        return {node_id: dict(self.nodes[node_id]) for node_id in node_ids}

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        node = self.nodes[node_id]
        if any(node.get(field) != expected for field, expected in conditions.items()):
            return False
        node.update(updates)
        return True

    def compare_and_set_node_embedding(self, node_id, conditions, updates, embedding):
        if not self.compare_and_set_node_fields(node_id, conditions, updates):
            return False
        self.indexed.append(node_id)
        return True


def test_backfill_excludes_secrets_graph_and_secret_label_by_construction():
    from agent_utilities.security.secrets_client import SECRET_LABEL, SECRETS_GRAPH

    backend = _SecretsAwareBackend(
        {
            "safe-1": {
                "id": "safe-1",
                "name": "widget",
                "graph_name": "code:some-repo",
            },
            "secret-manifest-1": {
                "id": "secret-manifest-1",
                "source_uri": "/au/agent_utilities/prompts/python_programmer.json",
                "category": "prompt_base",
                "graph_name": SECRETS_GRAPH,
                "node_type": "IngestManifest",
            },
            "secret-node-2": {
                "id": "secret-node-2",
                "name": "some credential",
                "node_type": SECRET_LABEL,
            },
        }
    )
    engine = MagicMock(backend=backend)

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding() for _ in texts],
    ):
        report = GraphMaintainer(engine).backfill_entity_embeddings(limit=10)

    # Only the non-secret node was ever a candidate: the secrets-graph
    # manifest and the directly-labeled Secret node never entered `scanned`.
    assert report["scanned"] == 1
    assert report["embedded"] == 1
    assert backend.indexed == ["safe-1"]
    assert "secret-manifest-1" not in backend.indexed
    assert "secret-node-2" not in backend.indexed


def test_embedding_backfill_type_scope_clause_inlines_literal_list():
    """D-HYD-4 addendum, 2026-08-06: the type-scope clause is what lets a
    caller target the discovery-relevant corpus instead of the default blind
    ``ORDER BY n.id`` sweep. Values are inlined (not `$`-bound), matching the
    established pattern for this backend (``research/loop_controller.py``'s
    watermark query, ``retrieval/governance_rules.py``'s active-rule query)."""
    from agent_utilities.knowledge_graph.enrichment.semantic import (
        DISCOVERY_NODE_TYPES,
        embedding_backfill_type_scope_clause,
    )

    clause = embedding_backfill_type_scope_clause(["Tool", "Skill"])
    assert clause == "AND n.node_type IN ['Tool', 'Skill'] "

    # A caller-controlled alias is honored.
    assert embedding_backfill_type_scope_clause(["Tool"], alias="x").startswith(
        "AND x.node_type"
    )

    # Empty input -> no-op clause, never a malformed `IN []`.
    assert embedding_backfill_type_scope_clause([]) == ""

    # A stray quote is stripped rather than producing an unterminated literal.
    assert "O'Brien" not in embedding_backfill_type_scope_clause(["O'Brien"])

    # DISCOVERY_NODE_TYPES is the shared enum this whole addendum exists to
    # let callers target — assert its membership stays what find_tools /
    # find_relevant_callable_resources / delegation actually search over.
    assert set(DISCOVERY_NODE_TYPES) == {
        "Tool",
        "WorkflowDefinition",
        "Skill",
        "CallableResource",
        "Concept",
        "Prompt",
        "MCPServer",
        "NativeTool",
    }


def test_backfill_entity_embeddings_node_types_scopes_the_candidate_query():
    """``node_types`` must narrow the SAME query the default sweep uses, not
    replace it with a second code path that could drift."""
    backend = DummyBackend(
        execute_results=[
            [
                {
                    "id": "tool-1",
                    "props": {
                        "id": "tool-1",
                        "node_type": "Tool",
                        "name": "x",
                        "description": "y",
                    },
                }
            ]
        ]
    )
    engine = MagicMock()
    engine.backend = backend

    with patch(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        return_value=lambda texts: [_embedding() for _ in texts],
    ):
        report = GraphMaintainer(engine).backfill_entity_embeddings(
            limit=10, node_types=("Tool", "Skill")
        )

    assert report["scanned"] == 1
    assert report["embedded"] == 1
    candidate_query = backend.queries[0]["query"]
    assert "n.node_type IN ['Tool', 'Skill']" in candidate_query

    # The default (node_types=None) path is UNCHANGED: no type clause at all.
    backend2 = DummyBackend(execute_results=[[]])
    engine2 = MagicMock(backend=backend2)
    GraphMaintainer(engine2).backfill_entity_embeddings(limit=10)
    assert "node_type IN" not in backend2.queries[0]["query"]
