from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.query_router import (
    QueryRouter,
    QueryTier,
    QueryType,
)


def test_query_router_classification():
    router = QueryRouter()
    assert router._classify("MATCH (n) RETURN n") == QueryType.FILTERED_MATCH
    assert router._classify("pagerank(graph)") == QueryType.TOPOLOGICAL
    assert router._classify("SELECT ?s ?p ?o WHERE { ?s ?p ?o }") == QueryType.SPARQL
    assert router._classify("INSERT DATA { <A> <B> <C> }") == QueryType.MUTATION


def test_query_router_cost_heuristics():
    graph_mock = MagicMock()
    wsm_mock = MagicMock()
    wsm_mock.has_relevant_data.return_value = True

    router = QueryRouter(graph_engine=graph_mock, working_set_manager=wsm_mock)

    # 1. Freshness bypasses cache and hits L3
    tier = router._select_tier(
        QueryType.FILTERED_MATCH, expected_hops=1, requires_freshness=True
    )
    assert tier == QueryTier.L3_PERSISTENT

    # 2. Freshness for Topological hits L1
    tier = router._select_tier(
        QueryType.TOPOLOGICAL, expected_hops=2, requires_freshness=True
    )
    assert tier == QueryTier.L1_RUST

    # 3. Complex multi-hop hits L2 cache if available
    tier = router._select_tier(
        QueryType.FILTERED_MATCH, expected_hops=3, requires_freshness=False
    )
    assert tier == QueryTier.L2_CACHE

    # 4. Without WSM data, hits L1 rust
    wsm_mock.has_relevant_data.return_value = False
    tier = router._select_tier(
        QueryType.FILTERED_MATCH, expected_hops=3, requires_freshness=False
    )
    assert tier == QueryTier.L1_RUST


def test_query_router_execute():
    # Test L3 execution fallback
    backend_mock = MagicMock()
    backend_mock.execute.return_value = [{"a": 1}]

    router = QueryRouter(persistent_backend=backend_mock)
    res = router.route("MATCH (n) RETURN n", expected_hops=1)

    assert res == [{"a": 1}]
    backend_mock.execute.assert_called_once()
