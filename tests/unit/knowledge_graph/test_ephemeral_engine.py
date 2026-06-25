#!/usr/bin/python
"""Proof: engine-backed tests run against the REAL ephemeral engine (CONCEPT:KG-2.238).

USER DIRECTIVE: no SQLite for niche test use cases — tests use our real DB,
deployed ephemerally and destroyed after. These tests exercise the
``tiny_engine`` (session, one real ``epistemic-graph-server``) + ``engine_graph``
(function, a fresh isolated tenant per test) fixtures and assert:

* a node round-trip persists + reads back through the real engine;
* a tenant op (create/use/purge) works;
* per-test isolation — test A's data is NOT visible in test B (the tenant-purge
  teardown gives each test a clean graph).

``@pytest.mark.engine`` opts the test into the real database. On a box with no
binary AND no Rust toolchain the fixtures skip with a clear message.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.engine, pytest.mark.concept("KG-2.238")]


def test_node_roundtrip_against_real_engine(engine_graph):
    """A node added through the real engine reads back with its properties."""
    compute = engine_graph
    compute.add_node("alpha", {"type": "Agent", "score": 7})

    assert compute.has_node("alpha") is True
    assert compute.node_count() >= 1

    props = compute._client.nodes.properties("alpha")
    assert props.get("type") == "Agent"
    assert props.get("score") == 7


def test_edge_roundtrip_against_real_engine(engine_graph):
    """A directed edge added through the real engine is queryable."""
    compute = engine_graph
    compute.add_node("a", {"type": "Node"})
    compute.add_node("b", {"type": "Node"})
    compute.add_edge("a", "b", {"weight": 1.5})

    assert compute.has_node("a") is True
    assert compute.has_node("b") is True
    assert compute.has_edge("a", "b") is True


# --- per-test isolation: a unique sentinel written in test A must NOT be visible
# in test B. Each test gets its OWN tenant graph (tenant-purged on teardown), so a
# node from one never leaks into another. ---

_ISOLATION_NODE = "isolation_sentinel_node"


def test_isolation_a_writes_sentinel(engine_graph):
    """Test A: write the sentinel into THIS test's fresh tenant graph."""
    compute = engine_graph
    assert compute.has_node(_ISOLATION_NODE) is False  # fresh graph
    compute.add_node(_ISOLATION_NODE, {"written_by": "test_a"})
    assert compute.has_node(_ISOLATION_NODE) is True


def test_isolation_b_does_not_see_a_sentinel(engine_graph):
    """Test B: the sentinel from test A is INVISIBLE — a different, fresh tenant."""
    compute = engine_graph
    # If isolation failed (shared graph / leaked state), this node would exist.
    assert compute.has_node(_ISOLATION_NODE) is False
    assert compute.node_count() == 0
