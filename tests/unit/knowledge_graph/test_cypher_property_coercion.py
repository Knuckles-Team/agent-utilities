"""Regression: Cypher mirror backends must not send Map/nested property values.

Neo4j/FalkorDB raise ``Neo.ClientError.Statement.TypeError`` ("Property values can
only be of primitive types or arrays thereof") on a Map (dict) property — and on a
fan-out *mirror* that stalls replication forever (the outbox entry at the bad seq
retries every cycle, dragging the write path). ``coerce_cypher_property`` serializes
such values to a lossless JSON string while leaving primitives + primitive arrays
(embedding vectors) intact. (CONCEPT:AU-KG.backend.mirror-health-repair)
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.backends.base import coerce_cypher_property


@pytest.mark.concept("AU-KG.backend.mirror-health-repair")
def test_map_property_is_serialized():
    # The exact value that stalled prod-neo4j / team-falkor at outbox seq 481.
    v = {"version": "0.1.0", "author": "Genius"}
    out = coerce_cypher_property(v)
    assert isinstance(out, str)
    assert json.loads(out) == v  # lossless round-trip


@pytest.mark.concept("AU-KG.backend.mirror-health-repair")
@pytest.mark.parametrize("v", ["s", 1, 1.5, True, False, None])
def test_primitives_pass_through(v):
    assert coerce_cypher_property(v) == v


@pytest.mark.concept("AU-KG.backend.mirror-health-repair")
def test_primitive_array_is_preserved():
    # Embedding vectors MUST stay a list (Neo4j/FalkorDB accept arrays of primitives
    # and index them) — never serialized to a string.
    emb = [0.1, 0.2, 0.3]
    assert coerce_cypher_property(emb) == emb
    assert coerce_cypher_property(["a", "b"]) == ["a", "b"]


@pytest.mark.concept("AU-KG.backend.mirror-health-repair")
def test_array_of_maps_is_serialized():
    v = [{"a": 1}, {"b": 2}]
    out = coerce_cypher_property(v)
    assert isinstance(out, str)
    assert json.loads(out) == v


@pytest.mark.concept("AU-KG.backend.mirror-health-repair")
def test_bytes_become_text():
    assert coerce_cypher_property(b"hello") == "hello"
