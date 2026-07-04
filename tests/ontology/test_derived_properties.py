#!/usr/bin/python
from __future__ import annotations

"""Tests for derived (function-backed / computed) properties (CONCEPT:KG-2.40).

Exercises all four backings: FUNCTION + EMBEDDING end-to-end (offline, via the
real FunctionRuntime and a real CapabilityIndex), CYPHER dispatch against a
small in-test facade, and the graceful degradation + caching/invalidation
contract. No backend services required.
"""

from agent_utilities.knowledge_graph.ontology.derived_properties import (
    DEFAULT_DERIVED_ENGINE,
    DEFAULT_DERIVED_REGISTRY,
    DerivedBacking,
    DerivedProperty,
    DerivedPropertyEngine,
    DerivedPropertyRegistry,
    EmbeddingDerivation,
    compute_derived,
)
from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex
from agent_utilities.numeric import xp as np


# ---------------------------------------------------------------------------
# Small in-test facade
# ---------------------------------------------------------------------------
class _FakeFacade:
    """Minimal KnowledgeGraph-shaped facade for CYPHER/EMBEDDING dispatch."""

    def __init__(self, rows=None, index=None):
        self._rows = rows or []
        self.retrieval = (
            index if index is not None else CapabilityIndex(prefer_backend="numpy")
        )
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params))
        return list(self._rows)


def _unit_index(dim=8):
    idx = CapabilityIndex(dim=dim, prefer_backend="numpy")
    a = np.zeros(dim, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(dim, dtype=np.float32)
    b[1] = 1.0
    idx.add("retriever-1", a.tolist(), {"retrieval"})
    idx.add("planner-1", b.tolist(), {"planning"})
    return idx, a, b


# ---------------------------------------------------------------------------
# Registry built-ins are live (not an empty shell)
# ---------------------------------------------------------------------------
def test_default_registry_has_real_builtins():
    names = {p.name for p in DEFAULT_DERIVED_REGISTRY.list_all()}
    assert {"summary", "degree", "nearest_capability"} <= names
    backings = {p.backing for p in DEFAULT_DERIVED_REGISTRY.list_all()}
    assert DerivedBacking.FUNCTION in backings
    assert DerivedBacking.CYPHER in backings
    assert DerivedBacking.EMBEDDING in backings


# ---------------------------------------------------------------------------
# FUNCTION backing — end-to-end via the real FunctionRuntime (offline)
# ---------------------------------------------------------------------------
def test_function_backing_end_to_end():
    obj = {
        "id": "obj-1",
        "name": "Acme Corp",
        "type": "Account",
        "description": "A wholesale supplier",
    }
    res = compute_derived(obj, "summary")
    assert res.ok, res.error
    assert res.backing == DerivedBacking.FUNCTION
    # object.summarize renders name (type): description
    assert "Acme Corp" in res.value
    assert isinstance(res.value, str)
    assert res.audit_ref  # audited


def test_function_backing_with_input_map():
    engine = DerivedPropertyEngine(registry=DerivedPropertyRegistry())
    prop = DerivedProperty(
        name="total",
        output_type="double",
        backing=DerivedBacking.FUNCTION,
        function_name="numeric.aggregate",
        function_version="1.0.0",
        input_map={"values": "amounts"},
        static_inputs={"op": "sum"},
    )
    obj = {"id": "inv-9", "amounts": [10.0, 20.5, 4.5]}
    res = engine.compute(obj, prop)
    assert res.ok, res.error
    assert res.value == 35.0
    assert isinstance(res.value, float)


# ---------------------------------------------------------------------------
# EMBEDDING backing — end-to-end via a real CapabilityIndex (offline)
# ---------------------------------------------------------------------------
def test_embedding_backing_nearest_label_offline():
    idx, vec_a, _vec_b = _unit_index()
    facade = _FakeFacade(index=idx)
    # Deterministic offline embedding: object text -> vector A direction.
    engine = DerivedPropertyEngine(
        registry=DerivedPropertyRegistry(),
        embedding_fn=lambda _text: vec_a.tolist(),
    )
    prop = DerivedProperty(
        name="nearest_cap",
        output_type="string",
        backing=DerivedBacking.EMBEDDING,
        embedding_text_property="description",
        embedding_derivation=EmbeddingDerivation.NEAREST_LABEL,
    )
    obj = {"id": "task-1", "description": "find documents about X"}
    res = engine.compute(obj, prop, facade)
    assert res.ok, res.error
    assert res.value == "retrieval"  # nearest to vector A == retriever-1


def test_embedding_backing_similarity_and_precomputed_vector():
    idx, vec_a, _vec_b = _unit_index()
    facade = _FakeFacade(index=idx)
    engine = DerivedPropertyEngine(registry=DerivedPropertyRegistry())
    prop = DerivedProperty(
        name="cap_score",
        output_type="double",
        backing=DerivedBacking.EMBEDDING,
        embedding_vector_property="vec",
        embedding_derivation=EmbeddingDerivation.SIMILARITY,
    )
    obj = {"id": "task-2", "vec": vec_a.tolist()}
    res = engine.compute(obj, prop, facade)
    assert res.ok, res.error
    # Cosine of identical unit vectors ~ 1.0.
    assert res.value > 0.99


def test_embedding_backing_degrades_with_empty_index():
    facade = _FakeFacade(index=CapabilityIndex(prefer_backend="numpy"))
    engine = DerivedPropertyEngine(
        registry=DerivedPropertyRegistry(),
        embedding_fn=lambda _t: [1.0] * 8,
    )
    prop = DerivedProperty(
        name="nearest_cap",
        output_type="string",
        backing=DerivedBacking.EMBEDDING,
        embedding_text_property="description",
    )
    res = engine.compute({"id": "x", "description": "hello"}, prop, facade)
    assert res.ok
    assert res.value is None  # graceful degradation, not an error


# ---------------------------------------------------------------------------
# CYPHER backing — dispatch against the small facade
# ---------------------------------------------------------------------------
def test_cypher_backing_dispatch_against_facade():
    facade = _FakeFacade(rows=[{"degree": 7}])
    res = DEFAULT_DERIVED_ENGINE.compute(
        {"id": "node-1", "type": "Account"}, "degree", facade, use_cache=False
    )
    assert res.ok, res.error
    assert res.backing == DerivedBacking.CYPHER
    assert res.value == 7
    # The object id was bound into the query params.
    assert facade.queries
    _cypher, params = facade.queries[0]
    assert params == {"id": "node-1"}


def test_cypher_backing_degrades_without_facade():
    res = DEFAULT_DERIVED_ENGINE.compute(
        {"id": "node-2"}, "degree", graph=None, use_cache=False
    )
    assert res.ok
    assert res.value is None  # no facade -> documented graceful degradation


# ---------------------------------------------------------------------------
# SPARQL backing — dispatch + graceful degradation
# ---------------------------------------------------------------------------
class _SparqlFacade:
    def __init__(self, rows):
        class _Sem:
            def __init__(self, rows):
                self._rows = rows
                self.seen = []

            def query_sparql(self, sparql):
                self.seen.append(sparql)
                return list(self._rows)

        self.semantic = _Sem(rows)


def test_sparql_backing_dispatch():
    facade = _SparqlFacade(rows=[{"label": "Supplier"}])
    engine = DerivedPropertyEngine(registry=DerivedPropertyRegistry())
    prop = DerivedProperty(
        name="rdf_label",
        output_type="string",
        backing=DerivedBacking.SPARQL,
        expression="SELECT ?label WHERE { ?id rdfs:label ?label } LIMIT 1",
    )
    res = engine.compute({"id": "iri-1"}, prop, facade)
    assert res.ok, res.error
    assert res.value == "Supplier"
    assert facade.semantic.seen  # the bridge was invoked


def test_sparql_backing_degrades_without_semantic_layer():
    engine = DerivedPropertyEngine(registry=DerivedPropertyRegistry())
    prop = DerivedProperty(
        name="rdf_label",
        output_type="string",
        backing=DerivedBacking.SPARQL,
        expression="SELECT ?l WHERE { ?id rdfs:label ?l }",
    )
    res = engine.compute({"id": "iri-2"}, prop, graph=_FakeFacade())
    assert res.ok
    assert res.value is None


# ---------------------------------------------------------------------------
# Caching + explicit invalidation
# ---------------------------------------------------------------------------
def test_cache_and_invalidation():
    facade = _FakeFacade(rows=[{"degree": 3}])
    engine = DerivedPropertyEngine(registry=DEFAULT_DERIVED_REGISTRY)
    obj = {"id": "n-cache", "type": "Account"}

    first = engine.compute(obj, "degree", facade)
    assert first.ok and not first.cached
    assert first.value == 3
    n_queries = len(facade.queries)

    second = engine.compute(obj, "degree", facade)
    assert second.cached  # served from cache
    assert second.value == 3
    assert len(facade.queries) == n_queries  # no new backend hit

    removed = engine.invalidate("degree", object_id="n-cache")
    assert removed == 1

    facade._rows = [{"degree": 9}]
    third = engine.compute(obj, "degree", facade)
    assert not third.cached
    assert third.value == 9  # recomputed after invalidation


def test_invalidate_object_drops_all_props():
    engine = DerivedPropertyEngine(registry=DEFAULT_DERIVED_REGISTRY)
    obj = {"id": "obj-multi", "name": "Thing", "type": "Account"}
    engine.compute(obj, "summary")  # FUNCTION cached
    assert engine.cache_size() >= 1
    dropped = engine.invalidate_object("obj-multi")
    assert dropped >= 1
    assert engine.cache_size() == 0


def test_unknown_property_returns_error_not_raise():
    res = DEFAULT_DERIVED_ENGINE.compute({"id": "z"}, "does_not_exist")
    assert not res.ok
    assert "unknown derived property" in res.error


def test_output_coercion_to_declared_type():
    engine = DerivedPropertyEngine(registry=DerivedPropertyRegistry())
    # numeric.aggregate returns float; declare output as long -> coerced to int.
    prop = DerivedProperty(
        name="count_long",
        output_type="long",
        backing=DerivedBacking.FUNCTION,
        function_name="numeric.aggregate",
        function_version="1.0.0",
        input_map={"values": "vals"},
        static_inputs={"op": "count"},
    )
    res = engine.compute({"id": "c", "vals": [1, 2, 3, 4]}, prop)
    assert res.ok, res.error
    assert res.value == 4
    assert isinstance(res.value, int)
