"""`_ensure_ontology_graph` must treat the engine's own "already exists"
rejection as the PRIMARY reconciliation signal, not `tenants.list()`.

Mirrors the BUG-PE-049 fix already applied to
`GraphComputeEngine._ensure_local_session_graph` (see
`agent_utilities/knowledge_graph/core/graph_compute.py`,
`_is_graph_already_exists_error`): against a store that already holds the
ontology graph, `tenants.create` is rejected by the engine, but the bootstrap
verified context's `tenants.list()` does not report the graph back -- so a
list-only reconciliation fails closed on a graph that genuinely already
exists. `_is_graph_already_exists_error` pins the whole engine rejection
sentence, including the graph's own name, so an already-exists error naming a
*different* graph must not be adopted either.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology import lifecycle


class _TenantsAlreadyExistsButNotListed:
    """Simulates the exact bootstrap-context mismatch this fix addresses:

    the engine rejects `create` because the graph already exists, but
    `list()` (evaluated under the same verified context) does not report it.
    """

    def __init__(self, graph_name: str) -> None:
        self._graph_name = graph_name
        self.create_calls: list[tuple[str, str]] = []

    def list(self):
        return []  # Deliberately does NOT report the graph.

    def create(self, name: str, graph_type: str) -> None:
        self.create_calls.append((name, graph_type))
        raise RuntimeError(f"Graph '{self._graph_name}' already exists")


class _TenantsUnrelatedError:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str]] = []

    def list(self):
        return []

    def create(self, name: str, graph_type: str) -> None:
        self.create_calls.append((name, graph_type))
        raise RuntimeError("connection reset by peer")


class _TenantsAlreadyExistsDifferentGraph:
    """The rejection names a DIFFERENT graph than the one being provisioned --
    must not be adopted as success (would otherwise degrade to a blanket
    `except Exception: pass`)."""

    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str]] = []

    def list(self):
        return []

    def create(self, name: str, graph_type: str) -> None:
        self.create_calls.append((name, graph_type))
        raise RuntimeError("Graph 'some_other_tenant__ontology' already exists")


class _FakeClient:
    def __init__(self, tenants) -> None:
        self.tenants = tenants


class _FakeGraphComputeEngine:
    def __init__(self, tenants) -> None:
        self.client = _FakeClient(tenants)


def setup_function() -> None:
    lifecycle.reset_registry()


def teardown_function() -> None:
    lifecycle.reset_registry()


def test_already_exists_rejection_is_adopted_as_success():
    graph_name = "tenant__acme__ontology"
    tenants = _TenantsAlreadyExistsButNotListed(graph_name)
    gc = _FakeGraphComputeEngine(tenants)

    # Must not raise: the engine's "already exists" rejection IS the
    # guarantee this function exists to provide.
    lifecycle._ensure_ontology_graph(gc, graph_name)

    assert tenants.create_calls == [(graph_name, "Global")]
    assert graph_name in lifecycle._KNOWN_ONTOLOGY_GRAPHS


def test_unrelated_create_error_still_raises_ontology_error():
    graph_name = "tenant__acme__ontology"
    tenants = _TenantsUnrelatedError()
    gc = _FakeGraphComputeEngine(tenants)

    with pytest.raises(lifecycle.OntologyError):
        lifecycle._ensure_ontology_graph(gc, graph_name)

    assert graph_name not in lifecycle._KNOWN_ONTOLOGY_GRAPHS


def test_already_exists_error_naming_a_different_graph_still_raises():
    graph_name = "tenant__acme__ontology"
    tenants = _TenantsAlreadyExistsDifferentGraph()
    gc = _FakeGraphComputeEngine(tenants)

    with pytest.raises(lifecycle.OntologyError):
        lifecycle._ensure_ontology_graph(gc, graph_name)

    assert graph_name not in lifecycle._KNOWN_ONTOLOGY_GRAPHS
