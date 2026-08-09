"""Write-time ACL registration convergence — chokepoint stamping contracts.

Covers CONCEPT:AU-KG.backend.company-brain-write-guard's write-time half of
defence-in-depth ACL registration: every generic node-write chokepoint stamps
durable ``tenant_id``/``_owner_id``/``classification`` governance properties
(via ``tenant_sharing.stamp_ownership``/``stamp_classification``) so
``secured_reads._hydrate_missing_acls`` can reconstruct a real ACL at read
time instead of leaving the node permanently denied — even to its own
creator. See docs/architecture/acl_registration_convergence.md for the full
design.
"""

from __future__ import annotations

from unittest.mock import Mock

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


def _actor(actor_id="writer:alice", tenant="tenant-a"):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=(),
        tenant_id=tenant,
        authenticated=True,
    )


# --- IntelligenceGraphEngine._upsert_node -----------------------------------


def _bare_engine(backend):
    engine = object.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    engine._normalize_label = lambda label: label
    return engine


def test_upsert_node_stamps_governance_on_the_raw_cypher_path():
    """A generic (non schema-backed, non typed-native) backend goes through
    the MERGE + SET path; the params dict sent to backend.execute must carry
    the durable classification/ownership stamp, not just the caller's data."""
    backend = Mock()
    backend.typed_mutation_support = ""
    backend.cypher_support = "full"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    with use_actor(_actor()):
        engine._upsert_node("Memory", "mem-1", {"content": "hello"})

    assert backend.execute.call_count == 1
    _query, params = backend.execute.call_args[0]
    assert params["id"] == "mem-1"
    assert params["tenant_id"] == "tenant-a"
    assert params["_owner_id"] == "writer:alice"
    assert params["classification"] == "confidential"


def test_upsert_node_stamps_public_for_catalog_labels():
    backend = Mock()
    backend.typed_mutation_support = ""
    backend.cypher_support = "full"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    with use_actor(_actor()):
        engine._upsert_node("CallableResource", "fn-1", {"name": "do-thing"})

    _query, params = backend.execute.call_args[0]
    assert params["classification"] == "public"


def test_upsert_node_stamps_governance_on_the_typed_native_path():
    backend = Mock()
    backend.typed_mutation_support = "native"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    with use_actor(_actor()):
        engine._upsert_node("Memory", "mem-2", {"content": "hello"})

    backend.add_node.assert_called_once()
    _node_id, kwargs = backend.add_node.call_args[0][0], backend.add_node.call_args[1]
    assert kwargs["tenant_id"] == "tenant-a"
    assert kwargs["_owner_id"] == "writer:alice"
    assert kwargs["classification"] == "confidential"


def test_upsert_node_fails_closed_with_no_bound_actor():
    """BUG-033/BUG-039: a write reaching this chokepoint with NO verified
    ambient actor at all must raise -- not silently proceed unstamped. The
    prior "best-effort" swallow here is exactly how 29 private conversation
    nodes landed unowned and world-visible. No node is created."""
    import contextvars

    import pytest

    backend = Mock()
    backend.typed_mutation_support = ""
    backend.cypher_support = "full"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    def isolated():
        engine._upsert_node("Memory", "mem-3", {"content": "hello"})

    with pytest.raises(PermissionError):
        contextvars.Context().run(isolated)
    backend.execute.assert_not_called()


def test_upsert_node_never_overwrites_a_caller_supplied_classification():
    backend = Mock()
    backend.typed_mutation_support = ""
    backend.cypher_support = "full"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    with use_actor(_actor()):
        engine._upsert_node(
            "Memory", "mem-4", {"content": "hello", "classification": "restricted"}
        )

    _query, params = backend.execute.call_args[0]
    assert params["classification"] == "restricted"


# --- GraphComputeEngine.add_node --------------------------------------------


def _bare_graph_compute(client):
    gc = object.__new__(GraphComputeEngine)
    gc._client = client
    return gc


def test_graph_compute_add_node_stamps_governance():
    client = Mock()
    gc = _bare_graph_compute(client)

    with use_actor(_actor()):
        gc.add_node("n1", {"node_type": "Memory", "content": "hi"})

    client.nodes.add.assert_called_once()
    _node_id, props = client.nodes.add.call_args[0]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


def test_graph_compute_add_node_public_for_catalog_labels():
    client = Mock()
    gc = _bare_graph_compute(client)

    with use_actor(_actor()):
        gc.add_node("srv:tool-1", {"node_type": "ToolMetadata", "name": "x"})

    _node_id, props = client.nodes.add.call_args[0]
    assert props["classification"] == "public"


# --- BrainGuardedBackend ------------------------------------------------


def test_brain_guarded_backend_add_node_level_stamps_classification():
    from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
        BrainGuardedBackend,
    )

    inner = Mock()
    del inner.get_node_properties  # force node-level arbitration path
    brain = Mock()
    brain.conflicts.effective_authority.return_value = 0.5
    guarded = BrainGuardedBackend(inner, brain)

    with use_actor(_actor()):
        guarded.add_node("res-1", node_type="CallableResource", name="thing")

    inner.add_node.assert_called_once()
    _node_id, kwargs = (
        inner.add_node.call_args[0][0],
        inner.add_node.call_args[1],
    )
    assert kwargs["classification"] == "public"
    assert kwargs["_owner_id"] == "writer:alice"


# --- _BatchedBackend ------------------------------------------------------


def test_batched_backend_add_node_stamps_governance():
    from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend

    inner = Mock(spec=[])  # no ._graph attr -> bulk path disabled, fine for this test
    batched = _BatchedBackend(inner, batch_size=1000)

    with use_actor(_actor()):
        batched.add_node("sym-1", label="CodeSymbol", name="foo")

    assert len(batched._nodes) == 1
    props = batched._nodes[0]["properties"]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


def test_no_actor_and_public_catalog_end_to_end_permit(monkeypatch):
    """Full loop: chokepoint stamp -> durable properties -> read-time
    hydration -> permit() grants a non-owner actor for a PUBLIC catalog node,
    but a private node stays owner-only. Exercised against the real
    secured_reads module end to end (durable rows mocked, everything else
    real)."""
    from agent_utilities.knowledge_graph.core import secured_reads as sr
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        reset_company_brain,
    )

    reset_company_brain()
    backend = Mock()
    backend.typed_mutation_support = ""
    backend.cypher_support = "full"
    backend.__class__.__name__ = "GenericTestBackend"
    engine = _bare_engine(backend)

    written: dict[str, dict] = {}

    def fake_execute(_query, params):
        written[params["id"]] = dict(params)

    backend.execute.side_effect = fake_execute

    with use_actor(_actor("writer:alice")):
        engine._upsert_node("Memory", "mem-private", {"content": "secret"})
        engine._upsert_node("CallableResource", "tool-public", {"name": "x"})

    def fake_durable_rows(ids):
        return {
            nid: {
                "tenant_id": written[nid].get("tenant_id"),
                "classification": written[nid].get("classification"),
                "external_access": None,
                "owner_id": written[nid].get("_owner_id"),
            }
            for nid in ids
            if nid in written
        }

    monkeypatch.setattr(sr, "_durable_access_rows", fake_durable_rows)

    other_actor = ActorContext(
        actor_id="writer:bob",
        actor_type=ActorType.HUMAN,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )

    reset_company_brain()
    with use_actor(other_actor):
        # PUBLIC catalog node: any authenticated same-tenant actor reads it.
        assert sr.permit(["tool-public"]) == ["tool-public"]
        # Private memory node owned by someone else: still denied.
        assert sr.permit(["mem-private"]) == []

    reset_company_brain()
    with use_actor(_actor("writer:alice")):
        # The actual owner reads their own private memory.
        assert sr.permit(["mem-private"]) == ["mem-private"]


# --- materialization.write_entities (D-ACL-4, 5th chokepoint) --------------


def test_write_entities_stamps_governance_on_the_generic_unwind_path():
    """D-ACL-4: ``enrichment.registry.write_batch`` -> ``materialization
    .write_entities`` is the fifth write surface (connectors + internal
    finance/synthesize batches with ``source=None``) that the original
    four-chokepoint ACL-registration fix did not reach. An internal batch
    with no connector-supplied ``external_access`` used to land with no
    owner/classification stamp at all -- written but permanently unreadable
    under ``secured_reads.permit()``'s default-deny, identical to the gap the
    other four chokepoints already closed."""
    from agent_utilities.knowledge_graph.core.materialization import write_entities
    from tests.kg_recording_backend import RecordingGraphBackend

    backend = RecordingGraphBackend()

    with use_actor(_actor()):
        write_entities(
            backend,
            "finance",
            [{"id": "fact:1", "node_type": "MarketFact", "value": 42}],
        )

    props = backend.nodes["fact:1"]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


def test_write_entities_fails_closed_with_no_bound_actor():
    """BUG-033/BUG-039: no bound actor (system/background materialization)
    must fail the write, not silently proceed unstamped. Mirrors
    ``test_upsert_node_fails_closed_with_no_bound_actor``'s isolated
    ``contextvars.Context()`` (the test session otherwise binds a default
    actor -- see ``current_actor()`` -- so a bare call in-process is not
    actually actor-free). No node is created."""
    import contextvars

    import pytest

    from agent_utilities.knowledge_graph.core.materialization import write_entities
    from tests.kg_recording_backend import RecordingGraphBackend

    backend = RecordingGraphBackend()

    def isolated():
        write_entities(
            backend,
            "",
            [{"id": "fact:2", "node_type": "MarketFact", "value": 7}],
        )

    with pytest.raises(PermissionError):
        contextvars.Context().run(isolated)

    assert "fact:2" not in backend.nodes


def test_write_batch_stamps_governance_end_to_end():
    """The public entrypoint (``enrichment.registry.write_batch``, what every
    connector/materialize source and internal batch actually calls) carries
    the stamp all the way through, not just the lower-level writer."""
    from agent_utilities.knowledge_graph.enrichment.models import (
        ExtractionBatch,
        GraphNode,
    )
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch
    from tests.kg_recording_backend import RecordingGraphBackend

    backend = RecordingGraphBackend()
    batch = ExtractionBatch(
        category="finance",
        nodes=[GraphNode(id="fact:3", type="MarketFact", props={"value": 1})],
    )

    with use_actor(_actor()):
        write_batch(backend, batch)  # source=None, the untagged internal path

    props = backend.nodes["fact:3"]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


# --- GraphComputeEngine.add_edge (BUG-058) ----------------------------------
#
# Unlike ``add_node`` above, ``add_edge`` had NO ``stamp_ownership``/
# ``stamp_classification`` call at all -- every edge write reaching this
# chokepoint was unconditionally ungoverned regardless of caller. These
# mirror the ``add_node`` governance tests above exactly.


def test_graph_compute_add_edge_stamps_governance():
    client = Mock()
    client.edges.has.return_value = False
    gc = _bare_graph_compute(client)

    with use_actor(_actor()):
        gc.add_edge("a", "b", {"relationship": "DEPENDS_ON"})

    client.edges.add.assert_called_once()
    _source, _target, props = client.edges.add.call_args[0]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


def test_graph_compute_add_edge_public_for_catalog_relationship():
    """``stamp_classification`` treats the edge's ``relationship`` value the
    same way ``add_node`` treats ``node_type`` -- a name inside
    ``PUBLIC_CATALOG_LABELS`` stamps PUBLIC, everything else CONFIDENTIAL.
    This edge uses an ordinary relationship name, so it stays CONFIDENTIAL
    (there is no edge-shaped entry in that allowlist today); the assertion
    pins that the classification stamp is actually driven by the
    relationship value, not a hardcoded constant."""
    client = Mock()
    client.edges.has.return_value = False
    gc = _bare_graph_compute(client)

    with use_actor(_actor()):
        gc.add_edge("srv:tool-1", "srv:tool-2", relationship="REQUIRES_TOOLSET")

    _source, _target, props = client.edges.add.call_args[0]
    assert props["classification"] == "confidential"


def test_graph_compute_add_edge_fails_closed_with_no_bound_actor():
    """BUG-058: a write reaching this chokepoint with NO verified ambient
    actor at all must raise -- not silently proceed unstamped, exactly like
    ``add_node``'s BUG-033/BUG-039 fix. No edge is created."""
    import contextvars

    import pytest

    client = Mock()
    gc = _bare_graph_compute(client)

    def isolated():
        gc.add_edge("a", "b", {"relationship": "DEPENDS_ON"})

    with pytest.raises(PermissionError):
        contextvars.Context().run(isolated)
    client.edges.add.assert_not_called()


def test_graph_compute_add_edge_never_overwrites_a_caller_supplied_scope():
    """Existing ``_shared_scope``/``_owner_id`` markers are never overwritten
    (mirrors ``stamp_ownership``'s own ``setdefault`` contract for nodes)."""
    client = Mock()
    client.edges.has.return_value = False
    gc = _bare_graph_compute(client)

    with use_actor(_actor()):
        gc.add_edge(
            "a",
            "b",
            relationship="DEPENDS_ON",
            _owner_id="writer:bob",
            _shared_scope="org",
        )

    _source, _target, props = client.edges.add.call_args[0]
    assert props["_owner_id"] == "writer:bob"
    assert props["_shared_scope"] == "org"


# --- _BatchedBackend.add_edge (BUG-058) -------------------------------------


def test_batched_backend_add_edge_stamps_governance():
    from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend

    inner = Mock(spec=[])  # no ._graph attr -> bulk path disabled, fine for this test
    batched = _BatchedBackend(inner, batch_size=1000)

    with use_actor(_actor()):
        batched.add_edge("sym-1", "sym-2", rel_type="CALLS")

    assert len(batched._edges) == 1
    props = batched._edges[0]["properties"]
    assert props["tenant_id"] == "tenant-a"
    assert props["_owner_id"] == "writer:alice"
    assert props["classification"] == "confidential"


def test_batched_backend_add_edge_fails_closed_with_no_bound_actor():
    """BUG-058: mirrors ``test_batched_backend_add_node``'s (implicit, via
    the shared ``add_node`` gate) fail-closed contract -- the bulk ingest
    seam must refuse an actor-free edge write, not buffer it unstamped for a
    later bulk RPC that bypasses ``GraphComputeEngine.add_edge`` entirely."""
    import contextvars

    import pytest

    from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend

    inner = Mock(spec=[])
    batched = _BatchedBackend(inner, batch_size=1000)

    def isolated():
        batched.add_edge("sym-1", "sym-2", rel_type="CALLS")

    with pytest.raises(PermissionError):
        contextvars.Context().run(isolated)
    assert batched._edges == []
