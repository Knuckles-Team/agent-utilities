#!/usr/bin/python
from __future__ import annotations

"""Wave-2 combined live-path test — exercise the EXISTING entry points.

Wire-First: each Wave-2 capability is reached through the *real*
:class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` facade and its
``kg.ontology`` :class:`OntologySystem` — not the helper APIs in isolation — so
this asserts the integrator wiring actually invokes the new code on a live path.

Covers:
  (a) KG-2.43 edit ledger: record -> history -> revert via ``kg.ontology``.
  (b) KG-2.44 object index funnel: batch + incremental sync -> staleness ->
      reconcile, through ``kg.sync_object_index`` / ``kg.reindex_stale_objects``.
  (c) KG-2.46 default-on permissioning: ``enforce`` filters a marked node while
      a public row passes, on the ``kg.query()`` read seam.
  (d) KG-2.45 object sets: search_around + pivot + aggregate via ``kg.ontology``.
  (e) KG-2.48 document processing: process -> chunks linked to a Document with
      embeddings, through the live facade write path.
  (f) KG-2.42 actions: an action with two side-effects writes two edit-ledger
      records and undo reverts them (governed verb -> edit trail).
"""

import uuid

import pytest

from agent_utilities.knowledge_graph.actions import (
    ActionEffect,
    ActionEffectSpec,
    ActionExecutor,
    ActionParameter,
    ActionRegistry,
    ActionStatus,
    EffectKind,
    OntologyAction,
)
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.knowledge_graph.ontology.document_processing import (
    ChunkingConfig,
    DocumentProcessor,
)
from agent_utilities.knowledge_graph.ontology.edits import Edit, EditType
from agent_utilities.knowledge_graph.ontology.permissioning import (
    Marking,
    apply_marking,
    clear_markings,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.observability.escalation_matrix import make_decision_provider
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.permissions_kernel import AgentRole, PermissionsKernel


@pytest.fixture
def kg() -> KnowledgeGraph:
    return KnowledgeGraph(backend_type="memory")


@pytest.fixture(autouse=True)
def _clean_brain_state():
    reset_company_brain()
    clear_markings()
    yield
    reset_company_brain()
    clear_markings()


def _p() -> str:
    """Unique id prefix so this test never collides with shared-graph data."""
    return "wave2lp_" + uuid.uuid4().hex[:8]


# (a) ── KG-2.43 edit ledger through kg.ontology ───────────────────────────────
def test_edit_ledger_record_history_revert_via_ontology(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    assert ont is not None
    oid = f"{_p()}:doc"

    # Materialize the object first (OBJECT_CREATE) so subsequent property edits
    # have a real before-snapshot in the versioned graph_state.
    e0 = ont.record_edit(
        Edit(
            actor="alice",
            edit_type=EditType.OBJECT_CREATE,
            object_id=oid,
            after={"title": "Draft"},
        )
    )
    e1 = ont.set_property_edit(oid, {"title": "Final", "score": 9}, actor="alice")

    history = ont.history(oid)
    assert [e.id for e in history] == [e0.id, e1.id]

    # as_of reconstructs the snapshot at object-create time.
    snap = ont.as_of(oid, e0.timestamp)
    assert snap == {"title": "Draft"}

    # Reverting the property edit restores the prior values it overwrote.
    comp = ont.revert_edit(e1.id, actor="alice")
    assert comp.edit_type == EditType.PROPERTY_SET
    assert comp.after == {"title": "Draft", "score": None}
    # History grew (revert is itself recorded), original edits remain.
    assert len(ont.history(oid)) == 3


# (b) ── KG-2.44 object index funnel via facade accessors ──────────────────────
def test_object_index_batch_incremental_staleness_reindex(kg: KnowledgeGraph) -> None:
    funnel = kg.object_index_funnel
    # The funnel is constructed over the live retrieval index (same object the
    # router ranks against) — not a second index.
    assert funnel.index is kg.retrieval

    pre = _p()
    nodes = [
        {"id": f"{pre}:n1", "type": "tool", "embedding": [0.1] * 8, "capabilities": ["x"]},
        {"id": f"{pre}:n2", "type": "tool", "embedding": [0.2] * 8, "capabilities": ["y"]},
    ]
    batch = kg.sync_object_index(nodes)
    assert batch.rebuilt and batch.upserted == 2

    # Incremental upsert of a brand-new object (live delta, no full rebuild).
    from agent_utilities.knowledge_graph.ontology.indexing import FunnelDelta

    n3 = {"id": f"{pre}:n3", "type": "tool", "embedding": [0.3] * 8, "capabilities": ["z"]}
    inc = funnel.incremental_sync(FunnelDelta(upserts=[n3]))
    assert inc.upserted == 1
    assert f"{pre}:n3" in funnel.live_ids()

    # Drift detection: mutate n1's embedding => staleness ledger flags reindex.
    changed = [
        {"id": f"{pre}:n1", "type": "tool", "embedding": [0.9] * 8, "capabilities": ["x"]},
        nodes[1],
        n3,
    ]
    assert funnel.needs_reindex(changed) is True
    recon = kg.reindex_stale_objects(changed)
    # Exactly the changed object is re-upserted; nothing dropped.
    assert recon.upserted == 1 and recon.deleted == 0
    assert funnel.needs_reindex(changed) is False


# (c) ── KG-2.46 default-on enforce on the kg.query() read seam ────────────────
def test_query_enforce_filters_marked_passes_public(
    monkeypatch: pytest.MonkeyPatch, kg: KnowledgeGraph
) -> None:
    pre = _p()
    pub_id, sec_id = f"{pre}:public", f"{pre}:secret"
    apply_marking(sec_id, Marking("topsecret"))

    # Use a fixed in-memory store read so the test is deterministic and exercises
    # the facade query() path (scope -> filter_rows -> enforce_fine_grained -> audit).
    rows = [{"id": pub_id, "name": "open"}, {"id": sec_id, "name": "classified"}]

    class _Store:
        def execute(self, cypher, params=None):  # noqa: D401, ANN001
            return list(rows)

    kg._store = _Store()

    low = ActorContext(actor_id="analyst:1", actor_type=ActorType.HUMAN, roles=("analyst",))
    with use_actor(low):
        out = kg.query("MATCH (n) RETURN n")
    ids = {r["id"] for r in out}
    # Public row passes (allow-by-default); marked row is row-dropped.
    assert pub_id in ids
    assert sec_id not in ids

    # An actor holding the marking sees both.
    cleared = ActorContext(
        actor_id="admin:1", actor_type=ActorType.HUMAN, roles=("marking:topsecret",)
    )
    with use_actor(cleared):
        out2 = kg.query("MATCH (n) RETURN n")
    assert {r["id"] for r in out2} == {pub_id, sec_id}


def test_query_unmarked_data_passes_unchanged(kg: KnowledgeGraph) -> None:
    """Allow-by-default: unmarked/unACL'd rows are returned verbatim."""
    rows = [{"id": f"{_p()}:a", "name": "x"}, {"id": f"{_p()}:b", "name": "y"}]

    class _Store:
        def execute(self, cypher, params=None):  # noqa: ANN001
            return list(rows)

    kg._store = _Store()
    out = kg.query("MATCH (n) RETURN n")
    assert out == rows


# (d) ── KG-2.45 object sets: search_around + pivot + aggregate ────────────────
def test_object_set_search_around_pivot_aggregate(kg: KnowledgeGraph) -> None:
    store = kg.store
    pre = _p()
    a, b, c = f"{pre}:a", f"{pre}:b", f"{pre}:c"
    store.add_node(a, type="widget", name="alpha", amount=10)
    store.add_node(b, type="widget", name="beta", amount=20)
    store.add_node(c, type="gadget", name="gamma")
    store.add_edge(a, c, rel_type="USES")

    ont = kg.ontology
    base = ont.object_set([a, b])

    # aggregate: real sum over a numeric property.
    agg = base.aggregate("sum", field="amount")
    assert agg.value == 30.0

    # search_around: typed traversal to the related object set.
    around = base.search_around("USES", hops=1)
    assert around.ids() == [c]

    # pivot: follow the link and group the linked set by a target property.
    piv = base.pivot("USES", "type")
    assert {k: len(v) for k, v in piv.groups.items()} == {"gadget": 1}


# (e) ── KG-2.48 document processing through the live facade write path ────────
def test_document_process_chunks_linked_with_embeddings(kg: KnowledgeGraph) -> None:
    text = (
        "Alpha block is the first paragraph here.\n\n"
        "Beta block is the second paragraph and continues a while.\n\n"
        "Gamma block is the third and last paragraph of this document."
    )

    # Inject a deterministic embed_fn so the embedding leg is exercised even
    # without the optional embedding model installed — still the live write path.
    def _embed(texts):  # noqa: ANN001
        return [[float(len(t) % 7)] * 8 for t in texts]

    proc = DocumentProcessor(
        kg,
        chunking=ChunkingConfig(chunk_size=50, overlap=10),
        embed_fn=_embed,
    )
    result = proc.process(text, source=f"{_p()}://doc")

    assert result.chunk_count >= 2
    # One HAS_CHUNK + one CHUNK_OF per chunk.
    assert len(result.edges) == 2 * result.chunk_count
    # Every chunk links back to the one Document and carries an embedding.
    for cn in result.chunk_nodes:
        assert cn["document_id"] == result.document_id
        assert cn["embedding"] and len(cn["embedding"]) == 8
    edge_types = {e["type"] for e in result.edges}
    assert edge_types == {"HAS_CHUNK", "CHUNK_OF"}


# (f) ── KG-2.42 action with two side-effects -> two edits -> undo reverts ─────
def _onboard_action() -> OntologyAction:
    return OntologyAction(
        name="wave2lp.onboard",
        verb="onboard",
        description="Create a record object and link it to its owner.",
        parameters=[
            ActionParameter(name="record_id", required=True),
            ActionParameter(name="owner_id", required=True),
            ActionParameter(name="title", required=True),
        ],
        acts_on=["record"],
        required_capability="kg_write",
        produces_effect=ActionEffect.MUTATION,
        idempotent=False,
        side_effects=[
            ActionEffectSpec(
                kind=EffectKind.CREATE_OBJECT,
                target="$record_id",
                params={"title": "$title", "type": "record"},
            ),
            ActionEffectSpec(
                kind=EffectKind.ADD_LINK,
                target="$record_id",
                params={"link_target": "$owner_id", "link_label": "owned_by"},
            ),
        ],
    )


def test_action_two_side_effects_write_two_edits_and_undo(kg: KnowledgeGraph) -> None:
    kernel = PermissionsKernel()
    # The action executor journals through the SAME edit-ledger surface the
    # ontology exposes; bind it to the live ontology ledger so the governed verb
    # and the edit trail share one ledger.
    ledger = kg.ontology.edits

    reg = ActionRegistry()
    reg.register(_onboard_action(), handler=lambda p: {"ok": p["record_id"]})
    ex = ActionExecutor(reg, kernel=kernel, persist=False, ledger=ledger)

    writer = kernel.issue_identity(
        "agent:writer", role=AgentRole.SPECIALIST, capabilities=["kg_write"]
    )
    approve = make_decision_provider(
        {"wave2lp.onboard": {"approved": True, "approver": "ops", "approver_role": "operator"}}
    )
    rid, owner = f"{_p()}:record", f"{_p()}:user"

    inv = ex.execute(
        "wave2lp.onboard",
        writer,
        {"record_id": rid, "owner_id": owner, "title": "Q3 filing"},
        decision_provider=approve,
    )
    assert inv.status == ActionStatus.SUCCESS
    # Two side-effects -> two durable edit-ledger records, both linked to the inv.
    assert len(inv.edit_ids) == 2
    for eid in inv.edit_ids:
        assert ledger.get(eid).invocation_ref == inv.id
    assert ledger.graph_state["nodes"][rid]["title"] == "Q3 filing"
    assert (rid, owner, "owned_by") in ledger.graph_state["edges"]

    # Undo reverts both effects via the C1 revert path (compensating edits).
    compensating = ex.undo(inv, actor="agent:writer")
    assert len(compensating) == 2
    assert rid not in ledger.graph_state["nodes"]
    assert (rid, owner, "owned_by") not in ledger.graph_state["edges"]
