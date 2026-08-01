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
from agent_utilities.security.brain_context import ActorContext
from agent_utilities.security.permissions_kernel import AgentRole, PermissionsKernel


@pytest.fixture
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


class _FakeMarkingStore:
    """Minimal durable-store double for the mandatory marking authority.

    apply_marking requires a resolved marking store
    (agent_utilities/knowledge_graph/ontology/permissioning.py) -- process-wide
    and cached-once, so an unavailable store fails every test in the process,
    not just the first. use_marking_authority is the sanctioned DI/test seam
    (see tests/ontology/test_permissioning.py's clean_permissioning autouse
    fixture for the established pattern).
    """

    def __init__(self) -> None:
        self.persisted: dict[str, dict] = {}

    def execute(self, query, params):
        if query.startswith("MATCH"):
            return list(self.persisted.values())
        self.persisted[params["id"]] = {
            "node_id": params["n"],
            "tenant_id": params["tenant"],
            "markings": params["marks"],
        }
        return []


@pytest.fixture(autouse=True)
def _clean_brain_state():
    from agent_utilities.knowledge_graph.ontology import permissioning as p

    reset_company_brain()
    clear_markings()
    with p.use_marking_authority(_FakeMarkingStore()):
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
        {
            "id": f"{pre}:n1",
            "type": "tool",
            "embedding": [0.1] * 8,
            "capabilities": ["x"],
        },
        {
            "id": f"{pre}:n2",
            "type": "tool",
            "embedding": [0.2] * 8,
            "capabilities": ["y"],
        },
    ]
    batch = kg.sync_object_index(nodes)
    assert batch.rebuilt and batch.upserted == 2

    # Incremental upsert of a brand-new object (live delta, no full rebuild).
    from agent_utilities.knowledge_graph.ontology.indexing import FunnelDelta

    n3 = {
        "id": f"{pre}:n3",
        "type": "tool",
        "embedding": [0.3] * 8,
        "capabilities": ["z"],
    }
    inc = funnel.incremental_sync(FunnelDelta(upserts=[n3]))
    assert inc.upserted == 1
    assert f"{pre}:n3" in funnel.live_ids()

    # Drift detection: mutate n1's embedding => staleness ledger flags reindex.
    changed = [
        {
            "id": f"{pre}:n1",
            "type": "tool",
            "embedding": [0.9] * 8,
            "capabilities": ["x"],
        },
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
    from agent_utilities.core.config import config

    # kg.query()'s light epistemic-attach layer (CONCEPT:AU-KB-CURRENCY, "Native
    # by default") is on by default and needs self.compute (a bound engine) even
    # for a plain read -- unrelated to what THIS test actually exercises (the
    # marking-enforcement row filter). Opt out for this hermetic, no-engine test.
    monkeypatch.setattr(config, "epistemic_light_default", False)

    from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
    from agent_utilities.models.company_brain import DataClassification

    pre = _p()
    pub_id, sec_id = f"{pre}:public", f"{pre}:secret"
    apply_marking(sec_id, Marking("topsecret"))
    # enforce()/restricted_view() fail closed on a MISSING ACL regardless of
    # marking (permissioning.py's _acl_permits docstring: "fail-closed
    # discretionary ACL check") -- give BOTH an otherwise-permitting
    # PUBLIC-classification ACL, so sec_id's denial for the uncleared actor
    # (and its visibility for the cleared one) is purely a function of the
    # mandatory marking this test actually exercises, not a missing ACL.
    build_acl(pub_id, DataClassification.PUBLIC)
    build_acl(sec_id, DataClassification.PUBLIC)

    # Use a fixed in-memory store read so the test is deterministic and exercises
    # the facade query() path (scope -> filter_rows -> enforce_fine_grained -> audit).
    rows = [{"id": pub_id, "name": "open"}, {"id": sec_id, "name": "classified"}]

    class _Store:
        # kg.query() calls store.execute_read(...), not execute() (that's the
        # unfiltered raw store method secured_reads.scope() wraps around).
        def execute_read(self, cypher, params=None, *, include_epistemic=False):
            return list(rows)

    kg._store = _Store()

    # kg.query()'s resolve_session(session=None, ...) always resolves the
    # AMBIENT GraphSession (core/session.py: "Every caller must inherit the
    # authentication middleware's ambient session" -- an out-of-band actor
    # substitution is deliberately rejected), so use_actor(...) alone has no
    # effect on enforcement here. Swap the ambient session's actor itself via
    # with_actor (which requires a matching tenant_id) + use_session.
    from _test_engine import TEST_TENANT

    from agent_utilities.knowledge_graph.core.session import (
        current_session,
        use_session,
    )

    baseline = current_session()
    low = ActorContext(
        actor_id="analyst:1",
        actor_type=ActorType.HUMAN,
        roles=("analyst",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    with use_session(baseline.with_actor(low)):
        out = kg.query("MATCH (n) RETURN n")
    ids = {r["id"] for r in out}
    # Public row passes (allow-by-default); marked row is row-dropped.
    assert pub_id in ids
    assert sec_id not in ids

    # An actor holding the marking sees both.
    cleared = ActorContext(
        actor_id="admin:1",
        actor_type=ActorType.HUMAN,
        roles=("marking:topsecret",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    with use_session(baseline.with_actor(cleared)):
        out2 = kg.query("MATCH (n) RETURN n")
    assert {r["id"] for r in out2} == {pub_id, sec_id}


def test_query_unmarked_data_passes_unchanged(
    monkeypatch: pytest.MonkeyPatch, kg: KnowledgeGraph
) -> None:
    """Allow-by-default (given a permitting ACL): unmarked rows pass verbatim.

    permissioning.py's enforce()/restricted_view() fail closed on any row with
    no ACL at all ("Missing authority or policy infrastructure fails closed" --
    enforce()'s own docstring); "allow-by-default" here means an explicit
    PUBLIC-classification ACL and no marking, not the absence of any control.
    """
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
    from agent_utilities.models.company_brain import DataClassification

    # See test_query_enforce_filters_marked_passes_public: opt out of the
    # engine-requiring light epistemic-attach layer for this hermetic test.
    monkeypatch.setattr(config, "epistemic_light_default", False)

    rows = [{"id": f"{_p()}:a", "name": "x"}, {"id": f"{_p()}:b", "name": "y"}]
    for row in rows:
        build_acl(row["id"], DataClassification.PUBLIC)

    class _Store:
        # kg.query() calls store.execute_read(...), not execute().
        def execute_read(self, cypher, params=None, *, include_epistemic=False):
            return list(rows)

    kg._store = _Store()
    out = kg.query("MATCH (n) RETURN n")
    assert out == rows


@pytest.fixture
def kg_engine(engine_graph) -> KnowledgeGraph:
    """A KnowledgeGraph bound to the REAL ephemeral test engine.

    Unlike the bare ``kg`` fixture (used above with a hand-rolled fake
    ``_Store``), these two tests call real backend write methods
    (``store.add_node``/``add_edge``) and route a full ``DocumentProcessor``
    write path through ``kg.store`` -- exactly the "live path" this module
    docstring promises, so a fake store would defeat the point. Requesting
    ``engine_graph`` is how a test opts into the real database (tests/
    conftest.py); it skips cleanly via ``tiny_engine`` when no prebuilt
    engine binary is available in this environment, rather than hard-failing
    with "KnowledgeGraph requires the active process-owned graph engine".
    """
    kg = KnowledgeGraph()
    kg._store = engine_graph
    kg._compute = engine_graph
    return kg


# (d) ── KG-2.45 object sets: search_around + pivot + aggregate ────────────────
@pytest.mark.engine
def test_object_set_search_around_pivot_aggregate(kg_engine: KnowledgeGraph) -> None:
    store = kg_engine.store
    pre = _p()
    a, b, c = f"{pre}:a", f"{pre}:b", f"{pre}:c"
    # GraphComputeEngine.add_node/add_edge use the canonical "node_type"/
    # "relationship" property names -- "type"/"rel_type" raise ValueError
    # ("use canonical 'node_type'") on the real engine.
    store.add_node(a, node_type="widget", name="alpha", amount=10)
    store.add_node(b, node_type="widget", name="beta", amount=20)
    store.add_node(c, node_type="gadget", name="gamma")
    store.add_edge(a, c, relationship="USES")

    ont = kg_engine.ontology
    base = ont.object_set([a, b])

    # aggregate: real sum over a numeric property.
    agg = base.aggregate("sum", field="amount")
    assert agg.value == 30.0

    # search_around: typed traversal to the related object set.
    around = base.search_around("USES", hops=1)
    assert around.ids() == [c]

    # pivot: follow the link and group the linked set by a target property.
    piv = base.pivot("USES", "node_type")
    assert {k: len(v) for k, v in piv.groups.items()} == {"gadget": 1}


# (e) ── KG-2.48 document processing through the live facade write path ────────
@pytest.mark.engine
def test_document_process_chunks_linked_with_embeddings(
    kg_engine: KnowledgeGraph,
) -> None:
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
        kg_engine,
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
    # document_processing.py's _build_edges keys edges by "relationship", not
    # "type" (see tests/ontology/test_document_processing.py's identical fix).
    edge_types = {e["relationship"] for e in result.edges}
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
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
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
        {
            "wave2lp.onboard": {
                "approved": True,
                "approver": "ops",
                "approver_role": "operator",
            }
        }
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
