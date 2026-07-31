"""Governed candidate-claim promotion + fact supersession + dead-letter drain.

Wiring proofs for the universal-ingestion program's Track A (governed
validation/promotion) and Track B (incremental reconciliation) tracks
(CONCEPT:AU-KG.ingest.governed-claim-promotion,
CONCEPT:AU-KG.ingest.fact-supersession,
CONCEPT:AU-KG.ingest.dead-letter-drain):

* a claim failing SHACL is REFUSED (never materialized);
* a re-delivered envelope with the same ``idempotency_key`` does not
  double-write;
* a superseded fact remains inspectable with the evidence that retired it;
* a dead-lettered item is visible and drainable.

Reuses the SAME native-engine test double as
``test_native_envelope_ingest.py`` (``_Compute``/``_Client``/``_Changes``/
``_Nodes``/``_Rdf``) so ``ingest_envelope`` itself runs for real — never
mocked — and layers a thin Claim/WorkItem bookkeeping surface
(``query_cypher``/``add_node``/``compare_and_set_node_fields``/``link_nodes``)
on top, mirroring ``test_claim_flywheel.py``'s ``_FlywheelStubEngine`` style.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    reset_session,
    set_session,
)
from agent_utilities.knowledge_graph.ingestion import (
    dead_letter,
    promotion,
    supersession,
)
from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.research.claim_flywheel import ClaimLifecycleState
from agent_utilities.models.company_brain import ActorType, DataClassification
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.brain_context import ActorContext
from tests.unit.knowledge_graph.ingestion.test_native_envelope_ingest import (
    _Compute,
)

pytestmark = pytest.mark.concept("AU-KG.ingest.governed-claim-promotion")


@pytest.fixture(autouse=True)
def _native_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PROFILE", "dev")
    actor = ActorContext(
        actor_id="fixture-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="fixture-tenant",
        authenticated=True,
    )
    token = set_session(
        GraphSession(
            actor=actor,
            tenant="fixture-tenant",
            scopes=frozenset({"kg:read", "kg:write"}),
            graph="fixture-graph",
            policy_version="fixture-policy",
            audience="fixture-audience",
        )
    )
    try:
        yield
    finally:
        reset_session(token)


class _PromotionEngine(_Compute):
    """``_Compute`` (real native ``ingest_envelope`` path) + a thin Claim/
    WorkItem bookkeeping surface (``query_cypher``/``add_node``/
    ``compare_and_set_node_fields``/``link_nodes``) — the two facets a real
    engine exposes, kept as two independent stores here since this double
    only needs to exercise both call shapes, not unify their storage."""

    def __init__(self, graph: str = "graph-promotion") -> None:
        super().__init__(graph)
        self._nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []

    def add_node(
        self, node_id: str, label: str, properties: dict[str, Any] | None = None
    ) -> None:
        self._nodes[node_id] = {"id": node_id, "_label": label, **(properties or {})}

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: Any,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: Any = None,
    ) -> None:
        self.edges.append((source_id, target_id, str(rel_type), dict(properties or {})))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "ClaimLifecycleEvent" in query:
            rows = [
                n
                for n in self._nodes.values()
                if n.get("_label") == "ClaimLifecycleEvent"
            ]
            cid = params.get("id")
            if cid is not None:
                rows = [r for r in rows if r.get("claim_id") == cid]
            return [
                {
                    "claim_id": r.get("claim_id"),
                    "from_state": r.get("from_state"),
                    "to_state": r.get("to_state"),
                    "reason": r.get("reason"),
                    "actor": r.get("actor"),
                    "governance_valid": r.get("governance_valid"),
                    "action_decision": r.get("action_decision"),
                    "timestamp": r.get("timestamp"),
                }
                for r in rows
            ]
        if "c:Claim {id:" in query:
            node = self._nodes.get(params.get("id"))
            if node is None or node.get("_label") != "Claim":
                return []
            return [{"metadata": node.get("metadata")}]
        if "c:Claim)" in query:
            domain = params.get("domain")
            rows = [n for n in self._nodes.values() if n.get("_label") == "Claim"]
            if domain is not None:
                rows = [r for r in rows if r.get("domain") == domain]
            return [
                {"id": r.get("id"), "claim_text": r.get("claim_text")} for r in rows
            ]
        if "WorkItem" in query and "dead_letter" in query:
            rows = [
                n
                for n in self._nodes.values()
                if n.get("_label") == "WorkItem" and n.get("status") == "dead_letter"
            ]
            kind = params.get("kind")
            if kind:
                rows = [r for r in rows if r.get("kind") == kind]
            return [dict(r, id=r["id"]) for r in rows]
        if "WorkItem" in query and "{id: $id}" in query:
            # ``get_work_item``'s get-by-id shape — return the raw node
            # (a real backend would project just the requested fields; every
            # caller here reads by key, so the superset is harmless).
            node = self._nodes.get(params.get("id"))
            if node is None or node.get("_label") != "WorkItem":
                return []
            return [dict(node, id=node["id"])]
        return []

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        node = self._nodes.get(node_id)
        if node is None:
            return False
        for key, expected in conditions.items():
            if node.get(key) != expected:
                return False
        node.update(updates)
        return True


def _proposed_envelope(
    *, source_object_id: str = "svc-1", statement: str = "svc-1 is a payments service"
) -> ChangeEnvelope:
    return ChangeEnvelope(
        connector="fixture-ingest",
        operation="upsert",
        source_object_id=source_object_id,
        source_version="1",
        typed_payload={"id": source_object_id, "type": "Service", "name": statement},
        source_acl=ExternalAccess(is_public=False, read_roles=["kg:read"]),
        classification=DataClassification.INTERNAL,
    )


def _claim(engine_domain: str = "cmdb", **overrides: Any) -> promotion.CandidateClaim:
    envelope = overrides.pop("envelope", None) or _proposed_envelope()
    return promotion.CandidateClaim(
        domain=engine_domain,
        statement=overrides.pop("statement", "svc-1 is a payments service"),
        confidence=overrides.pop("confidence", 0.9),
        envelope=envelope,
        **overrides,
    )


# ---------------------------------------------------------------------------
# 1. A claim failing SHACL is refused, never materialized.
# ---------------------------------------------------------------------------


def test_shacl_failing_claim_is_rejected_and_never_materialized() -> None:
    engine = _PromotionEngine()
    engine.client.rdf.reports = [{"conforms": False, "results": [{}]}]
    claim = _claim()

    outcome = promotion.evaluate_and_advance(engine, claim)

    assert outcome["verdict"]["decision"] == "rejected"
    assert outcome["current_state"] == ClaimLifecycleState.RETRACTED.value

    # Refused means never materialized: attempting to materialize an
    # unaccepted (retracted) claim must not write the real fact either.
    result = promotion.materialize_on_claim_accepted(engine, claim.claim_id)
    assert result is not None  # a proposed_envelope IS present...
    # ...but nothing was actually committed to the native engine.
    assert engine.client.changes.applied == []
    assert engine.client.nodes.values == {}


def test_shacl_conforming_claim_clears_and_awaits_steward_accept() -> None:
    engine = _PromotionEngine()
    claim = _claim()

    outcome = promotion.evaluate_and_advance(engine, claim)

    assert outcome["verdict"]["decision"] == "cleared"
    assert outcome["current_state"] == ClaimLifecycleState.VALIDATED.value
    # Still not a fact: no native write has happened yet.
    assert engine.client.changes.applied == []


# ---------------------------------------------------------------------------
# 2. A re-delivered envelope with the same idempotency_key does not
#    double-write (redelivery of the SAME accepted claim's materialization).
# ---------------------------------------------------------------------------


def test_materialize_on_claim_accepted_is_idempotent_on_redelivery() -> None:
    engine = _PromotionEngine()
    claim = _claim()
    promotion.evaluate_and_advance(engine, claim)  # -> VALIDATED

    first = promotion.materialize_on_claim_accepted(engine, claim.claim_id)
    assert first["status"] == "success"
    assert len(engine.client.changes.applied) == 1

    # Redelivery: the SAME accept path invoked again for the SAME claim.
    second = promotion.materialize_on_claim_accepted(engine, claim.claim_id)

    assert second["status"] == "skipped"
    assert second["reason"] == "already materialized"
    # No second native apply — the underlying idempotency_key was never
    # even re-submitted to the native engine a second time.
    assert len(engine.client.changes.applied) == 1


def test_native_redelivery_of_the_same_idempotency_key_is_also_a_native_skip() -> None:
    """Even bypassing this module's own short-circuit, the underlying
    ``ingest_envelope`` boundary independently refuses to double-write the
    SAME idempotency_key (defense in depth: two independent guards)."""
    engine = _PromotionEngine()
    envelope = _proposed_envelope()

    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        ingest_envelope,
    )

    first = ingest_envelope(engine, envelope)
    assert first["status"] == "success"
    replay = ingest_envelope(engine, envelope)
    assert replay["status"] == "skipped"
    assert len(engine.client.changes.applied) == 1


# ---------------------------------------------------------------------------
# 3. A superseded fact remains inspectable with the evidence that retired it.
# ---------------------------------------------------------------------------


def test_retire_fact_tombstones_without_deleting_and_links_evidence() -> None:
    engine = _PromotionEngine()
    engine.client.nodes.values["svc-1"] = {
        "domain": "fixture-ingest",
        "externalToolId": "svc-1",
        "name": "svc-1 is a payments service",
    }

    result = supersession.retire_fact(
        engine,
        entity_id="svc-1",
        connector="fixture-ingest",
        reason="corrected by a later, higher-confidence source",
        retracted_by_claim="claim:ingest:new-version",
    )

    assert result["tombstone"]["status"] == "success"
    assert result["evidence_linked"] is True
    # Inspectable, not deleted: the node still exists with its history.
    retired = engine.client.nodes.values["svc-1"]
    assert retired["archived"] is True
    assert retired["name"] == "svc-1 is a payments service"
    assert retired["externalToolId"] == "svc-1"
    # The evidence that retired it is a durable, traversable edge.
    assert engine.edges == [
        (
            "claim:ingest:new-version",
            "svc-1",
            "supersedes",
            {
                "_rel": "SUPERSEDES",
                "reason": "corrected by a later, higher-confidence source",
                "concept": "AU-KG.ingest.fact-supersession",
            },
        )
    ]


def test_retract_and_supersede_retires_a_materialized_claims_fact() -> None:
    engine = _PromotionEngine()
    claim = _claim()
    promotion.evaluate_and_advance(engine, claim)
    promotion.materialize_on_claim_accepted(engine, claim.claim_id)
    assert engine.client.nodes.values["svc-1"].get("archived") is not True

    outcome = promotion.retract_and_supersede(
        engine, claim.claim_id, reason="found to be wrong"
    )

    assert outcome["transition"]["to_state"] == ClaimLifecycleState.RETRACTED.value
    assert outcome["superseded_fact"]["tombstone"]["status"] == "success"
    assert engine.client.nodes.values["svc-1"]["archived"] is True
    # The claim's own lifecycle audit trail is untouched history, not deleted.
    assert engine.edges[-1][:2] == (claim.claim_id, "svc-1")


# ---------------------------------------------------------------------------
# 4. A dead-lettered item is visible and drainable.
# ---------------------------------------------------------------------------


def test_dead_letter_item_is_listed_and_drainable() -> None:
    engine = _PromotionEngine()
    engine._nodes["wi-1"] = {
        "id": "wi-1",
        "_label": "WorkItem",
        "status": "dead_letter",
        "kind": "ingest_task",
        "queue": "ingestion",
        "tenant": "fixture-tenant",
        "payload_ref": "ref-1",
        "error_ref": "boom",
        "resource_class": "default",
        "max_attempts": 3,
        "updated_at": 100.0,
        "correlation_id": "corr-1",
        "dag_id": "",
        "depends_on": [],
        "downstream_ids": [],
        "metadata": {},
    }

    listed = dead_letter.list_dead_letter_items(engine)
    assert len(listed) == 1
    assert listed[0]["id"] == "wi-1"
    assert listed[0]["error_ref"] == "boom"

    drained = dead_letter.drain_dead_letter_item(
        engine, "wi-1", actor="operator@example.invalid", reason="upstream fixed"
    )

    assert drained["status"] == "drained"
    replacement_id = drained["replacement_item_id"]
    assert replacement_id != "wi-1"
    replacement = engine._nodes[replacement_id]
    assert replacement["kind"] == "ingest_task"
    assert replacement["metadata"]["drained_from"] == "wi-1"
    # The ORIGINAL dead-lettered item is untouched — still visible, never
    # silently dropped or mutated out from under an auditor.
    assert engine._nodes["wi-1"]["status"] == "dead_letter"
    assert engine.edges[-1] == (
        "wi-1",
        replacement_id,
        "drained_as",
        {"_rel": "DRAINED_AS"},
    )
    # Draining an unknown / non-dead-lettered id is a defined no-op, never a
    # silent resubmit of an item that never actually dead-lettered.
    assert dead_letter.drain_dead_letter_item(
        engine, "does-not-exist", actor="operator@example.invalid"
    ) == {"status": "not_found", "item_id": "does-not-exist"}
