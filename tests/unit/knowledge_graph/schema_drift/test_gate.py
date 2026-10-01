"""AU-SEC-R004/R005: contain drift, propose a repair, activate only an approved one."""

from __future__ import annotations

import hashlib
import json

import pytest

from agent_utilities.knowledge_graph.schema_drift import (
    GateRequest,
    commit_contract,
    parse_policy,
    run_gate,
)
from agent_utilities.knowledge_graph.schema_drift.candidate import (
    APPROVED_CANDIDATE_DOMAIN,
    approved_candidate_digest,
    record_contract,
)
from agent_utilities.knowledge_graph.schema_drift.repair import (
    RepairContext,
    resolve_renames,
)
from agent_utilities.knowledge_graph.schema_drift.shape import infer_shape
from tests.unit.knowledge_graph.schema_drift.fakes import FakeEngine, FakePort

SOURCE = "container-manager-mcp"
TENANT = "tenant-a"
V1 = [
    {"id": "c1", "name": "web", "image": "x"},
    {"id": "c2", "name": "db", "image": "y"},
]
V2 = [
    {"id": "c1", "title": "web", "image": "x"},
    {"id": "c2", "title": "db", "image": "y"},
]


class Harness:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.port = FakePort()
        self.shadow_ingested: list[str] = []

    def ingest(self, graph: str) -> tuple[int, int]:
        self.shadow_ingested.append(graph)
        return 2, 0

    def context(self) -> RepairContext:
        return RepairContext(
            self.engine, self.port, TENANT, self.ingest, lambda f, c, fb: fb
        )

    def sync(self, records: list[dict], policy: str = ""):
        request = GateRequest(
            engine=self.engine,
            source=SOURCE,
            records=records,
            record_ids=[r["id"] for r in records],
            policy=parse_policy(policy),
            repair=self.context,
        )
        outcome = run_gate(request)
        if outcome.proceed:
            commit_contract(self.engine, outcome, applied_cleanly=True)
        return outcome


@pytest.fixture
def harness() -> Harness:
    h = Harness()
    first = h.sync(V1)
    assert first.proceed and first.detail == {"contract": "bootstrap"}
    return h


def test_the_first_observation_becomes_the_contract_only_after_a_clean_apply() -> None:
    h = Harness()
    outcome = run_gate(GateRequest(h.engine, SOURCE, V1, ["c1", "c2"]))
    assert commit_contract(h.engine, outcome, applied_cleanly=False) == "held"
    assert h.engine.labelled("SourceRecordContract") == {}
    assert commit_contract(h.engine, outcome, applied_cleanly=True) == "bootstrap"
    (contract,) = h.engine.labelled("SourceRecordContract").values()
    assert json.loads(contract["shape"])["name"] == {
        "types": ["string"],
        "required": True,
    }


def test_a_matching_delta_proceeds_and_records_nothing(harness: Harness) -> None:
    outcome = harness.sync([{"id": "c3", "name": "q", "image": "z"}])
    assert outcome.proceed and outcome.detail == {"verdict": "no_drift"}
    assert harness.engine.labelled("SchemaDriftReport") == {}


def test_breaking_drift_is_quarantined_reported_gapped_and_proposed(
    harness: Harness,
) -> None:
    outcome = harness.sync(V2)
    assert not outcome.proceed and outcome.advance is None
    detail = outcome.detail
    assert detail["verdict"] == "quarantine" and detail["records_held"] == 2
    assert [c["kind"] for c in detail["changes"]] == ["rename_candidate"]
    (report,) = harness.engine.labelled("SchemaDriftReport").values()
    assert report["epistemic_class"] == "observation"
    assert detail["gap_id"] in harness.engine.labelled("Gap")
    repair = detail["repair"]
    assert repair["renames"] == {"title": "name"}
    assert repair["shadow"]["validated"] and repair["shadow"]["ingested"] == 2
    (lease,) = harness.port.leases.values()
    assert lease["kind"] == "action.approval" and lease["status"] == "active"
    assert lease["grant"]["target"] == f"approved:{SOURCE}"
    assert lease["grant"]["candidate_digest"] == repair["candidate_digest"]
    # validated by EG on the shadow graph (typed contract, no RDF), never attached
    assert [(g, op) for g, op, *_ in harness.port.attached] == [
        (repair["shadow"]["graph"], "validate")
    ]
    assert harness.shadow_ingested == [repair["shadow"]["graph"]]
    (contract,) = harness.engine.labelled("SourceRecordContract").values()
    assert contract["approved_by"] == "bootstrap"


def test_a_resync_of_the_held_delta_does_not_propose_twice(harness: Harness) -> None:
    harness.sync(V2)
    again = harness.sync(V2)
    assert not again.proceed and again.detail["repair"] == "pending_approval"
    assert len(harness.port.leases) == 1 and len(harness.shadow_ingested) == 1


def test_only_an_approved_repair_is_activated_and_then_the_delta_flows(
    harness: Harness,
) -> None:
    proposed = harness.sync(V2).detail["repair"]
    lease_id = proposed["approval_lease_id"]
    assert not harness.sync(V2).proceed  # still pending: nothing activated
    harness.port.decide(lease_id, "consumed")  # a human approved it
    after = harness.sync(V2)
    assert after.proceed and after.detail["verdict"] == "no_drift"
    assert after.detail["activation"]["approval_lease_id"] == lease_id
    live = [a for a in harness.port.attached if a[0] == "live"]
    assert live == [
        ("live", lease_id, f"approved:{SOURCE}", record_contract(infer_shape(V2)))
    ]
    assert harness.port.leases[lease_id]["status"] == "expired"
    (contract,) = harness.engine.labelled("SourceRecordContract").values()
    assert contract["approved_by"] == lease_id
    (proposal,) = harness.engine.labelled("SchemaRepairProposal").values()
    assert proposal["status"] == "activated"


def test_a_refused_approval_retires_its_proposal_and_keeps_containing(
    harness: Harness,
) -> None:
    lease_id = harness.sync(V2).detail["repair"]["approval_lease_id"]
    harness.port.decide(lease_id, "revoked")
    held = harness.sync(V2)
    assert not held.proceed
    statuses = {
        p["status"] for p in harness.engine.labelled("SchemaRepairProposal").values()
    }
    assert "refused" in statuses
    assert not [a for a in harness.port.attached if a[0] == "live"]


def test_an_eg_refusal_of_the_attach_leaves_the_contract_and_the_hold(
    harness: Harness,
) -> None:
    lease_id = harness.sync(V2).detail["repair"]["approval_lease_id"]
    harness.port.decide(lease_id, "consumed")
    harness.port.refuse_approved = "SCHEMA_APPROVAL_MISMATCH: candidate_digest"
    outcome = harness.sync(V2)
    assert not outcome.proceed
    assert "SCHEMA_APPROVAL_MISMATCH" in outcome.detail["activation"]["error"]
    (contract,) = harness.engine.labelled("SourceRecordContract").values()
    assert contract["approved_by"] == "bootstrap"


def test_declared_compatible_drift_continues_and_widens_the_contract(
    harness: Harness,
) -> None:
    grown = [{**r, "zone": "z"} for r in V1]
    held = harness.sync(grown)
    assert not held.proceed, "undeclared additive drift is contained too"
    outcome = harness.sync(grown, policy=f"{SOURCE}=additive_required")
    assert outcome.proceed and outcome.detail["verdict"] == "continue"
    (contract,) = harness.engine.labelled("SourceRecordContract").values()
    assert contract["approved_by"] == "policy" and "zone" in json.loads(
        contract["shape"]
    )


def test_an_unreadable_contract_store_quarantines(harness: Harness) -> None:
    harness.engine.backend.fail_reads = True
    outcome = run_gate(GateRequest(harness.engine, SOURCE, V1, ["c1", "c2"]))
    assert not outcome.proceed and "store offline" in outcome.detail["reason"]


def test_renames_are_one_to_one_and_follow_the_decision() -> None:
    from agent_utilities.knowledge_graph.schema_drift import classify

    changes = classify(infer_shape(V1), infer_shape(V2))
    assert resolve_renames(changes, lambda f, c, fb: fb) == {"title": "name"}
    assert resolve_renames(changes, lambda f, c, fb: None) == {}


def test_the_candidate_digest_is_egs_framing() -> None:
    # Pins the same bytes EG's approval tests pin for this one-field contract.
    contract = {"fields": {"name": {"required": True, "types": ["string"]}}}
    source_id = "approved:container-manager-mcp"
    framed = f"{APPROVED_CANDIDATE_DOMAIN}\0{source_id}\0"
    framed += '{"fields":{"name":{"required":true,"types":["string"]}}}\0'
    expected = hashlib.sha256(framed.encode()).hexdigest()
    assert approved_candidate_digest(source_id, contract) == expected
    assert record_contract(infer_shape([{"name": "x"}])) == contract


def test_the_fleet_drain_never_actuates_a_schema_repair_approval() -> None:
    from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

    row = {
        "a": {"id": "action_approval:x", "kind": "schema_repair", "status": "consumed"}
    }
    assert FleetReconciler._approval_candidate_props(row) is None
