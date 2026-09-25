"""The single L5 writer builds a TerminalOutcomeExtension EG accepts.

``_check_bound`` mirrors the binding rules of EG's
``eg_types::outcome_bundle`` (``validate_receipt_nodes``,
``validate_receipt_properties``, ``RunEvent::validate_for_bundle``) that the
engine applies before committing, with the engine-filled ``outbox_id`` /
``payload_digest`` left empty exactly as EG-4 requires.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import msgpack
import pytest

from agent_utilities.layers.contracts import (
    RunEvent,
    RunEventKind,
    RunResult,
    RunStatus,
    TraceCompleteness,
    UsageRecord,
)
from agent_utilities.layers.l5_writer import (
    DelegationBinding,
    L5CommitRejected,
    L5WriterUnavailable,
    LeaseClaim,
    RunOutcomeWriter,
    delegation_metadata,
)

DIGEST = hashlib.sha256(b"x").hexdigest()
BINDING = DelegationBinding(
    delegation_id="job-1",
    delegator_id="agent:caller",
    selected_agent_id="expert",
    capability_digest=DIGEST,
    catalog_digest=DIGEST,
    policy_digest=DIGEST,
    model_digest=DIGEST,
    carrier_digest=DIGEST,
)
CLAIM = LeaseClaim(
    tenant="tenant:t",
    work_item_id="wi-1",
    worker_id="worker-7",
    lease_epoch=3,
    fencing_token=9,
)
IDENTITY = (
    "delegation_id",
    "delegator_id",
    "selected_agent_id",
    "executor_lease_actor",
    "outcome",
    "work_item_id",
    "run_id",
    "outbox_id",
    "capability_digest",
    "catalog_digest",
    "policy_digest",
    "model_digest",
)


def _event(seq: int, kind: RunEventKind, name: str = "") -> RunEvent:
    return RunEvent(
        run_id="run-1",
        seq=seq,
        kind=kind,
        evidence="observation",
        fidelity="tool-calls",
        spec_digest=DIGEST,
        name=name,
    )


TRACE = (
    _event(0, "started"),
    _event(1, "tool_call", "Bash"),
    _event(2, "tool_result", "Bash"),
    _event(3, "completed"),
)


def _result(
    status: RunStatus = "succeeded", trace: TraceCompleteness = "complete"
) -> RunResult:
    return RunResult(
        run_id="run-1",
        spec_digest=DIGEST,
        harness="claude-code",
        status=status,
        output="DONE" if status == "succeeded" else "",
        usage=UsageRecord(quality="measured", source="t", input_tokens=3),
        error="" if status == "succeeded" else "boom",
        environment="caller-managed-host",
        fidelity="tool-calls",
        trace=trace,
        high_watermark=3,
    )


class _WorkItems:
    def __init__(self, answer: Any = None, error: BaseException | None = None):
        self.answer = answer if answer is not None else {"status": "committed"}
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.landed: dict[str, Any] | None = None

    @staticmethod
    def receipt_properties(properties: dict[str, Any]) -> bytes:
        return bytes(msgpack.packb(properties, use_bin_type=True))

    async def commit_result(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.answer

    async def get_outcome(self, *, tenant: str, work_item_id: str):
        return self.landed


def _check_bound(extension: dict[str, Any], call: dict[str, Any]) -> None:
    bundle = extension["outcome_bundle"]
    _check_bundle(bundle, call)
    _check_nodes(bundle, extension["receipt_nodes"])
    _check_event(bundle, extension["run_event"])


def _check_bundle(bundle: dict[str, Any], call: dict[str, Any]) -> None:
    assert bundle["outcome"] == call["outcome"]
    assert bundle["work_item_id"] == call["work_item_id"]
    assert bundle["fence_token"] == call["fencing_token"]
    assert bundle["result_ref"] == call["result_ref"]
    assert bundle["executor_lease_actor"] == call["worker_id"]
    assert (bundle["result_ref"] is None) == (bundle["result_digest"] is None)
    refs = [bundle["trace_ref"], bundle["outcome_ref"], *bundle["tool_call_refs"]]
    assert len(set(refs)) == len(refs)
    assert all("/" not in ref and "://" not in ref for ref in refs)


def _check_nodes(bundle: dict[str, Any], receipt_nodes: list[dict[str, Any]]) -> None:
    refs = [bundle["trace_ref"], bundle["outcome_ref"], *bundle["tool_call_refs"]]
    nodes = {node["node_id"]: node for node in receipt_nodes}
    assert set(nodes) == set(refs)
    kinds = {
        bundle["trace_ref"]: "run_trace",
        bundle["outcome_ref"]: "outcome_evaluation",
    }
    for node_id, node in nodes.items():
        assert node["kind"] == kinds.get(node_id, "tool_call")
        assert node["outbox_id"] == "" and node["payload_digest"] == ""
        assert node["result_ref"] == bundle["result_ref"]
        assert node["fence_token"] == bundle["fence_token"]
        props = msgpack.unpackb(node["properties_msgpack"])
        for key in IDENTITY:
            assert props[key] == bundle[key], key
        assert props["node_id"] == node_id
        assert props["kind"] == node["kind"]
        assert props["payload_ref"] == node["payload_ref"]
        assert props["fence_token"] == bundle["fence_token"]
        assert props["event_sequence"] == bundle["event_sequence"]
        assert props["completeness"] == bundle["completeness"]
        assert props["missing_refs"] == bundle["missing_refs"]
        assert props["result_digest"] == bundle["result_digest"]


def _check_event(bundle: dict[str, Any], event: dict[str, Any]) -> None:
    for key in (*IDENTITY, "fence_token", "result_ref", "event_sequence"):
        assert event[key] == bundle[key], key
    assert event["completeness"] == bundle["completeness"]
    assert event["missing_refs"] == bundle["missing_refs"]
    assert event["outcome_ref"] == bundle["outcome_ref"]
    expected_kind = "outcome" if bundle["completeness"] == "complete" else "degraded"
    assert event["kind"] == expected_kind
    for digest in (event["payload_digest"], event["carrier_digest"]):
        assert len(digest) == 64 and digest == digest.lower()


async def test_successful_run_commits_complete_provenance_atomically() -> None:
    work_items = _WorkItems()
    receipt = await RunOutcomeWriter(work_items).commit(
        CLAIM, BINDING, _result(), TRACE
    )
    (call,) = work_items.calls
    extension = call["outcome_extension"]
    _check_bound(extension, call)
    assert receipt.status == "committed"
    assert receipt.completeness == "complete"
    assert extension["outcome_bundle"]["missing_refs"] == []
    assert len(receipt.tool_call_refs) == 1
    assert call["retryable"] is False and call["error_ref"] is None
    outcome = msgpack.unpackb(extension["receipt_nodes"][-1]["properties_msgpack"])
    assert outcome["body"]["output"] == "DONE"
    assert outcome["body"]["usage"]["quality"] == "measured"


async def test_uncertain_run_is_a_non_retryable_degraded_failure() -> None:
    work_items = _WorkItems()
    receipt = await RunOutcomeWriter(work_items).commit(
        CLAIM, BINDING, _result("outcome_uncertain", "incomplete"), TRACE
    )
    (call,) = work_items.calls
    _check_bound(call["outcome_extension"], call)
    bundle = call["outcome_extension"]["outcome_bundle"]
    assert call["outcome"] == "failed" and call["retryable"] is False
    assert call["result_ref"] is None and call["error_ref"]
    assert bundle["completeness"] == "degraded"
    assert "reconciliation_receipt" in bundle["missing_refs"]
    assert "trace_after_seq:3" in bundle["missing_refs"]
    assert receipt.outcome == "failed"


async def test_transport_failure_is_reconciled_by_reading_back() -> None:
    work_items = _WorkItems(error=ConnectionResetError("lost"))
    writer = RunOutcomeWriter(work_items)
    uncertain = await writer.commit(CLAIM, BINDING, _result(), TRACE)
    assert uncertain.status == "uncertain" and uncertain.reconciled
    work_items.landed = {"outcome_ref": uncertain.outcome_ref}
    landed = await writer.commit(CLAIM, BINDING, _result(), TRACE)
    assert landed.status == "committed" and landed.reconciled


async def test_engine_rejection_and_unknown_status_are_typed() -> None:
    rejected = _WorkItems(error=RuntimeError("CONFLICT: fence"))
    with pytest.raises(L5CommitRejected):
        await RunOutcomeWriter(rejected).commit(CLAIM, BINDING, _result(), TRACE)
    odd = _WorkItems(answer={"status": "mystery"})
    with pytest.raises(L5CommitRejected):
        await RunOutcomeWriter(odd).commit(CLAIM, BINDING, _result(), TRACE)
    replay = await RunOutcomeWriter(_WorkItems(answer={"status": "noop"})).commit(
        CLAIM, BINDING, _result(), TRACE
    )
    assert replay.status == "replayed"


async def test_writer_fails_closed_without_the_eg_extension_surface() -> None:
    class _Old:
        async def commit_result(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("must not commit without receipts")

    with pytest.raises(L5WriterUnavailable):
        await RunOutcomeWriter(_Old()).commit(CLAIM, BINDING, _result(), TRACE)
    with pytest.raises(L5WriterUnavailable):
        RunOutcomeWriter(None)


def test_admission_metadata_carries_the_bindings_eg_checks() -> None:
    assert delegation_metadata(BINDING, "run-1") == {
        "delegation_id": "job-1",
        "run_id": "run-1",
        "agent_id": "expert",
        "capability_digest": DIGEST,
    }


async def test_engine_writer_uses_one_paired_work_item_transport() -> None:
    work_items = _WorkItems()
    engine = SimpleNamespace(
        client=SimpleNamespace(work_items=work_items),
        async_client=SimpleNamespace(work_items=work_items),
    )
    receipt = await RunOutcomeWriter.from_engine(engine).commit(
        CLAIM, BINDING, _result(), TRACE
    )
    assert receipt.status == "committed"
    assert len(work_items.calls) == 1
    with pytest.raises(L5WriterUnavailable):
        RunOutcomeWriter.from_engine(SimpleNamespace(client=engine.client))
