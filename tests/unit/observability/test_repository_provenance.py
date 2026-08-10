"""RMDD-19 — repository job provenance bridge: event/redaction/idempotency/query tests.

Covers the lane's required test/evidence list from the lane brief:
monotonic/idempotent lifecycle across retries/restart (no duplicate terminal
event), the WorkItem -> RunTrace -> ToolCall chain queryable from one job ID,
cross-tenant query denial, a redaction corpus proving a known-bad input is
caught, bounded/no-graph-bloat payload storage, a stale fence refusing to
claim effect success, and graph-unavailable behavior that refuses loudly
instead of fabricating evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from agent_utilities.observability.repository_provenance import (
    REPOSITORY_EVENT_KINDS,
    STALE_FENCE_REFUSAL_CODE,
    RepositoryProvenanceUnavailable,
    StaleFenceError,
    explain_repository_job,
    query_repository_provenance,
    reconciliation_report,
    repository_event_id,
    repository_run_id,
    write_repository_event,
)


class _FakeEngine:
    """Minimal in-memory stand-in for the graph engine (add_node/link_nodes/query_cypher).

    Mirrors the ad hoc fake-engine pattern already used by
    ``tests/unit/observability/test_trace_ontology.py`` — ``add_node`` is an
    upsert (matches the real ``epistemic_graph_backend.add_node`` MERGE
    semantics) so replay-idempotency assertions are meaningful.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.add_node_calls = 0

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.add_node_calls += 1
        existing = self.nodes.get(node_id, {})
        merged = {**existing, **(properties or {}), "node_type": node_type}
        self.nodes[node_id] = merged

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: Any = None,
    ) -> None:
        edge = (source_id, target_id, rel_type)
        if edge not in self.edges:
            self.edges.append(edge)

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        work_item_ref = params.get("work_item_ref")
        rows = [
            node
            for node in self.nodes.values()
            if node.get("node_type") == "ToolCall"
            and node.get("work_item_ref") == work_item_ref
        ]
        if "tenant_ref" in params:
            rows = [n for n in rows if n.get("tenant_ref") == params.get("tenant_ref")]
        if "fence_ref" in query and "fence_ref" not in params:
            rows = [n for n in rows if n.get("fence_ref")]
        rows = sorted(rows, key=lambda n: int(n.get("event_sequence") or 0), reverse=True)
        limit = int(params.get("limit") or len(rows) or 1)
        if "RETURN t.fence_ref" in query:
            return [
                {"fence_ref": n.get("fence_ref"), "event_sequence": n.get("event_sequence")}
                for n in rows[:limit]
            ]
        return [{"t": n} for n in rows[:limit]]


class _UnreachableEngine:
    def add_node(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("private endpoint host down")

    def link_nodes(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("private endpoint host down")

    def query_cypher(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise ConnectionError("private endpoint host down")


def test_all_lane_brief_event_kinds_are_registered() -> None:
    expected = {
        "submitted",
        "dependency_ready",
        "lease_claimed",
        "admission_placement",
        "started",
        "heartbeat",
        "checkpoint",
        "cancelled",
        "retried",
        "dead_lettered",
        "command_result",
        "artifact_published",
        "validation_certificate",
        "candidate_event",
        "generation_event",
        "bisection_event",
        "concept_event",
        "landing_push",
        "gc_reconcile",
    }
    assert REPOSITORY_EVENT_KINDS == expected


def test_unknown_kind_is_rejected() -> None:
    engine = _FakeEngine()
    with pytest.raises(ValueError, match="unknown repository event kind"):
        write_repository_event(
            engine,
            work_item_id="workitem:repository_manager:fixture-1",
            attempt=1,
            kind="not_a_real_kind",
            occurrence=0,
            status="ok",
        )


def test_engine_unavailable_refuses_loudly_never_fabricates_success() -> None:
    """H-12: an emitter that cannot reach the graph must refuse, not silently drop."""

    with pytest.raises(RepositoryProvenanceUnavailable, match="graph authority"):
        write_repository_event(
            None,
            work_item_id="workitem:repository_manager:fixture-2",
            attempt=1,
            kind="submitted",
            occurrence=0,
            status="submitted",
        )

    with pytest.raises(RepositoryProvenanceUnavailable) as exc_info:
        write_repository_event(
            _UnreachableEngine(),
            work_item_id="workitem:repository_manager:fixture-2",
            attempt=1,
            kind="submitted",
            occurrence=0,
            status="submitted",
        )
    # The underlying transport error is never swallowed silently, but its raw
    # text also never leaks past the boundary as the ONLY evidence available.
    assert "repository provenance write failed" in str(exc_info.value)


def test_replay_of_same_event_is_idempotent_no_duplicate_node() -> None:
    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-3"
    first = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="started",
        occurrence=0,
        status="running",
        payload={"note": "attempt started"},
    )
    node_count_after_first = len(engine.nodes)
    add_node_calls_after_first = engine.add_node_calls

    # Simulate a retry/restart replaying the exact same logical event.
    second = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="started",
        occurrence=0,
        status="running",
        payload={"note": "attempt started"},
    )

    assert first["event_id"] == second["event_id"]
    assert first["run_id"] == second["run_id"]
    # add_node was called again (upsert), but it resolved to the SAME node ids
    # -- no new node was created.
    assert len(engine.nodes) == node_count_after_first
    assert engine.add_node_calls > add_node_calls_after_first


def test_deterministic_ids_never_depend_on_wall_clock() -> None:
    run_id_a = repository_run_id("workitem:repository_manager:fixture-4", 1)
    run_id_b = repository_run_id("workitem:repository_manager:fixture-4", 1)
    assert run_id_a == run_id_b
    event_id_a = repository_event_id(
        "workitem:repository_manager:fixture-4", 1, "heartbeat", 3
    )
    event_id_b = repository_event_id(
        "workitem:repository_manager:fixture-4", 1, "heartbeat", 3
    )
    assert event_id_a == event_id_b
    # A different attempt or occurrence must never collide.
    assert event_id_a != repository_event_id(
        "workitem:repository_manager:fixture-4", 2, "heartbeat", 3
    )
    assert event_id_a != repository_event_id(
        "workitem:repository_manager:fixture-4", 1, "heartbeat", 4
    )


def test_full_chain_queryable_from_one_job_id() -> None:
    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-5"
    tenant_ref = "tenant:fixture"

    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="submitted",
        occurrence=0,
        status="submitted",
        correlations={"repo": "repository-manager"},
    )
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-1",
        correlations={"owner": "worker-a"},
    )
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="started",
        occurrence=0,
        status="running",
        fence="fence-epoch-1",
    )
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="command_result",
        occurrence=0,
        status="succeeded",
        fence="fence-epoch-1",
        correlations={"artifact_id": "artifact-123"},
    )
    # Stamp every fake-engine node with the tenant the query below asks for
    # (the real writer stamps this from ambient actor context; the fake
    # engine has no ambient actor in this unit test, so stamp directly).
    for node in engine.nodes.values():
        node["tenant_ref"] = tenant_ref

    events = query_repository_provenance(
        engine, work_item_id=work_item_id, tenant_ref=tenant_ref
    )
    assert [event["event_kind"] for event in events] == [
        "submitted",
        "lease_claimed",
        "started",
        "command_result",
    ]
    # Ordered by the monotonic event_sequence, never by wall-clock/lexical order.
    sequences = [int(event["event_sequence"]) for event in events]
    assert sequences == sorted(sequences)

    explanation = explain_repository_job(
        engine, work_item_id=work_item_id, tenant_ref=tenant_ref
    )
    assert explanation["found"] is True
    assert explanation["event_count"] == 4
    assert explanation["latest_kind"] == "command_result"
    assert explanation["latest_status"] == "succeeded"
    assert explanation["terminal"] is True


def test_cross_tenant_query_denies() -> None:
    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-6"
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="submitted",
        occurrence=0,
        status="submitted",
    )
    for node in engine.nodes.values():
        node["tenant_ref"] = "tenant:owner"

    same_tenant = query_repository_provenance(
        engine, work_item_id=work_item_id, tenant_ref="tenant:owner"
    )
    assert len(same_tenant) == 1

    other_tenant = query_repository_provenance(
        engine, work_item_id=work_item_id, tenant_ref="tenant:intruder"
    )
    assert other_tenant == []


def test_query_without_any_tenant_scope_refuses() -> None:
    engine = _FakeEngine()
    with pytest.raises(ValueError, match="tenant scope"):
        query_repository_provenance(
            engine, work_item_id="workitem:repository_manager:fixture-7"
        )


def test_genuinely_empty_result_is_a_normal_pass_not_a_degraded_read() -> None:
    engine = _FakeEngine()
    events = query_repository_provenance(
        engine,
        work_item_id="workitem:repository_manager:fixture-empty",
        tenant_ref="tenant:fixture",
    )
    assert events == []
    explanation = explain_repository_job(
        engine,
        work_item_id="workitem:repository_manager:fixture-empty",
        tenant_ref="tenant:fixture",
    )
    assert explanation == {
        "work_item_id": "workitem:repository_manager:fixture-empty",
        "found": False,
        "event_count": 0,
        "events": [],
        "latest_status": None,
        "latest_kind": None,
        "latest_event_sequence": None,
        "terminal": False,
    }


def test_query_unreachable_engine_raises_not_empty() -> None:
    with pytest.raises(RepositoryProvenanceUnavailable):
        query_repository_provenance(
            _UnreachableEngine(),
            work_item_id="workitem:repository_manager:fixture-8",
            tenant_ref="tenant:fixture",
        )


def test_reconciliation_report_links_facts_to_a_proposed_repair_without_submitting_one() -> (
    None
):
    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-9"
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-1",
    )
    for node in engine.nodes.values():
        node["tenant_ref"] = "tenant:fixture"

    report = reconciliation_report(
        engine, work_item_id=work_item_id, tenant_ref="tenant:fixture"
    )
    assert report["terminal"] is False
    assert report["proposed_repair"] == {
        "kind": "reclaim_and_relaunch",
        "reason": "leased but never started",
    }
    assert any("events recorded" in fact for fact in report["observed_facts"])


def test_stale_fence_cannot_claim_effect_success() -> None:
    """Lane acceptance gate: a stale fence event cannot claim effect success."""

    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-10"

    # Attempt 1 leases with fence F1 and (eventually, believed by attempt 1's
    # own worker) succeeds.
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-1",
    )
    ok = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="command_result",
        occurrence=0,
        status="succeeded",
        fence="fence-epoch-1",
    )
    assert ok["status"] == "succeeded"

    # The lane reclaims the job (the original worker went dark) and re-leases
    # under a NEW fence for attempt 2.
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=2,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-2",
    )

    # The ORIGINAL (attempt 1) worker was merely slow, not dead, and now
    # delivers its own "succeeded" using the now-superseded fence. This must
    # be refused, not recorded as if it were the current outcome.
    with pytest.raises(StaleFenceError) as exc_info:
        write_repository_event(
            engine,
            work_item_id=work_item_id,
            attempt=1,
            kind="command_result",
            occurrence=1,
            status="succeeded",
            fence="fence-epoch-1",
        )
    assert exc_info.value.refusal_code == STALE_FENCE_REFUSAL_CODE
    assert exc_info.value.refusal_code == "stale_fence_duplicate_effect"

    # The CURRENT (attempt 2) fence is still free to claim success.
    current = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=2,
        kind="command_result",
        occurrence=0,
        status="succeeded",
        fence="fence-epoch-2",
    )
    assert current["status"] == "succeeded"


# --- H-9: prove the redaction gate catches a known-bad input --------------


def _flatten_values(node: Mapping[str, Any]) -> str:
    return json.dumps(node, sort_keys=True, default=str)


def test_redaction_gate_catches_a_known_bad_input_secret_path_and_raw_command() -> (
    None
):
    """H-9: feed the writer a fixture with a real-shaped secret, an absolute
    machine path, and a raw command body, and prove none of the three ever
    reach the graph verbatim -- and that the guard actually detected them
    (not merely a lucky truncation).
    """

    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-redaction"

    real_shaped_secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEF"
    absolute_machine_path = "/home/exampleuser/.ssh/id_rsa"
    raw_command_body = (
        f"scp {absolute_machine_path} deploy@internal:/var/backups && "
        f"curl -H 'Authorization: Bearer {real_shaped_secret}' "
        "https://internal.example/api"
    )

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="command_result",
        occurrence=0,
        status="failed",
        error="command failed",
        payload={"argv": raw_command_body},
        correlations={"artifact_id": absolute_machine_path},
    )

    event_node = engine.nodes[result["event_id"]]
    dumped = _flatten_values(event_node)

    # 1. None of the three known-bad inputs appear verbatim anywhere in the
    #    persisted node.
    assert real_shaped_secret not in dumped
    assert absolute_machine_path not in dumped
    assert raw_command_body not in dumped

    # 2. The payload is never stored raw -- only a digest + bounded count --
    #    regardless of whether a specific pattern matched (bounded/no-bloat
    #    requirement).
    assert event_node["args"] == ""
    assert event_node["args_digest"]
    assert event_node["args_character_count"] > 0

    # 3. The guard did not merely blank the field -- it actually recognized
    #    the bearer-token / api-token / posix-user-path shapes before
    #    blanking them (proves detection, not a blind truncation).
    detected_types = set(event_node.get("privacy_types") or [])
    assert detected_types & {"bearer_token", "api_token", "posix_user_path"}
    assert event_node["privacy_redactions"] > 0

    # 4. The correlation reference derived from the raw path is an opaque
    #    HMAC reference, never the raw path itself or a trivially reversible
    #    encoding of it.
    assert event_node["artifact_id_ref"].startswith("pref_artifact_id_")
    assert absolute_machine_path not in event_node["artifact_id_ref"]


def test_oversized_payload_is_bounded_not_graph_bloat() -> None:
    engine = _FakeEngine()
    huge_payload = {"blob": "x" * 50_000}
    result = write_repository_event(
        engine,
        work_item_id="workitem:repository_manager:fixture-oversized",
        attempt=1,
        kind="artifact_published",
        occurrence=0,
        status="ok",
        payload=huge_payload,
    )
    event_node = engine.nodes[result["event_id"]]
    # tool_call_properties truncates the sanitize input to 4000 chars before
    # digesting -- the stored representation must never scale with the raw
    # payload size.
    assert event_node["args"] == ""
    assert len(json.dumps(event_node)) < 5_000


def test_no_fence_supplied_skips_fence_guard_best_effort() -> None:
    """A kind with no fence concept (e.g. concept_event) must not be forced
    to supply one -- the guard is opt-in via a non-empty fence, not
    mandatory for every terminal-shaped kind."""

    engine = _FakeEngine()
    result = write_repository_event(
        engine,
        work_item_id="workitem:repository_manager:fixture-nofence",
        attempt=1,
        kind="concept_event",
        occurrence=0,
        status="reserved",
    )
    assert result["kind"] == "concept_event"


def test_landing_push_landed_status_is_also_fence_protected() -> None:
    """``landing_push``'s success vocabulary is 'landed', not 'succeeded'
    (repository_manager.development.enums.LandingOutcome) -- the fence guard
    must protect it exactly like a command_result success, not silently
    exempt it because the status string differs."""

    engine = _FakeEngine()
    work_item_id = "workitem:repository_manager:fixture-landing"

    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-1",
    )
    landed = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="landing_push",
        occurrence=0,
        status="landed",
        fence="fence-epoch-1",
    )
    assert landed["status"] == "landed"

    # A new lease supersedes the fence.
    write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=2,
        kind="lease_claimed",
        occurrence=0,
        status="leased",
        fence="fence-epoch-2",
    )
    # A stale replay of the OLD landing success must be refused.
    with pytest.raises(StaleFenceError):
        write_repository_event(
            engine,
            work_item_id=work_item_id,
            attempt=1,
            kind="landing_push",
            occurrence=1,
            status="landed",
            fence="fence-epoch-1",
        )
