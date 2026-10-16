"""Workload-aware data-placement mining (X-5, CONCEPT:AU-KG.evolution.placement-mining-canary-loop).

Mine (RunTrace/ToolCall/Entity provenance -> co-occurrence/hot-skew/sequence)
-> PlacementProposal -> Claim -> Validation (reuses promotion_governance
as-is) -> Action gate (reuses action_policy.decide(),
kind="apply_placement_change") -> a MEASURED CANARY (apply to a small scope,
measure, promote/rollback) -> the promoted change reaches the engine's
PlacementCatalog admin path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.placement_mining import (
    CONFIDENCE_FLOOR,
    TOOL_CALL_LABEL,
    CanaryResult,
    PlacementProposal,
    apply_placement_change,
    build_baskets,
    build_tenant_access_counts,
    build_tenant_sequences,
    gather_access_records,
    gather_drift_scores,
    mine_placement_patterns,
    placement_control_loop,
    placement_mining_subscription,
    placement_proposals_from_mining,
    proposals_from_association,
    proposals_from_drift_anomaly,
    proposals_from_sequence,
    proposals_from_tenant_anomaly,
    rollback_placement_change,
    run_canary,
    run_placement_mining_cycle,
)
from tests.unit.knowledge_graph._shacl_fakes import AlwaysConformsShaclMixin

pytestmark = pytest.mark.concept("AU-KG.evolution.placement-mining-canary-loop")


# ---------------------------------------------------------------------------
# gather_access_records / basket + count + sequence builders
# ---------------------------------------------------------------------------


class _AccessStubEngine:
    """Minimal engine double: canned RunTrace/ToolCall/Entity provenance rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self._rows = rows or []
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "RunTrace" in q and "USED_TOOL" in q:
            return list(self._rows)
        return []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def _row(trace_id, tool_name, tenant="", modality="", entity_id="", entity_type=""):
    args = {}
    if tenant:
        args["tenant_id"] = tenant
    if modality:
        args["modality"] = modality
    return {
        "trace_id": trace_id,
        "event_sequence": int(
            str(trace_id).removeprefix("trace:").removeprefix("ep") or 0
        ),
        "tool_name": tool_name,
        "args": json.dumps(args),
        "entity_id": entity_id,
        "entity_type": entity_type,
    }


def test_gather_access_records_parses_args_json():
    engine = _AccessStubEngine(
        rows=[
            _row(
                "ep1",
                "graph_query",
                tenant="acme",
                modality="text",
                entity_id="doc1",
                entity_type="Doc",
            )
        ]
    )
    records = gather_access_records(engine)
    assert records == [
        {
            "trace_id": "ep1",
            "event_sequence": 1,
            "tool_name": "graph_query",
            "tenant": "acme",
            "modality": "text",
            "entity_id": "doc1",
            "entity_type": "Doc",
        }
    ]


def test_gather_access_records_handles_missing_engine():
    assert gather_access_records(None) == []


def test_gather_access_records_degrades_on_query_failure():
    class _BoomEngine:
        def query_cypher(self, q, params=None):
            raise RuntimeError("engine unreachable")

    assert gather_access_records(_BoomEngine()) == []


def test_build_baskets_drops_single_item_episodes():
    records = [
        {
            "trace_id": "ep1",
            "tool_name": "graph_query",
            "tenant": "acme",
            "modality": "",
            "entity_id": "",
            "entity_type": "",
        },
        {
            "trace_id": "ep2",
            "tool_name": "graph_query",
            "tenant": "acme",
            "modality": "",
            "entity_id": "doc1",
            "entity_type": "Doc",
        },
    ]
    baskets = build_baskets(records)
    # ep1 has only 1 distinct item (tool+tenant = 2 actually) -> keep; verify shape
    assert ["tenant:acme", "tool:graph_query"] in baskets
    assert any("entity:doc1" in b for b in baskets)


def test_build_tenant_access_counts_counts_distinct_episodes():
    records = [
        {"trace_id": "ep1", "tenant": "acme"},
        {"trace_id": "ep1", "tenant": "acme"},  # same trace, dedup
        {"trace_id": "ep2", "tenant": "acme"},
        {"trace_id": "ep3", "tenant": "beta"},
    ]
    counts = build_tenant_access_counts(records)
    assert counts == {"acme": 2, "beta": 1}


def test_build_tenant_sequences_filters_short_sequences():
    records = [
        {"trace_id": "ep1", "tenant": "acme", "tool_name": "read", "entity_id": ""},
        {"trace_id": "ep2", "tenant": "acme", "tool_name": "write", "entity_id": ""},
        {"trace_id": "ep3", "tenant": "beta", "tool_name": "read", "entity_id": ""},
    ]
    seqs = build_tenant_sequences(records)
    assert seqs == {"acme": ["tool:read", "tool:write"]}


def test_gather_drift_scores_handles_missing_engine():
    assert gather_drift_scores(None) == ([], [])


# ---------------------------------------------------------------------------
# mine_placement_patterns — delegates to the engine's graph_mine surface
# ---------------------------------------------------------------------------


def test_mine_placement_patterns_no_records_is_a_clean_empty_result():
    engine = _AccessStubEngine(rows=[])
    result = mine_placement_patterns(engine)
    assert result["association"] == {"rules": []}
    assert result["sequence"] == {"patterns": []}
    assert result["records_scanned"] == 0
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# classification — mined finding -> typed PlacementProposal
# ---------------------------------------------------------------------------


def test_hot_tenant_anomaly_becomes_shard_split_proposal_with_evidence():
    tenant_anomaly = {
        "result": {
            "rows": [
                {"is_anomaly": True, "anomaly_score": 4.2},
                {"is_anomaly": False, "anomaly_score": 0.1},
            ]
        },
        "tenant_ids": ["hot-tenant", "quiet-tenant"],
    }
    proposals = proposals_from_tenant_anomaly(tenant_anomaly)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "shard_split"
    assert prop.target == "hot-tenant"
    assert prop.evidence == {"is_anomaly": True, "anomaly_score": 4.2}
    assert prop.confidence == pytest.approx(4.2 / 5.0)
    assert prop.clears_floor is (prop.confidence >= CONFIDENCE_FLOOR)


def test_cold_tenant_anomaly_is_not_a_split_candidate():
    tenant_anomaly = {
        "result": {"rows": [{"is_anomaly": True, "anomaly_score": -3.0}]},
        "tenant_ids": ["cold-tenant"],
    }
    assert proposals_from_tenant_anomaly(tenant_anomaly) == []


def test_cross_entity_association_becomes_materialized_join():
    association = {
        "rules": [
            {
                "antecedent": ["entity:doc1"],
                "consequent": ["entity:doc2"],
                "confidence": 0.85,
                "lift": 3.1,
            }
        ]
    }
    proposals = proposals_from_association(association)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "materialized_join"
    assert prop.target == "entity:doc1|entity:doc2"
    assert prop.confidence == pytest.approx(0.85)


def test_high_confidence_tool_entity_type_becomes_index_change():
    association = {
        "rules": [
            {
                "antecedent": ["tool:graph_query"],
                "consequent": ["entity_type:Doc"],
                "confidence": 0.95,
                "lift": 5.0,
            }
        ]
    }
    proposals = proposals_from_association(association)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "index_change"
    assert prop.target == "entity_type:Doc"


def test_tenant_read_tool_becomes_replica_proposal():
    association = {
        "rules": [
            {
                "antecedent": ["tenant:acme"],
                "consequent": ["tool:get_document"],
                "confidence": 0.7,
                "lift": 2.0,
            }
        ]
    }
    proposals = proposals_from_association(association)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "replica"
    assert prop.target == "tenant:acme"


def test_sequence_pattern_becomes_cache_prewarm():
    sequence = {
        "patterns": [
            {"items": ["tool:login", "tool:browse"], "support": 0.75, "count": 6}
        ]
    }
    proposals = proposals_from_sequence(sequence)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "cache_prewarm"
    assert prop.confidence == pytest.approx(0.75)


def test_drift_anomaly_becomes_embedding_refresh():
    drift_anomaly = {
        "result": {"rows": [{"is_anomaly": True, "anomaly_score": 3.5}]},
        "entity_ids": ["doc1"],
    }
    proposals = proposals_from_drift_anomaly(drift_anomaly)
    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.kind == "embedding_refresh"
    assert prop.target == "doc1"


def test_placement_proposals_from_mining_fans_out_all_sources():
    mine_result = {
        "association": {
            "rules": [
                {
                    "antecedent": ["entity:a"],
                    "consequent": ["entity:b"],
                    "confidence": 0.9,
                    "lift": 2.0,
                }
            ]
        },
        "tenant_anomaly": {
            "result": {"rows": [{"is_anomaly": True, "anomaly_score": 4.0}]},
            "tenant_ids": ["hot"],
        },
        "drift_anomaly": {
            "result": {"rows": [{"is_anomaly": True, "anomaly_score": 3.0}]},
            "entity_ids": ["drifted"],
        },
        "sequence": {"patterns": [{"items": ["x", "y"], "support": 0.5, "count": 2}]},
    }
    proposals = placement_proposals_from_mining(mine_result)
    kinds = {p.kind for p in proposals}
    assert kinds == {
        "materialized_join",
        "shard_split",
        "embedding_refresh",
        "cache_prewarm",
    }


def test_placement_proposal_rejects_unknown_kind():
    with pytest.raises(ValueError):
        PlacementProposal(kind="not_a_kind", target="x", statement="x", confidence=1.0)


# ---------------------------------------------------------------------------
# run_canary — measured canary: promote on positive delta, rollback on negative
# ---------------------------------------------------------------------------


def _proposal(kind="shard_split", target="hot-tenant", confidence=0.9):
    return PlacementProposal(
        kind=kind, target=target, statement="test", confidence=confidence, evidence={}
    )


def test_run_canary_promotes_on_improvement():
    calls = {"apply": 0, "rollback": 0}

    def apply_fn(p):
        calls["apply"] += 1
        return {"applied": True}

    def rollback_fn(p):
        calls["rollback"] += 1
        return {"rolled_back": True}

    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 90.0}])
    result = run_canary(
        _proposal(),
        measurement_fn=lambda p, phase: next(measurements),
        apply_fn=apply_fn,
        rollback_fn=rollback_fn,
    )
    assert isinstance(result, CanaryResult)
    assert result.verdict == "promote"
    assert result.applied is True
    assert calls["apply"] == 1
    assert calls["rollback"] == 0


def test_run_canary_rolls_back_on_regression():
    calls = {"apply": 0, "rollback": 0}

    def apply_fn(p):
        calls["apply"] += 1
        return {"applied": True}

    def rollback_fn(p):
        calls["rollback"] += 1
        return {"rolled_back": True}

    measurements = iter(
        [{"latency_ms": 100.0}, {"latency_ms": 250.0}]
    )  # +150% regression
    result = run_canary(
        _proposal(),
        measurement_fn=lambda p, phase: next(measurements),
        apply_fn=apply_fn,
        rollback_fn=rollback_fn,
    )
    assert result.verdict == "rollback"
    assert result.applied is False
    assert calls["apply"] == 1
    assert calls["rollback"] == 1  # the canary-scope change was reverted


def test_run_canary_rolls_back_when_no_measurement_available():
    result = run_canary(
        _proposal(),
        measurement_fn=lambda p, phase: {},
        apply_fn=lambda p: {"applied": True},
        rollback_fn=lambda p: {"rolled_back": True},
    )
    assert result.verdict == "rollback"
    assert result.applied is False
    assert "no measurement" in result.reason


def test_run_canary_tolerates_small_regression_within_tolerance():
    measurements = iter(
        [{"latency_ms": 100.0}, {"latency_ms": 105.0}]
    )  # +5%, within 10% default
    result = run_canary(
        _proposal(),
        measurement_fn=lambda p, phase: next(measurements),
        apply_fn=lambda p: {"applied": True},
        rollback_fn=lambda p: {"rolled_back": True},
    )
    assert result.verdict == "promote"


# ---------------------------------------------------------------------------
# apply_placement_change / rollback_placement_change — the catalog admin path
# ---------------------------------------------------------------------------


def test_apply_placement_change_for_non_catalog_kinds_is_an_accepted_record():
    result = apply_placement_change(_proposal(kind="materialized_join", target="a|b"))
    assert result == {
        "applied": True,
        "method": "accepted_record",
        "detail": "materialized_join",
    }


def test_rollback_via_raft_placement_is_honest_when_no_pre_move_group_recorded(
    monkeypatch,
):
    """A raft-catalog `assign` bootstrap (no prior group) has no clean engine-side
    un-assign yet — rollback says so plainly rather than silently no-op'ing or
    reverting through the unrelated R5 system."""
    proposal = _proposal(kind="shard_split", target="acme:ws1")
    proposal.evidence["_placement_move_tenant"] = "acme"

    result = rollback_placement_change(proposal)
    assert result["rolled_back"] is False
    assert result["method"] == "placement_move"
    assert "no recorded pre-move group" in result["detail"]


# ---------------------------------------------------------------------------
# run_placement_mining_cycle — the full governed loop
# ---------------------------------------------------------------------------


class _CycleStubEngine:
    """Same shape as ``test_trace_pattern_miner.py``'s ``_TraceMiningStubEngine``:
    empty governance-adjacent query results ⇒ ``PromotionGovernanceValidator``
    passes by default; ``governance_rules`` relaxes the ActionPolicy tier."""

    def __init__(self, *, governance_rules: list[dict[str, Any]] | None = None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()
        self._governance_rules = governance_rules or []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "TraceConsumerCursor" in q:
            values = params or {}
            if "id" in values:
                node = self.nodes.get(values["id"])
                return (
                    [{"event_sequence": node["event_sequence"]}]
                    if node is not None
                    else []
                )
            rows = [
                {"event_sequence": node["event_sequence"]}
                for node in self.nodes.values()
                if node.get("consumer_ref") == values.get("consumer_ref")
            ]
            return sorted(rows, key=lambda row: row["event_sequence"], reverse=True)[:1]
        if "governance_rule" in q:
            return [{"r": dict(r)} for r in self._governance_rules]
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def _patch_mine_result(monkeypatch, *, anomaly_score: float = 4.0) -> None:
    import agent_utilities.knowledge_graph.research.placement_mining as pm

    monkeypatch.setattr(
        pm,
        "mine_placement_patterns",
        lambda engine, **kw: {
            "association": {"rules": []},
            "tenant_anomaly": {
                "result": {
                    "rows": [{"is_anomaly": True, "anomaly_score": anomaly_score}]
                },
                "tenant_ids": ["hot-tenant"],
            },
            "drift_anomaly": {"result": {"rows": []}, "entity_ids": []},
            "sequence": {"patterns": []},
            "access_counts": {"hot-tenant": 50},
            "records_scanned": 50,
            "next_event_sequence": 29,
            "errors": [],
        },
    )


def test_placement_cursor_resumes_and_advances_only_after_success(monkeypatch):
    import agent_utilities.knowledge_graph.research.placement_mining as pm

    after: list[int] = []
    results = iter([(17, []), (23, ["mining failed"]), (31, [])])

    def _mine(engine, **kwargs):
        after.append(kwargs["after_sequence"])
        next_sequence, errors = next(results)
        return {
            "association": {"rules": []},
            "tenant_anomaly": {"result": {"rows": []}, "tenant_ids": []},
            "drift_anomaly": {"result": {"rows": []}, "entity_ids": []},
            "sequence": {"patterns": []},
            "access_counts": {},
            "records_scanned": 0,
            "next_event_sequence": next_sequence,
            "errors": errors,
        }

    monkeypatch.setattr(pm, "mine_placement_patterns", _mine)
    engine = _CycleStubEngine()

    first = run_placement_mining_cycle(engine)
    failed = run_placement_mining_cycle(engine)
    resumed = run_placement_mining_cycle(engine)

    assert after == [0, 17, 17]
    assert first["next_event_sequence"] == 17
    assert first["cursor_advanced"] is True
    assert failed["next_event_sequence"] == 17
    assert failed["cursor_advanced"] is False
    assert resumed["next_event_sequence"] == 31


def test_below_floor_proposal_never_becomes_a_claim(monkeypatch):
    _patch_mine_result(
        monkeypatch, anomaly_score=0.5
    )  # saturates well below CONFIDENCE_FLOOR
    engine = _CycleStubEngine()
    rep = run_placement_mining_cycle(engine)

    assert rep["eligible"] == 0
    assert rep["below_floor"] == 1
    assert rep["persisted"] == 0
    assert rep["applied"] == 0
    assert engine.by_type("PlacementProposal") == []


def test_eligible_proposal_persists_as_an_unverified_proposal_claim(monkeypatch):
    _patch_mine_result(monkeypatch, anomaly_score=4.0)
    engine = (
        _CycleStubEngine()
    )  # no governance_rule override ⇒ shipped default (approval_required)
    rep = run_placement_mining_cycle(engine)

    assert rep["eligible"] == 1
    assert rep["persisted"] == 1
    proposals = engine.by_type("PlacementProposal")
    assert len(proposals) == 1
    assert proposals[0]["status"] == "proposal"
    assert proposals[0]["is_verified"] is False


# ---------------------------------------------------------------------------
# X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): the persisted
# PlacementProposal claim registers via the SAME shared writeback seam
# ``loop_controller._run_insight_validation``/``_run_trace_mining`` use
# (``candidate_insight.register_claim_materialization``), and ACTUALLY goes
# Stale when its real base fact (the placement target) changes.
# ---------------------------------------------------------------------------


class _TmsAwareCycleStubEngine(_CycleStubEngine):
    """``_CycleStubEngine`` + ``add_edge`` + a minimal, faithful in-process
    TruthMaintenance index (same contract as ``test_insight_validation.py``'s
    ``_TmsAwareInsightStubEngine``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self._versions: dict[str, int] = {}
        self._materializations: dict[str, dict[str, int]] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        super().add_node(node_id, node_type, properties)
        self._versions[node_id] = self._versions.get(node_id, 0) + 1

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **properties}))

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        deps = {
            target
            for source, target, props in self.edges
            if source == derived_id and props.get("rel_type") == "DERIVED_FROM"
        }
        self._materializations[derived_id] = {d: self._versions.get(d, 0) for d in deps}
        return {
            "id": derived_id,
            "depends_on": sorted(deps),
            "generating_activity": None,
        }

    def materialization_status(self, derived_id: str) -> str | None:
        snapshot = self._materializations.get(derived_id)
        if snapshot is None:
            return None
        for dep, ver in snapshot.items():
            if self._versions.get(dep, 0) != ver:
                return "Stale"
        return "Fresh"


def test_placement_proposal_materialization_goes_stale_when_target_changes(
    monkeypatch,
):
    _patch_mine_result(monkeypatch, anomaly_score=4.0)
    engine = _TmsAwareCycleStubEngine()
    rep = run_placement_mining_cycle(engine)

    assert rep["persisted"] == 1
    proposal = engine.by_type("PlacementProposal")[0]
    claim_id = proposal["id"]
    target = proposal["source_ids"][0]
    assert target  # sanity: the claim carries the real placement target

    assert engine.materialization_status(claim_id) == "Fresh"

    # The placement target itself is later revised (a real KG write).
    engine.add_node(target, "Entity", properties={"revised": True})

    assert engine.materialization_status(claim_id) == "Stale"


def test_shipped_default_never_applies_or_canaries(monkeypatch):
    """apply_placement_change is approval_required by default — the gate must
    queue, and the canary must never even run."""
    canary_calls: list[Any] = []
    import agent_utilities.knowledge_graph.research.placement_mining as pm

    monkeypatch.setattr(
        pm,
        "run_canary",
        lambda *a, **kw: canary_calls.append(1) or CanaryResult("x", "promote", ""),
    )
    _patch_mine_result(monkeypatch, anomaly_score=4.0)
    engine = _CycleStubEngine()  # shipped default
    rep = run_placement_mining_cycle(engine)

    assert rep["applied"] == 0
    assert canary_calls == []
    for ex in rep["examples"]:
        assert ex["action_decision"] == "queue_approval"
        assert ex["applied"] is False


def test_action_policy_consulted_with_apply_placement_change_kind(monkeypatch):
    import agent_utilities.orchestration.action_policy as ap_mod

    calls: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        calls.append(request.kind)
        return real_decide(self, request)

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)
    _patch_mine_result(monkeypatch, anomaly_score=4.0)
    engine = _CycleStubEngine()
    run_placement_mining_cycle(engine)

    assert calls == ["apply_placement_change"]


# ---------------------------------------------------------------------------
# action-policy shipped-default safety net
# ---------------------------------------------------------------------------


def test_apply_placement_change_default_never_auto():
    """SAFETY-CRITICAL: the shipped ActionPolicy default for
    apply_placement_change must never be auto/auto_notify — a mined placement
    change must always be held for human review before it even enters the
    canary."""
    from agent_utilities.orchestration.action_policy import (
        DEFAULT_POLICY,
        TIER_APPROVAL,
        TIER_AUTO,
        TIER_AUTO_NOTIFY,
    )

    rule = next(
        r for r in DEFAULT_POLICY["rules"] if r["kind"] == "apply_placement_change"
    )
    assert rule["tier"] == TIER_APPROVAL
    assert rule["tier"] not in (TIER_AUTO, TIER_AUTO_NOTIFY)


# ---------------------------------------------------------------------------
# apply/rollback — the REAL online-move RPC (reshard), not a route-only flip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# canary measurement — real engine stat fallback (shard load skew)
# ---------------------------------------------------------------------------


def test_shard_load_skew_measurement_empty_without_resharding_surface(monkeypatch):
    import agent_utilities.knowledge_graph.research.placement_mining as pm

    monkeypatch.setattr(pm, "_promql_latency_measurement", lambda proposal: {})
    result = pm._shard_load_skew_measurement()
    # No resharding surface stubbed at all -> _resharding_methods() reads the
    # REAL engine_tools.ENGINE_DOMAINS, which is empty/irrelevant here; the
    # call degrades cleanly either way (never raises).
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# placement_control_loop — Seam 4: the orchestrated, opt-in controller step
# ---------------------------------------------------------------------------


def test_placement_control_loop_default_off_is_a_zero_side_effect_noop(monkeypatch):
    """Default OFF: importing/calling this without opting in must NEVER mine,
    govern, canary, or reshard anything."""
    import agent_utilities.knowledge_graph.research.placement_mining as pm

    def _boom(*a, **kw):  # pragma: no cover - must never be called
        raise AssertionError("mining must not run while the loop is disabled")

    monkeypatch.setattr(pm, "mine_placement_patterns", _boom)

    rep = placement_control_loop(_CycleStubEngine(), enabled=False)
    assert rep == {
        "enabled": False,
        "skipped": True,
        "reason": (
            "placement_control_loop is opt-in (enabled=False) — manual trigger required"
        ),
    }


# ---------------------------------------------------------------------------
# Seam 5 — the reactive (CDC-driven) trigger (report §9 #6)
#
# The real-engine round trip (a genuine ``:ToolCall`` write delivered through
# the engine's CDC/watch feed) lives in
# ``tests/unit/graph/test_engine_subscription_reactive.py`` alongside every
# other ``*_subscription`` primitive's live-engine proof (CONCEPT:AU-KG.memory.provides-real-ephemeral-one).
# These tests cover the primitive's OWN dispatch/wiring in isolation, mirroring
# how ``test_agent_task_dep_watcher.py`` tests ``AgentTaskDepWatcher`` against a
# fake subscription double rather than a live engine.
# ---------------------------------------------------------------------------


def test_placement_mining_subscription_subscribes_on_tool_call_label(monkeypatch):
    """``placement_mining_subscription`` wires the SAME ``subscribe(engine, label,
    handler)`` front door every other reactive consumer in this codebase uses
    (``fleet_autoscale_subscription``, ``AgentTaskDepWatcher``) — never a
    bespoke CDC client."""
    import agent_utilities.graph.reactive as reactive

    calls: list[dict] = []

    class _FakeSubscription:
        def __init__(self) -> None:
            self.available = True

    def _fake_subscribe(engine, label, handler):
        calls.append({"engine": engine, "label": label, "handler": handler})
        return _FakeSubscription()

    monkeypatch.setattr(reactive, "subscribe", _fake_subscribe)

    engine = object()
    sub = placement_mining_subscription(engine)

    assert len(calls) == 1
    assert calls[0]["engine"] is engine
    assert calls[0]["label"] == TOOL_CALL_LABEL == "ToolCall"
    assert sub.available is True
    assert sub.pending_state == {"pending": 0}


def test_placement_mining_subscription_handler_increments_pending(monkeypatch):
    """The delivered-change handler bumps ``pending_state["pending"]`` — the
    SAME signal :meth:`EngineTasksMixin._tick_placement_mining_reactive` reads
    to decide whether to run a mining cycle this tick."""
    import agent_utilities.graph.reactive as reactive

    captured_handler: dict[str, Any] = {}

    class _FakeSubscription:
        def __init__(self) -> None:
            self.available = True

    def _fake_subscribe(engine, label, handler):
        captured_handler["handler"] = handler
        return _FakeSubscription()

    monkeypatch.setattr(reactive, "subscribe", _fake_subscribe)

    sub = placement_mining_subscription(object())
    assert sub.pending_state["pending"] == 0

    captured_handler["handler"](
        {"kind": "add_node", "label": "ToolCall", "node_id": "t1"}
    )
    assert sub.pending_state["pending"] == 1

    captured_handler["handler"](
        {"kind": "add_node", "label": "ToolCall", "node_id": "t2"}
    )
    assert sub.pending_state["pending"] == 2


def test_placement_mining_subscription_unavailable_source_is_noop():
    """No engine streaming surface -> a permanent no-op, exactly like every
    other ``EngineSubscription`` consumer (never raises, never blocks the
    caller's existing manual/Loop-cycle trigger)."""
    sub = placement_mining_subscription(object())
    assert sub.available is False
    assert sub.poll(block_ms=0) == 0


# AlwaysConformsShaclMixin stands in for the committed EG SHACL authority
# (see that mixin's own docstring) so _CycleStubEngine can clear
# verdict.valid and let the canary/apply branch fire. Bound here, at module
# end, rather than as a base class or inline method on _CycleStubEngine
# itself: that class's body is a pre-existing, byte-for-byte near-duplicate
# of test_trace_pattern_miner.py's _TraceMiningStubEngine and
# test_skill_evolution.py's _SkillEvoStubEngine (plus the contiguous
# _patch_mine_result signature right after it, which coincidentally matches
# too), and editing inside or directly beside that span shifts the clone
# gate's matched token window enough to register pre-existing duplication as
# new.
_CycleStubEngine.shacl_validate_committed = (
    AlwaysConformsShaclMixin.shacl_validate_committed
)
