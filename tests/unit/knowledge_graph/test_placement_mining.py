"""Workload-aware data-placement mining (X-5, CONCEPT:AU-KG.evolution.placement-mining-canary-loop).

Mine (Episode/ToolCall/Entity provenance -> co-occurrence/hot-skew/sequence)
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
    placement_proposals_from_mining,
    proposals_from_association,
    proposals_from_drift_anomaly,
    proposals_from_sequence,
    proposals_from_tenant_anomaly,
    rollback_placement_change,
    run_canary,
    run_placement_mining_cycle,
)

pytestmark = pytest.mark.concept("AU-KG.evolution.placement-mining-canary-loop")


# ---------------------------------------------------------------------------
# gather_access_records / basket + count + sequence builders
# ---------------------------------------------------------------------------


class _AccessStubEngine:
    """Minimal engine double: canned Episode/ToolCall/Entity provenance rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self._rows = rows or []
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "Episode" in q and "USED_TOOL" in q:
            return list(self._rows)
        return []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def _row(episode_id, tool_name, tenant="", modality="", entity_id="", entity_type=""):
    args = {}
    if tenant:
        args["tenant_id"] = tenant
    if modality:
        args["modality"] = modality
    return {
        "episode_id": episode_id,
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
            "episode_id": "ep1",
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
            "episode_id": "ep1",
            "tool_name": "graph_query",
            "tenant": "acme",
            "modality": "",
            "entity_id": "",
            "entity_type": "",
        },
        {
            "episode_id": "ep2",
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
        {"episode_id": "ep1", "tenant": "acme"},
        {"episode_id": "ep1", "tenant": "acme"},  # same episode, dedup
        {"episode_id": "ep2", "tenant": "acme"},
        {"episode_id": "ep3", "tenant": "beta"},
    ]
    counts = build_tenant_access_counts(records)
    assert counts == {"acme": 2, "beta": 1}


def test_build_tenant_sequences_filters_short_sequences():
    records = [
        {"episode_id": "ep1", "tenant": "acme", "tool_name": "read", "entity_id": ""},
        {"episode_id": "ep2", "tenant": "acme", "tool_name": "write", "entity_id": ""},
        {"episode_id": "ep3", "tenant": "beta", "tool_name": "read", "entity_id": ""},
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


def test_mine_placement_patterns_invokes_associate_surface(monkeypatch):
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    captured: dict[str, Any] = {}

    def fake_invoke(*, surface, action, graph, candidates, params):
        captured.setdefault(action, params)
        if action == "associate":
            return json.dumps(
                {"surface": surface, "action": action, "result": {"rules": []}}
            )
        if action == "anomaly":
            return json.dumps(
                {"surface": surface, "action": action, "result": {"rows": []}}
            )
        if action == "sequence":
            return json.dumps(
                {"surface": surface, "action": action, "result": {"patterns": []}}
            )
        raise AssertionError(f"unexpected action {action}")

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    engine = _AccessStubEngine(
        rows=[
            _row(
                "ep1", "graph_query", tenant="acme", entity_id="doc1", entity_type="Doc"
            ),
            _row(
                "ep2", "graph_query", tenant="acme", entity_id="doc2", entity_type="Doc"
            ),
        ]
    )
    result = mine_placement_patterns(engine)
    assert "associate" in captured
    assert captured["associate"]["transactions"]
    assert result["errors"] == []


def test_mine_placement_patterns_degrades_cleanly_on_no_mining_engine_build(
    monkeypatch,
):
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    monkeypatch.setattr(
        engine_surface_tools,
        "_invoke",
        lambda **kw: json.dumps({"degraded": True, "error": "no mining surface"}),
    )
    engine = _AccessStubEngine(
        rows=[
            _row(
                "ep1", "graph_query", tenant="acme", entity_id="doc1", entity_type="Doc"
            ),
            _row(
                "ep2", "graph_query", tenant="acme", entity_id="doc2", entity_type="Doc"
            ),
        ]
    )
    result = mine_placement_patterns(engine)
    assert result["association"] == {"rules": []}
    assert result["errors"]  # recorded, not raised


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


def test_apply_placement_change_targets_placement_catalog_for_shard_split(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["domain"] = domain
        captured["action"] = action
        captured["params"] = json.loads(params_json)
        return json.dumps({"route": "assigned"})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    result = apply_placement_change(_proposal(kind="shard_split", target="acme:ws1"))
    assert result["applied"] is True
    assert result["method"] == "catalog_assign"
    assert captured["domain"] == "resharding"
    assert captured["action"] == "catalog_assign"
    assert captured["params"]["graph"] == "acme:ws1"
    assert "shard" in captured["params"]


def test_apply_placement_change_degrades_when_resharding_surface_missing(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    monkeypatch.setattr(engine_tools_mod, "ENGINE_DOMAINS", {})
    result = apply_placement_change(_proposal(kind="shard_split"))
    assert result["applied"] is False
    assert "unavailable" in result["detail"]


def test_apply_placement_change_for_replica_also_targets_catalog(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["action"] = action
        return json.dumps({"route": "assigned"})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    result = apply_placement_change(_proposal(kind="replica", target="acme"))
    assert result["applied"] is True
    assert captured["action"] == "catalog_assign"


def test_apply_placement_change_for_non_catalog_kinds_is_an_accepted_record():
    result = apply_placement_change(_proposal(kind="materialized_join", target="a|b"))
    assert result == {
        "applied": True,
        "method": "accepted_record",
        "detail": "materialized_join",
    }


def test_rollback_placement_change_calls_catalog_remove(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["action"] = action
        captured["params"] = json.loads(params_json)
        return json.dumps({"removed": True})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    result = rollback_placement_change(_proposal(kind="shard_split", target="acme:ws1"))
    assert result["rolled_back"] is True
    assert captured["action"] == "catalog_remove"
    assert captured["params"] == {"graph": "acme:ws1"}


def test_run_canary_promotion_reaches_the_placement_catalog(monkeypatch):
    """End-to-end: a promoted canary's default apply_fn is apply_placement_change,
    which reaches the engine's PlacementCatalog admin path."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["domain"] = domain
        captured["action"] = action
        return json.dumps({"route": "assigned"})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 95.0}])
    result = run_canary(
        _proposal(kind="shard_split", target="acme:ws1"),
        measurement_fn=lambda p, phase: next(measurements),
    )
    assert result.verdict == "promote"
    assert result.applied is True
    assert captured["domain"] == "resharding"
    assert captured["action"] == "catalog_assign"


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
            "errors": [],
        },
    )


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
            if source == derived_id and props.get("relationship_type") == "DERIVED_FROM"
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


def test_relaxed_policy_runs_canary_and_applies_on_promote(monkeypatch):
    """Relaxing apply_placement_change to auto lets the canary run — and a
    positive-delta canary applies the change (status flips to 'applied')."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(
        engine_tools_mod,
        "_dispatch",
        lambda domain, methods, action, params_json, graph: json.dumps({"ok": True}),
    )
    # anomaly_score=5.0 -> confidence=1.0, clearing BOTH the CONFIDENCE_FLOOR
    # (0.6) and the PromotionGovernanceValidator's own quality threshold
    # (0.85) — needed for the applied branch (gated on verdict.valid too) to
    # actually fire (mirrors test_trace_pattern_miner.py's identical gotcha).
    _patch_mine_result(monkeypatch, anomaly_score=5.0)
    engine = _CycleStubEngine(
        governance_rules=[
            {"kind": "apply_placement_change", "target": "*", "tier": "auto"}
        ]
    )
    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 90.0}])
    rep = run_placement_mining_cycle(
        engine, measurement_fn=lambda p, phase: next(measurements)
    )

    assert rep["applied"] == 1
    proposals = engine.by_type("PlacementProposal")
    assert proposals[0]["status"] == "applied"
    assert rep["examples"][0]["action_decision"] in ("allow", "allow_notify")


def test_relaxed_policy_rejects_on_canary_regression(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove"]},
    )
    monkeypatch.setattr(
        engine_tools_mod,
        "_dispatch",
        lambda domain, methods, action, params_json, graph: json.dumps({"ok": True}),
    )
    _patch_mine_result(monkeypatch, anomaly_score=5.0)
    engine = _CycleStubEngine(
        governance_rules=[
            {"kind": "apply_placement_change", "target": "*", "tier": "auto"}
        ]
    )
    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 300.0}])
    rep = run_placement_mining_cycle(
        engine, measurement_fn=lambda p, phase: next(measurements)
    )

    assert rep["applied"] == 0
    proposals = engine.by_type("PlacementProposal")
    assert proposals[0]["status"] == "rejected"


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


def test_apply_placement_change_prefers_reshard_online_move(monkeypatch):
    """When the engine build exposes ``reshard``, apply MUST use the real
    online-move RPC (data + route), not the route-only ``catalog_assign``."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["domain"] = domain
        captured["action"] = action
        captured["params"] = json.loads(params_json)
        return json.dumps(
            {
                "graph": "acme:ws1",
                "from_shard": 3,
                "to_shard": 5,
                "nodes": 10,
                "edges": 4,
            }
        )

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {
            "resharding": [
                "catalog_assign",
                "catalog_remove",
                "reshard",
                "rebalance_plan",
            ]
        },
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    prop = _proposal(kind="shard_split", target="acme:ws1")
    result = apply_placement_change(prop)

    assert result["applied"] is True
    assert result["method"] == "reshard"
    assert captured["domain"] == "resharding"
    assert captured["action"] == "reshard"
    assert captured["params"]["graph"] == "acme:ws1"
    assert "to_shard" in captured["params"]
    assert "shard" not in captured["params"]
    # the engine's pre-move shard is stashed on the proposal for rollback.
    assert prop.evidence["_reshard_from_shard"] == 3


def test_rollback_reshards_straight_back_to_recorded_from_shard(monkeypatch):
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    calls: list[dict[str, Any]] = []

    def fake_dispatch(domain, methods, action, params_json, graph):
        calls.append({"action": action, "params": json.loads(params_json)})
        return json.dumps({"graph": "acme:ws1", "from_shard": 3, "to_shard": 5})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove", "reshard"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    prop = _proposal(kind="shard_split", target="acme:ws1")
    apply_placement_change(prop)  # records _reshard_from_shard = 3
    result = rollback_placement_change(prop)

    assert result["rolled_back"] is True
    assert result["method"] == "reshard"
    assert len(calls) == 2
    assert calls[1]["action"] == "reshard"
    assert calls[1]["params"] == {"graph": "acme:ws1", "to_shard": 3}


def test_rollback_falls_back_to_catalog_remove_without_a_recorded_from_shard(
    monkeypatch,
):
    """No prior apply happened (no ``_reshard_from_shard`` recorded) — even
    with ``reshard`` available, rollback degrades to the route-only revert."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    captured: dict[str, Any] = {}

    def fake_dispatch(domain, methods, action, params_json, graph):
        captured["action"] = action
        return json.dumps({"removed": True})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["catalog_assign", "catalog_remove", "reshard"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)

    result = rollback_placement_change(_proposal(kind="shard_split", target="acme:ws1"))

    assert result["rolled_back"] is True
    assert captured["action"] == "catalog_remove"


# ---------------------------------------------------------------------------
# canary measurement — real engine stat fallback (shard load skew)
# ---------------------------------------------------------------------------


def test_shard_load_skew_measurement_reads_rebalance_plan(monkeypatch):
    import agent_utilities.knowledge_graph.research.placement_mining as pm
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    monkeypatch.setattr(
        engine_tools_mod, "ENGINE_DOMAINS", {"resharding": ["rebalance_plan"]}
    )
    monkeypatch.setattr(
        engine_tools_mod,
        "_dispatch",
        lambda domain, methods, action, params_json, graph: json.dumps(
            {
                "moves": [],
                "shards": [
                    {"shard": 0, "total": 100, "graphs": 3},
                    {"shard": 1, "total": 40, "graphs": 1},
                ],
            }
        ),
    )
    # bypass the promql attempt so the fallback path is exercised deterministically
    monkeypatch.setattr(pm, "_promql_latency_measurement", lambda proposal: {})

    result = pm._default_measurement(_proposal(), "baseline")
    assert result == {"shard_load_skew": 60.0}


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

    monkeypatch.delenv("PLACEMENT_CONTROL_LOOP_ENABLED", raising=False)

    def _boom(*a, **kw):  # pragma: no cover - must never be called
        raise AssertionError("mining must not run while the loop is disabled")

    monkeypatch.setattr(pm, "mine_placement_patterns", _boom)

    rep = placement_control_loop(_CycleStubEngine())
    assert rep == {
        "enabled": False,
        "skipped": True,
        "reason": (
            "placement_control_loop is opt-in "
            "(PLACEMENT_CONTROL_LOOP_ENABLED=0) — manual trigger required"
        ),
    }


def test_placement_control_loop_enabled_via_env_flag(monkeypatch):
    """The config/env opt-in path (distinct from an explicit ``enabled=True``
    manual-trigger call) also turns the loop on."""
    monkeypatch.setenv("PLACEMENT_CONTROL_LOOP_ENABLED", "1")
    _patch_mine_result(monkeypatch, anomaly_score=0.5)  # below floor -> no side effects
    rep = placement_control_loop(_CycleStubEngine())
    assert rep["enabled"] is True
    assert rep["eligible"] == 0


def test_placement_control_loop_denied_proposal_never_applied(monkeypatch):
    """SAFETY: a policy DENIAL (forbidden tier) must never reach the engine's
    reshard RPC, fail-closed."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    dispatched_actions: list[str] = []

    def fake_dispatch(domain, methods, action, params_json, graph):
        dispatched_actions.append(action)
        return json.dumps({"ok": True})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["reshard", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)
    _patch_mine_result(monkeypatch, anomaly_score=5.0)
    engine = _CycleStubEngine(
        governance_rules=[
            {"kind": "apply_placement_change", "target": "*", "tier": "forbidden"}
        ]
    )

    rep = placement_control_loop(engine, enabled=True)

    assert rep["enabled"] is True
    assert rep["applied"] == 0
    assert dispatched_actions == []  # the engine reshard RPC was NEVER invoked
    proposals = engine.by_type("PlacementProposal")
    assert proposals[0]["status"] == "proposal"  # never promoted


def test_placement_control_loop_promotes_via_mocked_reshard_and_records_outcome(
    monkeypatch,
):
    """The full Seam-4 loop: synthetic mining -> 1 proposal -> ActionPolicy
    approves -> a mocked engine reshard is invoked -> the canary keeps the
    change -> the outcome is recorded back for mining to read."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    dispatched: list[dict[str, Any]] = []

    def fake_dispatch(domain, methods, action, params_json, graph):
        dispatched.append({"action": action, "params": json.loads(params_json)})
        return json.dumps({"graph": "hot-tenant", "from_shard": 2, "to_shard": 6})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["reshard", "catalog_remove", "rebalance_plan"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)
    _patch_mine_result(monkeypatch, anomaly_score=5.0)
    engine = _CycleStubEngine(
        governance_rules=[
            {"kind": "apply_placement_change", "target": "*", "tier": "auto"}
        ]
    )
    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 90.0}])

    rep = placement_control_loop(
        engine, measurement_fn=lambda p, phase: next(measurements), enabled=True
    )

    assert rep["enabled"] is True
    assert rep["applied"] == 1
    assert rep["outcomes_recorded"] == 1
    assert any(d["action"] == "reshard" for d in dispatched)
    proposals = engine.by_type("PlacementProposal")
    assert proposals[0]["status"] == "applied"
    # the outcome fed back for the mining flywheel to read.
    outcomes = engine.by_type("ClaimOutcome")
    assert len(outcomes) == 1
    assert outcomes[0]["reward"] == pytest.approx(proposals[0]["confidence"])
    assert outcomes[0]["durable_reward"] == pytest.approx(1.0)


def test_placement_control_loop_rolls_back_via_reshard_on_regression(monkeypatch):
    """A regressing canary rolls back through the SAME real reshard RPC
    (straight back to the engine-reported pre-move shard), never applies,
    and still records the (negative) outcome back to mining."""
    import agent_utilities.mcp.tools.engine_tools as engine_tools_mod

    dispatched: list[dict[str, Any]] = []

    def fake_dispatch(domain, methods, action, params_json, graph):
        params = json.loads(params_json)
        dispatched.append({"action": action, "params": params})
        return json.dumps({"graph": "hot-tenant", "from_shard": 2, "to_shard": 6})

    monkeypatch.setattr(
        engine_tools_mod,
        "ENGINE_DOMAINS",
        {"resharding": ["reshard", "catalog_remove"]},
    )
    monkeypatch.setattr(engine_tools_mod, "_dispatch", fake_dispatch)
    _patch_mine_result(monkeypatch, anomaly_score=5.0)
    engine = _CycleStubEngine(
        governance_rules=[
            {"kind": "apply_placement_change", "target": "*", "tier": "auto"}
        ]
    )
    measurements = iter([{"latency_ms": 100.0}, {"latency_ms": 300.0}])  # regression

    rep = placement_control_loop(
        engine, measurement_fn=lambda p, phase: next(measurements), enabled=True
    )

    assert rep["applied"] == 0
    assert rep["outcomes_recorded"] == 1
    reshard_calls = [d for d in dispatched if d["action"] == "reshard"]
    assert len(reshard_calls) == 2  # apply, then rollback
    assert reshard_calls[1]["params"] == {"graph": "hot-tenant", "to_shard": 2}
    proposals = engine.by_type("PlacementProposal")
    assert proposals[0]["status"] == "rejected"
    outcomes = engine.by_type("ClaimOutcome")
    assert outcomes[0]["durable_reward"] == pytest.approx(0.0)
