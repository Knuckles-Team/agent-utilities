#!/usr/bin/python
"""Tests for the analytics-job feature/model/experiment registries (CONCEPT:INT-P2-1b, L41).

Fixtures mirror the EXACT claim/evidence property shape
``epistemic-graph``'s ``crates/eg-jobs/src/claim.rs::commit_result_claim`` writes
(the engine's job-result-claim convention), so these tests double as a
contract check against the engine's committed schema.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.retrieval.analytics_job_registry import (
    AlgoVersionLineage,
    AnalyticsJobRegistry,
    is_feature_family,
)


def _claim_props(
    *,
    job_id: str,
    graph: str,
    version: int,
    family: str,
    algorithm: str,
    params_digest: str,
    code_version: str = "2.19.0",
    env_version: str = "eg-jobs-v1",
    confidence: float = 0.87,
    validation_state: str = "unvalidated",
    result_ref: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    ref = result_ref or f"result:{family}:{algorithm}:{params_digest}:{graph}:{version}"
    props = {
        "type": "Claim",
        "family": family,
        "about": ref,
        "confidence": confidence,
        "validation_state": validation_state,
        "job_id": job_id,
        "input_snapshot_graph": graph,
        "input_snapshot_version": version,
        "algo_family": family,
        "algo_algorithm": algorithm,
        "algo_params_digest": params_digest,
        "algo_code_version": code_version,
        "algo_env_version": env_version,
        "result_ref": ref,
    }
    props.update(extra)
    return props


def _evidence_props(
    *,
    job_id: str,
    result_ref: str,
    family: str,
    tenant: str = "acme",
    actor: str = "agent:planner",
    purpose: str = "quarterly-mining",
    confidence: float = 0.87,
    validation_state: str = "unvalidated",
) -> dict[str, Any]:
    return {
        "type": "Evidence",
        "family": family,
        "about": result_ref,
        "provenance": f"job:{job_id}",
        "confidence": confidence,
        "validation_state": validation_state,
        "job_id": job_id,
        "tenant": tenant,
        "actor": actor,
        "purpose": purpose,
    }


# ---------------------------------------------------------------------------
# is_feature_family
# ---------------------------------------------------------------------------


def test_is_feature_family_recognizes_feature_and_embedding_roots() -> None:
    assert is_feature_family("feature.embedding.bge_m3")
    assert is_feature_family("embedding.refresh")
    assert not is_feature_family("mining.association")
    assert not is_feature_family("")


# ---------------------------------------------------------------------------
# Basic indexing + model registry view
# ---------------------------------------------------------------------------


def test_index_claim_and_evidence_builds_a_job_result_record() -> None:
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-0000000000000001",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence = _evidence_props(
        job_id="job-0000000000000001",
        result_ref=claim["result_ref"],
        family="mining.association",
    )

    reg.index_claim(claim)
    reg.index_evidence(evidence)

    assert len(reg) == 1
    record = reg.jobs_for_result_ref(claim["result_ref"])
    assert record is not None
    assert record.job_id == "job-0000000000000001"
    assert record.lineage == AlgoVersionLineage(
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
        code_version="2.19.0",
        env_version="eg-jobs-v1",
    )
    assert record.input_snapshot_graph == "g1"
    assert record.input_snapshot_version == 5
    assert record.confidence == pytest.approx(0.87)
    assert record.tenant == "acme"
    assert record.actor == "agent:planner"
    assert record.purpose == "quarterly-mining"


def test_claim_without_job_id_is_skipped_not_a_job_result_claim() -> None:
    """The synchronous mining.rs writeback claim shares the convention but has no
    job lineage — must not pollute the job-result index."""
    reg = AnalyticsJobRegistry()
    non_job_claim = {
        "type": "Claim",
        "family": "mining.association",
        "confidence": 0.9,
        # no job_id, no algo_* fields
    }
    assert reg.index_claim(non_job_claim) is None
    assert len(reg) == 0


def test_models_view_groups_mining_association_as_a_model_artifact() -> None:
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-1",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
        approval_state="approved",
        deployment_state="deployed",
        calibration={"method": "platt", "score": 0.02},
        cost_usd=0.15,
        risk_score=0.1,
    )
    evidence = _evidence_props(
        job_id="job-1", result_ref=claim["result_ref"], family="mining.association"
    )
    reg.index_rows([claim], [evidence])

    models = reg.models()
    assert len(models) == 1
    m = models[0]
    assert m.lineage.family == "mining.association"
    assert m.version == "2.19.0:deadbeef"
    assert m.evaluation_metrics["confidence"] == pytest.approx(0.87)
    assert m.approval_state == "approved"
    assert m.deployment_state == "deployed"
    assert m.calibration == {"method": "platt", "score": 0.02}
    assert m.cost_usd == pytest.approx(0.15)
    assert m.risk_score == pytest.approx(0.1)
    assert m.source_snapshots == [("g1", 5)]

    # A feature-family lineage never shows up in the model view.
    assert reg.features() == []


# ---------------------------------------------------------------------------
# Feature registry view
# ---------------------------------------------------------------------------


def test_features_view_surfaces_embedding_metadata_when_present() -> None:
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-feat-1",
        graph="docs",
        version=3,
        family="feature.embedding.bge_m3",
        algorithm="bge-m3",
        params_digest="cafef00d",
        feature_model="bge-m3",
        feature_dimension=1024,
        feature_tokenizer="bge-m3-tok",
        drift_score=0.04,
        reembedding_schedule="weekly",
    )
    evidence = _evidence_props(
        job_id="job-feat-1",
        result_ref=claim["result_ref"],
        family="feature.embedding.bge_m3",
        purpose="doc-embedding-refresh",
    )
    reg.index_rows([claim], [evidence])

    features = reg.features()
    assert len(features) == 1
    f = features[0]
    assert f.lineage.algorithm == "bge-m3"
    assert f.model == "bge-m3"
    assert f.dimension == 1024
    assert f.tokenizer == "bge-m3-tok"
    assert f.drift_score == pytest.approx(0.04)
    assert f.reembedding_schedule == "weekly"
    assert f.source_snapshots == [("docs", 3)]
    assert f.policy_purposes == ["doc-embedding-refresh"]

    assert reg.models() == []


def test_feature_record_omits_metadata_that_was_never_stamped() -> None:
    """Nothing fabricated: a claim without the optional embedding-metadata
    properties surfaces None rather than inventing a value."""
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-feat-2",
        graph="docs",
        version=1,
        family="feature.embedding.minilm",
        algorithm="minilm",
        params_digest="abc123",
    )
    reg.index_claim(claim)

    features = reg.features()
    assert len(features) == 1
    f = features[0]
    assert f.model is None
    assert f.dimension is None
    assert f.tokenizer is None
    assert f.drift_score is None
    assert f.reembedding_schedule is None


# ---------------------------------------------------------------------------
# Grouping by AlgoVersion lineage across snapshots ("a new claim with the same
# AlgoVersion is grouped")
# ---------------------------------------------------------------------------


def test_same_lineage_across_snapshots_is_grouped_into_one_lineage() -> None:
    reg = AnalyticsJobRegistry()
    claim_v5 = _claim_props(
        job_id="job-a",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence_v5 = _evidence_props(
        job_id="job-a", result_ref=claim_v5["result_ref"], family="mining.association"
    )
    # Same algorithm+params+build, but re-run over a LATER graph version —
    # a fresh result_ref (different snapshot), not a duplicate.
    claim_v6 = _claim_props(
        job_id="job-b",
        graph="g1",
        version=6,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence_v6 = _evidence_props(
        job_id="job-b", result_ref=claim_v6["result_ref"], family="mining.association"
    )

    reg.index_rows([claim_v5, claim_v6], [evidence_v5, evidence_v6])

    assert claim_v5["result_ref"] != claim_v6["result_ref"]
    lineages = reg.lineages()
    assert len(lineages) == 1

    jobs = reg.jobs_for_lineage("mining.association", "fpgrowth", "deadbeef")
    assert [j.job_id for j in jobs] == ["job-a", "job-b"]
    assert [j.input_snapshot_version for j in jobs] == [5, 6]

    models = reg.models()
    assert len(models) == 1
    assert len(models[0].jobs) == 2


def test_jobs_for_lineage_coarse_lookup_ignores_unset_fields() -> None:
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-a",
        graph="g1",
        version=1,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
        code_version="2.19.0",
    )
    reg.index_claim(claim)

    # Omitting params_digest/code_version still finds the record — a coarser
    # "every build of this algorithm" query.
    matches = reg.jobs_for_lineage("mining.association", "fpgrowth")
    assert len(matches) == 1
    assert matches[0].job_id == "job-a"

    # A mismatched params_digest excludes it.
    assert reg.jobs_for_lineage("mining.association", "fpgrowth", "wrongdigest") == []


def test_duplicate_result_ref_reindex_updates_in_place_no_duplicate_record() -> None:
    """Mirrors the engine's own idempotent-commit contract: re-indexing the SAME
    result_ref (e.g. a refreshed read of an unchanged claim) never creates a second
    record."""
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-a",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    reg.index_claim(claim)
    reg.index_claim(claim)
    assert len(reg) == 1


# ---------------------------------------------------------------------------
# Experiment registry view
# ---------------------------------------------------------------------------


def test_jobs_for_experiment_bundles_every_job_sharing_a_purpose() -> None:
    reg = AnalyticsJobRegistry()
    claim1 = _claim_props(
        job_id="job-a",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence1 = _evidence_props(
        job_id="job-a",
        result_ref=claim1["result_ref"],
        family="mining.association",
        purpose="quarterly-mining",
    )
    # A DIFFERENT algorithm run as part of the SAME experiment/purpose.
    claim2 = _claim_props(
        job_id="job-b",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="apriori",
        params_digest="beefdead",
    )
    evidence2 = _evidence_props(
        job_id="job-b",
        result_ref=claim2["result_ref"],
        family="mining.association",
        purpose="quarterly-mining",
    )
    # An unrelated experiment must not leak in.
    claim3 = _claim_props(
        job_id="job-c",
        graph="g2",
        version=1,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="00000000",
    )
    evidence3 = _evidence_props(
        job_id="job-c",
        result_ref=claim3["result_ref"],
        family="mining.association",
        purpose="ad-hoc-probe",
    )

    reg.index_rows([claim1, claim2, claim3], [evidence1, evidence2, evidence3])

    assert reg.experiments() == ["ad-hoc-probe", "quarterly-mining"]

    run = reg.jobs_for_experiment("quarterly-mining")
    assert run is not None
    assert run.purpose == "quarterly-mining"
    assert run.tenant == "acme"
    assert {j.job_id for j in run.jobs} == {"job-a", "job-b"}
    assert len(run.lineages) == 2  # two distinct algorithms in this one run

    assert reg.jobs_for_experiment("does-not-exist") is None


def test_jobs_for_experiment_scoped_by_tenant() -> None:
    reg = AnalyticsJobRegistry()
    claim = _claim_props(
        job_id="job-a",
        graph="g1",
        version=1,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence = _evidence_props(
        job_id="job-a",
        result_ref=claim["result_ref"],
        family="mining.association",
        tenant="acme",
        purpose="quarterly-mining",
    )
    reg.index_rows([claim], [evidence])

    assert reg.jobs_for_experiment("quarterly-mining", tenant="acme") is not None
    assert reg.jobs_for_experiment("quarterly-mining", tenant="other-tenant") is None


# ---------------------------------------------------------------------------
# refresh_from_engine
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Mimics ``IntelligenceGraphEngine.query_cypher``'s row shape (``{"c": {...}}``)."""

    def __init__(
        self, claim_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]
    ) -> None:
        self._claim_rows = claim_rows
        self._evidence_rows = evidence_rows

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        ctype = (params or {}).get("ctype")
        etype = (params or {}).get("etype")
        if ctype == "Claim":
            return [{"c": row} for row in self._claim_rows]
        if etype == "Evidence":
            return [{"e": row} for row in self._evidence_rows]
        return []


class _RaisingEngine:
    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        raise RuntimeError("engine unreachable")


def test_refresh_from_engine_indexes_via_query_cypher() -> None:
    claim = _claim_props(
        job_id="job-a",
        graph="g1",
        version=5,
        family="mining.association",
        algorithm="fpgrowth",
        params_digest="deadbeef",
    )
    evidence = _evidence_props(
        job_id="job-a", result_ref=claim["result_ref"], family="mining.association"
    )
    engine = _FakeEngine([claim], [evidence])

    reg = AnalyticsJobRegistry()
    count = reg.refresh_from_engine(engine)

    assert count == 1
    record = reg.job("job-a")
    assert record is not None
    assert record.tenant == "acme"


def test_refresh_from_engine_degrades_cleanly_when_engine_unreachable() -> None:
    reg = AnalyticsJobRegistry()
    count = reg.refresh_from_engine(_RaisingEngine())
    assert count == 0
    assert len(reg) == 0
