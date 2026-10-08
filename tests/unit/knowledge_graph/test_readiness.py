"""Proof that GraphOS readiness reports the TRUTH, not liveness (GOC-02).

The defect class under test: readiness reporting GREEN while a real graph
query cannot be answered (BUG-004's compiled sparse-index coverage of 0.0%
alongside a reachable transport). The governing rule this program recorded
eleven separate violations of: never trust a signal about state; check state.

``collect_readiness_snapshot``'s ``synthetic_query`` check runs the REAL
``build_code_context`` implementation — the same function
``graph_code(action="code_context")`` dispatches to — against a fake engine
that drives its actual Cypher-shaped call sites. Nothing in these tests mocks
``_check_synthetic_query`` or ``build_code_context`` themselves; only the
engine boundary they read through is faked, exactly like
``tests/unit/observability/test_runtime_health.py`` fakes the transport, not
the check.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime

import pytest

from agent_utilities.knowledge_graph import readiness as rd

_ONTOLOGY_ACTIVATION_MODULE = "agent_utilities.knowledge_graph.ontology.activation"


def _install_fake_activation_module(
    monkeypatch: pytest.MonkeyPatch, get_activation_status
) -> None:
    """Make ``from .ontology.activation import get_activation_status`` resolve
    inside ``_check_ontology_activation``, even though that submodule was
    permanently deleted (43197d7c6) and no longer exists on disk — Python's
    import machinery checks ``sys.modules`` before the filesystem, so
    inserting a fake module object there lets these tests exercise every
    branch of the function AS IF a replacement status source existed,
    without needing one to actually land first.
    """
    fake_module = types.ModuleType(_ONTOLOGY_ACTIVATION_MODULE)
    fake_module.get_activation_status = get_activation_status
    monkeypatch.setitem(sys.modules, _ONTOLOGY_ACTIVATION_MODULE, fake_module)


# --------------------------------------------------------------------------- #
# fakes — drive the REAL build_code_context/resolve_anchors code path
# --------------------------------------------------------------------------- #
class _FakeEngine:
    """A minimal ``query_cypher``-shaped engine.

    ``anchor_rows`` are returned for the anchor-resolution queries
    (``WHERE c.id = $id`` / ``WHERE c.name = $tok``); every other query
    returns an empty result, exactly like a real engine answering "nothing
    matches" for an unenriched section (call graph, similar-code, routes,
    coupling, docs) — never raises for those.
    """

    def __init__(
        self,
        *,
        anchor_rows: list[dict] | None = None,
        raise_exc: Exception | None = None,
    ):
        self._anchor_rows = anchor_rows or []
        self._raise_exc = raise_exc

    def query_cypher(self, cypher: str, params: dict):
        is_anchor_query = "c.id = $id" in cypher or "c.name = $tok" in cypher
        if is_anchor_query and self._raise_exc is not None:
            raise self._raise_exc
        if is_anchor_query:
            return list(self._anchor_rows)
        return []


_ANCHOR_ROW = {
    "id": "code:collect_readiness_snapshot",
    "name": "collect_readiness_snapshot",
    "file_path": "agent_utilities/knowledge_graph/readiness.py",
    "line": 42,
    "language": "python",
    "kind": "function",
    "instance": "",
    "source_system": "agent-utilities",
}


def _healthy_report(engine_ok: bool = True) -> dict:
    return {
        "status": "healthy" if engine_ok else "unhealthy",
        "generated_at": "2026-08-16T00:00:00+00:00",
        "checks": [
            {
                "name": "engine",
                "status": "ok" if engine_ok else "unhealthy",
                "reason": None if engine_ok else "no engine endpoint reachable",
                "detail": {
                    "resolved_mode": "shared",
                    "reachable_count": 1 if engine_ok else 0,
                },
                "latency_ms": 1.2,
            },
            {
                "name": "embedding_endpoint",
                "status": "ok",
                "detail": {"active_model": "bge-m3"},
                "latency_ms": 0.4,
            },
        ],
    }


# --------------------------------------------------------------------------- #
# KNOWN-BAD PROOF: engine unreachable during the synthetic query -> NOT ready
# --------------------------------------------------------------------------- #
def test_synthetic_query_reports_unavailable_when_engine_degraded(
    monkeypatch: pytest.MonkeyPatch,
):
    """The exact class the audit flagged: a transport-down engine must make
    readiness UNAVAILABLE, never a false 'ready' because the engine handle
    itself was non-None."""
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(raise_exc=ConnectionError("engine transport down"))

    snapshot = rd.collect_readiness_snapshot(engine, deadline_s=2.0)

    synth = snapshot["checks"]["synthetic_query"]
    assert synth["state"] == "unavailable"
    assert synth["reason"] == "engine_degraded"
    assert synth["evidence_count"] == 0
    assert snapshot["overall"] == "unavailable"
    assert "synthetic_query" in snapshot["required_failures"]
    assert rd.is_snapshot_ready(snapshot) is False


def test_readiness_unavailable_when_no_engine_supplied(monkeypatch: pytest.MonkeyPatch):
    """No engine handle at all must never be silently skipped into 'ready'."""
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )

    snapshot = rd.collect_readiness_snapshot(None, deadline_s=1.0)

    assert snapshot["checks"]["synthetic_query"]["state"] == "unavailable"
    assert snapshot["checks"]["synthetic_query"]["reason"] == "no_engine_supplied"
    assert snapshot["overall"] == "unavailable"
    assert rd.is_snapshot_ready(snapshot) is False


def test_synthetic_query_zero_evidence_is_degraded_not_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    """A REAL, non-degraded answer that grounds on nothing is DEGRADED with
    reason evidence_coverage_zero — regardless of index coverage — and
    must never read as a successful grounding."""
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(
        anchor_rows=[]
    )  # engine answers fine; genuinely nothing matches

    snapshot = rd.collect_readiness_snapshot(
        engine, synthetic_query="no_such_symbol_xyz", deadline_s=2.0
    )

    synth = snapshot["checks"]["synthetic_query"]
    assert synth["state"] == "degraded"
    assert synth["reason"] == "evidence_coverage_zero"
    assert synth["evidence_count"] == 0
    # An empty query result does not establish index-wide coverage.
    assert snapshot["checks"]["sparse_index"]["state"] == "not_configured"
    assert "coverage_pct" not in snapshot["checks"]["sparse_index"]
    # A required check (synthetic_query) is only DEGRADED here, not UNAVAILABLE,
    # so overall degrades but is not the harder failure state.
    assert snapshot["overall"] == "degraded"
    assert rd.is_snapshot_ready(snapshot) is False


# --------------------------------------------------------------------------- #
# KNOWN-GOOD PROOF: a real grounded answer -> ready, only when genuinely so
# --------------------------------------------------------------------------- #
def _healthy_engine_with_evidence(monkeypatch: pytest.MonkeyPatch):
    """A healthy health report + zero expected connectors + one real anchor
    row. Shared setup for test_synthetic_query_ready_when_real_evidence_found
    and test_ontology_activation_failure_makes_whole_snapshot_not_ready."""
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: [],
    )
    return _FakeEngine(anchor_rows=[_ANCHOR_ROW])


def test_synthetic_query_ready_when_real_evidence_found(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overall is ``"degraded"``, not ``"ready"``: ``ontology_activation`` is
    a separate, non-required check that fails closed to ``"unavailable"`` --
    in plain words, there is currently no replacement status source backed
    by the epistemic-graph-owned ontology authority for this leg to read
    (see ``readiness._check_ontology_activation``'s own comment), so overall
    ``"ready"`` is not reachable yet no matter what this test stubs. This
    test still proves its actual target: the synthetic-query canary itself
    reads real grounded evidence as ``"ready"``.
    """
    engine = _healthy_engine_with_evidence(monkeypatch)

    snapshot = rd.collect_readiness_snapshot(
        engine,
        synthetic_query="collect_readiness_snapshot",
        required_routes=rd.DEFAULT_REQUIRED_ROUTES,
        connector_freshness={},
        deadline_s=2.0,
    )

    synth = snapshot["checks"]["synthetic_query"]
    assert synth["state"] == "ready"
    assert synth["evidence_count"] >= 1
    assert synth["route"] == "graph_code_context"
    assert snapshot["checks"]["sparse_index"]["state"] == "not_configured"
    assert snapshot["checks"]["ontology_activation"]["state"] == "unavailable"
    assert snapshot["overall"] == "degraded"
    assert rd.is_snapshot_ready(snapshot) is False


def test_full_snapshot_ready_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """Every REQUIRED check green -> overall is the best currently-reachable
    state, "degraded" -- not "ready". ``ontology_activation`` (non-required,
    see ``REQUIRED_CHECKS``) fails closed to ``"unavailable"`` because its
    backing module was permanently deleted (43197d7c6) with no replacement
    epistemic-graph-backed status source yet; there is no way to stub it
    back to "ready" (the module it would read does not exist to patch). In
    plain words: overall "ready" is NOT reachable until that replacement
    exists. This test proves every OTHER check reaches its best state.
    """
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: ["leanix"],
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])

    class _Actor:
        authenticated = True

    class _Session:
        actor = _Actor()
        tenant = "tenant-a"
        policy_version = 7

    snapshot = rd.collect_readiness_snapshot(
        engine,
        session=_Session(),
        synthetic_query="collect_readiness_snapshot",
        # Within connector_coverage.DEFAULT_SLA_DAYS (7) of "now" — a fixed
        # past date goes stale as real time passes it (this field was
        # "2026-08-16", which cleared the SLA when this test was written but
        # does not today), making source_sync "degraded" for a reason
        # unrelated to what this test proves.
        connector_freshness={"leanix": datetime.now(UTC).isoformat()},
        deadline_s=2.0,
    )

    assert snapshot["overall"] == "degraded"
    assert snapshot["required_failures"] == []
    assert snapshot["checks"]["identity_policy"]["state"] == "ready"
    assert snapshot["checks"]["identity_policy"]["carrier"] == "verified"
    assert snapshot["checks"]["source_sync"]["state"] == "ready"
    assert snapshot["checks"]["ontology_activation"]["state"] == "unavailable"
    assert rd.is_snapshot_ready(snapshot) is False
    # Serializes cleanly with no live objects left behind.
    json.dumps(snapshot)


# --------------------------------------------------------------------------- #
# catalog — required_module_missing must fail startup readiness (acceptance
# gate 2 / BUG-013's certification-never-skips-required-modules invariant).
# --------------------------------------------------------------------------- #
def test_catalog_required_module_missing_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    bad_routes = {
        "graph_code_context": "agent_utilities.knowledge_graph.retrieval.code_context:build_code_context",
        "totally_missing_route": "agent_utilities.nonexistent_module:nonexistent_attr",
    }

    snapshot = rd.collect_readiness_snapshot(
        engine, required_routes=bad_routes, deadline_s=2.0
    )

    catalog = snapshot["checks"]["catalog"]
    assert catalog["state"] == "unavailable"
    assert catalog["reason"] == "required_module_missing"
    assert "totally_missing_route" in catalog["missing"]
    assert snapshot["overall"] == "unavailable"
    assert "catalog" in snapshot["required_failures"]
    assert rd.is_snapshot_ready(snapshot) is False


def test_catalog_ready_when_every_required_route_resolves():
    result = rd._check_catalog(dict(rd.DEFAULT_REQUIRED_ROUTES))
    assert result["state"] == "ready"
    assert result["missing"] == []
    assert result["digest"].startswith("sha256:")


# --------------------------------------------------------------------------- #
# engine check — reuses runtime_health, never a second authority
# --------------------------------------------------------------------------- #
def test_engine_check_reflects_unreachable_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=False)
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])

    snapshot = rd.collect_readiness_snapshot(engine, deadline_s=2.0)

    assert snapshot["checks"]["engine"]["state"] == "unavailable"
    assert snapshot["overall"] == "unavailable"
    assert "engine" in snapshot["required_failures"]


def test_engine_check_ready_reuses_health_report_verbatim():
    result = rd._check_engine(_healthy_report(engine_ok=True))
    assert result["state"] == "ready"
    assert result["version"] == "shared"


def test_readiness_unavailable_when_health_collector_itself_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    """A broken health collector must report unavailable, never fall back to ready.

    ``_collect_health_report`` itself already converts a raised exception into
    ``None`` (see its own try/except) — this proves the DOWNSTREAM checks that
    consume that ``None`` degrade honestly rather than assuming health.
    """
    monkeypatch.setattr(rd, "_collect_health_report", lambda: None)
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    snapshot = rd.collect_readiness_snapshot(engine, deadline_s=2.0)
    assert snapshot["checks"]["engine"]["state"] == "unavailable"
    assert snapshot["checks"]["dense_index"]["state"] == "unavailable"
    assert snapshot["overall"] == "unavailable"


# --------------------------------------------------------------------------- #
# identity_policy — never infers authority from a bare tenant/subject string
# --------------------------------------------------------------------------- #
def test_identity_policy_not_configured_without_session_or_principal():
    result = rd._check_identity_policy(
        session=None, subject="", tenant="", policy_epoch=0
    )
    assert result["state"] == "not_configured"


def test_identity_policy_degraded_when_principal_supplied_without_session():
    result = rd._check_identity_policy(
        session=None, subject="alice", tenant="tenant-a", policy_epoch=1
    )
    assert result["state"] == "degraded"
    assert result["carrier"] == "unverified"


def test_identity_policy_degraded_when_session_actor_unauthenticated():
    class _Actor:
        authenticated = False

    class _Session:
        actor = _Actor()
        tenant = "tenant-a"
        policy_version = 1

    result = rd._check_identity_policy(
        session=_Session(), subject="", tenant="", policy_epoch=0
    )
    assert result["state"] == "degraded"
    assert result["carrier"] == "unverified"


# --------------------------------------------------------------------------- #
# source_sync
# --------------------------------------------------------------------------- #
def test_source_sync_not_configured_when_nothing_expected(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: [],
    )
    result = rd._check_source_sync({})
    assert result["state"] == "not_configured"


def test_source_sync_stale_when_freshness_not_supplied(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: ["leanix"],
    )
    result = rd._check_source_sync(None)
    assert result["state"] == "stale"
    assert result["reason"] == "connector_freshness_not_supplied"


def test_source_sync_unavailable_when_zero_coverage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: ["leanix"],
    )
    result = rd._check_source_sync({"unrelated_source": "2026-01-01T00:00:00Z"})
    assert result["state"] == "unavailable"
    assert result["reason"] == "no_connector_coverage"


def test_source_sync_degraded_on_partial_coverage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: ["leanix", "servicenow"],
    )
    result = rd._check_source_sync({"leanix": "2026-08-15T00:00:00Z"})
    assert result["state"] == "degraded"
    assert "servicenow" in result["missing"]


# --------------------------------------------------------------------------- #
# ontology_activation (CONCEPT:AU-KG.ontology.integrity-bootstrap, NE-152) —
# fails closed: a graph never attempted, or that gave up, must never read
# "ready" just because a socket/engine handle exists. The backing module
# (agent_utilities/knowledge_graph/ontology/activation.py) was permanently
# deleted by 43197d7c6 with no replacement epistemic-graph-backed status
# source yet, so in THIS codebase today every real call always takes the
# ImportError branch -- these tests exercise every OTHER branch directly
# against the function (bypassing the import) so that contract stays
# covered and does not silently rot once a replacement does exist.
# --------------------------------------------------------------------------- #
def test_ontology_activation_unavailable_when_no_engine_supplied():
    result = rd._check_ontology_activation(None, "")
    assert result["state"] == "unavailable"
    assert result["reason"] == "no_engine_supplied"


def test_ontology_activation_unavailable_when_status_source_removed():
    """The real, current behavior in THIS codebase: the backing module is
    gone, so every call with a real engine takes the ImportError branch and
    fails closed instead of raising out of a readiness snapshot."""
    result = rd._check_ontology_activation(object(), "acme")
    assert result["state"] == "unavailable"
    assert result["reason"] == "ontology_activation_status_source_removed"


def test_ontology_activation_unavailable_when_never_attempted(
    monkeypatch: pytest.MonkeyPatch,
):
    """Once a replacement status source exists: a graph that was never
    attempted must still read unavailable, not ready."""
    _install_fake_activation_module(monkeypatch, lambda graph_name: None)
    result = rd._check_ontology_activation(object(), "")
    assert result["state"] == "unavailable"
    assert result["reason"] == "ontology_activation_not_attempted"


def test_ontology_activation_ready_when_recorded_ready(monkeypatch: pytest.MonkeyPatch):
    _install_fake_activation_module(
        monkeypatch,
        lambda graph_name: {
            "state": "ready",
            "reason": None,
            "detail": {"attempts": 1},
        },
    )
    result = rd._check_ontology_activation(object(), "acme")
    assert result["state"] == "ready"


def test_ontology_activation_unavailable_when_recorded_failed(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_activation_module(
        monkeypatch,
        lambda graph_name: {
            "state": "unavailable",
            "reason": "activation_timeout",
            "detail": {},
        },
    )
    result = rd._check_ontology_activation(object(), "acme")
    assert result["state"] == "unavailable"
    assert result["reason"] == "activation_timeout"


def test_ontology_activation_failure_makes_whole_snapshot_not_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end proof: every OTHER check green, but ontology activation
    never attempted -> overall must NOT read ready."""
    engine = _healthy_engine_with_evidence(monkeypatch)

    snapshot = rd.collect_readiness_snapshot(
        engine,
        synthetic_query="collect_readiness_snapshot",
        connector_freshness={},
        deadline_s=2.0,
    )

    assert snapshot["checks"]["ontology_activation"]["state"] == "unavailable"
    assert snapshot["overall"] != "ready"
    assert rd.is_snapshot_ready(snapshot) is False


# --------------------------------------------------------------------------- #
# schema + payload hygiene
# --------------------------------------------------------------------------- #
def test_snapshot_schema_shape(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    snapshot = rd.collect_readiness_snapshot(engine, deadline_s=2.0)

    assert snapshot["schema_version"] == "graphos.readiness.v1"
    assert snapshot["snapshot_id"].startswith("sha256:")
    assert set(snapshot["checks"]) == {
        "engine",
        "identity_policy",
        "catalog",
        "source_sync",
        "ontology_activation",
        "synthetic_query",
        "dense_index",
        "sparse_index",
    }
    assert isinstance(snapshot["required_failures"], list)
    assert isinstance(snapshot["degraded_reasons"], list)
    assert snapshot["refresh"]["mode"] in {"delta", "not_probed"}
    # observed_at is a real ISO timestamp
    assert "T" in snapshot["observed_at"]


def test_snapshot_never_carries_the_raw_query_text_or_answer_prose(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    secret_looking_query = "SELECT password FROM secrets WHERE token='abc123'"

    snapshot = rd.collect_readiness_snapshot(
        engine, synthetic_query=secret_looking_query, deadline_s=2.0
    )

    serialized = json.dumps(snapshot)
    assert secret_looking_query not in serialized
    assert "answer" not in snapshot["checks"]["synthetic_query"]
    assert "sections" not in snapshot["checks"]["synthetic_query"]


def test_snapshot_is_json_serializable_with_a_well_formed_digest(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    s1 = rd.collect_readiness_snapshot(engine, deadline_s=2.0)
    s2 = rd.collect_readiness_snapshot(engine, deadline_s=2.0)
    json.dumps(s1)
    json.dumps(s2)
    for snapshot in (s1, s2):
        digest = snapshot["snapshot_id"]
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64
        # digest is a deterministic function of the check payload (never a
        # random uuid) — recomputing it from the returned checks must match.
        recomputed = rd.hashlib.sha256(
            rd.json.dumps(
                {
                    "checks": snapshot["checks"],
                    "overall": snapshot["overall"],
                    "required_failures": snapshot["required_failures"],
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert digest == f"sha256:{recomputed}"


# --------------------------------------------------------------------------- #
# rollup semantics — the design contract table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("states", "required_failures", "expected"),
    [
        (
            {"engine": "ready", "catalog": "ready", "synthetic_query": "ready"},
            [],
            "ready",
        ),
        ({"engine": "unavailable"}, ["engine"], "unavailable"),
        ({"engine": "ready", "source_sync": "degraded"}, [], "degraded"),
        ({"engine": "ready", "dense_index": "stale"}, [], "stale"),
        ({"engine": "ready", "kg_mirrors": "not_configured"}, [], "ready"),
    ],
)
def test_rollup_matrix(states, required_failures, expected):
    checks = {name: {"state": state} for name, state in states.items()}
    assert rd._rollup(checks, required_failures) == expected


@pytest.mark.parametrize("source", ["docs/readiness.md", None])
def test_document_only_retrieval_is_not_empty_index_or_grounded_answer(
    monkeypatch, source
):
    from agent_utilities.knowledge_graph.retrieval.code_context import (
        build_code_context,
    )

    class DocumentEngine(_FakeEngine):
        def query_cypher(self, cypher, params):
            if "d:Chunk OR d:Document" in cypher:
                return [
                    {
                        "id": "doc:readiness",
                        "snippet": "Readiness documentation",
                        "source_path": source,
                    }
                ]
            return super().query_cypher(cypher, params)

    engine = DocumentEngine()
    answer = build_code_context(engine, query="readiness", intent="how", top_k=3)
    assert answer["status"] == "ok"
    assert len(answer["sections"]["docs"]) == 1
    assert answer["citations"] == []
    monkeypatch.setattr(
        rd, "_collect_health_report", lambda: _healthy_report(engine_ok=True)
    )
    snapshot = rd.collect_readiness_snapshot(
        engine, synthetic_query="readiness", deadline_s=2
    )
    synth = snapshot["checks"]["synthetic_query"]
    assert synth["state"] == "degraded"
    assert synth["reason"] == "evidence_coverage_zero"
    assert synth["evidence_count"] == 0
    assert synth["detail"] == {
        "retrieval_status": "ok",
        "retrieved_document_count": 1,
        "documents_with_source_count": int(bool(source)),
    }
    assert snapshot["checks"]["sparse_index"]["state"] == "not_configured"
    assert "coverage_pct" not in snapshot["checks"]["sparse_index"]
    assert snapshot["overall"] == "degraded"
    assert not rd.is_snapshot_ready(snapshot)


@pytest.mark.parametrize(
    "state,count,reason",
    [
        ("degraded", 0, "evidence_coverage_zero"),
        ("ready", 2, None),
        ("unavailable", 0, "engine_degraded"),
        ("unavailable", 0, "synthetic_query_timeout"),
    ],
)
def test_sparse_coverage_is_not_inferred_from_canary(state, count, reason):
    check = rd._check_sparse_index(
        {"state": state, "evidence_count": count, "reason": reason}
    )
    assert check["state"] == "not_configured"
    assert check["reason"] == "sparse_index_coverage_unmeasured"
    assert "coverage_pct" not in check
    assert check["detail"] == {
        "source": "synthetic_query",
        "synthetic_query_state": state,
        "citation_count": count,
    }
