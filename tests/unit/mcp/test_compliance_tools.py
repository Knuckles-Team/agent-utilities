"""Tests for the graph_compliance MCP tool (CONCEPT:AU-KG.enrichment.compliance-posture-rollup).

Mirrors the ``_CollectingMCP`` pattern of ``test_audit_tools.py``. ``posture``
monkeypatches ``kg_server._get_engine`` + ``audit_tools._verify`` (proving the
rollup REUSES the existing audit-ledger primitive rather than reimplementing
it); ``export`` monkeypatches ``engine_tools._client_for`` (proving bulk
export reuses the SAME ``explain_belief`` dispatch ``graph_epistemic``'s
``why`` action uses).
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import audit_tools, engine_tools
from agent_utilities.mcp.tools.compliance_tools import (
    _confidence_rollup,
    register_compliance_tools,
)


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _FakeComputeEngine:
    def __init__(
        self,
        nodes_by_label: dict[str, list[tuple[str, dict]]],
        stream_rows_by_label: dict[str, list[dict]] | None = None,
    ):
        self._nodes_by_label = nodes_by_label
        self._stream_rows_by_label = stream_rows_by_label

    def get_nodes_by_label(self, label, limit=0):
        return list(self._nodes_by_label.get(label, []))

    def stream_graph_confidence(self, label, *, batch_size=512, limit=0):
        # Only defined at all when the test opts a fake KnowledgeStream surface
        # in (mirrors GraphComputeEngine returning None when unavailable).
        if self._stream_rows_by_label is None:
            return None
        rows = self._stream_rows_by_label.get(label)
        if rows is None:
            return None
        return iter(rows)


class _FakeEngine:
    def __init__(self, graph, cypher_rows=None):
        self.graph = graph
        self._cypher_rows = cypher_rows or []

    def query_cypher(self, cypher, as_of=None):
        return list(self._cypher_rows)


def _register(monkeypatch):
    mcp = _CollectingMCP()
    register_compliance_tools(mcp)
    async_tool = mcp.tools["graph_compliance"]

    # graph_compliance is `async def` (D-50 — event-loop isolation for sync
    # MCP tool handlers with genuine blocking bodies). This suite predates
    # that change and calls the registered tool directly (bypassing
    # kg_server._execute_tool, which already awaits async tools) — wrap it so
    # every existing synchronous `tool(...)` call site keeps working unchanged.
    def _sync_tool(**kwargs):
        return asyncio.run(async_tool(**kwargs))

    return _sync_tool


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_compliance_tools(mcp)
    assert "graph_compliance" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_compliance") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_compliance") == "/compliance"


def test_posture_joins_audit_verify_and_node_counts(monkeypatch):
    tool = _register(monkeypatch)
    fake_graph = _FakeComputeEngine(
        {
            "Control": [
                ("c1", {"status": "satisfied"}),
                ("c2", {"status": "gap"}),
            ],
            "Incident": [("i1", {"status": "open"})],
        }
    )
    engine = _FakeEngine(fake_graph)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        audit_tools,
        "_verify",
        lambda: {"surface": "audit", "action": "verify", "ok": True, "entries": 4},
    )

    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["surface"] == "compliance"
    assert out["action"] == "posture"
    assert out["audit_ledger"]["ok"] is True
    assert out["node_counts"]["Control"] == 2
    assert out["node_counts"]["Incident"] == 1
    assert out["status_breakdown"]["Control"] == {"satisfied": 1, "gap": 1}


def test_posture_includes_confidence_rollup_when_knowledge_stream_available(
    monkeypatch,
):
    """CONCEPT:AU-KG.query.knowledge-stream-consumer (report §9 #3) — live path: when the compute
    engine exposes ``stream_graph_confidence`` (the ``Method::KnowledgeStream``
    consumer), ``posture`` streams a per-label confidence/contested rollup
    instead of one more full-materialize read."""
    tool = _register(monkeypatch)
    fake_graph = _FakeComputeEngine(
        {"Control": [("c1", {"status": "satisfied"})]},
        stream_rows_by_label={
            "Control": [
                {
                    "id": "opaque:1",
                    "kind": "graph_row",
                    "scores": {"score": None},
                    "confidence": 0.9,
                    "source_refs": [],
                    "valid_time": (None, None),
                    "tx_time": (None, None),
                    "policy_labels": [],
                    "contradiction_ids": [],
                    "proof_ids": [],
                },
                {
                    "id": "opaque:2",
                    "kind": "graph_row",
                    "scores": {"score": None},
                    "confidence": 0.2,
                    "source_refs": [],
                    "valid_time": (None, None),
                    "tx_time": (None, None),
                    "policy_labels": [],
                    "contradiction_ids": ["claim:x"],
                    "proof_ids": [],
                },
            ]
        },
    )
    engine = _FakeEngine(fake_graph)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        audit_tools,
        "_verify",
        lambda: {"surface": "audit", "action": "verify", "ok": True, "entries": 0},
    )

    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    rollup = out["confidence_rollup"]["Control"]
    assert rollup["sampled"] == 2
    assert rollup["avg_confidence"] == 0.55
    assert rollup["contested"] == 1  # the row with a contradiction_ids entry
    assert rollup["low_confidence"] == 0
    # A label with no streamed rows contributes no rollup entry.
    assert "Incident" not in out["confidence_rollup"]


def test_posture_omits_confidence_rollup_when_knowledge_stream_unavailable(
    monkeypatch,
):
    """Regression guard: a build/transport with no streaming surface (the
    existing ``_FakeComputeEngine`` default) reports posture exactly as
    before this feature — no empty/fabricated ``confidence_rollup`` key."""
    tool = _register(monkeypatch)
    fake_graph = _FakeComputeEngine({"Control": [("c1", {"status": "satisfied"})]})
    engine = _FakeEngine(fake_graph)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        audit_tools,
        "_verify",
        lambda: {"surface": "audit", "action": "verify", "ok": True, "entries": 0},
    )

    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "confidence_rollup" not in out


def test_confidence_rollup_has_no_opt_in_gate():
    """Query-seam closure ratchet (``reports/seam-closure-audit-2026-07-22.md``
    §4.3 — confirm ``KnowledgeStream`` is the DEFAULT, not merely reachable).

    ``_confidence_rollup`` is ``posture``'s only live default consumer of
    ``Method::KnowledgeStream``; the two tests above prove it runs whenever the
    engine exposes ``stream_graph_confidence`` and is silently omitted when it
    doesn't — i.e. capability detection is the ONLY gate, never an opt-in
    config/env flag a deployment must flip on. This is a source-level ratchet
    against that regressing quietly: if a future change wraps the call in
    ``config.*``/``os.environ``/``getenv`` (turning "default" back into
    "opt-in"), this test fails loudly instead of the seam silently reopening.
    """
    import inspect

    source = inspect.getsource(_confidence_rollup)
    for token in ("config.", "os.environ", "getenv", "Field(default"):
        assert token not in source, (
            f"_confidence_rollup must stay capability-gated only; found "
            f"opt-in-flag-shaped token {token!r} — KnowledgeStream is the "
            "documented default consumer, not opt-in"
        )


def test_posture_no_active_engine(monkeypatch):
    tool = _register(monkeypatch)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out


# --------------------------------------------------------------------------- #
# X4 — auth posture / secrets posture / policy version (CONCEPT:AU-KG.audit.
# compliance-posture-rollup). Aggregates the EXISTING, already-redacted
# agent-utilities-doctor checks (auth/outbound_auth/graph_authority/
# graph_identity for auth_posture; secrets/secrets_backend/mcp_fleet_secrets
# for secrets_posture) + the active ActionPolicy ruleset's own version — no
# new auth/secrets probe. Available even with no reachable KG engine.
# --------------------------------------------------------------------------- #


def test_posture_includes_policy_version_auth_and_secrets_posture(monkeypatch):
    """Live path: the REAL doctor.run_doctor + ActionPolicy, unmocked — proves
    genuine end-to-end wiring, not just that the aggregator calls a stub."""
    tool = _register(monkeypatch)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["policy_version"]["version"] == 1
    assert out["policy_version"]["default_tier"]
    assert out["policy_version"]["rule_count"] > 0
    assert "status" in out["auth_posture"]
    assert isinstance(out["auth_posture"]["checks"], list)
    assert {c["name"] for c in out["auth_posture"]["checks"]} == {
        "auth",
        "outbound_auth",
        "graph_authority",
        "graph_identity",
    }
    assert "status" in out["secrets_posture"]
    assert {c["name"] for c in out["secrets_posture"]["checks"]} == {
        "secrets",
        "secrets_backend",
        "mcp_fleet_secrets",
    }
    # X4 acceptance: these are present regardless of engine availability.
    assert out["error"] == "IntelligenceGraphEngine not active"


def test_posture_auth_and_secrets_posture_reuse_run_doctor_with_only(monkeypatch):
    """Proves REUSE, not reimplementation: posture() dispatches through
    ``doctor.run_doctor(only=[...])`` with exactly the documented check
    names — mirrors ``test_posture_joins_audit_verify_and_node_counts``'s
    proof that the audit-ledger half reuses ``audit_tools._verify``."""
    from agent_utilities.deployment import doctor as doctor_module

    calls: list[list[str]] = []

    def _fake_run_doctor(only=None, **kwargs):
        calls.append(list(only or []))
        return {
            "status": "healthy",
            "counts": {"ok": len(only or [])},
            "checks": [
                {"name": n, "status": "ok", "detail": "stub"} for n in (only or [])
            ],
        }

    monkeypatch.setattr(doctor_module, "run_doctor", _fake_run_doctor)
    tool = _register(monkeypatch)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)

    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert sorted(calls[0]) == sorted(
        ["auth", "outbound_auth", "graph_authority", "graph_identity"]
    )
    assert sorted(calls[1]) == sorted(
        ["secrets", "secrets_backend", "mcp_fleet_secrets"]
    )
    assert out["auth_posture"]["status"] == "healthy"
    assert out["secrets_posture"]["status"] == "healthy"


def test_posture_doctor_slice_degrades_on_import_failure(monkeypatch):
    """A broken doctor module must never break the whole posture rollup."""
    import builtins

    real_import = builtins.__import__

    def _boom_import(name, *a, **k):
        if name == "agent_utilities.deployment.doctor":
            raise RuntimeError("doctor module unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom_import)
    tool = _register(monkeypatch)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["auth_posture"]["status"] == "error"
    assert out["secrets_posture"]["status"] == "error"


# --------------------------------------------------------------------------- #
# X4 — bulk export redaction proof. ``_export`` is a thin pass-through over
# ``explain_belief`` (the SAME per-node redaction primitive graph_epistemic's
# 'why' action uses) — the real redaction decision lives server-side (policy-
# aware, disclosure_level-gated), not in this Python aggregator. These tests
# prove TWO things a Python-layer test CAN honestly prove: (1) a
# disclosure_level that the (simulated) engine redacts on genuinely produces
# NO secret/host-identity material anywhere in the bulk export output — with
# a NEGATIVE CONTROL (the SAME fixture at 'Full') proving the test can
# actually detect a leak, so the positive assertion isn't vacuous; and (2)
# ``_export``'s own aggregation never smuggles in anything beyond what
# explain_belief returned (no raw config/env value ever reaches the output).
# --------------------------------------------------------------------------- #

_FAKE_SECRET = "sk-live-fake-9f3e2b7a1c4d6e8f0a2b4c6d8e0f1a2b3c4d"  # nosec B105 - test fixture, not a real credential  # sanitizer:ignore — synthetic value, verifies compliance export redacts it
_FAKE_HOST_IDENTITY = "example-host-prod.internal.arpa"


class _LevelAwareQuery:
    """Mocks the engine's ``explain_belief`` as a REAL disclosure_level-aware
    redactor would: 'Full' returns the raw proof content (which, in this
    adversarial fixture, happens to embed a secret + a host identity);
    'Skeleton'/'ExistenceOnly' return a genuinely stripped, neutral payload —
    mirroring the real server's policy-aware proof redaction."""

    def __getattr__(self, name):
        def _call(**kwargs):
            node_id = kwargs.get("node_id")
            level = kwargs.get("disclosure_level", "Full")
            if level == "Full":
                return {
                    "root": {
                        "claim": node_id,
                        "rule": "Asserted",
                        # Adversarial: a raw proof step happens to carry
                        # secret/host material an unredacted level would leak.
                        "provenance": f"api_key={_FAKE_SECRET} host={_FAKE_HOST_IDENTITY}",
                    }
                }
            return {"root": {"claim": node_id, "rule": "[redacted]"}}

        return _call


class _LevelAwareClient:
    query = _LevelAwareQuery()


def test_export_redacted_disclosure_level_leaks_no_secret_or_host_identity(
    monkeypatch,
):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _LevelAwareClient())

    out_redacted = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["control:1", "control:2"]),
            disclosure_level="ExistenceOnly",
            as_of="",
            limit=200,
        )
    )
    redacted_blob = json.dumps(out_redacted)
    assert _FAKE_SECRET not in redacted_blob
    assert _FAKE_HOST_IDENTITY not in redacted_blob

    # NEGATIVE CONTROL: the SAME fixture at disclosure_level='Full' DOES leak
    # — proves the assertions above are a real detector, not vacuously true
    # because the strings never appear regardless of behavior.
    out_full = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["control:1", "control:2"]),
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    full_blob = json.dumps(out_full)
    assert _FAKE_SECRET in full_blob
    assert _FAKE_HOST_IDENTITY in full_blob


def test_export_never_adds_content_beyond_explain_beliefs_own_response(monkeypatch):
    """No Python-side smuggling channel: even with a real secret sitting in
    process config/environment, ``_export``'s aggregation surfaces EXACTLY
    ``{node_id, belief}`` per entry — nothing from config, nothing from the
    engine object, nothing beyond what the (here: intentionally
    secret-free) ``explain_belief`` response contained."""
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setenv("SOME_UNRELATED_TEST_SECRET", _FAKE_SECRET)

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                return {"root": {"claim": kwargs.get("node_id"), "rule": "Asserted"}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["control:1"]),
            disclosure_level="Skeleton",
            as_of="",
            limit=200,
        )
    )
    assert _FAKE_SECRET not in json.dumps(out)
    assert set(out["entries"][0].keys()) == {"node_id", "belief"}


def test_export_by_explicit_node_ids_reuses_explain_belief(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    calls: list[tuple[str, dict]] = []

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                calls.append((name, kwargs))
                return {"root": {"claim": kwargs.get("node_id"), "rule": "Asserted"}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["control:1", "control:2"]),
            disclosure_level="Skeleton",
            as_of="",
            limit=200,
        )
    )
    assert out["surface"] == "compliance"
    assert out["action"] == "export"
    assert out["exported"] == 2
    assert out["disclosure_level"] == "Skeleton"
    assert {c[0] for c in calls} == {"explain_belief"}
    assert all(c[1]["disclosure_level"] == "Skeleton" for c in calls)
    assert {e["node_id"] for e in out["entries"]} == {"control:1", "control:2"}


def test_export_by_cypher_selection(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(
        _FakeComputeEngine({}),
        cypher_rows=[{"id": "control:1"}, {"id": "control:2"}],
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                return {"root": {}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="MATCH (n:Control) RETURN n.id AS id",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["requested"] == 2
    assert out["exported"] == 2


def test_export_requires_ids_or_cypher(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out


def test_export_respects_limit_and_reports_truncation(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                return {"root": {}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["a", "b", "c"]),
            disclosure_level="Full",
            as_of="",
            limit=2,
        )
    )
    assert out["requested"] == 3
    assert out["exported"] == 2
    assert out["truncated"] is True


def test_unknown_action(monkeypatch):
    tool = _register(monkeypatch)
    out = json.loads(
        tool(
            action="bogus",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out
