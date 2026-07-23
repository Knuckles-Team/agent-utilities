"""Regression tests for cross-layer incident correlation + report-only
remediation (``agent_utilities.observability.incidents``) — Phase D/E of
``reports/unified-infra-intelligence-plan.md``. Mirrors
``agents/fan-manager/tests/test_kg_control.py``'s style: pure-function
assertions plus one fake-KG end-to-end pass.
"""

from __future__ import annotations

import time
from typing import Any

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
import agent_utilities.observability.health_ingest as hi
from agent_utilities.observability import incidents as inc

#: a reference "now" for building anomaly timestamps relative to the real
#: clock (never a hardcoded date — the correlation window/day-cutoff filters
#: are clock-relative, so a fixed date would drift stale against the sandbox's
#: real time).
_NOW = time.time()


def _ago(seconds: float) -> str:
    return inc._iso(_NOW - seconds)


class _Capture:
    """Captures every ``ingest_entities`` call (mirrors
    ``test_observability_health_ingest.py``'s fixture)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
                "domain": domain,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


class _FakeEngine:
    """Serves ``get_nodes_by_label`` per label from a fixed table, plus an
    optional ``get_neighbors`` table for the evidence-resolution tests."""

    def __init__(
        self,
        by_label: dict[str, list[tuple[str, dict]]],
        neighbors: dict[str, list[str]] | None = None,
    ) -> None:
        self._by_label = by_label
        self._neighbors = neighbors or {}

    def get_nodes_by_label(self, label: str, limit: int = 0):
        return self._by_label.get(label, [])

    def get_neighbors(self, node_id: str) -> list[str]:
        return list(self._neighbors.get(node_id, []))


def _anomaly(node_id, entity, signal, at, *, kind="above-baseline"):
    return (
        node_id,
        {
            "entity": entity,
            "signal": signal,
            "kind": kind,
            "zscore": 4.0,
            "observed": 90.0,
            "expected": 50.0,
            "observedAt": at,
        },
    )


# --- pure helpers -------------------------------------------------------- #
def test_asset_key_joins_producer_namespaces_on_shared_host():
    assert inc._asset_key("fan:host:storage-node-a") == "storage-node-a"
    assert inc._asset_key("systems:host:storage-node-a") == "storage-node-a"


def test_layer_of_maps_producer_prefixes():
    assert inc._layer_of("fan:host:storage-node-a") == "hardware"
    assert inc._layer_of("systems:host:storage-node-a") == "os"
    assert inc._layer_of("cm:node:analysis-node-a") == "orchestration"
    assert inc._layer_of("tunnel:path:analysis-node-a--storage-node-a") == "network"
    assert inc._layer_of("mystery:x:y") == "unknown"


def test_root_cause_layer_prefers_the_deepest():
    assert inc._root_cause_layer(["service", "hardware", "os"]) == "hardware"
    assert inc._root_cause_layer(["service", "os"]) == "os"
    assert inc._root_cause_layer(["unknown"]) == "unknown"


def test_severity_escalates_for_multi_layer_clusters():
    assert inc._severity_for(["hardware"]) == "warning"
    assert inc._severity_for(["hardware", "os"]) == "critical"


# --- correlate_incidents -------------------------------------------------- #
def test_correlate_incidents_groups_multi_layer_anomalies_on_shared_entity(monkeypatch):
    """Hardware + OS anomalies on the SAME host within the window collapse
    into ONE incident spanning both layers — the cross-layer payoff."""
    rows = {
        "HealthAnomaly": [
            _anomaly("a1", "fan:host:storage-node-a", "cpu_temp_c", _ago(120)),
            _anomaly("a2", "systems:host:storage-node-a", "load1", _ago(0)),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = inc.correlate_incidents(window_s=300, days=1)

    assert len(out) == 1
    incident = out[0]
    assert incident["layers"] == ["hardware", "os"]
    assert incident["signals"] == ["cpu_temp_c", "load1"]
    assert incident["entities"] == [
        "fan:host:storage-node-a",
        "systems:host:storage-node-a",
    ]
    assert incident["root_cause_layer"] == "hardware"
    assert incident["severity"] == "critical"
    assert set(incident["anomalies"]) == {"a1", "a2"}
    assert incident["written"] is True
    assert len(cap.calls) == 1
    incident_node = cap.calls[0]["entities"][0]
    assert incident_node["type"] == "Incident"
    assert incident_node["rootCauseLayer"] == "hardware"
    rel_types = {rel["type"] for rel in cap.calls[0]["relationships"]}
    assert rel_types == {"affectsEntity", "correlatesAnomaly"}


def test_correlate_incidents_leaves_unrelated_anomalies_separate(monkeypatch):
    """Anomalies on DIFFERENT assets, and anomalies on the same asset more than
    ``window_s`` apart, do NOT get collapsed into one incident."""
    rows = {
        "HealthAnomaly": [
            _anomaly("a1", "fan:host:storage-node-a", "cpu_temp_c", _ago(7200)),
            _anomaly("a2", "fan:host:compute-node-b", "cpu_temp_c", _ago(7190)),
            _anomaly("a3", "fan:host:storage-node-a", "cpu_temp_c", _ago(0)),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    out = inc.correlate_incidents(window_s=300, days=1)

    assert len(out) == 3
    assets = sorted(i["entity"] for i in out)
    assert assets == [
        "fan:host:compute-node-b",
        "fan:host:storage-node-a",
        "fan:host:storage-node-a",
    ]


def test_correlate_incidents_dedupes_already_open_incident(monkeypatch):
    rows = {
        "HealthAnomaly": [
            _anomaly("a1", "fan:host:storage-node-a", "cpu_temp_c", _ago(0)),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)

    # first pass writes the incident.
    cap1 = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap1)
    first = inc.correlate_incidents(window_s=300, days=1)
    assert len(cap1.calls) == 1
    written = first[0]
    assert written.get("deduped") is not True

    # seed the fake engine's Incident table with that now-open incident.
    rows["Incident"] = [
        (written["id"], {"status": "open", "signature": written["signature"]})
    ]

    # second pass over the SAME anomaly must not re-write the incident.
    cap2 = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap2)
    second = inc.correlate_incidents(window_s=300, days=1)
    assert len(cap2.calls) == 0
    assert second[0]["deduped"] is True
    assert second[0]["id"] == written["id"]


def test_correlate_incidents_no_engine_returns_empty(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert inc.correlate_incidents() == []


def test_correlate_incidents_ignores_anomalies_older_than_days(monkeypatch):
    rows = {
        "HealthAnomaly": [
            _anomaly(
                "a1",
                "fan:host:storage-node-a",
                "cpu_temp_c",
                "2020-01-01T00:00:00Z",
            ),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    assert inc.correlate_incidents(window_s=300, days=1) == []


# --- get_incident / get_incident_evidence ---------------------------------- #
def test_get_incident_returns_stored_props_by_id(monkeypatch):
    rows = {
        "Incident": [
            ("health:incident:a:1", {"status": "open", "summary": "a stress"}),
            ("health:incident:b:2", {"status": "resolved", "summary": "b stress"}),
        ]
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    got = inc.get_incident("health:incident:b:2")
    assert got == {
        "id": "health:incident:b:2",
        "status": "resolved",
        "summary": "b stress",
    }
    assert inc.get_incident("missing:id") is None


def test_get_incident_no_engine_returns_none(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert inc.get_incident("any") is None


def test_get_incident_evidence_resolves_anomalies_oldest_first_and_entities(
    monkeypatch,
):
    incident_id = "health:incident:storage-node-a:sig1"
    rows = {
        "HealthAnomaly": [
            _anomaly(
                "health:anomaly:storage-node-a:cpu:t2",
                "cm:node:storage-node-a",
                "cpu",
                "2026-07-02T00:00:00Z",
            ),
            _anomaly(
                "health:anomaly:storage-node-a:temp:t1",
                "fan:host:storage-node-a",
                "temp",
                "2026-07-01T00:00:00Z",
            ),
            # not correlated to this incident — must NOT show up in evidence.
            _anomaly(
                "health:anomaly:other-node:temp:t3",
                "fan:host:other-node",
                "temp",
                "2026-07-03T00:00:00Z",
            ),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(
        rows,
        neighbors={
            incident_id: [
                "health:anomaly:storage-node-a:cpu:t2",
                "health:anomaly:storage-node-a:temp:t1",
                "cm:node:storage-node-a",
                "fan:host:storage-node-a",
            ]
        },
    )
    monkeypatch.setattr(hi, "_engine", lambda: engine)

    evidence = inc.get_incident_evidence(incident_id)
    assert evidence is not None
    assert [a["id"] for a in evidence["anomalies"]] == [
        "health:anomaly:storage-node-a:temp:t1",
        "health:anomaly:storage-node-a:cpu:t2",
    ]
    assert evidence["entities"] == ["cm:node:storage-node-a", "fan:host:storage-node-a"]


def test_get_incident_evidence_no_neighbors_returns_empty(monkeypatch):
    engine = _FakeEngine({"HealthAnomaly": [], "Incident": []})
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    assert inc.get_incident_evidence("health:incident:x") == {
        "anomalies": [],
        "entities": [],
    }


def test_get_incident_evidence_no_engine_returns_none(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert inc.get_incident_evidence("any") is None


# --- set_incident_status ---------------------------------------------------- #
def _open_incident_row(incident_id: str) -> tuple[str, dict]:
    return (
        incident_id,
        {
            "kind": "hardware",
            "summary": "storage-node-a under thermal stress",
            "layers": ["hardware"],
            "signals": ["cpu_temp_c"],
            "severity": "warning",
            "rootCauseLayer": "hardware",
            "signature": "sig1",
            "status": "open",
            "observedAt": "2026-07-01T00:00:00Z",
        },
    )


def test_set_incident_status_acknowledges_and_preserves_other_fields(monkeypatch):
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine({"Incident": [_open_incident_row(incident_id)]})
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    result = inc.set_incident_status(incident_id, inc.STATUS_ACKNOWLEDGED, actor="op1")

    assert result == {"nodes": 1, "edges": 0}
    assert len(cap.calls) == 1
    node = cap.calls[0]["entities"][0]
    assert node["id"] == incident_id
    assert node["status"] == inc.STATUS_ACKNOWLEDGED
    assert node["ackedBy"] == "op1"
    assert node["ackedAt"]  # stamped, non-empty
    assert node.get("resolvedAt") is None
    # every pre-existing field round-trips instead of being blanked.
    assert node["kind"] == "hardware"
    assert node["layers"] == ["hardware"]
    assert node["signals"] == ["cpu_temp_c"]
    assert node["severity"] == "warning"
    assert node["rootCauseLayer"] == "hardware"
    assert node["signature"] == "sig1"
    assert node["observedAt"] == "2026-07-01T00:00:00Z"
    assert node["summary"] == "storage-node-a under thermal stress"


def test_set_incident_status_resolve_preserves_a_prior_ack(monkeypatch):
    incident_id = "health:incident:storage-node-a:sig1"
    acked_id, acked_props = _open_incident_row(incident_id)
    acked_props = {
        **acked_props,
        "status": inc.STATUS_ACKNOWLEDGED,
        "ackedAt": "2026-07-01T00:05:00Z",
        "ackedBy": "op1",
    }
    engine = _FakeEngine({"Incident": [(acked_id, acked_props)]})
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    result = inc.set_incident_status(incident_id, inc.STATUS_RESOLVED, actor="op2")

    assert result == {"nodes": 1, "edges": 0}
    node = cap.calls[0]["entities"][0]
    assert node["status"] == inc.STATUS_RESOLVED
    # the earlier ack survives the later resolve.
    assert node["ackedAt"] == "2026-07-01T00:05:00Z"
    assert node["ackedBy"] == "op1"
    assert node["resolvedBy"] == "op2"
    assert node["resolvedAt"]


def test_set_incident_status_unknown_incident_returns_none(monkeypatch):
    engine = _FakeEngine({"Incident": []})
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    assert inc.set_incident_status("missing:id", inc.STATUS_ACKNOWLEDGED) is None


def test_set_incident_status_no_engine_returns_none(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert inc.set_incident_status("any", inc.STATUS_ACKNOWLEDGED) is None


# --- propose_remediation --------------------------------------------------- #
def test_propose_remediation_writes_proposal_report_only(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)
    notified: list[str] = []
    monkeypatch.setattr(inc, "_notify", notified.append)

    incident = {
        "id": "health:incident:storage-node-a:abc",
        "entity": "systems:host:storage-node-a",
        "root_cause_layer": "os",
        "signals": ["disk_pct"],
    }
    proposal = inc.propose_remediation(incident)

    assert proposal is not None
    assert proposal["proposedAction"] == "disk_cleanup"
    assert proposal["targetPackage"] == "systems-manager"
    assert proposal["status"] == "proposed"
    assert len(cap.calls) == 1
    node = cap.calls[0]["entities"][0]
    assert node["type"] == "RemediationProposal"
    rel = cap.calls[0]["relationships"][0]
    assert rel["type"] == "proposesRemediation" and rel["target"] == incident["id"]
    assert notified  # best-effort notify fired


def test_propose_remediation_maps_every_layer():
    assert inc._proposed_action("hardware", [])["package"] == "fan-manager"
    assert (
        inc._proposed_action("orchestration", [])["package"] == "container-manager-mcp"
    )
    assert inc._proposed_action("service", [])["package"] == "lgtm-mcp"
    assert inc._proposed_action("network", [])["package"] == "tunnel-manager"
    assert inc._proposed_action("os", ["disk_pct"])["action"] == "disk_cleanup"
    assert inc._proposed_action("os", ["load1"])["action"] == "investigate_os_pressure"
    assert inc._proposed_action("unknown", [])["action"] == "investigate"


def test_propose_remediation_engine_unreachable_returns_none(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", lambda *a, **k: None)
    incident = {"id": "x", "root_cause_layer": "hardware", "signals": []}
    assert inc.propose_remediation(incident) is None


def test_notify_uses_bounded_shared_http_boundary(monkeypatch):
    import agent_utilities.core.config as config_module
    import agent_utilities.protocols.source_connectors.http_safety as http_safety

    monkeypatch.setattr(
        config_module,
        "setting",
        lambda key, default=None, cast=None: (
            "https://notify.example.invalid/events"
            if key == "INCIDENT_NOTIFY_URL"
            else default
        ),
    )
    calls = []
    monkeypatch.setattr(
        http_safety,
        "safe_post_json",
        lambda url, payload, **kwargs: calls.append((url, payload, kwargs)) or {},
    )

    inc._notify("bounded message")

    assert calls[0][1] == {"source": "incident-brain", "message": "bounded message"}
    assert calls[0][2]["max_request_bytes"] == 64 * 1024
    assert calls[0][2]["tls_service"] == "incident-notify"


# --- actuate_remediation (propose -> gate -> (held)) ---------------------- #
def test_actuate_remediation_refuses_non_restart_class_actions():
    proposal = {
        "proposedAction": "investigate_os_pressure",
        "entity": "systems:host:storage-node-a",
    }
    out = inc.actuate_remediation(proposal)
    assert out["status"] == "not_actuatable"


def test_actuate_remediation_refuses_when_no_target_entity():
    proposal = {"proposedAction": "restart_or_cordon_pod", "entity": ""}
    out = inc.actuate_remediation(proposal)
    assert out["status"] == "not_actuatable"


def test_actuate_remediation_defaults_to_held_pending_human_approval():
    """The whole point of the seam: with the SHIPPED default ActionPolicy
    (restart_service = approval_required), a safe restart-class proposal is
    always HELD, never executed — no monkeypatching of the policy at all."""
    proposal = {
        "id": "health:remediation:x",
        "proposedAction": "restart_or_cordon_pod",
        "entity": "cm:node:analysis-node-a",
        "incident": "health:incident:analysis-node-a:abc",
    }
    out = inc.actuate_remediation(proposal)
    assert out["status"] == "held"
    assert out["decision"] == "queue_approval"
    assert out["tier"] == "approval_required"
    assert out["action_kind"] == "restart_service"
    assert out["target"] == "analysis-node-a"


def test_actuate_remediation_only_executes_when_policy_explicitly_allows(monkeypatch):
    from agent_utilities.orchestration import action_policy as ap
    from agent_utilities.orchestration import fleet_actuation as fa

    class _AllowPolicy:
        def decide(self, request):
            return ap.ActionDecision(
                decision=ap.DECISION_ALLOW,
                tier=ap.TIER_AUTO,
                request=request,
                reason="test-allow",
            )

    monkeypatch.setattr(ap, "get_action_policy", lambda engine=None: _AllowPolicy())
    executed: list = []
    monkeypatch.setattr(
        fa,
        "execute_action",
        lambda engine, request, actuator: (
            executed.append(request) or {"ok": True, "dry_run": True}
        ),
    )

    proposal = {
        "id": "health:remediation:y",
        "proposedAction": "restart_or_cordon_pod",
        "entity": "cm:node:analysis-node-a",
    }
    out = inc.actuate_remediation(proposal)
    assert out["status"] == "executed"
    assert len(executed) == 1
    assert executed[0].kind == "restart_service"
    assert executed[0].target == "analysis-node-a"


# --- run_incident_correlation -------------------------------------------- #
def test_run_incident_correlation_summarizes_and_never_raises(monkeypatch):
    monkeypatch.setattr(
        inc,
        "correlate_incidents",
        lambda **kw: [
            {"id": "i1", "signature": "s1"},
            {"id": "i2", "signature": "s2", "deduped": True},
        ],
    )

    routed_calls: list[str] = []

    def fake_route(incident):
        routed_calls.append(incident["id"])
        return {"backend": "none", "ticket_status": "proposed"}

    proposed_calls: list[str] = []

    def fake_propose(incident):
        proposed_calls.append(incident["id"])
        return {"proposedAction": "investigate"}

    monkeypatch.setattr(
        "agent_utilities.observability.incident_router.route_incident", fake_route
    )
    monkeypatch.setattr(inc, "propose_remediation", fake_propose)

    summary = inc.run_incident_correlation()

    assert summary == {
        "incidents": 2,
        "new": 1,
        "deduped": 1,
        "routed": 2,
        "proposed": 2,
    }
    assert routed_calls == ["i1", "i2"]
    assert proposed_calls == ["i1", "i2"]


def test_run_incident_correlation_survives_a_routing_failure(monkeypatch):
    monkeypatch.setattr(
        inc, "correlate_incidents", lambda **kw: [{"id": "i1", "signature": "s1"}]
    )

    def boom(incident):
        raise RuntimeError("adapter exploded")

    monkeypatch.setattr(
        "agent_utilities.observability.incident_router.route_incident", boom
    )
    monkeypatch.setattr(inc, "propose_remediation", lambda incident: None)

    summary = inc.run_incident_correlation()
    assert summary["routed"] == 0
    assert summary["proposed"] == 0
    assert summary["incidents"] == 1


# --- actuation wiring in run_incident_correlation (default OFF) ----------- #
def test_run_incident_correlation_default_flag_off_never_attempts_actuation(
    monkeypatch,
):
    """CONCEPT:AU-OS.host.report-only-remediation-proposal — with
    INCIDENT_ACTUATION_ENABLED unset (the shipped default), the tick never
    even calls actuate_remediation and the summary shape is byte-identical
    to the pre-actuator-seam report-only behavior (no ``actuated``/``held``
    keys)."""
    monkeypatch.setattr(
        inc, "correlate_incidents", lambda **kw: [{"id": "i1", "signature": "s1"}]
    )
    monkeypatch.setattr(
        "agent_utilities.observability.incident_router.route_incident", lambda i: True
    )
    proposal = {
        "id": "p1",
        "proposedAction": "restart_or_cordon_pod",
        "entity": "cm:node:analysis-node-a",
    }
    monkeypatch.setattr(inc, "propose_remediation", lambda incident: proposal)
    attempted: list = []
    monkeypatch.setattr(inc, "actuate_remediation", lambda *a, **k: attempted.append(1))

    summary = inc.run_incident_correlation()

    assert attempted == []
    assert "actuated" not in summary
    assert "held" not in summary
    assert summary == {
        "incidents": 1,
        "new": 1,
        "deduped": 0,
        "routed": 1,
        "proposed": 1,
    }


def test_run_incident_correlation_enabled_flag_wires_through_to_held(monkeypatch):
    """With the flag explicitly on, the tick offers the eligible proposal to
    the SAME fail-closed gate — the default ActionPolicy still holds it, so
    ``held`` increments and ``actuated`` stays 0 (never autonomous by
    default even with the wiring switched on)."""
    import agent_utilities.core.config as cfg_mod

    def fake_setting(key, default=None, cast=None):
        if key == "INCIDENT_ACTUATION_ENABLED":
            return True
        return default

    monkeypatch.setattr(cfg_mod, "setting", fake_setting)
    monkeypatch.setattr(hi, "_engine", lambda: None)
    monkeypatch.setattr(
        inc, "correlate_incidents", lambda **kw: [{"id": "i1", "signature": "s1"}]
    )
    monkeypatch.setattr(
        "agent_utilities.observability.incident_router.route_incident", lambda i: True
    )
    proposal = {
        "id": "p1",
        "proposedAction": "restart_or_cordon_pod",
        "entity": "cm:node:analysis-node-a",
    }
    monkeypatch.setattr(inc, "propose_remediation", lambda incident: proposal)

    summary = inc.run_incident_correlation()

    assert summary["actuated"] == 0
    assert summary["held"] == 1
