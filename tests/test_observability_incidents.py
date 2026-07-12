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
    """Serves ``get_nodes_by_label`` per label from a fixed table."""

    def __init__(self, by_label: dict[str, list[tuple[str, dict]]]) -> None:
        self._by_label = by_label

    def get_nodes_by_label(self, label: str, limit: int = 0):
        return self._by_label.get(label, [])


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
    assert inc._asset_key("fan:host:r510") == "r510"
    assert inc._asset_key("systems:host:r510") == "r510"


def test_layer_of_maps_producer_prefixes():
    assert inc._layer_of("fan:host:r510") == "hardware"
    assert inc._layer_of("systems:host:r510") == "os"
    assert inc._layer_of("cm:node:r820") == "orchestration"
    assert inc._layer_of("tunnel:path:r820-r510") == "network"
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
            _anomaly("a1", "fan:host:r510", "cpu_temp_c", _ago(120)),
            _anomaly("a2", "systems:host:r510", "load1", _ago(0)),
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
    assert incident["entities"] == ["fan:host:r510", "systems:host:r510"]
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
            _anomaly("a1", "fan:host:r510", "cpu_temp_c", _ago(7200)),
            _anomaly("a2", "fan:host:r710", "cpu_temp_c", _ago(7190)),
            _anomaly("a3", "fan:host:r510", "cpu_temp_c", _ago(0)),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    out = inc.correlate_incidents(window_s=300, days=1)

    assert len(out) == 3
    assets = sorted(i["entity"] for i in out)
    assert assets == ["fan:host:r510", "fan:host:r510", "fan:host:r710"]


def test_correlate_incidents_dedupes_already_open_incident(monkeypatch):
    rows = {
        "HealthAnomaly": [
            _anomaly("a1", "fan:host:r510", "cpu_temp_c", _ago(0)),
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
            _anomaly("a1", "fan:host:r510", "cpu_temp_c", "2020-01-01T00:00:00Z"),
        ],
        "Incident": [],
    }
    engine = _FakeEngine(rows)
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    assert inc.correlate_incidents(window_s=300, days=1) == []


# --- propose_remediation --------------------------------------------------- #
def test_propose_remediation_writes_proposal_report_only(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)
    notified: list[str] = []
    monkeypatch.setattr(inc, "_notify", notified.append)

    incident = {
        "id": "health:incident:r510:abc",
        "entity": "systems:host:r510",
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
