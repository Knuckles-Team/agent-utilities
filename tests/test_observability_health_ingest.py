"""Regression tests for the shared health KG I/O
(``agent_utilities.observability.health_ingest``) — typed ``:HealthTrend``/
``:HealthBaseline``/``:HealthAnomaly``/``:Incident`` writers/readers over the unified
infrastructure ontology. Mirrors ``agents/fan-manager/tests``' read/write coverage of
``fan_manager.kg_ingest``, generalized off host/°C to any entity/signal pair.
"""

from __future__ import annotations

from typing import Any

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
from agent_utilities.observability import health_ingest as hi


class _Capture:
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


def test_ingest_health_trend_writes_typed_node_and_edge(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    trend = {
        "min": 40.0,
        "max": 60.0,
        "avg": 50.0,
        "avg_control": 30.0,
        "samples": 120,
        "window_s": 3600,
    }
    out = hi.ingest_health_trend(
        "sys:host:r510", "Host", "os", "cpu_temp_c", trend, host="r510"
    )
    assert out == {"nodes": 2, "edges": 1}
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["source"] == "agent-utilities-health"
    assert call["domain"] == "os"

    scaffold, trend_node = call["entities"]
    assert scaffold == {"id": "sys:host:r510", "type": "Host", "name": "sys:host:r510"}
    assert trend_node["type"] == "HealthTrend"
    assert trend_node["entity"] == "sys:host:r510"
    assert trend_node["signal"] == "cpu_temp_c"
    assert trend_node["layer"] == "os"
    assert (
        trend_node["avg"] == 50.0
        and trend_node["min"] == 40.0
        and trend_node["max"] == 60.0
    )
    assert trend_node["avgControl"] == 30.0
    assert trend_node["samples"] == 120 and trend_node["windowS"] == 3600
    assert trend_node["host"] == "r510"

    rel = call["relationships"][0]
    assert rel["source"] == trend_node["id"]
    assert rel["target"] == "sys:host:r510"
    assert rel["type"] == "affectsEntity"


def test_ingest_health_baseline_one_per_entity_signal(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    baseline = {
        "p50": 55.0,
        "p95": 62.0,
        "min_env": 45.0,
        "max_env": 60.0,
        "avg_control": 20.0,
        "inertia": 0.4,
        "windows": 8,
    }
    hi.ingest_health_baseline("cm:node:r820", "load1", baseline, entity_type="Node")
    call = cap.calls[0]
    _, node = call["entities"]
    assert node["id"] == "health:baseline:cm:node:r820:load1"
    assert node["type"] == "HealthBaseline"
    assert node["p50"] == 55.0 and node["p95"] == 62.0
    assert node["minEnv"] == 45.0 and node["maxEnv"] == 60.0
    assert node["inertia"] == 0.4 and node["windows"] == 8

    # re-ingesting the SAME entity+signal yields the SAME node id (overwrite semantics)
    cap2 = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap2)
    hi.ingest_health_baseline("cm:node:r820", "load1", baseline, entity_type="Node")
    assert cap2.calls[0]["entities"][1]["id"] == node["id"]


def test_ingest_health_anomaly_linked_affects_entity(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    anomaly = {
        "kind": "above-baseline",
        "zscore": 4.2,
        "observed": 80.0,
        "expected": 55.0,
    }
    hi.ingest_health_anomaly("cm:node:r820", "load1", anomaly, entity_type="Node")
    call = cap.calls[0]
    _, node = call["entities"]
    assert node["type"] == "HealthAnomaly"
    assert node["kind"] == "above-baseline"
    assert node["zscore"] == 4.2
    rel = call["relationships"][0]
    assert rel["type"] == "affectsEntity" and rel["target"] == "cm:node:r820"


def test_ingest_incident_links_every_entity(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    incident = {
        "kind": "thermal-and-load-stress",
        "summary": "r820 under thermal/compute stress",
        "entities": ["fan:host:r820", "cm:node:r820", "cm:pod:r820-1"],
    }
    hi.ingest_incident(incident)
    call = cap.calls[0]
    assert call["entities"][0]["type"] == "Incident"
    assert call["entities"][0]["summary"] == "r820 under thermal/compute stress"
    targets = {rel["target"] for rel in call["relationships"]}
    assert targets == set(incident["entities"])
    assert all(rel["type"] == "affectsEntity" for rel in call["relationships"])


def test_ingest_incident_carries_rich_correlation_fields_and_anomaly_edges(monkeypatch):
    """Phase D (agent_utilities.observability.incidents) populates the richer
    shape the original Phase-A docstring flagged as a later phase."""
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    incident = {
        "id": "health:incident:r820:sig1",
        "summary": "r820 under thermal/compute stress",
        "entities": ["fan:host:r820", "cm:node:r820"],
        "anomalies": ["health:anomaly:fan:host:r820:cpu_temp_c:t1"],
        "layers": ["hardware", "orchestration"],
        "signals": ["cpu_temp_c", "restart_count"],
        "severity": "critical",
        "root_cause_layer": "hardware",
        "signature": "sig1",
        "status": "open",
        "opened_at": "2026-07-11T00:00:00Z",
    }
    hi.ingest_incident(incident)
    call = cap.calls[0]
    node = call["entities"][0]
    assert node["layers"] == ["hardware", "orchestration"]
    assert node["signals"] == ["cpu_temp_c", "restart_count"]
    assert node["severity"] == "critical"
    assert node["rootCauseLayer"] == "hardware"
    assert node["signature"] == "sig1"
    assert node["status"] == "open"
    assert node["observedAt"] == "2026-07-11T00:00:00Z"

    by_type = {}
    for rel in call["relationships"]:
        by_type.setdefault(rel["type"], set()).add(rel["target"])
    assert by_type["affectsEntity"] == {"fan:host:r820", "cm:node:r820"}
    assert by_type["correlatesAnomaly"] == {
        "health:anomaly:fan:host:r820:cpu_temp_c:t1"
    }


def test_ingest_functions_are_engine_guarded_no_op(monkeypatch):
    """With no reachable engine, ingest_entities returns None and nothing raises."""
    monkeypatch.setattr(native_ingest, "ingest_entities", lambda *a, **k: None)
    assert hi.ingest_health_trend("e1", "Host", "os", "sig", {"avg": 1}) is None
    assert hi.ingest_health_baseline("e1", "sig", {"p50": 1, "p95": 2}) is None
    assert hi.ingest_health_anomaly("e1", "sig", {"kind": "above-baseline"}) is None
    assert hi.ingest_incident({"kind": "x", "entities": ["e1"]}) is None


# --- read_health_trends -------------------------------------------------------- #
class _FakeEngine:
    def __init__(self, rows: list[tuple[str, dict]]) -> None:
        self._rows = rows

    def get_nodes_by_label(self, label, limit=0):
        assert label == "HealthTrend"
        return self._rows


def test_read_health_trends_filters_entity_signal_and_days():
    rows = [
        (
            "t1",
            {
                "entity": "e1",
                "signal": "cpu",
                "observedAt": "2026-07-10T00:00:00Z",
                "avg": 1,
            },
        ),
        (
            "t2",
            {
                "entity": "e1",
                "signal": "disk",
                "observedAt": "2026-07-10T00:00:00Z",
                "avg": 2,
            },
        ),
        (
            "t3",
            {
                "entity": "e2",
                "signal": "cpu",
                "observedAt": "2026-07-10T00:00:00Z",
                "avg": 3,
            },
        ),
        (
            "t4",
            {
                "entity": "e1",
                "signal": "cpu",
                "observedAt": "2020-01-01T00:00:00Z",
                "avg": 4,
            },
        ),
        (
            "t5",
            {
                "entity": "e1",
                "signal": "cpu",
                "observedAt": "2026-07-11T00:00:00Z",
                "avg": 5,
            },
        ),
    ]
    eng = _FakeEngine(rows)
    out = hi.read_health_trends("e1", "cpu", days=3650, engine=eng)
    # only e1/cpu rows within the (huge) day window, sorted oldest -> newest
    assert [r["avg"] for r in out] == [4, 1, 5]


def test_read_health_trends_no_engine_returns_empty(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert hi.read_health_trends("e1", "cpu") == []
