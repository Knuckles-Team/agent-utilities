"""Fleet events webhook ingress + triage (CONCEPT:AU-OS.config.fleet-event-ingress).

Covers payload normalization for all three sender formats (Alertmanager v4,
Uptime Kuma, generic/Portainer), KG persistence + durable triage enqueue,
shared-secret token enforcement, the per-source storm cap, and the daemon-side
triage handler (correlation, failure_gap escalation, playbook dispatch seam).

@pytest.mark.concept("AU-OS.config.fleet-event-ingress")
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.gateway import fleet_events
from agent_utilities.knowledge_graph.adaptation import fleet_event_triage

pytestmark = pytest.mark.concept("AU-OS.config.fleet-event-ingress")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Req:
    """Minimal request double: json body + headers + query params."""

    def __init__(self, body=None, headers=None, query=None):
        self._body = body
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = query or {}

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _Backend:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params or {}))
        return []


class _Engine:
    """Fake engine honoring add_node/link_nodes/query_cypher/submit_task."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.submitted: list[dict] = []
        self.backend = _Backend()

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.edges.append((source_id, target_id, rel_type.upper()))

    def query_cypher(self, query, params=None):
        params = params or {}
        if "FleetEvent" in query and "RETURN e" in query:
            node = self.nodes.get(params.get("id"))
            return [{"e": dict(node)}] if node else []
        if "CONTAINS" in query:
            subject = str(params.get("subject", "")).lower()
            return [
                {"id": n["id"], "name": n.get("name")}
                for n in self.nodes.values()
                if n["type"] in {"Server", "Session", "Resource", "Tool"}
                and subject in str(n.get("name", "")).lower()
            ]
        return []

    def submit_task(
        self, target_path, is_codebase, provenance, task_type=None, skip_dedupe=False
    ):
        job_id = f"job-{len(self.submitted)}"
        self.submitted.append(
            {
                "job_id": job_id,
                "target": target_path,
                "task_type": task_type,
                "provenance": provenance,
            }
        )
        return job_id


@pytest.fixture
def engine(monkeypatch):
    eng = _Engine()
    monkeypatch.setattr(fleet_events, "_get_engine", lambda: eng)
    # Isolate the in-memory storm counters between tests.
    monkeypatch.setattr(fleet_events, "_rate_counters", {})
    return eng


async def _payload(resp):
    return json.loads(resp.body)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _alertmanager_payload():
    return {
        "version": "4",
        "receiver": "agent-os",
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighErrorRate",
                    "severity": "critical",
                    "service": "kg-gateway",
                },
                "annotations": {"summary": "error rate above 5%"},
            },
            {
                "status": "resolved",
                "labels": {"alertname": "DiskFull", "instance": "r820:9100"},
                "annotations": {"description": "disk back under threshold"},
            },
        ],
    }


class TestNormalization:
    def test_alertmanager_v4(self):
        events = fleet_events.normalize_payload(_alertmanager_payload())
        assert len(events) == 2
        first = events[0]
        assert first.source == "alertmanager"
        assert first.severity == "critical"
        assert first.subject == "kg-gateway"
        assert first.status == "firing"
        assert "error rate" in first.summary
        # resolved alerts normalize to info severity
        assert events[1].status == "resolved"
        assert events[1].severity == "info"

    def test_uptime_kuma_down(self):
        payload = {
            "heartbeat": {"status": 0, "msg": "timeout"},
            "monitor": {"name": "langfuse.arpa"},
            "msg": "[langfuse.arpa] is down",
        }
        (ev,) = fleet_events.normalize_payload(payload)
        assert ev.source == "uptime-kuma"
        assert ev.severity == "critical"
        assert ev.status == "down"
        assert ev.subject == "langfuse.arpa"

    def test_uptime_kuma_up_is_info(self):
        payload = {"heartbeat": {"status": 1}, "monitor": {"name": "egeria.arpa"}}
        (ev,) = fleet_events.normalize_payload(payload)
        assert ev.severity == "info"
        assert ev.status == "up"

    def test_generic_with_source_hint(self):
        payload = {"service": "portainer-stack", "severity": "error", "status": "down"}
        (ev,) = fleet_events.normalize_payload(payload, source_hint="portainer")
        assert ev.source == "portainer"
        assert ev.severity == "error"
        assert ev.subject == "portainer-stack"

    def test_generic_arbitrary_json_accepted(self):
        (ev,) = fleet_events.normalize_payload({"whatever": 1})
        assert ev.source == "generic"
        assert ev.severity == "info"

    def test_non_dict_rejected(self):
        assert fleet_events.normalize_payload([1, 2]) == []


# ---------------------------------------------------------------------------
# Endpoint: persistence + enqueue + auth + storm cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReceiveEndpoint:
    async def test_persists_nodes_and_enqueues_triage(self, engine):
        resp = await fleet_events.fleet_events_receive(_Req(_alertmanager_payload()))
        data = await _payload(resp)
        assert resp.status_code == 200
        assert data["accepted"] == 2
        fleet_nodes = [n for n in engine.nodes.values() if n["type"] == "FleetEvent"]
        assert len(fleet_nodes) == 2
        assert fleet_nodes[0]["triage_status"] == "pending"
        assert {t["task_type"] for t in engine.submitted} == {"fleet_event_triage"}
        # the queued target is the persisted FleetEvent node id
        assert engine.submitted[0]["target"] in engine.nodes

    async def test_header_source_hint_used(self, engine):
        req = _Req({"msg": "redeploy"}, headers={"X-Event-Source": "portainer"})
        resp = await fleet_events.fleet_events_receive(req)
        data = await _payload(resp)
        assert data["events"][0]["source"] == "portainer"

    async def test_token_required_when_configured(self, engine, monkeypatch):
        monkeypatch.setenv("FLEET_EVENTS_TOKEN", "s3cret")
        resp = await fleet_events.fleet_events_receive(_Req({"msg": "x"}))
        assert resp.status_code == 401
        ok = await fleet_events.fleet_events_receive(
            _Req({"msg": "x"}, headers={"X-Fleet-Events-Token": "s3cret"})
        )
        assert ok.status_code == 200

    async def test_wrong_token_rejected(self, engine, monkeypatch):
        monkeypatch.setenv("FLEET_EVENTS_TOKEN", "s3cret")
        resp = await fleet_events.fleet_events_receive(
            _Req({"msg": "x"}, headers={"X-Fleet-Events-Token": "nope"})
        )
        assert resp.status_code == 401

    async def test_invalid_json_is_400(self, engine):
        resp = await fleet_events.fleet_events_receive(_Req(None))
        assert resp.status_code == 400

    async def test_engine_unavailable_is_503(self, monkeypatch):
        monkeypatch.setattr(fleet_events, "_rate_counters", {})
        monkeypatch.setattr(fleet_events, "_get_engine", lambda: None)
        resp = await fleet_events.fleet_events_receive(_Req({"msg": "x"}))
        assert resp.status_code == 503

    async def test_storm_cap_returns_429(self, engine, monkeypatch):
        monkeypatch.setattr(fleet_events, "RATE_CAP_PER_MINUTE", 3)
        for _ in range(3):
            resp = await fleet_events.fleet_events_receive(_Req({"msg": "x"}))
            assert resp.status_code == 200
        resp = await fleet_events.fleet_events_receive(_Req({"msg": "x"}))
        assert resp.status_code == 429
        # a different source has its own counter
        other = await fleet_events.fleet_events_receive(
            _Req({"msg": "x"}, headers={"X-Event-Source": "other"})
        )
        assert other.status_code == 200


# ---------------------------------------------------------------------------
# Daemon-side triage handler
# ---------------------------------------------------------------------------


def _seed_event(engine, *, severity="critical", status="firing", subject="kg-gateway"):
    eid = "fleet_event:test1"
    engine.add_node(
        eid,
        "FleetEvent",
        properties={
            "source": "alertmanager",
            "severity": severity,
            "subject": subject,
            "status": status,
            "summary": "error rate above 5%",
            "triage_status": "pending",
        },
    )
    return eid


class TestTriage:
    def test_critical_event_files_failure_gap(self):
        engine = _Engine()
        eid = _seed_event(engine)
        report = fleet_event_triage.triage_fleet_event(engine, eid)
        assert report["triaged"] is True
        gaps = [
            n
            for n in engine.nodes.values()
            if n["type"] == "Concept" and n.get("kind") == "failure_gap"
        ]
        assert len(gaps) == 1
        assert gaps[0]["source"] == "fleet_event_triage"
        # provenance: event EVIDENCES gap
        assert (eid, gaps[0]["id"], "EVIDENCES") in engine.edges
        # node stamped triaged
        assert any("triage_status" in q for q, _ in engine.backend.executed)

    def test_info_event_does_not_escalate(self):
        engine = _Engine()
        eid = _seed_event(engine, severity="info", status="resolved")
        report = fleet_event_triage.triage_fleet_event(engine, eid)
        assert report["triaged"] is True
        assert "gap_topic" not in report
        assert not [n for n in engine.nodes.values() if n.get("kind") == "failure_gap"]

    def test_correlation_links_known_entities(self):
        engine = _Engine()
        engine.add_node("srv:kg-gateway", "Server", properties={"name": "kg-gateway"})
        eid = _seed_event(engine)
        report = fleet_event_triage.triage_fleet_event(engine, eid)
        assert "srv:kg-gateway" in report["correlated"]
        assert (eid, "srv:kg-gateway", "OBSERVED_ON") in engine.edges

    def test_missing_event_reports_not_triaged(self):
        engine = _Engine()
        report = fleet_event_triage.triage_fleet_event(engine, "fleet_event:nope")
        assert report["triaged"] is False

    def test_playbook_dispatch_seam(self, monkeypatch):
        engine = _Engine()
        eid = _seed_event(engine, severity="critical")
        calls = []
        monkeypatch.setitem(
            fleet_event_triage.PLAYBOOKS,
            "alertmanager:critical",
            lambda eng, ev: calls.append(ev["id"]) or {"playbook": "custom"},
        )
        report = fleet_event_triage.triage_fleet_event(engine, eid)
        assert calls == [eid]
        assert report["playbook"] == "custom"
