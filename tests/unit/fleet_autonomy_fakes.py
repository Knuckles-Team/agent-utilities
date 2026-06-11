"""Shared fakes for the autonomy control-plane tests (OS-5.24 — OS-5.27).

A minimal KG-engine double honoring exactly the surface the control plane
uses (``add_node`` / ``query_cypher`` / ``backend.execute`` / ``submit_task``
/ ``link_nodes``), plus observer/actuator/notifier doubles.
"""

from __future__ import annotations

import time
from typing import Any

from agent_utilities.orchestration.fleet_observation import ServiceObservation


def utc_now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class FakeBackend:
    def __init__(self, engine: FakeEngine):
        self.engine = engine
        self.executed: list[tuple[str, dict]] = []

    def execute(self, query: str, params: dict | None = None):
        params = params or {}
        self.executed.append((query, params))
        # Emulate the two SET shapes the control plane writes.
        node = self.engine.nodes.get(params.get("id"))
        if node is not None and "SET" in query:
            if "remediation_log" in query:
                node["remediation_log"] = params.get("log")
                node["remediation_status"] = params.get("status")
            if "a.status" in query and params.get("status"):
                node["status"] = params["status"]
        return []


class FakeEngine:
    """In-memory engine double for ActionPolicy/reconciler/playbook/watch tests."""

    def __init__(self):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.submitted: list[dict[str, Any]] = []
        self.backend = FakeBackend(self)

    # ── node surface ────────────────────────────────────────────────
    def add_node(self, node_id: str, node_type: str, properties: dict | None = None):
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            **(properties or {}),
        }

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.edges.append((source_id, target_id, rel_type))

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]

    # ── query surface (pattern-matched on the control plane's cyphers) ──
    def query_cypher(self, query: str, params: dict | None = None):
        params = params or {}
        if "governance_rule" in query:
            return [
                {"r": dict(n)}
                for n in self.by_type("governance_rule")
                if n.get("scope") == "action_policy"
            ]
        if "ActionApproval" in query and "'approved'" in query:
            return [
                {"a": dict(n)}
                for n in self.by_type("ActionApproval")
                if n.get("status") == "approved"
            ]
        if "ActionApproval" in query and "'pending'" in query:
            return [
                {"id": n["id"]}
                for n in self.by_type("ActionApproval")
                if n.get("status") == "pending"
                and n.get("kind") == params.get("kind")
                and n.get("target") == params.get("target")
            ][:1]
        if "ActionDecision" in query:
            return [
                {
                    "target": n.get("target"),
                    "decision": n.get("decision"),
                    "ts": n.get("decided_unix"),
                }
                for n in self.by_type("ActionDecision")
                if n.get("kind") == params.get("kind")
            ]
        if "FleetEvent {id" in query:
            node = self.nodes.get(params.get("id"))
            return [{"e": dict(node)}] if node else []
        if "MATCH (e:FleetEvent)" in query:
            return [
                {
                    "subject": n.get("subject"),
                    "status": n.get("status"),
                    "severity": n.get("severity"),
                    "received_at": n.get("received_at") or utc_now_str(),
                }
                for n in self.by_type("FleetEvent")
            ]
        if "Task {id" in query:
            node = self.nodes.get(params.get("id"))
            return [{"t": dict(node)}] if node else []
        if "CONTAINS" in query:
            return []
        return []

    # ── durable task queue surface ──────────────────────────────────
    def submit_task(
        self, target_path, is_codebase, provenance, task_type=None, skip_dedupe=False
    ):
        job_id = f"job-{len(self.submitted)}"
        self.nodes[job_id] = {
            "id": job_id,
            "type": "Task",
            "status": "pending",
            **(provenance or {}),
        }
        self.submitted.append(
            {"job_id": job_id, "target": target_path, "task_type": task_type}
        )
        return job_id


class FakeObserver:
    """Scriptable FleetObserver double."""

    name = "fake"

    def __init__(self, observations: dict[str, ServiceObservation] | None = None):
        self.observations = observations or {}

    def observe(self):
        return dict(self.observations)

    def service_status(self, name: str) -> ServiceObservation:
        return self.observations.get(name) or ServiceObservation(name=name)


def obs(name: str, status: str, replicas: int | None = None, flaps: int = 0):
    return ServiceObservation(
        name=name,
        status=status,
        replicas=replicas,
        flap_count=flaps,
        last_seen_unix=time.time(),
        sources=["fake"],
        detail=f"fake {status}",
    )


class CaptureNotifier:
    """Notification sink double recording every message."""

    def __init__(self):
        self.messages: list[str] = []

    def notify(self, spec, message: str):
        self.messages.append(message)
        return {"delivered": True, "transport": "capture", "message": message}


def write_policy(tmp_path, body: str):
    path = tmp_path / "policy.yml"
    path.write_text(body, encoding="utf-8")
    return str(path)
