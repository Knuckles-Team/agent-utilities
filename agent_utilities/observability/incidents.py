#!/usr/bin/python
from __future__ import annotations

"""Cross-layer incident correlation + report-only remediation — Phase D/E of the
unified infrastructure intelligence plan (``reports/unified-infra-intelligence-plan.md``).

CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation. Reads recent
``:HealthAnomaly`` rows written by every layer's producer (fan-manager=hardware,
systems-manager=os, and — once shipped — container-manager-mcp=orchestration,
lgtm-mcp/uptime-kuma-agent=service, tunnel-manager=network) across the shared
engine, groups the ones landing on the SAME asset (host/node) within a short
time window into ONE ``:Incident``, and estimates the deepest contributing
layer as the likely root cause — the payoff of the layered model: one incident
that explains a symptom by its cause across layers, not five independent
alerts.

:func:`run_incident_correlation` is the one-pass entry point (correlate → write
→ route to a ticketing system-of-record →
:func:`propose_remediation`, CONCEPT:AU-OS.host.report-only-remediation-proposal), meant for a CronJob /
``graph_loops`` tick (``python -m agent_utilities.observability.incidents``).
Routing/remediation dispatch is REPORT-ONLY by design — see
:mod:`agent_utilities.observability.incident_router` and
:func:`propose_remediation` for how an operator later turns either on.
Everything here is engine-guarded and best-effort: with no reachable engine,
every entry point degrades to an empty/no-op result rather than raising.

:func:`actuate_remediation` is that "turn it on" seam, finally wired: a
proposal for a narrow, safe restart-class action (see
``_SAFE_ACTUATION_KINDS``) CAN now flow propose → the SAME fail-closed
:class:`~agent_utilities.orchestration.action_policy.ActionPolicy` gate
``fleet_reconciler.py`` already uses → (held). It is deliberately NOT a
switch that turns on autonomous remediation: the shipped default policy tier
for ``restart_service`` is ``approval_required``
(``deploy/action-policy.default.yml``), so every call is held pending a
human `ActionApproval` grant unless an operator explicitly relaxes that
tier — nothing here auto-executes. The CronJob tick only even ATTEMPTS the
gate when ``INCIDENT_ACTUATION_ENABLED`` is set (default off, CONCEPT:AU-OS.host.report-only-remediation-proposal), so the
out-of-the-box behavior of :func:`run_incident_correlation` is byte-identical
to before this seam existed: propose-and-hold, never autonomous.
"""

import hashlib
import json
import logging
import time
import urllib.request
from typing import Any

from agent_utilities.observability import health_ingest

# Reused via the ``health_ingest`` module reference (never re-bound at import
# time) so tests can monkeypatch ``health_ingest._engine`` the same way
# ``test_observability_health_ingest.py`` already does for ``read_health_trends``.

logger = logging.getLogger("agent_utilities.observability.incidents")

# hardware < os < orchestration < service < network — the layer nearest the
# metal is presumed the root cause when several fire together within one
# correlation window (the ordering the unified-infra-intelligence plan documents).
LAYER_DEPTH: dict[str, int] = {
    "hardware": 0,
    "os": 1,
    "orchestration": 2,
    "service": 3,
    "network": 4,
}

# Producer entity-id namespace prefix -> layer. Every producer stamps entity
# ids as ``<namespace>:<host|node|...>:<slug>`` (``fan:host:r510``,
# ``systems:host:r510``, ``cm:node:r820``, ...) — the prefix is the one stable
# signal of which layer wrote the anomaly.
_LAYER_PREFIXES: dict[str, str] = {
    "fan": "hardware",
    "thermal": "hardware",
    "systems": "os",
    "os": "os",
    "cm": "orchestration",
    "container": "orchestration",
    "node": "orchestration",
    "pod": "orchestration",
    "workload": "orchestration",
    "k8s": "orchestration",
    "service": "service",
    "endpoint": "service",
    "uptime": "service",
    "lgtm": "service",
    "tunnel": "network",
    "network": "network",
}

DEFAULT_SEVERITY = "warning"
_MULTI_LAYER_SEVERITY = "critical"

# root_cause_layer -> (proposed action, owning package, rationale) for
# propose_remediation. REPORT-ONLY: this never executes anything.
_REMEDIATION_BY_LAYER: dict[str, tuple[str, str, str]] = {
    "hardware": (
        "apply_fan_policy",
        "fan-manager",
        "raise the fan curve for the thermally-stressed host",
    ),
    "orchestration": (
        "restart_or_cordon_pod",
        "container-manager-mcp",
        "restart the failing pod, or cordon/drain the node",
    ),
    "service": (
        "investigate_service_health",
        "lgtm-mcp",
        "latency/error-rate regression — investigate via lgtm-mcp/uptime-kuma-agent",
    ),
    "network": (
        "tunnel_failover",
        "tunnel-manager",
        "fail the network path over to a healthy tunnel",
    ),
}


def _layer_of(entity_id: str) -> str:
    prefix = (entity_id or "").split(":", 1)[0].lower()
    return _LAYER_PREFIXES.get(prefix, "unknown")


def _asset_key(entity_id: str) -> str:
    """The shared host/node slug an entity id concerns — its last ``:`` segment,
    so ``fan:host:r510`` and ``systems:host:r510`` join on ``r510``."""
    parts = (entity_id or "").split(":")
    return parts[-1] if parts and parts[-1] else (entity_id or "")


def _parse_ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return time.mktime(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _signature(asset: str, layers: list[str], signals: list[str]) -> str:
    """Stable dedupe key: same asset + same set of contributing layers/signals."""
    raw = f"{asset}|{','.join(sorted(set(layers)))}|{','.join(sorted(set(signals)))}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def _severity_for(layers: list[str]) -> str:
    return _MULTI_LAYER_SEVERITY if len(set(layers)) > 1 else DEFAULT_SEVERITY


def _root_cause_layer(layers: list[str]) -> str:
    known = [layer for layer in layers if layer in LAYER_DEPTH]
    if not known:
        return layers[0] if layers else "unknown"
    return min(known, key=lambda layer: LAYER_DEPTH[layer])


def _open_incident_signatures(engine: Any) -> set[str]:
    """Signatures of already-open ``:Incident`` nodes, for idempotent dedupe."""
    try:
        rows = engine.get_nodes_by_label("Incident", 0) or []
    except Exception as e:  # noqa: BLE001 — dedupe read is best-effort
        logger.debug("incident dedupe read failed: %s", e)
        return set()
    sigs: set[str] = set()
    for _id, props in rows:
        if not isinstance(props, dict):
            continue
        if str(props.get("status") or "open") != "open":
            continue
        sig = props.get("signature")
        if sig:
            sigs.add(str(sig))
    return sigs


def _notify(message: str) -> None:
    """Best-effort push to the operator notification endpoint
    (``INCIDENT_NOTIFY_URL``), mirroring ``systems_manager.os_health._notify`` /
    ``fan_manager.kg_control``'s notify convention."""
    from agent_utilities.core.config import setting

    logger.info(message)
    url = str(setting("INCIDENT_NOTIFY_URL", "") or "").strip()
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"source": "incident-brain", "message": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)  # noqa: S310  # nosec B310 — operator-configured URL
    except Exception as e:  # noqa: BLE001 — notification is best-effort
        logger.debug("incident notify skipped: %s", e)


def _synthesize_and_write(
    asset: str, cluster: list[dict[str, Any]], open_signatures: set[str]
) -> dict[str, Any]:
    """Build one incident dict from a time-clustered group of anomalies on
    ``asset`` and write it (unless an open incident with the same signature
    already exists)."""
    entity = str(cluster[0].get("entity"))
    layers = [_layer_of(str(a.get("entity") or "")) for a in cluster]
    signals = [str(a.get("signal")) for a in cluster if a.get("signal")]
    anomaly_ids = [str(a["id"]) for a in cluster]
    sig = _signature(asset, layers, signals)
    opened_at = min(a["_ts"] for a in cluster)
    root_cause_layer = _root_cause_layer(layers)
    severity = _severity_for(layers)
    pairs = sorted(
        {f"{layer}/{signal}" for layer, signal in zip(layers, signals, strict=False)}
    )
    summary = (
        f"{asset}: {' + '.join(pairs)} — correlated within "
        f"{len(cluster)} anomal{'y' if len(cluster) == 1 else 'ies'}"
    )

    incident: dict[str, Any] = {
        "id": f"health:incident:{asset}:{sig}",
        "kind": root_cause_layer,
        "entity": entity,
        "entities": sorted({str(a.get("entity")) for a in cluster}),
        "anomalies": anomaly_ids,
        "layers": sorted(set(layers)),
        "signals": sorted(set(signals)),
        "severity": severity,
        "root_cause_layer": root_cause_layer,
        "signature": sig,
        "status": "open",
        "summary": summary,
        "opened_at": _iso(opened_at),
    }

    if sig in open_signatures:
        incident["deduped"] = True
        return incident

    result = health_ingest.ingest_incident(incident)
    incident["written"] = result is not None
    open_signatures.add(sig)
    if result is not None:
        _notify(f"[incident-brain] opened {incident['id']} ({severity}): {summary}")
    return incident


def correlate_incidents(*, window_s: int = 300, days: int = 1) -> list[dict[str, Any]]:
    """One correlation pass: group recent cross-layer ``:HealthAnomaly`` rows into
    ``:Incident``s and write them.

    Reads every ``:HealthAnomaly`` node from the engine (via the
    :class:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine`
    read client, reused from :mod:`.health_ingest`), keeps the ones observed
    within ``days``, groups by :func:`_asset_key` (the shared host/node slug
    across producer namespaces), and — within each asset — clusters anomalies
    whose ``observedAt`` lands within ``window_s`` of the previous one already
    in the cluster. Each cluster becomes ONE synthesized incident (a cluster of
    size one is still a valid incident); anomalies on a different asset, or
    more than ``window_s`` apart on the same asset, land in separate incidents.
    Reuses :func:`agent_utilities.observability.health.correlate`'s systemic
    -collapse *pattern* one layer up — here the "entity" being collapsed is the
    time-window cluster, not a single anomaly kind.

    Idempotent: an already-open incident with the same entity+signature (see
    :func:`_signature`) is not duplicated — the returned dict for that cluster
    carries ``"deduped": True`` instead of being re-written.

    Best-effort: returns ``[]`` with no reachable engine.
    """
    engine = health_ingest._engine()
    if engine is None:
        return []
    try:
        rows = engine.get_nodes_by_label("HealthAnomaly", 0) or []
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.debug("incident correlation: anomaly read failed: %s", e)
        return []

    cutoff = time.time() - days * 86400
    anomalies: list[dict[str, Any]] = []
    for node_id, props in rows:
        if not isinstance(props, dict):
            continue
        ts = _parse_ts(props.get("observedAt"))
        if ts is None or ts < cutoff:
            continue
        entity = props.get("entity")
        if not entity:
            continue
        anomalies.append(
            {**props, "id": node_id, "_ts": ts, "_asset": _asset_key(str(entity))}
        )

    by_asset: dict[str, list[dict[str, Any]]] = {}
    for a in anomalies:
        by_asset.setdefault(a["_asset"], []).append(a)

    open_signatures = _open_incident_signatures(engine)
    incidents: list[dict[str, Any]] = []
    for asset, items in by_asset.items():
        items.sort(key=lambda a: a["_ts"])
        cluster: list[dict[str, Any]] = []
        for a in items:
            if cluster and a["_ts"] - cluster[-1]["_ts"] > window_s:
                incidents.append(_synthesize_and_write(asset, cluster, open_signatures))
                cluster = []
            cluster.append(a)
        if cluster:
            incidents.append(_synthesize_and_write(asset, cluster, open_signatures))
    return incidents


def _proposed_action(root_cause_layer: str, signals: list[str]) -> dict[str, str]:
    """Map an incident's root-cause layer (+ signal hints) to a REPORT-ONLY
    remediation action. os disk-fill gets its own mapping (systems-manager
    cleanup); every other os pressure falls back to "investigate"."""
    if root_cause_layer == "os":
        if any("disk" in s.lower() for s in signals):
            return {
                "action": "disk_cleanup",
                "package": "systems-manager",
                "detail": "reclaim disk space on the host (host-disk-reclaimer)",
            }
        return {
            "action": "investigate_os_pressure",
            "package": "systems-manager",
            "detail": "OS-layer resource pressure — investigate load/mem/CPU",
        }
    action, package, detail = _REMEDIATION_BY_LAYER.get(
        root_cause_layer,
        ("investigate", "", "unclassified root cause — manual investigation"),
    )
    return {"action": action, "package": package, "detail": detail}


# proposedAction -> ActionPolicy `kind` for the NARROW set of remediation
# proposals this module is willing to even OFFER to the actuation gate — safe,
# reversible, restart-class operations only. Every other proposedAction
# (fan-policy changes, tunnel failover, "investigate_*") has NO entry here and
# so :func:`actuate_remediation` refuses it outright before it ever reaches
# ActionPolicy; this table is the sole boundary of what "restart-class" means
# for this module — grow it deliberately, one safe/reversible kind at a time.
_SAFE_ACTUATION_KINDS: dict[str, str] = {
    "restart_or_cordon_pod": "restart_service",
}


def actuate_remediation(
    proposal: dict[str, Any],
    *,
    engine: Any = None,
    actuator: Any = None,
) -> dict[str, Any]:
    """propose → gate → (held) — the ONLY path by which a proposal can ever
    actuate, and only for the narrow, safe restart-class actions in
    :data:`_SAFE_ACTUATION_KINDS`.

    CONCEPT:AU-OS.host.report-only-remediation-proposal / CONCEPT:AU-OS.governance.action-policy-decision-point —
    this is additive plumbing composed 100% from the existing actuation seam
    (:mod:`agent_utilities.orchestration.action_policy` +
    :mod:`agent_utilities.orchestration.fleet_actuation`), NOT a new
    autonomy mechanism. Every call passes through the SAME fail-closed
    :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
    ``fleet_reconciler.py`` already gates restart/scale/deploy actions
    through — the shipped default tier for ``restart_service`` is
    ``approval_required``, so by default this call always resolves to
    ``"held"`` (a queued ``ActionApproval`` a human must grant), never
    ``"executed"``. It only executes when a policy rule *explicitly* allows
    the kind+target (``auto``/``auto_notify``) — exactly the same condition
    that lets any other autonomous fleet action run.

    A ``proposedAction`` outside :data:`_SAFE_ACTUATION_KINDS` never reaches
    the policy gate at all — returns ``{"status": "not_actuatable", ...}``.
    Best-effort: never raises; an internal error is reported as ``"error"``,
    never surfaced as a silent allow.
    """
    proposed_action = str(proposal.get("proposedAction") or "")
    kind = _SAFE_ACTUATION_KINDS.get(proposed_action)
    if kind is None:
        return {
            "status": "not_actuatable",
            "reason": f"{proposed_action or '(none)'} is not a safe restart-class action",
        }

    entity = str(proposal.get("entity") or "")
    target = _asset_key(entity) if entity else ""
    if not target:
        return {"status": "not_actuatable", "reason": "no target entity on proposal"}

    try:
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )
        from agent_utilities.orchestration.fleet_actuation import execute_action

        request = ActionRequest(
            kind=kind,
            target=target,
            source="incident-brain",
            reason=str(
                proposal.get("summary") or proposal.get("id") or "incident-brain"
            ),
            params={
                "incident": str(proposal.get("incident") or ""),
                "proposal": str(proposal.get("id") or ""),
            },
        )
        decision = get_action_policy(engine).decide(request)
    except Exception as e:  # noqa: BLE001 — the seam fails CLOSED, never silently allows
        logger.warning(
            "incident actuation: gate error for %s: %s", proposal.get("id"), e
        )
        return {"status": "error", "reason": str(e)}

    result: dict[str, Any] = {
        "status": "executed" if decision.allowed else "held",
        "decision": decision.decision,
        "tier": decision.tier,
        "approval_id": decision.approval_id,
        "reason": decision.reason,
        "action_kind": kind,
        "target": target,
    }
    if decision.allowed:
        result["execution"] = execute_action(engine, request, actuator)
        _notify(
            f"[incident-brain] actuated {kind}({target}) for proposal "
            f"{proposal.get('id')} — policy allowed ({decision.tier})"
        )
    else:
        _notify(
            f"[incident-brain] remediation HELD for {kind}({target}): "
            f"{decision.reason} (approval_id={decision.approval_id})"
        )
    return result


def propose_remediation(incident: dict[str, Any]) -> dict[str, Any] | None:
    """Map ``incident``'s root cause to a REPORT-ONLY remediation proposal.

    CONCEPT:AU-OS.host.report-only-remediation-proposal. Writes a
    ``:RemediationProposal`` node linked ``:proposesRemediation`` to the
    incident and best-effort-notifies ``INCIDENT_NOTIFY_URL`` — it NEVER
    executes anything. An operator later turns a proposal into action by
    wiring it onto the existing alert-bridge → ``graph_loops`` dispatch path
    (the same seam ``fleet_event_triage``/``remediation_playbooks`` already use
    for other subjects) once they are ready to enable autonomous actuation for
    that action kind; nothing in this module actuates on its own.

    Best-effort: returns ``None`` when the write fails (or no engine is
    reachable) — the caller (:func:`run_incident_correlation`) treats that as
    "not proposed", never as an error.
    """
    root_cause_layer = str(incident.get("root_cause_layer") or "unknown")
    signals = [str(s) for s in (incident.get("signals") or [])]
    mapped = _proposed_action(root_cause_layer, signals)
    incident_id = str(incident.get("id") or "")
    pid = f"health:remediation:{incident_id}"

    proposal = {
        "id": pid,
        "type": "RemediationProposal",
        "incident": incident_id,
        "proposedAction": mapped["action"],
        "targetPackage": mapped["package"],
        "summary": mapped["detail"],
        "rootCauseLayer": root_cause_layer,
        "status": "proposed",
        "observedAt": health_ingest._now(),
    }
    result = _write_remediation_proposal(proposal, incident_id=incident_id)
    if result is None:
        return None
    _notify(
        f"[incident-brain] proposal for {incident.get('entity')}: {mapped['action']} "
        f"({mapped['package'] or 'manual'}) — report-only, not executed"
    )
    return proposal


def _write_remediation_proposal(
    proposal: dict[str, Any], *, incident_id: str
) -> dict[str, int] | None:
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    relationships = (
        [
            {
                "source": proposal["id"],
                "target": incident_id,
                "type": "proposesRemediation",
            }
        ]
        if incident_id
        else []
    )
    return ingest_entities(
        [proposal], relationships, source="agent-utilities-health", domain="health"
    )


def run_incident_correlation(*, window_s: int = 300, days: int = 1) -> dict[str, Any]:
    """One correlate → write → route → propose-remediation pass.

    The CronJob / ``graph_loops`` tick entry point
    (``python -m agent_utilities.observability.incidents``). Ticket routing
    (:func:`agent_utilities.observability.incident_router.route_incident`) and
    remediation proposal are both best-effort per incident — a failure on one
    incident never stops the pass. Best-effort throughout: with no reachable
    engine this returns an all-zero summary rather than raising.

    Actuation (:func:`actuate_remediation`) is only even ATTEMPTED when
    ``INCIDENT_ACTUATION_ENABLED`` is truthy (default off) — with the flag
    off this function's behavior, including its returned summary shape, is
    byte-identical to before the actuator seam existed: propose-and-hold,
    report-only. With the flag on, every attempt still resolves through the
    fail-closed ActionPolicy gate, so the default OUTCOME stays "held"
    (see :func:`actuate_remediation`) — the flag only controls whether the
    tick offers eligible proposals to that gate at all.
    """
    from agent_utilities.core.config import setting

    actuation_enabled = bool(setting("INCIDENT_ACTUATION_ENABLED", False, cast=bool))
    engine = health_ingest._engine() if actuation_enabled else None

    incidents = correlate_incidents(window_s=window_s, days=days)
    routed = 0
    proposed = 0
    actuated = 0
    held = 0
    for incident in incidents:
        try:
            from agent_utilities.observability.incident_router import route_incident

            if route_incident(incident):
                routed += 1
        except Exception as e:  # noqa: BLE001 — routing must never break correlation
            logger.debug("incident routing failed for %s: %s", incident.get("id"), e)
        proposal = None
        try:
            proposal = propose_remediation(incident)
            if proposal is not None:
                proposed += 1
        except Exception as e:  # noqa: BLE001 — remediation must never break correlation
            logger.debug(
                "remediation proposal failed for %s: %s", incident.get("id"), e
            )
        if actuation_enabled and proposal is not None:
            try:
                outcome = actuate_remediation(proposal, engine=engine)
                if outcome.get("status") == "executed":
                    actuated += 1
                elif outcome.get("status") == "held":
                    held += 1
            except Exception as e:  # noqa: BLE001 — actuation must never break correlation
                logger.debug(
                    "remediation actuation failed for %s: %s", incident.get("id"), e
                )

    new = sum(1 for i in incidents if not i.get("deduped"))
    summary = {
        "incidents": len(incidents),
        "new": new,
        "deduped": len(incidents) - new,
        "routed": routed,
        "proposed": proposed,
    }
    # Only present when actuation was attempted this pass — keeps the summary
    # shape byte-identical to the pre-actuator-seam report-only default.
    if actuation_enabled:
        summary["actuated"] = actuated
        summary["held"] = held
    return summary


def main() -> None:
    """CLI (``python -m agent_utilities.observability.incidents``): one
    correlate→route→propose pass; prints a JSON summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    summary = run_incident_correlation()
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()
