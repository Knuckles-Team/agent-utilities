#!/usr/bin/python
from __future__ import annotations

"""Escalation-decision policy — the fuller-autonomy-with-escalation model
(``reports/autonomous-sdlc-loop-design.md`` §3.4, decision #3).

CONCEPT:AU-OS.host.report-only-remediation-proposal. The SDLC lifecycle loop
runs autonomously by default; :func:`evaluate_escalation` is consulted **before
every transition** the orchestrator is about to execute and decides whether a
human must be pulled in. It reads ONLY evidence that already exists in the graph
(no new instrumentation) across six signals, and — when any fires — emits a
queryable ``:EscalationRequest`` node linked to the entry node + the pending
transition, so "why did this need a human" is itself auditable. It returns the
request (or ``None`` when the loop may proceed autonomously).

**Report-only by design.** Emitting an ``:EscalationRequest`` records the
DECISION that a human is needed; it NEVER pauses or actuates on its own. The
lifecycle orchestrator carries the request as a consultable gate — it injects a
human-review gate ``:WorkflowStep`` (the suspend/resume gate of §7.1, see
:mod:`agent_utilities.workflows.runner`) at that transition so the run suspends
there rather than aborting, resuming when the request is resolved
(``graph_writeback action=approve`` / a watched ticket comment).

The six signals (each maps to already-available graph evidence, §3.4):

1. **Low agent confidence** — the bound run's ``:RunTrace`` ``status="degraded"``
   or a low ``graph_feedback`` reward on the capability.
2. **Large / risky diff** — files touched > N or ``graph_code(blast_radius)``
   fan-out over the changed symbols > M.
3. **Red CI past the retry cap** — a ``:PipelineRun`` still failed after the §5
   re-cycle loop exhausted its attempts.
4. **Governance/compliance gate requiring human sign-off** — a pending gate whose
   kind is ``owner_signoff``/``dpia``/``camunda_approval`` (always escalate).
5. **Critical-service blast radius** — the affected ``:Service``'s criticality
   (LeanIX capability tier) reaches a tier-0/critical CI.
6. **Novel / unseen transition** — no prior ``:RunTrace`` for this transition
   class on this service (cold-start caution).

Best-effort + engine-guarded: signals that need graph evidence degrade to
context-only hints when no engine is reachable; the request is written when an
engine is available and returned regardless. Never raises.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.observability import health_ingest

logger = logging.getLogger("agent_utilities.observability.escalation_policy")

_SOURCE = "agent-utilities-escalation"

# The gate kinds that ALWAYS escalate (compliance/governance human sign-off).
ALWAYS_ESCALATE_GATE_KINDS: frozenset[str] = frozenset(
    {"owner_signoff", "dpia", "camunda_approval"}
)


@dataclass(frozen=True)
class SignalHit:
    """One escalation signal that fired, with the evidence behind it."""

    signal: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# graph evidence reader (tiny, tolerant — mirrors lifecycle_orchestrator)
# --------------------------------------------------------------------------- #
@dataclass
class _GraphReader:
    engine: Any

    def node_props(self, node_id: str) -> dict[str, Any]:
        if not node_id or self.engine is None:
            return {}
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                data = graph.nodes[node_id]
                if data:
                    return dict(data)
            except Exception:  # noqa: BLE001 — fall through to backend
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (p) WHERE p.id = $pid RETURN p", {"pid": node_id}
                )
                if rows and isinstance(rows[0].get("p"), dict):
                    return dict(rows[0]["p"])
            except Exception:  # noqa: BLE001
                pass
        return {}

    def nodes_by_label(self, label: str) -> list[tuple[str, dict[str, Any]]]:
        if self.engine is None:
            return []
        try:
            return self.engine.get_nodes_by_label(label, 0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("escalation: get_nodes_by_label(%s) failed: %s", label, e)
            return []

    def out_edges(self, node_id: str) -> list[tuple[str, str]]:
        if self.engine is None:
            return []
        try:
            raw = self.engine.out_edges(node_id, data=True) or []
        except Exception:  # noqa: BLE001
            return []
        return [(_rel(props), tgt) for _s, tgt, props in raw]


def _rel(props: Any) -> str:
    if isinstance(props, dict):
        return str(props.get("rel_type") or props.get("type") or "")
    return ""


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# the six signal evaluators
# --------------------------------------------------------------------------- #
def _signal_low_confidence(
    reader: _GraphReader, ctx: dict[str, Any]
) -> SignalHit | None:
    """Degraded ``:RunTrace`` status, or a reward below the floor (§3.4 signal 1)."""
    reward_floor = _float(setting("ESCALATION_REWARD_FLOOR", "0.3"), 0.3)
    status = str(ctx.get("run_status") or "")
    reward = ctx.get("reward")
    run_trace = str(ctx.get("run_trace") or "")
    if not status and run_trace:
        props = reader.node_props(run_trace)
        status = str(props.get("status") or "")
        if reward is None:
            reward = props.get("reward")
    if status.lower() == "degraded":
        return SignalHit(
            "low_confidence",
            "bound run's RunTrace is degraded",
            {"run_trace": run_trace, "status": status},
        )
    if reward is not None and _float(reward, 1.0) < reward_floor:
        return SignalHit(
            "low_confidence",
            f"capability reward {reward} below floor {reward_floor}",
            {"reward": reward, "floor": reward_floor},
        )
    return None


def _signal_large_diff(reader: _GraphReader, ctx: dict[str, Any]) -> SignalHit | None:
    """Files touched > N, or blast-radius fan-out over changed symbols > M
    (§3.4 signal 2)."""
    file_cap = _int(setting("ESCALATION_DIFF_FILES", "10"), 10)
    fanout_cap = _int(setting("ESCALATION_BLAST_FANOUT", "20"), 20)
    n_files = _int(ctx.get("diff_files"), 0)
    if n_files > file_cap:
        return SignalHit(
            "large_diff",
            f"{n_files} files touched (> {file_cap})",
            {"diff_files": n_files, "cap": file_cap},
        )
    # Blast radius: count downstream :Code dependents of the changed symbols. The
    # caller may pre-resolve it (``blast_radius``), else we count out-edges of the
    # changed symbol nodes (best-effort).
    fanout = ctx.get("blast_radius")
    if fanout is None and ctx.get("changed_symbols"):
        total = 0
        for sym in ctx.get("changed_symbols") or []:
            total += len(
                [1 for rel, _t in reader.out_edges(str(sym)) if rel in ("calls", "depends_on")]
            )
        fanout = total
    if fanout is not None and _int(fanout, 0) > fanout_cap:
        return SignalHit(
            "large_diff",
            f"blast-radius fan-out {fanout} (> {fanout_cap})",
            {"blast_radius": fanout, "cap": fanout_cap},
        )
    return None


def _signal_red_ci_past_cap(
    reader: _GraphReader, ctx: dict[str, Any]
) -> SignalHit | None:
    """A ``:PipelineRun`` still failing after the re-cycle retry cap (§3.4 signal 3)."""
    retry_cap = _int(setting("ESCALATION_CI_RETRY_CAP", "3"), 3)
    attempts = _int(ctx.get("attempts"), 0)
    status = str(ctx.get("pipeline_status") or "")
    pr = str(ctx.get("pipeline_run") or "")
    if not status and pr:
        status = str(reader.node_props(pr).get("status") or "")
    if status.lower() in ("failed", "red", "error") and attempts >= retry_cap:
        return SignalHit(
            "red_ci_past_cap",
            f"CI still {status} after {attempts} attempts (cap {retry_cap})",
            {"pipeline_run": pr, "attempts": attempts, "cap": retry_cap},
        )
    return None


def _signal_governance_gate(
    _reader: _GraphReader, ctx: dict[str, Any]
) -> SignalHit | None:
    """A pending compliance/governance gate — ALWAYS escalate (§3.4 signal 4)."""
    gate_kind = str(ctx.get("gate_kind") or "").lower()
    if gate_kind in ALWAYS_ESCALATE_GATE_KINDS:
        return SignalHit(
            "governance_gate",
            f"gate '{gate_kind}' requires human sign-off",
            {"gate_kind": gate_kind},
        )
    return None


# criticality strings/tiers that count as a tier-0 / critical service.
_CRITICAL_TIERS: frozenset[str] = frozenset(
    {"0", "tier-0", "tier0", "critical", "mission_critical", "mission-critical"}
)


def _signal_critical_blast_radius(
    reader: _GraphReader, ctx: dict[str, Any]
) -> SignalHit | None:
    """The affected service is tier-0/critical (LeanIX capability tier, §3.4 signal 5)."""
    crit = str(ctx.get("service_criticality") or "").lower()
    service = str(ctx.get("service") or "")
    if not crit and service:
        props = reader.node_props(service)
        crit = str(
            props.get("criticality")
            or props.get("tier")
            or props.get("businessCriticality")
            or ""
        ).lower()
    if crit and crit in _CRITICAL_TIERS:
        return SignalHit(
            "critical_blast_radius",
            f"affected service is criticality '{crit}'",
            {"service": service, "criticality": crit},
        )
    return None


def _signal_cold_start(reader: _GraphReader, ctx: dict[str, Any]) -> SignalHit | None:
    """No prior ``:RunTrace`` for this transition class on this service (§3.4 signal 6).

    Caller may assert ``cold_start=True`` directly; otherwise we scan prior
    ``:RunTrace`` rows for a matching ``transition``+``service`` (best-effort —
    with no engine or no history this is treated as a cold start only when the
    caller explicitly opts in, never inferred silently from an empty graph)."""
    if ctx.get("cold_start") is True:
        return SignalHit(
            "cold_start",
            "first occurrence of this transition on this service",
            {"transition": ctx.get("transition"), "service": ctx.get("service")},
        )
    transition = str(ctx.get("transition") or "")
    service = str(ctx.get("service") or "")
    # Only infer from graph when the caller both named the pair AND enabled the
    # scan — an empty/unreachable graph must not manufacture a cold-start.
    if not (transition and service and ctx.get("scan_history")):
        return None
    for _id, props in reader.nodes_by_label("RunTrace"):
        if not isinstance(props, dict):
            continue
        if (
            str(props.get("transition") or "") == transition
            and str(props.get("service") or "") == service
        ):
            return None  # prior run exists — not novel
    return SignalHit(
        "cold_start",
        "no prior RunTrace for this transition on this service",
        {"transition": transition, "service": service},
    )


_EVALUATORS = (
    _signal_low_confidence,
    _signal_large_diff,
    _signal_red_ci_past_cap,
    _signal_governance_gate,
    _signal_critical_blast_radius,
    _signal_cold_start,
)


def _signature(entry_id: str, transition: str, signals: list[str]) -> str:
    raw = f"{entry_id}|{transition}|{'+'.join(sorted(signals))}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def evaluate_escalation(
    context: dict[str, Any], *, engine: Any | None = None, write: bool = True
) -> dict[str, Any] | None:
    """Decide whether a pending transition must escalate to a human (§3.4).

    ``context`` describes the transition the orchestrator is about to run::

        {
            "entry": "<spine node id>",     # the :Incident/:Ticket/:Spec/:MR
            "transition": "develop_spec",   # the transition name/class
            "service": "<service node id>", # affected service (optional)
            "run_trace": "<run trace id>",  # bound agent run (optional)
            "run_status": "degraded",       # or read from run_trace
            "reward": 0.2,                  # capability reward (optional)
            "diff_files": 14,               # diff size (optional)
            "changed_symbols": [...],       # for blast radius (optional)
            "blast_radius": 33,             # pre-resolved fan-out (optional)
            "pipeline_run": "<id>", "pipeline_status": "failed", "attempts": 3,
            "gate_kind": "dpia",            # a pending compliance gate (optional)
            "service_criticality": "tier-0",
            "cold_start": True,             # or scan_history=True to infer
        }

    Returns the emitted ``:EscalationRequest`` (a plain dict) when ANY signal
    fires, else ``None`` (the loop proceeds autonomously). Report-only — writing
    the request never actuates anything; with no engine it is returned unwritten.
    """
    eng = engine if engine is not None else health_ingest._engine()
    reader = _GraphReader(engine=eng)

    hits: list[SignalHit] = []
    for evaluate in _EVALUATORS:
        try:
            hit = evaluate(reader, context)
        except Exception as e:  # noqa: BLE001 — one signal must not break the policy
            logger.debug("escalation signal %s failed: %s", evaluate.__name__, e)
            hit = None
        if hit is not None:
            hits.append(hit)

    if not hits:
        return None

    entry_id = str(context.get("entry") or "")
    transition = str(context.get("transition") or "")
    signal_names = [h.signal for h in hits]
    sig = _signature(entry_id, transition, signal_names)
    request = {
        "id": f"escalation:{entry_id}:{sig}" if entry_id else f"escalation:{sig}",
        "type": "EscalationRequest",
        "entry": entry_id,
        "transition": transition,
        "service": str(context.get("service") or ""),
        "signals": signal_names,
        "reasons": [h.reason for h in hits],
        "evidence_json": json.dumps([h.evidence for h in hits], default=str),
        "autonomy": "escalate",
        "risk_tier": "high_stakes",  # per-transition output (design §3.1)
        "status": "open",
        "signature": sig,
        "observedAt": health_ingest._now(),
    }

    if write and eng is not None:
        _write_escalation(request, entry_id)
    return request


def _write_escalation(
    request: dict[str, Any], entry_id: str
) -> dict[str, int] | None:
    """MERGE the ``:EscalationRequest`` + an ``escalates`` edge from the entry node.

    Best-effort — a failed write returns ``None`` and the request is still
    returned to the caller (report-only never blocks on persistence)."""
    try:
        from agent_utilities.knowledge_graph.memory.native_ingest import (
            ingest_entities,
        )

        relationships = (
            [{"source": entry_id, "target": request["id"], "type": "escalates"}]
            if entry_id
            else []
        )
        return ingest_entities(
            [request], relationships, source=_SOURCE, domain="sdlc"
        )
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        logger.debug("escalation write skipped: %s", e)
        return None
