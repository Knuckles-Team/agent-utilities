#!/usr/bin/python
from __future__ import annotations

"""Operational health diagnosis over the live task/lane/queue state (CONCEPT:AU-KG.retrieval.ops-context).

The ``ops`` provider of the context plane (KG-2.136): answers *"is the system
healthy / why is the maint lane backing up / what's blocked?"* by synthesizing the
KG's own operational data — :Task nodes, their status/lane/kind, the dead-letter
and failed backlog — into one grounded answer with task/lane citations and a
remediation hint. Pure Cypher reads (best-effort, never raises) so a degraded
backend still yields a partial picture instead of a crash.

Intents: ``health`` (whole-queue overview, default), ``why`` (focus a lane/symptom
named in the question), ``impact`` (what's stuck/poisoned).
"""

from typing import Any

from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

VALID_INTENTS = ("health", "why", "impact")

# Lanes the scheduler stamps (CONCEPT:AU-ORCH.execution.two-level-fair-rotation/1.76/1.77) — used to focus a
# "why is the <lane> lane …" question without importing the scheduler.
_KNOWN_LANES = ("queries", "ingestion", "connectors", "research", "extraction", "maint")

# A lane is "backing up" when work is queued but almost nothing is being claimed.
_BACKLOG_PENDING = 20
_BACKLOG_RUNNING = 2


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _focus_lane(query: str) -> str:
    low = (query or "").lower()
    for lane in _KNOWN_LANES:
        if lane in low:
            return lane
    return ""


def diagnose_ops(
    engine: Any,
    *,
    query: str = "",
    intent: str = "health",
    top_k: int = 10,
    **_opts: Any,
) -> dict[str, Any]:
    """Synthesize task-queue health into a cited answer (see module docstring)."""
    intent = (intent or "health").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "health"
    limit = max(1, min(50, int(top_k)))

    status_counts = {
        str(r.get("status")): _as_int(r.get("n"))
        for r in read_rows(
            engine, "MATCH (t:Task) RETURN t.status AS status, count(t) AS n", {}
        )
    }
    active = read_rows(
        engine,
        "MATCH (t:Task) WHERE t.status IN ['pending','running'] "
        "RETURN t.lane AS lane, t.status AS status, count(t) AS n",
        {},
    )
    broken = read_rows(
        engine,
        "MATCH (t:Task) WHERE t.status IN ['failed','dead_letter'] "
        "RETURN t.lane AS lane, t.status AS status, count(t) AS n",
        {},
    )
    dead_sample = read_rows(
        engine,
        "MATCH (t:Task) WHERE t.status = 'dead_letter' "
        "RETURN t.id AS id, t.lane AS lane, t.tkind AS tkind LIMIT $k",
        {"k": limit},
    )

    # Per-lane roll-up.
    lanes: dict[str, dict[str, int]] = {}
    for r in active:
        lane = str(r.get("lane") or "unlaned")
        lanes.setdefault(lane, {})[str(r.get("status"))] = _as_int(r.get("n"))
    for r in broken:
        lane = str(r.get("lane") or "unlaned")
        lanes.setdefault(lane, {})[str(r.get("status"))] = _as_int(r.get("n"))

    # Signals — the actionable findings.
    signals: list[dict[str, Any]] = []
    for lane, st in lanes.items():
        pending, running = st.get("pending", 0), st.get("running", 0)
        if pending >= _BACKLOG_PENDING and running <= _BACKLOG_RUNNING:
            signals.append(
                {
                    "lane": lane,
                    "kind": "backing_up",
                    "detail": f"{pending} pending vs {running} running — workers not keeping pace",
                }
            )
    dead_total = status_counts.get("dead_letter", 0)
    failed_total = status_counts.get("failed", 0)
    if dead_total:
        signals.append(
            {
                "kind": "dead_letter",
                "detail": f"{dead_total} poison task(s) exhausted retries (dead_letter)",
            }
        )
    if failed_total >= 50:
        signals.append(
            {
                "kind": "failed",
                "detail": f"{failed_total} failed task(s) — check the engine breaker",
            }
        )

    focus = _focus_lane(query) if intent != "health" else ""
    citations: list[dict[str, Any]] = [
        {"type": "lane", "id": lane, **st} for lane, st in sorted(lanes.items())
    ]
    citations += [
        {
            "type": "task",
            "id": r.get("id"),
            "lane": r.get("lane"),
            "kind": r.get("tkind"),
        }
        for r in dead_sample
    ]

    answer = _synthesize(status_counts, lanes, signals, focus, intent)
    used = ["task_status", "lane_rollup"]
    if dead_sample:
        used.append("dead_letter_sample")
    return {
        "status": "ok",
        "domain": "ops",
        "intent": intent,
        "query": query,
        "answer": answer,
        "citations": citations,
        "sections": {
            "lanes": citations[: len(lanes)],
            "signals": signals,
            "status_counts": [status_counts],
        },
        "capability_id": f"ops:{intent}:{focus or 'queue'}",
        "used_primitives": used,
    }


def _synthesize(
    status_counts: dict[str, int],
    lanes: dict[str, dict[str, int]],
    signals: list[dict[str, Any]],
    focus: str,
    intent: str,
) -> str:
    total = sum(status_counts.values())
    head = (
        f"Task queue: {status_counts.get('pending', 0)} pending, "
        f"{status_counts.get('running', 0)} running, "
        f"{status_counts.get('dead_letter', 0)} dead-lettered, "
        f"{status_counts.get('failed', 0)} failed, "
        f"{status_counts.get('completed', 0)} completed (of {total})."
    )
    if focus and focus in lanes:
        st = lanes[focus]
        parts = [
            f"Lane '{focus}': "
            + ", ".join(f"{k}={v}" for k, v in sorted(st.items()))
            + "."
        ]
        sig = [s for s in signals if s.get("lane") == focus]
        if sig:
            parts.append(sig[0]["detail"] + ".")
            parts.append(
                "Likely cause: the worker pool is saturated or the engine circuit "
                "breaker is creeping open — a clean graph-os-host restart resets the "
                "breaker and resumes draining."
            )
        else:
            parts.append("No backlog signal on this lane.")
        return head + " " + " ".join(parts)

    lane_summ = "; ".join(
        f"{lane}({', '.join(f'{k}={v}' for k, v in sorted(st.items()))})"
        for lane, st in sorted(lanes.items())
    )
    out = [head, f"Lanes: {lane_summ}." if lane_summ else "No active lanes."]
    if signals:
        out.append("Signals: " + "; ".join(s["detail"] for s in signals) + ".")
        if any(s.get("kind") == "backing_up" for s in signals):
            out.append(
                "Remediation: a creeping engine breaker starves the worker pool — a "
                "clean graph-os-host restart resets it; inspect poison tasks via "
                "graph_query on status='dead_letter'."
            )
    else:
        out.append("No backlog or failure signals — queue is healthy.")
    return " ".join(out)
