#!/usr/bin/python
from __future__ import annotations

"""Live, legible state of the 24/7 self-evolution flywheel — transparent + steerable.

CONCEPT:KG-2.290 — EvolutionState live surface + per-stage progress beacon. The
deployed loop (``loop_controller.run_one_cycle``) only persisted an ``EvolutionCycle``
node at *finalize* — so until a cycle finished it was opaque, and an operator could
not see "what is it mining / distilling / developing right now, and why". This module
adds a single mutable :class:`StageBeacon` node the controller stamps at each stage
entry, plus :func:`read_evolution_state` which aggregates the live beacon + the
signals that already exist into ONE queryable read. You cannot steer what you cannot
see; this is the observation plane every steering action hangs off.

CONCEPT:KG-2.291 — Saturation gauge. Aggregates the four signals that already exist
(``open_gaps`` trend per cycle, the AHE-3.26 ``ImprovementVelocity`` verdict, the
OS-5.47 ``ingestion_coverage`` %, and the distilled-spec backlog yield) into ONE 0..1
reading: "how much have we extracted from the current corpus / are we saturated".
When saturated (gauge high + velocity stalling) it surfaces a *recommendation* to
request more research/codebases — it does NOT auto-fetch (acquisition stays a
steerable decision for Claude/the human).

Read-only + best-effort throughout: a missing engine / unsupported query yields an
empty-but-shaped reading and never raises, so the surface is safe on any backend.
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: The singleton live-beacon node — upserted in place at every stage entry so a
#: "what is the loop doing now" read is O(1) (one node, not a scan).
BEACON_NODE_ID = "evolution:beacon"
BEACON_LABEL = "EvolutionBeacon"

#: Saturation threshold above which (with a stalling velocity) the gauge recommends
#: acquiring more corpus. Tunable; deliberately conservative so "request more" is a
#: considered signal, not noise.
SATURATION_THRESHOLD = 0.66

SATURATION_SIGNAL_LABEL = "EvolutionSaturationSignal"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ────────────────────────────────────────────────────────────────────────
# The live per-stage progress beacon (KG-2.290)
# ────────────────────────────────────────────────────────────────────────
class StageBeacon:
    """A single mutable node tracking the cycle's CURRENT stage + rationale.

    One :class:`StageBeacon` is created per ``run_one_cycle`` and ``enter()`` is
    called at each stage boundary, upserting the singleton ``EvolutionBeacon`` node
    so that — mid-cycle — ``graph_loops(action="state")`` reports exactly which loop
    and stage is live, what it is acting on, and why (the ``open_gaps`` / topics it
    is mining). Every write is best-effort: instrumentation never aborts the cycle.
    """

    def __init__(
        self,
        engine: Any,
        *,
        cycle_id: str,
        loops_active: int = 0,
        triggered_by: str = "loop_engine",
        why: str = "",
    ) -> None:
        self.engine = engine
        self.cycle_id = cycle_id
        self.loops_active = loops_active
        self.triggered_by = triggered_by
        self.why = why
        self.stage = "starting"
        self.started_at = _now_iso()

    def _write(self, *, status: str, extra: dict[str, Any] | None = None) -> None:
        if self.engine is None:
            return
        meta = {
            "cycle_id": self.cycle_id,
            "stage": self.stage,
            "loops_active": self.loops_active,
            "triggered_by": self.triggered_by,
            "why": self.why,
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            **(extra or {}),
        }
        try:
            self.engine.add_node(
                BEACON_NODE_ID,
                BEACON_LABEL,
                properties={
                    "name": f"evolution cycle {self.cycle_id}",
                    "status": status,
                    "stage": self.stage,
                    "timestamp": meta["updated_at"],
                    "metadata": json.dumps(meta, default=str),
                },
            )
        except Exception as e:  # noqa: BLE001 — beacon is observability only
            logger.debug("StageBeacon write failed: %s", e)

    def enter(self, stage: str, *, detail: str = "", why: str = "") -> None:
        """Record that the cycle is now in ``stage`` (optionally what/why)."""
        self.stage = stage
        if why:
            self.why = why
        self._write(status="running", extra={"detail": detail} if detail else None)

    def finish(self, *, open_gaps: Any = 0, errors: int = 0, saturation: Any = None) -> None:
        """Mark the cycle complete and stamp the closing summary on the beacon."""
        self.stage = "idle"
        extra: dict[str, Any] = {
            "open_gaps": open_gaps,
            "error_count": errors,
            "finished_at": _now_iso(),
        }
        if saturation is not None:
            extra["saturation"] = saturation
        self._write(status="idle", extra=extra)


def read_beacon(engine: Any) -> dict[str, Any]:
    """Read the live beacon node → the cycle's current stage + rationale."""
    if engine is None:
        return {}
    try:
        rows = engine.query_cypher(
            "MATCH (n:EvolutionBeacon) RETURN n.status AS status, "
            "n.stage AS stage, n.timestamp AS timestamp, n.metadata AS metadata LIMIT 1"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("read_beacon query failed: %s", e)
        return {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        out: dict[str, Any] = {
            "status": r.get("status"),
            "stage": r.get("stage"),
            "updated_at": r.get("timestamp"),
        }
        try:
            meta = json.loads(r.get("metadata") or "{}")
            if isinstance(meta, dict):
                out.update(meta)
        except (TypeError, ValueError):
            pass
        return out
    return {}


# ────────────────────────────────────────────────────────────────────────
# The saturation gauge (KG-2.291)
# ────────────────────────────────────────────────────────────────────────
def _open_gaps_trend(engine: Any, *, window: int = 8) -> dict[str, Any]:
    """Recent per-cycle ``open_gaps`` series from the EvolutionCycle audit nodes."""
    if engine is None:
        return {"series": [], "recent": 0, "prior": 0}
    try:
        rows = engine.query_cypher(
            "MATCH (n:EvolutionCycle) RETURN n.created_at AS ts, n.metadata AS metadata"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("_open_gaps_trend query failed: %s", e)
        return {"series": [], "recent": 0, "prior": 0}
    series: list[int] = []
    for r in sorted(
        (r for r in (rows or []) if isinstance(r, dict)),
        key=lambda r: str(r.get("ts") or ""),
    ):
        try:
            meta = json.loads(r.get("metadata") or "{}")
            series.append(int(meta.get("open_gaps", 0) or 0))
        except (TypeError, ValueError):
            continue
    tail = series[-window:]
    half = max(1, len(tail) // 2)
    recent = round(sum(tail[-half:]) / half, 1) if tail else 0
    prior_slice = tail[:-half] or tail
    prior = round(sum(prior_slice) / len(prior_slice), 1) if prior_slice else recent
    return {"series": tail, "recent": recent, "prior": prior}


def _ingestion_coverage_pct() -> dict[str, Any]:
    """Best-effort OS-5.47 ingestion-coverage % (how much of OUR corpus is ingested)."""
    try:
        from agent_utilities.deployment.doctor import _check_ingestion_coverage

        res = _check_ingestion_coverage()
        data = res.get("data") if isinstance(res, dict) else None
        pct = float((data or {}).get("coverage_pct", 0.0)) if data else None
        return {
            "status": res.get("status") if isinstance(res, dict) else "skip",
            "coverage_pct": pct,
            "detail": res.get("detail") if isinstance(res, dict) else "",
        }
    except Exception as e:  # noqa: BLE001 — coverage probe is best-effort
        logger.debug("_ingestion_coverage_pct failed: %s", e)
        return {"status": "skip", "coverage_pct": None, "detail": str(e)}


def saturation_gauge(
    *,
    coverage_pct: float | None,
    velocity_verdict: str,
    gaps_recent: float,
    gaps_prior: float,
    pending_specs: int = 0,
) -> dict[str, Any]:
    """Aggregate the existing signals into ONE 0..1 saturation reading.

    Components (each 0..1, higher = more saturated):

    - ``velocity`` — the AHE-3.26 verdict: ``stalling`` → 1.0 (improvement plateaued),
      ``steady`` → 0.5, ``improving``/``idle`` → 0.0.
    - ``coverage`` — the OS-5.47 ingestion coverage fraction (our corpus is ingested,
      so there is little *un-mined* local material left). Unknown ⇒ neutral 0.5.
    - ``gaps`` — the ``open_gaps`` trend: still shrinking fast ⇒ low (we are actively
      extracting value); flat/rising ⇒ high (nothing new is being found).

    ``request_more`` fires only when the gauge is high AND velocity is stalling — a
    *recommendation* to acquire more research/codebases, never an auto-fetch.
    """
    verdict = (velocity_verdict or "idle").lower()
    velocity_c = {"stalling": 1.0, "steady": 0.5}.get(verdict, 0.0)

    coverage_c = 0.5 if coverage_pct is None else max(0.0, min(1.0, coverage_pct / 100.0))

    if gaps_prior > 0:
        shrink = max(0.0, (gaps_prior - gaps_recent) / gaps_prior)
        gaps_c = max(0.0, min(1.0, 1.0 - shrink))
    else:
        # No prior gaps to compare → if there are still open gaps we are NOT
        # saturated (work remains); if there are none, neutral.
        gaps_c = 0.5 if gaps_recent == 0 else 0.3

    gauge = round(0.4 * velocity_c + 0.3 * coverage_c + 0.3 * gaps_c, 4)
    saturated = gauge >= SATURATION_THRESHOLD
    request_more = bool(saturated and verdict == "stalling")
    recommendation = ""
    if request_more:
        recommendation = (
            "Saturation high and improvement stalling — the current corpus looks "
            "mined out. Recommend acquiring more material: enable discovery "
            "(graph_schedules run_now self_evolution with KG_LOOP_DISCOVER, or "
            "graph_research action=background_research) and/or clone more codebases "
            "into the breadth corpus. Acquisition is left to you/Claude (not auto-run)."
        )
    return {
        "gauge": gauge,
        "threshold": SATURATION_THRESHOLD,
        "saturated": saturated,
        "request_more": request_more,
        "recommendation": recommendation,
        "components": {
            "velocity": round(velocity_c, 4),
            "coverage": round(coverage_c, 4),
            "gaps": round(gaps_c, 4),
        },
        "inputs": {
            "velocity_verdict": verdict,
            "coverage_pct": coverage_pct,
            "gaps_recent": gaps_recent,
            "gaps_prior": gaps_prior,
            "pending_specs": pending_specs,
        },
    }


def emit_saturation_signal(engine: Any, gauge: dict[str, Any]) -> str | None:
    """Persist a queryable ``EvolutionSaturationSignal`` (request-more recommendation).

    Surfaced — never acted on — so Claude/the human can decide to acquire more.
    Best-effort; returns the node id or ``None``.
    """
    if engine is None or not gauge.get("request_more"):
        return None
    import uuid

    sig_id = f"evolution_saturation:{uuid.uuid4().hex[:10]}"
    try:
        engine.add_node(
            sig_id,
            SATURATION_SIGNAL_LABEL,
            properties={
                "name": "request more corpus (saturation)",
                "status": "open",
                "timestamp": _now_iso(),
                "metadata": json.dumps(gauge, default=str),
            },
        )
    except Exception as e:  # noqa: BLE001 — best-effort surfacing
        logger.debug("emit_saturation_signal failed: %s", e)
        return None
    return sig_id


# ────────────────────────────────────────────────────────────────────────
# The aggregated read surface (KG-2.290)
# ────────────────────────────────────────────────────────────────────────
def read_evolution_state(
    engine: Any, *, include_coverage: bool = True
) -> dict[str, Any]:
    """ONE legible read of the evolution: current stage, signals, gauge, spec backlog.

    The operator's "make it transparent so I can steer" surface. Reused by the
    ``graph_loops(action="state")`` MCP tool and its REST twin.
    """
    from .improvement_ledger import improvement_velocity
    from .spec_proposals import specs_summary

    beacon = read_beacon(engine)
    try:
        velocity = improvement_velocity(engine)
    except Exception as e:  # noqa: BLE001
        logger.debug("velocity read failed: %s", e)
        velocity = {"verdict": "idle"}
    gaps = _open_gaps_trend(engine)
    coverage = _ingestion_coverage_pct() if include_coverage else {"coverage_pct": None}
    specs = specs_summary(engine)

    gauge = saturation_gauge(
        coverage_pct=coverage.get("coverage_pct"),
        velocity_verdict=str(velocity.get("verdict", "idle")),
        gaps_recent=float(gaps.get("recent", 0) or 0),
        gaps_prior=float(gaps.get("prior", 0) or 0),
        pending_specs=int(specs.get("counts", {}).get("pending_review", 0) or 0),
    )

    return {
        "beacon": beacon,
        "velocity": velocity,
        "open_gaps": gaps,
        "ingestion_coverage": coverage,
        "specs": specs,
        "saturation": gauge,
        "steering": {
            "pause": "graph_schedules action=disable name=<evolution schedule>",
            "reprioritize": "graph_loops action=prioritize loop_id=<id> priority=high",
            "review_spec": "graph_loops action=review spec_id=<id> decision=approve|reject|edit",
            "request_more": gauge.get("recommendation") or "(not saturated)",
        },
    }


__all__ = [
    "BEACON_NODE_ID",
    "BEACON_LABEL",
    "SATURATION_THRESHOLD",
    "StageBeacon",
    "read_beacon",
    "saturation_gauge",
    "emit_saturation_signal",
    "read_evolution_state",
]
