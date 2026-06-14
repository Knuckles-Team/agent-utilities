#!/usr/bin/python
from __future__ import annotations

"""Recursive-improvement velocity ledger.

CONCEPT:AHE-3.26 — recursive-improvement instrumentation aggregating per-cycle proposal merge and capability-delta metrics so the self-evolution loop can measure its own rate and per-mechanism contribution
CONCEPT:SAFE-1.3 — recursive-improvement velocity tracker that surfaces whether the loop is still improving and flags a non-positive derivative as a research-gets-harder signal

The deployed loop already *writes* an audit trail every cycle — ``EvolutionCycle``
nodes (`loop_controller._finalize_metrics`), ``ProposalPublication`` nodes
(`change_publisher`, with ``kind`` = ``code``/``sdd_plan``), and
``CapabilityRatchetResult`` nodes (`capability_ratchet`, AHE-3.24) — but **nothing
reads them back**, so AU cannot answer "is the loop improving us, how fast, and
which mechanism contributes most" (the paper's dominant forecasting uncertainty).

This module is the missing consumer: a read-only aggregator over those persisted
nodes that produces a velocity summary (cycle cadence, proposal/publication counts,
the genotypic-vs-prose mechanism split, capability pass-rate, and a simple
improving/stalling verdict from the recent-vs-prior trend). The golden-loop tick
records one ``ImprovementVelocity`` node per cycle so the series itself is queryable
(and Grafana-able) — the loop self-instruments without any new write path.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImprovementVelocity:
    """A point-in-time read of the self-evolution loop's own dynamics."""

    cycles: int = 0
    recent_cycle_ms: float = 0.0
    prior_cycle_ms: float = 0.0
    publications: int = 0
    publications_ok: int = 0
    code_publications: int = 0
    prose_publications: int = 0
    capability_pass: int = 0
    capability_hold: int = 0
    verdict: str = "idle"  # idle | improving | steady | stalling
    signals: list[str] = field(default_factory=list)

    @property
    def code_fraction(self) -> float:
        total = self.code_publications + self.prose_publications
        return self.code_publications / total if total else 0.0

    @property
    def capability_pass_rate(self) -> float:
        total = self.capability_pass + self.capability_hold
        return self.capability_pass / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "recent_cycle_ms": self.recent_cycle_ms,
            "prior_cycle_ms": self.prior_cycle_ms,
            "publications": self.publications,
            "publications_ok": self.publications_ok,
            "code_publications": self.code_publications,
            "prose_publications": self.prose_publications,
            "code_fraction": round(self.code_fraction, 4),
            "capability_pass": self.capability_pass,
            "capability_hold": self.capability_hold,
            "capability_pass_rate": round(self.capability_pass_rate, 4),
            "verdict": self.verdict,
            "signals": self.signals,
        }


class ImprovementLedger:
    """Read-only aggregator over the loop's own persisted audit nodes."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _rows(self, query: str) -> list[dict[str, Any]]:
        if self.engine is None:
            return []
        try:
            rows = self.engine.query_cypher(query)
        except Exception as exc:  # noqa: BLE001 — no engine/query support ⇒ empty
            logger.debug("[AHE-3.26] ledger query failed: %s", exc)
            return []
        return [r for r in (rows or []) if isinstance(r, dict)]

    @staticmethod
    def _cycle_ms(metadata: Any) -> float:
        try:
            return float(json.loads(metadata).get("duration_ms", 0.0))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def summarize(self, *, recent: int = 5) -> ImprovementVelocity:
        """Aggregate the persisted audit streams into a velocity reading.

        ``recent`` sets the trailing window (in cycles) used for the
        improving/stalling derivative.
        """
        cycles = sorted(
            self._rows(
                "MATCH (n:EvolutionCycle) RETURN n.created_at AS ts, n.metadata AS metadata"
            ),
            key=lambda r: str(r.get("ts") or ""),
        )
        pubs = self._rows(
            "MATCH (n:ProposalPublication) RETURN n.kind AS kind, n.ok AS ok, n.published_at AS ts"
        )
        ratchet = self._rows(
            "MATCH (n:CapabilityRatchetResult) RETURN n.result AS result, n.recorded_at AS ts"
        )

        v = ImprovementVelocity(cycles=len(cycles))
        if cycles:
            window = cycles[-recent:]
            prior = cycles[-2 * recent : -recent]
            v.recent_cycle_ms = round(
                sum(self._cycle_ms(c.get("metadata")) for c in window) / len(window), 1
            )
            if prior:
                v.prior_cycle_ms = round(
                    sum(self._cycle_ms(c.get("metadata")) for c in prior) / len(prior),
                    1,
                )

        for p in pubs:
            v.publications += 1
            if p.get("ok") in (True, "True", "true", 1):
                v.publications_ok += 1
            if str(p.get("kind")) == "code":
                v.code_publications += 1
            else:
                v.prose_publications += 1

        for r in ratchet:
            if str(r.get("result")).lower() == "pass":
                v.capability_pass += 1
            elif str(r.get("result")).lower() == "hold":
                v.capability_hold += 1

        v.verdict, v.signals = self._verdict(v, cycles, ratchet, recent)
        return v

    @staticmethod
    def _verdict(
        v: ImprovementVelocity,
        cycles: list[dict[str, Any]],
        ratchet: list[dict[str, Any]],
        recent: int,
    ) -> tuple[str, list[str]]:
        signals: list[str] = []
        if v.cycles == 0:
            return "idle", signals
        # Genotypic-RSI signal: is the loop emitting code, or only prose?
        if v.code_publications == 0 and v.publications > 0:
            signals.append("prose-only: no genotypic (code) changes emitted")
        # Capability-pressure signal: recent holds outnumber passes ⇒ regressions rising.
        recent_ratchet = sorted(ratchet, key=lambda r: str(r.get("ts") or ""))[-recent:]
        holds = sum(1 for r in recent_ratchet if str(r.get("result")).lower() == "hold")
        passes = sum(
            1 for r in recent_ratchet if str(r.get("result")).lower() == "pass"
        )
        if holds > passes and recent_ratchet:
            signals.append("capability regressions outnumber confirmations recently")
        # Cadence signal: cycles getting markedly slower ⇒ research-gets-harder.
        slowing = v.prior_cycle_ms > 0 and v.recent_cycle_ms > v.prior_cycle_ms * 1.5
        if slowing:
            signals.append("cycle duration rising sharply (research-gets-harder)")

        if signals:
            return "stalling", signals
        if v.publications_ok > 0 or v.capability_pass > 0:
            return "improving", signals
        return "steady", signals

    def record(self, *, recent: int = 5) -> ImprovementVelocity:
        """Compute the velocity and persist a queryable ``ImprovementVelocity`` node."""
        v = self.summarize(recent=recent)
        if self.engine is not None:
            import uuid

            try:
                self.engine.add_node(
                    f"improvement_velocity:{uuid.uuid4().hex[:12]}",
                    "ImprovementVelocity",
                    properties={
                        "verdict": v.verdict,
                        "metrics_json": json.dumps(v.to_dict()),
                        "recorded_at": _now_iso(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — persistence is best-effort
                logger.debug("[AHE-3.26] could not persist velocity node: %s", exc)
        return v


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def improvement_velocity(engine: Any, *, recent: int = 5) -> dict[str, Any]:
    """Module-level read for MCP/REST surfaces: the current velocity as a dict."""
    return ImprovementLedger(engine).summarize(recent=recent).to_dict()
