"""CONCEPT:KG-2.27 — Agent Calibration and Reputation Tracking

A capability Palantir AIP and Fincept's persona registry both lack: track each
**agent/persona's** past directional calls against realized outcomes, score how
*calibrated* they are (Brier-based), and feed that reputation back into the
**weighted SwarmConsensus** so historically-accurate voices count more.

Closing the loop is the whole point (wire-first): storage alone is dead code.
This module therefore EXPOSES :func:`calibrated_role_weights`, which turns the
tracker's per-agent scores into a ``role_weights`` mapping that
:class:`~agent_utilities.domains.finance.trading_swarm.SwarmConfig` /
:class:`SwarmConsensus` aggregation already consumes — and a thin
:func:`apply_calibration_to_swarm` helper that mutates a live swarm's config.

KG/OWL uniqueness leveraged: each agent's calibration becomes a
``:AgentCalibration`` node (``CALIBRATION_OF`` the agent), so the reputation is a
*queryable, provenance-bearing fact* — a debate can ask "which persona has the
best Brier on tech shorts?" and weight accordingly, instead of treating every
voice as equal. The Brier maths uses the epistemic-graph engine's
``client.finance.brier_score`` when reachable, and a vetted local fallback
otherwise, so it runs offline.

Brier score: mean squared error of probabilistic forecasts vs binary outcomes,
in [0, 1] where **lower is better** (0 = perfect, 0.25 = a coin flip). We convert
to a [0, 1] *calibration score* where **higher is better** as ``1 - 2·brier``
clamped to [0, 1], so an uninformative agent (brier 0.25) scores 0.5.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Lazy, cached engine probe (mirrors forensic_screener) so import is offline-safe.
_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _engine() -> Any:
    """Return a connected SyncEpistemicGraphClient, or ``None`` if unavailable."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    if _ENGINE_PROBED:
        return _ENGINE_CLIENT
    _ENGINE_PROBED = True
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect()
        logger.info("epistemic-graph engine connected for calibration scoring")
    except Exception as exc:  # noqa: BLE001 — degrade to local Brier
        logger.debug("epistemic-graph engine unavailable for calibration: %s", exc)
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


def _local_brier(forecasts: list[float], outcomes: list[float]) -> float:
    """Vetted local Brier score = mean((f - o)^2). Matches the engine kernel."""
    if not forecasts:
        return 0.25
    return sum((f - o) ** 2 for f, o in zip(forecasts, outcomes, strict=False)) / len(
        forecasts
    )


def brier_score(forecasts: list[float], outcomes: list[float]) -> float:
    """Brier score via the engine when reachable, else the local fallback."""
    eng = _engine()
    if eng is not None:
        try:
            return float(eng.finance.brier_score(forecasts, outcomes))
        except Exception as exc:  # noqa: BLE001 — fall back, never invent
            logger.debug("engine brier_score failed, using local: %s", exc)
    return _local_brier(forecasts, outcomes)


@dataclass
class CallRecord:
    """One directional call by an agent and (once known) its outcome.

    ``direction`` is +1 (up/bull) or -1 (down/bear). ``confidence`` is the
    agent's probability that its direction is correct, in [0, 1]. ``correct``
    is filled by :meth:`CalibrationTracker.record_outcome`; ``None`` while open.
    """

    agent_id: str
    direction: int
    confidence: float
    subject: str = ""  # e.g. ticker or thesis id
    correct: bool | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class CalibrationScore:
    """A per-agent reputation summary."""

    agent_id: str
    n_calls: int
    accuracy: float  # fraction of resolved calls that were directionally right
    brier: float  # lower is better; uninformative ~ 0.25
    calibration: float  # higher is better, in [0, 1]; derived from brier

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "n_calls": self.n_calls,
            "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4),
            "calibration": round(self.calibration, 4),
        }


def _calibration_from_brier(brier: float) -> float:
    """Map a Brier (lower better) onto a [0, 1] score (higher better)."""
    return max(0.0, min(1.0, 1.0 - 2.0 * brier))


class CalibrationTracker:
    """Records directional calls + outcomes and scores each agent's calibration.

    Usage::

        tracker = CalibrationTracker()
        tracker.record_call("buffett_investor", direction=+1, confidence=0.8, subject="ACME")
        ...
        tracker.record_outcome("buffett_investor", subject="ACME", realized_direction=+1)
        score = tracker.score("buffett_investor")  # accuracy + Brier + calibration
    """

    def __init__(self) -> None:
        self._calls: dict[str, list[CallRecord]] = defaultdict(list)

    def record_call(
        self,
        agent_id: str,
        direction: int,
        confidence: float,
        subject: str = "",
    ) -> CallRecord:
        """Log an open directional call (outcome resolved later)."""
        rec = CallRecord(
            agent_id=agent_id,
            direction=1 if direction >= 0 else -1,
            confidence=max(0.0, min(1.0, float(confidence))),
            subject=subject,
        )
        self._calls[agent_id].append(rec)
        return rec

    def record_outcome(
        self,
        agent_id: str,
        realized_direction: int,
        subject: str = "",
    ) -> bool:
        """Resolve the most recent open call for ``(agent_id, subject)``.

        Returns ``True`` if a call was resolved, ``False`` if none was open.
        """
        realized = 1 if realized_direction >= 0 else -1
        for rec in reversed(self._calls.get(agent_id, [])):
            if rec.correct is None and (not subject or rec.subject == subject):
                rec.correct = rec.direction == realized
                return True
        return False

    def _forecasts_outcomes(self, agent_id: str) -> tuple[list[float], list[float]]:
        """Build (forecast prob the call was right, 1/0 it was) for resolved calls.

        The forecast is the agent's stated confidence that *its direction* is
        correct, so a well-calibrated agent's confidence matches its hit-rate.
        """
        fc: list[float] = []
        oc: list[float] = []
        for rec in self._calls.get(agent_id, []):
            if rec.correct is None:
                continue
            fc.append(rec.confidence)
            oc.append(1.0 if rec.correct else 0.0)
        return fc, oc

    def score(self, agent_id: str) -> CalibrationScore:
        """Compute accuracy + Brier + calibration for one agent."""
        fc, oc = self._forecasts_outcomes(agent_id)
        n = len(fc)
        if n == 0:
            # No resolved calls yet → neutral prior (calibration 0.5).
            return CalibrationScore(agent_id, 0, 0.0, 0.25, 0.5)
        accuracy = sum(oc) / n
        brier = brier_score(fc, oc)
        return CalibrationScore(
            agent_id=agent_id,
            n_calls=n,
            accuracy=accuracy,
            brier=brier,
            calibration=_calibration_from_brier(brier),
        )

    def all_scores(self) -> dict[str, CalibrationScore]:
        """Score every agent with at least one recorded call."""
        return {a: self.score(a) for a in self._calls}

    def to_batch(self) -> Any:
        """Build an ``ExtractionBatch`` of ``:AgentCalibration`` nodes (KG-2.27)."""
        from agent_utilities.knowledge_graph.enrichment.models import (
            EnrichmentEdge,
            ExtractionBatch,
            GraphNode,
        )

        nodes = []
        edges = []
        for agent_id, sc in self.all_scores().items():
            cid = f"agent_calibration:{agent_id}"
            nodes.append(
                GraphNode(
                    id=cid,
                    type="AgentCalibration",
                    props={
                        "agent_id": agent_id,
                        "n_calls": sc.n_calls,
                        "accuracy": round(sc.accuracy, 4),
                        "brier": round(sc.brier, 4),
                        "calibration": round(sc.calibration, 4),
                        "concept": "KG-2.27",
                    },
                )
            )
            edges.append(
                EnrichmentEdge(source=cid, target=agent_id, rel_type="CALIBRATION_OF")
            )
        return ExtractionBatch(category="agent_calibration", nodes=nodes, edges=edges)

    def persist(self, backend: Any) -> tuple[int, int]:
        """Persist all agent calibrations into the KG via ``write_batch``.

        ``None`` backend (offline) is a no-op returning ``(0, 0)``.
        """
        if backend is None:
            return (0, 0)
        from agent_utilities.knowledge_graph.enrichment.registry import write_batch

        n, e = write_batch(backend, self.to_batch())
        logger.info(
            "Persisted %d agent calibrations: %d nodes, %d edges",
            len(self._calls),
            n,
            e,
        )
        return n, e


def calibrated_role_weights(
    tracker: CalibrationTracker,
    agent_roles: dict[str, Any],
    base_weights: dict[Any, float] | None = None,
    floor: float = 0.25,
) -> dict[Any, float]:
    """Produce calibration-adjusted ``role_weights`` for ``SwarmConsensus``.

    Each role's weight is its base weight scaled by the average calibration of
    the agents mapped to that role (a value in ``[floor, 1]``), so a role staffed
    by historically-accurate agents counts more. Roles with no recorded calls
    keep their base weight (neutral prior).

    Args:
        tracker: the populated :class:`CalibrationTracker`.
        agent_roles: ``{agent_id: SwarmRole}`` — which role each agent fills.
        base_weights: starting role weights; defaults to ``SwarmConfig`` defaults.
        floor: minimum multiplier so a bad-but-improving agent is never zeroed.

    Returns:
        A ``{role: weight}`` mapping consumable directly by ``SwarmConfig`` /
        ``SwarmConsensus`` weighted aggregation.
    """
    from agent_utilities.domains.finance.trading_swarm import SwarmConfig

    base = (
        dict(base_weights)
        if base_weights is not None
        else dict(SwarmConfig().role_weights)
    )

    # Average calibration of agents per role.
    by_role: dict[Any, list[float]] = defaultdict(list)
    for agent_id, role in agent_roles.items():
        sc = tracker.score(agent_id)
        if sc.n_calls > 0:
            by_role[role].append(sc.calibration)

    weights = dict(base)
    for role, cals in by_role.items():
        avg_cal = sum(cals) / len(cals)
        multiplier = max(floor, avg_cal)  # in [floor, 1]
        weights[role] = base.get(role, 1.0) * multiplier
    return weights


def apply_calibration_to_swarm(
    swarm: Any,
    tracker: CalibrationTracker,
    floor: float = 0.25,
) -> dict[Any, float]:
    """Mutate a live ``TradingSwarm``'s config to use calibration-weighted roles.

    Reads the swarm's current agents (``agent.agent_id`` → ``agent.role``),
    computes calibrated weights against the swarm's own base weights, writes them
    back into ``swarm.config.role_weights``, and returns the new weights. This is
    the **live wire**: the very next ``swarm.analyze(...)`` aggregates with the
    updated weights, so a high-calibration quant outvotes a low-calibration one.
    """
    agent_roles = {a.agent_id: a.role for a in getattr(swarm, "agents", [])}
    new_weights = calibrated_role_weights(
        tracker,
        agent_roles,
        base_weights=dict(swarm.config.role_weights),
        floor=floor,
    )
    swarm.config.role_weights = new_weights
    logger.info(
        "Applied calibration-weighted role weights to swarm (%d roles)",
        len(new_weights),
    )
    return new_weights
