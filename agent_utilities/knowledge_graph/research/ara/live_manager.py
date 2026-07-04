#!/usr/bin/python
from __future__ import annotations

"""Live Research Manager — typed research events with provenance (CONCEPT:AU-KG.ontology.verified-by-implemented-by).

The paper's Live Research Manager continuously captures the *act* of research, not just
its conclusion: a Context-Harvester ingests the working stream, an Event-Router classifies
each move into a typed event, and a Maturity-Tracker crystallizes events into the artifact.
We make every event carry **provenance** — who originated it (``user`` / ``ai_suggested``
/ ``ai_executed`` / ``user_revised``, the EditLedger/ActorContext distinction, KG-2.43 /
OS-5.14) — and promote events into the one ontology so reasoning links research moves
across sessions and to the ecosystem.

This is the lightweight, embeddable core: an in-memory event log that classifies + stamps
provenance, crystallizes mature events onto a :class:`ResearchArtifact`'s layers, and can
flush events into the graph. It is deliberately dependency-free (callers wire the real
session collector / EditLedger), so it is unit-testable and never blocks a Loop.

Concept: live-research-manager
"""

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .artifact import Claim, Evidence, ExplorationNode, ResearchArtifact

logger = logging.getLogger(__name__)

#: the seven research-event types the router recognizes.
EventType = Literal[
    "question",
    "hypothesis",
    "decision",
    "experiment",
    "observation",
    "claim",
    "pivot",
]

#: provenance of a research move (the EditLedger / ActorContext distinction).
Provenance = Literal["user", "ai_suggested", "ai_executed", "user_revised"]

_MATURE: frozenset[str] = frozenset({"claim", "observation", "decision", "experiment"})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ResearchEvent(BaseModel):
    """One captured research move (CONCEPT:AU-KG.ontology.verified-by-implemented-by)."""

    id: str
    type: EventType
    text: str
    provenance: Provenance = "ai_executed"
    actor: str = ""
    timestamp: str = Field(default_factory=_now_iso)
    #: whether the Maturity-Tracker has crystallized this into the artifact.
    crystallized: bool = False


class LiveResearchManager:
    """Capture → route → mature research events for one artifact's lifecycle."""

    def __init__(self, article_id: str) -> None:
        self._aid = article_id
        self._events: list[ResearchEvent] = []
        self._n = 0

    @property
    def events(self) -> list[ResearchEvent]:
        return list(self._events)

    # -- capture / route -------------------------------------------------- #
    def capture(
        self,
        text: str,
        *,
        type: EventType | None = None,
        provenance: Provenance = "ai_executed",
        actor: str = "",
    ) -> ResearchEvent:
        """Record a research move, classifying its type when not given."""
        self._n += 1
        ev = ResearchEvent(
            id=f"exploration_node:{self._aid}:event:{self._n}",
            type=type or self._route(text),
            text=text,
            provenance=provenance,
            actor=actor,
        )
        self._events.append(ev)
        return ev

    @staticmethod
    def _route(text: str) -> EventType:
        """Heuristic Event-Router — classify a move from its surface form."""
        t = (text or "").lower().strip()
        if t.endswith("?") or t.startswith(("how ", "why ", "what ", "can ", "does ")):
            return "question"
        if t.startswith(("we hypothesize", "hypothesis", "suppose", "assume")):
            return "hypothesis"
        if t.startswith(("decide", "we chose", "pick", "select", "adopt")):
            return "decision"
        if t.startswith(("run", "experiment", "benchmark", "measure", "test")):
            return "experiment"
        if t.startswith(("pivot", "instead", "abandon", "reconsider")):
            return "pivot"
        if t.startswith(("observed", "result", "we find", "found", "measured")):
            return "observation"
        return "claim"

    # -- maturity / crystallize ------------------------------------------- #
    def crystallize(self, artifact: ResearchArtifact) -> dict[str, int]:
        """Fold mature events onto the artifact's layers (idempotent per event).

        claim → /logic, observation → /evidence, question/decision/experiment/pivot
        → /trace. Returns counts per layer crystallized this call.
        """
        counts = {"claims": 0, "evidence": 0, "exploration": 0}
        crystallizable = _MATURE | {
            "question",
            "pivot",
        }  # all but uncommitted hypothesis
        for ev in self._events:
            if ev.crystallized or ev.type not in crystallizable:
                continue
            if ev.type == "claim":
                artifact.claims.append(
                    Claim(id=f"claim:{self._aid}:{ev.id}", statement=ev.text)
                )
                counts["claims"] += 1
            elif ev.type == "observation":
                artifact.evidence.append(
                    Evidence(id=f"evidence:{self._aid}:{ev.id}", content=ev.text)
                )
                counts["evidence"] += 1
            else:  # question / decision / experiment / pivot → trace
                kind = (
                    "pivot"
                    if ev.type == "pivot"
                    else ("question" if ev.type == "question" else "decision")
                )
                artifact.exploration.append(
                    ExplorationNode(id=ev.id, kind=kind, text=ev.text)  # type: ignore[arg-type]  # kind is provably one of the ExplorationNode literals
                )
                counts["exploration"] += 1
            ev.crystallized = True
        return counts

    # -- promote to graph ------------------------------------------------- #
    def flush(self, engine: Any) -> int:
        """Promote captured events into the graph as exploration nodes (best-effort).

        Each event becomes an ``exploration_node`` carrying its provenance, so
        reasoning links research moves across sessions and to the ecosystem.
        """
        n = 0
        for ev in self._events:
            try:
                engine.add_node(
                    ev.id,
                    "exploration_node",
                    properties={
                        "name": ev.text[:80] or ev.id,
                        "exploration_kind": ev.type,
                        "text": ev.text,
                        "provenance": ev.provenance,
                        "actor": ev.actor,
                        "timestamp": ev.timestamp,
                    },
                )
                n += 1
            except Exception as e:  # noqa: BLE001 — best-effort persist
                logger.debug("LRM event flush failed (%s): %s", ev.id, e)
        return n


__all__ = ["LiveResearchManager", "ResearchEvent", "EventType", "Provenance"]
