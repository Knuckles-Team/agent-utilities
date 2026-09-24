"""The decision points AU hands to EG's ``Decide`` (DECIDE-LAYER-DESIGN §10).

One table row per decision point: which ledger row it closes, the typed
question EG is asked, how much is at stake (exploration is legal only for
``ordinary`` questions), which published ``FeatureSchema`` component it reads,
and how its records are logged. The table is the single owner of those facts;
a call site names its point and nothing else.

A point is only CONSULTED when a binding pins its feature schema (see
:class:`Binding`); an unbound point runs its deterministic fallback. That keeps
EG off every hot path until an operator has published the schema and, later,
a fitted head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class LogMode(Enum):
    """Whether a point's records are made durable in the decision log.

    ``SAMPLED`` is the evaluate-only mode §4.5 prescribes for high-frequency,
    low-stakes points: one record in :attr:`DecisionPoint.sample_every`,
    chosen by record digest so the sample is reproducible, never by a clock.
    """

    ALWAYS = "always"
    SAMPLED = "sampled"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """One decision AU asks EG to make."""

    row: str
    question_id: str
    kind: str
    safety: str = "ordinary"
    log_mode: LogMode = LogMode.ALWAYS
    sample_every: int = 1
    escalate: bool = False
    proposal_only: bool = False

    @property
    def schema_component_id(self) -> str:
        """The FeatureSchema component an operator publishes for this point."""
        return f"decide.schema.{self.question_id}"


def _point(row: str, question_id: str, kind: str, **extra: Any) -> DecisionPoint:
    return DecisionPoint(row=row, question_id=question_id, kind=kind, **extra)


_SAMPLED = {"log_mode": LogMode.SAMPLED, "sample_every": 16}

#: Every decision point, by question id. EG's ``QuestionKind`` and
#: ``QuestionSafety`` wire names (snake_case) are used verbatim.
POINTS: dict[str, DecisionPoint] = {
    p.question_id: p
    for p in (
        _point("EH-029", "au.retrieval.plan", "retrieval_plan"),
        _point("EH-030", "au.ingestion.lane", "ingestion_lane", **_SAMPLED),
        _point("EH-031", "au.enrichment.schedule", "enrichment_schedule"),
        _point(
            "EH-032",
            "au.entity.same_as",
            "resolve_entity",
            safety="irreversible",
            escalate=True,
            proposal_only=True,
        ),
        _point(
            "EH-033",
            "au.schema.mapping",
            "schema_mapping",
            safety="policy",
            escalate=True,
            proposal_only=True,
        ),
        _point(
            "EH-034",
            "au.tms.contradiction",
            "classify",
            safety="irreversible",
            escalate=True,
            proposal_only=True,
        ),
        _point("EH-035", "au.route.choice", "route", **_SAMPLED),
        _point("EH-035", "au.route.model", "route", **_SAMPLED),
        _point(
            "EH-038",
            "au.tool.risk",
            "classify",
            safety="security",
            proposal_only=True,
            **_SAMPLED,
        ),
        _point("EH-039", "au.route.cost", "route"),
        _point(
            "EH-407",
            "au.guardrail.profile",
            "classify",
            safety="policy",
            proposal_only=True,
        ),
        _point("EH-048", "au.swarm.topology", "route"),
        _point("EH-041", "au.connector.triage", "classify", **_SAMPLED),
        _point("EH-042", "au.connector.tool", "route", **_SAMPLED),
        _point(
            "EH-043",
            "au.connector.writeback",
            "route",
            safety="write_back",
            proposal_only=True,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class Binding:
    """The pinned components one point decides under.

    ``feature_schema`` and ``head`` are EG ``ComponentDependency`` dicts
    (``component_id``, ``kind``, ``definition_digest``); ``policy`` is a
    ``DecisionPolicyRef`` dict. No head means EG runs the deterministic ladder
    only, which under the default cold start abstains.
    """

    feature_schema: dict[str, Any]
    policy: dict[str, Any]
    head: dict[str, Any] | None = None


class Bindings(Protocol):
    """Where a point's binding comes from (config, a library lookup, ...)."""

    def binding_for(self, point: DecisionPoint) -> Binding | None: ...


@dataclass(frozen=True, slots=True)
class StaticBindings:
    """Bindings from a mapping of question id to :class:`Binding`."""

    by_question: dict[str, Binding]

    def binding_for(self, point: DecisionPoint) -> Binding | None:
        return self.by_question.get(point.question_id)


def point(question_id: str) -> DecisionPoint:
    """The registered point for ``question_id`` (``KeyError`` names it)."""
    return POINTS[question_id]


__all__ = [
    "POINTS",
    "Binding",
    "Bindings",
    "DecisionPoint",
    "LogMode",
    "StaticBindings",
    "point",
]
