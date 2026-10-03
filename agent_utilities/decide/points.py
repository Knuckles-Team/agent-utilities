"""The decision points AU hands to EG's ``Decide``.

One table row per decision point: a short label for the point, the typed
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
    #: The declared policy role whose holders may evaluate this point's
    #: committed records (EG grants them a record-scoped, expiring,
    #: evaluation-only lease; never the committer). A binding naming an
    #: evaluator principal overrides it.
    evaluator_role: str | None = None

    @property
    def schema_component_id(self) -> str:
        """The FeatureSchema component an operator publishes for this point."""
        return f"decide.schema.{self.question_id}"


def _point(row: str, question_id: str, kind: str, **extra: Any) -> DecisionPoint:
    return DecisionPoint(row=row, question_id=question_id, kind=kind, **extra)


_SAMPLED = {"log_mode": LogMode.SAMPLED, "sample_every": 16}

#: The policy role an independent evaluator of AU's retrieval runs holds
#: (granted to the evaluator's identity by the deployment, never to AU's).
DECIDE_EVALUATOR_ROLE = "decide-evaluator"

#: Every decision point, by question id. EG's ``QuestionKind`` and
#: ``QuestionSafety`` wire names (snake_case) are used verbatim.
POINTS: dict[str, DecisionPoint] = {
    p.question_id: p
    for p in (
        _point(
            "retrieval-plan",
            "au.retrieval.plan",
            "retrieval_plan",
            evaluator_role=DECIDE_EVALUATOR_ROLE,
        ),
        _point("ingestion-lane", "au.ingestion.lane", "ingestion_lane", **_SAMPLED),
        _point(
            "enrichment-schedule", "au.enrichment.schedule", "enrichment_schedule"
        ),
        _point(
            "entity-same-as",
            "au.entity.same_as",
            "resolve_entity",
            safety="irreversible",
            escalate=True,
            proposal_only=True,
        ),
        _point(
            "schema-mapping",
            "au.schema.mapping",
            "schema_mapping",
            safety="policy",
            escalate=True,
            proposal_only=True,
        ),
        _point(
            "tms-contradiction",
            "au.tms.contradiction",
            "classify",
            safety="irreversible",
            escalate=True,
            proposal_only=True,
        ),
        _point("route-choice", "au.route.choice", "route", **_SAMPLED),
        _point("route-model", "au.route.model", "route", **_SAMPLED),
        _point(
            "tool-risk",
            "au.tool.risk",
            "classify",
            safety="security",
            proposal_only=True,
            **_SAMPLED,
        ),
        _point("route-cost", "au.route.cost", "route"),
        # The statistical rung over a topology plan's legal (template, width,
        # rounds) options; advisory until a head is calibrated, never
        # explored or logged (AU-CONTROL-R020).
        _point(
            "swarm-topology",
            "au.swarm.topology",
            "template_choice",
            log_mode=LogMode.NEVER,
        ),
        # Continue / narrow / stop between rounds over a committed topology
        # plan; evaluate-only and sampled, options narrow-only
        # (AU-CONTROL-R019).
        _point("swarm-continue", "au.swarm.continue", "route", **_SAMPLED),
        _point("connector-triage", "au.connector.triage", "classify", **_SAMPLED),
        _point("connector-tool", "au.connector.tool", "route", **_SAMPLED),
        _point(
            "connector-writeback",
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
    #: The principal (EG persistence id) named at commit to evaluate
    #: this point's records -- the only way a committer-only record can be
    #: judged independently -- and how long that grant lives.
    evaluator: str | None = None
    evaluator_ttl_s: int = 3600


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
