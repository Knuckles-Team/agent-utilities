"""CONCEPT:AU-KG.domains.finance-recommendation-scorecard — AU-CONTEXT-R008.1

Calibrated, cited, informational-only holding/watchlist recommendations.

A recommendation is never useful (and never governable) unless it is pinned to
an exact strategy version, cites the evidence it was computed from, and is
unambiguously informational: :class:`AnalysisSnapshot` carries
``informational_only`` fixed to ``True`` (not settable) and this module
exposes no order-submission entry point of any kind. Live-order authority is
the agent connector SDK's governed write-back contract alone
(AU-CONTEXT-R006.1/.2) — nothing here can grant it.

The recommendation abstains (``RecommendationAction.ABSTAIN``) whenever the
evidence is inadequate: no cited evidence identifiers, or a calibrated
confidence below the caller-supplied threshold. Every non-abstaining snapshot
must cite at least one evidence identifier; :class:`AnalysisSnapshot`
enforces that invariant itself so a caller cannot construct one that skips it.

Per-strategy :class:`StrategyScorecard` tallies let several strategy versions
be compared (abstention rate, action mix) without re-deriving history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class RecommendationAction(StrEnum):
    """The informational action a recommendation conveys."""

    ACCUMULATE = "accumulate"
    HOLD = "hold"
    DE_RISK = "de_risk"
    ABSTAIN = "abstain"


class InsufficientEvidenceError(ValueError):
    """A non-abstaining :class:`AnalysisSnapshot` cited no evidence identifiers."""


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """One calibrated, cited, informational-only recommendation.

    ``informational_only`` is fixed to ``True`` by the dataclass (``init=False``,
    no setter — the instance is frozen); no caller can construct a snapshot
    that claims execution authority.
    """

    holding_id: str
    strategy_version: str
    action: RecommendationAction
    confidence: float
    evidence_ids: tuple[str, ...]
    as_of: datetime
    abstain_reason: str | None = None
    informational_only: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        if self.action is not RecommendationAction.ABSTAIN and not self.evidence_ids:
            raise InsufficientEvidenceError(
                f"AnalysisSnapshot for holding {self.holding_id!r} strategy "
                f"{self.strategy_version!r} action {self.action.value!r} must "
                "cite at least one evidence identifier"
            )


@dataclass
class StrategyScorecard:
    """Running per-strategy tally of recommendations, for cross-strategy comparison."""

    strategy_version: str
    total: int = 0
    abstentions: int = 0
    by_action: dict[str, int] = field(default_factory=dict)

    def record(self, snapshot: AnalysisSnapshot) -> None:
        if snapshot.strategy_version != self.strategy_version:
            raise ValueError(
                f"scorecard for {self.strategy_version!r} cannot record a "
                f"snapshot for strategy {snapshot.strategy_version!r}"
            )
        self.total += 1
        if snapshot.action is RecommendationAction.ABSTAIN:
            self.abstentions += 1
        self.by_action[snapshot.action.value] = (
            self.by_action.get(snapshot.action.value, 0) + 1
        )

    @property
    def abstention_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.abstentions / self.total


def recommend(
    *,
    holding_id: str,
    strategy_version: str,
    confidence: float,
    evidence_ids: Sequence[str],
    proposed_action: RecommendationAction,
    calibration_threshold: float,
    as_of: datetime | None = None,
) -> AnalysisSnapshot:
    """Build a calibrated, cited, informational-only :class:`AnalysisSnapshot`.

    Abstains when ``evidence_ids`` is empty or ``confidence`` falls below
    ``calibration_threshold`` — both are "evidence is inadequate" per
    AU-CONTEXT-R008. Otherwise records ``proposed_action`` with the supplied
    evidence. Never touches order execution: see module docstring.
    """
    if proposed_action is RecommendationAction.ABSTAIN:
        raise ValueError(
            "proposed_action must be a non-abstain action; abstention is derived"
        )
    moment = as_of or datetime.now(UTC)
    cited = tuple(evidence_ids)
    if not cited:
        return AnalysisSnapshot(
            holding_id=holding_id,
            strategy_version=strategy_version,
            action=RecommendationAction.ABSTAIN,
            confidence=confidence,
            evidence_ids=(),
            as_of=moment,
            abstain_reason="no evidence identifiers were cited",
        )
    if confidence < calibration_threshold:
        return AnalysisSnapshot(
            holding_id=holding_id,
            strategy_version=strategy_version,
            action=RecommendationAction.ABSTAIN,
            confidence=confidence,
            evidence_ids=cited,
            as_of=moment,
            abstain_reason=(
                f"confidence {confidence:.4f} below calibration threshold "
                f"{calibration_threshold:.4f}"
            ),
        )
    return AnalysisSnapshot(
        holding_id=holding_id,
        strategy_version=strategy_version,
        action=proposed_action,
        confidence=confidence,
        evidence_ids=cited,
        as_of=moment,
    )
