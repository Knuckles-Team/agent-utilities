"""AU-CONTEXT-R008.1: calibrated, cited, informational-only recommendations.

Locks the typed model (:class:`AnalysisSnapshot`, :class:`StrategyScorecard`)
and :func:`recommend`'s abstention behaviour: a recommendation abstains when
evidence is inadequate (missing or below the calibration threshold), and
every non-abstaining snapshot cites its strategy version and at least one
evidence identifier. ``informational_only`` is always ``True`` and not
settable — this module grants no order-submission authority; that is the
agent connector SDK's governed write-back contract (AU-CONTEXT-R006.2).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_utilities.domains.finance.analysis_snapshot import (
    AnalysisSnapshot,
    InsufficientEvidenceError,
    RecommendationAction,
    StrategyScorecard,
    recommend,
)


def test_recommendation_cites_strategy_version_and_evidence() -> None:
    snapshot = recommend(
        holding_id="AAPL",
        strategy_version="momentum-v3",
        confidence=0.9,
        evidence_ids=["filing:AAPL:10-Q:2026Q3", "indicator:AAPL:rsi:2026-10-09"],
        proposed_action=RecommendationAction.ACCUMULATE,
        calibration_threshold=0.6,
    )

    assert snapshot.action is RecommendationAction.ACCUMULATE
    assert snapshot.strategy_version == "momentum-v3"
    assert snapshot.evidence_ids == (
        "filing:AAPL:10-Q:2026Q3",
        "indicator:AAPL:rsi:2026-10-09",
    )
    assert snapshot.informational_only is True


def test_abstains_below_calibration_threshold() -> None:
    snapshot = recommend(
        holding_id="TSLA",
        strategy_version="momentum-v3",
        confidence=0.2,
        evidence_ids=["indicator:TSLA:rsi:2026-10-09"],
        proposed_action=RecommendationAction.DE_RISK,
        calibration_threshold=0.6,
    )

    assert snapshot.action is RecommendationAction.ABSTAIN
    assert snapshot.abstain_reason is not None
    assert "below calibration threshold" in snapshot.abstain_reason
    assert snapshot.informational_only is True


def test_abstains_when_no_evidence_cited() -> None:
    snapshot = recommend(
        holding_id="MSFT",
        strategy_version="momentum-v3",
        confidence=0.95,
        evidence_ids=[],
        proposed_action=RecommendationAction.HOLD,
        calibration_threshold=0.6,
    )

    assert snapshot.action is RecommendationAction.ABSTAIN
    assert snapshot.evidence_ids == ()
    assert snapshot.abstain_reason == "no evidence identifiers were cited"


def test_recommend_rejects_abstain_as_a_proposed_action() -> None:
    with pytest.raises(ValueError, match="abstention is derived"):
        recommend(
            holding_id="AAPL",
            strategy_version="momentum-v3",
            confidence=0.9,
            evidence_ids=["indicator:AAPL:rsi:2026-10-09"],
            proposed_action=RecommendationAction.ABSTAIN,
            calibration_threshold=0.6,
        )


def test_non_abstain_snapshot_requires_evidence_ids() -> None:
    with pytest.raises(InsufficientEvidenceError):
        AnalysisSnapshot(
            holding_id="AAPL",
            strategy_version="momentum-v3",
            action=RecommendationAction.ACCUMULATE,
            confidence=0.9,
            evidence_ids=(),
            as_of=datetime.now(UTC),
        )


def test_abstain_snapshot_may_omit_evidence_ids() -> None:
    snapshot = AnalysisSnapshot(
        holding_id="AAPL",
        strategy_version="momentum-v3",
        action=RecommendationAction.ABSTAIN,
        confidence=0.1,
        evidence_ids=(),
        as_of=datetime.now(UTC),
        abstain_reason="no evidence identifiers were cited",
    )

    assert snapshot.informational_only is True


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        AnalysisSnapshot(
            holding_id="AAPL",
            strategy_version="momentum-v3",
            action=RecommendationAction.ABSTAIN,
            confidence=1.5,
            evidence_ids=(),
            as_of=datetime.now(UTC),
        )


def test_informational_only_is_not_a_constructor_argument() -> None:
    with pytest.raises(TypeError):
        AnalysisSnapshot(
            holding_id="AAPL",
            strategy_version="momentum-v3",
            action=RecommendationAction.ACCUMULATE,
            confidence=0.9,
            evidence_ids=("indicator:AAPL:rsi:2026-10-09",),
            as_of=datetime.now(UTC),
            informational_only=False,  # type: ignore[call-arg]
        )


def test_strategy_scorecard_tracks_abstention_rate_and_action_mix() -> None:
    scorecard = StrategyScorecard(strategy_version="momentum-v3")

    scorecard.record(
        recommend(
            holding_id="AAPL",
            strategy_version="momentum-v3",
            confidence=0.9,
            evidence_ids=["indicator:AAPL:rsi:2026-10-09"],
            proposed_action=RecommendationAction.ACCUMULATE,
            calibration_threshold=0.6,
        )
    )
    scorecard.record(
        recommend(
            holding_id="TSLA",
            strategy_version="momentum-v3",
            confidence=0.2,
            evidence_ids=["indicator:TSLA:rsi:2026-10-09"],
            proposed_action=RecommendationAction.DE_RISK,
            calibration_threshold=0.6,
        )
    )

    assert scorecard.total == 2
    assert scorecard.abstentions == 1
    assert scorecard.abstention_rate == pytest.approx(0.5)
    assert scorecard.by_action == {"accumulate": 1, "abstain": 1}


def test_strategy_scorecard_rejects_mismatched_strategy_version() -> None:
    scorecard = StrategyScorecard(strategy_version="momentum-v3")
    other = recommend(
        holding_id="AAPL",
        strategy_version="mean-reversion-v1",
        confidence=0.9,
        evidence_ids=["indicator:AAPL:rsi:2026-10-09"],
        proposed_action=RecommendationAction.ACCUMULATE,
        calibration_threshold=0.6,
    )

    with pytest.raises(ValueError, match="cannot record"):
        scorecard.record(other)
