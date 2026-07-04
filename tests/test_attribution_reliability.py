#!/usr/bin/python
"""Tests for the self-attribution reliability metric on ManifestVerifier.

CONCEPT:AU-AHE.harness.harness-evolution
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.harness.evidence_corpus import EvidenceCorpus, EvidenceEntry
from agent_utilities.harness.manifest import (
    ChangeManifest,
    ComponentEdit,
    ComponentType,
)
from agent_utilities.harness.verifier import ManifestVerifier

pytestmark = pytest.mark.concept("AU-AHE.harness.harness-evolution")


def _manifest(predicted_fixes):
    m = ChangeManifest(baseline_score=0.5)
    m.add_edit(
        ComponentEdit(
            component_type=ComponentType.MIDDLEWARE,
            file_path="guard.py",
            edit_summary="x",
            predicted_fixes=list(predicted_fixes),
        )
    )
    return m


def _corpus(outcomes: dict[str, bool], score: float) -> EvidenceCorpus:
    return EvidenceCorpus(
        entries=[
            EvidenceEntry(task_id=t, pass_fail=p, score=1.0 if p else 0.0)
            for t, p in outcomes.items()
        ],
        benchmark_score=score,
    )


@pytest.mark.asyncio
async def test_reliable_when_prediction_beats_base_rate():
    verifier = ManifestVerifier(registry=MagicMock(), reliability_multiple=3.0)
    # One real fix among 5 tasks → base rate 0.2; predicting exactly it → precision 1.0 → 5x lift.
    baseline = _corpus(
        {"t1": False, "t2": True, "t3": True, "t4": True, "t5": True}, 0.5
    )
    new = _corpus({"t1": True, "t2": True, "t3": True, "t4": True, "t5": True}, 0.7)
    result = await verifier.verify(_manifest(["t1"]), baseline, new)

    assert result.fix_precision == 1.0
    assert result.random_baseline_precision == pytest.approx(0.2)
    assert result.attribution_lift == pytest.approx(5.0)
    assert result.attribution_reliable is True


@pytest.mark.asyncio
async def test_unreliable_when_base_rate_high():
    verifier = ManifestVerifier(registry=MagicMock(), reliability_multiple=3.0)
    # Both tasks fixed → base rate 1.0; a correct single prediction is unimpressive (1x lift).
    baseline = _corpus({"t1": False, "t2": False}, 0.0)
    new = _corpus({"t1": True, "t2": True}, 1.0)
    result = await verifier.verify(_manifest(["t1"]), baseline, new)

    assert result.fix_precision == 1.0
    assert result.random_baseline_precision == pytest.approx(1.0)
    assert result.attribution_lift == pytest.approx(1.0)
    assert result.attribution_reliable is False


@pytest.mark.asyncio
async def test_no_actual_fixes_is_not_reliable():
    verifier = ManifestVerifier(registry=MagicMock())
    baseline = _corpus({"t1": True, "t2": True}, 1.0)
    new = _corpus({"t1": True, "t2": True}, 1.0)
    result = await verifier.verify(_manifest(["t1"]), baseline, new)

    assert result.random_baseline_precision == 0.0
    assert result.attribution_lift == 0.0
    assert result.attribution_reliable is False


# ── Signature attribution falsifiability (CONCEPT:AU-AHE.evaluation.edit-claims-fix) ──────────────────
def test_signature_fired_detects_presence_and_absence():
    present = EvidenceCorpus(
        entries=[
            EvidenceEntry(task_id="t1", pass_fail=True, content="called WikiFetch ok")
        ]
    )
    absent = EvidenceCorpus(
        entries=[
            EvidenceEntry(task_id="t1", pass_fail=True, content="solved by guessing")
        ]
    )
    sig = {"tool_call": "WikiFetch", "min_count": 1}
    assert ManifestVerifier._signature_fired(sig, present) is True
    assert ManifestVerifier._signature_fired(sig, absent) is False
    # An empty signature trivially fires (back-compat for edits that declare none).
    assert ManifestVerifier._signature_fired({}, absent) is True


@pytest.mark.asyncio
async def test_unattributed_edit_is_not_confirmed():
    """An apparent fix whose attribution signature never fires must NOT be
    confirmed — it 'passed' only by coincidence (reward-hacking guard)."""
    verifier = ManifestVerifier(registry=MagicMock())
    manifest = ChangeManifest(baseline_score=0.5)
    manifest.add_edit(
        ComponentEdit(
            id="edit:wiki",
            component_type=ComponentType.TOOL_IMPLEMENTATION,
            file_path="wiki.py",
            edit_summary="add WikiFetch",
            predicted_fixes=["t1"],
            attribution_signature={"tool_call": "WikiFetch", "min_count": 1},
        )
    )
    baseline = _corpus({"t1": False}, 0.0)
    # t1 now passes, BUT the trace never shows WikiFetch → the edit is unattributed.
    new = EvidenceCorpus(
        entries=[EvidenceEntry(task_id="t1", pass_fail=True, content="guessed it")],
        benchmark_score=1.0,
    )
    result = await verifier.verify(manifest, baseline, new)
    assert "edit:wiki" in result.unattributed_edits
    assert result.recommendation != "confirm"
