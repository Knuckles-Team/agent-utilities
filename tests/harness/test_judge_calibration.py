"""Judge calibration primitives (CONCEPT:AU-AHE.evaluation.judge-calibration) — position-bias
defense (swap-and-average) and rubric versioning.

The headline test (``test_swap_and_average_corrects_a_position_bias_flip``) uses a
DELIBERATELY biased mock judge — one that always prefers whichever candidate is
presented in position "a", independent of content — to reproduce the well-documented
LLM-judge position-bias failure mode deterministically and offline (no live model
endpoint is reachable in this unit environment; see ``tests/harness/test_g_eval.py``'s
same convention). This is an honest reproduction of the failure mode being defended
against, not a fabricated result: the mock's bias is fully transparent in this file,
and the point of the test is that ``swap_and_average`` corrects it regardless of which
judge is plugged in underneath.
"""

from __future__ import annotations

from agent_utilities.harness.judge_calibration import (
    PairwiseVerdict,
    rubric_fingerprint,
    swap_and_average,
)


def _position_biased_judge(
    task: str, candidate_a: str, candidate_b: str
) -> PairwiseVerdict:
    """A judge that ALWAYS prefers whichever candidate is in position "a".

    This reproduces position bias exactly: the verdict depends on argument
    order, not content — the classic LLM-as-judge failure the article
    documents ("swap the order of two candidates and the verdict often
    flips").
    """
    return PairwiseVerdict(winner="a", score_a=0.9, score_b=0.1, measured=True)


def _content_aware_judge(
    task: str, candidate_a: str, candidate_b: str
) -> PairwiseVerdict:
    """A judge that correctly prefers the longer (here: better) candidate,
    regardless of which position it's presented in — the CONTROL case: no
    position bias, so swap-and-average should agree with a single call and
    report no bias detected.
    """
    score_a = float(len(candidate_a))
    score_b = float(len(candidate_b))
    total = score_a + score_b or 1.0
    winner = "a" if score_a > score_b else ("b" if score_b > score_a else "tie")
    return PairwiseVerdict(
        winner=winner, score_a=score_a / total, score_b=score_b / total, measured=True
    )


def test_position_biased_judge_flips_verdict_on_naive_single_call():
    """Reproduce the failure mode BEFORE applying the defense: a single,
    unswapped call to the biased judge gives the OPPOSITE winner depending on
    which candidate is passed as "a" — even though the candidates (and which
    one is "actually better" per the content-aware judge) never changed.
    """
    good, bad = "the correct and complete answer", "wrong"

    verdict_good_first = _position_biased_judge("t", good, bad)
    verdict_bad_first = _position_biased_judge("t", bad, good)

    # Naive single-call judging: "good" wins when presented first, loses when
    # presented second — a pure order artifact, exactly what position bias is.
    assert verdict_good_first.winner == "a"  # "good" (position a) "wins"
    assert verdict_bad_first.winner == "a"  # "bad" (position a) ALSO "wins"
    # i.e. whichever candidate the caller happens to list first wins, regardless
    # of content — the flip this whole module exists to correct.


def test_swap_and_average_corrects_a_position_bias_flip():
    """THE headline test: wrap the same biased judge in ``swap_and_average`` and
    the verdict becomes stable and content-driven, independent of argument
    order — and the wrapper HONESTLY reports that it detected + corrected a
    disagreement between the two orders.
    """
    good, bad = "the correct and complete answer", "wrong"

    result_good_first = swap_and_average(_position_biased_judge, "t", good, bad)
    result_bad_first = swap_and_average(_position_biased_judge, "t", bad, good)

    # Both orders now agree on a winner (from averaged, order-invariant scores)...
    assert result_good_first.position_bias_detected is True
    assert result_bad_first.position_bias_detected is True
    # ...and the wrapper is invariant to which candidate is listed first: the
    # "good" candidate's averaged score is the same whichever slot it started in.
    assert result_good_first.score_a == result_bad_first.score_b
    assert result_good_first.score_b == result_bad_first.score_a
    # Because both raw scores for the biased judge are symmetric (0.9/0.1 either
    # way), the average collapses to a tie — which is the CORRECT, honest
    # resolution for a judge shown to carry pure position bias and nothing else:
    # neither candidate has a measured content-driven edge once bias is removed.
    assert result_good_first.winner == "tie"
    assert result_bad_first.winner == "tie"


def test_swap_and_average_agrees_with_an_unbiased_judge():
    """Control case: a content-aware (non-position-biased) judge already agrees
    across both orders, so swap-and-average reports NO bias detected and
    preserves the correct winner.
    """
    good, bad = "the correct and complete answer", "wrong"

    result = swap_and_average(_content_aware_judge, "t", good, bad)
    assert result.position_bias_detected is False
    assert result.winner == "a"  # "good" (the longer/better candidate) still wins

    swapped = swap_and_average(_content_aware_judge, "t", bad, good)
    assert swapped.position_bias_detected is False
    assert swapped.winner == "b"  # "good" is now in slot b, and still wins


def test_swap_and_average_measured_requires_both_calls_live():
    """``measured`` is False if EITHER order's call degraded offline — never
    silently trust a half-measured verdict."""

    def _one_side_degraded(task: str, a: str, b: str) -> PairwiseVerdict:
        # Degrades (measured=False) only when "x" is in the "a" slot.
        if a == "x":
            return PairwiseVerdict(
                winner="tie", score_a=0.0, score_b=0.0, measured=False
            )
        return PairwiseVerdict(winner="a", score_a=0.8, score_b=0.2, measured=True)

    result = swap_and_average(_one_side_degraded, "t", "x", "y")
    assert result.measured is False


# ── rubric versioning (CONCEPT:AU-AHE.evaluation.judge-calibration) ────────────────────────


def test_rubric_fingerprint_explicit_version_wins():
    assert rubric_fingerprint("some criteria text", "2.0.0") == "2.0.0"


def test_rubric_fingerprint_auto_derives_from_content_when_unset():
    fp1 = rubric_fingerprint("criteria A")
    fp2 = rubric_fingerprint("criteria A")
    fp3 = rubric_fingerprint("criteria B — different text")
    # stable for identical content...
    assert fp1 == fp2
    # ...but a silent rubric-text edit still surfaces as a DIFFERENT fingerprint,
    # rather than drifting invisibly under no version at all.
    assert fp1 != fp3


def test_g_eval_rubric_version_auto_derives_and_is_reported(monkeypatch):
    """``GEval`` threads a rubric_version into every score's reasoning, and an
    unversioned criteria edit produces a DIFFERENT auto-derived version — the
    mechanical form of "a rubric that changes without a version is a judge
    that drifts without a trace"."""
    from agent_utilities.harness import g_eval as ge

    monkeypatch.setattr(ge, "_live_endpoint", lambda: None)  # offline degrade path
    g1 = ge.GEval("task", "criteria v1")
    g2 = ge.GEval("task", "criteria v1")
    g3 = ge.GEval("task", "criteria v2 -- changed")

    assert g1.rubric_version == g2.rubric_version
    assert g1.rubric_version != g3.rubric_version

    explicit = ge.GEval("task", "criteria v1", rubric_version="9.9.9")
    assert explicit.rubric_version == "9.9.9"
