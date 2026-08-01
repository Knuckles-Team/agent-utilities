#!/usr/bin/python
from __future__ import annotations

"""Judge calibration primitives (CONCEPT:AU-AHE.evaluation.judge-calibration).

LLM-as-judge is unreliable in specific, measurable ways (pydantic.dev "When agents
improve agents", 2026-07-31):

* **Position bias** — swap the order of two candidates and the verdict often
  flips. Defense: run both orders and average (:func:`swap_and_average`).
* **Self-enhancement bias** — a judge prefers its own model family's output.
  Not mitigated here (needs a judge model distinct from the generator; every
  judge in this package still resolves the SAME live endpoint the generator
  uses — see the audit in ``reports/program/pydantic-ai-native-adoption.md``
  track 8). Flagged, not fixed, in this module.
* **Inconsistency** — ask twice, get different answers. Not mitigated here
  (would need repeated-sampling + agreement scoring on top of this).

Defenses this module DOES implement:

* :func:`swap_and_average` — the position-bias defense: call a pairwise judge
  in both orders, detect a flip, and resolve to one stable verdict from the
  averaged per-candidate scores rather than trusting either single call.
* :func:`llm_pairwise_judge` — a real LLM pairwise comparator (the previously
  absent "duel" generator for :func:`~agent_utilities.harness.
  selection_operators.bradley_terry_scores` / :func:`~agent_utilities.harness.
  frontier_scorers.elo_from_duels`, which took ``(winner, loser)`` tuples as
  input but had no live caller producing them). Reuses the SAME live-endpoint
  resolution and degrade-offline discipline as :mod:`.g_eval`.
* Rubric versioning (:data:`DEFAULT_PAIRWISE_RUBRIC_VERSION` + the
  ``rubric_version`` threaded through every verdict) — "a rubric that changes
  without a version is a judge that drifts without a trace." An explicit
  version always wins; an unset one auto-derives from a content hash of the
  criteria so silent text drift still produces a different fingerprint.
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: Bump this when the pairwise judge PROMPT/SCORING LOGIC changes (not just the
#: criteria text passed to it) — a logic change is invisible to the
#: content-hash auto-fingerprint below, since that only hashes the criteria.
DEFAULT_PAIRWISE_RUBRIC_VERSION = "1.0.0"

Winner = Literal["a", "b", "tie"]


def rubric_fingerprint(criteria: str, version: str = "") -> str:
    """Resolve one traceable rubric identifier.

    An explicit ``version`` always wins (a human bumped it deliberately). With
    none given, auto-derive a stable fingerprint from the criteria text itself
    — so even an *unversioned* rubric edit changes its own fingerprint instead
    of silently drifting under the same identifier.
    """
    v = (version or "").strip()
    if v:
        return v
    return hashlib.sha256(criteria.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class PairwiseVerdict:
    """One pairwise judge verdict, in the ORIGINAL (unswapped) candidate labeling.

    ``measured`` is ``False`` when no live judge endpoint was reachable (the
    same offline-degrade discipline as :class:`~agent_utilities.orchestration.
    loop_guards.GoalEvaluation` — never fabricate a verdict).
    """

    winner: Winner
    score_a: float
    score_b: float
    measured: bool = True
    detail: str = ""
    rubric_version: str = ""
    position_bias_detected: bool = False


#: A single-order pairwise judge: ``(task, candidate_a, candidate_b) ->
#: PairwiseVerdict`` where the returned winner/scores are ALWAYS relative to the
#: literal argument order given (``"a"`` == first positional candidate).
PairwiseJudge = Callable[[str, str, str], PairwiseVerdict]


def _winner_from_scores(score_a: float, score_b: float, *, tie_eps: float = 1e-9) -> Winner:
    if abs(score_a - score_b) <= tie_eps:
        return "tie"
    return "a" if score_a > score_b else "b"


def _normalize_winner(raw: str) -> Winner:
    """Narrow an untrusted (e.g. judge-model-emitted) string to a :data:`Winner`."""
    if raw == "a":
        return "a"
    if raw == "b":
        return "b"
    return "tie"


def swap_and_average(
    judge: PairwiseJudge,
    task: str,
    candidate_a: str,
    candidate_b: str,
) -> PairwiseVerdict:
    """Position-bias defense: judge both orders, average, and report any flip.

    Calls ``judge`` twice — once as given, once with the candidates swapped —
    normalizes the swapped call's verdict back onto the ORIGINAL ``a``/``b``
    labeling, and returns the averaged-score verdict. When the two orders
    disagree on the winner (the position-bias failure mode this defends
    against), ``position_bias_detected`` is set and the averaged scores (not
    either single noisy call) decide the final winner — so
    ``swap_and_average(judge, t, X, Y)`` and the mirrored
    ``swap_and_average(judge, t, Y, X)`` (with labels swapped back) agree,
    even when the raw ``judge`` itself does not.

    ``measured`` is ``True`` only when BOTH calls were live measurements —
    either call degrading offline makes the combined verdict degraded too.
    """
    v1 = judge(task, candidate_a, candidate_b)
    v2_swapped_call = judge(task, candidate_b, candidate_a)

    # Normalize the swapped call back onto the ORIGINAL a/b labeling: in that
    # call "a" was candidate_b and "b" was candidate_a.
    swapped_winner: Winner
    if v2_swapped_call.winner == "a":
        swapped_winner = "b"
    elif v2_swapped_call.winner == "b":
        swapped_winner = "a"
    else:
        swapped_winner = "tie"
    swapped_score_a = v2_swapped_call.score_b
    swapped_score_b = v2_swapped_call.score_a

    agree = v1.winner == swapped_winner
    avg_score_a = (v1.score_a + swapped_score_a) / 2.0
    avg_score_b = (v1.score_b + swapped_score_b) / 2.0
    final_winner = _winner_from_scores(avg_score_a, avg_score_b)

    measured = v1.measured and v2_swapped_call.measured
    version = v1.rubric_version or v2_swapped_call.rubric_version

    if agree:
        detail = (
            f"swap-and-average: both orders agree (winner={v1.winner!r}); "
            f"avg scores a={avg_score_a:.3f} b={avg_score_b:.3f}"
        )
    else:
        detail = (
            f"swap-and-average: POSITION BIAS DETECTED — order 1 said "
            f"{v1.winner!r}, swapped order said {swapped_winner!r} "
            f"(raw: a={v1.score_a:.3f}/{v1.score_b:.3f}, "
            f"swapped-normalized: a={swapped_score_a:.3f}/{swapped_score_b:.3f}); "
            f"resolved via averaged scores -> {final_winner!r}"
        )
        logger.info("swap_and_average corrected a position-bias flip: %s", detail)

    return PairwiseVerdict(
        winner=final_winner,
        score_a=avg_score_a,
        score_b=avg_score_b,
        measured=measured,
        detail=detail,
        rubric_version=version,
        position_bias_detected=not agree,
    )


def llm_pairwise_judge(
    task: str,
    candidate_a: str,
    candidate_b: str,
    *,
    criteria: str = "",
    rubric_version: str = "",
) -> PairwiseVerdict:
    """A real LLM pairwise comparator — the missing duel generator for
    :func:`~agent_utilities.harness.selection_operators.bradley_terry_scores` /
    :func:`~agent_utilities.harness.frontier_scorers.elo_from_duels` (both took
    ``(winner, loser)`` tuples as input with no live caller producing them).

    Reuses :mod:`.g_eval`'s live-endpoint resolution + governed completion so
    this is the SAME model surface every other judge in this package uses (no
    second config path), and degrades to ``measured=False`` exactly like
    :class:`~agent_utilities.orchestration.loop_guards.GoalEvaluation` when no
    endpoint is reachable — never fabricates a winner offline.

    Callers wanting the position-bias defense should wrap this with
    :func:`swap_and_average` rather than trusting a single call directly.
    """
    from . import g_eval as _ge

    version = rubric_fingerprint(criteria or task, rubric_version)
    ep = _ge._live_endpoint()
    if ep is None:
        return PairwiseVerdict(
            winner="tie",
            score_a=0.0,
            score_b=0.0,
            measured=False,
            detail="llm_pairwise_judge unavailable (no model endpoint)",
            rubric_version=version,
        )
    client, model_name = ep
    prompt = (
        f"Task: {task}\n"
        f"Criteria: {criteria or 'Overall quality for the task above.'}\n\n"
        f"Candidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\n"
        "Which candidate better satisfies the task per the criteria? "
        'Respond with ONLY a single JSON line: {"winner": "a"|"b"|"tie", '
        '"score_a": <0..1>, "score_b": <0..1>, "reasoning": "<brief>"}'
    )
    try:
        r = _ge._complete(
            client,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        import json as _json

        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = _json.loads(raw)
        winner = _normalize_winner(str(parsed.get("winner", "tie")).strip().lower())
        score_a = max(0.0, min(1.0, float(parsed.get("score_a", 0.5))))
        score_b = max(0.0, min(1.0, float(parsed.get("score_b", 0.5))))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as exc:  # a judge/parse error must never crash the caller
        logger.debug("llm_pairwise_judge failed: %s", exc)
        return PairwiseVerdict(
            winner="tie",
            score_a=0.0,
            score_b=0.0,
            measured=False,
            detail=f"llm_pairwise_judge error: {exc}",
            rubric_version=version,
        )
    return PairwiseVerdict(
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        measured=True,
        detail=f"llm_pairwise_judge (rubric_version={version}): {reasoning}",
        rubric_version=version,
    )


__all__ = [
    "DEFAULT_PAIRWISE_RUBRIC_VERSION",
    "PairwiseJudge",
    "PairwiseVerdict",
    "Winner",
    "llm_pairwise_judge",
    "rubric_fingerprint",
    "swap_and_average",
]
