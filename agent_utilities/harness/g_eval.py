#!/usr/bin/python
from __future__ import annotations

"""Logprob-weighted G-Eval (CONCEPT:AU-AHE.harness.ahe-2).

G-Eval (Liu et al., 2023) makes an LLM-as-judge more robust two ways, both absorbed from
Opik and improved:

1. **Chain-of-thought rubric, generated once and cached.** From a task description +
   evaluation criteria the judge first writes explicit evaluation steps; that rubric is
   reused for every item (LRU-cached per ``(task, criteria, model)``), so the CoT is paid
   once, not per call.
2. **Logprob-weighted continuous score.** Instead of taking the single emitted score
   digit, request top-logprobs on the score token and compute a probability-weighted
   average over the candidate digits — turning a discrete 1–5 judgement into a smooth
   0..1 value that is more stable across runs. Degrades to the point score when the
   provider returns no logprobs.

Reuses the live model endpoint resolved by ``create_model`` (vLLM/OpenAI-style) — no
second config path. Sync (a thin ``openai.OpenAI`` built from the resolved endpoint), so
it slots into the sync ``EvalRunner`` judge surface.
"""

import logging
import math
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: Bump when the SCORING LOGIC/prompt shape changes (scale, instructions) — a
#: rubric-text-only change is caught automatically by the per-instance content
#: fingerprint (CONCEPT:AU-AHE.evaluation.judge-calibration); this constant covers drift
#: in the surrounding mechanism a text hash can't see.
RUBRIC_SCHEMA_VERSION = "1.0.0"

_SCALE = 5  # single-digit 1..5 rubric (one token → clean top-logprob weighting)
# Reasoning models (e.g. Qwen) emit a thinking block first, which buries the score token
# and nulls `content`. Disable thinking so the digit is the emitted token with clean
# top-logprobs. Ignored/retried-without for endpoints that don't accept it.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _complete(client: Any, **kw: Any) -> Any:
    """Governed completion with thinking disabled; retry without if rejected."""
    from agent_utilities.knowledge_graph.retrieval.context_compiler_serving import (
        compiled_chat_completion,
    )

    messages = kw.pop("messages", [])
    prompt = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    try:
        return compiled_chat_completion(
            prompt, client=client, extra_body=_NO_THINK, **kw
        )
    except Exception:
        return compiled_chat_completion(prompt, client=client, **kw)


def _live_endpoint() -> tuple[Any, str] | None:
    """A sync ``openai.OpenAI`` client + model name from the live ``create_model``
    endpoint (introspects the pydantic-ai model's provider client). ``None`` if no
    model/endpoint is reachable (callers degrade)."""
    try:
        from agent_utilities.knowledge_graph.retrieval.context_compiler_serving import (
            resolve_bundle_chat_client,
        )

        return resolve_bundle_chat_client()
    except Exception as exc:  # pragma: no cover - model optional offline
        logger.debug("g-eval endpoint unavailable (%s)", type(exc).__name__)
        return None


@lru_cache(maxsize=256)
def _rubric(task: str, criteria: str, model: str, version: str) -> str:
    """Generate (once, cached) the chain-of-thought evaluation steps for a task/criteria.

    ``version`` is part of the cache key (CONCEPT:AU-AHE.evaluation.judge-calibration) purely so an
    explicit rubric-version bump — even one that leaves ``task``/``criteria`` text
    unchanged (e.g. the grading intent shifted without a wording change) —
    invalidates the cached CoT steps rather than silently reusing a stale rubric.
    """
    ep = _live_endpoint()
    if ep is None:
        return ""
    client, model_name = ep
    prompt = (
        f"Task: {task}\nEvaluation criteria: {criteria}\n\n"
        "Write 3-5 concise, numbered evaluation steps a judge should follow to score an "
        "answer against the criteria. Output ONLY the numbered steps."
    )
    try:
        r = _complete(
            client,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        return r.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover  # noqa: BLE001 — returns "" on failure, which GEval's caller treats as "no rubric" and falls back to criteria-only scoring rather than crashing or silently using a stale rubric
        logger.debug("g-eval rubric generation failed: %s", exc)
        return ""


class GEval:
    """A reusable G-Eval scorer for one ``(task, criteria)`` (CONCEPT:AU-AHE.harness.ahe-2).

    ``rubric_version`` (CONCEPT:AU-AHE.evaluation.judge-calibration) is a traceable identifier for this
    exact criteria: an explicit value wins (bump it deliberately when you change
    grading intent), otherwise it auto-derives from a content hash of
    ``(task_introduction, evaluation_criteria)`` — so an unversioned rubric edit
    still surfaces as a *different* fingerprint instead of drifting silently
    under the same identity. Every score's reasoning carries it.
    """

    def __init__(
        self,
        task_introduction: str,
        evaluation_criteria: str,
        rubric_version: str = "",
    ) -> None:
        self.task = task_introduction
        self.criteria = evaluation_criteria
        from .judge_calibration import rubric_fingerprint

        self.rubric_version = rubric_fingerprint(
            f"{task_introduction}\n{evaluation_criteria}", rubric_version
        )

    def score(self, query: str, actual: str) -> tuple[float, str]:
        """Return ``(score 0..1, reasoning)``. Logprob-weighted when available."""
        ep = _live_endpoint()
        if ep is None:
            return 0.0, "g-eval unavailable (no model)"
        client, model_name = ep
        rubric = _rubric(self.task, self.criteria, model_name, self.rubric_version)
        prompt = (
            f"Task: {self.task}\nCriteria: {self.criteria}\n"
            f"Evaluation steps:\n{rubric}\n\n"
            f"Query: {query}\nAnswer: {actual}\n\n"
            f"Following the steps, rate the answer 1-{_SCALE} (1=worst, {_SCALE}=best). "
            f"Respond with ONLY the single digit."
        )
        try:
            r = _complete(
                client,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
            )
        except Exception as exc:  # pragma: no cover
            return 0.0, f"g-eval scoring failed: {exc}"

        choice = r.choices[0]
        raw = (choice.message.content or "").strip()
        point = _first_digit(raw)
        weighted = _logprob_weighted_score(choice)
        value = weighted if weighted is not None else (point or 0)
        score01 = max(0.0, min(1.0, value / _SCALE))
        how = "logprob-weighted" if weighted is not None else "point"
        return score01, (
            f"g-eval {how} score={value:.2f}/{_SCALE} (rubric-guided, "
            f"rubric_version={self.rubric_version})"
        )


def _first_digit(text: str) -> int | None:
    for ch in text:
        if ch.isdigit():
            return int(ch)
    return None


def _logprob_weighted_score(choice: Any) -> float | None:
    """Probability-weighted average over the digit candidates of the score token."""
    try:
        content = choice.logprobs.content if choice.logprobs else None
        if not content:
            return None
        # Find the first token position whose top candidates include a 1.._SCALE digit.
        for tok in content:
            cands = getattr(tok, "top_logprobs", None) or []
            num, den = 0.0, 0.0
            for c in cands:
                d = _first_digit((c.token or "").strip())
                if d is not None and 1 <= d <= _SCALE:
                    p = math.exp(c.logprob)
                    num += p * d
                    den += p
            if den > 0:
                return num / den
        return None
    except Exception:  # pragma: no cover
        return None


__all__ = ["GEval", "RUBRIC_SCHEMA_VERSION", "_SCALE"]
