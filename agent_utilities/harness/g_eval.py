#!/usr/bin/python
from __future__ import annotations

"""Logprob-weighted G-Eval (CONCEPT:AHE-3.65).

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

_SCALE = 5  # single-digit 1..5 rubric (one token → clean top-logprob weighting)
# Reasoning models (e.g. Qwen) emit a thinking block first, which buries the score token
# and nulls `content`. Disable thinking so the digit is the emitted token with clean
# top-logprobs. Ignored/retried-without for endpoints that don't accept it.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _complete(client: Any, **kw: Any) -> Any:
    """chat.completions.create with thinking disabled; retry without if rejected."""
    try:
        return client.chat.completions.create(extra_body=_NO_THINK, **kw)
    except Exception:
        return client.chat.completions.create(**kw)


def _live_endpoint() -> tuple[Any, str] | None:
    """A sync ``openai.OpenAI`` client + model name from the live ``create_model``
    endpoint (introspects the pydantic-ai model's provider client). ``None`` if no
    model/endpoint is reachable (callers degrade)."""
    try:
        import openai

        from agent_utilities.core.model_factory import create_model

        m = create_model()
        model_name = getattr(m, "model_name", None) or getattr(m, "_model_name", "")
        client = None
        for prov in (getattr(m, "provider", None), getattr(m, "_provider", None)):
            c = getattr(prov, "client", None) if prov is not None else None
            if c is not None and str(getattr(c, "base_url", "")):
                client = c
                break
        if client is None:
            return None
        sync = openai.OpenAI(
            base_url=str(client.base_url), api_key=client.api_key or "EMPTY"
        )
        return sync, str(model_name)
    except Exception as exc:  # pragma: no cover - model optional offline
        logger.debug("g-eval endpoint unavailable: %s", exc)
        return None


@lru_cache(maxsize=256)
def _rubric(task: str, criteria: str, model: str) -> str:
    """Generate (once, cached) the chain-of-thought evaluation steps for a task/criteria."""
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
    except Exception as exc:  # pragma: no cover
        logger.debug("g-eval rubric generation failed: %s", exc)
        return ""


class GEval:
    """A reusable G-Eval scorer for one ``(task, criteria)`` (CONCEPT:AHE-3.65)."""

    def __init__(self, task_introduction: str, evaluation_criteria: str) -> None:
        self.task = task_introduction
        self.criteria = evaluation_criteria

    def score(self, query: str, actual: str) -> tuple[float, str]:
        """Return ``(score 0..1, reasoning)``. Logprob-weighted when available."""
        ep = _live_endpoint()
        if ep is None:
            return 0.0, "g-eval unavailable (no model)"
        client, model_name = ep
        rubric = _rubric(self.task, self.criteria, model_name)
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
        return score01, f"g-eval {how} score={value:.2f}/{_SCALE} (rubric-guided)"


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


__all__ = ["GEval", "_SCALE"]
