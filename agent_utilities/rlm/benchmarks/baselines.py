"""Systems under test for the RLM benchmark: RLM + the paper's scaffolds (CONCEPT:AHE-3.32).

Three comparable systems, all answering the same :class:`TaskCase`:

* :class:`RLMSystem` — agent-utilities' RLM (``runner.run_rlm``), context handed in as the external
  variable; depth-tiered models and cost surfaced from the RunTrace usage.
* :class:`VanillaSystem` — a single LLM completion over a head/tail-truncated context (the paper's
  "vanilla long-context" baseline).
* :class:`CompactionSystem` — chunk → summarize-each (cheap model) → answer (strong model): the
  lossy "compaction" baseline the paper benchmarks against.

External scaffolds the paper also reports (CodeAct-with-sub-calls, Claude Code) are out of process
and are NOT run here; the scoreboard lists their published numbers and marks them "not run" rather
than silently omitting them.

The pure text operations (truncation, chunking) are importable and unit-tested; the model call is
abstracted behind :class:`Completer` so tests inject a deterministic fake instead of a live LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .base import TaskCase
from .cost import estimate_cost_usd


@dataclass
class Completion:
    text: str
    tokens: int


class Completer(Protocol):
    async def complete(self, system: str, user: str, *, model_id: str) -> Completion:
        ...


@dataclass
class SystemOutput:
    system: str
    prediction: str
    tokens: int = 0
    cost_usd: float = 0.0
    max_depth: int = 0


# ── pure text ops (unit-tested) ──


def head_tail_truncate(text: str, budget_chars: int) -> str:
    """Keep the head and tail of ``text`` within ``budget_chars`` (a vanilla-window proxy)."""
    if budget_chars <= 0 or len(text) <= budget_chars:
        return text
    half = budget_chars // 2
    return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]


def chunk_text(text: str, size: int) -> list[str]:
    """Split ``text`` into ``size``-char chunks (the unit a compaction pass summarizes)."""
    if size <= 0:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


# ── default live completer ──


class PydanticAICompleter:
    """A :class:`Completer` backed by a pydantic-ai ``Agent`` (live LLM)."""

    async def complete(self, system: str, user: str, *, model_id: str) -> Completion:
        from pydantic_ai import Agent

        agent = Agent(model=model_id, system_prompt=system)
        res = await agent.run(user)
        return Completion(text=str(res.output), tokens=_usage_tokens(res))


def _usage_tokens(res: object) -> int:
    """Best-effort total-token extraction across pydantic-ai versions."""
    try:
        usage = res.usage() if callable(getattr(res, "usage", None)) else None
    except Exception:  # noqa: BLE001
        usage = None
    if usage is None:
        return 0
    for attr in ("total_tokens",):
        val = getattr(usage, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    total = 0
    for attr in ("request_tokens", "response_tokens", "input_tokens", "output_tokens"):
        val = getattr(usage, attr, None)
        if isinstance(val, int):
            total += val
    return total


# ── systems ──


class System(ABC):
    name: str = "system"

    @abstractmethod
    async def answer(self, case: TaskCase) -> SystemOutput:
        raise NotImplementedError  # ABSTRACT-OK


class VanillaSystem(System):
    name = "vanilla"

    def __init__(
        self,
        model_id: str = "openai:gpt-4o-mini",
        *,
        window_chars: int = 100_000,
        completer: Completer | None = None,
    ):
        self.model_id = model_id
        self.window_chars = window_chars
        self.completer: Completer = completer or PydanticAICompleter()

    async def answer(self, case: TaskCase) -> SystemOutput:
        ctx = head_tail_truncate(case.context, self.window_chars)
        user = f"CONTEXT:\n{ctx}\n\nQUESTION: {case.question}"
        out = await self.completer.complete(
            "You answer questions using only the provided context.",
            user,
            model_id=self.model_id,
        )
        return SystemOutput(
            system=self.name,
            prediction=out.text,
            tokens=out.tokens,
            cost_usd=estimate_cost_usd(out.tokens, self.model_id),
        )


class CompactionSystem(System):
    name = "compaction"

    def __init__(
        self,
        model_large: str = "google:gemini-1.5-flash",
        model_small: str = "openai:gpt-4o-mini",
        *,
        chunk_chars: int = 8_000,
        max_chunks: int = 64,
        completer: Completer | None = None,
    ):
        self.model_large = model_large
        self.model_small = model_small
        self.chunk_chars = chunk_chars
        self.max_chunks = max_chunks
        self.completer: Completer = completer or PydanticAICompleter()

    async def answer(self, case: TaskCase) -> SystemOutput:
        chunks = chunk_text(case.context, self.chunk_chars)
        truncated = len(chunks) > self.max_chunks
        chunks = chunks[: self.max_chunks]
        summaries: list[str] = []
        tokens = 0
        cost = 0.0
        for ch in chunks:
            out = await self.completer.complete(
                "Summarize the text, preserving any detail relevant to the question.",
                f"QUESTION: {case.question}\n\nTEXT:\n{ch}",
                model_id=self.model_small,
            )
            summaries.append(out.text)
            tokens += out.tokens
            cost += estimate_cost_usd(out.tokens, self.model_small)
        digest = "\n".join(summaries)
        final = await self.completer.complete(
            "You answer using only the compacted notes.",
            f"NOTES:\n{digest}\n\nQUESTION: {case.question}",
            model_id=self.model_large,
        )
        tokens += final.tokens
        cost += estimate_cost_usd(final.tokens, self.model_large)
        note = "compaction truncated chunk set" if truncated else ""
        return SystemOutput(
            system=self.name + (f" ({note})" if note else ""),
            prediction=final.text,
            tokens=tokens,
            cost_usd=cost,
        )


class RLMSystem(System):
    name = "rlm"

    def __init__(self, config: object | None = None):
        from ..config import RLMConfig

        self.config = config or RLMConfig(enabled=True)

    async def answer(self, case: TaskCase) -> SystemOutput:
        from ..runner import run_rlm

        result = await run_rlm(
            case.question, input_text=case.context, config=self.config
        )
        usage = result.get("usage") or {}
        prompt_t = int(usage.get("prompt_tokens", 0))
        completion_t = int(usage.get("completion_tokens", 0))
        sub_t = int(usage.get("sub_lm_tokens", 0))
        cost = estimate_cost_usd(
            prompt_t + completion_t, getattr(self.config, "sub_llm_model_large", "")
        ) + estimate_cost_usd(sub_t, getattr(self.config, "sub_llm_model_small", ""))
        return SystemOutput(
            system=self.name,
            prediction=str(result.get("result", "")),
            tokens=prompt_t + completion_t + sub_t,
            cost_usd=cost,
            max_depth=int(result.get("max_depth", 0)),
        )
