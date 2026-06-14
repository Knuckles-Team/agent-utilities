"""CONCEPT:ORCH-1.54 — Drop-in RLM completion client and model-family-aware REPL prompt.

A thin, paper-shaped client so RLM can replace a plain ``llm.completion(prompt)`` call without
learning the structured ``run_rlm(task, input_text=...)`` signature:

    from agent_utilities.rlm import RLM
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-4o-mini"})
    print(rlm.completion("...very long prompt...").response)

The long prompt is handed to the RLM as its external ``context`` variable (the whole point of the
paradigm), so this works on inputs far beyond the model's window. Both a sync ``completion`` and an
async ``acompletion`` are provided; inside an event loop, use ``acompletion``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .config import RLMConfig
from .runner import run_rlm

_DEFAULT_INSTRUCTION = (
    "Read the user's request held in the `context` variable and produce the best possible "
    "response to it, navigating the content programmatically as needed."
)


@dataclass
class RLMResponse:
    """Result of an :class:`RLM` completion. ``response``/``text`` are the answer string."""

    response: str
    ok: bool = True
    usage: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def text(self) -> str:
        return self.response


class RLM:
    """A drop-in recursive-LM client wrapping ``run_rlm`` (CONCEPT:ORCH-1.54).

    ``backend``/``backend_kwargs`` mirror the reference RLM library ergonomics: the resolved
    ``{backend}:{model_name}`` becomes the root model. Pass an explicit ``config`` to control
    depth, sandbox, prompt family, etc.
    """

    def __init__(
        self,
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        *,
        config: RLMConfig | None = None,
    ):
        backend_kwargs = backend_kwargs or {}
        model_name = backend_kwargs.get("model_name") or backend_kwargs.get("model")
        self.config = config or RLMConfig(enabled=True)
        if model_name:
            self.config.sub_llm_model_large = f"{backend}:{model_name}"

    async def acompletion(
        self, prompt: str, *, context: str | None = None
    ) -> RLMResponse:
        """Run an RLM completion. If ``context`` is given, ``prompt`` is the question over it;
        otherwise the prompt itself is the external content to reason over."""
        if context is not None:
            task, input_text = prompt, context
        else:
            task, input_text = _DEFAULT_INSTRUCTION, prompt
        result = await run_rlm(task, input_text=input_text, config=self.config)
        return RLMResponse(
            response=str(result.get("result") or ""),
            ok=bool(result.get("ok")),
            usage=result.get("usage") or {},
            error=result.get("error"),
        )

    def completion(self, prompt: str, *, context: str | None = None) -> RLMResponse:
        """Synchronous wrapper around :meth:`acompletion` (use ``acompletion`` inside a loop)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acompletion(prompt, context=context))
        raise RuntimeError(
            "RLM.completion() called inside a running event loop; await RLM.acompletion() instead."
        )
