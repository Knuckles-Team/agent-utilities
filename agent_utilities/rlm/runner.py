"""CONCEPT:AU-ORCH.execution.predict-rlm-runtime — live Predict-RLM entry point.

The RLM runtime executes bounded recursive inference through :func:`run_rlm`.
Program optimization is a separate native epistemic-graph responsibility exposed
through ``graph_evolution action=optimize_component``; this module contains no prompt
optimizer or alternate model stack.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, create_model

from .config import RLMConfig
from .predict_rlm import InputField, OutputField, PredictRLM
from .telemetry import (  # CONCEPT:AU-ORCH.execution.typed-failure-classification
    SandboxFatalError,
    classify_failure,
)

logger = logging.getLogger(__name__)


def _dynamic_signature(
    task: str, output_field: str = "result", output_type: Any = str
) -> type[BaseModel]:
    """Build an ad-hoc Predict-RLM signature: one free-form input → one output field.

    ``output_type`` may be any type pydantic can validate — a primitive (``int``,
    ``bool``), a typing generic (``list[Model]``), or a Pydantic model — so the
    root contract is not limited to a free-form string (CONCEPT:AU-ORCH.execution.predict-rlm-runtime).
    """
    model = create_model(  # type: ignore[call-overload]  # pydantic dynamic-model field tuples
        "AdHocRLMSignature",
        input_text=(
            str,
            InputField(default="", description="The input to reason over."),
        ),
        **{output_field: (output_type, OutputField(description=task))},
    )
    model.__doc__ = task
    return model


async def run_rlm(
    task: str,
    input_text: str = "",
    *,
    config: RLMConfig | None = None,
    graph_deps: Any = None,
    output_field: str = "result",
    output_type: Any = str,
    skills: list[Any] | None = None,
) -> dict[str, Any]:
    """Run the Predict-RLM runtime on an ad-hoc task (CONCEPT:AU-ORCH.execution.predict-rlm-runtime entry point).

    Optional ``skills`` (composable :class:`~agent_utilities.rlm.skills.Skill` units, CONCEPT:AU-ORCH.adapter.composable-skills-environment)
    are merged and mounted into the runtime before execution. ``output_type`` lets the caller request
    a structured root contract (e.g. ``bool`` or a Pydantic model) instead of a free-form string.
    Returns ``{"ok": bool, "result": ...}``. Best-effort: a runtime/model failure returns
    ``{"ok": False, "error": ...}`` rather than raising.
    """
    sig = _dynamic_signature(task, output_field, output_type)
    try:
        rlm = PredictRLM(sig, config=config or RLMConfig(), graph_deps=graph_deps)
        if skills:
            from .skills import merge_skills

            rlm.mount_skill_unit(merge_skills(skills))
        out = await rlm.run(input_text=input_text)
        value = getattr(out, output_field, None)
        # CONCEPT:AU-AHE.rlm.token-usage-surface — surface token usage (root + folded sub-call) so a caller (e.g. the
        # benchmark harness) can compute per-query cost. Best-effort; absent trace → empty.
        trace = getattr(rlm, "last_run_trace", None)
        usage = trace.usage.model_dump() if trace is not None else {}
        if trace is not None:
            usage["total"] = trace.usage.total
        return {
            "ok": True,
            "result": value,
            "task": task,
            "usage": usage,
            "max_depth": (config or RLMConfig()).max_depth,
        }
    except SandboxFatalError:
        raise  # CONCEPT:AU-ORCH.execution.typed-failure-classification — fatal sandbox death must fast-fail, never be swallowed.
    except Exception as e:  # noqa: BLE001 - entry surface must not raise
        # CONCEPT:AU-ORCH.execution.typed-failure-classification — classify the failure so the caller/optimizer gets a typed signal.
        failure = classify_failure(e)
        logger.debug("run_rlm failed (%s): %s", failure, e)
        return {"ok": False, "error": str(e), "failure_class": failure, "task": task}
