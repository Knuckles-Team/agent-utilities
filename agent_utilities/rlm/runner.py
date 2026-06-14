"""CONCEPT:ORCH-1.12 / ORCH-1.13 — RLM-GEPA live entry-point glue.

`gepa.py` (`GEPAOptimizer`) and `predict_rlm.py` (`PredictRLM`) were previously library-only —
invoked manually, with no MCP/CLI/agent path reaching them (a Wire-First violation that left the
RLM-GEPA optimization features unreachable). This module is the thin, import-connecting entry point:

- :func:`run_rlm` — run the Predict-RLM runtime on an ad-hoc ``task`` over free-form ``input_text``,
  building a dynamic single-output signature so the RLM is callable without a hand-written schema.
- :func:`optimize_rlm_skill` — drive the GEPA loop over a small in-memory dataset with a default
  pass/contains evaluator, returning the best candidate prompt.

Both are reached from `graph_orchestrate(action="rlm_run" | "rlm_optimize")` (and a CLI), putting
`PredictRLM` and `GEPAOptimizer` ≤3 hops from a live entry point.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, create_model

from .config import RLMConfig
from .predict_rlm import InputField, OutputField, PredictRLM
from .telemetry import SandboxFatalError, classify_failure  # CONCEPT:ORCH-1.29

logger = logging.getLogger(__name__)


def _dynamic_signature(
    task: str, output_field: str = "result", output_type: Any = str
) -> type[BaseModel]:
    """Build an ad-hoc Predict-RLM signature: one free-form input → one output field.

    ``output_type`` may be any type pydantic can validate — a primitive (``int``,
    ``bool``), a typing generic (``list[Model]``), or a Pydantic model — so the
    root contract is not limited to a free-form string (CONCEPT:ORCH-1.12).
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
    """Run the Predict-RLM runtime on an ad-hoc task (CONCEPT:ORCH-1.12 entry point).

    Optional ``skills`` (composable :class:`~agent_utilities.rlm.skills.Skill` units, CONCEPT:ORCH-1.28)
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
        # CONCEPT:AHE-3.32 — surface token usage (root + folded sub-call) so a caller (e.g. the
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
        raise  # CONCEPT:ORCH-1.29 — fatal sandbox death must fast-fail, never be swallowed.
    except Exception as e:  # noqa: BLE001 - entry surface must not raise
        # CONCEPT:ORCH-1.29 — classify the failure so the caller/optimizer gets a typed signal.
        failure = classify_failure(e)
        logger.debug("run_rlm failed (%s): %s", failure, e)
        return {"ok": False, "error": str(e), "failure_class": failure, "task": task}


def _default_evaluator(instance: Any, prediction: Any, prompt: str) -> dict[str, Any]:
    """Default GEPA evaluator: substring/exact match of the reference in the prediction.

    Returns ``{"score": float, "feedback": str}`` — the textual feedback the proposer reflects on.
    """
    ref = str(getattr(instance, "reference_output", "") or "")
    pred = str(prediction)
    hit = bool(ref) and ref.lower() in pred.lower()
    return {
        "accuracy": 1.0 if hit else 0.0,
        "efficiency": 1.0 / (1 + len(prompt) / 1000.0),
        "feedback": "matched reference" if hit else f"missing reference {ref[:40]!r}",
    }


async def optimize_rlm_skill(
    base_prompt: str,
    dataset: list[dict[str, Any]],
    *,
    signature: type[BaseModel] | None = None,
    iterations: int = 2,
    batch_size: int = 5,
    config: RLMConfig | None = None,
    graph_deps: Any = None,
    agent_spec: Any = None,
    dev_fraction: float = 0.3,
    persist_run_id: str | None = None,
) -> dict[str, Any]:
    """Optimize a skill prompt via the GEPA loop (CONCEPT:ORCH-1.13 entry point).

    ``dataset`` rows are ``{"input": ..., "reference": ...}``. By default this **enables the
    generalizing-GEPA held-out split** (``dev_fraction=0.3``, ORCH-1.30) so the returned candidate is
    selected on unseen data, threads an optional ``agent_spec`` (ORCH-1.30 grounding), and — when
    ``persist_run_id`` is given — **persists the Pareto frontier** to the epistemic-graph for
    resumable optimization (ORCH-1.31). Best-effort: failures return ``{"ok": False, "error": ...}``.
    """
    from .gepa import GEPAInstance, GEPAOptimizer

    sig = signature or _dynamic_signature(base_prompt)
    try:
        instances = [
            GEPAInstance(
                id=str(row.get("id", i)),
                input_data={"input_text": row.get("input", "")},
                reference_output=row.get("reference", ""),
            )
            for i, row in enumerate(dataset)
        ]
        opt = GEPAOptimizer(
            signature_class=sig,
            base_prompt=base_prompt,
            evaluator_fn=_default_evaluator,
            config=config or RLMConfig(),
            graph_deps=graph_deps,
            agent_spec=agent_spec,  # ORCH-1.30 anti-overfit grounding
        )
        if persist_run_id:  # ORCH-1.31 — resume a prior frontier if one exists
            await opt.resume_frontier(persist_run_id)
        best = await opt.optimize(
            instances,
            iterations=iterations,
            batch_size=batch_size,
            dev_fraction=dev_fraction,  # ORCH-1.30 held-out selection (on by default)
        )
        if persist_run_id:  # ORCH-1.31 — persist the resulting frontier
            await opt.persist_frontier(persist_run_id)
        return {
            "ok": True,
            "best_prompt": best.prompt_text,
            "scores": best.scores,
            "generation": best.generation,
            "dev_fraction": dev_fraction,
        }
    except Exception as e:  # noqa: BLE001 - entry surface must not raise
        logger.debug("optimize_rlm_skill failed: %s", e)
        return {"ok": False, "error": str(e)}
