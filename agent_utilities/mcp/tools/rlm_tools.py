"""Focused graph-os Recursive Language Model operations."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_rlm_tools(mcp: Any) -> None:
    """Register RLM execution and long-context benchmarks."""

    @mcp.tool(
        name="graph_rlm",
        description=(
            "Run the confined RLM runtime. Actions: 'run' executes an ad-hoc RLM task; "
            "'benchmark' compares RLM, vanilla, and compaction on a supported "
            "long-context task; 'evolve_prompt' runs the GEPA (Genetic-Pareto) "
            "reflective prompt-evolution loop over a PredictRLM signature — the "
            "RLM-specific evolutionary optimizer (Pareto-frontier candidate pool, "
            "DW-GRPO dynamic multi-objective reward weighting on by default), distinct "
            "from and non-overlapping with generic program optimization, which stays "
            "owned exclusively by the native graph_evolution optimize_component "
            "surface (the engine's ProgramOptimize job)."
        ),
        tags=["graph-os", "rlm", "benchmark"],
    )
    async def graph_rlm(
        action: str = Field(
            default="run", description="run | benchmark | evolve_prompt"
        ),
        task: str = Field(
            default="",
            description="Task, skill prompt, or benchmark name; base prompt to evolve "
            "for evolve_prompt (defaults to a small built-in seed prompt).",
        ),
        input_text: str = Field(default="", description="Input context for run."),
        data_json: str = Field(
            default="{}",
            description="Benchmark options ({scales:[int],cases_per_scale:int}) or "
            "evolve_prompt options ({objectives:[str],iterations:int,batch_size:int,"
            "dataset:[{query:str,response:str}]}).",
        ),
    ) -> str:
        try:
            if action == "run":
                from agent_utilities.rlm.runner import run_rlm

                return json.dumps(
                    await run_rlm(task, input_text=input_text), default=str
                )
            if action == "evolve_prompt":
                return json.dumps(await _evolve_prompt(task, data_json), default=str)
            if action == "benchmark":
                from agent_utilities.rlm.benchmarks import (
                    list_tasks,
                    render_scoreboard,
                    run_benchmark,
                )

                options = json.loads(data_json) if data_json else {}
                if not isinstance(options, dict):
                    raise ValueError("data_json must decode to an object for benchmark")
                benchmark = task or "s_niah"
                if benchmark not in list_tasks():
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "unknown benchmark",
                            "tasks": list_tasks(),
                        }
                    )
                scales = options.get("scales") or [50_000]
                results = await run_benchmark(
                    benchmark,
                    scales=[int(scale) for scale in scales],
                    cases_per_scale=int(options.get("cases_per_scale", 3)),
                )
                return json.dumps(
                    {
                        "ok": True,
                        "results": [result.model_dump() for result in results],
                        "scoreboard": render_scoreboard(results),
                    },
                    default=str,
                )
            return f"Error: Unknown graph_rlm action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_rlm"] = graph_rlm
    kg_server.ACTION_TOOL_ROUTES["graph_rlm"] = "/graph/rlm"


#: A small, self-contained default corpus so `evolve_prompt` is directly usable
#: with no caller-supplied dataset — mirrors ``graph_rlm(action="benchmark")``'s
#: own built-in default (``task or "s_niah"``).
_DEFAULT_EVOLVE_PROMPT_DATASET: tuple[dict[str, str], ...] = (
    {"query": "What is the capital of France?", "response": "Paris"},
    {"query": "What is 2 + 2?", "response": "4"},
)


async def _evolve_prompt(task: str, data_json: str) -> dict[str, Any]:
    """Run one GEPA (Genetic-Pareto) reflective prompt-evolution loop.

    CONCEPT:AU-ORCH.optimization.optimize-skill-prompt-gepa. Builds a minimal generic
    PredictRLM signature (one input, one graded text output) and evaluates candidates
    with :func:`~agent_utilities.harness.program_optimization.graded_score` against
    ``dataset`` (a small built-in QA pair set when the caller supplies none) —
    :class:`~agent_utilities.rlm.gepa.GEPAOptimizer` enables DW-GRPO dynamic reward
    weighting (:mod:`~agent_utilities.rlm.dynamic_reward`) by default, so a single call
    here exercises the full evolutionary optimizer, not just its scaffolding.
    """
    from pydantic import BaseModel

    from agent_utilities.harness.program_optimization import graded_score
    from agent_utilities.rlm.gepa import GEPAInstance, GEPAOptimizer
    from agent_utilities.rlm.predict_rlm import InputField, OutputField

    class _PromptEvolutionSignature(BaseModel):
        """Answer the given query."""

        query: str = InputField(default="", description="The task input.")
        response: str = OutputField(default="", description="The model's answer.")

    options = json.loads(data_json) if data_json else {}
    if not isinstance(options, dict):
        raise ValueError("data_json must decode to an object for evolve_prompt")
    objectives = [str(o) for o in options.get("objectives") or ["accuracy"]]
    rows = options.get("dataset") or list(_DEFAULT_EVOLVE_PROMPT_DATASET)
    if not isinstance(rows, list) or not rows:
        raise ValueError("evolve_prompt dataset must be a non-empty list")
    dataset = [
        GEPAInstance(
            id=f"inst_{i}",
            input_data={"query": str(row.get("query", ""))},
            reference_output=str(row.get("response", "")),
        )
        for i, row in enumerate(rows)
    ]

    async def _evaluator(
        instance: GEPAInstance, model_output: Any, _trace: str
    ) -> tuple[dict[str, float], str]:
        score = graded_score(
            str(instance.reference_output or ""),
            getattr(model_output, "response", "") or "",
        )
        return {"accuracy": score}, f"graded_score={score:.3f}"

    optimizer = GEPAOptimizer(
        signature_class=_PromptEvolutionSignature,
        base_prompt=task or "Answer the user's query accurately and concisely.",
        evaluator_fn=_evaluator,
        objectives=objectives,
    )
    best = await optimizer.optimize(
        dataset,
        iterations=max(1, int(options.get("iterations", 1))),
        batch_size=max(1, int(options.get("batch_size", 2))),
    )
    return {
        "ok": True,
        "winning_prompt": best.prompt_text,
        "scores": best.scores,
        "generation": best.generation,
        # Proves DW-GRPO dynamic reward weighting (rlm.dynamic_reward) ran, not just
        # imported — uniform weights until enough generations produce a slope signal.
        "reward_weights": optimizer.pool.reward_weights,
        "frontier_size": len(optimizer.pool.get_frontier()),
    }
