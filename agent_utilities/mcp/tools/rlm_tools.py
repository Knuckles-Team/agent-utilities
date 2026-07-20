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
            "long-context task. Program optimization is owned exclusively by the "
            "native graph_evolution optimize_component surface."
        ),
        tags=["graph-os", "rlm", "benchmark"],
    )
    async def graph_rlm(
        action: str = Field(default="run", description="run | benchmark"),
        task: str = Field(
            default="", description="Task, skill prompt, or benchmark name."
        ),
        input_text: str = Field(default="", description="Input context for run."),
        data_json: str = Field(
            default="{}",
            description="Benchmark options ({scales:[int],cases_per_scale:int}).",
        ),
    ) -> str:
        try:
            if action == "run":
                from agent_utilities.rlm.runner import run_rlm

                return json.dumps(
                    await run_rlm(task, input_text=input_text), default=str
                )
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
