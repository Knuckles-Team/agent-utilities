"""Focused graph-os surface for the unified durable-execution plane (DE2).

One action-routed tool over :mod:`agent_utilities.orchestration.durable_tool_surface`
-- see that module's docstring for exactly which of DE0's five ``CallShape``s are
backed today and why. Matches ``graph_jobs``'s pattern (``mcp/tools/job_tools.py``):
one tool, an ``action`` dispatch, the same engine resolution.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_durable_tools(mcp: Any) -> None:
    """Register the unified durable-execution tool surface."""

    @mcp.tool(
        name="graph_durable",
        description=(
            "The unified durable-execution plane (durable-execution-native.md, "
            "supersedes restate natively). Actions: 'run' (start-or-resume a "
            "named DurableRun, returns run_id/resumed/definition_version), "
            "'checkpoint' (DE3: durably record that name already happened, "
            "auto-sequenced -- no hand-picked unique step name needed), "
            "'sleep' (a durable timer bound to one run's one named step -- "
            "fires once wake_at_ms has passed, durably WAITING until then), "
            "'status' (query the :DurableExecutionUnit KG family -- 'what is "
            "durably in flight, waiting on what' across mutation-store/jobs/"
            "statechart/DurableRun). 'state_get'/'state_set'/'call' are named "
            "by DE0's routing contract but NOT backed yet (they route to "
            "eg-statechart/eg-mutation-store, which have no Python-reachable "
            "wire-protocol surface today) -- calling them returns a clear "
            "error naming the gap, never a silent no-op."
        ),
        tags=["graph-os", "durable-execution", "orchestration"],
    )
    async def graph_durable(
        action: str = Field(
            default="status",
            description="run | checkpoint | sleep | status | state_get | state_set | call",
        ),
        session_id: str = Field(
            default="",
            description="Durable run session id (run|checkpoint|sleep|state_get|state_set|call).",
        ),
        name: str = Field(
            default="",
            description="Named step/state key (checkpoint|sleep|state_get|state_set|call).",
        ),
        wake_at_ms: int = Field(
            default=0, description="'sleep' only: epoch-ms deadline to wake at."
        ),
        value: str = Field(
            default="",
            description="'state_set'/'checkpoint' only: JSON value to write/record.",
        ),
        definition_version: str = Field(
            default="",
            description="Optional DE5 version pin for 'run'|'checkpoint'|'sleep'.",
        ),
        kinds: str = Field(
            default="", description="'status' only: JSON list of labels to filter to."
        ),
        status: str = Field(
            default="", description="'status' only: filter by durable_status."
        ),
        limit: int = Field(default=100, description="'status' only: max rows."),
    ) -> str:
        engine = kg_server._get_engine()
        from agent_utilities.orchestration.durable_tool_surface import (
            DurableCallNotBacked,
            durable_call,
            durable_checkpoint,
            durable_run,
            durable_sleep,
            durable_state_get,
            durable_state_set,
            durable_status,
        )

        try:
            if action == "run":
                if not session_id:
                    return "Error: session_id required"
                return json.dumps(
                    durable_run(
                        engine,
                        session_id,
                        definition_version=definition_version or None,
                    ),
                    default=str,
                )
            if action == "checkpoint":
                if not session_id or not name:
                    return "Error: session_id and name required"
                parsed_result = json.loads(value) if value else None
                return json.dumps(
                    durable_checkpoint(
                        engine,
                        session_id,
                        name,
                        parsed_result,
                        definition_version=definition_version or None,
                    ),
                    default=str,
                )
            if action == "sleep":
                if not session_id or not name:
                    return "Error: session_id and name required"
                return json.dumps(
                    durable_sleep(
                        engine,
                        session_id,
                        name,
                        wake_at_ms,
                        definition_version=definition_version or None,
                    ),
                    default=str,
                )
            if action == "status":
                if engine is None:
                    return "Error: IntelligenceGraphEngine not active."
                parsed_kinds = tuple(json.loads(kinds)) if kinds else None
                rows = durable_status(
                    engine,
                    kinds=parsed_kinds,
                    status=status or None,
                    limit=limit,
                )
                return json.dumps(rows, default=str)
            if action == "state_get":
                return json.dumps(
                    durable_state_get(engine, session_id, name), default=str
                )
            if action == "state_set":
                parsed_value = json.loads(value) if value else None
                return json.dumps(
                    durable_state_set(engine, session_id, name, parsed_value),
                    default=str,
                )
            if action == "call":
                return json.dumps(durable_call(engine, session_id, name), default=str)
            return f"Error: unknown action {action!r}"
        except DurableCallNotBacked as exc:
            return f"Error: {exc}"

    # Matches register_job_tools's own pattern (mcp/tools/job_tools.py): an
    # explicit REGISTERED_TOOLS entry (belt-and-suspenders alongside the
    # @mcp.tool() decorator) plus the REST twin, so
    # kg_server._mount_rest_routes's generic ACTION_TOOL_ROUTES loop mounts
    # POST /api/graph/durable for free (Two surfaces by default).
    kg_server.REGISTERED_TOOLS["graph_durable"] = graph_durable
    kg_server.ACTION_TOOL_ROUTES["graph_durable"] = "/graph/durable"
