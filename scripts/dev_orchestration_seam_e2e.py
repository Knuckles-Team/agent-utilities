#!/usr/bin/env python3
"""LIVE e2e harness for the orchestration EXECUTION SEAM.

Proves, end-to-end against the running engine + the local vLLM, that an *ingested
capability* is actually EXECUTED by a local LLM through a real MCP tool call, with
full provenance visible in the epistemic-graph:

  goal
   ├─ (ORCH-1.96) an ingested ``:Skill`` node resolves to a runnable CallableResource
   ├─ (ORCH-1.95) ``execute_workflow`` runs the STORED step-DAG via ``run_agent``
   │             (not the old ``dynamic_worker`` fallback)
   ├─ ``run_agent`` on the LOCAL vLLM binds a real MCP toolset and calls a tool
   ├─ (KG-2.296)  each tool call is persisted as a ``:ToolCall`` node on the RunTrace
   └─ (ORCH-1.97) the delegated run returns a trackable ``run_id`` handle

It stands up a tiny **local stdio MCP server** exposing ONE read-only tool
(``health_probe`` → the host load average) so the proof is non-destructive and needs
NO secrets. (A jwt-protected ``*.arpa`` fleet server exercises the identical
``run_agent`` path but needs the human-gated OIDC service-account creds — see
``scripts/dev_execute_agent.py`` and AGENTS.md → "Secrets & credential retrieval".)

Usage::

    python scripts/dev_orchestration_seam_e2e.py

Requires: a reachable engine (``GRAPH_SERVICE_SOCKET``) + a reachable chat model
(``config.json`` ``chat_models``, e.g. the GB10 qwen vLLM). Read-only; safe to run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

_PROBE_SERVER = '''
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("health-probe-mcp")


@mcp.tool()
def health_probe() -> str:
    """Return the host 1/5/15-minute load average and CPU count (read-only)."""
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        one = five = fifteen = -1.0
    return (
        f"HEALTH_OK load1={one:.2f} load5={five:.2f} load15={fifteen:.2f} "
        f"cpus={os.cpu_count()}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


async def main() -> int:
    from agent_utilities.core.config import load_config

    load_config()

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine()
    server_name = "health-probe-mcp"

    # Write the fixture stdio MCP server to a tempfile and register it as a Server.
    fd, probe_path = tempfile.mkstemp(suffix="_probe_server.py")
    with os.fdopen(fd, "w") as fh:
        fh.write(_PROBE_SERVER)
    py = sys.executable
    engine.add_node(
        f"srv:{server_name}",
        "Server",
        properties={
            "id": f"srv:{server_name}",
            "name": server_name,
            "url": f"stdio://{py} {probe_path}",
            "command": py,
            "args": probe_path,
            "tool_count": 1,
        },
    )

    ok = {
        "run_agent": False,
        "toolcalls": False,
        "run_id": False,
        "gap1": False,
        "gap2": False,
    }

    # ---- run_agent → local vLLM → real MCP tool call (ORCH-1.96 path + KG-2.296) ----
    from agent_utilities.orchestration.agent_runner import run_agent

    out = await asyncio.wait_for(
        run_agent(
            agent_name=server_name,
            task="Call the health_probe tool and report the host load average.",
            engine=engine,
            max_steps=3,
            return_mermaid=True,
        ),
        timeout=180,
    )
    payload = json.loads(out) if out.strip().startswith("{") else {"output": out}
    run_id = payload.get("run_id")
    print("OUTPUT:", str(payload.get("output"))[:300])
    print("RUN_ID:", run_id)
    ok["run_agent"] = (
        "HEALTH_OK" in str(payload.get("output"))
        or "load" in str(payload.get("output")).lower()
    )
    ok["run_id"] = bool(run_id and str(run_id).startswith("run:"))

    # ---- Visibility: query the RunTrace's :ToolCall provenance (KG-2.296) ----
    if run_id:
        rows = engine.backend.execute(
            "MATCH (t:RunTrace {id: $tid})-[:MADE_TOOL_CALL]->(tc:ToolCall) "
            "RETURN tc.tool_name AS tool, tc.status AS status, tc.result_preview AS result",
            {"tid": f"trace:{run_id}"},
        )
        print(f"ToolCall nodes under trace:{run_id}: {len(rows or [])}")
        for r in rows or []:
            print(
                f"  - {r.get('tool')} [{r.get('status')}] {str(r.get('result'))[:80]}"
            )
        ok["toolcalls"] = bool(rows)

    # ---- GAP1 (ORCH-1.95): execute_workflow runs the stored DAG ----
    from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
    from agent_utilities.models.graph import ExecutionStep, GraphPlan
    from agent_utilities.orchestration.manager import Orchestrator

    plan = GraphPlan(
        steps=[
            ExecutionStep(
                id=server_name,
                refined_subtask="Call health_probe and report the load average.",
            )
        ],
        metadata={"query": "host health"},
    )
    wf_name = f"seam-e2e-{os.getpid()}"
    WorkflowStore(engine).save_workflow(name=wf_name, plan=plan, description="seam e2e")
    wf = await Orchestrator(engine).execute_workflow(workflow_id=wf_name, task="health")
    step_nodes = [s.get("node_id") for s in wf.get("step_results", [])]
    print(
        "workflow status:",
        wf.get("status"),
        "steps:",
        step_nodes,
        "run_id:",
        wf.get("run_id"),
    )
    ok["gap1"] = bool(step_nodes) and all(
        "dynamic_worker" not in str(n) for n in step_nodes
    )

    # ---- GAP2 (ORCH-1.96): a bare :Skill resolves to a runnable CallableResource ----
    from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg

    skill_name = f"seam-skill-{os.getpid()}"
    engine.add_node(
        f"skill:{skill_name}",
        "Skill",
        properties={
            "id": f"skill:{skill_name}",
            "name": skill_name,
            "source": "universal-skills",
            "body": "You check host health. Call health_probe and summarize.",
        },
    )
    meta = _resolve_agent_from_kg(engine, skill_name)
    bound = engine.backend.execute(
        "MATCH (r:CallableResource) WHERE r.name = $n "
        "RETURN r.id AS id, r.resource_type AS rt, r.runnable_bound AS b",
        {"n": skill_name},
    )
    print("skill resolved type:", meta.get("type"), "bound resource:", bound)
    ok["gap2"] = (
        meta.get("type") == "skill"
        and bool(meta.get("system_prompt"))
        and bool(bound)
        and bound[0].get("rt") == "AGENT_SKILL"
    )

    os.unlink(probe_path)
    print("\nRESULT:", json.dumps(ok))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
