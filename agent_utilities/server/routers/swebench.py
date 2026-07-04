"""CONCEPT:AU-AHE.harness.swebench-http-surface — SWE-bench harness HTTP surface (run + report).

Mirrors the LongMemEval benchmark router shape: POST a set of instances to run, GET the aggregate
report. Optionally files failure-gap Concepts for unresolved instances (failure-driven
remediation, AHE-3.23) so the golden loop can self-improve.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["SWE-bench"], prefix="/api/swebench")

_RUNS: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(default_factory=list)
    ingest: bool = False
    remediate: bool = True
    prefer_docker: bool = True


@router.post("/run")
async def run(req: RunRequest) -> dict[str, Any]:
    """Run the SWE-bench suite over the posted instances and return the aggregate report."""
    if not req.instances:
        raise HTTPException(status_code=422, detail="no instances provided")

    from agent_utilities.harness.swebench_corpus import load_instances
    from agent_utilities.harness.swebench_harness import run_suite
    from agent_utilities.harness.swebench_remediation import remediate
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.runtime import create_workspace

    instances = load_instances(req.instances)

    def factory(inst: Any) -> Any:
        return create_workspace(
            run_id=f"swe-{inst.instance_id or uuid.uuid4().hex[:8]}",
            prefer_docker=req.prefer_docker,
        )

    engine = IntelligenceGraphEngine.get_active()
    suite = await run_suite(
        instances, workspace_factory=factory, ingest=req.ingest, kg=engine
    )
    report = suite.report

    remediation: dict[str, Any] | None = None
    if req.remediate and engine is not None:
        remediation = remediate(suite.results, engine)

    run_id = f"swebench:{uuid.uuid4().hex[:8]}"
    _RUNS[run_id] = {"report": report, "remediation": remediation}
    return {"run_id": run_id, "report": report, "remediation": remediation}


@router.get("/report/{run_id}")
async def report(run_id: str) -> dict[str, Any]:
    data = _RUNS.get(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return data
