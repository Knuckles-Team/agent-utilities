import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent_utilities.observability.approval_manager import ApprovalManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Human-in-the-Loop"])

# Singleton approval manager shared between the graph executor and
# the /api/approve endpoint. Created once at module import.
_approval_manager = ApprovalManager()


@router.post("/api/approve", summary="Resolve a pending tool approval or elicitation")
async def resolve_approval(request: Request):
    """Resolve a pending approval request from the graph executor.

    Expected JSON body::

        {
            "request_id": "<id from approval_required event>",
            "decisions": {
                "<tool_call_id>": "accept" | "deny",
                ...
            },
            "feedback": "optional text"
        }

    """
    try:
        data = await request.json()
        rid = data.get("request_id") or data.get("id")
        if not rid:
            return JSONResponse({"error": "request_id is required"}, status_code=400)
        if _approval_manager.resolve(rid, data):
            return {"status": "resolved", "request_id": rid}
        return JSONResponse(
            {"error": "Request not found or already resolved"},
            status_code=404,
        )
    except Exception as e:
        logger.exception("Approval resolution error")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post(
    "/api/runs/{run_id}/tool-result",
    summary="Inject a tool result to resume a run paused mid-turn (CONCEPT:ORCH-1.35)",
)
async def submit_tool_result(run_id: str, request: Request):
    """Resolve a run paused on a ``tool_use`` so it resumes the same turn.

    Expected JSON body::

        {"tool_use_id": "<id>" (optional), "result": { ... }}

    If ``tool_use_id`` is omitted, the first pending tool_use for the run is resolved.
    """
    from agent_utilities.core.execution.held_turns import get_held_turn_registry

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    result = data.get("result", data)
    tool_use_id = data.get("tool_use_id")
    registry = get_held_turn_registry()
    resolved = (
        registry.resolve(run_id, tool_use_id, result)
        if tool_use_id
        else registry.resolve_any(run_id, result)
    )
    if not resolved:
        return JSONResponse(
            {"error": "no run waiting for a tool result", "run_id": run_id},
            status_code=404,
        )
    return {"status": "resumed", "run_id": run_id}
