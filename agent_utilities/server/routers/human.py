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
