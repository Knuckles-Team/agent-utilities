"""CONCEPT:OS-5.33 / ORCH-1.46 — HTTP surface for the developer-workspace runtime.

Lets the gateway (and the agent-webui SWE view, OS-5.34) drive a sandboxed workspace over REST:
create a session, post typed actions, and stream the action/observation event log over SSE. The
session registry holds live :class:`~agent_utilities.runtime.DevWorkspace` objects (each owns a
container/subprocess), so sessions must be explicitly deleted (or reaped) to release resources.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_utilities.runtime import create_workspace
from agent_utilities.runtime.events import ACTION_ADAPTER

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runtime"], prefix="/api/runtime")


class _Session:
    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.events: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for q in list(self.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)


_SESSIONS: dict[str, _Session] = {}


class CreateSessionRequest(BaseModel):
    prefer_docker: bool = True
    image: str = "python:3.11-slim"
    actor: str | None = None


@router.post("/sessions")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    sid = uuid.uuid4().hex[:12]
    ws = create_workspace(
        run_id=sid, prefer_docker=req.prefer_docker, image=req.image, actor=req.actor
    )
    await ws.start()
    _SESSIONS[sid] = _Session(ws)
    return {
        "session_id": sid,
        "backend": ws.backend.name,
        "workdir": ws.backend.workdir,
    }


@router.post("/sessions/{sid}/act")
async def act(sid: str, action: dict[str, Any]) -> dict[str, Any]:
    session = _SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {sid}")
    try:
        typed = ACTION_ADAPTER.validate_python(action)
    except Exception as exc:  # noqa: BLE001 - surface a 422-style error to the caller
        raise HTTPException(status_code=422, detail=f"invalid action: {exc}") from exc
    observation = await session.ws.act(typed)
    obs_dict = observation.model_dump()
    session.publish({"action": typed.model_dump(), "observation": obs_dict})
    return obs_dict


@router.get("/sessions/{sid}")
async def status(sid: str) -> dict[str, Any]:
    session = _SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {sid}")
    return {
        "session_id": sid,
        "backend": session.ws.backend.name,
        "cwd": session.ws.state.cwd,
        "steps": len(session.events),
    }


@router.get("/sessions/{sid}/events")
async def stream_events(sid: str) -> StreamingResponse:
    session = _SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {sid}")

    async def gen():
        # Replay the log so a late subscriber sees the whole trajectory, then stream live.
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        for event in list(session.events):
            yield f"data: {json.dumps(event)}\n\n"
        session.subscribers.add(q)
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            session.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/sessions/{sid}/provenance")
async def provenance(sid: str) -> dict[str, Any]:
    """CONCEPT:OS-5.34 — the KG-provenance panel data for a run: the action/observation
    trajectory and the ``Code`` symbols each edit mutated (KG-2.64).

    This is what the agent-webui SWE view renders alongside the live SSE event stream — the
    differentiator over OpenHands' flat log: you see the symbol graph the agent reasoned over.
    Best-effort: returns an empty graph when the KG is cold.
    """
    return _run_provenance(sid)


def _run_provenance(run_id: str) -> dict[str, Any]:
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        if not callable(execute):
            return {"run_id": run_id, "actions": [], "mutated": []}
        actions = execute(
            "MATCH (a:WorkspaceAction {run_id: $rid}) "
            "OPTIONAL MATCH (a)-[:PRODUCED]->(o:WorkspaceObservation) "
            "RETURN a.id AS id, a.kind AS kind, a.step AS step, a.summary AS summary, "
            "o.kind AS obs_kind, o.summary AS obs_summary ORDER BY a.step",
            {"rid": run_id},
        )
        mutated = execute(
            "MATCH (a:WorkspaceAction {run_id: $rid})-[:MUTATED]->(c:Code) "
            "RETURN a.id AS action_id, c.id AS symbol_id",
            {"rid": run_id},
        )
        return {
            "run_id": run_id,
            "actions": [r for r in (actions or []) if isinstance(r, dict)],
            "mutated": [r for r in (mutated or []) if isinstance(r, dict)],
        }
    except Exception as exc:  # noqa: BLE001 - KG optional
        logger.debug("provenance query failed: %s", exc)
        return {"run_id": run_id, "actions": [], "mutated": []}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str) -> dict[str, Any]:
    session = _SESSIONS.pop(sid, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {sid}")
    await session.ws.stop()
    return {"session_id": sid, "stopped": True}
