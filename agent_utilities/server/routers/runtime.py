"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating / ORCH-1.46 — HTTP surface for the developer-workspace runtime.

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
import math
import re
import time
import uuid
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_utilities.runtime import DevWorkspace, DockerWorkspace, action_policy_gate
from agent_utilities.runtime.events import ACTION_ADAPTER
from agent_utilities.security.error_surface import public_error_payload

logger = logging.getLogger(__name__)
_MAX_EVENT_BYTES = 256 * 1024
_MAX_EVENT_STRING = 16_384

router = APIRouter(tags=["Runtime"], prefix="/api/runtime")


class _Session:
    def __init__(self, ws: Any, owner: str, max_events: int) -> None:
        self.ws = ws
        self.owner = owner
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.subscribers: set[asyncio.Queue] = set()
        self.last_access = time.monotonic()
        self.lock = asyncio.Lock()

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        from agent_utilities.security.persistence_privacy import (
            sanitize_for_persistence,
        )

        clean, _ = sanitize_for_persistence(event)
        clean = _bound_event_value(clean)
        if (
            not isinstance(clean, dict)
            or len(json.dumps(clean, default=str).encode("utf-8")) > _MAX_EVENT_BYTES
        ):
            clean = {"event": "truncated"}
        self.events.append(clean)
        self.last_access = time.monotonic()
        for q in list(self.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(clean)
        return clean


_SESSIONS: dict[str, _Session] = {}
_SESSION_REGISTRY_LOCK = asyncio.Lock()


def _bound_event_value(value: Any, *, depth: int = 0) -> Any:
    """Bound an event after privacy sanitization and before retention/fan-out."""
    if depth >= 8:
        return "[truncated]"
    if isinstance(value, str):
        return value[:_MAX_EVENT_STRING]
    if isinstance(value, dict):
        return {
            str(key)[:128]: _bound_event_value(item, depth=depth + 1)
            for key, item in list(value.items())[:64]
        }
    if isinstance(value, (list, tuple)):
        return [_bound_event_value(item, depth=depth + 1) for item in value[:128]]
    return value


class CreateSessionRequest(BaseModel):
    prefer_docker: bool = True
    image: str


def _require_runtime_capability(request: Request, *, mutate: bool) -> None:
    """Authorize the high-impact workspace surface from validated identity."""
    claims = getattr(request.state, "user_claims", None)
    if not claims:
        # The outer server boundary permits this only on an intentionally open,
        # loopback listener, which is the local-trust profile.
        return
    if claims.get("auth_type") == "api_key":
        return
    try:
        from agent_utilities.core.config import config
        from agent_utilities.security.identity import (
            base_capabilities,
            normalize_identity,
        )

        capabilities = set(
            base_capabilities(
                normalize_identity(claims), config.identity_group_capability_map
            )
        )
    except Exception:
        raise HTTPException(
            status_code=403, detail="runtime capability required"
        ) from None
    required = {"runtime:execute", "runtime:admin", "admin"}
    if not mutate:
        required.add("runtime:read")
    if not capabilities.intersection(required):
        raise HTTPException(status_code=403, detail="runtime capability required")


def _validate_action_bounds(action: Any) -> None:
    """Apply protocol-independent bounds before dispatching executable input."""
    payload = action.model_dump()
    for field in ("path", "cwd", "selector", "url"):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, str)
            or len(value.encode("utf-8")) > 4_096
            or any(ord(character) < 32 for character in value)
        ):
            raise HTTPException(status_code=422, detail="invalid action field")
    for field, limit in (
        ("command", 65_536),
        ("content", 1024 * 1024),
        ("old", 1024 * 1024),
        ("new", 1024 * 1024),
        ("text", 65_536),
        ("keys", 4_096),
        ("interaction", 65_536),
    ):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, str) or len(value.encode("utf-8")) > limit
        ):
            raise HTTPException(status_code=413, detail="action field too large")
    timeout = payload.get("timeout")
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 1.0 <= float(timeout) <= 900.0
    ):
        raise HTTPException(status_code=422, detail="invalid action timeout")


def _owner(request: Request) -> str:
    claims = getattr(request.state, "user_claims", None) or {}
    raw = (
        claims.get("tenant_id")
        or claims.get("tenant")
        or claims.get("sub")
        or claims.get("client_id")
        or claims.get("auth_type")
        or "local"
    )
    from agent_utilities.security.persistence_privacy import persistence_reference

    return persistence_reference("runtime_owner", raw)


async def _reap_sessions() -> None:
    from agent_utilities.core.config import config

    cutoff = time.monotonic() - config.runtime_session_ttl_seconds
    expired = [
        sid for sid, session in _SESSIONS.items() if session.last_access < cutoff
    ]
    for sid in expired:
        session = _SESSIONS.pop(sid, None)
        if session is not None:
            async with session.lock:
                with contextlib.suppress(Exception):
                    await session.ws.stop()


def _owned_session(sid: str, request: Request) -> _Session:
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        raise HTTPException(status_code=404, detail="session not found")
    session = _SESSIONS.get(sid)
    if session is None or session.owner != _owner(request):
        # Do not reveal whether another tenant owns the identifier.
        raise HTTPException(status_code=404, detail="session not found")
    session.last_access = time.monotonic()
    return session


@router.post("/sessions")
async def create_session(req: CreateSessionRequest, request: Request) -> dict[str, Any]:
    from agent_utilities.core.config import config

    _require_runtime_capability(request, mutate=True)
    if not req.prefer_docker:
        raise HTTPException(status_code=403, detail="isolated runtime required")
    async with _SESSION_REGISTRY_LOCK:
        await _reap_sessions()
        owner = _owner(request)
        if len(_SESSIONS) >= config.runtime_max_sessions or sum(
            session.owner == owner for session in _SESSIONS.values()
        ) >= max(1, config.runtime_max_sessions // 2):
            raise HTTPException(status_code=429, detail="runtime session limit reached")
        if req.image not in config.runtime_workspace_images:
            raise HTTPException(
                status_code=403, detail="workspace image is not approved"
            )
        sid = uuid.uuid4().hex
        backend = DockerWorkspace(
            run_id=sid,
            image=req.image,
            network=config.runtime_workspace_network,
        )
        if not backend.is_available():
            await backend.stop()
            raise HTTPException(status_code=503, detail="isolated runtime unavailable")
        ws = DevWorkspace(
            backend,
            run_id=sid,
            actor=owner,
            policy_gate=action_policy_gate(),
        )
        try:
            await ws.start()
        except Exception:
            await ws.stop()
            raise HTTPException(
                status_code=503, detail="isolated runtime unavailable"
            ) from None
        _SESSIONS[sid] = _Session(ws, owner, config.runtime_max_events)
    return {
        "session_id": sid,
        "backend": ws.backend.name,
    }


@router.post("/sessions/{sid}/act")
async def act(sid: str, action: dict[str, Any], request: Request) -> dict[str, Any]:
    _require_runtime_capability(request, mutate=True)
    session = _owned_session(sid, request)
    try:
        typed = ACTION_ADAPTER.validate_python(action)
    except Exception as exc:  # noqa: BLE001 - surface a 422-style error to the caller
        raise HTTPException(
            status_code=422,
            detail=public_error_payload(exc, logger=logger, code="invalid_request"),
        ) from None
    _validate_action_bounds(typed)
    async with session.lock:
        observation = await session.ws.act(typed)
        obs_dict = observation.model_dump()
        stored = session.publish(
            {"action": typed.model_dump(), "observation": obs_dict}
        )
    observation_payload = stored.get("observation")
    return (
        observation_payload
        if isinstance(observation_payload, dict)
        else {"kind": "error", "message": "event exceeded the response boundary"}
    )


@router.get("/sessions/{sid}")
async def status(sid: str, request: Request) -> dict[str, Any]:
    _require_runtime_capability(request, mutate=False)
    session = _owned_session(sid, request)
    return {
        "session_id": sid,
        "backend": session.ws.backend.name,
        "steps": len(session.events),
    }


@router.get("/sessions/{sid}/events")
async def stream_events(sid: str, request: Request) -> StreamingResponse:
    _require_runtime_capability(request, mutate=False)
    session = _owned_session(sid, request)
    if len(session.subscribers) >= 16:
        raise HTTPException(status_code=429, detail="subscriber limit reached")

    async def gen():
        # Replay the log so a late subscriber sees the whole trajectory, then stream live.
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        for event in list(session.events):
            yield f"data: {json.dumps(event)}\n\n"
        session.subscribers.add(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            session.subscribers.discard(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{sid}/provenance")
async def provenance(sid: str, request: Request) -> dict[str, Any]:
    """CONCEPT:AU-OS.scaling.kg-provenance-panel-data — the KG-provenance panel data for a run: the action/observation
    trajectory and the ``Code`` symbols each edit mutated (KG-2.64).

    This is what the agent-webui SWE view renders alongside the live SSE event stream — the
    differentiator over OpenHands' flat log: you see the symbol graph the agent reasoned over.
    Best-effort: returns an empty graph when the KG is cold.
    """
    _require_runtime_capability(request, mutate=False)
    _owned_session(sid, request)
    return _run_provenance(sid)


def _run_provenance(run_id: str) -> dict[str, Any]:
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.observability.trace_ontology import trace_id

        trace_ref = trace_id(run_id)
        engine = IntelligenceGraphEngine.get_active()
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        if not callable(execute):
            return {"trace_ref": trace_ref, "actions": [], "mutated": []}
        actions = execute(
            "MATCH (a:WorkspaceAction {trace_id: $trace_ref}) "
            "OPTIONAL MATCH (a)-[:PRODUCED]->(o:WorkspaceObservation) "
            "RETURN a.id AS id, a.kind AS kind, a.step AS step, "
            "a.payload_ref AS payload_ref, "
            "a.payload_field_count AS payload_field_count, "
            "o.kind AS obs_kind, o.payload_ref AS obs_payload_ref, "
            "o.payload_field_count AS obs_payload_field_count ORDER BY a.step",
            {"trace_ref": trace_ref},
        )
        mutated = execute(
            "MATCH (a:WorkspaceAction {trace_id: $trace_ref})-[:MUTATED]->(c:Code) "
            "RETURN a.id AS action_id, c.id AS symbol_id",
            {"trace_ref": trace_ref},
        )
        result = {
            "trace_ref": trace_ref,
            "actions": [r for r in (actions or []) if isinstance(r, dict)],
            "mutated": [r for r in (mutated or []) if isinstance(r, dict)],
        }
        from agent_utilities.security.persistence_privacy import (
            sanitize_for_persistence,
        )

        return sanitize_for_persistence(result)[0]
    except Exception as exc:  # noqa: BLE001 - KG optional
        logger.debug("provenance query failed (exception_type=%s)", type(exc).__name__)
        return {"trace_ref": "unavailable", "actions": [], "mutated": []}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, request: Request) -> dict[str, Any]:
    _require_runtime_capability(request, mutate=True)
    session = _owned_session(sid, request)
    _SESSIONS.pop(sid, None)
    async with session.lock:
        await session.ws.stop()
    return {"session_id": sid, "stopped": True}
