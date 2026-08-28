import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agent_utilities.core.config import (
    DEFAULT_APPROVAL_TIMEOUT,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
)
from agent_utilities.core.workspace import WORKSPACE_DIR
from agent_utilities.security.error_surface import public_error_payload

from ...models import AgentDeps
from ..dependencies import _build_model_from_registry, process_parts

logger = logging.getLogger(__name__)

_MAX_QUERY_BYTES = 1024 * 1024
_STREAM_SELECTOR_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")

_OWNER_CLAIM_KEYS = ("tenant_id", "tenant", "sub", "client_id", "auth_type")


async def _require_agent_invoke(request: Request) -> None:
    claims = getattr(request.state, "user_claims", None)
    if not claims or claims.get("auth_type") == "api_key":
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
            status_code=403, detail="agent invocation capability required"
        ) from None
    if not capabilities.intersection(
        {"agent:invoke", "model:invoke", "agent:admin", "admin"}
    ):
        raise HTTPException(
            status_code=403, detail="agent invocation capability required"
        )


router = APIRouter(tags=["Agent UI"], dependencies=[Depends(_require_agent_invoke)])


def _resolve_run_owner(claims: dict) -> str:
    """First non-empty identity claim, in priority order, else ``"local"``."""
    for key in _OWNER_CLAIM_KEYS:
        value = claims.get(key)
        if value:
            return value
    return "local"


def _scoped_run_id(request: Request, supplied: Any = None) -> str:
    """Bind a caller-provided continuity key to the authenticated identity."""
    claims = getattr(request.state, "user_claims", None) or {}
    owner = _resolve_run_owner(claims)
    candidate = str(supplied or "").strip()
    if len(candidate) > 256 or any(character in candidate for character in "\r\n\x00"):
        candidate = ""
    from agent_utilities.security.persistence_privacy import persistence_reference

    return persistence_reference("agent_run", f"{owner}\x00{candidate or 'new'}")


async def _parse_ag_ui_request(request: Request) -> tuple[Response | None, dict | None]:
    """Parse and validate the AG-UI request body.

    Returns ``(error_response, None)`` on failure or ``(None, parsed)`` on
    success — exactly one of the pair is ``None``.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON request"}, status_code=422), None
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid request"}, status_code=422), None
    query = body.get("query", body.get("prompt", ""))
    if not isinstance(query, str):
        return JSONResponse({"error": "invalid query"}, status_code=422), None
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        return JSONResponse({"error": "query too large"}, status_code=413), None
    raw_parts = body.get("parts", [])
    try:
        query_parts = await process_parts(raw_parts) if raw_parts else []
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return JSONResponse(
                {"error": exc.detail}, status_code=exc.status_code
            ), None
        return JSONResponse({"error": "invalid message parts"}, status_code=422), None
    return None, {"body": body, "query": query, "query_parts": query_parts}


@dataclass
class _AgUiFastPathCtx:
    """Bundled args for the direct graph-iter streaming path (avoids a
    long parameter list on the extracted generator)."""

    graph: Any
    graph_cfg: dict
    query: str
    run_id: str
    session_id: Any
    initialized_mcp_toolsets: list
    requested_model_id: Any
    graph_event_queue: "asyncio.Queue[Any]"


def _prime_fast_path_state(session_id: Any, graph_cfg: dict) -> tuple[Any, dict]:
    """Best-effort: get the active KG engine and recall the per-session
    memento keyed by the caller's raw ``session_id`` (see D-TC-6 — the seam
    is keyed by the raw session id, NOT the pseudonymized ``run_id``),
    merging it into the graph exec config.

    Returns ``(kg_engine_or_None, exec_cfg)``. Any failure anywhere in this
    best-effort priming is swallowed and the caller gets back ``(None,
    graph_cfg)`` or a partially-primed state, matching the original inline
    ``with suppress(Exception):`` block exactly.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.orchestration.session_continuity import (
        prime_session_context,
    )

    kg_engine = None
    exec_cfg = graph_cfg
    with suppress(Exception):
        kg_engine = IntelligenceGraphEngine.get_active()
        primed = prime_session_context(kg_engine, session_id)
        if primed:
            exec_cfg = {**graph_cfg, "invoker_context": primed}
    return kg_engine, exec_cfg


async def _persist_fast_path_turn(
    kg_engine: Any, session_id: Any, query: str, final_output: str, run_id: str
) -> None:
    """Best-effort: persist the turn (RunTrace + per-session memento) off
    the reply path so the NEXT turn — on this surface OR any other keyed to
    the same session — recalls it. Never affects the stream."""
    if kg_engine is None or not final_output:
        return
    with suppress(Exception):
        from agent_utilities.orchestration.session_continuity import (
            persist_session_turn,
        )
        from agent_utilities.security.persistence_privacy import (
            PersistencePrivacyGuard,
        )

        guard = PersistencePrivacyGuard()
        clean_query = guard.sanitize_text(query)[0]
        clean_output = guard.sanitize_text(final_output)[0]
        asyncio.create_task(
            persist_session_turn(
                kg_engine,
                # D-TC-6: key off the caller's raw ``session_id``, not the
                # pseudonymized ``run_id``.
                session_id,
                clean_query,
                clean_output,
                agent_name="agent-ui",
                run_id=run_id,
            )
        )


def _drain_sideband_chunks(
    queue: "asyncio.Queue[Any]", emitter: Any
) -> Iterator[bytes]:
    while not queue.empty():
        ev = queue.get_nowait()
        if ev:
            yield from emitter._format_sideband(ev)


async def _ag_ui_fast_path_stream(ctx: _AgUiFastPathCtx) -> AsyncIterator[bytes]:
    from agent_utilities.protocols.agui_emitter import AGUIGraphEmitter

    from ...graph.protocol_agnostic_execution import execute_graph_iter

    logger.info("AG-UI direct graph execution")
    kg_engine, exec_cfg = _prime_fast_path_state(ctx.session_id, ctx.graph_cfg)

    emitter = AGUIGraphEmitter()
    final_output: str = ""
    try:
        async for event in execute_graph_iter(
            graph=ctx.graph,
            config=exec_cfg,
            query=ctx.query,
            run_id=ctx.run_id,
            mode="ask",
            mcp_toolsets=ctx.initialized_mcp_toolsets,
            requested_model_id=ctx.requested_model_id,
        ):
            if isinstance(event, dict) and event.get("type") in (
                "graph_complete",
                "final_output",
            ):
                out = event.get("output") or event.get("content")
                if out:
                    final_output = str(out)
            for chunk in emitter.translate(event):
                yield chunk
            for chunk in _drain_sideband_chunks(ctx.graph_event_queue, emitter):
                yield chunk
    except Exception as exc:
        error_data = json.dumps(
            {"type": "error", **public_error_payload(exc, logger=logger)}
        )
        yield f"data: {error_data}\n\n".encode()
    finally:
        await _persist_fast_path_turn(
            kg_engine, ctx.session_id, ctx.query, final_output, ctx.run_id
        )


@dataclass
class _AgUiAdapterPathCtx:
    """Bundled args for the AGUIAdapter streaming path."""

    request: Request
    agent_instance: Any
    override_model: Any
    query: str
    query_parts: Any
    deps: AgentDeps
    graph_event_queue: "asyncio.Queue[Any]"
    elicitation_queue: "asyncio.Queue[Any]"


async def _dispatch_agent_response(
    ctx: _AgUiAdapterPathCtx,
) -> tuple[bytes | None, Any]:
    """Build the override context and dispatch through AGUIAdapter.

    Returns ``(error_chunk, None)`` on failure or ``(None, agent_response)``
    on success.
    """
    from pydantic_ai.ui.ag_ui import AGUIAdapter

    if ctx.agent_instance is None:
        raise RuntimeError("Agent instance not initialized on app state")
    run_input: Any = ctx.query_parts if ctx.query_parts else ctx.query
    override_ctx = (
        ctx.agent_instance.override(model=ctx.override_model)
        if ctx.override_model is not None
        else nullcontext()
    )
    try:
        with override_ctx:
            adapter = AGUIAdapter(agent=ctx.agent_instance, run_input=run_input)
            logger.info("[LAYER:ACP] AG-UI: Dispatching request")
            if ctx.override_model is not None:
                logger.info("AG-UI: Applying authorized per-turn model override")
            agent_response = await adapter.dispatch_request(
                ctx.request, agent=ctx.agent_instance, deps=ctx.deps
            )
        logger.info("[LAYER:ACP] AG-UI: Dispatch successful. Stream established.")
    except Exception as exc:
        failure = {"type": "error", **public_error_payload(exc, logger=logger)}
        return f"data: {json.dumps(failure)}\n\n".encode(), None
    return None, agent_response


def _stream_chunk_bytes(chunk: Any) -> bytes | bytearray:
    if isinstance(chunk, bytes | bytearray):
        return chunk
    if isinstance(chunk, memoryview):
        return bytes(chunk)
    return chunk.encode("utf-8")


async def _poll_agent_response(
    agent_response: Any, combined_queue: "asyncio.Queue[Any]"
) -> None:
    try:
        async for chunk in agent_response.body_iterator:
            chunk_str = (
                chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            )
            await combined_queue.put(("chunk", _stream_chunk_bytes(chunk)))
            if (
                chunk_str.startswith("2:")
                or chunk_str.startswith("9:")
                or '"tool_calls"' in chunk_str
            ):
                await combined_queue.put(("chunk", b'0 " "\n'))
                await asyncio.sleep(0.01)
    except Exception as exc:
        logger.error("Agent stream error (exception_type=%s)", type(exc).__name__)
    finally:
        await combined_queue.put(("done", None))


async def _emit_sideband_event(
    task: "asyncio.Task[Any]", combined_queue: "asyncio.Queue[Any]"
) -> None:
    try:
        ev = await task
        if ev:
            packet = f"8:{json.dumps(ev)}\n".encode()
            await combined_queue.put(("chunk", packet))
            await combined_queue.put(("chunk", b'0 " "\n'))
            await asyncio.sleep(0.01)
    except Exception as exc:
        logger.error(
            "Error processing sideband event (exception_type=%s)", type(exc).__name__
        )


async def _cancel_pending(tasks: Iterable["asyncio.Task[Any]"]) -> None:
    for task in tasks:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _poll_sideband_events(
    graph_event_queue: "asyncio.Queue[Any]",
    elicitation_queue: "asyncio.Queue[Any]",
    combined_queue: "asyncio.Queue[Any]",
) -> None:
    while True:
        try:
            tasks = [
                asyncio.create_task(graph_event_queue.get()),
                asyncio.create_task(elicitation_queue.get()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                await _emit_sideband_event(task, combined_queue)
            await _cancel_pending(pending)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(
                "Sideband poller error (exception_type=%s)", type(exc).__name__
            )
            break


async def _drain_combined_queue(
    combined_queue: "asyncio.Queue[Any]",
    agent_task: "asyncio.Task[Any]",
    graph_event_queue: "asyncio.Queue[Any]",
    elicitation_queue: "asyncio.Queue[Any]",
) -> AsyncIterator[bytes]:
    while True:
        try:
            msg_type, data = await asyncio.wait_for(combined_queue.get(), timeout=0.1)
            if msg_type == "done":
                await asyncio.sleep(0.1)
                if not graph_event_queue.empty() or not elicitation_queue.empty():
                    continue
                break
            yield data
            combined_queue.task_done()
        except TimeoutError:
            yield b'0 " "\n'
            if agent_task.done() and combined_queue.empty():
                break
            continue


async def _ag_ui_adapter_path_stream(ctx: _AgUiAdapterPathCtx) -> AsyncIterator[bytes]:
    error, agent_response = await _dispatch_agent_response(ctx)
    if error is not None:
        yield error
        return

    if not isinstance(agent_response, StreamingResponse):
        yield agent_response.body
        return

    combined_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    agent_task = asyncio.create_task(
        _poll_agent_response(agent_response, combined_queue)
    )
    sideband_task = asyncio.create_task(
        _poll_sideband_events(
            ctx.graph_event_queue, ctx.elicitation_queue, combined_queue
        )
    )

    try:
        async for chunk in _drain_combined_queue(
            combined_queue, agent_task, ctx.graph_event_queue, ctx.elicitation_queue
        ):
            yield chunk
    finally:
        agent_task.cancel()
        sideband_task.cancel()


@router.post("/ag-ui", summary="AG-UI Streaming Endpoint")
async def ag_ui_endpoint(request: Request) -> Response:
    """Primary streaming endpoint for the Agent UI (FastAG-UI).

    CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Supports sideband graph activity annotations, session resumption,
        and rich media attachments. This endpoint handles high-fidelity
        SSE streaming with sideband data.
    """
    try:
        from pydantic_ai.ui.ag_ui import AGUIAdapter  # noqa: F401
    except ImportError:
        logger.error(
            "AG-UI: AGUIAdapter not found in pydantic_ai. Ensure pydantic-ai[ag-ui] is installed."
        )
        return JSONResponse(
            {"status": "error", "message": "AG-UI not available"},
            status_code=501,
        )
    from uuid import uuid4

    run_id = _scoped_run_id(request, uuid4().hex)
    logger.info("AG-UI request received")

    error, parsed = await _parse_ag_ui_request(request)
    if error is not None:
        return error
    assert parsed is not None
    body, query, query_parts = parsed["body"], parsed["query"], parsed["query_parts"]

    session_id = body.get("session_id") or body.get("run_id")
    if session_id:
        run_id = _scoped_run_id(request, session_id)
        logger.info("AG-UI session resumed")
    concurrency_strategy = body.get("concurrency_strategy", "enqueue")
    if concurrency_strategy not in {"enqueue", "reject", "interrupt", "rollback"}:
        concurrency_strategy = "reject"

    cm = getattr(request.app.state, "concurrency_manager", None)

    graph_event_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
    elicitation_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)

    from ...patterns.manager import PatternManager

    _initialized_mcp_toolsets = getattr(request.app.state, "mcp_toolsets", [])
    _agent_instance = getattr(request.app.state, "agent_instance", None)
    graph_bundle = getattr(request.app.state, "graph_bundle", None)

    deps = AgentDeps(
        workspace_path=Path(WORKSPACE_DIR or "."),
        graph_event_queue=graph_event_queue,
        elicitation_queue=elicitation_queue,
        request_id=run_id,
        approval_timeout=DEFAULT_APPROVAL_TIMEOUT,
        provider=DEFAULT_LLM_PROVIDER,
        model_id=DEFAULT_LLM_MODEL_ID,
        base_url=DEFAULT_LLM_BASE_URL,
        api_key=DEFAULT_LLM_API_KEY,
        mcp_toolsets=_initialized_mcp_toolsets,
    )
    deps.patterns = PatternManager(deps)
    logger.info("AG-UI session context established")

    requested_model_id = getattr(request.state, "requested_model_id", None)
    override_model = _build_model_from_registry(
        getattr(request.app.state, "model_registry", None),
        requested_model_id,
    )

    async def merged_stream() -> AsyncIterator[bytes]:
        use_fast_path = False
        if graph_bundle:
            graph_obj, _ = graph_bundle
            use_fast_path = hasattr(graph_obj, "iter")

        if use_fast_path:
            assert graph_bundle is not None
            graph, graph_cfg = graph_bundle
            fast_ctx = _AgUiFastPathCtx(
                graph=graph,
                graph_cfg=graph_cfg,
                query=query,
                run_id=run_id,
                session_id=session_id,
                initialized_mcp_toolsets=_initialized_mcp_toolsets,
                requested_model_id=requested_model_id,
                graph_event_queue=graph_event_queue,
            )
            async for chunk in _ag_ui_fast_path_stream(fast_ctx):
                yield chunk
            return

        adapter_ctx = _AgUiAdapterPathCtx(
            request=request,
            agent_instance=_agent_instance,
            override_model=override_model,
            query=query,
            query_parts=query_parts,
            deps=deps,
            graph_event_queue=graph_event_queue,
            elicitation_queue=elicitation_queue,
        )
        async for chunk in _ag_ui_adapter_path_stream(adapter_ctx):
            yield chunk

    async def merged_stream_with_lock() -> AsyncIterator[bytes]:
        try:
            async for chunk in merged_stream():
                yield chunk
        finally:
            if cm:
                await cm.release(run_id)

    if cm:
        try:
            await cm.acquire(run_id, strategy=concurrency_strategy)
        except HTTPException as exc:
            return JSONResponse(
                {"status": "error", "message": exc.detail},
                status_code=exc.status_code,
            )

    return StreamingResponse(
        merged_stream_with_lock(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _parse_stream_request(
    request: Request,
) -> tuple[Response | None, dict | None]:
    """Parse and validate the generic ``/stream`` request body.

    Returns ``(error_response, None)`` on failure or ``(None, parsed)`` on
    success — exactly one of the pair is ``None``.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid request"}, status_code=400), None
    if not isinstance(data, dict):
        return JSONResponse({"error": "invalid request"}, status_code=422), None
    query = data.get("query", data.get("prompt", ""))
    if not isinstance(query, str) or len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        return JSONResponse({"error": "query too large"}, status_code=413), None
    raw_parts = data.get("parts", [])
    query_parts = await process_parts(raw_parts) if raw_parts else []
    mode = data.get("mode", "ask")
    topology = data.get("topology", "basic")
    if not isinstance(mode, str) or not _STREAM_SELECTOR_RE.fullmatch(mode):
        return JSONResponse({"error": "invalid mode"}, status_code=422), None
    if not isinstance(topology, str) or not _STREAM_SELECTOR_RE.fullmatch(topology):
        return JSONResponse({"error": "invalid topology"}, status_code=422), None
    return (
        None,
        {
            "data": data,
            "query": query,
            "query_parts": query_parts,
            "mode": mode,
            "topology": topology,
        },
    )


async def _acquire_stream_slot(
    cm: Any, session_id: Any, concurrency_strategy: str
) -> Response | None:
    if not (cm and session_id):
        return None
    try:
        await cm.acquire(session_id, strategy=concurrency_strategy)
    except HTTPException as e:
        return JSONResponse(
            {"status": "error", "message": e.detail}, status_code=e.status_code
        )
    return None


@router.post("/stream", summary="SSE Stream Endpoint")
async def stream_endpoint(request: Request) -> Response:
    """Generic SSE stream endpoint for high-fidelity graph agent execution."""
    error, parsed = await _parse_stream_request(request)
    if error is not None:
        return error
    assert parsed is not None
    data = parsed["data"]
    query = parsed["query"]
    query_parts = parsed["query_parts"]
    mode = parsed["mode"]
    topology = parsed["topology"]
    requested_model_id = getattr(request.state, "requested_model_id", None)

    session_id = data.get("session_id") or data.get("run_id")
    concurrency_strategy = data.get("concurrency_strategy", "enqueue")
    if session_id:
        session_id = _scoped_run_id(request, session_id)
    if concurrency_strategy not in {"enqueue", "reject", "interrupt", "rollback"}:
        concurrency_strategy = "reject"

    cm = getattr(request.app.state, "concurrency_manager", None)
    acquire_error = await _acquire_stream_slot(cm, session_id, concurrency_strategy)
    if acquire_error is not None:
        return acquire_error

    graph_bundle = getattr(request.app.state, "graph_bundle", None)
    _initialized_mcp_toolsets = getattr(request.app.state, "mcp_toolsets", [])

    if not graph_bundle:
        if cm and session_id:
            await cm.release(session_id)
        return JSONResponse(
            {"error": "No graph bundle provided for streaming"}, status_code=400
        )

    from ...orchestration.engine import AgentOrchestrationEngine

    graph, config = graph_bundle

    async def graph_stream_with_lock() -> AsyncIterator[bytes]:
        try:
            engine = AgentOrchestrationEngine()
            async for chunk in engine.stream_graph(
                graph,
                config,
                query,
                mode=mode,
                topology=topology,
                mcp_toolsets=_initialized_mcp_toolsets,
                query_parts=query_parts,
                requested_model_id=requested_model_id,
            ):
                yield chunk
        finally:
            if cm and session_id:
                await cm.release(session_id)

    return StreamingResponse(
        graph_stream_with_lock(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
