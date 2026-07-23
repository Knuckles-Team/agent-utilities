import asyncio
import json
import logging
import re
from contextlib import suppress
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


def _scoped_run_id(request: Request, supplied: Any = None) -> str:
    """Bind a caller-provided continuity key to the authenticated identity."""
    claims = getattr(request.state, "user_claims", None) or {}
    owner = (
        claims.get("tenant_id")
        or claims.get("tenant")
        or claims.get("sub")
        or claims.get("client_id")
        or claims.get("auth_type")
        or "local"
    )
    candidate = str(supplied or "").strip()
    if len(candidate) > 256 or any(character in candidate for character in "\r\n\x00"):
        candidate = ""
    from agent_utilities.security.persistence_privacy import persistence_reference

    return persistence_reference("agent_run", f"{owner}\x00{candidate or 'new'}")


@router.post("/ag-ui", summary="AG-UI Streaming Endpoint")
async def ag_ui_endpoint(request: Request) -> Response:
    """Primary streaming endpoint for the Agent UI (FastAG-UI).

    CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Supports sideband graph activity annotations, session resumption,
        and rich media attachments. This endpoint handles high-fidelity
        SSE streaming with sideband data.
    """
    try:
        from pydantic_ai.ui.ag_ui import AGUIAdapter
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
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON request"}, status_code=422)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid request"}, status_code=422)
    query = body.get("query", body.get("prompt", ""))
    if not isinstance(query, str):
        return JSONResponse({"error": "invalid query"}, status_code=422)
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        return JSONResponse({"error": "query too large"}, status_code=413)
    raw_parts = body.get("parts", [])
    try:
        query_parts = await process_parts(raw_parts) if raw_parts else []
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
        return JSONResponse({"error": "invalid message parts"}, status_code=422)

    concurrency_strategy = "enqueue"
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

    async def merged_stream():
        from contextlib import nullcontext

        _use_fast_path = False
        if graph_bundle:
            _graph_obj, _ = graph_bundle
            _use_fast_path = hasattr(_graph_obj, "iter")

        if _use_fast_path:
            from agent_utilities.protocols.agui_emitter import AGUIGraphEmitter

            from ...graph.protocol_agnostic_execution import execute_graph_iter

            logger.info("AG-UI direct graph execution")
            assert graph_bundle is not None
            graph, graph_cfg = graph_bundle

            # CONCEPT:AU-ORCH.session.session-continuity-entrypoint — Unified Agent Entrypoint: the streaming surface streams
            # the SAME graph as the run_agent seam but cannot return through it, so join the
            # shared continuity model here. Recall the per-session mementos (keyed by the
            # caller's stable session_id == run_id) and inject them as invoker_context so
            # this turn inherits cross-surface memory; persist the turn afterwards.
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )
            from agent_utilities.orchestration.session_continuity import (
                persist_session_turn,
                prime_session_context,
            )

            _kg_engine = None
            _exec_cfg = graph_cfg
            with suppress(Exception):
                _kg_engine = IntelligenceGraphEngine.get_active()
                _primed = prime_session_context(_kg_engine, run_id)
                if _primed:
                    _exec_cfg = {**graph_cfg, "invoker_context": _primed}

            emitter = AGUIGraphEmitter()
            _final_output: str = ""
            try:
                async for event in execute_graph_iter(
                    graph=graph,
                    config=_exec_cfg,
                    query=query,
                    run_id=run_id,
                    mode="ask",
                    mcp_toolsets=_initialized_mcp_toolsets,
                    requested_model_id=requested_model_id,
                ):
                    if isinstance(event, dict) and event.get("type") in (
                        "graph_complete",
                        "final_output",
                    ):
                        _out = event.get("output") or event.get("content")
                        if _out:
                            _final_output = str(_out)
                    for chunk in emitter.translate(event):
                        yield chunk
                    while not graph_event_queue.empty():
                        ev = graph_event_queue.get_nowait()
                        if ev:
                            for chunk in emitter._format_sideband(ev):
                                yield chunk
            except Exception as exc:
                error_data = json.dumps(
                    {"type": "error", **public_error_payload(exc, logger=logger)}
                )
                yield f"data: {error_data}\n\n".encode()
            finally:
                # CONCEPT:AU-ORCH.session.session-continuity-entrypoint — persist the turn (RunTrace + per-session memento)
                # off the reply path so the NEXT turn — on this surface OR any other keyed
                # to the same session — recalls it. Best-effort; never affects the stream.
                if _kg_engine is not None and _final_output:
                    with suppress(Exception):
                        from agent_utilities.security.persistence_privacy import (
                            PersistencePrivacyGuard,
                        )

                        guard = PersistencePrivacyGuard()
                        clean_query = guard.sanitize_text(query)[0]
                        clean_output = guard.sanitize_text(_final_output)[0]
                        asyncio.create_task(
                            persist_session_turn(
                                _kg_engine,
                                run_id,
                                clean_query,
                                clean_output,
                                agent_name="agent-ui",
                                run_id=run_id,
                            )
                        )
            return

        # The AG-UI adapter's `run_input` is dynamically shaped (the assembled
        # multimodal parts list, or the bare query string) — annotate `Any` rather
        # than pin it to the adapter's declared `RunAgentInput` protocol type.
        run_input: Any = query_parts if query_parts else query
        override_ctx = (
            _agent_instance.override(model=override_model)  # type: ignore[union-attr]
            if override_model is not None
            else nullcontext()
        )
        if _agent_instance is None:
            raise RuntimeError("Agent instance not initialized on app state")
        try:
            with override_ctx:
                adapter = AGUIAdapter(agent=_agent_instance, run_input=run_input)
                logger.info("[LAYER:ACP] AG-UI: Dispatching request")
                if override_model is not None:
                    logger.info("AG-UI: Applying authorized per-turn model override")
                agent_response = await adapter.dispatch_request(
                    request, agent=_agent_instance, deps=deps
                )
            logger.info("[LAYER:ACP] AG-UI: Dispatch successful. Stream established.")
        except Exception as exc:
            failure = {"type": "error", **public_error_payload(exc, logger=logger)}
            yield f"data: {json.dumps(failure)}\n\n"
            return

        if not isinstance(agent_response, StreamingResponse):
            yield agent_response.body
            return

        combined_queue: asyncio.Queue = asyncio.Queue(maxsize=512)

        async def poll_agent():
            try:
                async for chunk in agent_response.body_iterator:
                    chunk_str = (
                        chunk.decode("utf-8")
                        if isinstance(chunk, bytes)
                        else str(chunk)
                    )
                    if (
                        chunk_str.startswith("2:")
                        or chunk_str.startswith("9:")
                        or '"tool_calls"' in chunk_str
                    ):
                        await combined_queue.put(
                            (
                                "chunk",
                                chunk
                                if isinstance(chunk, bytes | bytearray)
                                else bytes(chunk)
                                if isinstance(chunk, memoryview)
                                else chunk.encode("utf-8"),
                            )
                        )
                        await combined_queue.put(("chunk", b'0 " "\n'))
                        await asyncio.sleep(0.01)
                    else:
                        await combined_queue.put(
                            (
                                "chunk",
                                chunk
                                if isinstance(chunk, bytes | bytearray)
                                else bytes(chunk)
                                if isinstance(chunk, memoryview)
                                else chunk.encode("utf-8"),
                            )
                        )
            except Exception as exc:
                logger.error(
                    "Agent stream error (exception_type=%s)", type(exc).__name__
                )
            finally:
                await combined_queue.put(("done", None))

        async def poll_sideband():
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
                        try:
                            ev = await task
                            if ev:
                                packet = f"8:{json.dumps(ev)}\n".encode()
                                await combined_queue.put(("chunk", packet))
                                await combined_queue.put(("chunk", b'0 " "\n'))
                                await asyncio.sleep(0.01)
                        except Exception as exc:
                            logger.error(
                                "Error processing sideband event (exception_type=%s)",
                                type(exc).__name__,
                            )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(
                        "Sideband poller error (exception_type=%s)",
                        type(exc).__name__,
                    )
                    break

        agent_task = asyncio.create_task(poll_agent())
        sideband_task = asyncio.create_task(poll_sideband())

        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(
                        combined_queue.get(), timeout=0.1
                    )
                    if msg_type == "done":
                        await asyncio.sleep(0.1)
                        if (
                            not graph_event_queue.empty()
                            or not elicitation_queue.empty()
                        ):
                            continue
                        break
                    yield data
                    combined_queue.task_done()
                except TimeoutError:
                    yield b'0 " "\n'
                    if agent_task.done() and combined_queue.empty():
                        break
                    continue
        finally:
            agent_task.cancel()
            sideband_task.cancel()

    async def merged_stream_with_lock():
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


@router.post("/stream", summary="SSE Stream Endpoint")
async def stream_endpoint(request: Request) -> Response:
    """Generic SSE stream endpoint for high-fidelity graph agent execution."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid request"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "invalid request"}, status_code=422)
    query = data.get("query", data.get("prompt", ""))
    if not isinstance(query, str) or len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        return JSONResponse({"error": "query too large"}, status_code=413)
    raw_parts = data.get("parts", [])
    query_parts = await process_parts(raw_parts) if raw_parts else []
    mode = data.get("mode", "ask")
    topology = data.get("topology", "basic")
    if not isinstance(mode, str) or not _STREAM_SELECTOR_RE.fullmatch(mode):
        return JSONResponse({"error": "invalid mode"}, status_code=422)
    if not isinstance(topology, str) or not _STREAM_SELECTOR_RE.fullmatch(topology):
        return JSONResponse({"error": "invalid topology"}, status_code=422)
    requested_model_id = getattr(request.state, "requested_model_id", None)

    session_id = data.get("session_id") or data.get("run_id")
    concurrency_strategy = data.get("concurrency_strategy", "enqueue")
    if session_id:
        session_id = _scoped_run_id(request, session_id)
    if concurrency_strategy not in {"enqueue", "reject", "interrupt", "rollback"}:
        concurrency_strategy = "reject"

    cm = getattr(request.app.state, "concurrency_manager", None)
    if cm and session_id:
        try:
            await cm.acquire(session_id, strategy=concurrency_strategy)
        except HTTPException as e:
            return JSONResponse(
                {"status": "error", "message": e.detail}, status_code=e.status_code
            )

    graph_bundle = getattr(request.app.state, "graph_bundle", None)
    _initialized_mcp_toolsets = getattr(request.app.state, "mcp_toolsets", [])

    if graph_bundle:
        from ...orchestration.engine import AgentOrchestrationEngine

        graph, config = graph_bundle

        async def graph_stream_with_lock():
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
    else:
        if cm and session_id:
            await cm.release(session_id)
        return JSONResponse(
            {"error": "No graph bundle provided for streaming"}, status_code=400
        )
