import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agent_utilities.core.config import (
    DEFAULT_APPROVAL_TIMEOUT,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER,
)
from agent_utilities.core.workspace import WORKSPACE_DIR

from ...models import AgentDeps
from ..dependencies import _build_model_from_registry, process_parts

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent UI"])


@router.post("/ag-ui", summary="AG-UI Streaming Endpoint")
async def ag_ui_endpoint(request: Request) -> Response:
    """Primary streaming endpoint for the Agent UI (FastAG-UI).

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

    run_id = uuid4().hex
    logger.info(
        f"[LAYER:ACP] AG-UI Request Received. Assigned internal run_id: {run_id}"
    )
    with suppress(Exception):
        body = await request.json()
        if body:
            session_id = body.get("session_id") or body.get("run_id")
            if session_id:
                run_id = session_id
                logger.info(f"[LAYER:ACP] AG-UI: Resuming session: {run_id}")

    graph_event_queue: asyncio.Queue[Any] = asyncio.Queue()
    elicitation_queue: asyncio.Queue[Any] = asyncio.Queue()

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
        provider=DEFAULT_PROVIDER,
        model_id=DEFAULT_MODEL_ID,
        base_url=DEFAULT_LLM_BASE_URL,
        api_key=DEFAULT_LLM_API_KEY,
        mcp_toolsets=_initialized_mcp_toolsets,
    )
    deps.patterns = PatternManager(deps)
    logger.info(f"AG-UI session context: {run_id}")

    requested_model_id = getattr(request.state, "requested_model_id", None)
    override_model = _build_model_from_registry(
        getattr(request.app.state, "model_registry", None),
        requested_model_id,
    )

    async def merged_stream():
        from contextlib import nullcontext

        query = ""
        query_parts = []
        with suppress(Exception):
            body = await request.json()
            query = body.get("query", body.get("prompt", ""))
            raw_parts = body.get("parts", [])
            if raw_parts:
                query_parts = await process_parts(raw_parts)

        from agent_utilities.core.config import DEFAULT_GRAPH_DIRECT_EXECUTION

        _use_fast_path = False
        if graph_bundle and DEFAULT_GRAPH_DIRECT_EXECUTION:
            _graph_obj, _ = graph_bundle
            _use_fast_path = hasattr(_graph_obj, "iter")

        if _use_fast_path:
            from agent_utilities.protocols.agui_emitter import AGUIGraphEmitter

            from ...graph.unified import execute_graph_iter

            logger.info(
                f"[LAYER:AG-UI] Direct graph execution fast-path for query: '{query[:50]}...'"
            )
            assert graph_bundle is not None
            graph, graph_cfg = graph_bundle
            emitter = AGUIGraphEmitter()
            try:
                async for event in execute_graph_iter(
                    graph=graph,
                    config=graph_cfg,
                    query=query,
                    run_id=run_id,
                    mode="ask",
                    mcp_toolsets=_initialized_mcp_toolsets,
                    requested_model_id=requested_model_id,
                ):
                    for chunk in emitter.translate(event):
                        yield chunk
                    while not graph_event_queue.empty():
                        ev = graph_event_queue.get_nowait()
                        if ev:
                            for chunk in emitter._format_sideband(ev):
                                yield chunk
            except Exception as e:
                logger.exception(f"AG-UI direct graph error: {e}")
                error_data = json.dumps({"type": "error", "error": str(e)})
                yield f"data: {error_data}\n\n".encode()
            return

        run_input = query_parts if query_parts else query
        override_ctx = (
            _agent_instance.override(model=override_model)  # type: ignore[union-attr]
            if override_model is not None
            else nullcontext()
        )
        try:
            with override_ctx:
                adapter = AGUIAdapter(agent=_agent_instance, run_input=run_input)
                _q_preview = query[:50]
                logger.info(
                    f"[LAYER:ACP] AG-UI: Dispatching request for query: '{_q_preview}...'"
                )
                if override_model is not None:
                    logger.info(
                        "AG-UI: Applying per-turn model override '%s'",
                        requested_model_id,
                    )
                agent_response = await adapter.dispatch_request(
                    request, agent=_agent_instance, deps=deps
                )
            logger.info("[LAYER:ACP] AG-UI: Dispatch successful. Stream established.")
        except Exception as e:
            logger.exception(f"AG-UI: Dispatch error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        if not isinstance(agent_response, StreamingResponse):
            yield agent_response.body
            return

        combined_queue: asyncio.Queue = asyncio.Queue()

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
                                if isinstance(chunk, bytes)
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
                                if isinstance(chunk, bytes)
                                else chunk.encode("utf-8"),
                            )
                        )
            except Exception as e:
                logger.error(f"Agent stream error: {e}")
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
                        except Exception as e:
                            logger.error(f"Error processing sideband event: {e}")
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Sideband poller error: {e}")
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

    return StreamingResponse(
        merged_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/stream", summary="SSE Stream Endpoint")
async def stream_endpoint(request: Request) -> Response:
    """Generic SSE stream endpoint for high-fidelity graph agent execution."""
    data = await request.json()
    query = data.get("query", data.get("prompt", ""))
    raw_parts = data.get("parts", [])
    query_parts = await process_parts(raw_parts) if raw_parts else []
    mode = data.get("mode", "ask")
    topology = data.get("topology", "basic")
    requested_model_id = getattr(request.state, "requested_model_id", None)

    graph_bundle = getattr(request.app.state, "graph_bundle", None)
    _initialized_mcp_toolsets = getattr(request.app.state, "mcp_toolsets", [])

    if graph_bundle:
        from ...graph_orchestration import run_graph_stream

        graph, config = graph_bundle
        return StreamingResponse(
            run_graph_stream(
                graph,
                config,
                query,
                mode=mode,
                topology=topology,
                mcp_toolsets=_initialized_mcp_toolsets,
                query_parts=query_parts,
                requested_model_id=requested_model_id,
            ),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(
            {"error": "No graph bundle provided for streaming"}, status_code=400
        )
