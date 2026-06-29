#!/usr/bin/python
from __future__ import annotations

"""Cross-surface session continuity for the AG-UI / streaming REST gateway.

CONCEPT:ORCH-1.104 — Unified Agent Entrypoint: shared KG memory + provenance for the
streaming surface.

The instrumented seam :func:`agent_utilities.orchestration.agent_runner.run_agent`
(used by the graph-os MCP ``graph_orchestrate``, the ``messaging`` framework, and the
``WorkflowRunner``) primes recalled mementos at the start of every turn and persists a
``:RunTrace`` (+ ``:ToolCall`` provenance, KG-2.296) and a per-session memento at the
end — so conversation memory and run provenance are SHARED across those surfaces via the
Knowledge Graph, keyed by a memento ``source``.

The AG-UI / SSE REST gateway (``server/routers/agent_ui.py`` — ``/ag-ui`` and
``/stream``, the path the ``agent-webui`` / ``agent-terminal-ui`` frontends hit directly)
streams the SAME pydantic-graph through
:meth:`AgentOrchestrationEngine.iter_graph` for token-by-token delivery. That streaming
contract cannot return through ``run_agent``'s string return value, so historically the
streaming surface ran the shared graph but SKIPPED the seam's two memory/provenance
concerns — leaving webui/terminal-ui memory siloed (no recall, no write-back, no
RunTrace).

This module factors those two concerns into a tiny, reusable pair so the streaming
surface joins the SAME continuity model WITHOUT a parallel orchestrator:

* :func:`prime_session_context` — recall the recent per-session mementos as an
  ``invoker_context`` string (the same store ``run_agent`` primes from), so a turn on the
  streaming surface inherits prior context.
* :func:`persist_session_turn` — record a ``:RunTrace`` and compress the turn into a
  per-session memento (the SAME core primitives the messaging path uses), so the next
  turn — on ANY surface keyed to the same ``session_id`` — recalls it.

Both functions key off the caller-supplied ``session_id`` used VERBATIM as the memento
``source``. Pass a stable, user-scoped id (e.g. the same id the messaging channel uses)
to get true cross-surface recall; both are best-effort and never raise on the hot path.
"""

import asyncio
import contextlib
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

SESSION_MEMENTO_LIMIT = 3


def prime_session_context(
    engine: Any, session_id: str | None, *, limit: int = SESSION_MEMENTO_LIMIT
) -> str:
    """Recall recent per-session mementos as an ``invoker_context`` string.

    Reads the SAME memento store ``run_agent`` primes from (keyed by
    ``source=session_id``), so the streaming surface inherits cross-surface continuity.
    Zero-I/O on a warm session cache; best-effort — returns ``""`` on any failure or when
    there is nothing to recall.

    Args:
        engine: The intelligence/graph engine (the AG-UI ``app.state`` engine).
        session_id: The session/source key — used verbatim as the memento ``source``.
        limit: How many recent mementos to recall.

    Returns:
        A short markdown context block to inject as ``config["invoker_context"]``, or
        ``""`` when nothing is recalled.

    """
    if not engine or not session_id:
        return ""
    try:
        from agent_utilities.knowledge_graph.memory.memento_compressor import (
            get_recent_mementos,
        )

        mementos = get_recent_mementos(engine, source=session_id, limit=limit)
    except Exception as e:  # noqa: BLE001 — recall is advisory; never block the turn
        logger.debug("[ORCH-1.104] memento recall skipped for %s: %s", session_id, e)
        return ""
    joined = "\n".join(f"- {m}" for m in (mementos or []) if m)
    if not joined:
        return ""
    return "Recent conversation memory (recalled from the knowledge graph):\n" + joined


async def persist_session_turn(
    engine: Any,
    session_id: str | None,
    query: str,
    reply: str,
    *,
    agent_name: str = "agent-ui",
    run_id: str | None = None,
) -> None:
    """Persist one streamed turn into the shared KG: provenance + per-session memento.

    Mirrors ``run_agent``'s post-run wrapper for the streaming surface so a
    webui/terminal-ui turn (a) is queryable as a ``:RunTrace`` (KG-2.296 provenance
    parity) and (b) seeds the next turn's recall on ANY surface keyed to the same
    ``session_id``. Runs strictly off the reply path (each KG write is dispatched via
    ``asyncio.to_thread``); best-effort, never raises.

    Args:
        engine: The intelligence/graph engine.
        session_id: The session/source key — used verbatim as the memento ``source``.
        query: The user prompt for this turn.
        reply: The assistant's final reply text for this turn.
        agent_name: The surface label recorded on the RunTrace.
        run_id: Optional run id; one is generated when omitted.

    """
    if not engine or not session_id:
        return

    rid = run_id or f"run:{uuid.uuid4().hex[:8]}"

    # 1) Provenance parity — record a :RunTrace via the seam's own recorder, then anchor
    #    it to its Session (mirrors run_agent Step 5 / CONCEPT:ORCH-1.40).
    with contextlib.suppress(Exception):
        from agent_utilities.orchestration.agent_runner import _record_execution_trace

        await asyncio.to_thread(
            _record_execution_trace,
            engine,
            rid,
            agent_name,
            query,
            "completed",
            None,
            None,
            str(reply)[:500],
        )
        snode = f"session:{session_id}"
        with contextlib.suppress(Exception):
            add_node = getattr(engine, "add_node", None)
            add_edge = getattr(engine, "add_edge", None)
            if callable(add_node) and callable(add_edge):
                add_node(
                    snode,
                    "Session",
                    properties={"id": snode, "session_id": session_id},
                )
                add_edge(snode, f"trace:{rid}", "HAS_RUN")

    # 2) Memory parity — compress this turn into a per-session memento via the SAME core
    #    primitive the messaging path uses (CONCEPT:ECO-4.78), then refresh the session
    #    cache so the next turn reads it from memory (CONCEPT:KG-2.131).
    if not (query or reply):
        return
    with contextlib.suppress(Exception):
        from agent_utilities.knowledge_graph.memory.memento_compressor import (
            compress_to_memento,
        )
        from agent_utilities.knowledge_graph.memory.session_memento_cache import (
            refresh_session_memento_cache,
        )

        turn = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": reply},
        ]
        await asyncio.to_thread(
            compress_to_memento, engine, turn, source=session_id, refine=False
        )
        await asyncio.to_thread(refresh_session_memento_cache, engine, session_id)
