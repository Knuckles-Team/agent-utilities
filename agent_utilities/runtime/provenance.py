"""CONCEPT:AU-KG.temporal.workspace-action-provenance — Workspace-Action Provenance Graph.

Every workspace action/observation is mirrored into the KG so a run is *replayable* and
*attributable*. The surpassing move over a flat event log: a file edit is grounded to the
``Code`` symbol nodes it mutated (the code ontology is already live — KG-2.8 ingestion emits
``Code``/``Test`` nodes with ``calls``/``covers`` edges), so the golden loop (AHE-3.23) can ask
"which kind of edit on which symbol class failed?" as a graph query rather than parsing logs.

Graph shape::

    (:RunTrace {id: trace:<run_id>}) -[:HAS_ACTION]-> (:WorkspaceAction)
    (:WorkspaceAction) -[:NEXT]-> (:WorkspaceAction)            # replay order
    (:WorkspaceAction) -[:PRODUCED]-> (:WorkspaceObservation)
    (:WorkspaceAction) -[:MUTATED]-> (:Code)                    # for file edits/writes

All writes are best-effort: a cold or absent KG must never break the workspace, so every graph
call is wrapped and failures are logged at debug level only.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import Action, Observation

logger = logging.getLogger(__name__)

# Code-node property names ingestion may use for the source path (we match any).
_PATH_PROPS = ("file", "path", "file_path", "source_path", "filepath")


def _active_engine() -> Any | None:
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception:  # noqa: BLE001 - KG optional; provenance degrades silently
        return None


class ProvenanceMirror:
    """Mirror workspace actions/observations into the KG, grounding edits to Code symbols."""

    def __init__(self) -> None:
        # last action node id per run, for NEXT replay edges
        self._last_action: dict[str, str] = {}

    def mirror_action(self, action: Action) -> None:
        engine = _active_engine()
        if engine is None:
            return
        run_id = action.run_id or "unknown"
        node_id = f"wsaction:{run_id}:{action.step}"
        props = {
            "kind": getattr(action, "kind", ""),
            "run_id": run_id,
            "step": action.step,
            "ts": action.ts,
            "actor": action.actor or "",
            "summary": self._summarize_action(action),
        }
        with contextlib.suppress(Exception):
            engine.add_node(node_id, "WorkspaceAction", properties=props)
            engine.add_edge(f"trace:{run_id}", node_id, "HAS_ACTION")
            prev = self._last_action.get(run_id)
            if prev:
                engine.add_edge(prev, node_id, "NEXT")
            self._last_action[run_id] = node_id

    def mirror_observation(
        self, action: Action, observation: Observation, root: Path
    ) -> None:
        engine = _active_engine()
        if engine is None:
            return
        run_id = action.run_id or "unknown"
        action_id = f"wsaction:{run_id}:{action.step}"
        obs_id = f"wsobs:{run_id}:{action.step}"
        props = {
            "kind": getattr(observation, "kind", ""),
            "run_id": run_id,
            "step": action.step,
            "ts": observation.ts,
            "summary": self._summarize_observation(observation),
        }
        with contextlib.suppress(Exception):
            engine.add_node(obs_id, "WorkspaceObservation", properties=props)
            engine.add_edge(action_id, obs_id, "PRODUCED")

        # Ground file mutations to the Code symbols they touched.
        path = getattr(action, "path", None)
        if path and getattr(action, "kind", "") in {"file_edit", "file_write"}:
            for symbol_id in self._resolve_symbols(engine, path):
                with contextlib.suppress(Exception):
                    engine.add_edge(action_id, symbol_id, "MUTATED")

    # ── symbol grounding ──────────────────────────────────────────────────────
    @staticmethod
    def _resolve_symbols(engine: Any, path: str) -> list[str]:
        """Return Code node ids whose stored source path matches ``path`` (suffix match)."""
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        if not callable(execute):
            return []
        rel = path.lstrip("/")
        where = " OR ".join(f"c.{p} ENDS WITH $rel" for p in _PATH_PROPS)
        cypher = f"MATCH (c:Code) WHERE {where} RETURN c.id AS id LIMIT 50"
        try:
            rows = execute(cypher, {"rel": rel})
        except Exception:  # noqa: BLE001 - backend may not support this dialect
            return []
        out: list[str] = []
        for row in rows or []:
            sid = row.get("id") if isinstance(row, dict) else None
            if sid:
                out.append(str(sid))
        return out

    # ── summaries ──────────────────────────────────────────────────────────────
    @staticmethod
    def _summarize_action(action: Action) -> str:
        kind = getattr(action, "kind", "")
        if kind == "cmd_run":
            return f"$ {getattr(action, 'command', '')[:200]}"
        if kind in {"file_read", "file_write", "file_edit"}:
            return f"{kind} {getattr(action, 'path', '')}"
        if kind == "test_run":
            return f"test {getattr(action, 'selector', '') or '<suite>'}"
        if kind == "computer_use":
            op = getattr(action, "op", "")
            target = getattr(action, "element_id", "") or (
                f"@{getattr(action, 'x', '')},{getattr(action, 'y', '')}"
            )
            detail = getattr(action, "text", "") or getattr(action, "keys", "")
            return f"gui {op} {detail or target}".strip()
        if kind == "agent_finish":
            return getattr(action, "summary", "")[:200]
        return kind

    @staticmethod
    def _summarize_observation(observation: Observation) -> str:
        kind = getattr(observation, "kind", "")
        if kind == "cmd_output":
            return f"exit={getattr(observation, 'exit_code', '?')}"
        if kind == "test_result":
            return getattr(observation, "report", "")
        if kind == "error":
            return f"error: {getattr(observation, 'message', '')[:200]}"
        if kind == "screen":
            err = getattr(observation, "error", "")
            if err:
                return f"screen error: {err[:160]}"
            n = len(getattr(observation, "elements", []) or [])
            w = getattr(observation, "width", 0)
            h = getattr(observation, "height", 0)
            return f"screen {w}x{h}, {n} elements"
        return kind
