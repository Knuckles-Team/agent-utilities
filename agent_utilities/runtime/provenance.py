"""CONCEPT:AU-KG.temporal.workspace-action-provenance — Workspace-Action Provenance Graph.

Every workspace action/observation is mirrored into the KG so a run is *replayable* and
*attributable*. The surpassing move over a flat event log: a file edit is grounded to the
``Code`` symbol nodes it mutated (the code ontology is already live — KG-2.8 ingestion emits
``Code``/``Test`` nodes with ``calls``/``covers`` edges), so the golden loop (AHE-3.23) can ask
"which kind of edit on which symbol class failed?" as a graph query rather than parsing logs.

Graph shape::

    (:RunTrace {id: trace:<opaque-ref>}) -[:HAS_ACTION]-> (:WorkspaceAction)
    (:WorkspaceAction) -[:NEXT]-> (:WorkspaceAction)            # replay order
    (:WorkspaceAction) -[:PRODUCED]-> (:WorkspaceObservation)
    (:WorkspaceAction) -[:MUTATED]-> (:Code)                    # for file edits/writes

All writes are best-effort: a cold or absent KG must never break the workspace, so every graph
call is wrapped and failures are logged at debug level only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_utilities.observability.trace_ontology import trace_id
from agent_utilities.security.persistence_privacy import persistence_reference

if TYPE_CHECKING:
    from .events import Action, Observation

logger = logging.getLogger(__name__)

# Code-node property names ingestion may use for the source path (we match any).
_PATH_PROPS = ("file", "path", "file_path", "source_path", "filepath")
_KIND_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_ACTIVE_RUNS = 4096


def _safe_kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    return kind if _KIND_RE.fullmatch(kind) else "unknown"


def _bounded_int(value: Any, *, minimum: int = 0, maximum: int = 2**31 - 1) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return minimum


def _payload_metadata(event: Any, *, namespace: str) -> dict[str, Any]:
    """Return one-way metadata for event fields without durable raw content."""

    dump = getattr(event, "model_dump", None)
    if not callable(dump):
        return {"payload_field_count": 0, "payload_character_count": 0}
    raw = dump(exclude={"actor", "kind", "run_id", "step", "ts"})
    if not isinstance(raw, dict):
        return {"payload_field_count": 0, "payload_character_count": 0}
    canonical = json.dumps(
        raw,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda value: f"<{type(value).__name__}>",
    )
    return {
        "payload_field_count": min(len(raw), 256),
        "payload_character_count": min(len(canonical), 2**31 - 1),
        "payload_ref": persistence_reference(
            "workspace_payload", canonical, namespace=namespace
        ),
    }


def _observation_counts(observation: Any) -> dict[str, int | bool]:
    """Retain bounded operational counts only; never output/report/error text."""

    kind = _safe_kind(getattr(observation, "kind", ""))
    fields: tuple[str, ...]
    if kind == "test_result":
        fields = ("passed", "failed", "errors", "exit_code")
    elif kind == "cmd_output":
        fields = ("exit_code",)
    elif kind == "file_write_ok":
        fields = ("bytes_written",)
    elif kind == "file_edit_ok":
        fields = ("replacements",)
    elif kind == "browser":
        fields = ("status",)
    elif kind == "screen":
        fields = ("width", "height")
    else:
        fields = ()
    counts: dict[str, int | bool] = {
        field: _bounded_int(
            getattr(observation, field, 0), minimum=-(2**16), maximum=2**31 - 1
        )
        for field in fields
    }
    if kind == "file_edit_ok":
        counts["applied"] = bool(getattr(observation, "applied", False))
    if kind == "screen":
        elements = getattr(observation, "elements", None)
        counts["element_count"] = min(len(elements), 100_000) if isinstance(elements, list) else 0
    return counts


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
        trace_ref = trace_id(run_id)
        run_ref = trace_ref.removeprefix("trace:")
        step = _bounded_int(action.step)
        node_id = f"wsaction:{run_ref}:{step}"
        props = {
            "kind": _safe_kind(getattr(action, "kind", "")),
            "trace_id": trace_ref,
            "step": step,
            "ts": action.ts,
            "actor_ref": persistence_reference(
                "actor", action.actor or "", namespace="workspace-action"
            ),
            **_payload_metadata(action, namespace="workspace-action"),
        }
        with contextlib.suppress(Exception):
            engine.add_node(node_id, "WorkspaceAction", properties=props)
            engine.add_edge(trace_ref, node_id, "HAS_ACTION")
            prev = self._last_action.get(trace_ref)
            if prev:
                engine.add_edge(prev, node_id, "NEXT")
            self._last_action[trace_ref] = node_id
            if len(self._last_action) > _MAX_ACTIVE_RUNS:
                self._last_action.pop(next(iter(self._last_action)))

    def mirror_observation(
        self, action: Action, observation: Observation, root: Path
    ) -> None:
        engine = _active_engine()
        if engine is None:
            return
        run_id = action.run_id or "unknown"
        trace_ref = trace_id(run_id)
        run_ref = trace_ref.removeprefix("trace:")
        step = _bounded_int(action.step)
        action_id = f"wsaction:{run_ref}:{step}"
        obs_id = f"wsobs:{run_ref}:{step}"
        props = {
            "kind": _safe_kind(getattr(observation, "kind", "")),
            "trace_id": trace_ref,
            "step": step,
            "ts": observation.ts,
            **_payload_metadata(observation, namespace="workspace-observation"),
            **_observation_counts(observation),
        }
        with contextlib.suppress(Exception):
            engine.add_node(obs_id, "WorkspaceObservation", properties=props)
            engine.add_edge(action_id, obs_id, "PRODUCED")

        # Ground file mutations to the Code symbols they touched.
        path = getattr(action, "path", None)
        if path and getattr(action, "kind", "") in {"file_edit", "file_write"}:
            for symbol_id in self._resolve_symbols(engine, path, root):
                with contextlib.suppress(Exception):
                    engine.add_edge(action_id, symbol_id, "MUTATED")

    # ── symbol grounding ──────────────────────────────────────────────────────
    @staticmethod
    def _resolve_symbols(engine: Any, path: str, root: Path) -> list[str]:
        """Resolve symbols using only a workspace-relative path in the query."""
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        if not callable(execute):
            return []
        try:
            resolved_root = root.expanduser().resolve()
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = resolved_root / candidate
            rel = candidate.resolve().relative_to(resolved_root).as_posix()
        except (OSError, RuntimeError, ValueError):
            return []
        if not rel or len(rel) > 4096:
            return []
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
