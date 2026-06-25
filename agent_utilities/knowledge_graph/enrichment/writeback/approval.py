"""Approval queue for high-stakes write-backs (CONCEPT:KG-2.9 / KG-2.208).

High-stakes sinks (finance trades, legal filings, destructive infra) must NEVER
auto-execute. Their proposed writes are persisted here as ``pending`` proposals; a
human/gate later ``approve``s a proposal id, which replays the exact ops through
:func:`run_writeback` with an approval token.

Dual-mode like ``DeltaManifest`` (CONCEPT:KG-2.208): when a durable graph backend
(the epistemic-graph engine authority) is available, proposals live as
``:WritebackProposal`` nodes ON THE ENGINE — queryable, beside the one authority —
instead of a local ``writeback_proposals.json``. Only the zero-infra ``tiny``
profile (no durable backend) keeps the JSON store under the data dir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent_utilities.core.paths import data_dir

logger = logging.getLogger(__name__)

_LABEL = "WritebackProposal"


def _store_path() -> Path:
    return data_dir() / "writeback_proposals.json"


def _node(row: Any) -> dict[str, Any] | None:
    """Extract the node dict from an ``execute()`` row (``{'p': {...}}`` or flat)."""
    if isinstance(row, dict):
        inner = row.get("p")
        if isinstance(inner, dict):
            return inner
        return row
    return None


class ProposalQueue:
    """Durable pending/approved write-back proposals, keyed by a stable id.

    Engine-graph mode (``:WritebackProposal`` nodes) when a durable backend is
    passed/active; the JSON file fallback otherwise (CONCEPT:KG-2.208).
    """

    def __init__(self, path: str | Path | None = None, backend: Any = None) -> None:
        from ...backends.base import is_durable_backend

        if backend is None:
            try:
                from ...backends import get_active_backend

                backend = get_active_backend()
            except Exception:  # noqa: BLE001 - no backend wired -> JSON fallback
                backend = None
        self._backend: Any = backend if is_durable_backend(backend) else None
        self.mode = "graph" if self._backend is not None else "json"
        self._path = Path(path) if path else _store_path()

    # -- JSON fallback ----------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:  # noqa: BLE001 - empty/missing store
            return {"_seq": 0, "proposals": {}}

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:  # noqa: BLE001
            logger.warning("could not persist writeback proposals", exc_info=True)

    # -- public API (dual-mode) ------------------------------------------
    def enqueue(
        self, target: str, ops: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> str:
        clean_ops = {k: v for k, v in ops.items() if not k.startswith("_")}
        if self.mode == "graph":
            return self._enqueue_graph(target, clean_ops, proposals)
        data = self._load()
        seq = int(data.get("_seq", 0)) + 1
        data["_seq"] = seq
        pid = f"wbp:{target}:{seq}"
        data.setdefault("proposals", {})[pid] = {
            "id": pid,
            "target": target,
            "ops": clean_ops,
            "proposals": proposals,
            "status": "pending",
        }
        self._save(data)
        return pid

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if self.mode == "graph":
            return self._list_graph(status)
        items = list(self._load().get("proposals", {}).values())
        return [p for p in items if status is None or p.get("status") == status]

    def get(self, pid: str) -> dict[str, Any] | None:
        if self.mode == "graph":
            return self._get_graph(pid)
        return self._load().get("proposals", {}).get(pid)

    def mark(self, pid: str, status: str) -> None:
        if self.mode == "graph":
            self._backend.execute(
                f"MATCH (p:{_LABEL} {{id: $id}}) SET p.status = $status",
                {"id": pid, "status": status},
            )
            return
        data = self._load()
        if pid in data.get("proposals", {}):
            data["proposals"][pid]["status"] = status
            self._save(data)

    # -- engine-graph mode (CONCEPT:KG-2.208) ----------------------------
    def _next_seq(self, target: str) -> int:
        rows = self._backend.execute(
            f"MATCH (p:{_LABEL}) WHERE p.target = $target RETURN count(p) AS c",
            {"target": target},
        )
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                val = row.get("c", row.get("count(p)"))
                if val is not None:
                    return int(val) + 1
        return 1

    def _enqueue_graph(
        self, target: str, ops: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> str:
        seq = self._next_seq(target)
        pid = f"wbp:{target}:{seq}"
        self._backend.execute(
            f"MERGE (p:{_LABEL} {{id: $id}}) SET "
            "p.target = $target, p.ops_json = $ops, p.proposals_json = $props, "
            "p.status = 'pending'",
            {
                "id": pid,
                "target": target,
                "ops": json.dumps(ops, default=str),
                "props": json.dumps(proposals, default=str),
            },
        )
        return pid

    def _decode(self, node: dict[str, Any]) -> dict[str, Any]:
        try:
            ops = json.loads(node.get("ops_json") or "{}")
        except (ValueError, json.JSONDecodeError):
            ops = {}
        try:
            props = json.loads(node.get("proposals_json") or "[]")
        except (ValueError, json.JSONDecodeError):
            props = []
        return {
            "id": node.get("id"),
            "target": node.get("target"),
            "ops": ops,
            "proposals": props,
            "status": node.get("status", "pending"),
        }

    def _get_graph(self, pid: str) -> dict[str, Any] | None:
        rows = self._backend.execute(
            f"MATCH (p:{_LABEL} {{id: $id}}) RETURN p", {"id": pid}
        )
        for row in rows if isinstance(rows, list) else []:
            node = _node(row)
            if node:
                return self._decode(node)
        return None

    def _list_graph(self, status: str | None) -> list[dict[str, Any]]:
        if status is None:
            rows = self._backend.execute(f"MATCH (p:{_LABEL}) RETURN p", {})
        else:
            rows = self._backend.execute(
                f"MATCH (p:{_LABEL}) WHERE p.status = $status RETURN p",
                {"status": status},
            )
        out: list[dict[str, Any]] = []
        for row in rows if isinstance(rows, list) else []:
            node = _node(row)
            if node:
                out.append(self._decode(node))
        return out


def approve_proposal(
    pid: str,
    *,
    backend: Any = None,
    engine: Any = None,
    queue: ProposalQueue | None = None,
) -> dict[str, Any]:
    """Apply a queued high-stakes proposal (replays its ops with an approval token)."""
    from .core import run_writeback

    queue = queue or ProposalQueue(backend=backend)
    proposal = queue.get(pid)
    if proposal is None:
        return {"status": "error", "error": f"unknown proposal {pid!r}"}
    if proposal.get("status") != "pending":
        return {
            "status": "skipped",
            "reason": f"proposal {pid} is {proposal.get('status')}",
        }
    result = run_writeback(
        proposal["target"],
        backend=backend,
        engine=engine,
        dry_run=False,
        _approved=True,
        **(proposal.get("ops") or {}),
    )
    queue.mark(pid, "approved" if result.get("status") == "completed" else "failed")
    result["proposal_id"] = pid
    return result
