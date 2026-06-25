"""Approval queue for high-stakes write-backs (CONCEPT:KG-2.9 / KG-2.247).

High-stakes sinks (finance trades, legal filings, destructive infra) must NEVER
auto-execute. Their proposed writes are persisted here as ``pending`` proposals; a
human/gate later ``approve``s a proposal id, which replays the exact ops through
:func:`run_writeback` with an approval token.

Engine-only (CONCEPT:KG-2.247): proposals live as ``:WritebackProposal`` nodes ON
THE ONE epistemic-graph engine authority — queryable, beside the graph — with NO
local ``writeback_proposals.json`` fallback. When no engine backend is supplied
the queue resolves the active engine backend (the OS-5.63 resolver auto-starts the
pi-tier engine in prod; the KG-2.238 fixture provides a real ephemeral one in
tests), raising a clear error if the engine is genuinely unreachable.
"""

from __future__ import annotations

import builtins
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LABEL = "WritebackProposal"


def _node(row: Any) -> dict[str, Any] | None:
    """Extract the node dict from an ``execute()`` row (``{'p': {...}}`` or flat)."""
    if isinstance(row, dict):
        inner = row.get("p")
        if isinstance(inner, dict):
            return inner
        return row
    return None


class ProposalQueue:
    """Engine-backed pending/approved write-back proposals, keyed by a stable id.

    CONCEPT:KG-2.247 — proposals are ``:WritebackProposal`` nodes on the one
    epistemic-graph engine authority; there is no JSON-file fallback.
    """

    def __init__(self, backend: Any = None) -> None:
        from ...backends.base import (
            is_engine_authority_backend,
            require_engine_authority_backend,
        )

        if is_engine_authority_backend(backend):
            self._backend: Any = backend
        else:
            self._backend = require_engine_authority_backend(
                "high-stakes write-back approval queue (CONCEPT:KG-2.247)"
            )

    # -- public API ------------------------------------------------------
    def enqueue(
        self, target: str, ops: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> str:
        clean_ops = {k: v for k, v in ops.items() if not k.startswith("_")}
        return self._enqueue_graph(target, clean_ops, proposals)

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._list_graph(status)

    def get(self, pid: str) -> dict[str, Any] | None:
        return self._get_graph(pid)

    def mark(self, pid: str, status: str) -> None:
        self._backend.execute(
            f"MATCH (p:{_LABEL} {{id: $id}}) SET p.status = $status",
            {"id": pid, "status": status},
        )

    # -- engine-graph implementation (CONCEPT:KG-2.247) ------------------
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
        self,
        target: str,
        ops: dict[str, Any],
        proposals: builtins.list[dict[str, Any]],
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

    def _list_graph(self, status: str | None) -> builtins.list[dict[str, Any]]:
        if status is None:
            rows = self._backend.execute(f"MATCH (p:{_LABEL}) RETURN p", {})
        else:
            rows = self._backend.execute(
                f"MATCH (p:{_LABEL}) WHERE p.status = $status RETURN p",
                {"status": status},
            )
        out: builtins.list[dict[str, Any]] = []
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
