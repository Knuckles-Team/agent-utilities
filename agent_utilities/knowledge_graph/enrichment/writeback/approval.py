"""Approval queue for high-stakes write-backs (CONCEPT:AU-KG.ingest.enterprise-source-extractor / KG-2.247).

High-stakes sinks (finance trades, legal filings, destructive infra) must NEVER
auto-execute. Their proposed writes are persisted here as ``pending`` proposals; a
human/gate later ``approve``s a proposal id, which replays the exact ops through
:func:`run_writeback` with an approval token.

Engine-only (CONCEPT:AU-KG.enrichment.proposals-live-as): proposals live as ``:WritebackProposal`` nodes ON
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


class ProposalQueue:
    """Engine-backed pending/approved write-back proposals, keyed by a stable id.

    CONCEPT:AU-KG.enrichment.proposals-live-as — proposals are ``:WritebackProposal`` nodes on the one
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
                "high-stakes write-back approval queue (CONCEPT:AU-KG.enrichment.proposals-live-as)"
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
        node = self._backend.get_node_properties(pid)
        if not node:
            return
        # Re-upsert the node with the new status (the engine's add_node is an
        # upsert keyed by node id).
        self._backend.add_node(
            pid,
            label=_LABEL,
            id=pid,
            target=node.get("target"),
            ops_json=node.get("ops_json") or "{}",
            proposals_json=node.get("proposals_json") or "[]",
            status=status,
        )

    # -- engine-graph implementation (CONCEPT:AU-KG.enrichment.proposals-live-as) ------------------
    def _next_seq(self, target: str) -> int:
        existing = self._backend.nodes_by_label(_LABEL)
        same_target = sum(
            1 for _nid, props in existing if (props or {}).get("target") == target
        )
        return same_target + 1

    def _enqueue_graph(
        self,
        target: str,
        ops: dict[str, Any],
        proposals: builtins.list[dict[str, Any]],
    ) -> str:
        seq = self._next_seq(target)
        pid = f"wbp:{target}:{seq}"
        self._backend.add_node(
            pid,
            label=_LABEL,
            id=pid,
            target=target,
            ops_json=json.dumps(ops, default=str),
            proposals_json=json.dumps(proposals, default=str),
            status="pending",
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
        node = self._backend.get_node_properties(pid)
        return self._decode(node) if node else None

    def _list_graph(self, status: str | None) -> builtins.list[dict[str, Any]]:
        out: builtins.list[dict[str, Any]] = []
        for _nid, props in self._backend.nodes_by_label(_LABEL):
            if not props:
                continue
            decoded = self._decode(props)
            if status is None or decoded.get("status") == status:
                out.append(decoded)
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
