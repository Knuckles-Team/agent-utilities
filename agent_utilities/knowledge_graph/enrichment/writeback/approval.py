"""Approval queue for high-stakes write-backs (CONCEPT:KG-2.9).

High-stakes sinks (finance trades, legal filings, destructive infra) must NEVER
auto-execute. Their proposed writes are persisted here as ``pending`` proposals; a
human/gate later ``approve``s a proposal id, which replays the exact ops through
:func:`run_writeback` with an approval token. Durable JSON store under the data dir
(same pattern as the skill scheduler's state file).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent_utilities.core.paths import data_dir

logger = logging.getLogger(__name__)


def _store_path() -> Path:
    return data_dir() / "writeback_proposals.json"


class ProposalQueue:
    """Durable pending/approved write-back proposals, keyed by a stable id."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _store_path()

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

    def enqueue(
        self, target: str, ops: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> str:
        data = self._load()
        seq = int(data.get("_seq", 0)) + 1
        data["_seq"] = seq
        pid = f"wbp:{target}:{seq}"
        data.setdefault("proposals", {})[pid] = {
            "id": pid,
            "target": target,
            "ops": {k: v for k, v in ops.items() if not k.startswith("_")},
            "proposals": proposals,
            "status": "pending",
        }
        self._save(data)
        return pid

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        items = list(self._load().get("proposals", {}).values())
        return [p for p in items if status is None or p.get("status") == status]

    def get(self, pid: str) -> dict[str, Any] | None:
        return self._load().get("proposals", {}).get(pid)

    def mark(self, pid: str, status: str) -> None:
        data = self._load()
        if pid in data.get("proposals", {}):
            data["proposals"][pid]["status"] = status
            self._save(data)


def approve_proposal(
    pid: str,
    *,
    backend: Any = None,
    engine: Any = None,
    queue: ProposalQueue | None = None,
) -> dict[str, Any]:
    """Apply a queued high-stakes proposal (replays its ops with an approval token)."""
    from .core import run_writeback

    queue = queue or ProposalQueue()
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
