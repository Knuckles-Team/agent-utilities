"""CONCEPT:AU-ORCH.runvcs.fork-revert — fork/revert a *live* run from any commit.

This is the run-scoped API shepherd exposes as ``scope.fork``/``checkpoint``/``resume`` and the
firecracker ``branch`` verb, generalized over ALL three fork substrates at once. A
:class:`RunSession` owns a run's workspace, its message history, and its
:class:`~.kernel.RunEventLog`, and drives them together:

* :meth:`commit` — snapshot fs + checkpoint messages + cut the event log into ONE
  :class:`~.run_commit.RunCommit` (cheap; commit at every step).
* :meth:`revert` — restore *this* run's files **and** process/event frontier **and** messages
  to an exact prior commit. Mid-run time-travel to a known-good world.
* :meth:`fork` — spawn a NEW run seeded from any commit into a fresh workspace, leaving the
  parent untouched (the review/what-if path — gate the child, then discard or merge).

A :class:`RunSessionRegistry` keeps live sessions addressable by ``run_id`` so the MCP/REST
surface (``graph_runvcs``) can operate on a running session across calls.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from agent_utilities.capabilities.checkpointing import (
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
)

from .carrier import FsCarrier
from .kernel import RunEventLog
from .run_commit import RunCommit, RunCommitStore

logger = logging.getLogger(__name__)


class RunSession:
    """A live, forkable, revertable agent run over a real workspace.

    ``messages`` is the mutable conversation history (any list; ``pydantic_ai`` ``ModelMessage``
    in production, plain dicts in tests — the in-memory checkpoint store preserves objects). The
    event log records the typed, content-addressed effect stream. The carrier snapshots files.
    """

    def __init__(
        self,
        run_id: str,
        root: str | Path,
        *,
        commit_store: RunCommitStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        engine: Any | None = None,
    ) -> None:
        self.run_id = run_id
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.messages: list[Any] = []
        self.log = RunEventLog(run_id, engine=engine)
        self.carrier = FsCarrier(self.root)
        self.commits = commit_store or RunCommitStore()
        self.checkpoints = checkpoint_store or InMemoryCheckpointStore()
        self._engine = engine

    # ── commit ─────────────────────────────────────────────────────────────────
    async def commit(self, label: str = "") -> RunCommit:
        """Snapshot fs + messages + event frontier into ONE content-addressed commit."""
        snapshot = self.carrier.snapshot()
        checkpoint_id = f"ckpt_{self.run_id}_{uuid.uuid4().hex[:8]}"
        checkpoint = Checkpoint(
            id=checkpoint_id,
            label=label or "run-commit",
            turn=len(self.messages),
            messages=list(self.messages),
            metadata={"run_id": self.run_id},
        )
        await self.checkpoints.save(checkpoint)
        parent = self.commits.head
        commit = RunCommit(
            run_id=self.run_id,
            checkpoint_id=checkpoint_id,
            fs_snapshot=snapshot,
            event_cut=self.log.cut(),
            label=label,
            parent_commit_id=parent.commit_id if parent else None,
        )
        self.commits.save(commit)
        self.log.append(
            "run_commit",
            {"commit_id": commit.commit_id, "label": label},
            mode="capture",
        )
        return commit

    # ── revert ──────────────────────────────────────────────────────────────────
    async def revert(self, commit: RunCommit | str) -> dict[str, Any]:
        """Restore THIS run's files + event frontier + messages to ``commit``.

        The exact-world restore shepherd calls a mid-run revert: files come back via the
        carrier, the event log rewinds to the commit's cut, and the conversation is rewound to
        the commit's checkpoint. Returns a summary of what was restored.
        """
        commit = self._resolve(commit)
        fs_result = self.carrier.restore(commit.fs_snapshot)
        self.log.truncate_to(commit.event_cut)
        checkpoint = await self.checkpoints.get(commit.checkpoint_id)
        restored_msgs = 0
        if checkpoint is not None:
            self.messages = list(checkpoint.messages)
            restored_msgs = len(self.messages)
        return {
            "reverted_to": commit.commit_id,
            "files_written": fs_result["written"],
            "files_removed": fs_result["removed"],
            "messages_restored": restored_msgs,
            "event_ordinal": commit.event_cut.through_ordinal,
        }

    def discard(self) -> dict[str, Any]:
        """Drop the uncommitted world delta by rewinding to the head commit's frontier.

        Files are NOT touched (discard is about the *event/message* delta since the last
        commit); use :meth:`revert` to also restore files. If there is no commit, clears the log.
        """
        head = self.commits.head
        if head is None:
            n = len(self.log.events)
            self.log = RunEventLog(self.run_id, engine=self._engine)
            return {"discarded_events": n, "reverted_to": None}
        before = len(self.log.events)
        self.log.truncate_to(head.event_cut)
        return {
            "discarded_events": before - len(self.log.events),
            "reverted_to": head.commit_id,
        }

    # ── fork ────────────────────────────────────────────────────────────────────
    async def fork(
        self,
        commit: RunCommit | str,
        *,
        new_run_id: str | None = None,
        new_root: str | Path | None = None,
    ) -> RunSession:
        """Spawn a NEW run seeded from ``commit`` into a fresh workspace (parent untouched).

        The child gets the committed files (restored into ``new_root``), the committed message
        history, and the committed event prefix — a true branch from any point in the run's
        history. The parent :class:`RunSession` is not mutated.
        """
        commit = self._resolve(commit)
        child_id = new_run_id or f"{self.run_id}-fork-{uuid.uuid4().hex[:6]}"
        child_root = Path(new_root) if new_root else self.root.parent / child_id
        child = RunSession(
            child_id,
            child_root,
            commit_store=RunCommitStore(),
            checkpoint_store=self.checkpoints,
            engine=self._engine,
        )
        # Restore committed files into the child's fresh workspace.
        self.carrier.restore(commit.fs_snapshot, target=child_root)
        # Seed the child's event log with the committed prefix (re-based onto its path).
        child.log = RunEventLog.from_events(
            child_id, self.log.project(commit.event_cut), engine=self._engine
        )
        # Seed the child's conversation from the committed checkpoint.
        checkpoint = await self.checkpoints.get(commit.checkpoint_id)
        if checkpoint is not None:
            child.messages = list(checkpoint.messages)
        child.log.append(
            "run_fork",
            {"from_commit": commit.commit_id, "parent_run": self.run_id},
            mode="capture",
        )
        return child

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _resolve(self, commit: RunCommit | str) -> RunCommit:
        if isinstance(commit, RunCommit):
            return commit
        found = self.commits.get(commit)
        if found is None:
            raise KeyError(f"no commit {commit!r} in run {self.run_id!r}")
        return found

    def status(self) -> dict[str, Any]:
        head = self.commits.head
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "events": len(self.log.events),
            "commits": len(self.commits.list()),
            "head_commit": head.commit_id if head else None,
            "messages": len(self.messages),
            "log_digest": self.log.digest_at(),
        }


class RunSessionRegistry:
    """Process-wide registry of live :class:`RunSession` objects, keyed by ``run_id``.

    The seam the ``graph_runvcs`` MCP/REST surface uses to reach a running session across calls
    (mirrors :class:`~agent_utilities.runtime.warm_registry.WarmParentRegistry`). Singleton per
    process; sessions are weakly held only by strong ref here until explicitly dropped.
    """

    _instance: RunSessionRegistry | None = None

    def __init__(self) -> None:
        self._sessions: dict[str, RunSession] = {}

    @classmethod
    def get(cls) -> RunSessionRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, session: RunSession) -> None:
        self._sessions[session.run_id] = session

    def acquire(self, run_id: str) -> RunSession | None:
        return self._sessions.get(run_id)

    def drop(self, run_id: str) -> bool:
        return self._sessions.pop(run_id, None) is not None

    def list_ids(self) -> list[str]:
        return list(self._sessions)
