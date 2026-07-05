"""CONCEPT:AU-ORCH.runvcs.run-commit — ONE commit unifying messages + filesystem + process.

Our three fork primitives never composed: the message checkpoint
(:class:`~agent_utilities.capabilities.checkpointing.Checkpoint`) rewinds a conversation, the
warm-fork sandbox forks a *snippet*, and the fs carrier snapshots *files*. Shepherd's insight is
that an agent↔environment interaction only reverts correctly if all three move together — a
reverted conversation over a mutated filesystem is a corrupt world.

A :class:`RunCommit` is that single object: it binds one message checkpoint, one
:class:`~.carrier.FsSnapshot`, and one :class:`~.kernel.RunCut` (the process/event frontier)
into a content-addressed commit whose id is a digest of the three parts. Committing is cheap —
the carrier dedups blobs and the event cut is metadata — so a run can commit at every step and
fork/revert from any of them.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .carrier import FsSnapshot
from .kernel import RunCut


@dataclass(frozen=True)
class RunCommit:
    """One content-addressed run commit binding messages + fs + process-frontier.

    * ``checkpoint_id`` — the message-list checkpoint (conversation state).
    * ``fs_snapshot`` — the filesystem CoW snapshot.
    * ``event_cut`` — the run-event-log frontier (the "process"/effect boundary).

    ``commit_id`` digests the three so two runs that reach the same world share a commit id —
    the run-scope analogue of the event kernel's ``record_id == digest``.
    """

    run_id: str
    checkpoint_id: str
    fs_snapshot: FsSnapshot
    event_cut: RunCut
    label: str = ""
    parent_commit_id: str | None = None
    ts: float = field(default_factory=time.time)

    @property
    def commit_id(self) -> str:
        material = json.dumps(
            {
                "run_id": self.run_id,
                "checkpoint_id": self.checkpoint_id,
                "fs": self.fs_snapshot.snapshot_id,
                "cut": self.event_cut.frontier_id,
                "parent": self.parent_commit_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "runcommit:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_id": self.commit_id,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "fs_snapshot": self.fs_snapshot.to_dict(),
            "event_cut": {
                "run_id": self.event_cut.run_id,
                "through_ordinal": self.event_cut.through_ordinal,
                "through_record_id": self.event_cut.through_record_id,
            },
            "label": self.label,
            "parent_commit_id": self.parent_commit_id,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunCommit:
        cut = data["event_cut"]
        return cls(
            run_id=str(data["run_id"]),
            checkpoint_id=str(data["checkpoint_id"]),
            fs_snapshot=FsSnapshot.from_dict(data["fs_snapshot"]),
            event_cut=RunCut(
                run_id=str(cut["run_id"]),
                through_ordinal=int(cut["through_ordinal"]),
                through_record_id=str(cut["through_record_id"]),
            ),
            label=str(data.get("label") or ""),
            parent_commit_id=data.get("parent_commit_id"),
            ts=float(data.get("ts") or time.time()),
        )


class RunCommitStore:
    """A durable, ordered store of :class:`RunCommit` objects for a run's history.

    In-memory by default; pass ``directory`` to persist each commit as JSON so a run's history
    survives a process restart (the fork/revert graph is durable). Ordered by insertion, which
    is the commit lineage.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._commits: dict[str, RunCommit] = {}
        self._order: list[str] = []
        self.directory = Path(directory).resolve() if directory else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        assert self.directory is not None
        for path in sorted(self.directory.glob("*.json")):
            try:
                commit = RunCommit.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, KeyError):
                continue
            self._commits[commit.commit_id] = commit
            self._order.append(commit.commit_id)

    def save(self, commit: RunCommit) -> str:
        cid = commit.commit_id
        if cid not in self._commits:
            self._order.append(cid)
        self._commits[cid] = commit
        if self.directory is not None:
            (self.directory / f"{cid.replace(':', '_')}.json").write_text(
                json.dumps(commit.to_dict()), encoding="utf-8"
            )
        return cid

    def get(self, commit_id: str) -> RunCommit | None:
        return self._commits.get(commit_id)

    def list(self) -> list[RunCommit]:
        return [self._commits[c] for c in self._order]

    @property
    def head(self) -> RunCommit | None:
        return self._commits[self._order[-1]] if self._order else None
