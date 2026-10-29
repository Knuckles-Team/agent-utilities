"""Run-local changeset cycle; the host supplies the source and sandbox boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from agent_utilities import host_ports
from agent_utilities.harness.edit_engine import Edit
from agent_utilities.runtime.events import CmdOutputObservation
from agent_utilities.runtime.workspace import DevWorkspace


@dataclass(frozen=True)
class GateVerdict:
    index: int
    exit_code: int


@dataclass(frozen=True)
class PendingChangeset:
    base_head: str
    edits: tuple[Edit, ...]
    run_ref: str
    files: tuple[tuple[str, str], ...]
    gate_verdicts: tuple[GateVerdict, ...]


@dataclass(frozen=True)
class Submitted:
    """Host reports submission only; landing remains an owner-side decision."""

    changeset_id: str


@dataclass(frozen=True)
class SourceConflict:
    paths: tuple[str, ...]


class ChangesetSubmitter(Protocol):
    """Host authority for source snapshots and submission, bound to one run."""

    async def read_head(self) -> str: ...

    async def materialize(self, head: str, workspace: DevWorkspace) -> None:
        """Replace the workspace with exactly the source snapshot at head."""
        ...

    async def submit(self, pending: PendingChangeset) -> Submitted | SourceConflict:
        """Submit a candidate; this outcome does not certify a landing."""
        ...


class SandboxGateRunner(Protocol):
    """Host-bound edit, read and gate operations on one isolated workspace."""

    async def apply_edits(
        self, workspace: DevWorkspace, edits: tuple[Edit, ...]
    ) -> bool:
        """Apply only after validating the entire edit set inside the sandbox."""
        ...

    async def read_files(
        self, workspace: DevWorkspace, paths: tuple[str, ...]
    ) -> tuple[tuple[str, str], ...]:
        """Read exact path/content pairs from that same isolated workspace."""
        ...

    async def run_gate(
        self, workspace: DevWorkspace, command: str
    ) -> CmdOutputObservation: ...


@dataclass(frozen=True)
class CycleResult:
    reason: str
    changeset_id: str = ""
    conflict_paths: tuple[str, ...] = ()
    symbol_path: str = ""
    symbol_id: str = ""


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class SymbolChange:
    """The exact symbol edit the host must send to the engine owner."""

    repo: str
    branch: str
    base_rev: str
    symbol_id: str
    kind: Literal[
        "replace_symbol", "replace_body", "insert_before", "insert_after", "delete"
    ]
    anchor_span_digest: str
    new_text: str
    message: str
    run_trace_id: str


@dataclass(frozen=True)
class SymbolSnapshot:
    """One symbol and anchor read together at the host's current branch head."""

    repo: str
    branch: str
    base_rev: str
    symbol_id: str
    anchor_span_digest: str


@dataclass(frozen=True)
class SymbolAccepted:
    changeset_id: str


@dataclass(frozen=True)
class SymbolBaseStale:
    code: Literal["EDIT_BASE_STALE"] = "EDIT_BASE_STALE"


@dataclass(frozen=True)
class SymbolRoundtripMismatch:
    path: str
    symbol_id: str
    code: Literal["EDIT_ROUNDTRIP_MISMATCH"] = "EDIT_ROUNDTRIP_MISMATCH"


class SymbolSubmitter(Protocol):
    """Host port; the engine owns symbol resolution and file projection."""

    async def submit_symbol(
        self, change: SymbolChange
    ) -> SymbolAccepted | SymbolBaseStale | SymbolRoundtripMismatch: ...

    async def read_current_symbol(self, change: SymbolChange) -> SymbolSnapshot | None:
        """Read the symbol and anchor from one authorized current head."""
        ...


async def run_symbol_edit(
    change: SymbolChange, *, submitter: SymbolSubmitter | None
) -> CycleResult:
    """Submit one symbol edit and retry one stale base without local projection."""
    if submitter is None:
        return CycleResult("submitter_unavailable")
    if (
        change.kind
        not in (
            "replace_symbol",
            "replace_body",
            "insert_before",
            "insert_after",
            "delete",
        )
        or not isinstance(change.new_text, str)
        or not all(
            _nonblank(value)
            for value in (
                change.repo,
                change.branch,
                change.base_rev,
                change.symbol_id,
                change.anchor_span_digest,
                change.message,
                change.run_trace_id,
            )
        )
    ):
        return CycleResult("invalid_symbol_edit")
    for attempt in range(2):
        outcome = await submitter.submit_symbol(change)
        if isinstance(outcome, SymbolAccepted):
            if not _nonblank(outcome.changeset_id):
                return CycleResult("submit_refused")
            return CycleResult("accepted", changeset_id=outcome.changeset_id)
        if isinstance(outcome, SymbolRoundtripMismatch):
            if not _nonblank(outcome.path) or not _nonblank(outcome.symbol_id):
                return CycleResult("submit_refused")
            return CycleResult(
                "EDIT_ROUNDTRIP_MISMATCH",
                symbol_path=outcome.path,
                symbol_id=outcome.symbol_id,
            )
        if not isinstance(outcome, SymbolBaseStale):
            return CycleResult("submit_refused")
        if attempt == 1:
            return CycleResult("EDIT_BASE_STALE")
        snapshot = await submitter.read_current_symbol(change)
        if (
            snapshot is None
            or snapshot.repo != change.repo
            or snapshot.branch != change.branch
            or snapshot.symbol_id != change.symbol_id
            or not _nonblank(snapshot.base_rev)
            or not _nonblank(snapshot.anchor_span_digest)
        ):
            return CycleResult("symbol_unavailable")
        change = replace(
            change,
            base_rev=snapshot.base_rev,
            anchor_span_digest=snapshot.anchor_span_digest,
        )
    raise AssertionError("unreachable")


def _safe_paths(root: Path, edits: tuple[Edit, ...]) -> bool:
    try:
        root = root.resolve()
        for edit in edits:
            if not isinstance(edit.path, str) or not edit.path or "\0" in edit.path:
                return False
            path = Path(edit.path)
            if path.is_absolute():
                return False
            try:
                resolved = (root / path).resolve(strict=True)
            except FileNotFoundError:
                resolved = (root / path).resolve()
            if resolved == root or not resolved.is_relative_to(root):
                return False
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _valid_files(files: object, paths: tuple[str, ...]) -> bool:
    if not isinstance(files, tuple) or len(files) != len(paths):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and item[0] == path
        and isinstance(item[1], str)
        for item, path in zip(files, paths, strict=True)
    )


async def run_changeset_cycle(
    workspace: DevWorkspace,
    edits: tuple[Edit, ...],
    *,
    base_head: str,
    run_ref: str,
    fast_gates: tuple[str, ...],
    submitter: ChangesetSubmitter | None,
    gate_runner: SandboxGateRunner | None = None,
    max_rebases: int = 3,
) -> CycleResult:
    """Apply edits to a host-provided snapshot and submit after sandbox gates.

    A submitted outcome does not establish that the changeset landed.
    The workspace must be disposable and exclusive to this cycle; its caller
    owns stopping it on success, failure, and cancellation. Materialization
    and source authorization are host responsibilities. The host also binds
    and verifies the isolated gate runner. An absent provider or gate list
    refuses before touching the workspace.
    """
    if submitter is None:
        return CycleResult("submitter_unavailable")
    if (
        not isinstance(fast_gates, tuple)
        or not fast_gates
        or any(not _nonblank(gate) for gate in fast_gates)
        or not _nonblank(base_head)
        or not _nonblank(run_ref)
        or type(max_rebases) is not int
        or not 0 <= max_rebases <= 3
        or not isinstance(edits, tuple)
        or not edits
        or any(not isinstance(edit, Edit) for edit in edits)
    ):
        return CycleResult("invalid_cycle")
    root = workspace.backend.root
    if not _safe_paths(root, edits):
        return CycleResult("invalid_edit_path")
    if gate_runner is None:
        return CycleResult("sandbox_unavailable")
    await workspace.start()
    head = base_head
    for attempt in range(max_rebases + 1):
        await submitter.materialize(head, workspace)
        # Snapshot materialization can introduce symlinks, so check again.
        if not _safe_paths(root, edits):
            return CycleResult("invalid_edit_path")
        if await gate_runner.apply_edits(workspace, edits) is not True:
            return CycleResult("edit_failed")
        paths = tuple(sorted({edit.path for edit in edits}))
        try:
            files = await gate_runner.read_files(workspace, paths)
        except (OSError, UnicodeError):
            return CycleResult("edit_failed")
        if not _valid_files(files, paths):
            return CycleResult("edit_failed")
        verdicts: list[GateVerdict] = []
        for index, command in enumerate(fast_gates):
            observation = await gate_runner.run_gate(workspace, command)
            if not isinstance(observation, CmdOutputObservation):
                return CycleResult("gate_failed")
            verdict = GateVerdict(index, observation.exit_code)
            verdicts.append(verdict)
            host_ports.emit_run_event(
                {
                    "kind": "changeset_gate",
                    "run_ref": run_ref,
                    "attempt": attempt,
                    "gate_index": index,
                    "exit_code": observation.exit_code,
                }
            )
            if observation.exit_code:
                return CycleResult("gate_failed")
        if not _safe_paths(root, edits):
            return CycleResult("invalid_edit_path")
        try:
            after_gates = await gate_runner.read_files(workspace, paths)
        except (OSError, UnicodeError):
            return CycleResult("gate_modified_edits")
        if not _valid_files(after_gates, paths) or after_gates != files:
            return CycleResult("gate_modified_edits")
        outcome = await submitter.submit(
            PendingChangeset(head, edits, run_ref, files, tuple(verdicts))
        )
        if isinstance(outcome, Submitted):
            if not _nonblank(outcome.changeset_id):
                return CycleResult("submit_refused")
            host_ports.emit_run_event(
                {
                    "kind": "changeset_submitted",
                    "run_ref": run_ref,
                    "changeset_id": outcome.changeset_id,
                }
            )
            return CycleResult("submitted", changeset_id=outcome.changeset_id)
        if not isinstance(outcome, SourceConflict):
            return CycleResult("submit_refused")
        if (
            not isinstance(outcome.paths, tuple)
            or not outcome.paths
            or not all(_nonblank(path) for path in outcome.paths)
        ):
            return CycleResult("submit_refused")
        host_ports.emit_run_event(
            {
                "kind": "changeset_conflict",
                "run_ref": run_ref,
                "attempt": attempt,
                "paths": outcome.paths,
            }
        )
        if attempt == max_rebases:
            return CycleResult("conflict_retry_exhausted", conflict_paths=outcome.paths)
        head = await submitter.read_head()
        if not _nonblank(head):
            return CycleResult("source_head_unavailable")
    raise AssertionError("unreachable")
