from dataclasses import replace
from pathlib import Path

import pytest

from agent_utilities.harness.changeset_loop import (
    SourceConflict,
    Submitted,
    SymbolAccepted,
    SymbolBaseStale,
    SymbolChange,
    SymbolRoundtripMismatch,
    SymbolSnapshot,
    run_changeset_cycle,
    run_symbol_edit,
)
from agent_utilities.harness.edit_engine import Edit, apply_edits
from agent_utilities.runtime.events import CmdRunAction
from agent_utilities.runtime.local_workspace import LocalWorkspace
from agent_utilities.runtime.workspace import DevWorkspace


class Source:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.heads = []
        self.submitted = []

    async def materialize(self, head, workspace):
        self.heads.append(head)
        (workspace.backend.root / "a.txt").write_text("before\n")

    async def read_head(self):
        return "new-head"

    async def submit(self, pending):
        self.submitted.append(pending)
        return next(self.outcomes)

    async def run_gate(self, workspace, command):
        # A test double for control flow; LocalWorkspace is not a sandbox.
        return await workspace.act(CmdRunAction(command=command))

    async def apply_edits(self, workspace, edits):
        if not apply_edits(list(edits), root=workspace.backend.root, dry_run=True).ok:
            return False
        return apply_edits(list(edits), root=workspace.backend.root).ok

    async def read_files(self, workspace, paths):
        return tuple(
            (path, (workspace.backend.root / path).read_text(encoding="utf-8"))
            for path in paths
        )


def workspace(root: Path) -> DevWorkspace:
    return DevWorkspace(LocalWorkspace(root), run_id="run")


class SymbolSource:
    def __init__(self, outcomes, snapshot):
        self.outcomes = iter(outcomes)
        self.snapshot = snapshot
        self.submitted = []
        self.reads = []

    async def submit_symbol(self, change):
        self.submitted.append(change)
        return next(self.outcomes)

    async def read_current_symbol(self, change):
        self.reads.append(change)
        return self.snapshot


def symbol_change():
    return SymbolChange(
        repo="repo",
        branch="main",
        base_rev="old",
        symbol_id="symbol",
        kind="replace_body",
        anchor_span_digest="old-anchor",
        new_text="new body",
        message="update symbol",
        run_trace_id="run",
    )


@pytest.mark.asyncio
async def test_symbol_stale_reads_one_current_anchor_and_retries():
    change = symbol_change()
    source = SymbolSource(
        [SymbolBaseStale(), SymbolAccepted("change")],
        SymbolSnapshot("repo", "main", "new", "symbol", "new-anchor"),
    )
    result = await run_symbol_edit(change, submitter=source)
    assert result.reason == "accepted"
    assert result.reason == "accepted" and result.changeset_id == "change"
    assert source.reads == [change]
    assert source.submitted[0] == change
    assert source.submitted[1] == SymbolChange(
        repo="repo",
        branch="main",
        base_rev="new",
        symbol_id="symbol",
        kind="replace_body",
        anchor_span_digest="new-anchor",
        new_text="new body",
        message="update symbol",
        run_trace_id="run",
    )


@pytest.mark.asyncio
async def test_symbol_roundtrip_refusal_is_returned_without_retry():
    source = SymbolSource([SymbolRoundtripMismatch("pkg/a.py", "symbol")], None)
    result = await run_symbol_edit(symbol_change(), submitter=source)
    assert result.reason == "EDIT_ROUNDTRIP_MISMATCH"
    assert result.symbol_path == "pkg/a.py"
    assert result.symbol_id == "symbol"
    assert len(source.submitted) == 1
    assert source.reads == []


@pytest.mark.asyncio
async def test_symbol_second_stale_stops_after_one_retry():
    source = SymbolSource(
        [SymbolBaseStale(), SymbolBaseStale()],
        SymbolSnapshot("repo", "main", "new", "symbol", "new-anchor"),
    )
    result = await run_symbol_edit(symbol_change(), submitter=source)
    assert result.reason == "EDIT_BASE_STALE"
    assert len(source.submitted) == 2
    assert len(source.reads) == 1


@pytest.mark.asyncio
async def test_symbol_changed_identity_refuses_before_retry():
    source = SymbolSource(
        [SymbolBaseStale()],
        SymbolSnapshot("repo", "main", "new", "other", "new-anchor"),
    )
    result = await run_symbol_edit(symbol_change(), submitter=source)
    assert result.reason == "symbol_unavailable"
    assert len(source.submitted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("repo,branch", [("other", "main"), ("repo", "other")])
async def test_symbol_changed_source_refuses_before_retry(repo, branch):
    source = SymbolSource(
        [SymbolBaseStale()], SymbolSnapshot(repo, branch, "new", "symbol", "new-anchor")
    )
    result = await run_symbol_edit(symbol_change(), submitter=source)
    assert result.reason == "symbol_unavailable"
    assert len(source.submitted) == 1


@pytest.mark.asyncio
async def test_symbol_unbound_refuses_without_source_effect():
    result = await run_symbol_edit(symbol_change(), submitter=None)
    assert result.reason == "submitter_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "repo",
        "branch",
        "base_rev",
        "symbol_id",
        "anchor_span_digest",
        "message",
        "run_trace_id",
    ],
)
async def test_invalid_symbol_request_refuses_before_source_effect(field):
    source = SymbolSource([], None)
    result = await run_symbol_edit(
        replace(symbol_change(), **{field: " "}), submitter=source
    )
    assert result.reason == "invalid_symbol_edit"
    assert source.submitted == []


@pytest.mark.asyncio
async def test_empty_symbol_acceptance_is_not_reported_as_success():
    source = SymbolSource([SymbolAccepted("")], None)
    result = await run_symbol_edit(symbol_change(), submitter=source)
    assert result.reason == "submit_refused"


@pytest.mark.asyncio
async def test_rebases_on_conflict(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        "agent_utilities.harness.changeset_loop.host_ports.emit_run_event",
        events.append,
    )
    source = Source([SourceConflict(("a.txt",)), Submitted("change")])
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "submitted" and result.changeset_id == "change"
    assert source.heads == ["old", "new-head"]
    assert [p.base_head for p in source.submitted] == ["old", "new-head"]
    assert [
        [(v.index, v.exit_code) for v in p.gate_verdicts] for p in source.submitted
    ] == [[(0, 0)], [(0, 0)]]
    assert [e["kind"] for e in events] == [
        "changeset_gate",
        "changeset_conflict",
        "changeset_gate",
        "changeset_submitted",
    ]
    assert events[1]["paths"] == ("a.txt",)
    assert (tmp_path / "a.txt").read_text() == "after\n"


@pytest.mark.asyncio
async def test_failed_gate_does_not_submit(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        "agent_utilities.harness.changeset_loop.host_ports.emit_run_event",
        events.append,
    )
    source = Source([])
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("false",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "gate_failed"
    assert source.submitted == []
    assert events == [
        {
            "kind": "changeset_gate",
            "run_ref": "run",
            "attempt": 0,
            "gate_index": 0,
            "exit_code": 1,
        }
    ]


@pytest.mark.asyncio
async def test_unbound_refuses_before_edit(tmp_path):
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("a.txt", "", "after"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=None,
    )
    assert result.reason == "submitter_unavailable"
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.asyncio
async def test_missing_sandbox_runner_refuses_before_materialization(tmp_path):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("a.txt", "before", "after"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
    )
    assert result.reason == "sandbox_unavailable"
    assert source.heads == []
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.asyncio
async def test_conflict_bound(tmp_path):
    source = Source([SourceConflict(("a.txt",))] * 4)
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "conflict_retry_exhausted"
    assert result.conflict_paths == ("a.txt",)
    assert len(source.submitted) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, 2, 3])
async def test_received_conflict_limit_bounds_submissions(tmp_path, limit):
    source = Source([SourceConflict(("a.txt",))] * (limit + 1))
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
            max_rebases=limit,
        )
    finally:
        await ws.stop()
    assert result.reason == "conflict_retry_exhausted"
    assert len(source.submitted) == limit + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [Submitted(""), Submitted("  "), SourceConflict(()), SourceConflict((" ",))],
)
async def test_invalid_source_outcome_refuses_without_retry(tmp_path, outcome):
    source = Source([outcome])
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "submit_refused"
    assert len(source.submitted) == 1


@pytest.mark.asyncio
async def test_unavailable_new_head_refuses_before_retry(tmp_path):
    source = Source([SourceConflict(("a.txt",))])

    async def missing_head():
        return " "

    source.read_head = missing_head
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "source_head_unavailable"
    assert len(source.submitted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [4, -1, True, 0.5])
async def test_invalid_rebase_limit_refused_before_workspace_changes(tmp_path, limit):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("a.txt", "before", "after"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
        max_rebases=limit,
    )
    assert result.reason == "invalid_cycle"
    assert source.heads == []


@pytest.mark.asyncio
async def test_invalid_gate_sequence_refused_before_workspace_changes(tmp_path):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("a.txt", "before", "after"),),
        base_head="old",
        run_ref="run",
        fast_gates="true",
        submitter=source,
    )
    assert result.reason == "invalid_cycle"
    assert source.heads == []


@pytest.mark.asyncio
async def test_invalid_edit_refused_before_workspace_changes(tmp_path):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        ("invalid",),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
    )
    assert result.reason == "invalid_cycle"
    assert source.heads == []


@pytest.mark.asyncio
async def test_gate_modifying_edit_refuses_submission(tmp_path):
    source = Source([])
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("printf changed > a.txt",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "gate_modified_edits"
    assert source.submitted == []


@pytest.mark.asyncio
async def test_sandbox_cannot_substitute_a_different_path(tmp_path):
    source = Source([])

    async def wrong_path(workspace, paths):
        return (("other.txt", "after\n"),)

    source.read_files = wrong_path
    ws = workspace(tmp_path)
    try:
        result = await run_changeset_cycle(
            ws,
            (Edit("a.txt", "before", "after"),),
            base_head="old",
            run_ref="run",
            fast_gates=("true",),
            submitter=source,
            gate_runner=source,
        )
    finally:
        await ws.stop()
    assert result.reason == "edit_failed"
    assert source.submitted == []


@pytest.mark.asyncio
async def test_path_escape_refused_before_materialization(tmp_path):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("../escape", "", "bad"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
    )
    assert result.reason == "invalid_edit_path"
    assert source.heads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["", "bad\0path", None])
async def test_malformed_path_refused_before_materialization(tmp_path, path):
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit(path, "", "bad"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
    )
    assert result.reason == "invalid_edit_path"
    assert source.heads == []


@pytest.mark.asyncio
async def test_symlink_loop_refused_before_materialization(tmp_path):
    (tmp_path / "loop").symlink_to("loop")
    source = Source([])
    result = await run_changeset_cycle(
        workspace(tmp_path),
        (Edit("loop/file", "", "bad"),),
        base_head="old",
        run_ref="run",
        fast_gates=("true",),
        submitter=source,
    )
    assert result.reason == "invalid_edit_path"
    assert source.heads == []
