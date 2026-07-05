"""CONCEPT:AU-ORCH.runvcs.run-commit — agent-native run version-control tests.

Wire-First: these exercise the REAL run-VCS primitives against REAL temp workspaces — a run
commits its files+messages+event-frontier, forks a child into a fresh workspace, mutates it, and
reverts an exact world back; the event kernel is content-addressed; replay is deterministic; and
the retained-output gate holds a world delta until the action-policy accepts it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.runtime.run_vcs.carrier import FsCarrier
from agent_utilities.runtime.run_vcs.kernel import (
    RunEventLog,
    content_digest,
)
from agent_utilities.runtime.run_vcs.replay import ReplayModel, replay_run
from agent_utilities.runtime.run_vcs.retained_output import (
    RetainedRunGate,
    RetainedRunProposal,
)
from agent_utilities.runtime.run_vcs.run_session import RunSession, RunSessionRegistry


# ── Phase 1: content-addressed typed event kernel ────────────────────────────────
def test_event_identity_is_content_digest():
    log = RunEventLog("run-1")
    e1 = log.declare("cmd_run", {"command": "ls"})
    e2 = log.capture("cmd_run", {"exit": 0}, of=e1)
    # record_id == digest over (schema_ref, mode, payload, caused_by), not position.
    assert e1.record_id == content_digest(
        "cmd_run", "declaration", {"command": "ls"}, ()
    )
    assert e2.caused_by == (e1.record_id,)
    assert e1.ordinal == 0 and e2.ordinal == 1
    assert e1.mode == "declaration" and e2.mode == "capture"


def test_identical_content_dedups_by_digest():
    a = RunEventLog("a").declare("note", {"x": 1})
    b = RunEventLog("b").declare("note", {"x": 1})
    # Same content ⇒ same identity (the CoW/dedup property), independent of run/position.
    assert a.record_id == b.record_id


def test_cut_and_project_returns_causal_prefix():
    log = RunEventLog("run-2")
    for i in range(5):
        log.append("step", {"i": i})
    cut = log.cut(2)
    projected = log.project(cut)
    assert [e.payload["i"] for e in projected] == [0, 1, 2]
    assert cut.through_ordinal == 2
    # The whole log projects when no cut is given.
    assert len(log.project()) == 5


def test_truncate_rewinds_to_frontier():
    log = RunEventLog("run-3")
    for i in range(4):
        log.append("step", {"i": i})
    log.truncate_to(log.cut(1))
    assert [e.payload["i"] for e in log.events] == [0, 1]


# ── Phase 2: carrier CoW snapshot + unified commit ───────────────────────────────
def test_carrier_snapshot_restore_roundtrip(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("world")
    carrier = FsCarrier(root)
    snap = carrier.snapshot()

    # Mutate: change a file, add one, delete one.
    (root / "a.txt").write_text("CHANGED")
    (root / "c.txt").write_text("new")
    (root / "sub" / "b.txt").unlink()

    carrier.restore(snap)
    assert (root / "a.txt").read_text() == "hello"
    assert (root / "sub" / "b.txt").read_text() == "world"
    assert not (root / "c.txt").exists()  # the added file is pruned


def test_snapshot_is_content_addressed(tmp_path: Path):
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    for r in (r1, r2):
        r.mkdir()
        (r / "f.txt").write_text("same")
    assert FsCarrier(r1).snapshot().snapshot_id == FsCarrier(r2).snapshot().snapshot_id


# ── Phase 3: fork / revert of a live run (THE end-to-end demo) ────────────────────
async def test_end_to_end_fork_mutate_revert(tmp_path: Path):
    """Fork→mutate→revert restores files + process(event frontier) + messages."""
    root = tmp_path / "parent"
    session = RunSession("parent-run", root)

    # Build up a run: files + messages + a typed event stream.
    (root / "code.py").write_text("v1")
    session.messages.append({"role": "user", "content": "start"})
    session.log.append("file_write", {"path": "code.py", "content": "v1"})
    commit = await session.commit("checkpoint-A")

    # Advance the run past the commit: mutate files, messages, and events.
    (root / "code.py").write_text("v2-BROKEN")
    (root / "scratch.txt").write_text("junk")
    session.messages.append({"role": "assistant", "content": "made a mess"})
    session.log.append("file_write", {"path": "code.py", "content": "v2-BROKEN"})
    assert len(session.log.events) > commit.event_cut.through_ordinal + 1

    # FORK a child from commit-A into a fresh workspace — parent untouched.
    child = await session.fork(commit, new_run_id="child-run")
    assert child.run_id == "child-run"
    assert (child.root / "code.py").read_text() == "v1"  # child sees committed world
    assert not (child.root / "scratch.txt").exists()
    assert (root / "code.py").read_text() == "v2-BROKEN"  # parent still mutated
    # Child's messages + event prefix came from the commit.
    assert child.messages == [{"role": "user", "content": "start"}]

    # REVERT the parent to commit-A: files + events + messages all restored.
    result = await session.revert(commit)
    assert (root / "code.py").read_text() == "v1"
    assert not (root / "scratch.txt").exists()
    assert session.messages == [{"role": "user", "content": "start"}]
    assert result["files_removed"] >= 1
    # Event frontier rewound to the commit's cut.
    assert all(
        e.ordinal <= commit.event_cut.through_ordinal for e in session.log.events
    )


async def test_discard_drops_event_delta_without_touching_files(tmp_path: Path):
    root = tmp_path / "ws"
    session = RunSession("run-d", root)
    (root / "f.txt").write_text("keep")
    session.log.append("step", {"n": 0})
    await session.commit("A")
    session.log.append("step", {"n": 1})
    (root / "f.txt").write_text("still-here")
    out = session.discard()
    assert out["discarded_events"] >= 1
    # Files are deliberately NOT reverted by discard.
    assert (root / "f.txt").read_text() == "still-here"


# ── Phase 4: deterministic trace replay ──────────────────────────────────────────
def test_replay_is_deterministic_and_offline():
    log = RunEventLog("run-r")
    # Record two model exchanges as declaration→capture pairs.
    d1 = log.declare("model_exchange", {"request": "2+2?"})
    log.capture("model_exchange", {"response": "4"}, of=d1)
    d2 = log.declare("model_exchange", {"request": "capital of France?"})
    log.capture("model_exchange", {"response": "Paris"}, of=d2)

    result = replay_run(log)
    assert result.deterministic
    assert result.model_calls == 2  # replay used the recorded model, not a live one
    assert result.reconstructed == ["4", "Paris"]


def test_replay_model_answers_by_request_digest():
    log = RunEventLog("run-r2")
    d = log.declare("model_exchange", {"request": "ping"})
    log.capture("model_exchange", {"response": "pong"}, of=d)
    model = ReplayModel.from_log(log)
    assert model.respond("ping") == "pong"


# ── Phase 5: retained-run-output review gate ─────────────────────────────────────
async def test_retained_output_discard_leaves_world_untouched(tmp_path: Path):
    root = tmp_path / "ws"
    session = RunSession("run-p", root)
    (root / "out.txt").write_text("final")
    commit = await session.commit("final")
    proposal = RetainedRunProposal(run_id="run-p", commit=commit)

    gate = RetainedRunGate()  # no engine → offline, audit best-effort
    out = gate.discard(proposal)
    assert out["discarded"] and out["world_touched"] is False
    assert proposal.discarded


async def test_retained_output_select_gated_by_action_policy(tmp_path: Path):
    # Default policy: run.select is approval_required → the world is NOT materialized.
    root = tmp_path / "ws"
    target = tmp_path / "live"
    session = RunSession("run-s", root)
    (root / "out.txt").write_text("proposed")
    commit = await session.commit("final")
    proposal = RetainedRunProposal(run_id="run-s", commit=commit)

    gate = RetainedRunGate()
    res = gate.select(proposal, session.carrier, target_root=target)
    assert res["materialized"] is False
    assert res["decision"] in {"queue_approval", "deny"}
    assert not (target / "out.txt").exists()


async def test_retained_output_select_materializes_when_policy_allows(tmp_path: Path):
    """With a policy that auto-allows run.select, accept materializes the held fs delta."""
    policy_yml = tmp_path / "policy.yml"
    policy_yml.write_text(
        "version: 1\n"
        "defaults: {tier: approval_required}\n"
        "rules:\n"
        "  - {kind: run.select, target: '*', tier: auto}\n",
        encoding="utf-8",
    )
    from agent_utilities.orchestration.action_policy import ActionPolicy

    root = tmp_path / "ws"
    target = tmp_path / "live"
    session = RunSession("run-ok", root)
    (root / "out.txt").write_text("accepted")
    commit = await session.commit("final")
    proposal = RetainedRunProposal(run_id="run-ok", commit=commit)

    gate = RetainedRunGate(policy=ActionPolicy(policy_path=policy_yml))
    res = gate.select(proposal, session.carrier, target_root=target)
    assert res["materialized"] is True
    assert (target / "out.txt").read_text() == "accepted"
    assert proposal.materialized


# ── registry + surface plumbing ──────────────────────────────────────────────────
def test_session_registry_addresses_live_runs(tmp_path: Path):
    session = RunSession("reg-run", tmp_path / "ws")
    RunSessionRegistry.get().register(session)
    assert "reg-run" in RunSessionRegistry.get().list_ids()
    assert RunSessionRegistry.get().acquire("reg-run") is session
    assert RunSessionRegistry.get().drop("reg-run")


@pytest.mark.asyncio
async def test_graph_runvcs_surface_is_registered():
    """The MCP tool + REST twin exist (surface parity)."""
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    assert "graph_runvcs" in kg_server.REGISTERED_TOOLS
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_runvcs") == "/graph/runvcs"
