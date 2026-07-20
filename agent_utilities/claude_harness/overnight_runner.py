#!/usr/bin/python
from __future__ import annotations

"""Unattended 'run while you sleep' loop-driver core.

CONCEPT:AU-AHE.harness.overnight-loop-driver — Unattended overnight loop-driver + morning-summary write-back
CONCEPT:AU-OS.scaling.unattended-session-stop-ask — Unattended-session stop-on-ask containment

The ``unattended-loop-driver`` skill hands Claude Code a goal and a permission
fence; this module is the testable core it leans on. It drives the existing
:class:`LoopController` (KG-2.78) — the same feature-extraction + innovation-
distillation cycle the daemon ticks — once per iteration, commits after each
productive cycle, stops when the loop converges (no new progress) or a cap is
hit, and writes a morning summary into ``MEMORY.md`` so the existing memory
bridge (``inject_project_context``, KG-2.1) surfaces it on the next SessionStart.

There is no human to answer an ``ask`` mid-run, so this core never auto-approves
anything: the *permission fence* (OS-5.40/5.41) halts ``ask``/``deny`` tool calls,
and this loop only advances the propose-only Loop cycle, which writes proposals,
never executes high-stakes actions (SAFE-1.8 containment).
"""

import logging
import subprocess  # nosec B404 — git invocation with a fixed, non-shell argv
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["run_session", "write_morning_summary", "cycle_productive"]

_MEMORY_START = "<!-- agent-utilities:unattended-loop-driver START -->"
_MEMORY_END = "<!-- agent-utilities:unattended-loop-driver END -->"


def cycle_productive(report: dict[str, Any]) -> bool:
    """True when a Loop cycle advanced something worth committing.

    Reads the structured report from :meth:`LoopController.run_one_cycle`:
    resolved topics, newly linked sources, distilled skill proposals, or an
    executed develop/skill Loop all count as progress.
    """
    if not isinstance(report, dict):
        return False
    score = int(report.get("topics_resolved") or 0) + int(
        report.get("sources_linked") or 0
    )
    proposals = report.get("skill_proposals")
    if isinstance(proposals, dict):
        score += int(proposals.get("proposed") or 0)
    executed = report.get("executed")
    if isinstance(executed, dict):
        score += int(executed.get("advanced") or executed.get("executed") or 0)
    return score > 0


def _acquire_engine() -> Any:
    """The live engine, via the same singleton the MCP/daemon paths use."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return engine
    from agent_utilities.mcp import kg_server

    return kg_server._get_engine()


def _git_commit(workspace: Path, message: str) -> dict[str, Any]:
    """Stage all + commit in ``workspace``. Best-effort; never raises."""
    try:
        subprocess.run(  # nosec B603,B607 — fixed argv, no shell
            ["git", "-C", str(workspace), "add", "-A"],
            check=True,
            capture_output=True,
            text=True,
        )
        proc = subprocess.run(  # nosec B603,B607
            ["git", "-C", str(workspace), "commit", "-m", message],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return {"committed": True, "message": message}
        return {
            "committed": False,
            "reason": (proc.stdout or proc.stderr or "").strip()[:200],
        }
    except Exception as e:  # noqa: BLE001 — a commit failure must not abort the run
        return {"committed": False, "reason": str(e)}


def run_session(
    *,
    max_cycles: int = 6,
    max_topics: int = 5,
    commit: bool = True,
    convergence_patience: int = 2,
    workspace: str | Path | None = None,
    engine: Any = None,
    controller: Any = None,
    write_summary: bool = True,
) -> dict[str, Any]:
    """Drive Loop cycles unattended until convergence or ``max_cycles``.

    Args:
        max_cycles: hard cap on cycles (the morning is finite).
        max_topics: Loops advanced per cycle (passed to ``run_one_cycle``).
        commit: commit after each productive cycle.
        convergence_patience: stop after this many consecutive empty cycles.
        workspace: repo root for commits + ``MEMORY.md`` (default: cwd).
        engine / controller: injectable for tests (default: live singletons).
        write_summary: write the ``MEMORY.md`` morning summary on exit.

    Returns a JSON-able session report (cycles, commits, stop reason, summary
    path).
    """
    root = Path(workspace).expanduser() if workspace else Path.cwd()
    if controller is None:
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )

        controller = LoopController(engine or _acquire_engine())

    cycles: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    empty_streak = 0
    stop_reason = "max_cycles"
    started = time.time()

    for i in range(1, max_cycles + 1):
        report = controller.run_one_cycle(max_topics=max_topics)
        productive = cycle_productive(report)
        metrics = report.get("metrics") if isinstance(report, dict) else None
        cycles.append(
            {
                "cycle": i,
                "productive": productive,
                "topics_resolved": (report or {}).get("topics_resolved"),
                "sources_linked": (report or {}).get("sources_linked"),
                "errors": len((report or {}).get("errors") or []),
                "metrics": metrics,
            }
        )
        if productive:
            empty_streak = 0
            if commit:
                msg = _commit_message(i, report)
                commits.append(_git_commit(root, msg))
        else:
            empty_streak += 1
            if empty_streak >= convergence_patience:
                stop_reason = "converged"
                break

    report_out: dict[str, Any] = {
        "cycles_run": len(cycles),
        "productive_cycles": sum(1 for c in cycles if c["productive"]),
        "commits": commits,
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - started, 2),
        "cycles": cycles,
    }
    if write_summary:
        report_out["summary_path"] = str(write_morning_summary(report_out, root))
    return report_out


def _commit_message(cycle: int, report: dict[str, Any]) -> str:
    resolved = (report or {}).get("topics_resolved") or 0
    linked = (report or {}).get("sources_linked") or 0
    proposals = report.get("skill_proposals") if isinstance(report, dict) else None
    proposed = proposals.get("proposed") if isinstance(proposals, dict) else 0
    return (
        f"chore(loop): unattended cycle {cycle} — "
        f"{resolved} topics resolved, {linked} sources linked, {proposed} skill proposals"
    )


def write_morning_summary(report: dict[str, Any], workspace: str | Path) -> Path:
    """Write the session summary into ``<workspace>/MEMORY.md``.

    Reuses the memory bridge's read path (``inject_project_context`` reads
    ``MEMORY.md`` on SessionStart, KG-2.1) — no new surfacing mechanism. The
    block is delimited by stable markers so successive runs replace it in place
    rather than appending duplicates.
    """
    root = Path(workspace).expanduser()
    path = root / "MEMORY.md"
    block = _render_summary(report)

    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — unreadable → start fresh
            logger.warning("overnight_runner: unreadable MEMORY.md (%s)", e)
            existing = ""

    if _MEMORY_START in existing and _MEMORY_END in existing:
        pre = existing.split(_MEMORY_START, 1)[0]
        post = existing.split(_MEMORY_END, 1)[1]
        new_text = f"{pre}{block}{post}"
    else:
        sep = "" if not existing or existing.endswith("\n") else "\n"
        new_text = f"{existing}{sep}\n{block}\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return path


def _render_summary(report: dict[str, Any]) -> str:
    commits = report.get("commits") or []
    committed = sum(1 for c in commits if c.get("committed"))
    lines = [
        _MEMORY_START,
        "## Unattended loop session (CONCEPT:AU-AHE.harness.overnight-loop-driver)",
        "",
        f"- Cycles run: **{report.get('cycles_run', 0)}** "
        f"({report.get('productive_cycles', 0)} productive)",
        f"- Commits: **{committed}**",
        f"- Stop reason: `{report.get('stop_reason', 'unknown')}`",
        f"- Elapsed: {report.get('elapsed_s', 0)}s",
    ]
    for c in report.get("cycles", []):
        lines.append(
            f"  - cycle {c.get('cycle')}: "
            f"{'productive' if c.get('productive') else 'empty'}, "
            f"{c.get('topics_resolved') or 0} resolved, "
            f"{c.get('errors') or 0} errors"
        )
    lines += [
        "",
        "_Review the commit history + proposals; approve any `ask`-list items "
        "the gate halted (SAFE-1.8)._",
        _MEMORY_END,
    ]
    return "\n".join(lines)
