"""CONCEPT:AHE-3.22 — SWE-bench evaluation harness.

Per instance: provision a developer workspace (OS-5.33), clone the repo at ``base_commit``,
optionally ingest it into the KG so the SWE agent's grounding tools work (KG-2.65), run the
agent on the problem statement, apply the gold ``test_patch``, run the FAIL_TO_PASS and
PASS_TO_PASS selectors, and score "resolved". Aggregation mirrors the LongMemEval router shape
(``server/routers/benchmark.py``): a per-repo breakdown plus a resolved-rate.

The orchestration is injectable — ``solve`` defaults to the KG-grounded SWE agent (ORCH-1.47)
but a test can pass a scripted solver, and the scoring helpers (:func:`is_resolved`,
:func:`aggregate_report`) are pure and LLM-free.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.runtime.events import CmdRunAction, FileWriteAction, TestRunAction

from .swebench_corpus import SweBenchInstance

logger = logging.getLogger(__name__)

# A solver: given the instance + a started workspace, edit the working tree to fix the bug.
Solver = Callable[[SweBenchInstance, Any], Awaitable[str]]


@dataclass
class InstanceResult:
    instance_id: str
    repo: str
    resolved: bool
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    patch: str = ""
    error: str = ""
    trace_run_id: str = ""


def is_resolved(
    f2p_passed: int,
    f2p_total: int,
    p2p_passed: int,
    p2p_total: int,
    f2p_failed: int,
    p2p_failed: int,
) -> bool:
    """Resolved iff every FAIL_TO_PASS now passes AND every PASS_TO_PASS still passes."""
    f2p_ok = f2p_failed == 0 and (f2p_total == 0 or f2p_passed >= f2p_total)
    p2p_ok = p2p_failed == 0 and (p2p_total == 0 or p2p_passed >= p2p_total)
    return f2p_ok and p2p_ok


async def _default_solver(instance: SweBenchInstance, workspace: Any) -> str:
    """Run the KG-grounded SWE agent (ORCH-1.47) against the instance's problem statement."""
    from agent_utilities.models import AgentDeps
    from agent_utilities.orchestration.swe_agent import run_swe_task

    deps = AgentDeps(workspace=workspace)
    result = await run_swe_task(instance.problem_statement, deps)
    return result.patch


async def _provision(instance: SweBenchInstance, workspace: Any) -> None:
    """Clone the repo @ base_commit and run setup commands inside the workspace."""
    if instance.repo_url:
        await workspace.act(
            CmdRunAction(
                command=(
                    f"git clone {instance.repo_url} . && "
                    f"git checkout {instance.base_commit} 2>/dev/null || true"
                ),
                timeout=600,
            )
        )
    for cmd in instance.setup_commands:
        await workspace.act(CmdRunAction(command=cmd, timeout=900))
    # Tag the pre-solve state so the patch can be extracted even if the agent self-commits.
    await workspace.act(
        CmdRunAction(command="git tag -f au_base >/dev/null 2>&1 || true")
    )


async def _apply_test_patch(instance: SweBenchInstance, workspace: Any) -> bool:
    """Apply the gold test_patch (adds/updates the tests). Returns True on success."""
    if not instance.test_patch:
        return True
    await workspace.act(
        FileWriteAction(path=".au_test.patch", content=instance.test_patch)
    )
    obs = await workspace.act(
        CmdRunAction(command="git apply --whitespace=nowarn .au_test.patch")
    )
    return obs.exit_code == 0


async def _run_selectors(workspace: Any, selectors: list[str]) -> tuple[int, int, int]:
    """Run a set of test selectors; return (passed, failed, total_requested)."""
    if not selectors:
        return 0, 0, 0
    obs = await workspace.act(TestRunAction(selector=" ".join(selectors)))
    return obs.passed, obs.failed + obs.errors, len(selectors)


async def evaluate_instance(
    instance: SweBenchInstance,
    *,
    workspace: Any,
    solve: Solver | None = None,
    ingest: bool = False,
    kg: Any = None,
) -> InstanceResult:
    """Provision -> solve -> apply test_patch -> run selectors -> score one instance."""
    solve = solve or _default_solver
    try:
        await _provision(instance, workspace)
        if ingest and kg is not None:
            _ingest_repo(kg, workspace)
        patch = await solve(instance, workspace)
        if not await _apply_test_patch(instance, workspace):
            return InstanceResult(
                instance.instance_id,
                instance.repo,
                False,
                patch=patch,
                error="test_patch failed to apply",
                trace_run_id=getattr(workspace, "run_id", ""),
            )
        f2p_pass, f2p_fail, f2p_total = await _run_selectors(
            workspace, instance.fail_to_pass
        )
        p2p_pass, p2p_fail, p2p_total = await _run_selectors(
            workspace, instance.pass_to_pass
        )
        resolved = is_resolved(
            f2p_pass, f2p_total, p2p_pass, p2p_total, f2p_fail, p2p_fail
        )
        return InstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            resolved=resolved,
            fail_to_pass_passed=f2p_pass,
            fail_to_pass_total=f2p_total,
            pass_to_pass_passed=p2p_pass,
            pass_to_pass_total=p2p_total,
            patch=patch,
            trace_run_id=getattr(workspace, "run_id", ""),
        )
    except Exception as exc:  # noqa: BLE001 - one bad instance must not sink the suite
        logger.warning("instance %s errored: %s", instance.instance_id, exc)
        return InstanceResult(
            instance.instance_id,
            instance.repo,
            False,
            error=str(exc),
            trace_run_id=getattr(workspace, "run_id", ""),
        )


def _ingest_repo(kg: Any, workspace: Any) -> None:
    """Best-effort: ingest the cloned repo into the KG so grounding tools have a graph."""
    try:
        root = str(getattr(workspace.backend, "root", "")) or "."
        ingest = getattr(kg, "ingest_codebase", None) or getattr(
            kg, "ingest_path", None
        )
        if callable(ingest):
            ingest(root)
    except Exception as exc:  # noqa: BLE001
        logger.debug("repo ingest skipped: %s", exc)


def aggregate_report(results: list[InstanceResult]) -> dict[str, Any]:
    """Summarize a suite: resolved-rate + per-repo breakdown (mirrors benchmark.aggregate_report)."""
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    by_repo: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_repo.setdefault(r.repo or "?", {"total": 0, "resolved": 0})
        bucket["total"] += 1
        bucket["resolved"] += int(r.resolved)
    return {
        "total": total,
        "resolved": resolved,
        "unresolved": total - resolved,
        "resolved_rate": round(resolved / total, 4) if total else 0.0,
        "by_repo": by_repo,
        "instances": [
            {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "resolved": r.resolved,
                "error": r.error,
            }
            for r in results
        ],
    }


@dataclass
class SuiteReport:
    results: list[InstanceResult] = field(default_factory=list)

    @property
    def report(self) -> dict[str, Any]:
        return aggregate_report(self.results)


async def run_suite(
    instances: list[SweBenchInstance],
    *,
    workspace_factory: Callable[[SweBenchInstance], Any],
    solve: Solver | None = None,
    ingest: bool = False,
    kg: Any = None,
) -> SuiteReport:
    """Evaluate every instance, each in its own workspace (started + stopped per instance)."""
    report = SuiteReport()
    for inst in instances:
        ws = workspace_factory(inst)
        await ws.start()
        try:
            report.results.append(
                await evaluate_instance(
                    inst, workspace=ws, solve=solve, ingest=ingest, kg=kg
                )
            )
        finally:
            await ws.stop()
    return report
