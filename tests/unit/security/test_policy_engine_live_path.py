"""Wire-First live-path test for the guardrails ``PolicyEngine`` (OS-5.x).

Before this wiring, ``agent_utilities.security.guardrails.PolicyEngine`` — and
its ``PromptInjectionPolicy``/``ContentFilterPolicy``/``CostBudgetPolicy``
rules — was importable and fully unit-tested but never invoked on any live
call path: the only places ``PolicyEngine()`` appeared outside tests were
docstring ``Example::`` blocks in ``threat_defense_engine.py`` and
``execution_stability_engine.py`` (never executed) plus the re-export in
``security/__init__.py``. Classic "reachable != invoked" dead code.

``Orchestrator.__init__`` (``agent_utilities/orchestration/manager.py``) now
builds one ``PolicyEngine`` with those rules registered, and
``Orchestrator._scan_task`` — the chokepoint every ``dispatch_task``/
``execute_agent``/``execute_workflow`` call runs through — evaluates the task
text through it. This test drives the REAL ``Orchestrator.dispatch_task``
entry point (not a bare ``PolicyEngine()`` construction) and asserts a policy
violation actually blocks the dispatch as a side effect.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.manager import Orchestrator


def _orchestrator() -> Orchestrator:
    # ``_scan_task`` never touches ``self.engine`` — a bare ``object()`` is the
    # same fixture pattern used by other Orchestrator unit tests (e.g.
    # ``tests/unit/test_structured_response_format.py``).
    return Orchestrator(engine=object())


class TestPolicyEngineLivePath:
    async def test_dispatch_task_blocked_by_policy_engine_live_path(self) -> None:
        """A PII-bearing task is rejected by ``dispatch_task`` itself.

        This is the real production entry point (an MCP job-dispatch tool
        calls it) — not a direct ``PolicyEngine().evaluate(...)`` call — so it
        proves the engine is actually wired into the live request path, and
        will fail again if a future refactor orphans the wiring.
        """
        orchestrator = _orchestrator()

        with pytest.raises(ValueError, match="Security Alert"):
            await orchestrator.dispatch_task(
                "Please onboard this employee, SSN 123-45-6789, into payroll."
            )

    async def test_dispatch_task_allows_benign_task_live_path(self) -> None:
        """A benign task is unaffected — the gate is not a blanket block."""
        orchestrator = _orchestrator()

        # dispatch_task's downstream KG plumbing can't complete against a bare
        # ``object()`` fixture engine — but that failure happens strictly
        # AFTER _scan_task, so any exception raised here must NOT be the
        # policy gate's "Security Alert" ValueError. That is the side effect
        # this test asserts: the benign task cleared the live policy gate.
        with pytest.raises(Exception) as exc_info:
            await orchestrator.dispatch_task("Summarize last night's log volume.")
        assert "Security Alert" not in str(exc_info.value)

    def test_orchestrator_policy_engine_reachable(self) -> None:
        """The engine constructed in ``Orchestrator.__init__`` carries the rules.

        Guards against the wiring being silently narrowed back down to just
        the raw scanner call.
        """
        orchestrator = _orchestrator()
        rule_names = {rule.name for rule in orchestrator._policy_engine.rules}
        assert {"prompt_injection", "content_filter", "cost_budget"} <= rule_names
