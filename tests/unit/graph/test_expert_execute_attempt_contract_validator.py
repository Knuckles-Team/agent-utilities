"""BUG-CX-070: ``_expert_execute_attempt`` used to mask a genuine
``ContractValidator`` crash as an intentional "not configured" skip.

``ContractValidator.validate_pre``/``validate_post`` (agent_utilities/harness/
contract_validator.py) already catch every exception a registered
pre-condition/post-condition callable can raise internally and turn it into a
plain ``False`` return -- so an ``Exception`` reaching
``_expert_execute_attempt``'s own ``try/except`` around the contract check can
only come from the contract-check plumbing itself (e.g. ``ContractValidator
.instance()`` raising, or building ``state_context`` raising), never from a
"contract not configured for this node" condition. That except block treated
ANY such exception the same way as "no contract registered" -- logging it at
DEBUG as "Contract pre-validation skipped" and letting execution proceed --
which silently swallows a real bug in the validation path and reports a
contract violation as an all-clear. Fail closed: a crash must fail the
attempt, not be treated as a skip.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_graph.step import StepContext

from agent_utilities.graph import _router_impl
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.harness import contract_validator
from agent_utilities.models import ExecutionStep


class _CrashingValidator:
    """Stands in for a genuinely broken ContractValidator plumbing path.

    Real ``ContractValidator.validate_pre``/``validate_post`` never raise --
    they catch everything from a registered contract callable internally and
    return ``False``. So this simulates the ONLY realistic way an exception
    reaches ``_expert_execute_attempt``'s contract-check try/except: something
    upstream of the boolean check itself is broken.
    """

    def validate_pre(self, *_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("contract validator internal crash")

    def validate_post(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


@pytest.mark.asyncio
async def test_expert_execute_attempt_does_not_mask_a_genuine_validator_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract_validator.ContractValidator,
        "instance",
        classmethod(lambda _cls: _CrashingValidator()),
    )

    dispatched = {"called": False}

    async def _fake_dispatch(_ctx: StepContext, _node_id: str, _step: Any) -> None:
        dispatched["called"] = True

    monkeypatch.setattr(_router_impl, "_expert_dispatch_step_handler", _fake_dispatch)

    state = GraphState(query="test query for contract validator crash")
    deps = GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])
    ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)
    step = ExecutionStep(id="researcher", description="do the thing")

    with pytest.raises(RuntimeError, match="contract validator internal crash"):
        await _router_impl._expert_execute_attempt(
            ctx, "researcher", step, max_retries=0
        )

    assert dispatched["called"] is False, (
        "the node was dispatched/executed despite a genuine ContractValidator "
        "crash -- the crash was masked as an intentional 'skip' instead of "
        "failing the attempt"
    )
