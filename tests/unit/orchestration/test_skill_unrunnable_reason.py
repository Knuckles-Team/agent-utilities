"""Regression coverage for D-SNV-5's "degraded read is not an honest negative" fix.

``_skill_unrunnable_reason`` used to collapse two very different situations
into the same plain string: "the graph told us the skill is genuinely
unrunnable" and "we could not even read whether it is runnable" (a transient
engine/session failure, e.g. ``SessionExpiredError`` mid-delegation). The
caller in ``run_agent`` then always raised ``LookupError(f"... is not
runnable: {reason}")`` — phrasing a failed read as a confirmed negative
finding about the skill itself, which cost real diagnostic time chasing a
"broken skill" that was never actually checked.

The fix makes ``_skill_unrunnable_reason`` return ``(reason, degraded)`` and
has ``run_agent`` raise a different, honestly-labeled exception
(``RuntimeError``, not ``LookupError``) when ``degraded`` is True.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration.agent_runner import _skill_unrunnable_reason


def _engine_with_backend(execute) -> MagicMock:
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute = execute
    return engine


# --------------------------------------------------------------------------- #
# _skill_unrunnable_reason: the (reason, degraded) contract
# --------------------------------------------------------------------------- #


def test_read_failure_is_reported_as_degraded_not_a_finding():
    """A raised exception during the read is a DEGRADED result, not a negative."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("engine unreachable")

    engine = _engine_with_backend(_boom)

    reason, degraded = _skill_unrunnable_reason(
        engine, "servicenow-incident-management"
    )

    assert degraded is True
    assert "could not be read" in reason
    assert "RuntimeError" in reason
    assert "engine unreachable" in reason


def test_missing_skill_node_is_a_real_finding_not_degraded():
    engine = _engine_with_backend(lambda *_a, **_k: [])

    reason, degraded = _skill_unrunnable_reason(engine, "never-ingested-skill")

    assert degraded is False
    assert "no Skill node" in reason


def test_confirmed_blocked_precondition_is_a_real_finding_not_degraded():
    engine = _engine_with_backend(
        lambda *_a, **_k: [{"blocked": "child_reachable", "server": "servicenow-mcp"}]
    )

    reason, degraded = _skill_unrunnable_reason(
        engine, "servicenow-incident-management"
    )

    assert degraded is False
    assert "unmet precondition 'child_reachable'" in reason
    assert "servicenow-mcp" in reason


def test_resource_confirmation_read_failure_is_also_degraded():
    """The second read (CallableResource existence) can fail independently —
    that must be labeled degraded too, not folded into "no resource"."""
    calls = {"n": 0}

    def _execute(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"blocked": None, "server": ""}]
        raise RuntimeError("second read failed")

    engine = _engine_with_backend(_execute)

    reason, degraded = _skill_unrunnable_reason(
        engine, "servicenow-incident-management"
    )

    assert degraded is True
    assert "runnable resource could not be confirmed" in reason


# --------------------------------------------------------------------------- #
# run_agent: the exception TYPE must match what was actually established
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_agent_raises_lookuperror_for_a_confirmed_unrunnable_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda _e, _n: {"type": "unknown"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_skill_unrunnable_reason",
        lambda _engine, _name: ("unmet precondition 'skill_ingested'", False),
    )

    with pytest.raises(LookupError, match="is not runnable"):
        await agent_runner.run_agent(
            agent_name="never-ingested-skill",
            skill_name="never-ingested-skill",
            task="q",
            engine=object(),
        )


@pytest.mark.asyncio
async def test_run_agent_raises_runtimeerror_not_lookuperror_for_a_degraded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-SNV-5: a SessionExpiredError mid-delegation must never be reported as
    "the skill is not runnable" — that is a false negative about the skill.
    """
    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda _e, _n: {"type": "unknown"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_skill_unrunnable_reason",
        lambda _engine, _name: (
            "its blocking precondition could not be read "
            "(SessionExpiredError: Verified graph authority has expired)",
            True,
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await agent_runner.run_agent(
            agent_name="servicenow-incident-management",
            skill_name="servicenow-incident-management",
            task="q",
            engine=object(),
        )

    message = str(excinfo.value)
    assert "could not determine whether" in message
    assert "is not runnable" not in message
    assert not isinstance(excinfo.value, LookupError)
