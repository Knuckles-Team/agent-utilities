"""Continuation and stopping of a running swarm over a committed topology plan.

Running the committed ``StopRule`` is EXECUTION, not decision:
:func:`stop_reason` evaluates it over the run's L5 observations
(:class:`RoundProgress`), and budget or deadline exhaustion always stops.

Between rounds the runtime may ask the continuation question
(``au.swarm.continue``, evaluate-only and sampled). Its options only NARROW
-- ``continue``, ``narrow`` (one fewer agent per slot next round) or
``stop``; there is no widen option, because an extra round, a wider slot or
a deeper level is a NEW plan decision and a new lease (AU-CONTROL-R019). The
question cites the parent plan's record as a request parameter, so the
continuation needs no new record field. Its fallback is ``continue`` while
the stop rule holds, so an unbound point changes nothing.

``stop`` releases the plan's leases through the installed release port
(graph-os's ``ReleaseCapacity`` book): the ledger's leases are per cell, so
narrowing keeps them until the stop -- runtime never grants itself capacity.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

logger = logging.getLogger(__name__)

QUESTION = "au.swarm.continue"


@dataclass(frozen=True, slots=True)
class RoundProgress:
    """What L5 observed after a round of a running plan."""

    rounds_done: int
    verifier_passed: bool = False
    votes: int = 0
    tokens_spent: int | None = None
    token_ceiling: int | None = None
    elapsed_ms: int = 0
    deadline_ms: int | None = None


def _max_rounds(rule: Mapping[str, Any], p: RoundProgress) -> bool:
    return p.rounds_done >= int(rule.get("n") or rule.get("max_rounds") or 1)


def _quorum(rule: Mapping[str, Any], p: RoundProgress) -> bool:
    return p.votes >= int(rule.get("k") or 1)


def _verifier_pass(rule: Mapping[str, Any], p: RoundProgress) -> bool:
    return p.verifier_passed or _max_rounds(rule, p)


def _never(rule: Mapping[str, Any], p: RoundProgress) -> bool:
    """Budget/deadline exhaustion is checked in :func:`stop_reason` directly,
    never by the rule's own tag."""
    return False


#: stop rule tag -> "its own condition is met".
_RULE_MET: dict[str, Callable[[Mapping[str, Any], RoundProgress], bool]] = {
    "max_rounds": _max_rounds,
    "quorum": _quorum,
    "verifier_pass": _verifier_pass,
    "budget": _never,
    "deadline": _never,
}


def stop_reason(rule: Mapping[str, Any], progress: RoundProgress) -> str | None:
    """Why the committed stop rule ends the run now, or ``None``.

    Budget or deadline exhaustion stops whatever the rule says.
    """
    if (
        progress.token_ceiling is not None
        and (progress.tokens_spent or 0) >= progress.token_ceiling
    ):
        return "budget"
    if progress.deadline_ms is not None and progress.elapsed_ms >= progress.deadline_ms:
        return "deadline"
    tag = str(rule.get("rule") or "")
    met = _RULE_MET.get(tag)
    if met is None:
        return f"unknown_rule:{tag}"
    return tag if met(rule, progress) else None


def continuation_options(width: int) -> list[Option]:
    """The narrow-only option set: never an option that widens."""
    options = [
        Option("continue", {"width": float(width)}),
        Option("stop", {"width": 0.0}),
    ]
    if width > 1:
        options.insert(1, Option("narrow", {"width": float(width - 1)}))
    return options


@dataclass(frozen=True, slots=True)
class Continuation:
    """The runtime's next step for a running plan."""

    action: str
    reason: str
    record_id: str | None = None


Release = Callable[[str], Awaitable[Any]]
_RELEASE: list[Release | None] = [None]


def install_release(release: Release | None) -> None:
    """Install the plan-lease release port (graph-os's ``ReleaseCapacity`` book)."""
    _RELEASE[0] = release


async def _released(parent_record: str) -> None:
    release = _RELEASE[0]
    if release is None:
        return
    try:
        await release(parent_record)
    except (RuntimeError, ConnectionError, TimeoutError, ValueError) as exc:
        logger.warning(
            "plan %s leases left to expire (%s)", parent_record, type(exc).__name__
        )


async def continue_or_stop(
    parent_record: str,
    stop: Mapping[str, Any],
    progress: RoundProgress,
    *,
    width: int,
) -> Continuation:
    """After a round: stop when the rule or a budget says so, else ask EG
    (evaluate-only) whether to continue, narrow or stop; release on stop."""
    reason = stop_reason(stop, progress)
    if reason is not None:
        await _released(parent_record)
        return Continuation("stop", reason)
    choice = await decide.achoose(
        QUESTION,
        continuation_options(width),
        lambda: "continue",
        params=[text_param("parent_record", parent_record)],
    )
    action = choice.option_id or "continue"
    if action == "stop":
        await _released(parent_record)
    return Continuation(action, choice.reason, choice.record_id)


#: The manifest metadata key a plan-driven execution carries its plan under:
#: ``{"record_id": ..., "stop": <StopRule>, "token_ceiling": .., "deadline_ms": ..}``.
PLAN_METADATA_KEY = "topology_plan"


async def after_wave(
    metadata: Mapping[str, Any], wave_index: int, waves: list[list[Any]]
) -> bool:
    """The parallel engine's between-round hook; ``False`` stops the run.

    Only a manifest executing a committed plan is governed; ``narrow`` drops
    one agent from every remaining multi-agent wave (runtime only narrows).
    """
    plan = metadata.get(PLAN_METADATA_KEY)
    if not isinstance(plan, Mapping):
        return True
    if wave_index + 1 >= len(waves):
        await _released(str(plan.get("record_id") or ""))
        return True
    progress = RoundProgress(
        rounds_done=wave_index + 1,
        token_ceiling=plan.get("token_ceiling"),
        deadline_ms=plan.get("deadline_ms"),
    )
    width = max(len(wave) for wave in waves[wave_index + 1 :])
    step = await continue_or_stop(
        str(plan.get("record_id") or ""),
        plan.get("stop") or {},
        progress,
        width=width,
    )
    if step.action == "narrow":
        for wave in waves[wave_index + 1 :]:
            if len(wave) > 1:
                wave.pop()
    return step.action != "stop"


__all__ = [
    "PLAN_METADATA_KEY",
    "QUESTION",
    "Continuation",
    "RoundProgress",
    "after_wave",
    "continuation_options",
    "continue_or_stop",
    "install_release",
    "stop_reason",
]
