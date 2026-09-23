"""AU's consumers of EG's Decide layer (DECIDE-LAYER-DESIGN §10, ledger EH-029..EH-048).

Every decision point that used to be ad-hoc code or an LLM call asks EG
first and keeps its old rule as the deterministic fallback:

    from agent_utilities import decide
    choice = decide.choose("au.retrieval.plan", options, fallback=lambda: mode)

With no runner installed (no engine connected), :func:`choose` is exactly the
fallback, reason ``no_runner``. The process installs one runner once it has a
verified engine session (:func:`install_runner`); tests install fakes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

from agent_connector_sdk import decide as sdk_decide

from agent_utilities.decide.options import Option, iri_list_param, q32, text_param
from agent_utilities.decide.outcome import Choice
from agent_utilities.decide.points import (
    POINTS,
    Binding,
    DecisionPoint,
    LogMode,
    StaticBindings,
)
from agent_utilities.decide.runner import DecisionRunner, Escalated, Fallback

_RUNNER: ContextVar[DecisionRunner | None] = ContextVar(
    "au_decide_runner", default=None
)
_PROCESS: list[DecisionRunner | None] = [None]


def install_runner(runner: DecisionRunner | None) -> None:
    """Install ``runner`` for the whole process (``None`` uninstalls).

    The connector SDK's own decision points (EH-042/043) ask through the SDK's
    runner slot; AU installs the SAME runner there, so there is one runner and
    one answer per question whichever side asks.
    """
    _PROCESS[0] = runner
    sdk_decide.install_runner(runner)


def use_runner(runner: DecisionRunner | None) -> Any:
    """Scope ``runner`` to the current context (AU and SDK); returns the reset token."""
    return (_RUNNER.set(runner), sdk_decide.use_runner(runner))


def reset_runner(token: Any) -> None:
    """Undo :func:`use_runner`."""
    for part in token:
        part.var.reset(part)


def current_runner() -> DecisionRunner | None:
    """The context's runner, else the process runner, else ``None``."""
    return _RUNNER.get() or _PROCESS[0]


def _no_runner(fallback: Fallback) -> Choice:
    answer = fallback()
    option = answer.option_id if isinstance(answer, Escalated) else answer
    return Choice(option, False, "no_runner")


def choose(
    question_id: str,
    options: Sequence[Option],
    fallback: Fallback,
    *,
    params: Iterable[Mapping[str, Any]] = (),
    candidates: Mapping[str, Any] | None = None,
) -> Choice:
    """Decide ``question_id`` from a sync call site."""
    runner = current_runner()
    if runner is None:
        return _no_runner(fallback)
    return runner.choose(
        question_id, options, fallback, params=params, candidates=candidates
    )


async def achoose(
    question_id: str,
    options: Sequence[Option],
    fallback: Fallback,
    *,
    params: Iterable[Mapping[str, Any]] = (),
    candidates: Mapping[str, Any] | None = None,
) -> Choice:
    """Decide ``question_id`` from an async call site."""
    runner = current_runner()
    if runner is None:
        return _no_runner(fallback)
    return await runner.achoose(
        question_id, options, fallback, params=params, candidates=candidates
    )


__all__ = [
    "POINTS",
    "Binding",
    "Choice",
    "DecisionPoint",
    "DecisionRunner",
    "Escalated",
    "LogMode",
    "Option",
    "StaticBindings",
    "achoose",
    "choose",
    "current_runner",
    "install_runner",
    "iri_list_param",
    "q32",
    "reset_runner",
    "text_param",
    "use_runner",
]
