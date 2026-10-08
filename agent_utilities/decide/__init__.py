"""AU's consumers of EG's Decide layer.

Every decision point that used to be ad-hoc code or an LLM call asks EG
first and keeps its old rule as the deterministic fallback:

    from agent_utilities import decide
    choice = decide.choose("au.retrieval.plan", options, fallback=lambda: mode)

With no runner installed (no engine connected), :func:`choose` is exactly the
fallback, reason ``no_runner``. The process installs one runner once it has a
verified engine session (:func:`install_runner`); tests install fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def install_runner(runner: DecisionRunner | None) -> None:
    """Install ``runner`` for the whole process (``None`` uninstalls).

    There is ONE runner slot -- the connector SDK's -- so AU's and the SDK's
    own decision points ask through the same runner.
    """
    sdk_decide.install_runner(runner)


def use_runner(runner: DecisionRunner | None) -> Any:
    """Scope ``runner`` to the current context; returns the reset token."""
    return sdk_decide.use_runner(runner)


def reset_runner(token: Any) -> None:
    """Undo :func:`use_runner`."""
    token.var.reset(token)


def current_runner() -> DecisionRunner | None:
    """The installed AU runner, if any (the SDK slot holds it)."""
    runner = sdk_decide.current_runner()
    return runner if isinstance(runner, DecisionRunner) else None


def _au(choice: Any) -> Choice:
    """An SDK choice (no runner installed) as AU's, which adds the record fields."""
    if isinstance(choice, Choice):
        return choice
    return Choice(
        choice.option_id, choice.decided, choice.reason, advisory=choice.advisory
    )


def choose(
    question_id: str, options: Sequence[Option], fallback: Fallback, **context: Any
) -> Choice:
    """Decide from a sync call site (``context``: ``params`` / ``candidates``)."""
    return _au(sdk_decide.choose(question_id, options, fallback, **context))


async def achoose(
    question_id: str, options: Sequence[Option], fallback: Fallback, **context: Any
) -> Choice:
    """Decide from an async call site (``context``: ``params`` / ``candidates``)."""
    return _au(await sdk_decide.achoose(question_id, options, fallback, **context))


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
