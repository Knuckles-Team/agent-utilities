"""The delegated run a retrieval serves (EH-394 topology -> skill, EH-395).

``run_agent`` -- the one entrypoint every ``graph_orchestrate`` delegation
reaches -- executes inside :func:`run_scoped`:

* the run's pinned ``skill_name`` is the current skill, so a retrieval the run
  makes records WHICH skill it served (a proven path's ``skill_ref``, which
  ``graph.assemble()`` pins for the task class);
* every retrieval the run plans is noted, and when the run's answer is in,
  each one's outcome is attested -- what it returned, and which returned
  units the answer names (``HybridRetriever.record_answer_usage``). Only an
  independent evaluator's verdict on the committed plan record turns that into
  labels (EG enforces it).

Light at import: the orchestration runner imports it at module load.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class RunScope:
    """One delegated run: its skill and the retrievals it planned."""

    skill_ref: str | None = None
    retrievals: list[tuple[Any, str]] = field(default_factory=list)


_RUN: contextvars.ContextVar[RunScope | None] = contextvars.ContextVar(
    "retrieval_run_scope", default=None
)


def skill_ref_of(skill_name: str) -> str:
    """The skill's stable ``skill://<slug>`` reference (its ingested identity)."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        skill_reference,
    )

    return skill_reference(skill_name)


def current_skill_ref() -> str | None:
    """The skill the current run executes under, if it pinned one."""
    scope = _RUN.get()
    return None if scope is None else scope.skill_ref


def note_retrieval(retriever: Any, query: str) -> None:
    """Remember that the current run planned a retrieval of ``query``."""
    scope = _RUN.get()
    if scope is not None:
        scope.retrievals.append((retriever, query))


def _attest(scope: RunScope, answer: Any) -> None:
    from agent_utilities.decide.learning.runs import attest_answer

    try:
        attest_answer(scope.retrievals, answer)
    except Exception as exc:
        logger.warning("retrieval outcomes of the run not attested: %s", exc)


def run_scoped(
    fn: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Run ``fn`` as one delegated run: its ``skill_name`` keyword is the
    current skill, and its answer attests the retrievals it planned."""

    @functools.wraps(fn)
    async def scoped(*args: Any, **kwargs: Any) -> T:
        name = kwargs.get("skill_name")
        scope = RunScope(skill_ref_of(str(name)) if name else None)
        token = _RUN.set(scope)
        try:
            answer = await fn(*args, **kwargs)
            if scope.retrievals:
                await asyncio.to_thread(_attest, scope, answer)
            return answer
        finally:
            _RUN.reset(token)

    return scoped


__all__ = [
    "RunScope",
    "current_skill_ref",
    "note_retrieval",
    "run_scoped",
    "skill_ref_of",
]
