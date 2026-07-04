"""Task contract + result model for the RLM long-context benchmark (CONCEPT:AU-AHE.rlm.long-context-benchmark)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

GraderKind = Literal["substring", "numeric", "choice"]
TaskMode = Literal["real", "synthetic"]


class TaskCase(BaseModel):
    """One concrete long-context problem instance produced by a :class:`LongContextTask`.

    ``context`` is the (large) external document, ``question`` the query a system must answer
    against it, and ``answer`` the reference. ``grade`` scores a prediction in ``[0, 1]`` using
    the task's ``grader_kind`` so grading is deterministic and unit-testable without an LLM.
    """

    context: str
    question: str
    answer: str
    grader_kind: GraderKind = "substring"
    mode: TaskMode = "synthetic"
    meta: dict = Field(default_factory=dict)

    def grade(self, prediction: str) -> float:
        pred = (prediction or "").strip()
        ref = self.answer.strip()
        if self.grader_kind == "numeric":
            want = _first_number(ref)
            got = _first_number(pred)
            if want is None or got is None:
                return 0.0
            return 1.0 if abs(want - got) < 1e-9 else 0.0
        if self.grader_kind == "choice":
            # A multiple-choice answer is correct if the chosen letter/label appears as a token.
            return 1.0 if _choice_hit(ref, pred) else 0.0
        # substring (default): reference phrase must appear in the prediction.
        return 1.0 if ref.lower() in pred.lower() else 0.0


def _first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def _choice_hit(ref: str, pred: str) -> bool:
    ref_l = ref.strip().lower()
    # Match a standalone occurrence of the option label (e.g. "B" or "B)") in the prediction.
    return bool(re.search(rf"\b{re.escape(ref_l)}\b", pred.lower()))


class BenchResult(BaseModel):
    """Aggregated outcome of running one system over one task at one scale (CONCEPT:AU-AHE.rlm.long-context-benchmark)."""

    task: str
    complexity: str
    system: str
    scale: int
    accuracy: float
    n: int
    cost_usd: float = 0.0
    total_tokens: int = 0
    wall_s: float = 0.0
    max_depth: int = 0
    mode: TaskMode = "synthetic"
    notes: str = ""


class LongContextTask(ABC):
    """A long-context benchmark task: builds gradeable cases at a requested ``scale``.

    ``scale`` is the approximate context size in characters, letting one task span from a few
    thousand chars up to the 10M-token stress regime by turning a single knob.
    """

    name: str = "task"
    complexity: str = "O(?)"
    real_dataset: bool = False

    @abstractmethod
    def build(self, scale: int, *, seed: int = 0) -> TaskCase:
        """Return one :class:`TaskCase` whose context is ~``scale`` characters."""
        raise NotImplementedError  # ABSTRACT-OK


# ── task registry (populated by tasks/ at import) ──
_REGISTRY: dict[str, LongContextTask] = {}


def register_task(task: LongContextTask) -> LongContextTask:
    _REGISTRY[task.name] = task
    return task


def get_task(name: str) -> LongContextTask:
    from . import tasks as _tasks  # noqa: F401  ensure registrations ran

    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown RLM benchmark task {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    from . import tasks as _tasks  # noqa: F401  ensure registrations ran

    return sorted(_REGISTRY)
