#!/usr/bin/python
"""Lightweight, unified ingestion profiling (CONCEPT:AU-OS.observability.ingestion-profile-report/70/71).

The task profiler (``profile_report``, OS-5.55) times every queued task end-to-end
per lane, but three things were invisible:

* **token/cost** — the LLM/embed spend per ingest was never captured, so
  ``profile_report`` reported ``tokens=0`` (CONCEPT:AU-OS.observability.ingestion-profile-report);
* **per-stage breakdown** — a paper's ~5s ingest was opaque (read vs LLM-extract vs
  embed vs graph-write) (CONCEPT:AU-OS.observability.ingest-stage-breakdown);
* **off-queue work** — the embed-backfill, concept-registry embedding and the
  assimilation passes run OFF the task queue, so they never appeared (CONCEPT:AU-OS.observability.embed-stage-profile).

This module closes all three with ONE primitive: a **contextvar-scoped**
``IngestProfile``. An ingest activates one for its duration; the shared
LLM (``make_llm_fn``) and embed (``make_embed_fn``) wrappers find it on the
contextvar and record their token usage into it automatically — no parameter
threading. Ingest code times stages into it; off-queue passes activate one and emit
its record so the unified report covers every path.

Tokens are the real signal; cost is a derived, config-discipline default (local
vLLM is effectively free, but the per-1k rates keep the number cross-provider
comparable without an env knob).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

#: the active profile for the current async/thread context (None when no ingest is
#: being profiled — the wrappers then no-op at ~zero cost).
_ACTIVE: ContextVar[IngestProfile | None] = ContextVar("ingest_profile", default=None)

#: derived-cost rates per 1k tokens (config discipline — one value, no env knob).
#: Local vLLM is effectively free; kept non-zero-capable for cross-provider compare.
_LLM_COST_PER_1K = 0.0
_EMBED_COST_PER_1K = 0.0

#: when a token count is unavailable (e.g. an embedding endpoint omits usage), this
#: chars-per-token ratio gives a stable estimate so the signal is never simply lost.
_CHARS_PER_TOKEN = 4


@dataclass
class IngestProfile:
    """Accumulates stage timings + LLM/embed token usage for one ingest unit."""

    label: str = ""
    stages: dict[str, float] = field(default_factory=dict)  # stage name -> ms
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    llm_calls: int = 0
    embed_calls: int = 0

    @staticmethod
    def active() -> IngestProfile | None:
        """The profile bound to the current context, or ``None``."""
        return _ACTIVE.get()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage; re-entrant for the same name (durations sum)."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (
                self.stages.get(name, 0.0) + (time.perf_counter() - t) * 1000.0
            )

    def record_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.completion_tokens += max(0, int(completion_tokens or 0))
        self.llm_calls += 1

    def record_embed(self, n_tokens: int) -> None:
        self.embed_tokens += max(0, int(n_tokens or 0))
        self.embed_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embed_tokens

    @property
    def cost(self) -> float:
        chat = (self.prompt_tokens + self.completion_tokens) / 1000.0 * _LLM_COST_PER_1K
        embed = self.embed_tokens / 1000.0 * _EMBED_COST_PER_1K
        return round(chat + embed, 6)

    def to_dict(self) -> dict[str, Any]:
        """Compact, JSON-safe record for task metadata / the profiler."""
        return {
            "stages_ms": {k: round(v, 1) for k, v in self.stages.items()},
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embed_tokens": self.embed_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "embed_calls": self.embed_calls,
            "cost": self.cost,
        }


@contextmanager
def profile_ingest(label: str = "") -> Iterator[IngestProfile]:
    """Activate a profile for the block. Nested activations REUSE the outer profile
    so one ingest accumulates across its helper calls + the llm/embed wrappers
    (re-entrant, never double-counts by stacking)."""
    existing = _ACTIVE.get()
    if existing is not None:
        yield existing
        return
    prof = IngestProfile(label=label)
    token = _ACTIVE.set(prof)
    try:
        yield prof
    finally:
        _ACTIVE.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a stage against the active ingest profile; a no-op when none is active
    (so call sites stay clean: ``with ingest_profile.stage("read"): ...``)."""
    p = _ACTIVE.get()
    if p is None:
        yield
        return
    with p.stage(name):
        yield


def record_llm_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """Called by ``make_llm_fn``; no-op (cheap) when no ingest profile is active."""
    p = _ACTIVE.get()
    if p is not None:
        p.record_llm(prompt_tokens, completion_tokens)


def record_embed_usage(n_tokens: int = 0, *, texts: list[str] | None = None) -> None:
    """Called by ``make_embed_fn``. Estimates tokens from text length when the
    embedding endpoint doesn't return a usage count, so the signal is never lost."""
    p = _ACTIVE.get()
    if p is None:
        return
    if not n_tokens and texts:
        n_tokens = sum(len(t) for t in texts) // _CHARS_PER_TOKEN
    p.record_embed(n_tokens)


def estimate_tokens(text: str) -> int:
    """Stable chars→tokens estimate for paths with no usage metadata."""
    return max(0, len(text) // _CHARS_PER_TOKEN)


def record_offqueue_span(engine: Any, kind: str, profile: IngestProfile) -> None:
    """Persist an OFF-QUEUE ingest profile as a ``:ProfileSpan`` so ``profile_report``
    covers paths that never become ``:Task`` nodes — the embed-backfill, the
    concept-registry embedding, the assimilation passes (CONCEPT:AU-OS.observability.embed-stage-profile).

    Written to the control graph alongside ``:Task`` with a task-shaped metadata
    envelope (``type='offqueue:<kind>'`` + ``profile`` + ``duration_ms``) so the
    report aggregates it through the SAME path. Best-effort: it logs the profile
    regardless, and never raises into the pass it is measuring.
    """
    import logging

    logging.getLogger(__name__).info(
        "offqueue profile [%s]: %s", kind, profile.to_dict()
    )
    try:
        import json as _json
        import time as _time
        from datetime import UTC, datetime

        from ..backends.epistemic_graph_backend import EpistemicGraphBackend

        now = datetime.now(UTC).isoformat()
        envelope = {
            "type": f"offqueue:{kind}",
            "kind": kind,
            "profile": profile.to_dict(),
            "duration_ms": round(sum(profile.stages.values()), 1),
            "started_at": now,
            "completed_at": now,
        }
        span_id = f"profilespan:{kind}:{int(_time.time() * 1000)}"
        EpistemicGraphBackend(graph_name="__control__").add_node(
            span_id, type="ProfileSpan", metadata=_json.dumps(envelope)
        )
    except Exception:  # noqa: BLE001 — persistence is best-effort; the log remains
        pass
