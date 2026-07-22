#!/usr/bin/python
from __future__ import annotations

"""``LangfuseSignalProvider`` — SkillOpt's ``SkillSignalProvider`` protocol,
Langfuse-backed (CONCEPT:AU-AHE.optimization.skillopt-langfuse-signal).

Closes the loop the ``skill_evolution`` module's pluggable-signal seam was built
for (see ``skill_evolution.py``'s "Native Langfuse" coordination note): the
ReflACT cycle's Rollout/Evaluate/Reflect stages now have a real production-signal
implementation of :class:`~agent_utilities.knowledge_graph.research.skill_evolution.SkillSignalProvider`,
sourced entirely from the deployed Langfuse instance —

- Rollout (① ``train_tasks``/``holdout_tasks``) — a Langfuse DATASET, via
  :func:`~agent_utilities.harness.langfuse_signal.fetch_dataset_tasks`.
- Evaluate (⑥ ``score``) — Langfuse SCORES, via
  :func:`~agent_utilities.harness.langfuse_signal.fetch_reward`, blended with a
  cheap internal expected-output check through
  :func:`~agent_utilities.harness.langfuse_signal.blend_reward` (the SAME
  blend-not-replace contract ``ClaimFlywheel.record_outcome`` already uses).
- Reflect (② ``failure_traces``) — prior low-scoring Langfuse traces, via
  :func:`~agent_utilities.harness.langfuse_signal.fetch_low_score_traces`.

This module never talks to Langfuse's HTTP API directly and never re-implements
a client — every Langfuse read goes THROUGH the existing
:mod:`agent_utilities.harness.langfuse_signal` adapter (itself a thin wrapper over
:class:`~agent_utilities.harness.trace_backend.LangfuseTraceBackend`). Everything
here is best-effort and degrades: Langfuse being unconfigured, unreachable, or
returning nothing collapses to the SAME zero-infra
:class:`~agent_utilities.knowledge_graph.research.skill_evolution.InternalCorpusSignalProvider`
behavior a caller already gets today — :func:`select_skill_signal_provider` is the
config/param opt-in seam that makes this an additive, not a default-changing,
capability: the internal corpus is ALWAYS the default; Langfuse is used only when
BOTH a dataset name is configured (config field or explicit call argument) AND
Langfuse credentials are configured.
"""

import logging
from typing import Any

from agent_utilities.core.config import config
from agent_utilities.harness.langfuse_signal import (
    blend_reward,
    fetch_dataset_tasks,
    fetch_low_score_traces,
    fetch_reward,
)
from agent_utilities.knowledge_graph.research.skill_evolution import (
    InternalCorpusSignalProvider,
    SkillSignalProvider,
    SkillTask,
    SkillTrajectory,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LangfuseSignalProvider",
    "is_langfuse_configured",
    "select_skill_signal_provider",
]


def is_langfuse_configured() -> bool:
    """True when this deployment has Langfuse credentials configured.

    Same gate :mod:`~agent_utilities.harness.langfuse_signal` checks internally
    (``config.langfuse_secret_key_ref``) — exposed here so a caller can decide
    whether to even attempt to construct a :class:`LangfuseSignalProvider`.
    """
    return bool(config.langfuse_secret_key_ref)


def _task_from_dataset_row(row: dict[str, Any], *, index: int) -> SkillTask:
    """Normalize one ``fetch_dataset_tasks`` row into a :class:`SkillTask`.

    Rows are ``{"id", "input", "expected_output"}`` (the shape
    :meth:`~agent_utilities.harness.trace_backend.LangfuseTraceBackend.list_dataset_items`
    already normalizes to). ``expected_output`` travels through as task metadata
    so :meth:`LangfuseSignalProvider.score`'s internal half can use it without a
    second Langfuse round trip.

    Foundation simplification (mirrors ``skill_evolution.py``'s own §6-style
    disclosure): ``list_dataset_items`` does not currently pass through a
    linked source-trace id, so the SCORES lookup key defaults to the dataset
    item's own id (``row["trace_id"]``/``row["sourceTraceId"]`` win when present
    for forward compatibility) — correct when dataset items are curated by
    promoting a named trace into the benchmark (``add_to_dataset``'s own
    ``sourceTraceId`` convention), which is exactly how a Langfuse-curated
    regression dataset is normally built.
    """
    row_id = str(row.get("id") or "") or f"langfuse-item-{index}"
    raw_prompt = row.get("input")
    prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt or "")
    trace_id = (
        row.get("trace_id")
        or row.get("sourceTraceId")
        or row.get("source_trace_id")
        or row_id
    )
    metadata: dict[str, Any] = {
        "source": "langfuse",
        "langfuse_item_id": row_id,
        "trace_id": str(trace_id),
    }
    expected = row.get("expected_output")
    if expected is not None:
        metadata["expected_output"] = expected
    return SkillTask(id=row_id, prompt=prompt, metadata=metadata)


class LangfuseSignalProvider:
    """:class:`SkillSignalProvider` implementation backed by Langfuse.

    Structurally-typed against the protocol (``runtime_checkable``) — no explicit
    base class needed, but every method below fulfills it: ``train_tasks``,
    ``holdout_tasks``, ``score``, ``record_trajectory``, ``failure_traces``.
    """

    def __init__(
        self,
        *,
        train_dataset: str,
        holdout_dataset: str | None = None,
        score_name: str | None = None,
        weight: float = 0.5,
        limit: int = 50,
        engine: Any = None,
        fallback: SkillSignalProvider | None = None,
    ) -> None:
        """
        Args:
            train_dataset: Langfuse dataset name sourcing Rollout (①) tasks.
            holdout_dataset: Langfuse dataset name sourcing the Evaluate (⑥)
                held-out set; defaults to ``train_dataset`` when omitted (mirrors
                a single curated benchmark dataset used for both).
            score_name: Optional Langfuse score name filter passed to
                :func:`~agent_utilities.harness.langfuse_signal.fetch_reward`
                (``None`` averages every score on the trace).
            weight: Langfuse score's share in :func:`~agent_utilities.harness.
                langfuse_signal.blend_reward` (default an even split with the
                internal expected-output check).
            limit: Max rows/items pulled per Langfuse read.
            engine: Optional KG engine, forwarded to the internal fallback
                provider for trajectory persistence (same convention
                :class:`InternalCorpusSignalProvider` uses).
            fallback: The :class:`SkillSignalProvider` this provider degrades to
                when Langfuse returns nothing (unconfigured, unreachable, or an
                empty dataset/score). Defaults to a fresh, empty
                :class:`InternalCorpusSignalProvider` — degrading to ``[]``/a
                non-empty-output check, never a fabricated pass.
        """
        self.train_dataset = train_dataset
        self.holdout_dataset = holdout_dataset or train_dataset
        self.score_name = score_name
        self.weight = weight
        self.limit = limit
        self.engine = engine
        self._fallback: SkillSignalProvider = fallback or InternalCorpusSignalProvider(
            [], [], engine=engine
        )

    # ── ① Rollout ───────────────────────────────────────────────────────────
    def train_tasks(self) -> list[SkillTask]:
        rows = fetch_dataset_tasks(self.train_dataset, limit=self.limit)
        if not rows:
            return self._fallback.train_tasks()
        return [_task_from_dataset_row(r, index=i) for i, r in enumerate(rows)]

    def holdout_tasks(self) -> list[SkillTask]:
        rows = fetch_dataset_tasks(self.holdout_dataset, limit=self.limit)
        if not rows:
            return self._fallback.holdout_tasks()
        return [_task_from_dataset_row(r, index=i) for i, r in enumerate(rows)]

    # ── ⑥ Evaluate ──────────────────────────────────────────────────────────
    def score(self, task: SkillTask, output: str) -> tuple[bool, float, str]:
        """Blend a Langfuse SCORE with a cheap internal expected-output check.

        The internal half never fabricates a pass on an empty task: with no
        ``expected_output`` metadata it falls back to the same non-empty-output
        rule :class:`InternalCorpusSignalProvider` uses. The Langfuse half is
        looked up by the task's ``trace_id`` metadata (present when the dataset
        item is linked to a source production trace); absent a trace id or a
        matching score, :func:`~agent_utilities.harness.langfuse_signal.blend_reward`
        passes the internal reward through unchanged — the zero-overhead default.
        """
        expected = task.metadata.get("expected_output")
        if isinstance(expected, str) and expected.strip():
            internal_ok = expected.strip() in (output or "")
        else:
            internal_ok = bool(output and output.strip())
        internal_reward = 1.0 if internal_ok else 0.0

        trace_id = task.metadata.get("trace_id")
        langfuse_reward: float | None = None
        if trace_id:
            try:
                langfuse_reward = fetch_reward(
                    trace_id=str(trace_id), name=self.score_name
                )
            except Exception as e:  # noqa: BLE001 — reward fetch is best-effort
                logger.debug(
                    "LangfuseSignalProvider: score fetch failed for %s (%s).",
                    trace_id,
                    type(e).__name__,
                )
                langfuse_reward = None

        blended = blend_reward(internal_reward, langfuse_reward, weight=self.weight)
        success = blended >= 0.5
        if success:
            fail_reason = ""
        elif langfuse_reward is not None:
            fail_reason = "langfuse_low_score"
        else:
            fail_reason = "internal_check_failed"
        return success, blended, fail_reason

    # ── durable trajectory persistence — delegated, not duplicated ─────────
    def record_trajectory(
        self, skill_id: str, skill_version_id: str, trajectory: SkillTrajectory
    ) -> None:
        self._fallback.record_trajectory(skill_id, skill_version_id, trajectory)

    # ── ② Reflect ───────────────────────────────────────────────────────────
    def failure_traces(self, skill_id: str) -> list[dict[str, Any]]:
        """Prior low-scoring Langfuse traces as Reflect-stage failure material.

        ``[]`` (well, the fallback's — normally ``[]`` too) when Langfuse has
        nothing below threshold; never raises.
        """
        try:
            rows = fetch_low_score_traces(score_name=self.score_name, limit=self.limit)
        except Exception as e:  # noqa: BLE001 — trace fetch is best-effort
            logger.debug(
                "LangfuseSignalProvider: failure_traces fetch failed (%s).",
                type(e).__name__,
            )
            rows = []
        if not rows:
            return self._fallback.failure_traces(skill_id)
        return [{**row, "skill_id": skill_id, "source": "langfuse"} for row in rows]


def select_skill_signal_provider(
    skill_id: str,
    train_tasks: list[SkillTask],
    holdout_tasks: list[SkillTask],
    *,
    engine: Any = None,
    train_dataset: str | None = None,
    holdout_dataset: str | None = None,
    score_name: str | None = None,
    weight: float | None = None,
    limit: int = 50,
) -> SkillSignalProvider:
    """The ReflACT cycle's signal-provider selector — config/param opt-in.

    Returns a :class:`LangfuseSignalProvider` ONLY when a Langfuse train dataset
    name is available (explicit ``train_dataset`` argument, else
    ``config.kg_skill_evolution_langfuse_train_dataset``) AND Langfuse
    credentials are configured (:func:`is_langfuse_configured`); otherwise
    returns a plain :class:`~agent_utilities.knowledge_graph.research.
    skill_evolution.InternalCorpusSignalProvider` seeded with ``train_tasks``/
    ``holdout_tasks`` — the SAME shipped default every existing caller already
    gets. This is the single seam a real ``skill_eval_targets_provider``
    (``LoopController(engine, skill_eval_targets_provider=...)``) calls to build
    each target's ``signal``, so enabling Langfuse for a skill is a config/call
    change, never a control-flow change to :func:`~agent_utilities.
    knowledge_graph.research.skill_evolution.run_reflact_cycle`.

    When Langfuse IS selected, the internal task lists are still passed through
    as the :class:`LangfuseSignalProvider`'s fallback, so an empty/unreachable
    Langfuse dataset degrades to those tasks rather than an empty rollout.
    """
    resolved_dataset = train_dataset or config.kg_skill_evolution_langfuse_train_dataset
    if resolved_dataset and is_langfuse_configured():
        fallback = InternalCorpusSignalProvider(
            train_tasks, holdout_tasks, engine=engine
        )
        resolved_holdout = (
            holdout_dataset
            or config.kg_skill_evolution_langfuse_holdout_dataset
            or resolved_dataset
        )
        resolved_score_name = (
            score_name or config.kg_skill_evolution_langfuse_score_name
        )
        resolved_weight = (
            weight if weight is not None else config.kg_skill_evolution_langfuse_weight
        )
        logger.info(
            "select_skill_signal_provider(%s): Langfuse signal selected "
            "(train_dataset=%s, holdout_dataset=%s).",
            skill_id,
            resolved_dataset,
            resolved_holdout,
        )
        return LangfuseSignalProvider(
            train_dataset=resolved_dataset,
            holdout_dataset=resolved_holdout,
            score_name=resolved_score_name,
            weight=resolved_weight,
            limit=limit,
            engine=engine,
            fallback=fallback,
        )
    return InternalCorpusSignalProvider(train_tasks, holdout_tasks, engine=engine)
