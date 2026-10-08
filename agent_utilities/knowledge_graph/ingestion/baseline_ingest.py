"""Baseline ingest after daemon boot (spec: baseline-ingestion).

A fresh store holds no skills, prompts or code, so retrieval refuses every
chat for a sparse index. The daemon role closes that gap without a manual
step: once the graph is writable, one background thread enqueues the baseline
as durable WorkItems and returns. The existing task workers drain them.

Baseline content is the grounding corpus only: the prompt library and the
configured skill providers (:mod:`.baseline_items`) plus the in-scope
workspace code (:mod:`.baseline_workspace`). External connector data is out
of scope. ``source_sync source=all`` copies external systems into the graph;
the baseline never calls it.

Properties:

* **Non-blocking.** Planning and submission run on one daemon thread. A
  failure logs and ends the thread; serving never waits on it or sees it.
* **Idempotent.** Every leg is content-hash delta, so a repeat boot costs a
  hash check per file. ``submit_task`` deduplicates a target that still has a
  live WorkItem from an earlier boot.
* **Ordered and bounded.** Items enqueue in EG queue-class order (fast before
  medium; see :mod:`..core.semantic_tiers`) at the class priority bucket. The
  lane admission policy bounds concurrency. Vector work stays on the dedicated
  embedding backfill thread.
* **Observable.** Each WorkItem carries ``baseline`` metadata (leg, name,
  queue class, entry stage), so the existing ``jobs`` and ``job_status`` task
  surface reports progress.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from ..core.semantic_tiers import (
    entry_stage_for_task_type,
    priority_for_queue_class,
    queue_class_rank,
)
from .baseline_items import BaselineItem, csv_names, prompt_item, skill_items
from .baseline_workspace import codebase_items

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def _settings() -> Any:
    from agent_utilities.core.config import config

    return config


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-") or "item"


def plan_baseline() -> list[BaselineItem]:
    """Return the configured baseline in queue-class drain order."""
    settings = _settings()
    items = [
        prompt_item(),
        *skill_items(csv_names(settings.kg_baseline_skill_providers)),
    ]
    scope = str(settings.kg_baseline_codebases).strip().lower()
    try:
        items.extend(codebase_items(scope, int(settings.kg_baseline_max_codebases)))
    except Exception as exc:  # noqa: BLE001 - code is optional; skills/prompts still run
        logger.warning(
            "baseline ingest: workspace scan failed (%s)", type(exc).__name__
        )
    return sorted(items, key=lambda item: queue_class_rank(item.queue_class))


def _submit(engine: Any, item: BaselineItem, boot_id: str) -> str:
    queue_class = item.queue_class
    metadata = dict(item.extra_meta)
    metadata["baseline"] = {
        "leg": item.leg,
        "name": item.name,
        "queue_class": queue_class,
        "stage": entry_stage_for_task_type(item.task_type),
    }
    return engine.submit_task(
        target_path=item.target,
        is_codebase=item.is_codebase,
        provenance={"source": "baseline_ingest", "boot": boot_id},
        task_type=item.task_type,
        priority=priority_for_queue_class(queue_class),
        job_id=f"baseline-{boot_id}-{item.leg}-{_slug(item.name)}",
        extra_meta=metadata,
    )


def enqueue_baseline(
    engine: Any, items: list[BaselineItem], boot_id: str
) -> dict[str, Any]:
    """Submit each item and report queued and rejected entries.

    One rejected item never blocks the rest.
    """
    queued: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for item in items:
        entry = {"leg": item.leg, "name": item.name, "queue_class": item.queue_class}
        try:
            entry["job_id"] = _submit(engine, item, boot_id)
        except Exception as exc:  # noqa: BLE001 - per-item rejection is reported
            rejected.append({**entry, "reason": type(exc).__name__})
            continue
        queued.append(entry)
    return {"boot_id": boot_id, "queued": queued, "rejected": rejected}


def run_baseline_ingest(engine: Any) -> dict[str, Any] | None:
    """Plan and enqueue the baseline once. Never raises."""
    try:
        report = enqueue_baseline(engine, plan_baseline(), uuid.uuid4().hex[:12])
    except Exception as exc:  # noqa: BLE001 - the thread must end quietly
        logger.error("baseline ingest failed before enqueue (%s)", type(exc).__name__)
        return None
    logger.info(
        "baseline ingest enqueued %d WorkItem(s), rejected %d (boot=%s)",
        len(report["queued"]),
        len(report["rejected"]),
        report["boot_id"],
    )
    return report


def _baseline_wanted(engine: Any) -> bool:
    if not bool(_settings().kg_baseline_ingest):
        return False
    return getattr(engine, "_baseline_ingest_thread", None) is None


def _baseline_thread(engine: Any, session: Any) -> Any:
    from ..core.engine_tasks import _authorized_background_thread

    thread = _authorized_background_thread(
        session, run_baseline_ingest, name="KG-Baseline-Ingest", args=(engine,)
    )
    engine._baseline_ingest_thread = thread
    return thread


def start_baseline_ingest(engine: Any, session: Any) -> bool:
    """Start the baseline thread once per engine; return whether it started.

    Disabled by ``KG_BASELINE_INGEST=0``. A launch failure logs and returns
    ``False``; it never propagates into daemon startup.
    """
    if not _baseline_wanted(engine):
        return False
    try:
        _baseline_thread(engine, session).start()
    except Exception as exc:  # noqa: BLE001 - never block or crash daemon start
        logger.error("baseline ingest launch failed (%s)", type(exc).__name__)
        return False
    return True


__all__ = [
    "enqueue_baseline",
    "plan_baseline",
    "run_baseline_ingest",
    "start_baseline_ingest",
]
