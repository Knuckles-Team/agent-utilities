"""Baseline ingest after daemon boot (spec: baseline-ingestion).

A fresh store holds no skills, prompts or code, so retrieval refuses every
chat for a sparse index. The daemon role closes that gap without a manual
step: once the graph is writable, one background thread enqueues the baseline
as durable WorkItems and returns. The existing task workers drain them.

Baseline content is the grounding corpus only:

* **prompts** - the prompt library (``ingest_prompts_to_graph``), as one
  ``scheduled_job`` WorkItem that runs the ``baseline_prompts`` maintenance tick.
* **skills** - one ``skill_workflows`` WorkItem per configured skill provider
  (default ``agent-utilities``, ``graph-os`` and ``universal-skills``). The
  target is a ``skill-provider:<name>`` reference, resolved to the provider's
  verified root when the worker runs.
* **codebase** - one ``codebase`` WorkItem per workspace repository in the
  configured scope, capped by ``KG_BASELINE_MAX_CODEBASES``.

External connector data is out of scope. ``source_sync source=all`` copies
external systems into the graph; the baseline never calls it.

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

import asyncio
import contextvars
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.semantic_tiers import (
    entry_stage_for_task_type,
    priority_for_queue_class,
    queue_class_for_task_type,
    queue_class_rank,
)

logger = logging.getLogger(__name__)

#: Target prefix that names a skill provider instead of a filesystem path.
SKILL_PROVIDER_TARGET_PREFIX = "skill-provider:"
#: The installed universal-skills package sentinel ``skill_workflows`` accepts.
_UNIVERSAL_SKILLS_SENTINEL = "universal-skills"
#: Maintenance tick that ingests the prompt library.
PROMPTS_MAINTENANCE_REF = "baseline_prompts"

_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class BaselineItem:
    """One durable WorkItem the baseline enqueues."""

    leg: str
    name: str
    target: str
    task_type: str
    is_codebase: bool = False
    extra_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_class(self) -> str:
        return queue_class_for_task_type(self.task_type)


def _settings() -> Any:
    from agent_utilities.core.config import config

    return config


def _csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-") or "item"


# ── skills ──────────────────────────────────────────────────────────────


def resolve_skill_corpus_root(target: str) -> str | None:
    """Resolve a ``skill_workflows`` target to the corpus root it names.

    ``universal-skills`` keeps its meaning (the installed package default,
    returned as ``None``). ``skill-provider:<name>`` resolves through the
    verified provider registry at run time, so the durable target never holds
    a machine path. Any other value is an explicit root and passes through.

    Raises:
        LookupError: the named provider is not installed in this process.
    """
    if target == _UNIVERSAL_SKILLS_SENTINEL:
        return None
    if not target.startswith(SKILL_PROVIDER_TARGET_PREFIX):
        return target
    name = target.removeprefix(SKILL_PROVIDER_TARGET_PREFIX)
    from agent_utilities.core.providers import SKILL_PROVIDER_GROUP, iter_provider_dirs

    for provider, root in iter_provider_dirs(SKILL_PROVIDER_GROUP):
        if provider == name:
            return str(root)
    raise LookupError(f"skill provider {name!r} is not installed")


def _skill_items(providers: list[str]) -> list[BaselineItem]:
    return [
        BaselineItem(
            leg="skills",
            name=name,
            target=f"{SKILL_PROVIDER_TARGET_PREFIX}{name}",
            task_type="skill_workflows",
        )
        for name in providers
    ]


# ── prompts ─────────────────────────────────────────────────────────────


def _prompt_item() -> BaselineItem:
    return BaselineItem(
        leg="prompts",
        name="prompt-library",
        target="baseline:prompts",
        task_type="scheduled_job",
        extra_meta={"payload": {"kind": "maint", "ref": PROMPTS_MAINTENANCE_REF}},
    )


def ingest_prompt_library() -> None:
    """Ingest the prompt library from synchronous worker code.

    The worker calls maintenance ticks from inside its running event loop, so
    the coroutine runs on a fresh loop in a helper thread. The copied context
    carries the worker's verified session and actor into that thread.
    """
    from agent_utilities.agent.registry_builder import ingest_prompts_to_graph

    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kg-baseline") as pool:
        pool.submit(context.run, asyncio.run, ingest_prompts_to_graph()).result()


# ── codebases ───────────────────────────────────────────────────────────


def _load_workspace_manifest() -> dict[str, Any] | None:
    import yaml

    from agent_utilities.core.workspace import get_agent_workspace
    from agent_utilities.core.workspace_config import get_workspace_yml_path

    for path in (get_agent_workspace() / "workspace.yml", get_workspace_yml_path()):
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    return None


def _scoped_subtree(agent_packages: dict[str, Any], scope: str) -> dict[str, Any]:
    if scope != "core":
        return agent_packages
    children = agent_packages.get("subdirectories") or {}
    return {
        "repositories": agent_packages.get("repositories") or [],
        "subdirectories": {"skills": children.get("skills") or {}},
    }


def workspace_repositories(data: dict[str, Any], scope: str) -> list[tuple[str, Path]]:
    """Return ``(name, path)`` for agent-packages repositories in ``scope``.

    ``core`` covers the agent-packages top level plus the skills subtree. Any
    other non-empty scope except ``none`` covers the whole subtree, filtered to
    the listed names when ``scope`` is a comma-separated list. Only checked-out
    directories qualify.
    """
    if scope in {"", "none"}:
        return []
    named = set() if scope in {"core", "all"} else set(_csv(scope))
    return [
        item
        for item in _checked_out_repositories(data, scope)
        if not named or item[0] in named
    ]


def _checked_out_repositories(
    data: dict[str, Any], scope: str
) -> list[tuple[str, Path]]:
    from agent_utilities.core.workspace_config import (
        _extract_repositories,
        _workspace_base_path,
    )

    agent_packages = (data.get("subdirectories") or {}).get("agent-packages") or {}
    root = _workspace_base_path(data, require_resolved=False) / "agent-packages"
    pairs = _extract_repositories(_scoped_subtree(agent_packages, scope), root)
    return [
        (path.name, path)
        for path, _url in pairs
        if not path.name.startswith(".") and path.is_dir()
    ]


def _codebase_items(scope: str, limit: int) -> list[BaselineItem]:
    data = _load_workspace_manifest()
    if data is None or limit <= 0:
        return []
    return [
        BaselineItem(
            leg="codebase",
            name=name,
            target=str(path),
            task_type="codebase",
            is_codebase=True,
        )
        for name, path in workspace_repositories(data, scope)[:limit]
    ]


# ── plan and enqueue ────────────────────────────────────────────────────


def plan_baseline() -> list[BaselineItem]:
    """Return the configured baseline in queue-class drain order."""
    settings = _settings()
    items = [_prompt_item()]
    items.extend(_skill_items(_csv(settings.kg_baseline_skill_providers)))
    try:
        items.extend(
            _codebase_items(
                str(settings.kg_baseline_codebases).strip().lower(),
                int(settings.kg_baseline_max_codebases),
            )
        )
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
            queued.append(entry)
        except Exception as exc:  # noqa: BLE001 - per-item rejection is reported
            rejected.append({**entry, "reason": type(exc).__name__})
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


def start_baseline_ingest(engine: Any, session: Any) -> bool:
    """Start the baseline thread once per engine; return whether it started.

    Disabled by ``KG_BASELINE_INGEST=0``. A launch failure logs and returns
    ``False``; it never propagates into daemon startup.
    """
    try:
        if not bool(_settings().kg_baseline_ingest):
            return False
        if getattr(engine, "_baseline_ingest_thread", None) is not None:
            return False
        from ..core.engine_tasks import _authorized_background_thread

        thread = _authorized_background_thread(
            session, run_baseline_ingest, name="KG-Baseline-Ingest", args=(engine,)
        )
        engine._baseline_ingest_thread = thread
        thread.start()
        return True
    except Exception as exc:  # noqa: BLE001 - never block or crash daemon start
        logger.error("baseline ingest launch failed (%s)", type(exc).__name__)
        return False


__all__ = [
    "PROMPTS_MAINTENANCE_REF",
    "SKILL_PROVIDER_TARGET_PREFIX",
    "BaselineItem",
    "enqueue_baseline",
    "ingest_prompt_library",
    "plan_baseline",
    "resolve_skill_corpus_root",
    "run_baseline_ingest",
    "start_baseline_ingest",
    "workspace_repositories",
]
