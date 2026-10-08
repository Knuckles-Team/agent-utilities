"""Baseline WorkItem shapes for prompts and skills (spec: baseline-ingestion).

* **prompts** - the prompt library (``ingest_prompts_to_graph``), as one
  ``scheduled_job`` WorkItem that runs the ``baseline_prompts`` maintenance tick.
* **skills** - one ``skill_workflows`` WorkItem per configured skill provider.
  The target is a ``skill-provider:<name>`` reference, resolved to the
  provider's verified root when the worker runs, so the durable target never
  holds a machine path.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..core.semantic_tiers import queue_class_for_task_type

#: Target prefix that names a skill provider instead of a filesystem path.
SKILL_PROVIDER_TARGET_PREFIX = "skill-provider:"
#: The installed universal-skills package sentinel ``skill_workflows`` accepts.
_UNIVERSAL_SKILLS_SENTINEL = "universal-skills"
#: Maintenance tick that ingests the prompt library.
PROMPTS_MAINTENANCE_REF = "baseline_prompts"


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


def csv_names(value: str) -> list[str]:
    """Split a comma-separated setting into trimmed, non-empty names."""
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def resolve_skill_corpus_root(target: str) -> str | None:
    """Resolve a ``skill_workflows`` target to the corpus root it names.

    ``universal-skills`` keeps its meaning (the installed package default,
    returned as ``None``). ``skill-provider:<name>`` resolves through the
    verified provider registry at run time. Any other value is an explicit
    root and passes through.

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


def skill_items(providers: list[str]) -> list[BaselineItem]:
    """One ``skill_workflows`` item per provider name."""
    return [
        BaselineItem(
            leg="skills",
            name=name,
            target=f"{SKILL_PROVIDER_TARGET_PREFIX}{name}",
            task_type="skill_workflows",
        )
        for name in providers
    ]


def prompt_item() -> BaselineItem:
    """The prompt-library item, run by the ``baseline_prompts`` tick."""
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


__all__ = [
    "PROMPTS_MAINTENANCE_REF",
    "SKILL_PROVIDER_TARGET_PREFIX",
    "BaselineItem",
    "csv_names",
    "ingest_prompt_library",
    "prompt_item",
    "resolve_skill_corpus_root",
    "skill_items",
]
