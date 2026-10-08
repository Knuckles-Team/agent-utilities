"""Semantic ingestion tiers that mirror the EG S1-S6 stage queue (spec: baseline-ingestion).

epistemic-graph defines six semantic-index stages in
``crates/eg-types/src/semantic_index/stage.rs`` and maps each stage to one of
three queue classes:

=====  =====================  ===========
Stage  Name                   Queue class
=====  =====================  ===========
S1     SourceCommit           fast
S2     GraphProjection        medium
S3     LexicalIndex           medium
S4     Vector                 slow_heavy
S5     AnnIndex               slow_heavy
S6     ReconcileAndActivate   slow_heavy
=====  =====================  ===========

The EG ``SemanticIndex`` method serves that queue only for SQL-sourced
semantic bindings. It has no admission op for file, skill, prompt or code
content, and EG decision-based ingestion lane routing is not
built. AU therefore keeps its own durable lanes and orders work by the same
stage sequence: source commit first, graph projection and lexical work next,
vector work last. The class names are the EG wire values, so a later switch to
an EG-routed decision changes the router, not the vocabulary.

The table is pure data. The slow class needs no producer here: the dedicated
``KG-Embedding-Backfill`` daemon thread already embeds committed nodes off the
queue, so vector work never blocks the fast or medium classes.
"""

from __future__ import annotations

from typing import Final

FAST: Final = "fast"
MEDIUM: Final = "medium"
SLOW_HEAVY: Final = "slow_heavy"

#: EG stage id -> EG queue class (``SemanticStage::queue_class``).
STAGE_QUEUE_CLASS: Final[dict[str, str]] = {
    "S1": FAST,
    "S2": MEDIUM,
    "S3": MEDIUM,
    "S4": SLOW_HEAVY,
    "S5": SLOW_HEAVY,
    "S6": SLOW_HEAVY,
}

#: Queue classes in drain order. A class never waits on a later class.
QUEUE_CLASS_ORDER: Final[tuple[str, ...]] = (FAST, MEDIUM, SLOW_HEAVY)

#: Durable WorkItem priority bucket per class (0 critical .. 3 background).
_CLASS_PRIORITY: Final[dict[str, int]] = {FAST: 1, MEDIUM: 2, SLOW_HEAVY: 3}

#: The entry stage each AU task type performs first. Metadata-sized work
#: (skills, prompts, connector metadata) is a source commit. Code and document
#: parsing projects structure into the graph and its lexical surface.
_TASK_TYPE_ENTRY_STAGE: Final[dict[str, str]] = {
    "skill_workflows": "S1",
    "scheduled_job": "S1",
    "connector_sync": "S1",
    "capability_hydration": "S1",
    "self_tool_surface": "S1",
    "codebase": "S2",
    "document": "S2",
    "diff": "S2",
    "content_url": "S2",
    "enrichment_backfill": "S4",
}


def entry_stage_for_task_type(task_type: str | None) -> str:
    """Return the EG stage id a task type enters at; unknown types enter at S2."""
    return _TASK_TYPE_ENTRY_STAGE.get(task_type or "", "S2")


def queue_class_for_task_type(task_type: str | None) -> str:
    """Return the EG queue class (``fast``/``medium``/``slow_heavy``) of a task type."""
    return STAGE_QUEUE_CLASS[entry_stage_for_task_type(task_type)]


def priority_for_queue_class(queue_class: str) -> int:
    """Return the WorkItem priority bucket of a queue class; unknown is background."""
    return _CLASS_PRIORITY.get(queue_class, 3)


def queue_class_rank(queue_class: str) -> int:
    """Return the drain position of a queue class; unknown classes drain last."""
    try:
        return QUEUE_CLASS_ORDER.index(queue_class)
    except ValueError:
        return len(QUEUE_CLASS_ORDER)


__all__ = [
    "FAST",
    "MEDIUM",
    "QUEUE_CLASS_ORDER",
    "SLOW_HEAVY",
    "STAGE_QUEUE_CLASS",
    "entry_stage_for_task_type",
    "priority_for_queue_class",
    "queue_class_for_task_type",
    "queue_class_rank",
]
