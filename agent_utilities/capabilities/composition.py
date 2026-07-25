#!/usr/bin/python
from __future__ import annotations

"""The single default-capability assembly for every governed agent.

This module is the ONE place the default-ON ``AbstractCapability`` set is built --
the single composition seam. Historically only ``agent/factory.py``'s
``create_agent`` assembled the reliability capabilities (stuck-loop detection, context
warnings, tool-output eviction, live Memento compaction), so the ~80 agents that the
graph executor and the KG-internal sites build directly through
``core/contextual_model.py`` ``create_context_agent`` ran BARE — no compaction, no
stuck-loop guard, no eviction. This module holds the assembly so BOTH seams share it:
``create_context_agent`` applies it by default to every agent it constructs, and
``create_agent`` builds its richly-configured list from the same function. See
``create_context_agent(default_capabilities=...)``.

The reliability capabilities are safe to attach unconditionally: each one only acts
when its own trigger fires (compaction only over budget, eviction only on an oversized
tool result, the warner only near the token limit, the stuck-loop guard only on
repeated identical tool calls), so a normal run is unaffected.
"""

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai.capabilities import AbstractCapability


def default_runtime_capabilities(
    *,
    stuck_loop_detection: bool = True,
    stuck_loop_max_repeated: int = 3,
    stuck_loop_action: Literal["warn", "error"] = "warn",
    context_warnings: bool = True,
    max_context_tokens: int | None = None,
    output_eviction: bool = True,
    eviction_threshold_chars: int = 80_000,
    memento_compaction: bool = True,
    include_checkpoints: bool = False,
    checkpoint_store: Any | None = None,
    checkpoint_frequency: Literal[
        "every_tool", "every_turn", "manual_only"
    ] = "every_tool",
    include_teams: bool = False,
) -> list[AbstractCapability[Any]]:
    """Build the default-ON reliability capability set (excluding ``HooksCapability``).

    ``HooksCapability`` is deliberately left to the caller: ``create_agent`` folds the
    RLM large-output hook into its hooks list, while ``create_context_agent`` attaches a
    hookless one. Every other default-ON capability is assembled here so the two seams
    can never drift.

    Each flag reproduces the corresponding ``create_agent`` default, so a bare
    ``create_context_agent`` call now receives exactly what a factory-built agent always
    had.
    """
    from .checkpointing import CheckpointMiddleware, InMemoryCheckpointStore
    from .context_warnings import ContextLimitWarner
    from .eviction import ToolOutputEviction
    from .memento import MementoCompaction
    from .stuck_loop import StuckLoopDetection
    from .teams import TeamCapability

    capabilities: list[AbstractCapability[Any]] = []

    if stuck_loop_detection:
        capabilities.append(
            StuckLoopDetection(
                max_repeated=stuck_loop_max_repeated,
                action=stuck_loop_action,
            )
        )

    if context_warnings:
        capabilities.append(ContextLimitWarner(max_tokens=max_context_tokens))

    if output_eviction:
        capabilities.append(
            ToolOutputEviction(threshold_chars=eviction_threshold_chars)
        )

    # CONCEPT:AU-KG.memory.memento-compress-evict — the live block-compress-evict sawtooth
    # (default ON). Routes eviction through the tool-pair-safe helper so a compacted
    # message list never orphans a tool-call/tool-return pair.
    if memento_compaction:
        capabilities.append(MementoCompaction(max_tokens=max_context_tokens))

    if include_checkpoints:
        store = checkpoint_store or InMemoryCheckpointStore()
        capabilities.append(
            CheckpointMiddleware(store=store, frequency=checkpoint_frequency)
        )

    if include_teams:
        capabilities.append(TeamCapability())

    return capabilities


def merge_capabilities(
    existing: list[AbstractCapability[Any]],
    defaults: list[AbstractCapability[Any]],
) -> list[AbstractCapability[Any]]:
    """Return ``existing`` plus every default whose concrete type is not already present.

    De-duplication is by concrete capability type: a caller that already supplied its
    own ``MementoCompaction`` keeps it, and the default one is not added on top. Caller
    capabilities are preserved in their original order; missing defaults are appended.
    """
    present = {type(cap) for cap in existing}
    merged = list(existing)
    for cap in defaults:
        if type(cap) not in present:
            merged.append(cap)
            present.add(type(cap))
    return merged
