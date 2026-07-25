#!/usr/bin/python
from __future__ import annotations

"""Tool-pair-safety invariant for message-history compaction.

CONCEPT:AU-KG.memory.mementified-context — the ONE shared enforcement of the
tool-pair-safety rule every compaction path must obey: a compacted message list may never carry an
orphaned tool call (a ``ToolCallPart`` whose ``ToolReturnPart`` was dropped) or an
orphaned tool return (the reverse). An OpenAI/vLLM-compatible provider rejects such a
history with HTTP 400 mid-run, so any capability that removes messages from the
outgoing history must route its decision through here.

This mirrors the tool-pair-safe cutoff logic in ``pydantic_ai_harness.compaction._shared``
(the harness reference the ecosystem tracks), adapted to message *indices* because the
live Memento sawtooth (``capabilities/memento.py``) evicts arbitrary, non-contiguous
blocks rather than a single tail cutoff — so the constraint is a transitive both-or-
neither closure over pairs, not a windowed cutoff check.
"""

from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

__all__ = ["iter_tool_pairs", "enforce_tool_pair_safety"]


def iter_tool_pairs(messages: Sequence[ModelMessage]) -> list[tuple[int, int]]:
    """Return ``(call_index, return_index)`` message-index pairs.

    A pair is a ``ToolCallPart`` (in a ``ModelResponse``) matched by ``tool_call_id`` to
    a ``ToolReturnPart`` (in a later ``ModelRequest``). Only calls that HAVE a matching
    return are returned — a call still awaiting its result has no partner to strand, and
    a return whose call is absent from ``messages`` is a pre-existing orphan this module
    does not manufacture. Both endpoints are message indices into ``messages``.
    """
    call_index: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse):
            for call_part in msg.parts:
                if isinstance(call_part, ToolCallPart) and call_part.tool_call_id:
                    call_index[call_part.tool_call_id] = i

    pairs: list[tuple[int, int]] = []
    for j, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            for return_part in msg.parts:
                if (
                    isinstance(return_part, ToolReturnPart)
                    and return_part.tool_call_id in call_index
                ):
                    pairs.append((call_index[return_part.tool_call_id], j))
    return pairs


def enforce_tool_pair_safety(
    messages: Sequence[ModelMessage], evicted: set[int]
) -> set[int]:
    """Return a copy of ``evicted`` in which no tool-call / tool-return pair is split.

    For every pair, its two message indices must end up wholly inside or wholly outside
    the evicted set. Where the requested plan would evict exactly one side, BOTH sides
    are removed from the eviction (kept raw) rather than dragging the surviving partner
    out of the recent context that compaction deliberately preserves. Applied to a fixed
    point so a chain of pairs that share a message resolves transitively (a message that
    both returns one call and precedes another's return keeps its whole component
    together). The result is always a subset of ``evicted``.
    """
    pairs = iter_tool_pairs(messages)
    if not pairs:
        return set(evicted)

    safe = set(evicted)
    changed = True
    while changed:
        changed = False
        for call_i, return_j in pairs:
            if (call_i in safe) != (return_j in safe):
                # Exactly one half is evicted — un-evict both so neither is orphaned.
                safe.discard(call_i)
                safe.discard(return_j)
                changed = True
    return safe
