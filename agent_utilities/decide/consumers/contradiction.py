"""EH-034: TMS contradiction-handling SUGGESTIONS through EG ``Decide``.

A friction finding (two similar claims that oppose each other) gets a
suggested handling -- retract the new claim, retract the existing one, or keep
both for a human -- from a ``classify`` question with irreversible safety
(never explored). It is a suggestion attached to the finding and nothing
more: the detector stays propose-only, no claim is ever retracted here, and
the deterministic fallback is ``keep_both`` (surface it, let a human decide),
which is exactly what the detector did before.
"""

from __future__ import annotations

from agent_utilities import decide
from agent_utilities.decide.options import Option

KEEP_BOTH = "keep_both"


def suggest_handling(
    new_id: str, conflict_id: str, similarity: float
) -> tuple[str, str | None]:
    """``(suggestion, record_id)`` for one finding; ``keep_both`` when EG does not decide."""
    options = [
        Option(f"retract:{new_id}", {"confidence": similarity, "support": 0.0}),
        Option(f"retract:{conflict_id}", {"confidence": similarity, "support": 0.0}),
        Option(KEEP_BOTH, {"confidence": 1.0 - similarity, "support": 1.0}),
    ]
    choice = decide.choose("au.tms.contradiction", options, lambda: KEEP_BOTH)
    return str(choice.option_id or KEEP_BOTH), choice.record_id


__all__ = ["KEEP_BOTH", "suggest_handling"]
