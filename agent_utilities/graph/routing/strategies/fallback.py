"""Multi-level fallback chain (R13).

When LLM planning fails or times out, the router escalates through fallbacks
ending in an unstructured natural-language extraction: it asks the model for a
plain comma-separated list of specialist names and matches them against the
available specialists. The prompt and the name-matching are extracted here as
the single source; the router still owns the agent invocation + control flow.
"""

from __future__ import annotations

from collections.abc import Iterable


def unstructured_fallback_prompt(system_prompt_str: str) -> str:
    """R13: instruct the model to emit only a comma-separated specialist list."""
    return (
        system_prompt_str + "\n\nCRITICAL: You failed JSON validation. "
        "Please reply ONLY with a simple text list of the exact agent names you "
        "want to use from the AVAILABLE SPECIALIST NODES list (separated by "
        "commas). DO NOT output conversational text, just the comma-separated "
        "agent names."
    )


def match_specialists_in_text(raw_text: str, available: Iterable[str]) -> list[str]:
    """R13: extract the known specialist names that appear in free-text output.

    Preserves the order of ``available`` and is case-insensitive (the monolith's
    original substring match).
    """
    low = (raw_text or "").lower()
    return [spec for spec in available if spec.lower() in low]
