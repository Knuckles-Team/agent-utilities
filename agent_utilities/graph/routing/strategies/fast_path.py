"""R1 — Fast-path / adaptive model routing (CONCEPT:KG-2.1).

Trivial/conversational queries short-circuit the full planning pipeline and get
a direct lightweight-model response. This module is the *single source of truth*
for the trivial-query DETECTION; the response execution still lives in the
router implementation (it needs the agent/model factory + graph deps), but both
call :func:`is_trivial_query` so the rule is defined once.
"""

from __future__ import annotations

# Conversational openers that, in a short utterance, do not require specialist
# tools or planning.
TRIVIAL_PREFIXES: tuple[str, ...] = (
    "hello",
    "hi ",
    "hey ",
    "thanks",
    "thank you",
    "ok",
    "bye",
    "what can you",
)

MAX_TRIVIAL_WORDS = 6


def is_trivial_query(query: str) -> bool:
    """Return True if ``query`` is a short conversational utterance.

    A query is trivial when it has at most :data:`MAX_TRIVIAL_WORDS` words and
    starts with one of :data:`TRIVIAL_PREFIXES` (case-insensitive).
    """
    if not query:
        return False
    query_lower = query.strip().lower()
    word_count = len(query.split())
    return word_count <= MAX_TRIVIAL_WORDS and any(
        query_lower.startswith(p) for p in TRIVIAL_PREFIXES
    )


class FastPathStrategy:
    """RoutingStrategy wrapper for R1.

    ``decide`` returns the sentinel string ``"fast_path"`` when the query is
    trivial (signalling the Router/monolith to take the direct-response path),
    otherwise ``None`` so the pipeline continues.
    """

    name = "fast_path"
    concept = "KG-2.1"

    async def decide(self, ctx) -> str | None:
        query = getattr(getattr(ctx, "state", None), "query", "") or ""
        return "fast_path" if is_trivial_query(query) else None
