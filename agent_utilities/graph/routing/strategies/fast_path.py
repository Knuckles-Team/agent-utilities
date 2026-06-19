"""R1 — Fast-path / adaptive model routing (CONCEPT:KG-2.1, widened CONCEPT:ORCH-1.63).

Simple conversational/Q&A turns short-circuit the full multi-agent planning pipeline and
get a single lightweight-model round instead of router → planner → expert → verifier →
synthesizer. This module is the *single source of truth* for the simple-turn DETECTION;
the response execution still lives in the router implementation (it needs the agent/model
factory + graph deps), but both call :func:`is_trivial_query` so the rule is defined once.
It is universal — used by every entrypoint through the router, not a messaging-only patch.

CONCEPT:ORCH-1.63 — the original rule was far too narrow: ≤6 words AND starting with a
fixed greeting prefix. So a normal simple question ("what's the status of X?", "can you
summarise this?") did NOT qualify and paid for the full graph. The widened classifier is
**rules-first**: a turn is simple unless it shows a concrete signal that it needs tools,
specialists, or multi-step planning (an imperative/tool keyword, an explicit ``/``-command,
multiple clauses, or sheer length). Greetings remain trivial; normal short questions now
also take the one-round fast path. Only genuinely tool/plan-shaped turns escalate.
"""

from __future__ import annotations

import re

# Conversational openers that, in a short utterance, do not require specialist tools or
# planning. Kept for the explicit "definitely trivial" shortcut.
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

# A greeting-only utterance stays trivial up to this many words.
MAX_TRIVIAL_WORDS = 6

# Above this word count a turn is treated as substantial enough to warrant the full graph
# (long asks tend to bundle several requirements / multi-step work).
MAX_SIMPLE_WORDS = 40

# Tokens that signal the turn wants the agent to *do* something with tools, specialists, or a
# multi-step plan — escalate these to the full orchestration graph rather than answer on the
# single-round fast path. Word-boundary matched so "deploy" matches but "redeployment" prose
# in a question still has to clear the other gates.
_ESCALATION_KEYWORDS: tuple[str, ...] = (
    # tool / fleet actions
    "deploy",
    "provision",
    "restart",
    "build",
    "install",
    "configure",
    "run",
    "execute",
    "create",
    "delete",
    "remove",
    "update",
    "migrate",
    "refactor",
    "implement",
    "fix",
    "debug",
    "patch",
    "merge",
    "commit",
    "push",
    "ingest",
    "scan",
    "search",
    "query",
    "fetch",
    "scrape",
    "crawl",
    "analyze",
    "analyse",
    "orchestrate",
    "schedule",
    "trigger",
    "compute",
    "calculate",
    "optimize",
    "optimise",
    "design",
    "generate",
    "forecast",
    "simulate",
    "train",
    "evaluate",
    # multi-step / specialist asks
    "step by step",
    "step-by-step",
    "plan",
    "workflow",
    "pipeline",
    "and then",
)


def _looks_multi_clause(query_lower: str) -> bool:
    """True when the turn bundles several requests (multi-step → full graph)."""
    # Several sentences, or an explicit conjunction of actions.
    sentence_breaks = (
        query_lower.count(". ") + query_lower.count("? ") + query_lower.count("; ")
    )
    return sentence_breaks >= 2 or " and then " in query_lower


def needs_full_orchestration(query: str) -> bool:
    """True when a turn genuinely needs tools / specialists / multi-step planning.

    This is the escalation gate: the fast path handles everything that does NOT trip it.
    """
    if not query:
        return False
    q = query.strip()
    query_lower = q.lower()

    # Explicit slash-command (e.g. ``/deploy``, ``/skill foo``) — a directed action.
    if q.startswith("/"):
        return True

    # Long turns tend to bundle multiple requirements / multi-step work.
    if len(q.split()) > MAX_SIMPLE_WORDS:
        return True

    # Multiple clauses / chained actions.
    if _looks_multi_clause(query_lower):
        return True

    # A tool/action/plan keyword (word-boundary matched for single words; substring for the
    # multi-word phrases which are already specific).
    for kw in _ESCALATION_KEYWORDS:
        if " " in kw or "-" in kw:
            if kw in query_lower:
                return True
        elif re.search(rf"\b{re.escape(kw)}\b", query_lower):
            return True

    return False


def orchestration_signal_strength(query: str) -> int:
    """Count how STRONGLY a turn signals a need for full orchestration (CONCEPT:ORCH-1.69).

    Built from the same signals as :func:`needs_full_orchestration` so the rule stays
    single-sourced, but graded instead of boolean:

    * ``0`` — no signal: a trivial/conversational turn (take the lean direct-completion shape).
    * ``1`` — a single weak signal: the *ambiguous middle* the escalating planner sends to the
      KG capability stage to disambiguate (is it a real tool task, or just conversational?).
    * ``2+`` — a strong, unambiguous need for the full multi-agent graph.

    A slash-command, an over-length turn, or multiple clauses each count as strong on their own;
    each distinct action keyword counts as one.
    """
    if not query:
        return 0
    q = query.strip()
    ql = q.lower()
    strength = 0
    if q.startswith("/"):
        strength += 2
    if len(q.split()) > MAX_SIMPLE_WORDS:
        strength += 2
    if _looks_multi_clause(ql):
        strength += 2
    for kw in _ESCALATION_KEYWORDS:
        if " " in kw or "-" in kw:
            if kw in ql:
                strength += 1
        elif re.search(rf"\b{re.escape(kw)}\b", ql):
            strength += 1
    return strength


def is_trivial_query(query: str) -> bool:
    """Return True if ``query`` should take the single-round fast path (CONCEPT:ORCH-1.63).

    Rules-first: a turn is fast-path eligible when it does NOT show a concrete signal that
    it needs tools, specialists, or multi-step planning (see :func:`needs_full_orchestration`).
    A pure short greeting is always trivial; a normal simple question is now also handled on
    the fast path instead of paying for the full multi-agent graph.
    """
    if not query:
        return False

    query_lower = query.strip().lower()
    word_count = len(query.split())

    # Explicit-greeting shortcut (the original behaviour) — always trivial when short.
    if word_count <= MAX_TRIVIAL_WORDS and any(
        query_lower.startswith(p) for p in TRIVIAL_PREFIXES
    ):
        return True

    # Otherwise: simple unless it needs full orchestration.
    return not needs_full_orchestration(query)


class FastPathStrategy:
    """RoutingStrategy wrapper for R1.

    ``decide`` returns the sentinel string ``"fast_path"`` when the query is simple
    (signalling the Router/monolith to take the direct-response path), otherwise ``None``
    so the pipeline continues into the full orchestration graph.
    """

    name = "fast_path"
    concept = "KG-2.1"

    async def decide(self, ctx) -> str | None:
        query = getattr(getattr(ctx, "state", None), "query", "") or ""
        return "fast_path" if is_trivial_query(query) else None
