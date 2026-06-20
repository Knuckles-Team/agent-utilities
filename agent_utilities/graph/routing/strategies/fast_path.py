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
**rules-first**: a turn is simple unless it shows a concrete STRUCTURAL signal that it needs
tools, specialists, or multi-step planning (an explicit ``/``-command, multiple clauses, or
sheer length). Greetings remain trivial; normal short questions now also take the one-round
fast path.

CONCEPT:EG-010, ORCH-1.73 — this module is now PURELY STRUCTURAL. The old hardcoded
``_ESCALATION_KEYWORDS`` list was deleted: domain/capability vocabulary lives in the KG, and a
turn that names a real fleet capability escalates via the engine's ``match_ontology_terms``
lexical gate (in ``orchestration.execution_profile``), not a frozen word list here.
"""

from __future__ import annotations

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

# NOTE: domain/action vocabulary used to live here as a hardcoded ``_ESCALATION_KEYWORDS``
# list (deploy/restart/list/…). It was deleted (CONCEPT:EG-010, ORCH-1.73): an unbounded word
# list is the wrong gate — it both missed real capabilities (no read verbs) and could not name
# the fleet ("portainer"). The domain vocabulary now lives in the KG as capability nodes, and a
# turn that names one escalates via the engine's ``match_ontology_terms`` lexical gate
# (``orchestration.execution_profile``). This module stays PURELY STRUCTURAL.


def _looks_multi_clause(query_lower: str) -> bool:
    """True when the turn bundles several requests (multi-step → full graph)."""
    # Several sentences, or an explicit conjunction of actions.
    sentence_breaks = (
        query_lower.count(". ") + query_lower.count("? ") + query_lower.count("; ")
    )
    return sentence_breaks >= 2 or " and then " in query_lower


def needs_full_orchestration(query: str) -> bool:
    """True when a turn shows a STRUCTURAL signal that it needs the full graph.

    Purely structural (CONCEPT:EG-010, ORCH-1.73): slash-command, over-length, or multiple
    clauses. Domain/capability escalation (a turn that names a real fleet tool) is no longer
    decided here from a hardcoded word list — it is decided by the engine's ontology lexical
    gate in ``orchestration.execution_profile`` against the live KG. This is the escalation
    gate's structural half: the fast path handles everything that does NOT trip it.
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

    return False


def orchestration_signal_strength(query: str) -> int:
    """Graded STRUCTURAL signal that a turn needs full orchestration (CONCEPT:ORCH-1.69/1.73).

    Built from the same structural signals as :func:`needs_full_orchestration` so the rule
    stays single-sourced, but graded:

    * ``0`` — no structural signal: hand off to the ontology lexical gate, which escalates if
      the turn names a real fleet capability and otherwise leaves it lean.
    * ``2+`` — a strong, unambiguous structural need for the full multi-agent graph.

    A slash-command, an over-length turn, or multiple clauses each count as strong on their
    own. Domain/action vocabulary is intentionally NOT scored here (it lives in the KG); the
    keyword-driven "strength 1" middle is gone — the lexical gate replaces it.
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
