"""Deterministic free-text -> EG task-IRI classification.

**Why this exists.** A caller proposing a mapping from a free-text task
description to one of EG's native task IRIs must never have that proposal
treated as an authoritative identity -- only as a labelled claim requiring
evidence (AU-CONTROL-R008). This module is a deterministic, LLM-free
classifier that proposes a `task_iri` for EG's ontology search from the free
text itself, recorded as a LABELLED CLAIM (never a proof) with its own
provenance: no model inference, no network call, no LLM of any kind. An
operator who later wants a proposal from an actual LLM run instead gets a
SEPARATE, still-claim-classed premise -- this module never becomes one by
adding a model behind the same name.

**Keyword vocabulary provenance.** Every keyword below is mined directly
from EG's agent ontology -- each task IRI's own label, plus the path
segments of the capability IRIs its `needs` list names (e.g.
``eg:capability/generation/code`` contributes ``generation`` and ``code``).
Nothing here is a hand-invented synonym list; if the ontology's
task/capability shape changes, this table is the one place that must be
re-derived from it.

**Abstention is the safe default.** `classify_task_text` returns ``None``
(never a fabricated best guess) when: no token overlaps any IRI's keyword
set, the best score is below :data:`CONFIDENCE_FLOOR`, or the top two
IRIs are too close to call (:data:`AMBIGUITY_MARGIN`). An abstention leaves
the caller in its prior, safe, fail-closed shape -- using this module is
never observably worse than not having it at all.
"""

from __future__ import annotations

import hashlib
import math
import re

from agent_utilities.api.agent_control_contracts import TaskClassificationClaim, TaskIri

#: Mined from EG's agent ontology -- see module docstring. Each set is the
#: task's own label token(s) plus the path segments of its `needs`
#: capability IRIs (the slug after the LAST `/` in each, and the segment
#: right before it when that segment is itself a meaningful word rather than
#: a bare category like ``capability``).
_TASK_KEYWORDS: dict[TaskIri, frozenset[str]] = {
    "eg:task/research": frozenset(
        {"research", "retrieval", "analysis", "summarize", "reasoning", "plan"}
    ),
    "eg:task/implement": frozenset(
        {
            "implement",
            "generation",
            "code",
            "retrieval",
            "document",
            "read",
            "action",
            "file",
            "write",
        }
    ),
    "eg:task/review": frozenset(
        {
            "review",
            "analysis",
            "evaluate",
            "reasoning",
            "critique",
            "retrieval",
            "document",
            "read",
        }
    ),
    "eg:task/operate": frozenset(
        {"operate", "action", "retrieval", "graph", "query", "reasoning", "verify"}
    ),
    "eg:task/communicate": frozenset(
        {
            "communicate",
            "generation",
            "text",
            "analysis",
            "summarize",
            "action",
            "message",
            "send",
        }
    ),
}

#: Minimum Ochiai (cosine-on-binary-vectors) overlap score to accept a
#: classification. Below this, the free text is too lexically distant from
#: every task IRI's vocabulary to assert even a claim.
CONFIDENCE_FLOOR = 0.25

#: Minimum margin the top-scoring task IRI must hold over the second-best.
#: A near-tie is exactly the case a deterministic classifier must not
#: silently resolve by insertion order -- it must abstain instead.
AMBIGUITY_MARGIN = 0.05

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "for",
        "and",
        "or",
        "in",
        "on",
        "with",
        "this",
        "that",
        "please",
        "task",
        "need",
        "needs",
        "is",
        "are",
        "it",
    }
)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    )


def _text_digest(text: str) -> str:
    """SHA-256 of the normalized text -- never the raw text itself, so the
    claim can be logged/persisted without duplicating the task string."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _score(query_tokens: frozenset[str], keywords: frozenset[str]) -> float:
    """Ochiai coefficient (cosine similarity of binary bag-of-words vectors)
    -- bounded in [0, 1], symmetric, and does not reward a longer query or a
    larger keyword set on its own the way a raw intersection count would."""
    overlap = query_tokens & keywords
    if not overlap:
        return 0.0
    return len(overlap) / math.sqrt(len(query_tokens) * len(keywords))


def classify_task_text(text: str) -> TaskClassificationClaim | None:
    """Propose a task IRI for free-text ``text``, or abstain (``None``).

    Deterministic and LLM-free: bag-of-words lexical overlap only, no model
    inference and no network call. The result, when not ``None``, is always
    ``evidence_class="claim"`` -- callers must never promote it to a proof.
    """
    tokens = _tokenize(text)
    if not tokens:
        return None

    scored = sorted(
        (
            (_score(tokens, keywords), task_iri, tokens & keywords)
            for task_iri, keywords in _TASK_KEYWORDS.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_iri, best_overlap = scored[0]
    if best_score < CONFIDENCE_FLOOR:
        return None
    if len(scored) > 1 and (best_score - scored[1][0]) < AMBIGUITY_MARGIN:
        return None

    return TaskClassificationClaim(
        task_iri=best_iri,
        confidence=round(min(best_score, 1.0), 4),
        method="lexical_keyword_overlap",
        matched_keywords=tuple(sorted(best_overlap)),
        text_digest=_text_digest(text),
    )


__all__ = [
    "AMBIGUITY_MARGIN",
    "CONFIDENCE_FLOOR",
    "classify_task_text",
]
