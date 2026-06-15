#!/usr/bin/python
from __future__ import annotations

"""Explicit node-vs-node contradiction / friction detection.

CONCEPT:KG-2.83

Adopts the "Building a Second Brain" *night-shift Critic* role: when a new note
is filed against an existing belief, a good second brain does not silently
overwrite the old note — it surfaces the tension as **[FRICTION]** so a human
decides which belief survives. Today the KG only ever expresses contradiction
*implicitly* — through failure analysis (:mod:`.failure_analyzer`), dedup, and
SHACL value constraints. None of those compares one *claim node* against another
to say "these two beliefs oppose each other." This module is that missing
explicit node↔node contradiction surface.

It is deliberately **propose-only**: :class:`ContradictionDetector` emits
:class:`FrictionFinding` records and never mutates, resolves, or deletes a
claim. Resolution is left to the existing proposal/governance spine — the Critic
flags, it does not arbitrate.

The default similarity is **zero-infra**: a deterministic lexical heuristic
(content-token Jaccard blended with bigram/phrase overlap, mirroring
:mod:`agent_utilities.knowledge_graph.retrieval.reasoning_reranker`) so the
detector runs with no embeddings, model, or network. When embeddings *are*
available a caller can inject ``similarity_fn`` for a stronger topical signal;
the opposition (polarity) check stays lexical and deterministic regardless.

Concept: contradiction-detection
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "Claim",
    "FrictionFinding",
    "lexical_similarity",
    "opposes",
    "ContradictionDetector",
]

# Keep intra-word apostrophes so contractions like ``doesn't`` survive as one
# token (``doesn't``) for negation detection, then normalize the apostrophe out.
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
# Glue words that should not drive topical overlap (mirrors reasoning_reranker).
_STOP = frozenset(
    "a an the of to in on for and or is are be was were with as at by from this that "
    "it its into about over under what which who whom how why when where will would "
    "can could should than then so but if".split()
)

# Conceptual-polarity lexicons: a statement that frames a subject as an
# *unresolved problem/blocker* vs. one that frames it as *resolved/overcome*.
# Used only when both sides share subject matter (conservative). This catches
# friction the antonym table can't, e.g. "X is the binding constraint" vs.
# "Y undercuts X on cost".
_BLOCKER_CUES = frozenset(
    "constraint constraints bottleneck blocker blockers barrier barriers limit "
    "limiting limitation binding obstacle problem expensive costly prohibitive".split()
)
_RESOLVER_CUES = frozenset(
    "undercut undercuts undercutting solve solves solved resolves resolved "
    "overcome overcomes eliminates eliminate cheaper affordable removes "
    "alleviates alleviate".split()
)

# Negation cues whose presence flips the polarity of a statement.
_NEGATIONS = frozenset(
    "not no never none cannot cant dont doesnt didnt wont isnt arent wasnt werent "
    "without fails failed".split()
)

# Antonym pairs whose co-occurrence on a shared topic signals opposing polarity.
# Stored bidirectionally so order does not matter at lookup time. Conservative
# on purpose — only well-separated, low-ambiguity opposites (favor precision).
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("rise", "fall"),
    ("rises", "falls"),
    ("rose", "fell"),
    ("rising", "falling"),
    ("improve", "degrade"),
    ("improves", "degrades"),
    ("improved", "degraded"),
    ("strengthen", "weaken"),
    ("strengthens", "weakens"),
    ("cheaper", "costlier"),
    ("cheap", "expensive"),
    ("undercut", "exceed"),
    ("undercuts", "exceeds"),
    ("true", "false"),
    ("supports", "contradicts"),
    ("support", "contradict"),
    ("enable", "prevent"),
    ("enables", "prevents"),
    ("up", "down"),
    ("higher", "lower"),
    ("more", "less"),
    ("faster", "slower"),
    ("win", "lose"),
    ("wins", "loses"),
    ("gain", "loss"),
    ("positive", "negative"),
)

# Build the bidirectional lookup once at import time (deterministic).
_ANTONYMS: dict[str, frozenset[str]] = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYMS[_a] = _ANTONYMS.get(_a, frozenset()) | {_b}
    _ANTONYMS[_b] = _ANTONYMS.get(_b, frozenset()) | {_a}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Claim:
    """An atomic belief stored as a KG node: a stable id and its text."""

    id: str
    text: str


@dataclass
class FrictionFinding:
    """One surfaced contradiction between a new claim and an existing one.

    Propose-only: it records *that* two beliefs conflict and *why*, never how to
    resolve them.
    """

    new_id: str
    conflict_id: str
    similarity: float  # topical overlap of the two claims, in [0, 1]
    reason: str  # human-readable explanation of the opposition
    severity: str  # "high" | "medium" | "low"


def _content_tokens(text: str) -> list[str]:
    """Lowercased content words, glue stripped (deterministic order preserved).

    Apostrophes are stripped *after* matching so a contraction like ``doesn't``
    normalizes to ``doesnt`` (caught by the ``endswith('nt')`` negation cue).
    """
    return [
        t.replace("'", "")
        for t in _WORD.findall((text or "").lower())
        if t.replace("'", "") not in _STOP
    ]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False)) if len(tokens) > 1 else set()


def lexical_similarity(a: str, b: str) -> float:
    """Deterministic, dependency-free topical similarity in [0, 1].

    Blends content-token Jaccard with phrase (bigram) overlap — the same lexical
    signal :mod:`reasoning_reranker` uses — so two statements about the *same
    subject* score high regardless of whether they agree. No model, no network.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    ba, bb = _bigrams(ta), _bigrams(tb)
    if ba or bb:
        bigram = len(ba & bb) / len(ba | bb)
    else:
        bigram = 0.0
    raw = 0.7 * jaccard + 0.3 * bigram
    return max(0.0, min(1.0, raw))


def _negation_count(tokens: list[str]) -> int:
    """Count polarity-flipping cues, including contracted ``n't`` forms."""
    return sum(1 for t in tokens if t in _NEGATIONS or t.endswith("nt"))


def _shared_topic_tokens(ta: list[str], tb: list[str]) -> set[str]:
    """Content tokens common to both statements, excluding the polarity markers."""
    shared = set(ta) & set(tb)
    return {t for t in shared if t not in _NEGATIONS and t not in _ANTONYMS}


def opposes(a: str, b: str) -> bool:
    """True when two statements are topically related but assert opposing polarity.

    Conservative by design (precision over recall): returns ``True`` only when
    the statements share subject matter AND one of these polarity signals fires:

    * **Negation flip** — the same claim, but exactly one side carries a negation
      cue (``not``/``no``/``never``/``n't``/...). Both-negated or both-plain do
      not oppose.
    * **Antonym flip** — each side carries one half of a known antonym pair
      (``increase``/``decrease``, ``improve``/``degrade``, ``cheaper``/
      ``costlier``, ``supports``/``contradicts``, ...) over a shared topic.
    * **Numeric contradiction** — the same subject is asserted with different
      numeric values.
    * **Frame flip** — one side frames a shared subject as an unresolved
      blocker/cost (``constraint``/``binding``/``expensive``) while the other
      frames it as resolved/overcome (``undercut``/``solves``/``cheaper``).

    Unrelated statements (no shared topic) and merely-different statements (no
    polarity signal) return ``False``.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return False

    # Some shared subject matter is required for any verdict — otherwise the two
    # statements are simply about different things, not in conflict.
    shared_topic = _shared_topic_tokens(ta, tb)
    sa, sb = set(ta), set(tb)

    # 1. Antonym flip: each side holds one half of an antonym pair over a topic
    #    they actually share. This needs a shared subject token to be meaningful.
    if shared_topic:
        for tok in sa:
            for anti in _ANTONYMS.get(tok, frozenset()):
                if anti in sb:
                    return True

    # 2. Negation flip on an otherwise-shared claim: exactly one side is negated
    #    and the two share enough subject matter to be the same assertion.
    neg_a, neg_b = _negation_count(ta), _negation_count(tb)
    if (neg_a > 0) != (neg_b > 0):
        # Compare the statements with polarity cues removed; if what remains is
        # substantially the same claim, the lone negation is a genuine flip.
        core_a = [t for t in ta if t not in _NEGATIONS and not t.endswith("nt")]
        core_b = [t for t in tb if t not in _NEGATIONS and not t.endswith("nt")]
        ca, cb = set(core_a), set(core_b)
        if ca and cb:
            # Containment over the smaller core: tolerant of light stemming
            # noise (``pass``/``passes``) while still requiring the bulk of the
            # shorter claim to be shared subject matter.
            inter = len(ca & cb)
            containment = inter / min(len(ca), len(cb))
            if containment >= 0.5:
                return True

    # 3. Numeric contradiction: same subject, differing numbers.
    if shared_topic:
        nums_a = _NUM.findall(a or "")
        nums_b = _NUM.findall(b or "")
        if nums_a and nums_b and set(nums_a) != set(nums_b):
            return True

    # 4. Frame flip: one side frames the shared subject as an unresolved
    #    blocker/cost, the other as resolved/overcome.
    if shared_topic:
        a_blocks = bool(sa & _BLOCKER_CUES)
        b_blocks = bool(sb & _BLOCKER_CUES)
        a_resolves = bool(sa & _RESOLVER_CUES)
        b_resolves = bool(sb & _RESOLVER_CUES)
        if (a_blocks and b_resolves) or (b_blocks and a_resolves):
            return True

    return False


def _severity_for(similarity: float) -> str:
    """Map topical similarity to a coarse severity band (deterministic)."""
    if similarity >= 0.6:
        return "high"
    if similarity >= 0.4:
        return "medium"
    return "low"


class ContradictionDetector:
    """Explicit node↔node contradiction/friction surface (CONCEPT:KG-2.83).

    The night-shift Critic: given a candidate claim and the existing belief set,
    surface every existing belief that is topically similar yet opposes the
    candidate. It only proposes :class:`FrictionFinding` records — it never
    resolves, merges, or overwrites a claim.

    Args:
        similarity_fn: Optional injected ``(a, b) -> float`` topical similarity
            in [0, 1] (e.g. an embedding cosine). Defaults to the zero-infra
            :func:`lexical_similarity`. The opposition check is always lexical.
        min_similarity: A pair must reach at least this topical similarity before
            its polarity is even considered — keeps unrelated statements out.
            Kept modest because :func:`opposes` already enforces shared subject
            matter; this gate is a coarse secondary pre-filter, not the primary
            topicality check.
    """

    def __init__(
        self,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        min_similarity: float = 0.15,
    ) -> None:
        self.similarity_fn = similarity_fn or lexical_similarity
        self.min_similarity = float(min_similarity)

    def _friction(self, new: Claim, existing: Claim) -> FrictionFinding | None:
        """Build a finding if ``existing`` is similar-and-opposing to ``new``."""
        sim = float(self.similarity_fn(new.text, existing.text))
        if sim < self.min_similarity:
            return None
        if not opposes(new.text, existing.text):
            return None
        reason = (
            f"[FRICTION] new claim '{new.text}' opposes existing belief "
            f"'{existing.text}' (topical similarity {sim:.2f})"
        )
        return FrictionFinding(
            new_id=new.id,
            conflict_id=existing.id,
            similarity=round(sim, 6),
            reason=reason,
            severity=_severity_for(sim),
        )

    def check(
        self, new_claim: Claim, existing: Sequence[Claim]
    ) -> list[FrictionFinding]:
        """Surface every existing belief that contradicts ``new_claim``.

        For each existing claim that is topically similar (``>= min_similarity``)
        AND opposes the new one, emit a :class:`FrictionFinding`. Severity scales
        with similarity. Results are sorted most-similar first, with the conflict
        id as a stable tiebreaker for determinism. Never mutates anything.
        """
        findings: list[FrictionFinding] = []
        for other in existing:
            if other.id == new_claim.id:
                continue
            finding = self._friction(new_claim, other)
            if finding is not None:
                findings.append(finding)
        findings.sort(key=lambda f: (-f.similarity, f.conflict_id))
        return findings

    def scan(self, claims: Sequence[Claim]) -> list[FrictionFinding]:
        """All-pairs friction scan over a belief set.

        Each unordered pair is checked once; symmetric duplicates are dropped by
        orienting every finding so ``new_id`` is the lexicographically smaller id.
        Sorted most-similar first, then by the pair ids for determinism.
        """
        findings: list[FrictionFinding] = []
        items = list(claims)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                # Orient deterministically so the symmetric pair is counted once.
                lo, hi = (a, b) if a.id <= b.id else (b, a)
                finding = self._friction(lo, hi)
                if finding is not None:
                    findings.append(finding)
        findings.sort(key=lambda f: (-f.similarity, f.new_id, f.conflict_id))
        return findings
