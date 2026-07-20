from __future__ import annotations

"""Research-craft discipline primitives: failure triage + disconfirming-evidence log.

CONCEPT:AU-AHE.harness.research-craft-discipline — Research-Craft Discipline

Operationalizes two pieces of researcher craft that keep self-improving
agents honest about their own failures and beliefs:

1. A **failure-transcript triage queue** — Andrew Ng's "Be good at research"
   advice: pull 100 failures, sort them into piles, and attack the biggest pile
   first. :class:`FailureTriage` clusters failing cases by pile label and
   surfaces the largest cluster so effort goes where it removes the most loss.

2. A **disconfirming-evidence research log** — Darwin's rule of recording any
   fact that runs *against* a working hypothesis the moment it appears, because
   (as he observed) the mind forgets inconvenient evidence far faster than
   convenient evidence. This guards against Feynman's "the first principle is
   that you must not fool yourself — and you are the easiest person to fool."
   :class:`ResearchLog` records belief evidence with an explicit supports/refutes
   flag, makes disconfirming evidence first-class and queryable, and flags
   hypotheses that are *contested* (carry both supporting and refuting facts).

Both structures are deterministic, pure-Python, and consume the harness's own
:mod:`evidence_corpus` shapes: :class:`FailureTriage.from_evidence_corpus`
duck-types a ``continuous_evaluation_engine`` ``EvidenceCorpus`` (its
``failure_clusters`` and ``entries``) so triage is fed directly by trace
distillation.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FailureCase",
    "BeliefEntry",
    "FailureTriage",
    "ResearchLog",
]


@dataclass
class FailureCase:
    """A single failing transcript pulled into the triage queue.

    Attributes:
        case_id: Stable identifier for the failing case/task.
        summary: One-line human-readable description of the failure.
        pile: Cluster/category label the case sorts into.
        transcript: Optional full transcript text to read when attacking a pile.
    """

    case_id: str
    summary: str
    pile: str
    transcript: str = ""


@dataclass
class BeliefEntry:
    """A piece of evidence recorded for or against a hypothesis.

    Attributes:
        hypothesis: The working hypothesis the evidence bears on.
        evidence: The observed fact.
        supports: ``True`` if the evidence confirms the hypothesis, ``False`` if
            it disconfirms it (Darwin's rule: record the disconfirming kind on
            the spot, before the mind forgets it).
        timestamp: Caller-supplied timestamp string, or ``""`` if unspecified.
            Kept as an opaque string so the log stays deterministic and
            stdlib-only (no clock reads).
    """

    hypothesis: str
    evidence: str
    supports: bool
    timestamp: str = ""


class FailureTriage:
    """Pull-N-failures → sort into piles → attack the biggest (CONCEPT:AU-AHE.harness.research-craft-discipline).

    A deterministic triage queue: every failing case is added with a pile label,
    piles are ranked by size, and the largest pile is surfaced as the one to
    attack first. This is the research-craft move of looking at a representative
    sample of failures and clustering them rather than chasing the most recent
    one.
    """

    def __init__(self) -> None:
        """Create an empty triage queue."""
        self._cases: list[FailureCase] = []

    def add_failure(
        self,
        case_id: str,
        summary: str,
        pile: str,
        transcript: str = "",
    ) -> FailureCase:
        """Add a failing case to the triage queue.

        Args:
            case_id: Stable identifier for the failing case.
            summary: One-line description of the failure.
            pile: Cluster/category label to sort the case under.
            transcript: Optional full transcript for later reading.

        Returns:
            The created :class:`FailureCase`.
        """
        case = FailureCase(
            case_id=case_id, summary=summary, pile=pile, transcript=transcript
        )
        self._cases.append(case)
        return case

    def piles(self) -> dict[str, int]:
        """Return ``pile -> count``, ordered by count descending.

        Ties are broken deterministically by pile label (ascending) so the same
        inputs always yield the same ordering.

        Returns:
            An insertion-ordered mapping with the biggest pile first.
        """
        counts: Counter[str] = Counter(c.pile for c in self._cases)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return dict(ordered)

    def biggest_pile(self) -> tuple[str, list[FailureCase]] | None:
        """Return the largest pile and its cases — the one to attack first.

        Returns:
            A ``(pile_label, cases)`` tuple for the biggest cluster, or ``None``
            when no failures have been triaged yet. Ties are broken by pile
            label (ascending) for determinism.
        """
        piles = self.piles()
        if not piles:
            return None
        top_label = next(iter(piles))
        return top_label, [c for c in self._cases if c.pile == top_label]

    def sample(self, pile: str, k: int = 5) -> list[FailureCase]:
        """Read up to ``k`` transcripts from a pile.

        Returns cases in insertion order (deterministic), so repeated sampling
        of the same pile yields the same first ``k`` cases.

        Args:
            pile: The pile label to sample from.
            k: Maximum number of cases to return.

        Returns:
            Up to ``k`` :class:`FailureCase` objects belonging to ``pile``.
        """
        if k <= 0:
            return []
        return [c for c in self._cases if c.pile == pile][:k]

    def from_evidence_corpus(self, corpus: Any) -> int:
        """Ingest failure clusters from an ``EvidenceCorpus``-like object.

        Defensively duck-typed: the corpus shape varies across the harness, so
        this reads whatever is plausibly present and degrades gracefully.

        Preference order:
          1. ``corpus.failure_clusters`` — each cluster contributes one
             :class:`FailureCase` per ``task_id``, sorted into a pile named by
             the cluster ``label`` (falling back to ``root_cause_summary``).
          2. ``corpus.entries`` — any entry whose ``passed``/``pass_fail`` flag
             is falsey becomes a case, piled by its ``root_cause``.

        Args:
            corpus: A ``continuous_evaluation_engine`` ``EvidenceCorpus`` or any
                object exposing the duck-typed attributes above. Mapping/dict
                shapes are also tolerated.

        Returns:
            The number of failure cases added to the queue.
        """
        added = 0
        clusters = _read_attr(corpus, "failure_clusters")
        if clusters:
            for cluster in clusters:
                label = (
                    _read_attr(cluster, "label")
                    or _read_attr(cluster, "root_cause_summary")
                    or "unknown"
                )
                summary = _read_attr(cluster, "root_cause_summary") or str(label)
                task_ids = _read_attr(cluster, "task_ids") or []
                for task_id in task_ids:
                    self.add_failure(
                        case_id=str(task_id), summary=str(summary), pile=str(label)
                    )
                    added += 1
            if added:
                return added

        entries = _read_attr(corpus, "entries") or []
        for entry in entries:
            passed = _read_attr(entry, "pass_fail")
            if passed is None:
                passed = _read_attr(entry, "passed")
            if passed:
                continue
            task_id = (
                _read_attr(entry, "task_id") or _read_attr(entry, "id") or "unknown"
            )
            root_cause = _read_attr(entry, "root_cause") or "unknown"
            self.add_failure(
                case_id=str(task_id),
                summary=str(root_cause),
                pile=str(root_cause),
                transcript=str(_read_attr(entry, "content") or ""),
            )
            added += 1
        return added


class ResearchLog:
    """Disconfirming-evidence belief log (CONCEPT:AU-AHE.harness.research-craft-discipline).

    Records evidence for and against hypotheses, with the disconfirming kind
    made first-class so it can't be quietly dropped. The log never reads a clock
    (timestamps are caller-supplied), so it is fully deterministic.
    """

    def __init__(self) -> None:
        """Create an empty research log."""
        self._entries: list[BeliefEntry] = []

    def record(
        self,
        hypothesis: str,
        evidence: str,
        *,
        supports: bool,
        timestamp: str = "",
    ) -> BeliefEntry:
        """Record a piece of evidence bearing on a hypothesis.

        Args:
            hypothesis: The working hypothesis the evidence relates to.
            evidence: The observed fact.
            supports: ``True`` if it confirms the hypothesis, ``False`` if it
                disconfirms it. Disconfirming facts should be recorded the
                moment they appear (Darwin's rule).
            timestamp: Optional caller-supplied timestamp string.

        Returns:
            The created :class:`BeliefEntry`.
        """
        entry = BeliefEntry(
            hypothesis=hypothesis,
            evidence=evidence,
            supports=supports,
            timestamp=timestamp,
        )
        self._entries.append(entry)
        return entry

    def disconfirming(self, hypothesis: str | None = None) -> list[BeliefEntry]:
        """Return disconfirming evidence — the facts we'd otherwise forget.

        Args:
            hypothesis: When given, restrict to evidence against that exact
                hypothesis; otherwise return all disconfirming entries.

        Returns:
            The matching disconfirming :class:`BeliefEntry` objects, in the
            order they were recorded.
        """
        return [
            e
            for e in self._entries
            if not e.supports and (hypothesis is None or e.hypothesis == hypothesis)
        ]

    def balance(self, hypothesis: str) -> dict[str, int]:
        """Tally supporting vs refuting evidence for a hypothesis.

        Args:
            hypothesis: The hypothesis to tally.

        Returns:
            ``{"supports": <count>, "refutes": <count>}`` for the hypothesis.
        """
        supports = 0
        refutes = 0
        for e in self._entries:
            if e.hypothesis != hypothesis:
                continue
            if e.supports:
                supports += 1
            else:
                refutes += 1
        return {"supports": supports, "refutes": refutes}

    def contested(self) -> list[str]:
        """Return hypotheses carrying BOTH supporting and disconfirming evidence.

        These are the live, genuinely-uncertain beliefs — the ones worth more
        investigation rather than premature commitment.

        Returns:
            Hypothesis strings with at least one of each evidence kind, sorted
            for deterministic output.
        """
        supported: set[str] = set()
        refuted: set[str] = set()
        for e in self._entries:
            (supported if e.supports else refuted).add(e.hypothesis)
        return sorted(supported & refuted)


def _read_attr(obj: Any, name: str) -> Any:
    """Read ``name`` from an object or mapping, returning ``None`` if absent.

    Tolerates both attribute-style (pydantic models/dataclasses) and
    mapping-style (dict) corpora so :meth:`FailureTriage.from_evidence_corpus`
    stays robust to shape drift.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
