"""Iterative, relevance-feedback query expansion over the knowledge graph.

CONCEPT:KG-2.88 — assimilated from ADORE: "Iterative Query Expansion with
Retrieval-Grounded Relevance Feedback" (github.com/aminbigdeli/ADORE).

ADORE runs a bounded loop of *reformulate → retrieve → judge*: each round a
reformulator emits pseudo-passages, an alpha-repetition query is built from them
(term-frequency-balanced against the much longer passages), the retriever returns
the top-k, newly-seen documents are graded 0..3 (UMBRELA-style), and the graded
evidence conditions the next round's reformulation. The loop terminates on a round
budget, on coverage saturation (no new documents — delegated to
:class:`IterativeStopper`), or on quality saturation (everything judged maximally
relevant). The accumulated, best-per-document ranking is the output.

This module is the graph-native port. It is **fully dependency-injected** —
callers supply ``retrieve_fn``, ``judge_fn`` and an optional ``reformulate_fn`` —
so the whole policy is unit-testable with no LLM and no network. Termination is
delegated to the sibling :class:`~.adaptive_stopping.IterativeStopper`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .adaptive_stopping import IterativeStopper

# (expanded_query, top_k) -> result nodes, each a dict with an id + a text-ish field.
RetrieveFn = Callable[[str, int], list[dict[str, Any]]]
# (query, doc_text) -> graded relevance 0..3 (UMBRELA-style).
JudgeFn = Callable[[str, str], int]
# (original_query, prev_pseudo_passages, graded_doc_texts_by_grade) -> new pseudo-passages.
ReformulateFn = Callable[[str, list[str], dict[int, list[str]]], list[str]]


def build_expanded_query(
    query: str, pseudo_passages: list[str], *, alpha: int = 5
) -> str:
    """Build a retrieval query via ADORE/MuGI alpha-repetition.

    The original ``query`` is repeated enough times that its term frequencies
    stay competitive against the much longer concatenated ``pseudo_passages``::

        reps = max(1, (docs_len // query_len) // alpha)
        expanded = (query + " ") * reps + pseudo_passages

    Character lengths and integer floor division match ADORE faithfully. With no
    pseudo-passages this returns the (stripped) query repeated once.
    """
    q0 = query.strip()
    all_pseudo = " ".join(p.strip() for p in pseudo_passages if p and p.strip())

    query_len = len(q0)
    docs_len = len(all_pseudo)
    if query_len > 0:
        reps = max(1, (docs_len // query_len) // max(alpha, 1))
    else:
        reps = 1

    return (q0 + " ") * reps + all_pseudo


@dataclass
class RoundRecord:
    """Trace of a single reformulate → retrieve → judge round."""

    round_index: int
    pseudo_passages: list[str]
    expanded_query: str
    retrieved_ids: list[str]
    judged: dict[str, int]  # doc_id -> grade (only docs judged this round)
    new_doc_count: int


@dataclass
class SearchHistory:
    """Per-query state: every round plus the accumulated best-first ranking."""

    query_id: str
    query: str
    rounds: list[RoundRecord] = field(default_factory=list)
    final_ranking: list[tuple[str, float]] = field(
        default_factory=list
    )  # (doc_id, score)


class IterativeQueryExpander:
    """Run the bounded ADORE relevance-feedback loop for a single query.

    All external effects are injected, so the loop is deterministic and testable
    without an LLM or a retrieval backend:

    - ``retrieve_fn(expanded_query, top_k)`` returns candidate node dicts.
    - ``judge_fn(query, doc_text)`` grades a document 0..3 against the query.
    - ``reformulate_fn(query, prev_passages, graded_texts_by_grade)`` proposes new
      pseudo-passages; round 1 receives empty feedback (zero-shot). When omitted,
      pseudo-passages default to the original query.
    """

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        judge_fn: JudgeFn,
        reformulate_fn: ReformulateFn | None = None,
        *,
        max_rounds: int = 5,
        k_pseudo: int = 3,
        judge_depth: int = 10,
        top_k: int = 20,
        alpha: int = 5,
        text_keys: tuple[str, ...] = ("text", "content", "summary", "snippet", "name"),
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.judge_fn = judge_fn
        self.reformulate_fn = reformulate_fn
        self.max_rounds = max_rounds
        self.k_pseudo = k_pseudo
        self.judge_depth = judge_depth
        self.top_k = top_k
        self.alpha = alpha
        self.text_keys = text_keys

    # ------------------------------------------------------------------ helpers

    def _doc_id(self, node: dict[str, Any]) -> str | None:
        """Extract a stable id from a result node, if present."""
        nid = node.get("id")
        return str(nid) if nid is not None else None

    def _doc_text(self, node: dict[str, Any]) -> str:
        """Extract the first present text-ish field from a result node."""
        for key in self.text_keys:
            val = node.get(key)
            if val:
                return str(val)
        return ""

    def _score(self, node: dict[str, Any], rank: int, count: int) -> float:
        """Retrieval score for a node: explicit ``_score``/``score`` or a rank fallback.

        The rank fallback decreases with position so retrieval order is preserved
        even when the backend returns no numeric scores.
        """
        for key in ("_score", "score"):
            val = node.get(key)
            if val is not None:
                return float(val)
        return float(count - rank) / float(count) if count else 0.0

    def _reformulate(
        self,
        query: str,
        prev_passages: list[str],
        graded_by_grade: dict[int, list[str]],
    ) -> list[str]:
        """Produce pseudo-passages for a round, honouring ``k_pseudo``."""
        if self.reformulate_fn is None:
            return [query]
        passages = self.reformulate_fn(query, prev_passages, graded_by_grade)
        passages = [p for p in passages if p and p.strip()]
        if not passages:
            passages = [query]
        return passages[: self.k_pseudo]

    # --------------------------------------------------------------------- main

    def run(self, query_id: str, query: str) -> SearchHistory:
        """Run the loop and return the completed :class:`SearchHistory`."""
        history = SearchHistory(query_id=query_id, query=query)
        stopper = IterativeStopper(max_rounds=self.max_rounds, min_new_evidence=1)

        seen: set[str] = set()
        best_score: dict[str, float] = {}
        best_grade: dict[str, int] = {}
        # Graded doc texts partitioned by grade, accumulated across rounds.
        graded_by_grade: dict[int, list[str]] = {g: [] for g in range(4)}
        pseudo_passages: list[str] = []

        for round_index in range(1, self.max_rounds + 1):
            pseudo_passages = self._reformulate(query, pseudo_passages, graded_by_grade)
            expanded = build_expanded_query(query, pseudo_passages, alpha=self.alpha)
            results = self.retrieve_fn(expanded, self.top_k)

            retrieved_ids: list[str] = []
            new_docs: list[
                tuple[str, str]
            ] = []  # (doc_id, text), first-seen this round
            count = len(results)
            for rank, node in enumerate(results):
                doc_id = self._doc_id(node)
                if doc_id is None:
                    continue
                retrieved_ids.append(doc_id)
                score = self._score(node, rank, count)
                if score > best_score.get(doc_id, float("-inf")):
                    best_score[doc_id] = score
                if doc_id not in seen:
                    new_docs.append((doc_id, self._doc_text(node)))

            # Judge up to judge_depth NEW documents; dedup guarantees no double-judge.
            judged: dict[str, int] = {}
            for doc_id, text in new_docs[: self.judge_depth]:
                grade = int(self.judge_fn(query, text))
                grade = max(0, min(3, grade))
                judged[doc_id] = grade
                seen.add(doc_id)
                if grade > best_grade.get(doc_id, -1):
                    best_grade[doc_id] = grade
                graded_by_grade.setdefault(grade, []).append(text)

            history.rounds.append(
                RoundRecord(
                    round_index=round_index,
                    pseudo_passages=list(pseudo_passages),
                    expanded_query=expanded,
                    retrieved_ids=retrieved_ids,
                    judged=judged,
                    new_doc_count=len(new_docs),
                )
            )

            # Quality saturation: everything judged this round is maximally relevant.
            if judged and all(grade == 3 for grade in judged.values()):
                break

            # Coverage / budget saturation is delegated to the shared stopper.
            answer = " ".join(retrieved_ids[: min(len(retrieved_ids), 10)])
            decision = stopper.update(answer=answer, evidence_ids=retrieved_ids)
            if decision.stop:
                break

        # Accumulated best-first ranking: best grade first, retrieval score as tiebreak.
        ranked = sorted(
            best_score.keys(),
            key=lambda d: (best_grade.get(d, 0), best_score.get(d, 0.0)),
            reverse=True,
        )
        history.final_ranking = [(d, best_score.get(d, 0.0)) for d in ranked]
        return history


__all__ = [
    "RetrieveFn",
    "JudgeFn",
    "ReformulateFn",
    "build_expanded_query",
    "RoundRecord",
    "SearchHistory",
    "IterativeQueryExpander",
]
