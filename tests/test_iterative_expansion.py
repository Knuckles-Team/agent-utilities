"""Unit tests for the ADORE-style iterative query expander (CONCEPT:AU-KG.query.adore-concept-expansion).

Fully deterministic: a fake corpus + keyword-overlap retriever + keyword judge +
fixed-pseudo-passage reformulator. No LLM, no network.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.retrieval.iterative_expansion import (
    IterativeQueryExpander,
    build_expanded_query,
)

# Fake corpus: high-grade docs mention the query terms strongly; low-grade ones don't.
CORPUS: dict[str, str] = {
    "d_perfect_1": "vector embedding similarity search relevance feedback ranking",
    "d_perfect_2": "relevance feedback vector embedding retrieval similarity ranking",
    "d_good": "embedding similarity for documents and ranking heuristics",
    "d_weak": "a short note about unrelated cooking recipes and gardening",
    "d_offtopic": "weather forecast and sports results for the weekend",
}

QUERY = "vector embedding similarity relevance feedback"
QUERY_TERMS = set(QUERY.split())


def _overlap(text: str) -> int:
    return len(QUERY_TERMS & set(text.lower().split()))


def retrieve_fn(expanded_query: str, top_k: int) -> list[dict[str, Any]]:
    """Return corpus docs scored by keyword overlap with the original query terms."""
    nodes = [
        {"id": doc_id, "text": text, "_score": float(_overlap(text))}
        for doc_id, text in CORPUS.items()
    ]
    nodes.sort(key=lambda n: n["_score"], reverse=True)
    return nodes[:top_k]


def judge_fn(query: str, doc_text: str) -> int:
    """Grade 0..3 by keyword overlap (UMBRELA-style stub)."""
    return min(3, _overlap(doc_text))


def reformulate_fn(
    query: str,
    prev_pseudo_passages: list[str],
    graded_doc_texts_by_grade: dict[int, list[str]],
) -> list[str]:
    """Append a fixed pseudo-passage; feedback is accepted but ignored deterministically."""
    return [query, "auxiliary pseudo passage about embeddings and ranking"]


def _build_expander(**kwargs: Any) -> IterativeQueryExpander:
    return IterativeQueryExpander(
        retrieve_fn=retrieve_fn,
        judge_fn=judge_fn,
        reformulate_fn=reformulate_fn,
        max_rounds=5,
        judge_depth=10,
        top_k=20,
        **kwargs,
    )


def test_build_expanded_query_repeats_query_and_appends_passages() -> None:
    expanded = build_expanded_query(
        "cat", ["a much longer pseudo passage here"] * 3, alpha=5
    )
    # Query token repeated at least twice (alpha-repetition against long passages).
    assert expanded.split().count("cat") >= 2
    # Pseudo passages appended verbatim.
    assert "pseudo passage" in expanded


def test_build_expanded_query_no_passages_returns_query_once() -> None:
    assert build_expanded_query("hello world", [], alpha=5).strip() == "hello world"


def test_runs_multiple_rounds_then_stops() -> None:
    history = _build_expander().run("q1", QUERY)
    # Loop runs and terminates within the round budget.
    assert 1 <= len(history.rounds) <= 5
    # Round indices are sequential starting at 1.
    assert [r.round_index for r in history.rounds] == list(
        range(1, len(history.rounds) + 1)
    )


def test_dedup_no_doc_judged_twice() -> None:
    history = _build_expander().run("q1", QUERY)
    all_judged: list[str] = []
    for rd in history.rounds:
        all_judged.extend(rd.judged.keys())
    assert len(all_judged) == len(set(all_judged)), (
        "a doc was judged in more than one round"
    )


def test_final_ranking_best_first_with_high_grade_docs_first() -> None:
    history = _build_expander().run("q1", QUERY)
    ranking = history.final_ranking
    assert ranking, "expected a non-empty final ranking"
    ranked_ids = [doc_id for doc_id, _ in ranking]
    # The two perfect docs rank ahead of the off-topic doc.
    assert ranked_ids.index("d_perfect_1") < ranked_ids.index("d_offtopic")
    assert ranked_ids.index("d_perfect_2") < ranked_ids.index("d_offtopic")
    # Scores are non-increasing once grade tiers are equal at the top.
    top_score = ranking[0][1]
    assert all(top_score >= s for _, s in ranking)


def test_quality_saturation_stops_early() -> None:
    """A corpus where every retrieved doc grades 3 should stop after round 1."""
    perfect_corpus = {
        "a": QUERY + " extra",
        "b": QUERY + " more",
    }

    def perfect_retrieve(expanded_query: str, top_k: int) -> list[dict[str, Any]]:
        return [
            {"id": k, "text": v, "_score": float(_overlap(v))}
            for k, v in perfect_corpus.items()
        ][:top_k]

    expander = IterativeQueryExpander(
        retrieve_fn=perfect_retrieve,
        judge_fn=judge_fn,
        reformulate_fn=reformulate_fn,
        max_rounds=5,
    )
    history = expander.run("q1", QUERY)
    # All judged grade 3 -> quality saturation after the first round.
    assert len(history.rounds) == 1
    assert history.rounds[0].judged
    assert all(g == 3 for g in history.rounds[0].judged.values())


def test_default_reformulator_uses_query_as_passage() -> None:
    expander = IterativeQueryExpander(
        retrieve_fn=retrieve_fn, judge_fn=judge_fn, reformulate_fn=None, max_rounds=2
    )
    history = expander.run("q1", QUERY)
    assert history.rounds[0].pseudo_passages == [QUERY]
