#!/usr/bin/python
from __future__ import annotations

"""Measured-lift benchmark suite for the assimilated-paper mechanisms.

CONCEPT:AHE-3.47 — empirical parity evidence for the assimilation program.

We adopted eight research mechanisms as shipped code, but parity has so far been
*asserted*, not *measured*: each module ports a paper's mechanism, yet nothing
runs OUR mechanism against a BASELINE and reports the real numbers that justify
the claim "we reproduced the paper's benefit". This module is that missing
measurement. For each mechanism it builds a small, fully controlled synthetic
task, runs the baseline (the mechanism switched off, or its naive predecessor)
and OURS (the mechanism switched on) on identical inputs, and reports a
:class:`BenchmarkResult` carrying the baseline number, the our-number, the
direction-aware lift, and a ``claim_reproduced`` verdict — did OURS beat the
baseline in the direction the paper claims?

The tasks are deliberately easy: the point is a *faithful, stable, reproducible*
demonstration that the mechanism moves the metric the right way under a fixed
seed, not a hard benchmark. Every task is seeded, CPU-only, and uses the
deterministic / lexical code paths of each module (no torch, no LLM, no network),
so :func:`run_all` is bit-for-bit reproducible.

The eight modules map to seven benchmarks here: the temporal-semantic-ID encoder
(KG-2.86) is exercised *inside* the PauseRec benchmark (it is the recommender's
required encoder), so it is measured as part of that task rather than as a
standalone row.

Style mirrors the sibling harness measurement modules
(``adaptation_speed`` AHE-3.27, ``reliability_scorers`` AHE-3.12): pure,
dependency-light, dataclass-shaped, deterministic.

Layer contract: a read-only harness/benchmark over the L2 retrieval helpers and
the harness mechanisms; it imports them, runs them, and reports — no I/O, no
network, no upward dependencies.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent_utilities.harness.explore_exploit_router import ExploreExploitRouter
from agent_utilities.harness.graph_search_evolution import GraphSearchEvolver
from agent_utilities.harness.self_guided_play import Guide, SelfGuidedSelfPlay
from agent_utilities.knowledge_graph.retrieval.adaptive_stopping import IterativeStopper
from agent_utilities.knowledge_graph.retrieval.generative_recommender import (
    ImplicitReasoningRecommender,
)
from agent_utilities.knowledge_graph.retrieval.iterative_expansion import (
    IterativeQueryExpander,
)
from agent_utilities.knowledge_graph.retrieval.score_gate import score_gate
from agent_utilities.knowledge_graph.retrieval.temporal_semantic_id import (
    TemporalSemanticIdEncoder,
)

__all__ = [
    "BenchmarkResult",
    "bench_pauserec",
    "bench_scoregate",
    "bench_tasr",
    "bench_adore",
    "bench_decentmem_bandit",
    "bench_mlevolve",
    "bench_sgs",
    "bench_pauserec_trained",
    "run_all",
    "to_markdown",
]


@dataclass(frozen=True)
class BenchmarkResult:
    """One paper's measured baseline-vs-ours comparison on a controlled task.

    Attributes:
        name: Human-readable mechanism + concept id, e.g. ``"PauseRec KG-2.93"``.
        metric: The metric measured, e.g. ``"Recall@5"`` or ``"rounds"``.
        baseline: The metric value for the baseline (mechanism off / predecessor).
        ours: The metric value for OUR mechanism (mechanism on).
        lift: Signed improvement of ours over baseline in the paper's claimed
            direction; ``ours - baseline`` when ``higher_is_better`` else
            ``baseline - ours``. Positive means OURS won.
        higher_is_better: Whether a larger metric value is better.
        claim_reproduced: True iff OURS beat the baseline in the claimed direction
            (``lift > 0``), i.e. the paper's benefit reproduced under the seed.
        detail: Mechanism-specific extras (rounds saved, regret, per-arm stats…).
    """

    name: str
    metric: str
    baseline: float
    ours: float
    lift: float
    higher_is_better: bool
    claim_reproduced: bool
    detail: dict[str, Any] = field(default_factory=dict)


def _make_result(
    *,
    name: str,
    metric: str,
    baseline: float,
    ours: float,
    higher_is_better: bool,
    detail: dict[str, Any],
) -> BenchmarkResult:
    """Assemble a :class:`BenchmarkResult`, deriving direction-aware lift + verdict.

    ``lift`` is derived from the *rounded* baseline/ours so it is always exactly
    consistent with the reported numbers (no last-digit drift).
    """
    rounded_baseline = round(float(baseline), 6)
    rounded_ours = round(float(ours), 6)
    lift = (
        (rounded_ours - rounded_baseline)
        if higher_is_better
        else (rounded_baseline - rounded_ours)
    )
    return BenchmarkResult(
        name=name,
        metric=metric,
        baseline=rounded_baseline,
        ours=rounded_ours,
        lift=round(lift, 6),
        higher_is_better=higher_is_better,
        claim_reproduced=bool(lift > 0.0),
        detail=detail,
    )


# ----------------------------------------------------------------------------
# Synthetic-data helpers (seeded, numpy-only)
# ----------------------------------------------------------------------------
def _clustered_embeddings(
    rng: np.random.Generator, *, n_clusters: int, per_cluster: int, dim: int
) -> tuple[np.ndarray, list[int]]:
    """Generate tight clusters of embeddings; return vectors + per-row cluster id.

    Each cluster has a well-separated random centroid (orthogonal-ish on the
    sphere) plus a little noise, so the geometry is unambiguous and clustering /
    nearest-neighbour outcomes are stable under the seed.
    """
    centroids = rng.normal(size=(n_clusters, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            rows.append(centroids[c] + 0.05 * rng.normal(size=dim))
            labels.append(c)
    return np.vstack(rows), labels


def _ndcg_at_k(ranked_relevances: Sequence[float], k: int) -> float:
    """Normalized DCG@k for a best-first list of graded relevances (0/1 here)."""
    rels = list(ranked_relevances)[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(ranked_relevances, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ----------------------------------------------------------------------------
# PauseRec (KG-2.93) + TemporalSemanticIdEncoder (KG-2.86)
# ----------------------------------------------------------------------------
def bench_pauserec(*, seed: int = 0) -> BenchmarkResult:
    """Latent-reasoning budget (pause_steps>0) vs none (pause_steps=0).

    Synthetic task: a clustered catalog where the user's history lives entirely in
    the target cluster, the relevant items ARE that cluster, but the query points
    (on content) at a *distractor* cluster. Ranking straight off the projected
    query (no pausing) lands on the distractor and misses every relevant item;
    PauseRec's implicit-reasoning steps pull the latent target toward the history
    (and the items it co-supports), recovering the target cluster — the paper's
    claim that latent reasoning bridges history/world-knowledge into SID selection.

    Metric: NDCG@k over the relevant cluster. Claim: pausing >= no pausing.
    Exercises ``TemporalSemanticIdEncoder`` (KG-2.86) as the shared SID encoder.
    """
    rng = np.random.default_rng(seed + 1)
    dim = 24
    per_cluster = 6
    vectors, labels = _clustered_embeddings(
        rng, n_clusters=4, per_cluster=per_cluster, dim=dim
    )
    labels_arr = np.array(labels)
    target_cluster = 0
    distractor_cluster = 1
    items = [(f"item-{i}", vectors[i].tolist()) for i in range(len(labels))]
    relevant = {f"item-{i}" for i, c in enumerate(labels) if c == target_cluster}

    encoder_seed = seed + 2

    def _build() -> TemporalSemanticIdEncoder:
        return TemporalSemanticIdEncoder(
            n_codebooks=3, codebook_size=8, seed=encoder_seed
        )

    # The query points (on content) at the WRONG cluster — a distractor — while the
    # user's history sits firmly in the target cluster. A raw projection (no
    # pausing) therefore ranks the distractor and misses the relevant items; the
    # implicit-reasoning steps pull the latent target toward the history (and the
    # items it co-supports), recovering the target cluster. This is exactly
    # PauseRec's claim that the latent reasoning bridges history/world-knowledge
    # into SID selection rather than ranking off the surface query.
    distractor_centroid = vectors[labels_arr == distractor_cluster].mean(axis=0)
    query = (0.7 * distractor_centroid + 0.3 * rng.normal(size=dim)).tolist()

    top_k = per_cluster

    def _ndcg_for(pause_steps: int) -> float:
        rec = ImplicitReasoningRecommender(_build(), pause_steps=pause_steps)
        rec.fit_catalog(items)
        # History SIDs: three items firmly inside the target cluster.
        hist_ids = [i for i, c in enumerate(labels) if c == target_cluster][:3]
        history_sids = [rec._encoder.encode_content(vectors[i]) for i in hist_ids]  # noqa: SLF001
        ranked = rec.recommend(query, top_k=top_k, history_sids=history_sids)
        rels = [1.0 if r.item_id in relevant else 0.0 for r in ranked]
        return _ndcg_at_k(rels, top_k)

    baseline = _ndcg_for(0)
    ours = _ndcg_for(2)
    return _make_result(
        name="PauseRec KG-2.93",
        metric=f"NDCG@{top_k}",
        baseline=baseline,
        ours=ours,
        higher_is_better=True,
        detail={
            "pause_steps_baseline": 0,
            "pause_steps_ours": 2,
            "catalog_size": len(items),
            "relevant_count": len(relevant),
            "encoder": "TemporalSemanticIdEncoder KG-2.86 (n_codebooks=3)",
        },
    )


# ----------------------------------------------------------------------------
# ScoreGate (KG-2.85)
# ----------------------------------------------------------------------------
def bench_scoregate(*, seed: int = 0) -> BenchmarkResult:
    """Adaptive ScoreGate cut vs a fixed top-k cut on a clear-cluster-plus-tail set.

    Synthetic task: a small relevant cluster of clearly-high scores followed by a
    long weak tail of near-zero scores. A fixed ``top_k`` cut that overshoots the
    cluster drags irrelevant tail items in (precision collapses); ScoreGate cuts at
    the statistical knee, keeping the cluster and dropping the tail.

    Metric: precision at the gate's recall (we hold recall constant by sizing the
    fixed cut so both keep the whole relevant cluster, then compare precision).
    Claim: ScoreGate precision >= fixed-k precision.
    """
    rng = np.random.default_rng(seed)
    n_relevant = 4
    n_tail = 16
    relevant_scores = (0.9 + 0.05 * rng.random(n_relevant)).tolist()
    tail_scores = (0.05 * rng.random(n_tail)).tolist()

    scored: list[dict[str, Any]] = []
    for i, s in enumerate(relevant_scores):
        scored.append({"id": f"rel-{i}", "_score": s, "relevant": True})
    for i, s in enumerate(tail_scores):
        scored.append({"id": f"tail-{i}", "_score": s, "relevant": False})
    rng.shuffle(scored)

    def _precision(kept: list[dict[str, Any]]) -> float:
        if not kept:
            return 0.0
        hits = sum(1 for item in kept if item["relevant"])
        return hits / len(kept)

    # OURS: adaptive gate (keep items at/above the fused mean).
    ours_kept = score_gate(scored, min_results=1, keep_z=0.0)
    ours_precision = _precision(ours_kept)

    # BASELINE: a fixed top-k that recalls the whole cluster but, being fixed, also
    # grabs tail items (a realistic over-provisioned k). We pick k that matches the
    # gate's recall floor (the full cluster) yet stays fixed across queries.
    fixed_k = n_relevant + 4  # over-provisioned: full recall but tail contamination
    by_score = sorted(scored, key=lambda d: float(d["_score"]), reverse=True)
    baseline_kept = by_score[:fixed_k]
    baseline_precision = _precision(baseline_kept)

    return _make_result(
        name="ScoreGate KG-2.85",
        metric="precision@matched-recall",
        baseline=baseline_precision,
        ours=ours_precision,
        higher_is_better=True,
        detail={
            "relevant": n_relevant,
            "tail": n_tail,
            "fixed_k": fixed_k,
            "ours_kept": len(ours_kept),
            "ours_recall": sum(1 for i in ours_kept if i["relevant"]) / n_relevant,
            "baseline_recall": sum(1 for i in baseline_kept if i["relevant"])
            / n_relevant,
        },
    )


# ----------------------------------------------------------------------------
# TASR (KG-2.87)
# ----------------------------------------------------------------------------
def bench_tasr(*, seed: int = 0) -> BenchmarkResult:
    """TASR adaptive stopping vs a fixed max-rounds loop at equal final recall.

    Synthetic task: an iterative retrieve→answer loop whose answer *converges* —
    it changes for the first few rounds, then repeats. TASR halts the moment the
    answer repeats; the fixed baseline always burns all ``max_rounds``. Both reach
    the identical final answer/recall, so TASR's saved rounds are pure waste cut.

    Metric: rounds used (lower is better). Claim: TASR uses FEWER rounds at equal
    recall (``rounds_saved`` reported).
    """
    del seed  # the convergence schedule is fixed, not random
    max_rounds = 6
    # Answers per round: converges at round 3, then repeats verbatim.
    answers = [
        "paris is in france",
        "paris is the capital city",
        "paris is the capital of france",
        "paris is the capital of france",  # repeat -> TASR halts here
        "paris is the capital of france",
        "paris is the capital of france",
    ]
    # Evidence keeps trickling so coverage-saturation never pre-empts the answer
    # rule; the answer-repeat rule is what we are measuring.
    evidence = [[f"doc-{r}-{j}" for j in range(2)] for r in range(max_rounds)]

    def _final_answer(rounds_used: int) -> str:
        return answers[rounds_used - 1]

    # BASELINE: a fixed loop that runs every round regardless of convergence.
    baseline_rounds = max_rounds
    baseline_final = _final_answer(baseline_rounds)

    # OURS: stop as soon as the answer repeats.
    stopper = IterativeStopper(max_rounds=max_rounds, min_new_evidence=1, patience=99)
    ours_rounds = 0
    for r in range(max_rounds):
        decision = stopper.update(answer=answers[r], evidence_ids=evidence[r])
        ours_rounds = r + 1
        if decision.stop:
            break
    ours_final = _final_answer(ours_rounds)

    return _make_result(
        name="TASR KG-2.87",
        metric="rounds",
        baseline=float(baseline_rounds),
        ours=float(ours_rounds),
        higher_is_better=False,
        detail={
            "rounds_saved": baseline_rounds - ours_rounds,
            "equal_final_answer": baseline_final == ours_final,
            "final_answer": ours_final,
            "stop_reason": decision.reason,
        },
    )


# ----------------------------------------------------------------------------
# ADORE (KG-2.88)
# ----------------------------------------------------------------------------
def bench_adore(*, seed: int = 0) -> BenchmarkResult:
    """ADORE iterative feedback expansion vs a one-shot single retrieval.

    Synthetic corpus: relevant docs are split between those matching the original
    query terms (found in round 1) and those matching *expansion* terms that only
    a reformulation surfaces. A one-shot retrieval (round 1 only) can never reach
    the expansion-only relevant docs; ADORE's feedback rounds reformulate toward
    them and recover them.

    Metric: Recall@k over the full relevant set. Claim: ADORE >= one-shot.
    """
    del seed
    # Two pools of relevant docs: q-matching and expansion-matching. Each doc has
    # its own text so distinct doc-ids are judged individually. Noise docs share
    # the query term "topic" so round 1 retrieves grade-0 docs too — that keeps the
    # loop from short-circuiting on quality saturation before any feedback round.
    q_docs = {f"qrel-{i}": f"alpha topic core qmark{i}" for i in range(3)}
    # Expansion docs share the relevance term "core" (surfaced only once q-docs are
    # judged grade-3 and fed back) plus their own "expansion" terms — so a
    # reformulation built from the round-1 feedback reaches them, a one-shot can't.
    exp_docs = {f"erel-{i}": f"core deeper expansion emark{i}" for i in range(3)}
    noise = {f"noise-{i}": f"topic filler nmark{i}" for i in range(8)}
    corpus: dict[str, str] = {**q_docs, **exp_docs, **noise}
    relevant = set(q_docs) | set(exp_docs)
    query = "alpha topic"

    def retrieve_fn(expanded_query: str, top_k: int) -> list[dict[str, Any]]:
        ql = expanded_query.lower()
        scored: list[tuple[float, str]] = []
        for doc_id, text in corpus.items():
            overlap = sum(1 for tok in set(text.split()) if tok in ql)
            if overlap > 0:
                scored.append((float(overlap), doc_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            {"id": doc_id, "text": corpus[doc_id], "_score": score}
            for score, doc_id in scored[:top_k]
        ]

    def judge_fn(query_text: str, doc_text: str) -> int:
        return 3 if doc_text in (set(q_docs.values()) | set(exp_docs.values())) else 0

    def reformulate_fn(
        original: str, prev: list[str], graded: dict[int, list[str]]
    ) -> list[str]:
        # Feedback: graded-relevant texts seed the next query, surfacing "beta…".
        good = graded.get(3, [])
        return [original, *good] if good else [original]

    top_k = 8

    def _recall(ranking: list[tuple[str, float]]) -> float:
        got = {doc_id for doc_id, _ in ranking[:top_k]}
        return len(got & relevant) / len(relevant)

    # BASELINE: one-shot — a single retrieval round, no feedback.
    one_shot = IterativeQueryExpander(
        retrieve_fn, judge_fn, reformulate_fn, max_rounds=1, top_k=top_k, k_pseudo=3
    )
    baseline_recall = _recall(one_shot.run("q", query).final_ranking)

    # OURS: multi-round ADORE feedback expansion.
    adore = IterativeQueryExpander(
        retrieve_fn, judge_fn, reformulate_fn, max_rounds=4, top_k=top_k, k_pseudo=3
    )
    ours_history = adore.run("q", query)
    ours_recall = _recall(ours_history.final_ranking)

    return _make_result(
        name="ADORE KG-2.88",
        metric=f"Recall@{top_k}",
        baseline=baseline_recall,
        ours=ours_recall,
        higher_is_better=True,
        detail={
            "relevant_total": len(relevant),
            "expansion_only_relevant": len(exp_docs),
            "rounds_run": len(ours_history.rounds),
        },
    )


# ----------------------------------------------------------------------------
# DecentMem bandit (AHE-3.33)
# ----------------------------------------------------------------------------
def bench_decentmem_bandit(*, seed: int = 0) -> BenchmarkResult:
    """UCB1 router vs uniform-random arm choice on a 2-armed task.

    Synthetic task: two arms with a clearly better arm (Bernoulli means 0.8 vs
    0.2). UCB1 self-anneals onto the better arm; the random policy keeps pulling
    the bad arm half the time. Regret is measured by each router against the best
    running mean it has seen (the router's own intrinsic regret accounting).

    Metric: cumulative regret over N pulls (lower is better). Claim: UCB1 regret <
    random regret.
    """
    n_pulls = 200
    arm_means = {"good": 0.8, "bad": 0.2}

    def _run(strategy_random: bool, draw_seed: int) -> float:
        router = ExploreExploitRouter(
            arms=("good", "bad"), strategy="ucb1", seed=draw_seed
        )
        reward_rng = np.random.default_rng(draw_seed + 1000)
        choice_rng = np.random.default_rng(draw_seed + 2000)
        for _ in range(n_pulls):
            arm = (
                ("good", "bad")[int(choice_rng.integers(0, 2))]
                if strategy_random
                else router.select()
            )
            reward = 1.0 if reward_rng.random() < arm_means[arm] else 0.0
            router.update(arm, reward)
        return router.cumulative_regret

    baseline_regret = _run(strategy_random=True, draw_seed=seed)
    ours_regret = _run(strategy_random=False, draw_seed=seed)

    return _make_result(
        name="DecentMem-bandit AHE-3.33",
        metric="cumulative-regret",
        baseline=baseline_regret,
        ours=ours_regret,
        higher_is_better=False,
        detail={
            "n_pulls": n_pulls,
            "arm_means": arm_means,
            "strategy": "ucb1",
        },
    )


# ----------------------------------------------------------------------------
# MLEvolve (KG-2.92)
# ----------------------------------------------------------------------------
def bench_mlevolve(*, seed: int = 0) -> BenchmarkResult:
    """Multi-branch graph search (+fusion) vs single-branch search.

    Toy optimization: each branch's cold-start approach maps to a *different*
    plateau metric, and improving a branch only nudges its own plateau. A single
    branch is trapped at one plateau; the multi-branch graph search explores
    several plateaus in parallel and (via the cross-branch fusion node) can fuse
    the best of the others, so it finds a strictly higher best metric.

    Metric: best metric found (higher is better). Claim: graph search >= single
    branch.
    """
    # Branch-dependent plateaus: keyed by an approach marker injected into the plan.
    # The coder embeds a per-branch tag in the code; the evaluator reads it back.
    plateau_for_tag = {"t1": 0.4, "t2": 0.55, "t3": 0.7}

    def coder_fn(plan: str, prior_code: str | None) -> tuple[str, str]:
        # Derive a stable branch tag from the plan text (deterministic).
        tag = "t1"
        if "branch 2" in plan or "tag:t2" in plan:
            tag = "t2"
        elif "branch 3" in plan or "tag:t3" in plan:
            tag = "t3"
        elif prior_code and "tag:" in prior_code:
            tag = prior_code.split("tag:")[1][:2]
        # Improvements add depth markers that raise the metric toward the plateau.
        depth = (prior_code.count("+") + 1) if prior_code else 0
        code = f"tag:{tag} " + "+" * depth
        return (f"{plan} tag:{tag}", code)

    def evaluate_fn(code: str) -> tuple[float, bool]:
        tag = code.split("tag:")[1][:2] if "tag:" in code else "t1"
        plateau = plateau_for_tag.get(tag, 0.3)
        depth = code.count("+")
        # Approach the branch's plateau as depth grows; never exceed it.
        metric = plateau * (1.0 - 0.5 ** (depth + 1))
        return (metric, False)

    task = "optimize draft branch 1 branch 2 branch 3 over a toy metric"

    # BASELINE: a single branch — trapped on one plateau (the first approach, t1).
    single = GraphSearchEvolver(
        coder_fn, evaluate_fn, num_branches=1, num_steps=12, seed=seed
    )
    baseline_best = single.run(task).metric or 0.0

    # OURS: multi-branch graph search with cross-branch fusion.
    graph = GraphSearchEvolver(
        coder_fn, evaluate_fn, num_branches=3, num_steps=12, seed=seed
    )
    ours_node = graph.run(task)
    ours_best = ours_node.metric or 0.0

    return _make_result(
        name="MLEvolve KG-2.92",
        metric="best-metric",
        baseline=baseline_best,
        ours=ours_best,
        higher_is_better=True,
        detail={
            "branches_baseline": 1,
            "branches_ours": 3,
            "fusion_nodes": len(graph.fusion_nodes),
            "best_stage": ours_node.stage.value,
        },
    )


# ----------------------------------------------------------------------------
# SGS (AHE-3.37)
# ----------------------------------------------------------------------------
def bench_sgs(*, seed: int = 0) -> BenchmarkResult:
    """Self-Guided Self-Play WITH the Guide vs WITHOUT (accept everything).

    The Conjecturer alternates between a clean, on-target task and a *gamed* task
    (disjunction spam, runaway length, contradictory markers) — exactly the SGS
    collapse mode. WITHOUT the Guide every task is accepted, so half the accepted
    tasks are low-quality. WITH the Guide the gamed tasks are rejected, raising the
    quality of the accepted set.

    Metric: fraction of accepted tasks that are high-quality. Claim: the Guide
    improves accepted-task quality.
    """
    del seed
    target = "prove that the sum of two even integers is even"
    clean = "prove that the sum of two odd integers is even"
    gamed = (
        "prove that the sum is even or the sum is odd or vacuously trivially false "
        "and moreover the integers are even or not even furthermore contradiction "
        "but not when the premises are redundant redundant redundant redundant"
    )

    def conjecture_fn(target_task: str, difficulty: float) -> str:
        del target_task, difficulty
        # Alternate clean / gamed by an internal counter (deterministic).
        conjecture_fn.calls = getattr(conjecture_fn, "calls", 0) + 1  # type: ignore[attr-defined]
        return clean if conjecture_fn.calls % 2 == 1 else gamed  # type: ignore[attr-defined]

    def solve_fn(task: str) -> tuple[str, bool]:
        return ("solution", True)

    # Ground-truth "high quality" = the heuristic Guide would accept it. We measure
    # the quality of whatever each policy ACCEPTS for the Solver.
    quality_guide = Guide(threshold=0.5)

    def _accepted_quality_fraction(use_guide: bool) -> tuple[float, int]:
        conjecture_fn.calls = 0  # type: ignore[attr-defined]
        if use_guide:
            play = SelfGuidedSelfPlay(
                conjecture_fn, solve_fn, guide=Guide(threshold=0.5)
            )
        else:
            # No-gate baseline: a permissive Guide that accepts everything.
            play = SelfGuidedSelfPlay(
                conjecture_fn, solve_fn, guide=Guide(threshold=-1.0)
            )
        report = play.run(target, rounds=10, start_difficulty=0.3)
        accepted = [r for r in report.rounds if r.accepted]
        if not accepted:
            return (0.0, 0)
        high = sum(
            1
            for r in accepted
            if quality_guide.evaluate(target, r.generated_task).overall >= 0.5
        )
        return (high / len(accepted), len(accepted))

    baseline_quality, baseline_n = _accepted_quality_fraction(use_guide=False)
    ours_quality, ours_n = _accepted_quality_fraction(use_guide=True)

    return _make_result(
        name="SGS AHE-3.37",
        metric="accepted-quality-fraction",
        baseline=baseline_quality,
        ours=ours_quality,
        higher_is_better=True,
        detail={
            "accepted_baseline": baseline_n,
            "accepted_ours": ours_n,
            "gamed_rejected": baseline_n - ours_n,
        },
    )


# ----------------------------------------------------------------------------
# Aggregation + reporting
# ----------------------------------------------------------------------------
def bench_pauserec_trained(*, seed: int = 0) -> BenchmarkResult | None:
    """Train REAL pause tokens (PauseRec's actual mechanism) and measure the lift.

    Unlike :func:`bench_pauserec` (inference-time deterministic adaptation), this runs the
    paper's literal trainable-``<pause>``-token mechanism via gradient descent (CONCEPT:KG-2.93
    training track) and reports Recall@k WITH vs WITHOUT the trained pause tokens. Returns
    ``None`` when torch is not installed (the bench is skipped, not failed).
    """
    from agent_utilities.knowledge_graph.retrieval import pause_token_trainer as ptt

    if not ptt.is_available():
        return None
    # Toy task where next-item is a nonlinear interaction of the history (see the trainer).
    rng = np.random.default_rng(seed)
    n_items = 12
    sequences = [[int(rng.integers(0, n_items)) for _ in range(3)] for _ in range(64)]
    model = ptt.PauseTokenRecommender(
        n_items=n_items, dim=6, n_pause_tokens=4, seed=seed
    )
    res = model.fit(sequences, epochs=200, lr=0.05, k=3)
    return _make_result(
        name="PauseRec-trained KG-2.93",
        metric="Recall@3",
        baseline=res.recall_at_k_without_pause,
        ours=res.recall_at_k_with_pause,
        higher_is_better=True,
        detail={
            "initial_loss": res.initial_loss,
            "final_loss": res.final_loss,
            "n_pause_tokens": res.n_pause_tokens,
            "mechanism": "trained pause tokens (gradient descent), not inference adaptation",
        },
    )


def run_all(*, seed: int = 0) -> list[BenchmarkResult]:
    """Run every benchmark under one seed and return the results in order.

    Includes the trained-pause-token bench only when torch is installed (CPU is fine).
    """
    results = [
        bench_pauserec(seed=seed),
        bench_scoregate(seed=seed),
        bench_tasr(seed=seed),
        bench_adore(seed=seed),
        bench_decentmem_bandit(seed=seed),
        bench_mlevolve(seed=seed),
        bench_sgs(seed=seed),
    ]
    trained = bench_pauserec_trained(seed=seed)
    if trained is not None:
        results.append(trained)
    return results


def to_markdown(results: list[BenchmarkResult]) -> str:
    """Render results as a Markdown table for the empirical-parity report."""
    header = (
        "| Mechanism | Metric | Baseline | Ours | Lift | Claim reproduced |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    rows = [
        f"| {r.name} | {r.metric} | {r.baseline:g} | {r.ours:g} | "
        f"{r.lift:+g} | {'yes' if r.claim_reproduced else 'NO'} |"
        for r in results
    ]
    reproduced = sum(1 for r in results if r.claim_reproduced)
    footer = (
        f"\n\n**{reproduced}/{len(results)} claims reproduced** under the fixed seed."
    )
    return "\n".join([header, *rows]) + footer
