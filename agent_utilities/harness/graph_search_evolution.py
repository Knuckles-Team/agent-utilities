#!/usr/bin/python
from __future__ import annotations

"""Monte-Carlo Graph Search for self-evolving code/solution discovery.

CONCEPT:KG-2.92 — Progressive Monte-Carlo Graph Search (MCGS) with cross-branch
fusion edges, retrospective memory, and decoupled hierarchical planning/coding.

Assimilated from MLEvolve (arXiv:2606.06473, "MLEvolve: A Self-Evolving Framework
for Automated Machine Learning Algorithm Discovery"). AU's prior evolutionary
search (AHE-3.2 ``VariantPool``, AHE-3.3 regressor evolution, KG-2.69 program
synthesis) searched *tree*- or population-structured spaces where each branch is
information-isolated: a strong solution found in one branch can never seed
another. MLEvolve's central advance is to make the search a **directed graph**:
beyond the parent→child *primary* edges that carry credit assignment, nodes gain
*reference* edges that import knowledge from strong nodes in **other** branches
(``Section 3.2``). When a branch stagnates, a fusion node is created whose
reference set is the best nodes of the other branches — cross-branch knowledge
flow that a tree cannot express.

This module ports that mechanism in a dependency-injected, network-free form so
it is unit-testable with stub code-generation and evaluation callables. It also
ports the three other MLEvolve pillars:

* **Retrospective Memory** (``Section 3.3``): a static :class:`ColdStartKB` of
  task-category → recommended approaches that seeds an otherwise cold task, plus
  a dynamic :class:`GlobalCodeMemory` that accumulates per-attempt records with a
  ``-1 / 0 / 1`` reward label and retrieves similar past experience (lexical by
  default, or an injected embedding similarity).
* **Hierarchical Planning + Adaptive Code Generation** (``Section 3.4``): the
  planner reasons in free text and is then refined against *similar success*
  records from memory (decoupled planning↔coding), and :func:`select_coding_mode`
  chooses among full-rewrite / stepwise / diff modes by the current search state.
* **Progressive exploration schedule** (``Section 3.2.2``): :func:`exploration_schedule`
  decays the UCT exploration constant ``c`` piecewise from broad exploration to
  focused exploitation over the search horizon.

All randomness is seeded and varies deterministically by step index, so a fixed
``seed`` reproduces an identical search.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random

__all__ = [
    "Stage",
    "SearchNode",
    "exploration_schedule",
    "MemRecord",
    "GlobalCodeMemory",
    "ColdStartKB",
    "select_coding_mode",
    "GraphSearchEvolver",
]


class Stage(StrEnum):
    """The MCGS expansion stage that produced a node (MLEvolve ``Section 3.2.2``)."""

    ROOT = "root"
    DRAFT = "draft"
    IMPROVE = "improve"
    DEBUG = "debug"
    FUSION = "fusion"
    EVOLUTION = "evolution"


@dataclass
class SearchNode:
    """One candidate solution in the search graph (MLEvolve ``search_node.py``).

    Primary edges are encoded by ``parent_id`` (used for UCT selection and reward
    backpropagation). ``reference_ids`` encode the *reference* edges that make the
    structure a graph rather than a tree: cross-branch knowledge flow that does
    **not** participate in credit assignment.
    """

    node_id: str
    code: str
    plan: str
    metric: float | None  # task performance; None = not yet evaluated
    stage: Stage
    branch_id: int
    parent_id: str | None = None
    visits: int = 0
    total_reward: float = 0.0
    is_buggy: bool = False
    #: Cross-branch reference edges — the graph, not a tree (MLEvolve Eq. E_ref).
    reference_ids: list[str] = field(default_factory=list)

    def uct(self, parent_visits: int, c: float) -> float:
        """Upper Confidence bound for Trees: exploitation + ``c``·exploration.

        Returns ``+inf`` for an unvisited node so it is always tried first
        (MLEvolve ``SearchNode.uct_value``). Otherwise the score is the mean
        reward plus the exploration bonus ``c·sqrt(ln(N)/n)`` where ``N`` is the
        parent visit count and ``n`` is this node's visit count.
        """
        if self.visits == 0:
            return math.inf
        exploitation = self.total_reward / self.visits
        # ``parent_visits`` is clamped to >= 1 so ln() is well-defined for a
        # freshly-rooted branch whose parent has a single visit.
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return exploitation + exploration


def exploration_schedule(
    step: int,
    total_steps: int,
    *,
    initial_c: float = 1.4,
    lower_bound: float = 0.2,
    t1_frac: float = 0.3,
    t2_frac: float = 0.7,
) -> float:
    """Piecewise progressive exploration constant (MLEvolve ``node_selection.py``).

    Transitions the search from broad exploration to focused exploitation:

    * constant ``initial_c`` until ``t1 = t1_frac·total_steps``;
    * linear decay from ``initial_c`` down to ``lower_bound`` between ``t1`` and
      ``t2 = t2_frac·total_steps``;
    * constant ``lower_bound`` after ``t2``.

    The result is monotone non-increasing in ``step``. ``total_steps <= 0`` (and a
    degenerate ``t1 == t2``) collapse cleanly to ``lower_bound`` past the knee.
    """
    if total_steps <= 0:
        return lower_bound
    t1 = t1_frac * total_steps
    t2 = t2_frac * total_steps
    if step <= t1:
        return initial_c
    if step >= t2 or t2 <= t1:
        return lower_bound
    decay_progress = (step - t1) / (t2 - t1)
    return initial_c - (initial_c - lower_bound) * decay_progress


@dataclass
class MemRecord:
    """A single retrospective-memory experience record (MLEvolve ``record.py``).

    ``label`` is the reward category used to filter retrieval:
    ``-1`` failed / worsened, ``0`` tried but no improvement, ``1`` success.
    """

    record_id: str
    plan: str
    code_summary: str
    stage: Stage
    label: int  # -1 failed, 0 tried-no-improve, 1 success
    metric: float | None = None


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric word set — the lexical retrieval primitive."""
    token = []
    tokens: set[str] = set()
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            tokens.add("".join(token))
            token = []
    if token:
        tokens.add("".join(token))
    return tokens


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity over word sets — the default (FAISS-free) similarity."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class GlobalCodeMemory:
    """Dynamic experience replay over code attempts (MLEvolve ``global_memory.py``).

    Accumulates :class:`MemRecord` entries during search and retrieves the most
    similar prior experience to guide the next plan. Retrieval defaults to a
    lexical (Jaccard) similarity so the module needs neither FAISS nor an
    embedding model; an embedding ``similarity_fn(query, candidate) -> float`` may
    be injected to replace it.
    """

    def __init__(
        self, *, similarity_fn: Callable[[str, str], float] | None = None
    ) -> None:
        self._similarity = similarity_fn or _jaccard
        self._records: dict[str, MemRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[MemRecord]:
        """All stored records, in insertion order."""
        return list(self._records.values())

    def save(self, record: MemRecord) -> None:
        """Persist a record, idempotent by ``record_id`` (first write wins)."""
        if record.record_id in self._records:
            return
        self._records[record.record_id] = record

    def retrieve(
        self,
        query_plan: str,
        *,
        k: int = 3,
        label: int | None = None,
        stage: Stage | None = None,
        min_similarity: float = 0.0,
    ) -> list[MemRecord]:
        """Return up to ``k`` records most similar to ``query_plan``.

        Records are first filtered by ``label`` and/or ``stage`` (when given),
        then ranked by descending similarity of their ``plan`` to ``query_plan``;
        records below ``min_similarity`` are dropped. Ties break by ``record_id``
        for determinism.
        """
        scored: list[tuple[float, str, MemRecord]] = []
        for rec in self._records.values():
            if label is not None and rec.label != label:
                continue
            if stage is not None and rec.stage != stage:
                continue
            sim = self._similarity(query_plan, rec.plan)
            if sim < min_similarity:
                continue
            scored.append((sim, rec.record_id, rec))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [rec for _, _, rec in scored[:k]]


class ColdStartKB:
    """Static domain knowledge base for cold-start seeding (MLEvolve ``Section 3.3.1``).

    Maps a task *category* (matched by keyword against the task text) to a list of
    recommended approaches that seed the initial drafts of an otherwise cold task,
    mitigating the high cold-start error rate of relying on LLM priors alone.
    """

    #: A small, real default table keyed by task category. Each category lists the
    #: model families / techniques MLEvolve curates for that domain (``coldstart/``).
    DEFAULT_TABLE: dict[str, list[str]] = {
        "image": [
            "fine-tune a pretrained ResNet/ConvNeXt backbone",
            "apply strong data augmentation (mixup, cutmix, random crop)",
            "ensemble multiple checkpoints with test-time augmentation",
        ],
        "text": [
            "fine-tune a pretrained transformer encoder (DeBERTa/RoBERTa)",
            "add TF-IDF lexical features alongside embeddings",
            "use stratified k-fold cross-validation with early stopping",
        ],
        "tabular": [
            "gradient-boosted trees (XGBoost/LightGBM/CatBoost)",
            "careful target encoding of high-cardinality categoricals",
            "k-fold out-of-fold stacking of diverse base learners",
        ],
        "audio": [
            "convert to log-mel spectrograms and use a CNN classifier",
            "apply SpecAugment time/frequency masking",
            "pretrain on a larger corpus then fine-tune",
        ],
        "timeseries": [
            "engineer lag and rolling-window features",
            "gradient-boosted trees on windowed features",
            "respect temporal ordering in validation splits",
        ],
    }

    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self._table = table if table is not None else dict(self.DEFAULT_TABLE)

    def recommend(self, task_text: str) -> list[str]:
        """Return recommended approaches for every category whose key appears.

        Categories are matched by substring against the lowercased ``task_text``;
        the union of their approaches (de-duplicated, order-preserving) is
        returned. An empty list means the task did not match any known category.
        """
        text = task_text.lower()
        out: list[str] = []
        seen: set[str] = set()
        for category, approaches in self._table.items():
            if category in text:
                for approach in approaches:
                    if approach not in seen:
                        seen.add(approach)
                        out.append(approach)
        return out


def select_coding_mode(*, stagnating: bool, code_size: int, has_error: bool) -> str:
    """Adaptive coding-mode dispatch (MLEvolve ``Section 3.4.2``).

    Picks the code-generation granularity from the current search state:

    * ``"stepwise"`` — module-by-module regeneration, used on an execution error
      or retry (recover by re-deriving the broken module rather than diffing it);
    * ``"diff"`` — targeted localized edits, used when a working solution is large
      (``code_size > 2000``) or the branch is stagnating and needs controlled,
      stable refinement;
    * ``"single"`` — full rewrite from scratch, the default initial-draft mode.
    """
    if has_error:
        return "stepwise"
    if stagnating or code_size > 2000:
        return "diff"
    return "single"


class GraphSearchEvolver:
    """Monte-Carlo Graph Search with cross-branch fusion (MLEvolve ``agent_search.py``).

    The search is dependency-injected and network-free:

    * ``coder_fn(plan, prior_code) -> (plan_text, code)`` generates a child's plan
      and code given a (possibly memory-refined) plan and the parent code (or
      ``None`` for a root draft) — the decoupled planner+coder seam;
    * ``evaluate_fn(code) -> (metric, is_buggy)`` runs/scores the candidate.

    Each step: select a node by UCT under the progressive ``c`` schedule, choose a
    coding mode, refine the plan against similar *success* memory, expand a child,
    evaluate it, backpropagate the reward along primary edges, and save a labelled
    memory record. When a branch stagnates, a **fusion** node is created whose
    ``reference_ids`` point at the best nodes of the *other* branches.
    """

    def __init__(
        self,
        coder_fn: Callable[[str, str | None], tuple[str, str]],
        evaluate_fn: Callable[[str], tuple[float, bool]],
        *,
        num_branches: int = 3,
        num_steps: int = 12,
        seed: int = 0,
        memory: GlobalCodeMemory | None = None,
        coldstart: ColdStartKB | None = None,
        stagnation_patience: int = 3,
    ) -> None:
        self._coder = coder_fn
        self._evaluate = evaluate_fn
        self._num_branches = max(1, num_branches)
        self._num_steps = max(0, num_steps)
        self._seed = seed
        self._rng = Random(seed)  # nosec B311 — reproducible search exploration, not crypto
        self.memory = memory if memory is not None else GlobalCodeMemory()
        self.coldstart = coldstart if coldstart is not None else ColdStartKB()
        self._stagnation_patience = max(1, stagnation_patience)

        self._counter = 0
        self.nodes: dict[str, SearchNode] = {}
        #: branch_id -> ordered list of that branch's node_ids.
        self._branch_nodes: dict[int, list[str]] = {}
        #: branch_id -> best metric seen + consecutive non-improving expansions.
        self._branch_best: dict[int, float] = {}
        self._branch_stall: dict[int, int] = {}
        #: node_ids that are fusion nodes (have cross-branch reference edges).
        self.fusion_nodes: list[str] = []

    # ── node bookkeeping ────────────────────────────────────────────────

    def _new_id(self, stage: Stage) -> str:
        self._counter += 1
        return f"{stage.value}-{self._counter:04d}"

    def _register(self, node: SearchNode) -> None:
        self.nodes[node.node_id] = node
        self._branch_nodes.setdefault(node.branch_id, []).append(node.node_id)

    def _evaluate_and_record(self, node: SearchNode, query_plan: str) -> None:
        """Run the evaluator, store the metric, and persist a labelled memory record."""
        metric, is_buggy = self._evaluate(node.code)
        node.metric = None if is_buggy else metric
        node.is_buggy = is_buggy

        label = self._label_for(node)
        self.memory.save(
            MemRecord(
                record_id=f"rec-{node.node_id}",
                plan=query_plan,
                code_summary=node.code[:200],
                stage=node.stage,
                label=label,
                metric=node.metric,
            )
        )

    def _label_for(self, node: SearchNode) -> int:
        """Reward label for a node (MLEvolve ``_determine_label``): -1/0/1."""
        if node.is_buggy or node.metric is None:
            return -1
        parent = self.nodes.get(node.parent_id) if node.parent_id else None
        if parent is None or parent.metric is None:
            return 1  # a fresh valid draft/root child is a success
        if node.metric > parent.metric:
            return 1
        if node.metric < parent.metric:
            return -1
        return 0

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        """Propagate reward to the root along *primary* edges only (MLEvolve §3.2.2).

        Reference edges are excluded from credit assignment by construction —
        only ``parent_id`` is traversed here.
        """
        current: SearchNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = self.nodes.get(current.parent_id) if current.parent_id else None

    def _reward(self, node: SearchNode) -> float:
        """Immediate reward (MLEvolve simulation): buggy=0, valid-improve weighted."""
        if node.is_buggy or node.metric is None:
            return 0.0
        parent = self.nodes.get(node.parent_id) if node.parent_id else None
        if parent is None or parent.metric is None:
            return 1.0
        return 1.0 if node.metric > parent.metric else 0.5

    # ── branch stagnation tracking ──────────────────────────────────────

    def _note_branch_outcome(self, branch_id: int, metric: float | None) -> None:
        prev_best = self._branch_best.get(branch_id)
        improved = metric is not None and (prev_best is None or metric > prev_best)
        if improved:
            self._branch_best[branch_id] = metric  # type: ignore[assignment]
            self._branch_stall[branch_id] = 0
        else:
            self._branch_stall[branch_id] = self._branch_stall.get(branch_id, 0) + 1

    def _is_stagnant(self, branch_id: int) -> bool:
        return self._branch_stall.get(branch_id, 0) >= self._stagnation_patience

    # ── selection ───────────────────────────────────────────────────────

    def _selectable(self) -> list[SearchNode]:
        """Non-buggy nodes eligible to be expanded (the tree backbone)."""
        return [
            n for n in self.nodes.values() if not n.is_buggy and n.stage != Stage.ROOT
        ]

    def _select(self, step: int) -> SearchNode:
        """UCT-select an expandable node under the progressive ``c`` schedule."""
        c = exploration_schedule(step, self._num_steps)
        candidates = self._selectable()
        # Parent visit count for the bound: use the max branch root visits as the
        # graph-level N (selection operates over the primary-edge backbone).
        parent_visits = max((n.visits for n in self.nodes.values()), default=1)

        def key(node: SearchNode) -> tuple[float, str]:
            return (node.uct(parent_visits, c), node.node_id)

        # Deterministic: ties break by node_id via the secondary key.
        return max(candidates, key=key)

    # ── expansion ───────────────────────────────────────────────────────

    def _refine_plan(self, base_plan: str) -> str:
        """Decoupled planning: refine a free-text plan using similar success records.

        Ports MLEvolve's planner↔coder decoupling (``planner_with_memory.py``):
        the planner's free-text plan is augmented with the single most similar
        *successful* prior experience before it is handed to the coder.
        """
        similar = self.memory.retrieve(base_plan, k=1, label=1)
        if not similar:
            return base_plan
        return f"{base_plan} | reuse:{similar[0].code_summary[:40]}"

    def _expand_child(self, parent: SearchNode, step: int) -> SearchNode:
        """Generate, evaluate, and backpropagate one primary-edge child."""
        stagnating = self._is_stagnant(parent.branch_id)
        mode = select_coding_mode(
            stagnating=stagnating,
            code_size=len(parent.code),
            has_error=parent.is_buggy,
        )
        refined_plan = self._refine_plan(f"{parent.plan} [{mode}]")
        plan_text, code = self._coder(refined_plan, parent.code)

        stage = Stage.DEBUG if parent.is_buggy else Stage.IMPROVE
        child = SearchNode(
            node_id=self._new_id(stage),
            code=code,
            plan=plan_text,
            metric=None,
            stage=stage,
            branch_id=parent.branch_id,
            parent_id=parent.node_id,
        )
        self._register(child)
        self._evaluate_and_record(child, refined_plan)
        self._backpropagate(child, self._reward(child))
        self._note_branch_outcome(parent.branch_id, child.metric)
        return child

    def _best_nodes_excluding(self, branch_id: int) -> list[SearchNode]:
        """Best valid node of every branch other than ``branch_id`` (for fusion)."""
        best: list[SearchNode] = []
        for other_branch, node_ids in self._branch_nodes.items():
            if other_branch == branch_id:
                continue
            valid = [
                self.nodes[nid]
                for nid in node_ids
                if not self.nodes[nid].is_buggy and self.nodes[nid].metric is not None
            ]
            if valid:
                best.append(max(valid, key=lambda n: (n.metric, n.node_id)))  # type: ignore[arg-type]
        best.sort(key=lambda n: (n.metric, n.node_id), reverse=True)  # type: ignore[arg-type, return-value]
        return best

    def _fuse(self, parent: SearchNode, step: int) -> SearchNode:
        """Create a cross-branch fusion node (MLEvolve cross-branch reference §3.2.2).

        The fusion node's ``reference_ids`` are the best nodes of the *other*
        branches — knowledge flow that a tree cannot represent. Reference edges do
        not participate in backpropagation, only generation.
        """
        references = self._best_nodes_excluding(parent.branch_id)
        ref_summaries = " ; ".join(r.plan for r in references) or parent.plan
        fusion_plan = self._refine_plan(f"fuse({parent.plan}) <- {ref_summaries}")
        plan_text, code = self._coder(fusion_plan, parent.code)

        fusion = SearchNode(
            node_id=self._new_id(Stage.FUSION),
            code=code,
            plan=plan_text,
            metric=None,
            stage=Stage.FUSION,
            branch_id=parent.branch_id,
            parent_id=parent.node_id,
            reference_ids=[r.node_id for r in references],
        )
        self._register(fusion)
        self.fusion_nodes.append(fusion.node_id)
        self._evaluate_and_record(fusion, fusion_plan)
        self._backpropagate(fusion, self._reward(fusion))
        self._note_branch_outcome(parent.branch_id, fusion.metric)
        # Reset the stall so a fused branch gets a fresh exploitation window.
        self._branch_stall[parent.branch_id] = 0
        return fusion

    # ── public entry point ──────────────────────────────────────────────

    def _seed_branches(self, task_text: str) -> None:
        """Seed one draft branch per ``num_branches`` from the cold-start KB."""
        root = SearchNode(
            node_id=self._new_id(Stage.ROOT),
            code="",
            plan="(root)",
            metric=None,
            stage=Stage.ROOT,
            branch_id=0,
        )
        self._register(root)

        recommendations = self.coldstart.recommend(task_text)
        for branch_id in range(1, self._num_branches + 1):
            if recommendations:
                approach = recommendations[(branch_id - 1) % len(recommendations)]
            else:
                approach = f"baseline approach {branch_id}"
            seed_plan = f"draft branch {branch_id}: {approach}"
            plan_text, code = self._coder(seed_plan, None)
            draft = SearchNode(
                node_id=self._new_id(Stage.DRAFT),
                code=code,
                plan=plan_text,
                metric=None,
                stage=Stage.DRAFT,
                branch_id=branch_id,
                parent_id=root.node_id,
            )
            self._register(draft)
            self._evaluate_and_record(draft, seed_plan)
            self._backpropagate(draft, self._reward(draft))
            self._note_branch_outcome(branch_id, draft.metric)

    def run(self, task_text: str) -> SearchNode:
        """Run the full progressive MCGS and return the best node found.

        1. Seed ``num_branches`` draft branches from :class:`ColdStartKB`.
        2. For each step: UCT-select a node under the progressive exploration
           schedule, expand a memory-guided child, evaluate it, and backpropagate.
        3. When the selected node's branch has stagnated for
           ``stagnation_patience`` non-improving expansions, create a cross-branch
           **fusion** node referencing the best nodes of the other branches.
        4. Return the highest-metric valid node (the root if none is valid).
        """
        self._seed_branches(task_text)

        for step in range(self._num_steps):
            if not self._selectable():
                break
            parent = self._select(step)
            if self._is_stagnant(parent.branch_id) and self._num_branches > 1:
                self._fuse(parent, step)
            else:
                self._expand_child(parent, step)

        return self._best_node()

    def _best_node(self) -> SearchNode:
        valid = [
            n for n in self.nodes.values() if not n.is_buggy and n.metric is not None
        ]
        if not valid:
            # Fall back to the root when nothing evaluated cleanly.
            return next(n for n in self.nodes.values() if n.stage == Stage.ROOT)
        return max(valid, key=lambda n: (n.metric, n.node_id))  # type: ignore[arg-type, return-value]
