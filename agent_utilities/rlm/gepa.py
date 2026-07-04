"""GEPA (Genetic-Pareto) Prompt Optimization Loop for Recursive LMs.

CONCEPT:AU-ORCH.optimization.optimize-skill-prompt-gepa — GEPA Reflective Prompt Optimizer
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..graph.client import create_or_merge_node
from ..graph.state import GraphDeps
from .config import RLMConfig
from .predict_rlm import PredictRLM

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    """A prompt candidate representing a hypothesis in the evolutionary search."""

    id: str
    prompt_text: str
    generation: int
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    parent_ids: list[str] = Field(default_factory=list)


class GEPAInstance(BaseModel):
    """An evaluation instance containing dynamic project payloads (JSON, GraphQL)."""

    id: str
    input_data: dict[str, Any]
    reference_output: Any = None
    rubric: str = ""


# ── CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — Generalizing GEPA (held-out split, AgentSpec grounding, patch-merge) ──
# Assimilated from the GEPA paper (Agrawal et al., ICLR 2026; D_train → D_feedback + D_pareto) and
# predict-rlm's AgentSpec (anti-overfit grounding). Makes optimized skills *transfer* off the split.


class AgentSpec(BaseModel):
    """Grounds the proposer so optimized skills capture a general SOP, not benchmark glue.

    CONCEPT:AU-ORCH.optimization.selection-on-unseen-data. ``counterfactual_axis`` names the dimension the skill must generalize across
    (e.g. "different app/API set"), steering the proposer away from memorizing the training split.
    """

    use_cases: list[str] = Field(default_factory=list)
    runtime_grounding: list[str] = Field(
        default_factory=list
    )  # tools, env constraints, protocol facts
    scoring_rule: str = ""
    counterfactual_axis: str = ""

    def as_prompt(self) -> str:
        parts = [
            "# Agent Specification (ground your edits in this — do not overfit the examples)"
        ]
        if self.use_cases:
            parts.append("## Use cases\n" + "\n".join(f"- {u}" for u in self.use_cases))
        if self.runtime_grounding:
            parts.append(
                "## Runtime surface & constraints\n"
                + "\n".join(f"- {r}" for r in self.runtime_grounding)
            )
        if self.scoring_rule:
            parts.append(f"## Scoring rule\n{self.scoring_rule}")
        if self.counterfactual_axis:
            parts.append(
                f"## Generalization requirement\nThe skill MUST generalize across: "
                f"{self.counterfactual_axis}. Prefer general procedures over example-specific rules."
            )
        return "\n\n".join(parts)


def split_dataset(
    dataset: list[GEPAInstance], dev_fraction: float, *, seed: int = 0
) -> tuple[list[GEPAInstance], list[GEPAInstance]]:
    """Split into (feedback_set, held-out pareto/dev_set) — GEPA's D_feedback / D_pareto (ORCH-1.30).

    Deterministic (seeded). ``dev_fraction`` in (0,1) reserves that fraction for selection. With
    ``dev_fraction <= 0`` the pareto set is empty (current behavior — select on feedback set).
    """
    import random as _random

    if dev_fraction <= 0 or len(dataset) < 2:
        return list(dataset), []
    rng = _random.Random(seed)  # nosec B311 - deterministic train/dev split, not security
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    n_dev = max(1, int(round(len(dataset) * min(dev_fraction, 0.9))))
    dev_ids = set(idx[:n_dev])
    feedback = [d for i, d in enumerate(dataset) if i not in dev_ids]
    pareto = [d for i, d in enumerate(dataset) if i in dev_ids]
    return feedback, pareto


def select_best_on_heldout(
    candidates: list[Candidate], heldout_scores: dict[str, float]
) -> Candidate:
    """Pick the candidate with the highest held-out score (GEPA selection; patch-merge graft).

    CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — selection happens on UNSEEN data, so a candidate that merely memorized the
    feedback minibatch does not win. Ties break toward the earlier generation (simpler) candidate.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    return max(
        candidates,
        key=lambda c: (heldout_scores.get(c.id, float("-inf")), -c.generation),
    )


class ParetoCandidatePool:
    """Maintains a set of non-dominated candidates (the Pareto Frontier)."""

    def __init__(
        self, objectives: list[str], max_size: int = 10, dynamic_weighting: bool = False
    ):
        self.objectives = objectives  # e.g., ["accuracy", "efficiency", "error_rate"]
        self.max_size = max_size
        self.pool: list[Candidate] = []
        # CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — DW-GRPO anti-seesaw reward weighting (opt-in).
        self._weighter: Any = None
        if dynamic_weighting:
            from .dynamic_reward import DynamicRewardWeighter

            self._weighter = DynamicRewardWeighter(objectives)

    def is_dominated(self, c1: Candidate, c2: Candidate) -> bool:
        """Returns True if c1 is dominated by c2.

        c1 is dominated by c2 if:
          1. c2 is no worse than c1 in all objectives.
          2. c2 is strictly better than c1 in at least one objective.
        """
        no_worse = True
        strictly_better = False

        for obj in self.objectives:
            # We assume ALL objectives are maximized.
            # For error_rate, we should pass (1.0 - error_rate) or negative error_rate to maximize it.
            v1 = c1.scores.get(obj, 0.0)
            v2 = c2.scores.get(obj, 0.0)

            if v2 < v1:
                no_worse = False
                break
            if v2 > v1:
                strictly_better = True

        return no_worse and strictly_better

    def update(self, new_candidates: list[Candidate]):
        """Adds new candidates and prunes dominated ones to maintain the Pareto frontier."""
        all_candidates = self.pool + new_candidates
        non_dominated: list[Candidate] = []

        for c1 in all_candidates:
            dominated = False
            for c2 in all_candidates:
                if c1.id != c2.id and self.is_dominated(c1, c2):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(c1)

        # Remove duplicates by ID
        unique_candidates = {}
        for c in non_dominated:
            unique_candidates[c.id] = c

        # Sort by primary objective (first one in objectives) and slice
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda c: c.scores.get(self.objectives[0], 0.0),
            reverse=True,
        )

        self.pool = sorted_candidates[: self.max_size]

    def get_frontier(self) -> list[Candidate]:
        """Return the current active Pareto frontier candidates."""
        return self.pool

    def observe(self) -> dict[str, float]:
        """Record the current best-per-objective into the DW-GRPO weighter.

        No-op (returns ``{}``) when dynamic weighting is disabled. CONCEPT:AU-ORCH.optimization.selection-on-unseen-data.
        """
        if self._weighter is None or not self.pool:
            return {}
        best = {
            obj: max(c.scores.get(obj, 0.0) for c in self.pool)
            for obj in self.objectives
        }
        self._weighter.observe(best)
        return self._weighter.weights()

    def weighted_best(self) -> Candidate | None:
        """Best candidate by DW-GRPO weighted scalarization (CONCEPT:AU-ORCH.optimization.selection-on-unseen-data).

        Falls back to the primary-objective best (identical to ``get_frontier()[0]``)
        when dynamic weighting is off or there is not yet a slope signal — so
        behaviour is unchanged until the anti-seesaw signal is meaningful.
        """
        if not self.pool:
            return None
        if self._weighter is None or not self._weighter.ready:
            return self.pool[0]
        return max(self.pool, key=lambda c: self._weighter.scalarize(c.scores))

    @property
    def reward_weights(self) -> dict[str, float]:
        """Current DW-GRPO reward weights (uniform when disabled/cold)."""
        if self._weighter is None:
            n = len(self.objectives) or 1
            return {obj: 1.0 / n for obj in self.objectives}
        return self._weighter.weights()

    # ── CONCEPT:AU-ORCH.optimization.graph-native-optimization-state — Graph-Native Optimization State (resumable GEPA) ──────────

    def to_snapshot(self) -> list[dict[str, Any]]:
        """Serialize the frontier (candidates + ancestry) for durable persistence (pure)."""
        return [c.model_dump() for c in self.pool]

    def load_snapshot(self, snapshot: list[dict[str, Any]]) -> int:
        """Rebuild candidates from a snapshot and merge them into the pool. Returns the count."""
        restored = [Candidate(**row) for row in snapshot]
        self.update(restored)
        return len(restored)


class ReflectiveMutator:
    """Uses natural language execution traces and feedback to propose prompt updates."""

    def __init__(self, model: Any = "openai:gpt-4o-mini"):
        self.model = model
        self.agent = Agent(
            model=model,
            system_prompt=(
                "You are an expert Evolutionary Prompt Mutator.\n"
                "Your task is to analyze previous prompt variants, their execution traces (stdout, code written), "
                "and evaluator/OWL validation feedback to produce a superior prompt instruction set.\n"
                "Focus on critical instruction changes that solve the highlighted failures."
            ),
        )

    async def mutate(
        self,
        parent: Candidate,
        traces: list[dict[str, Any]],
        feedback: list[str],
        generation: int,
        agent_spec: Any = None,
    ) -> Candidate:
        """Propose a mutated prompt variant using natural language gradients.

        CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — when an ``agent_spec`` is supplied, its grounding (use cases, runtime
        surface, counterfactual axis) is prepended so the proposer writes a general SOP rather than
        rules that overfit the training minibatch.
        """
        spec_block = ""
        if agent_spec is not None:
            try:
                spec_block = agent_spec.as_prompt() + "\n\n"
            except Exception:  # noqa: BLE001
                spec_block = ""
        trace_summary = ""
        for i, t in enumerate(traces[:5]):  # limit to top 5 to avoid context blowup
            trace_summary += (
                f"Evaluation Task {i + 1}:\n"
                f"  - Code Run: {t.get('code', 'N/A')}\n"
                f"  - Trace stdout/stderr: {t.get('stdout', 'N/A')[:500]}...\n\n"
            )

        prompt = (
            f"{spec_block}"
            f"PARENT PROMPT IDENTIFIER: {parent.id}\n"
            f"PARENT PROMPT INSTRUCTIONS:\n"
            f'"""\n{parent.prompt_text}\n"""\n\n'
            f"EXECUTION TRACES & ANOMALIES:\n"
            f"{trace_summary}\n"
            f"EVALUATOR & COMPLIANCE FEEDBACK:\n"
            f"{chr(10).join(feedback)}\n\n"
            f"INSTRUCTIONS:\n"
            f"  1. Review parent prompt and what failed.\n"
            f"  2. Propose a new set of prompt instructions that systematically mitigates these failures.\n"
            f"  3. Keep the prompt self-contained, clear, and actionable.\n"
            f"  4. Provide a 1-sentence rationale for the modifications.\n\n"
            f"Output your proposal strictly in this JSON format:\n"
            f"{{\n"
            f'  "rationale": "explanation of prompt modifications",\n'
            f'  "mutated_prompt": "the complete new prompt instructions"\n'
            f"}}"
        )

        res = await self.agent.run(prompt)
        import json

        # Clean JSON markdown blocks if any
        res_text = res.output.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:]
        if res_text.endswith("```"):
            res_text = res_text[:-3]
        res_text = res_text.strip()

        data = json.loads(res_text)

        cand_id = f"cand_gen{generation}_{int(time.time())}"
        return Candidate(
            id=cand_id,
            prompt_text=data["mutated_prompt"],
            generation=generation,
            rationale=data["rationale"],
            parent_ids=[parent.id],
        )

    async def crossover(
        self,
        parent1: Candidate,
        parent2: Candidate,
        generation: int,
    ) -> Candidate:
        """Perform crossover by merging strategies of two Pareto-optimal prompts."""
        prompt = (
            f"We are optimizing RLM prompts. We have two high-performing parent prompts:\n\n"
            f"PARENT 1 ({parent1.id}):\n"
            f'"""\n{parent1.prompt_text}\n"""\n\n'
            f"PARENT 2 ({parent2.id}):\n"
            f'"""\n{parent2.prompt_text}\n"""\n\n'
            f"INSTRUCTIONS:\n"
            f"  1. Synthesize the core successful strategies from both parent prompts into a single hybrid prompt.\n"
            f"  2. Eliminate redundant constraints or conflicting instructions.\n"
            f"  3. Output the hybrid prompt in this JSON format:\n"
            f"{{\n"
            f'  "rationale": "crossover strategy explanation",\n'
            f'  "mutated_prompt": "the complete crossed-over prompt instructions"\n'
            f"}}"
        )

        res = await self.agent.run(prompt)
        import json

        res_text = res.output.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:]
        if res_text.endswith("```"):
            res_text = res_text[:-3]
        res_text = res_text.strip()

        data = json.loads(res_text)

        cand_id = f"crossover_{parent1.id}_{parent2.id}_{int(time.time())}"
        return Candidate(
            id=cand_id,
            prompt_text=data["mutated_prompt"],
            generation=generation,
            rationale=data["rationale"],
            parent_ids=[parent1.id, parent2.id],
        )


class GEPAOptimizer:
    """Coordinates the full Genetic-Pareto prompt optimization loop."""

    def __init__(
        self,
        signature_class: type[BaseModel],
        base_prompt: str,
        evaluator_fn: Callable[[GEPAInstance, BaseModel, str], Any],
        objectives: list[str] | None = None,
        config: RLMConfig | None = None,
        graph_deps: GraphDeps | None = None,
        agent_spec: AgentSpec | None = None,
    ):
        self.signature_class = signature_class
        self.base_prompt = base_prompt
        self.evaluator_fn = evaluator_fn
        self.config = config or RLMConfig()
        self.graph_deps = graph_deps
        self.agent_spec = agent_spec  # CONCEPT:AU-ORCH.optimization.selection-on-unseen-data anti-overfit grounding

        self.objectives = objectives or ["accuracy", "efficiency"]
        # CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — DW-GRPO anti-seesaw weighting on by default for the optimizer.
        self.pool = ParetoCandidatePool(
            objectives=self.objectives, dynamic_weighting=True
        )
        # CONCEPT:AU-ORCH.optimization.proposer-strong-model — the proposer is the STRONG model (resolved via the rlm-proposer role),
        # decoupled from the cheap executor/sub-LM. Falls back to the configured small model.
        from .roles import rlm_role_model

        self.mutator = ReflectiveMutator(
            model=rlm_role_model(
                "rlm-proposer", fallback=self.config.sub_llm_model_small
            )
        )

        # Initialize candidate pool with base prompt
        base_candidate = Candidate(
            id="base_prompt",
            prompt_text=base_prompt,
            generation=0,
            scores={obj: 0.0 for obj in self.objectives},
            rationale="Initial baseline prompt.",
        )
        self.pool.update([base_candidate])

    async def optimize(
        self,
        dataset: list[GEPAInstance],
        iterations: int = 3,
        batch_size: int = 5,
        enable_schema_diversity: bool = False,
        dev_fraction: float = 0.0,
    ) -> Candidate:
        """Run the GEPA optimization loop over the provided dataset.

        CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — when ``dev_fraction > 0``, the dataset is split into a feedback set (for
        proposing) and a held-out Pareto/dev set; the final candidate is selected by held-out score
        so optimized skills generalize off the optimization split (no overfitting to the minibatch).
        """
        feedback_set, pareto_set = split_dataset(dataset, dev_fraction)
        dataset = feedback_set  # propose/evaluate only on the feedback split
        for gen in range(1, iterations + 1):
            logger.info(f"--- Starting GEPA Generation {gen} ---")
            frontier = self.pool.get_frontier()

            new_candidates = []

            # 1. Propose mutations/crossovers
            for parent in frontier:
                # Run parent evaluations to capture traces
                traces: list[dict[str, Any]] = []
                feedback: list[str] = []
                scores_accumulator = {obj: 0.0 for obj in self.objectives}

                # Evaluate on a mini-batch subset
                import random

                mini_batch = random.sample(dataset, min(len(dataset), batch_size))  # nosec B311

                if enable_schema_diversity:
                    diverse_batch = []
                    for inst in mini_batch:
                        diverse_batch.append(inst)
                        if random.random() > 0.5:  # nosec B311
                            diverse_batch.append(self._perturb_instance(inst))
                    mini_batch = diverse_batch

                for instance in mini_batch:
                    # Modify current prompt of signature class dynamically
                    # We inject the prompt_text into PredictRLM runtime instructions
                    harness = PredictRLM(
                        signature=self.signature_class,
                        config=self.config,
                        graph_deps=self.graph_deps,
                    )
                    # Override the description of the signature class temporarily or instruct the runner
                    harness.signature.__doc__ = parent.prompt_text

                    try:
                        # Capture startTime for efficiency tracking
                        start_time = time.time()
                        result_model = await harness.run(**instance.input_data)
                        duration = time.time() - start_time

                        # Run custom evaluator function
                        eval_scores, explanation = await self.evaluator_fn(
                            instance, result_model, str(result_model.model_dump())
                        )

                        # Accumulate scores
                        for obj in self.objectives:
                            if obj == "efficiency":
                                # Efficiency: higher is better (e.g. 1 / duration)
                                scores_accumulator[obj] += 1.0 / max(duration, 0.1)
                            else:
                                scores_accumulator[obj] += eval_scores.get(obj, 0.0)

                        feedback.append(explanation)
                        traces.append(
                            {
                                "instance_id": instance.id,
                                "code": "PredictRLM Execution Complete",
                                "stdout": f"Result: {result_model.model_dump()}",
                            }
                        )

                        # Persist Trajectory to Epistemic Graph
                        await self._persist_trajectory_to_graph(
                            parent, instance, eval_scores, explanation
                        )

                    except Exception as e:
                        logger.error(
                            f"Execution failed for instance {instance.id}: {e}"
                        )
                        feedback.append(f"Traceback/Exception: {e}")

                # Compute mean scores
                mean_scores = {}
                for obj in self.objectives:
                    mean_scores[obj] = scores_accumulator[obj] / max(len(mini_batch), 1)

                parent.scores = mean_scores

                # Generate new mutated child candidate
                child = await self.mutator.mutate(
                    parent, traces, feedback, gen, agent_spec=self.agent_spec
                )
                new_candidates.append(child)

            # 2. Perform Crossover if we have multiple Pareto candidates
            if len(frontier) >= 2:
                crossover_child = await self.mutator.crossover(
                    frontier[0], frontier[1], gen
                )
                new_candidates.append(crossover_child)

            # 3. Update Pool + advance DW-GRPO reward weights (anti-seesaw, CONCEPT:AU-ORCH.optimization.selection-on-unseen-data)
            self.pool.update(new_candidates)
            self.pool.observe()

        # CONCEPT:AU-ORCH.optimization.selection-on-unseen-data — select the final candidate on the HELD-OUT pareto set (generalization),
        # not on the feedback minibatch the candidates were tuned on.
        frontier = self.pool.get_frontier()
        if pareto_set and len(frontier) > 1:
            heldout = {
                c.id: await self._score_candidate_on(c, pareto_set) for c in frontier
            }
            best_candidate = select_best_on_heldout(frontier, heldout)
            logger.info(
                "Optimization finished. Held-out best: %s (score %.3f)",
                best_candidate.id,
                heldout.get(best_candidate.id, 0.0),
            )
        else:
            # DW-GRPO anti-seesaw selection (falls back to frontier[0] until the
            # slope signal is meaningful) — CONCEPT:AU-ORCH.optimization.selection-on-unseen-data.
            best_candidate = self.pool.weighted_best() or frontier[0]
            logger.info(
                "Optimization finished. Best candidate: %s (reward weights=%s)",
                best_candidate.id,
                {k: round(v, 3) for k, v in self.pool.reward_weights.items()},
            )
        return best_candidate

    async def _score_candidate_on(
        self, candidate: Candidate, instances: list[GEPAInstance]
    ) -> float:
        """Mean accuracy of a candidate prompt over a (held-out) instance set (CONCEPT:AU-ORCH.optimization.selection-on-unseen-data)."""
        total = 0.0
        for instance in instances:
            harness = PredictRLM(
                signature=self.signature_class,
                config=self.config,
                graph_deps=self.graph_deps,
            )
            harness.signature.__doc__ = candidate.prompt_text
            try:
                result_model = await harness.run(**instance.input_data)
                eval_scores, _ = await self.evaluator_fn(
                    instance, result_model, str(result_model.model_dump())
                )
                total += float(eval_scores.get("accuracy", 0.0))
            except Exception as e:  # noqa: BLE001 - a failed eval scores 0 for this instance
                logger.debug("held-out eval failed for %s: %s", instance.id, e)
        return total / max(len(instances), 1)

    def _perturb_instance(self, instance: GEPAInstance) -> GEPAInstance:
        """Create a synthetically perturbed version of an instance for schema diversity sweeps.
        This prevents the RLM from overfitting to static API shapes.
        """
        import copy
        import random

        perturbed_data = copy.deepcopy(instance.input_data)

        # Simple schema perturbations
        if isinstance(perturbed_data, dict):
            # 1. Nest payload under a 'data' or 'payload' key
            if random.random() > 0.5:  # nosec B311
                wrapper = random.choice(["data", "payload", "result", "response"])  # nosec B311
                perturbed_data = {wrapper: perturbed_data}

            # 2. Add synthetic noise fields
            if random.random() > 0.5:  # nosec B311
                perturbed_data["_metadata"] = {
                    "timestamp": time.time(),
                    "synthetic": True,
                }

            # 3. Randomize key casing (snake_case to camelCase mapping representation)
            if random.random() > 0.7:  # nosec B311
                camel_data = {}
                for k, v in list(perturbed_data.items()):
                    if isinstance(k, str) and "_" in k:
                        parts = k.split("_")
                        new_k = parts[0] + "".join(p.title() for p in parts[1:])
                        camel_data[new_k] = v
                    else:
                        camel_data[k] = v
                perturbed_data = camel_data

        return GEPAInstance(
            id=f"{instance.id}_perturbed_{int(time.time())}",
            input_data=perturbed_data,
            reference_output=instance.reference_output,
            rubric=instance.rubric,
        )

    async def _persist_trajectory_to_graph(
        self,
        candidate: Candidate,
        instance: GEPAInstance,
        eval_scores: dict[str, float],
        explanation: str,
    ):
        """Persist RLM prompt execution metrics and lineage to the Epistemic Graph using standard nodes."""
        try:
            # We map OptimizationTrajectoryNode to TrajectoryNode (which already exists in the graph schema!)
            # Reusing the existing graph nodes enables full compatibility.
            # TrajectoryNode has thinker_id, query_hash, answer, reasoning_summary, score, is_correct, model_id.
            from ..models.knowledge_graph import (
                EvaluatorFeedbackNode,
                OptimizationTrajectoryNode,
                RegistryNodeType,
            )

            timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # 1. Create Trajectory Node for prompt candidate instance
            traj_node_id = f"traj_{candidate.id}_{instance.id}_{int(time.time())}"
            traj = OptimizationTrajectoryNode(
                id=traj_node_id,
                type=RegistryNodeType.OPTIMIZATION_TRAJECTORY,
                name=f"GEPA Candidate {candidate.id} on {instance.id}",
                thinker_id=candidate.id,
                query_hash=instance.id,
                answer=candidate.prompt_text,
                reasoning_summary=candidate.rationale,
                score=eval_scores.get("accuracy", 0.0),
                model_id=self.config.sub_llm_model_small,
                timestamp=timestamp_str,
            )

            # 2. Create Outcome Evaluation Node linking to the trajectory node
            eval_node_id = f"eval_{traj_node_id}"
            outcome_eval = EvaluatorFeedbackNode(
                id=eval_node_id,
                type=RegistryNodeType.EVALUATOR_FEEDBACK,
                name=f"Evaluation of {traj_node_id}",
                reward=eval_scores.get("accuracy", 0.0),
                feedback_text=explanation,
                timestamp=timestamp_str,
            )

            # Write nodes dynamically to the Epistemic Graph using our GraphNode-based client adapter
            # (which automatically serializes props and handles Cypher integration)
            from ..graph.models import GraphNode

            g_traj = GraphNode(
                id=traj.id,
                labels=["Trajectory"],
                properties=traj.model_dump(exclude_none=True),
            )
            g_eval = GraphNode(
                id=outcome_eval.id,
                labels=["OutcomeEvaluation"],
                properties=outcome_eval.model_dump(exclude_none=True),
            )

            await create_or_merge_node(g_traj)
            await create_or_merge_node(g_eval)

            logger.debug(
                f"Persisted trajectory and outcome eval nodes to KG: {traj_node_id}"
            )

        except Exception as e:
            logger.warning(f"Could not persist trajectory node to graph: {e}")

    # ── CONCEPT:AU-ORCH.optimization.graph-native-optimization-state — Graph-Native Optimization State ──────────────────────────

    async def persist_frontier(self, run_id: str) -> bool:
        """Snapshot the Pareto frontier (candidates + ancestry) to the durable epistemic-graph.

        CONCEPT:AU-ORCH.optimization.graph-native-optimization-state — enables resumable, cross-session GEPA: a killed run can resume from the
        persisted frontier, and prior frontiers accumulate as reusable optimization state. Best-effort.
        """
        import json as _json

        try:
            from ..graph.models import GraphNode

            snapshot = self.pool.to_snapshot()
            node = GraphNode(
                id=f"gepa_frontier_{run_id}",
                labels=["GEPAFrontier"],
                properties={
                    "run_id": run_id,
                    "snapshot_json": _json.dumps(snapshot),
                    "candidate_count": len(snapshot),
                    "objectives": ",".join(self.objectives),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            await create_or_merge_node(node)
            logger.info(
                "[ORCH-1.31] persisted GEPA frontier %s (%d candidates)",
                run_id,
                len(snapshot),
            )
            return True
        except Exception as e:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Could not persist GEPA frontier: %s", e)
            return False

    async def resume_frontier(self, run_id: str) -> int:
        """Load a persisted frontier snapshot into the pool. Returns candidates restored (0 if none).

        CONCEPT:AU-ORCH.optimization.graph-native-optimization-state. Best-effort: a missing snapshot or absent backend returns 0.
        """
        import json as _json

        try:
            from ..graph.client import get_graph_client

            client = get_graph_client()
            rows = await client.query(
                "MATCH (n:GEPAFrontier {run_id: $rid}) RETURN n.snapshot_json AS snap",
                {"rid": run_id},
            )
            if not rows:
                return 0
            snap = rows[0].get("snap") if isinstance(rows[0], dict) else None
            if not snap:
                return 0
            return self.pool.load_snapshot(_json.loads(snap))
        except Exception as e:  # noqa: BLE001 - resume is best-effort
            logger.debug("Could not resume GEPA frontier %s: %s", run_id, e)
            return 0
