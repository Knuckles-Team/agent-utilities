"""GEPA (Genetic-Pareto) Prompt Optimization Loop for Recursive LMs.

CONCEPT:ORCH-1.31 — GEPA Reflective Prompt Optimizer
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


class ParetoCandidatePool:
    """Maintains a set of non-dominated candidates (the Pareto Frontier)."""

    def __init__(self, objectives: list[str], max_size: int = 10):
        self.objectives = objectives  # e.g., ["accuracy", "efficiency", "error_rate"]
        self.max_size = max_size
        self.pool: list[Candidate] = []

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
    ) -> Candidate:
        """Propose a mutated prompt variant using natural language gradients."""
        trace_summary = ""
        for i, t in enumerate(traces[:5]):  # limit to top 5 to avoid context blowup
            trace_summary += (
                f"Evaluation Task {i + 1}:\n"
                f"  - Code Run: {t.get('code', 'N/A')}\n"
                f"  - Trace stdout/stderr: {t.get('stdout', 'N/A')[:500]}...\n\n"
            )

        prompt = (
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
    ):
        self.signature_class = signature_class
        self.base_prompt = base_prompt
        self.evaluator_fn = evaluator_fn
        self.config = config or RLMConfig()
        self.graph_deps = graph_deps

        self.objectives = objectives or ["accuracy", "efficiency"]
        self.pool = ParetoCandidatePool(objectives=self.objectives)
        self.mutator = ReflectiveMutator(model=self.config.sub_llm_model_small)

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
    ) -> Candidate:
        """Run the GEPA optimization loop over the provided dataset."""
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
                child = await self.mutator.mutate(parent, traces, feedback, gen)
                new_candidates.append(child)

            # 2. Perform Crossover if we have multiple Pareto candidates
            if len(frontier) >= 2:
                crossover_child = await self.mutator.crossover(
                    frontier[0], frontier[1], gen
                )
                new_candidates.append(crossover_child)

            # 3. Update Pool
            self.pool.update(new_candidates)

        # Return the best candidate from the frontier
        best_candidate = self.pool.get_frontier()[0]
        logger.info(f"Optimization finished. Best candidate: {best_candidate.id}")
        return best_candidate

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
