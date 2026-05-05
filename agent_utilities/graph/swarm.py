#!/usr/bin/python
"""CONCEPT:ORCH-1.0 — Swarm Orchestration with Emergent Hierarchies.

Dynamically spawns agent swarms based on task decomposition, forming
emergent hierarchies instead of relying on static specialist registration.

Architecture:
    1. **Decompose**: LLM breaks a complex task into a ``TaskTree``
    2. **Score affinity**: Tri-signal scoring (semantic + structural + historical)
    3. **Spawn**: Create sub-agent graphs for each subtask cluster
    4. **Execute**: Fan-out parallel for independent subtasks, chain sequential for dependent
    5. **Recurse**: Nested spawning for complex subtasks (up to ``max_depth``)

Integrates with:
    - AU-013 (OGM): Coalition tracking nodes via ``KGMapper``
    - Existing HSM: ``run_orthogonal_regions()`` for parallel fan-out
    - Existing engine: ``spawn_specialized_agent()`` for agent creation
    - Existing builder: ``discover_agents()`` for specialist registry

See docs/emergent-architecture.md §AU-014.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from ..knowledge_graph.engine import cosine_similarity
from ..knowledge_graph.ogm import KGMapper
from ..models.knowledge_graph import (
    SwarmCoalitionNode,
)
from .swarm_models import SwarmResult, TaskTree

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine
    from .state import GraphDeps

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Dynamically spawns agent swarms based on task decomposition.

    CONCEPT:ORCH-1.0 — Swarm Orchestration

    Replaces static specialist dispatch with dynamic swarm formation.
    For each incoming task:

    1. Decomposes it into a ``TaskTree`` using LLM planning
    2. Computes affinity scores between available specialists and subtasks
    3. Spawns sub-agent graphs for each subtask cluster
    4. Executes in parallel where independent, sequential where dependent
    5. Recursively spawns sub-swarms for complex subtasks

    The swarm hierarchy is persisted as ``SwarmCoalitionNode`` in the KG
    for observability and evolutionary feedback.

    Args:
        engine: The ``IntelligenceGraphEngine`` for KG access.
        max_depth: Maximum nesting depth for recursive sub-swarms.
        max_agents: Maximum total agents in a single swarm.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine,
        max_depth: int = 3,
        max_agents: int = 10,
    ) -> None:
        self.engine = engine
        self.ogm = KGMapper(engine)
        self.max_depth = max_depth
        self.max_agents = max_agents

    # ── Affinity Scoring ──────────────────────────────────────────────

    def compute_affinity(
        self,
        specialist_id: str,
        subtask: str,
    ) -> float:
        """Compute affinity between a specialist and a subtask.

        Uses a tri-signal scoring approach:
            1. **Semantic similarity**: Embedding cosine similarity between
               specialist description and subtask text
            2. **Structural overlap**: Number of shared tool capabilities
            3. **Historical success**: Mean reward from past executions

        Args:
            specialist_id: ID of the candidate specialist agent.
            subtask: The subtask description text.

        Returns:
            Affinity score in [0.0, 1.0].
        """
        scores: list[float] = []

        # 1. Semantic similarity via embeddings
        specialist_data = self.engine.graph.nodes.get(specialist_id, {})
        specialist_emb = specialist_data.get("embedding")
        if specialist_emb:
            # Generate embedding for subtask if possible
            try:
                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                model = create_embedding_model()
                subtask_emb = model.get_text_embedding(subtask)
                sim = cosine_similarity(specialist_emb, subtask_emb)
                scores.append(sim)
            except Exception:
                # Fallback to keyword matching
                desc = specialist_data.get("description", "").lower()
                overlap = sum(1 for w in subtask.lower().split() if w in desc)
                scores.append(min(1.0, overlap / max(len(subtask.split()), 1)))
        else:
            # Keyword-based fallback
            desc = specialist_data.get("description", "").lower()
            name = specialist_data.get("name", "").lower()
            combined = f"{desc} {name}"
            overlap = sum(1 for w in subtask.lower().split() if w in combined)
            scores.append(min(1.0, overlap / max(len(subtask.split()), 1)))

        # 2. Historical success from KG
        if self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (a {id: $aid})<-[:EXECUTED_BY]-(ep)"
                    "-[:PRODUCED_OUTCOME]->(eval:OutcomeEvaluation) "
                    "RETURN avg(eval.reward) as avg_reward",
                    {"aid": specialist_id},
                )
                if results and results[0].get("avg_reward") is not None:
                    scores.append(float(results[0]["avg_reward"]))
            except Exception:
                pass  # nosec B110

        return sum(scores) / len(scores) if scores else 0.0

    def rank_specialists(
        self,
        subtask: str,
        available_specialists: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Rank specialists by affinity to a subtask.

        Args:
            subtask: The subtask description.
            available_specialists: List of specialist agent IDs.
            top_k: Maximum number of specialists to return.

        Returns:
            List of (specialist_id, affinity_score) tuples, sorted descending.
        """
        scored = [
            (sid, self.compute_affinity(sid, subtask)) for sid in available_specialists
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Task Decomposition ────────────────────────────────────────────

    def decompose_task(self, task: str, deps: GraphDeps) -> TaskTree:
        """Decompose a complex task into a TaskTree using LLM planning.

        Falls back to a flat single-node tree if LLM decomposition fails.

        Args:
            task: The complex task description to decompose.
            deps: Runtime dependencies for LLM access.

        Returns:
            A ``TaskTree`` representing the decomposed subtasks.
        """
        try:
            from agent_utilities.core.model_factory import create_model

            model = create_model(
                provider=deps.provider,
                base_url=deps.base_url,
                api_key=deps.api_key,
            )
            decomp_agent = Agent(
                model=model,
                system_prompt=(
                    "You are a task decomposition engine. Given a complex task, "
                    "break it into 2-5 subtasks. For each subtask, indicate:\n"
                    "- The subtask description\n"
                    "- Whether it can run in parallel with other subtasks\n"
                    "- Any dependencies on other subtasks\n\n"
                    "Respond as a JSON array of objects with keys: "
                    "'task', 'parallelizable', 'dependencies' (list of task strings)."
                ),
            )
            result = decomp_agent.run_sync(task)

            # Parse LLM response
            import json

            subtask_data = json.loads(result.data)
            subtasks = [
                TaskTree(
                    task=st.get("task", ""),
                    parallelizable=st.get("parallelizable", True),
                    dependencies=st.get("dependencies", []),
                )
                for st in subtask_data
                if st.get("task")
            ]

            return TaskTree(task=task, subtasks=subtasks)

        except Exception as e:
            logger.warning("Task decomposition failed, using flat tree: %s", e)
            return TaskTree(task=task, subtasks=[], parallelizable=True)

    # ── Swarm Execution ───────────────────────────────────────────────

    async def decompose_and_spawn(
        self,
        task: str,
        deps: GraphDeps,
        depth: int = 0,
    ) -> SwarmResult:
        """Decompose a task and spawn a swarm to execute it.

        This is the main entry point for swarm-based execution.

        Args:
            task: The task description.
            deps: Runtime graph dependencies.
            depth: Current recursion depth (internal).

        Returns:
            ``SwarmResult`` with aggregated results from all subtasks.
        """
        swarm_id = f"swarm:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Register coalition node
        coalition = SwarmCoalitionNode(
            id=swarm_id,
            name=f"Swarm: {task[:60]}",
            task_description=task,
            timestamp=ts,
            status="active",
        )
        self.ogm.upsert(coalition)

        # Decompose
        tree = self.decompose_task(task, deps)

        if not tree.subtasks:
            # Leaf node — execute directly
            result = await self._execute_leaf(task, deps, swarm_id)
            coalition.status = "completed"
            coalition.agents_spawned = 1
            coalition.depth_reached = depth
            self.ogm.upsert(coalition)
            return SwarmResult(
                swarm_id=swarm_id,
                agents_spawned=1,
                depth_reached=depth,
                results={"root": result},
                parallelism_achieved=0.0,
            )

        # Separate parallel and sequential subtasks
        parallel_tasks = [st for st in tree.subtasks if st.parallelizable]
        sequential_tasks = [st for st in tree.subtasks if not st.parallelizable]

        results: dict[str, Any] = {}
        total_agents = 0

        # Execute parallel subtasks
        if parallel_tasks:
            if depth < self.max_depth and total_agents < self.max_agents:
                parallel_results = await self._execute_parallel(
                    parallel_tasks, deps, swarm_id, depth
                )
                results.update(parallel_results)
                total_agents += len(parallel_tasks)

        # Execute sequential subtasks
        for st in sequential_tasks:
            if total_agents >= self.max_agents:
                results[st.task] = "Skipped: max agents reached"
                continue
            if depth < self.max_depth:
                sub_result = await self.decompose_and_spawn(st.task, deps, depth + 1)
                results[st.task] = sub_result.results
                total_agents += sub_result.agents_spawned

        # Update coalition
        parallelism = (
            len(parallel_tasks) / max(len(tree.subtasks), 1) if tree.subtasks else 0.0
        )
        coalition.status = "completed"
        coalition.agents_spawned = total_agents
        coalition.depth_reached = depth
        coalition.parallelism_achieved = parallelism
        self.ogm.upsert(coalition)

        return SwarmResult(
            swarm_id=swarm_id,
            agents_spawned=total_agents,
            depth_reached=depth,
            results=results,
            parallelism_achieved=parallelism,
        )

    async def _execute_leaf(
        self,
        task: str,
        deps: GraphDeps,
        swarm_id: str,
    ) -> str:
        """Execute a leaf-node task using the best-matching specialist.

        Args:
            task: The leaf task description.
            deps: Runtime dependencies.
            swarm_id: Parent swarm coalition ID.

        Returns:
            The specialist's output string.
        """
        # Find best specialist
        available = list(deps.nodes.keys()) if deps.nodes else []
        if not available:
            return f"No specialists available for: {task}"

        ranked = self.rank_specialists(task, available, top_k=1)
        if not ranked:
            return f"No matching specialist for: {task}"

        best_id, score = ranked[0]
        logger.info(
            "Swarm leaf: %s → specialist %s (affinity=%.2f)", task[:40], best_id, score
        )

        # Execute via the specialist agent if available
        agent = deps.sub_agents.get(best_id)
        if isinstance(agent, Agent):
            try:
                result = await asyncio.wait_for(
                    agent.run(task, deps=deps),
                    timeout=120.0,
                )
                return str(result.output)
            except Exception as e:
                return f"Specialist {best_id} failed: {e}"

        return f"Specialist {best_id} matched but not executable (score={score:.2f})"

    async def _execute_parallel(
        self,
        subtasks: list[TaskTree],
        deps: GraphDeps,
        swarm_id: str,
        depth: int,
    ) -> dict[str, Any]:
        """Execute multiple subtasks in parallel.

        Uses ``asyncio.gather`` for concurrent execution of independent
        subtasks, mirroring the existing ``run_orthogonal_regions`` pattern.

        Args:
            subtasks: List of parallelizable subtasks.
            deps: Runtime dependencies.
            swarm_id: Parent swarm coalition ID.
            depth: Current recursion depth.

        Returns:
            Dict mapping task description → result.
        """

        async def run_one(st: TaskTree) -> tuple[str, Any]:
            try:
                sub_result = await self.decompose_and_spawn(st.task, deps, depth + 1)
                return (st.task, sub_result.results)
            except Exception as e:
                logger.warning("Parallel subtask failed: %s — %s", st.task[:40], e)
                return (st.task, f"Error: {e}")

        tasks = [run_one(st) for st in subtasks]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, Any] = {}
        for r in raw_results:
            if isinstance(r, tuple):
                results[r[0]] = r[1]
            elif isinstance(r, Exception):
                results[f"error_{id(r)}"] = str(r)

        return results
