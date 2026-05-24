"""CONCEPT:ORCH-1.25 — Parallel Engine.

The single engine that handles every execution from a trivial 1-agent
LLM call to a 300-agent enterprise swarm. The **same code path** runs
for all scales.

Replaces the following fragmented systems:
    - ``DynamicSubgraphOrchestrator`` (team execution)
    - ``HeavyThinkingOrchestrator`` (parallel reasoning + deliberation)
    - ``RLMEnvironment.run_parallel_sub_calls()`` (parallel sub-calls)
    - ``SubagentPatternRouter`` (pattern selection)
    - ``CoordinationLayer`` (protocol selection — now a subcomponent)
    - ``WorkflowRunner`` (wave-based batch execution)

Execution flow:
    1. Receive ``ExecutionManifest`` (from planner, workflow, skill, or preset)
    2. Resolve ``auto`` fields (execution_mode, synthesis, coordination)
    3. Build dependency DAG from agent specs
    4. Chromatic-schedule parallel groups into waves
    5. Execute waves with semaphore-governed concurrency
    6. Synthesize outputs using RLM-native strategy
    7. Persist results to KG
    8. Return ``ExecutionResult``

See docs/pillars/1_graph_orchestration/ORCH-1.25-Parallel_Engine.md
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

import networkx as nx
from pydantic_ai import Agent

from agent_utilities.core.config import config

from ..models.execution_manifest import (
    AgentExecutionResult,
    AgentSpec,
    ExecutionManifest,
    ExecutionResult,
    SynthesisSpec,
    WaveResult,
)
from .coordination import CoordinationLayer

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from .state import GraphDeps

logger = logging.getLogger(__name__)


# ── Circuit Breaker ─────────────────────────────────────────────────


class _CircuitBreaker:
    """Per-agent-type circuit breaker.

    CONCEPT:ORCH-1.25 — Parallel Engine

    Tracks consecutive failures per agent type. When failures exceed
    ``threshold``, the agent type is disabled and skipped in subsequent
    waves until the breaker is reset.
    """

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._failures: dict[str, int] = {}

    def record_failure(self, agent_id: str) -> None:
        self._failures[agent_id] = self._failures.get(agent_id, 0) + 1

    def record_success(self, agent_id: str) -> None:
        self._failures.pop(agent_id, None)

    def is_open(self, agent_id: str) -> bool:
        return self._failures.get(agent_id, 0) >= self.threshold

    def reset(self, agent_id: str | None = None) -> None:
        if agent_id:
            self._failures.pop(agent_id, None)
        else:
            self._failures.clear()


# ── Parallel Engine ─────────────────────────────────────────────────


class ParallelEngine:
    """CONCEPT:ORCH-1.25 — Parallel Engine.

    The single engine that handles every agent execution from 1 to 300+
    agents. Uses ``asyncio.Semaphore`` for concurrency backpressure and
    wave-based scheduling for dependency ordering.

    Key design: The SAME code path handles:
        - 1 agent (trivial query → inline execution)
        - 3-5 agents (team of specialists → standard parallel)
        - 10-50 agents (department-scale → wave batching)
        - 50-300+ agents (enterprise swarm → hierarchical synthesis)

    Args:
        engine: Optional KG engine for persistence and topology lookups.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None) -> None:
        self.engine = engine
        self.coordination = CoordinationLayer(engine=engine)
        self._circuit_breaker = _CircuitBreaker(
            threshold=getattr(config, "circuit_breaker_threshold", 3)
        )
        from ..capabilities.auto_healing import AutoHealingEngine

        self.auto_healing = AutoHealingEngine(
            skill_evolver=None,
            fallback_router=None,
            enabled=getattr(config, "enable_auto_healing", False),
        )

    # ── Public API ──────────────────────────────────────────────────

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. This is the **only** entry point.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            manifest: The execution specification.
            graph_deps: Optional graph runtime dependencies.

        Returns:
            Complete ``ExecutionResult`` with synthesis output and per-wave results.
        """
        start_time = time.monotonic()

        # 1. Resolve auto-configuration
        resolved = self._resolve_manifest(manifest)

        logger.info(
            "[CONCEPT:ORCH-1.25] Executing manifest '%s' — %d agents, mode=%s, "
            "synthesis=%s, source=%s",
            resolved.name or resolved.manifest_id,
            resolved.agent_count,
            resolved.execution_mode,
            resolved.synthesis.strategy,
            resolved.source or "direct",
        )

        # 2. Build DAG and schedule waves
        waves = self._schedule_waves(resolved)

        # 3. Select coordination protocol
        protocol = self.coordination.select_protocol(
            agent_count=resolved.agent_count,
            execution_mode=resolved.execution_mode,
        )

        # 4. Execute waves with backpressure
        concurrency = resolved.max_concurrency
        if concurrency is None:
            concurrency = getattr(config, "max_parallel_agents", 60) or 60
        semaphore = asyncio.Semaphore(int(concurrency))
        wave_results: list[WaveResult] = []

        for wave_idx, wave_agents in enumerate(waves):
            logger.info(
                "[CONCEPT:ORCH-1.25] Wave %d/%d — %d agents",
                wave_idx + 1,
                len(waves),
                len(wave_agents),
            )

            wave_result = await self._execute_wave(
                wave_agents, wave_idx, semaphore, resolved, graph_deps, wave_results
            )
            wave_results.append(wave_result)

            logger.info(
                "[CONCEPT:ORCH-1.25] Wave %d complete — success_rate=%.1f%%, "
                "duration=%.0fms",
                wave_idx + 1,
                wave_result.success_rate * 100,
                wave_result.duration_ms,
            )

        # 5. Synthesize outputs (RLM-native)
        all_results = [r for w in wave_results for r in w.results]
        synthesis_output = await self._synthesize(
            all_results, resolved.synthesis, resolved.query, graph_deps
        )

        # Adversarial verification on final run synthesized output
        from ..capabilities.adversarial_verifier import ADVERSARIAL_ENABLED

        if ADVERSARIAL_ENABLED:
            try:
                from ..capabilities.adversarial_verifier import run_adversarial_pass

                # Mock GraphState/Deps if missing
                class MockGraphState:
                    def __init__(self, q):
                        self.query = q
                        self.mode = "execute"
                        self.signal_board = {}

                class MockGraphDeps:
                    def __init__(self, model, eq=None):
                        self.agent_model = model
                        self.verifier_timeout = 120.0
                        self.event_queue = eq

                from typing import cast

                from ..graph.state import GraphDeps, GraphState

                m_state = cast(GraphState, MockGraphState(resolved.query))
                model_id = resolved.synthesis.model_id or (
                    str(graph_deps.agent_model) if graph_deps else "openai:gpt-4o-mini"
                )
                m_deps = cast(
                    GraphDeps,
                    MockGraphDeps(
                        model_id, graph_deps.event_queue if graph_deps else None
                    ),
                )

                logger.info(
                    "[CONCEPT:AHE-3.1] Running final adversarial verification pass..."
                )
                adv_res = await run_adversarial_pass(m_state, m_deps, synthesis_output)
                if adv_res and adv_res.vulnerabilities_found:
                    logger.warning(
                        "[CONCEPT:AHE-3.1] Adversarial pass found vulnerabilities: %s",
                        adv_res.findings,
                    )
                    # Attach findings to resolved metadata or final execution log
                    resolved.metadata["adversarial_findings"] = adv_res.findings
            except Exception as adv_err:
                logger.warning("Adversarial pass failed (non-fatal): %s", adv_err)

        total_duration = (time.monotonic() - start_time) * 1000

        # 6. Persist to KG
        execution_id = self._persist_execution(resolved, wave_results, synthesis_output)

        result = ExecutionResult(
            manifest_id=resolved.manifest_id,
            execution_id=execution_id,
            synthesis_output=synthesis_output,
            wave_results=wave_results,
            agent_count=resolved.agent_count,
            protocol=protocol.name,
            total_duration_ms=total_duration,
            synthesis_strategy=resolved.synthesis.strategy,
            success=all(r.success for r in all_results) if all_results else True,
        )

        logger.info(
            "[CONCEPT:ORCH-1.25] Execution complete — %d agents, %d waves, "
            "%.0fms total, success=%s",
            result.agent_count,
            len(wave_results),
            total_duration,
            result.success,
        )

        return result

    # ── Manifest Resolution ─────────────────────────────────────────

    def _resolve_manifest(self, manifest: ExecutionManifest) -> ExecutionManifest:
        """Resolve ``auto`` fields based on agent count and complexity.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Auto-resolution rules:
            - execution_mode: sequential (1), parallel (≤5), wave (>5)
            - synthesis: flat (≤10), hierarchical (≤50), rlm (>50)
            - coordination: delegation (1), consensus (2), voting (3+)
        """
        resolved = copy.deepcopy(manifest)

        if resolved.execution_mode == "auto":
            if resolved.is_trivial:
                resolved.execution_mode = "sequential"
            elif resolved.agent_count <= 5 and not resolved.has_dependencies:
                resolved.execution_mode = "parallel"
            else:
                resolved.execution_mode = "wave"

        if resolved.synthesis.strategy == "auto":
            if resolved.agent_count <= 1:
                resolved.synthesis.strategy = "flat"
            elif resolved.agent_count <= 10:
                resolved.synthesis.strategy = "flat"
            elif resolved.agent_count <= 50:
                resolved.synthesis.strategy = "hierarchical"
            else:
                resolved.synthesis.strategy = "rlm"

        return resolved

    # ── DAG Scheduling ──────────────────────────────────────────────

    def _schedule_waves(self, manifest: ExecutionManifest) -> list[list[AgentSpec]]:
        """Build a dependency DAG and schedule agents into execution waves.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Uses topological sort on the dependency graph to determine
        execution order, then groups agents by topological level
        into parallel waves.

        Args:
            manifest: Resolved execution manifest.

        Returns:
            List of waves, each containing agents that can run concurrently.
        """
        if manifest.execution_mode == "sequential":
            # Each agent is its own wave
            return [[a] for a in self._expand_partitions(manifest)]

        expanded = self._expand_partitions(manifest)

        if not manifest.has_dependencies:
            # No DAG — batch by configured batch size
            b_size = manifest.batch_size
            if b_size is None:
                b_size = getattr(config, "parallel_batch_size", 25) or 25
            batch_size = int(b_size)
            waves = []
            for i in range(0, len(expanded), batch_size):
                waves.append(expanded[i : i + batch_size])
            return waves

        # Build DAG from depends_on edges
        dag = nx.DiGraph()
        agent_map: dict[str, AgentSpec] = {}

        for agent in expanded:
            dag.add_node(agent.agent_id)
            agent_map[agent.agent_id] = agent
            for dep in agent.depends_on:
                if dep in {a.agent_id for a in expanded}:
                    dag.add_edge(dep, agent.agent_id)

        # Group by topological generation (parallel levels)
        try:
            generations = list(nx.topological_generations(dag))
        except nx.NetworkXUnfeasible:
            logger.warning(
                "[CONCEPT:ORCH-1.25] Dependency cycle detected — falling back "
                "to sequential execution"
            )
            return [[a] for a in expanded]

        topological_waves: list[list[AgentSpec]] = []
        b_size = manifest.batch_size
        if b_size is None:
            b_size = getattr(config, "parallel_batch_size", 25) or 25
        batch_size = int(b_size)

        for generation in generations:
            gen_agents = [agent_map[nid] for nid in generation if nid in agent_map]
            # Sub-batch within a generation if it exceeds batch_size
            for i in range(0, len(gen_agents), batch_size):
                topological_waves.append(gen_agents[i : i + batch_size])

        return topological_waves

    def _expand_partitions(self, manifest: ExecutionManifest) -> list[AgentSpec]:
        """Expand fan-out partitions into individual agent specs.

        CONCEPT:ORCH-1.25 — Parallel Engine

        If an ``AgentSpec`` has partitions, create one copy per partition
        with ``{{partition}}`` replaced in the task template and a unique
        agent_id suffix.
        """
        expanded: list[AgentSpec] = []
        for agent in manifest.agents:
            if agent.partitions:
                for partition in agent.partitions:
                    expanded_agent = agent.model_copy(deep=True)
                    expanded_agent.agent_id = f"{agent.agent_id}:{partition}"
                    expanded_agent.task_template = agent.task_template.replace(
                        "{{partition}}", partition
                    )
                    expanded_agent.partitions = []  # Already expanded
                    expanded.append(expanded_agent)
            else:
                expanded.append(agent)
        return expanded

    # ── Wave Execution ──────────────────────────────────────────────

    async def _execute_wave(
        self,
        agents: list[AgentSpec],
        wave_idx: int,
        semaphore: asyncio.Semaphore,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
    ) -> WaveResult:
        """Execute one wave of agents concurrently with semaphore backpressure.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            agents: Agents in this wave.
            wave_idx: Zero-based wave index.
            semaphore: Concurrency governor.
            manifest: The full manifest for context.
            graph_deps: Optional runtime dependencies.
            wave_results: Accumulated results from preceding waves.

        Returns:
            ``WaveResult`` with all agent outcomes.
        """
        start_time = time.monotonic()

        async def _run_one(agent: AgentSpec) -> AgentExecutionResult:
            if self._circuit_breaker.is_open(agent.agent_id):
                return AgentExecutionResult(
                    agent_id=agent.agent_id,
                    role=agent.role,
                    success=False,
                    error=f"Circuit breaker open for {agent.agent_id}",
                )

            async with semaphore:
                return await self._execute_agent(
                    agent, manifest, graph_deps, wave_results
                )

        tasks = [_run_one(a) for a in agents]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[AgentExecutionResult] = []
        for raw in raw_results:
            if isinstance(raw, AgentExecutionResult):
                results.append(raw)
                # Update circuit breaker
                if raw.success:
                    self._circuit_breaker.record_success(raw.agent_id)
                else:
                    self._circuit_breaker.record_failure(raw.agent_id)
            elif isinstance(raw, Exception):
                results.append(
                    AgentExecutionResult(
                        agent_id="unknown",
                        success=False,
                        error=str(raw),
                    )
                )

        duration_ms = (time.monotonic() - start_time) * 1000
        return WaveResult(
            wave_index=wave_idx,
            results=results,
            duration_ms=duration_ms,
        )

    async def _execute_agent(
        self,
        agent: AgentSpec,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
    ) -> AgentExecutionResult:
        """Execute a single agent invocation with full capability wiring.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            agent: The agent specification.
            manifest: The parent manifest for shared context.
            graph_deps: Optional runtime dependencies.
            wave_results: Preceding wave results for context injection.

        Returns:
            ``AgentExecutionResult`` with the agent's output.
        """
        start_time = time.monotonic()
        timeout = agent.timeout or getattr(config, "agent_execution_timeout", 120.0)

        # Build the task prompt
        task = agent.task_template or manifest.query

        # Ingest dependency outputs (Fan-In / Fan-Out topological context flow)
        dependency_contexts = []
        if agent.depends_on:
            for dep_id in agent.depends_on:
                for wave_res in wave_results:
                    for agent_res in wave_res.results:
                        if (
                            agent_res.agent_id == dep_id
                            or agent_res.agent_id.startswith(f"{dep_id}:")
                        ) and agent_res.success:
                            role_str = (
                                f"Role: {agent_res.role}" if agent_res.role else ""
                            )
                            part_str = (
                                f", Partition: {agent_res.partition}"
                                if agent_res.partition
                                else ""
                            )
                            dependency_contexts.append(
                                f"### Output from dependent agent '{agent_res.agent_id}' ({role_str}{part_str}):\n"
                                f"{agent_res.output}"
                            )

        if dependency_contexts:
            dep_text = "\n\n".join(dependency_contexts)
            task = (
                f"{task}\n\n"
                f"## DEPENDENCY OUTPUTS\n"
                f"The following dependent upstream steps have completed successfully. "
                f"Use their outputs to complete your task:\n\n"
                f"{dep_text}"
            )

        if manifest.context:
            task = f"{task}\n\nContext:\n{manifest.context}"

        # Determine model
        model_id = agent.model_id
        if not model_id and graph_deps:
            model_id = str(graph_deps.agent_model)
        if not model_id:
            model_id = "openai:gpt-4o-mini"  # Fallback

        system_prompt = agent.system_prompt or (
            f"You are a {agent.role or agent.agent_id} specialist agent. "
            f"Provide your best analysis and response."
        )

        try:
            from ..agent.factory import create_agent

            # Setup provider & model override
            provider = "openai"
            prov_model = model_id
            if ":" in model_id:
                provider, prov_model = model_id.split(":", 1)

            # Map manifest settings / metadata
            metadata = manifest.metadata or {}
            from ..capabilities.checkpointing import CheckpointStore

            checkpoint_store: CheckpointStore | None = None
            if metadata.get("checkpoint_store") == "file":
                from ..capabilities.checkpointing import FileCheckpointStore

                checkpoint_store = FileCheckpointStore(
                    directory=metadata.get("checkpoint_dir", "./checkpoints")
                )
            elif metadata.get("checkpoint_store") == "graph":
                from ..capabilities.checkpointing import GraphCheckpointStore

                checkpoint_store = GraphCheckpointStore(engine=self.engine)

            # Wire up all 8 capabilities natively using factory
            llm_agent, _ = create_agent(
                provider=provider,
                model_id=prov_model,
                system_prompt=system_prompt,
                name=agent.agent_id,
                enable_skills=True,
                enable_universal_tools=True,
                mcp_config=metadata.get("mcp_config"),
                tool_tags=agent.tools,
                stuck_loop_detection=metadata.get("stuck_loop_detection", True),
                stuck_loop_max_repeated=metadata.get("stuck_loop_max_repeated", 3),
                context_warnings=metadata.get("context_warnings", True),
                max_context_tokens=metadata.get("max_context_tokens"),
                output_eviction=metadata.get("output_eviction", True),
                eviction_threshold_chars=metadata.get(
                    "eviction_threshold_chars", 80_000
                ),
                include_checkpoints=metadata.get("include_checkpoints", False),
                checkpoint_store=checkpoint_store,
                checkpoint_frequency=metadata.get("checkpoint_frequency", "every_tool"),
                include_teams=metadata.get("include_teams", False),
            )

            result = await asyncio.wait_for(
                llm_agent.run(task),
                timeout=timeout,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            output = result.output

            logger.debug(
                "[CONCEPT:ORCH-1.25] Agent %s (%s) completed in %.0fms — "
                "output=%d chars",
                agent.agent_id,
                agent.role,
                duration_ms,
                len(output),
            )

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                partition=agent.partitions[0] if agent.partitions else "",
                output=output,
                success=True,
                duration_ms=duration_ms,
                model_id=model_id,
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[CONCEPT:ORCH-1.25] Agent %s timed out after %.0fms",
                agent.agent_id,
                duration_ms,
            )
            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                success=False,
                error=f"Timeout after {duration_ms:.0f}ms",
                duration_ms=duration_ms,
                model_id=model_id,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[CONCEPT:ORCH-1.25] Agent %s failed: %s",
                agent.agent_id,
                e,
            )
            # Register failure with Auto-Healing retry engine
            try:
                self.auto_healing.report_failure(
                    task_name=agent.agent_id,
                    error_context=str(e),
                )
            except Exception as ah_err:
                logger.debug("Auto-healing trigger failed: %s", ah_err)

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                model_id=model_id,
            )

    # ── Output Synthesis ────────────────────────────────────────────

    async def _synthesize(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Synthesize agent outputs using the specified strategy.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        The key insight: outputs are stored as Pydantic objects and
        processed programmatically, never dumped into context windows.

        Args:
            results: All agent execution results.
            spec: Synthesis specification.
            query: Original user query.
            graph_deps: Optional runtime dependencies.

        Returns:
            Synthesized output string.
        """
        successful = [r for r in results if r.success]
        if not successful:
            return "No successful agent outputs to synthesize."

        if len(successful) == 1:
            return successful[0].output

        if spec.strategy == "flat":
            return self._flat_synthesis(successful)

        elif spec.strategy == "hierarchical":
            return await self._hierarchical_synthesis(
                successful, spec, query, graph_deps
            )

        elif spec.strategy == "rlm":
            return await self._rlm_synthesis(successful, spec, query, graph_deps)

        elif spec.strategy == "progressive":
            return await self._progressive_synthesis(
                successful, spec, query, graph_deps
            )

        # Default fallback
        return self._flat_synthesis(successful)

    def _flat_synthesis(self, results: list[AgentExecutionResult]) -> str:
        """Simple concatenation synthesis for small agent counts.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis
        """
        parts: list[str] = []
        for r in results:
            header = f"## {r.role or r.agent_id}"
            if r.partition:
                header += f" [{r.partition}]"
            parts.append(f"{header}\n\n{r.output}")
        return "\n\n---\n\n".join(parts)

    async def _hierarchical_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Tiered synthesis: group → sub-summaries → final summary.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Groups outputs by ``spec.ratio`` (default 10), generates a
        sub-summary for each group, then synthesizes sub-summaries
        into a final output. Recurses if needed for very large sets.
        """
        ratio = spec.ratio

        # Base case: small enough for direct synthesis
        if len(results) <= ratio:
            return await self._synthesize_group(results, query, graph_deps)

        # Tier 1: Create sub-summaries
        sub_summaries: list[str] = []
        for i in range(0, len(results), ratio):
            group = results[i : i + ratio]
            summary = await self._synthesize_group(group, query, graph_deps)
            sub_summaries.append(summary)

        logger.info(
            "[CONCEPT:ORCH-1.26] Hierarchical synthesis: %d results → "
            "%d sub-summaries → final",
            len(results),
            len(sub_summaries),
        )

        # Tier 2: Final synthesis of sub-summaries
        if len(sub_summaries) > ratio:
            # Recurse for very large sets
            pseudo_results = [
                AgentExecutionResult(
                    agent_id=f"sub_summary_{i}",
                    output=s,
                    success=True,
                )
                for i, s in enumerate(sub_summaries)
            ]
            return await self._hierarchical_synthesis(
                pseudo_results, spec, query, graph_deps
            )

        return await self._synthesize_group(
            [
                AgentExecutionResult(
                    agent_id=f"sub_summary_{i}",
                    output=s,
                    success=True,
                )
                for i, s in enumerate(sub_summaries)
            ],
            query,
            graph_deps,
        )

    async def _rlm_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Full RLM synthesis for massive-scale (50+ agent) output processing.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Uses the RLM environment to programmatically process outputs
        stored as Pydantic objects, not dumped into the context window.
        Falls back to hierarchical synthesis if RLM is unavailable.
        """
        try:
            from ..rlm.config import RLMConfig
            from ..rlm.repl import RLMEnvironment

            # Serialize outputs as environment context
            outputs_json = json.dumps(
                [
                    {
                        "agent_id": r.agent_id,
                        "role": r.role,
                        "partition": r.partition,
                        "output": r.output[:2000],  # Truncate for metadata
                        "success": r.success,
                    }
                    for r in results
                ],
                indent=2,
            )

            rlm_config = RLMConfig(
                metadata_only_root=True,
                async_enabled=True,
            )

            env = RLMEnvironment(
                context=outputs_json,
                config=rlm_config,
                graph_deps=graph_deps,
            )

            return await env.run_full_rlm(
                f"Synthesize {len(results)} agent outputs for query: {query}"
            )

        except Exception as e:
            logger.warning(
                "[CONCEPT:ORCH-1.26] RLM synthesis failed, falling back to "
                "hierarchical: %s",
                e,
            )
            return await self._hierarchical_synthesis(results, spec, query, graph_deps)

    async def _progressive_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Progressive synthesis: incrementally merge as results arrive.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Processes results one at a time, maintaining a running summary
        that grows as each agent's output is incorporated.
        """
        if not results:
            return ""

        running_summary = results[0].output

        for r in results[1:]:
            running_summary = await self._merge_pair(
                running_summary, r, query, graph_deps
            )

        return running_summary

    async def _synthesize_group(
        self,
        results: list[AgentExecutionResult],
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Synthesize a group of results using an LLM.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis
        """
        model_id = "openai:gpt-4o-mini"
        if graph_deps:
            model_id = str(graph_deps.agent_model)

        combined = "\n\n---\n\n".join(
            f"**{r.role or r.agent_id}**: {r.output}" for r in results
        )

        try:
            synthesizer = Agent(
                model=model_id,
                system_prompt=(
                    "You are a synthesis agent. Merge the following agent outputs "
                    "into a single coherent response. Preserve key findings, "
                    "resolve contradictions, and maintain provenance."
                ),
            )

            result = await asyncio.wait_for(
                synthesizer.run(f"Query: {query}\n\nAgent Outputs:\n{combined}"),
                timeout=120.0,
            )
            return result.output
        except Exception as e:
            logger.warning(
                "[CONCEPT:ORCH-1.26] Group synthesis failed, using flat: %s", e
            )
            return combined

    async def _merge_pair(
        self,
        running_summary: str,
        new_result: AgentExecutionResult,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Merge a new result into the running summary.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis
        """
        model_id = "openai:gpt-4o-mini"
        if graph_deps:
            model_id = str(graph_deps.agent_model)

        try:
            merger = Agent(
                model=model_id,
                system_prompt=(
                    "Merge the new agent output into the existing summary. "
                    "Add new information, resolve conflicts, keep it concise."
                ),
            )
            result = await asyncio.wait_for(
                merger.run(
                    f"Existing summary:\n{running_summary}\n\n"
                    f"New output from {new_result.role or new_result.agent_id}:\n"
                    f"{new_result.output}"
                ),
                timeout=60.0,
            )
            return result.output
        except Exception:
            return f"{running_summary}\n\n---\n\n{new_result.output}"

    # ── KG Persistence ──────────────────────────────────────────────

    def _persist_execution(
        self,
        manifest: ExecutionManifest,
        wave_results: list[WaveResult],
        synthesis_output: str,
    ) -> str:
        """Persist execution results to the Knowledge Graph with verbose hierarchy.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Creates a ``ParallelExecution`` node, individual ``AgentExecutionResult`` nodes
        linked via ``PART_OF_EXECUTION`` edges, and dependency edges linked via
        ``DEPENDS_ON`` edges.

        Args:
            manifest: The executed manifest.
            wave_results: Per-wave results.
            synthesis_output: Final synthesis output.

        Returns:
            The execution node ID.
        """
        execution_id = f"pe:{uuid.uuid4().hex[:8]}"

        if self.engine is None:
            return execution_id

        try:
            all_results = [r for w in wave_results for r in w.results]
            total_duration = sum(w.duration_ms for w in wave_results)
            success_count = sum(1 for r in all_results if r.success)

            node_data = {
                "id": execution_id,
                "type": "ParallelExecution",
                "name": f"PE: {manifest.name or manifest.manifest_id}",
                "manifest_id": manifest.manifest_id,
                "agent_count": manifest.agent_count,
                "wave_count": len(wave_results),
                "success_count": success_count,
                "failure_count": len(all_results) - success_count,
                "total_duration_ms": total_duration,
                "synthesis_strategy": manifest.synthesis.strategy,
                "execution_mode": manifest.execution_mode,
                "source": manifest.source,
                "synthesis_preview": synthesis_output[:500],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "importance_score": 0.7,
            }

            self.engine.graph.add_node(execution_id, **node_data)

            # Persist individual AgentExecutionResult nodes and connect them
            kg_node_map = {}
            for result in all_results:
                node_uuid = f"agent_exec_res:{uuid.uuid4().hex[:8]}"
                res_data = {
                    "id": node_uuid,
                    "type": "AgentExecutionResult",
                    "agent_id": result.agent_id,
                    "role": result.role,
                    "partition": result.partition,
                    "success": result.success,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "model_id": result.model_id,
                    "output_preview": result.output[:500] if result.output else "",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.engine.graph.add_node(node_uuid, **res_data)
                self.engine.graph.add_edge(
                    execution_id, node_uuid, type="PART_OF_EXECUTION"
                )
                kg_node_map[result.agent_id] = node_uuid

            # Reconstruct and persist dependency topology edges inside KG
            for agent_spec in manifest.agents:
                for dep in agent_spec.depends_on:
                    source_kg = kg_node_map.get(dep)
                    target_kg = kg_node_map.get(agent_spec.agent_id)
                    if source_kg and target_kg:
                        self.engine.graph.add_edge(
                            source_kg, target_kg, type="DEPENDS_ON"
                        )

            logger.info(
                "[CONCEPT:ORCH-1.25] Persisted execution hierarchy %s to KG "
                "(%d agents, %d waves, %d topology edges)",
                execution_id,
                manifest.agent_count,
                len(wave_results),
                sum(len(a.depends_on) for a in manifest.agents),
            )

        except Exception as e:
            logger.debug("ParallelEngine: KG persistence failed: %s", e)

        return execution_id
