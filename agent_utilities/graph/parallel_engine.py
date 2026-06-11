"""CONCEPT:ORCH-1.8 — Parallel Engine.

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

See docs/pillars/1_graph_orchestration/ORCH-1.8-Parallel_Engine.md
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from agent_utilities.core.config import config
from agent_utilities.knowledge_graph.core import graph_primitives as rx
from agent_utilities.knowledge_graph.core.engine_breaker import CircuitBreaker
from agent_utilities.orchestration.resilience import (
    ResiliencePolicy,
    run_with_resilience,
)

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


class AgentBreakerOpenError(ConnectionError):
    """A chronically failing agent type's circuit is open — skip it this wave."""


class AgentAttemptFailedError(Exception):
    """One in-wave agent attempt produced an unsuccessful result (retryable).

    Carries the failed :class:`AgentExecutionResult` so the wave keeps the
    last attempt's result when retries are exhausted (SWARM-5 semantics).
    """

    def __init__(self, result: AgentExecutionResult) -> None:
        self.result = result
        super().__init__(result.error or "agent attempt failed")


class AgentTypeCircuitBreaker(CircuitBreaker):
    """The canonical OS-5.23 breaker state machine, per parallel-engine agent type.

    CONCEPT:ORCH-1.8 — Parallel Engine

    Subclass-parameterized exactly like the multiplexer's per-child breaker
    (CONCEPT:ECO-4.34). Profile differences from the engine-client breaker:

    * ``cooldown`` is infinite — once open, the agent type stays disabled for
      subsequent waves until a recorded success (the historical ORCH-1.8
      semantics had no half-open probe window).
    * No gauge export — agent-type ids are unbounded per run, so they must
      not become Prometheus label values.

    Note the canonical ``threshold=0 disables the breaker`` convention now
    applies (the deleted fork treated 0 as "always open", a footgun).
    """

    error_cls = AgentBreakerOpenError
    subject = "parallel-engine agent type"

    def __init__(self, agent_id: str, threshold: int) -> None:
        super().__init__(agent_id, threshold=threshold, cooldown=float("inf"))

    def _export_state(self) -> None:
        return None


# ── Swarm helpers — CONCEPT:ORCH-1.32 KG-Governed Agent Swarm


def enforce_structured_output(output: str, schema: str | None) -> tuple[bool, str]:
    """SWARM-4: validate a sub-agent output against an expected JSON shape (pure, testable).

    Kimi guardrail #3 — "prose from intermediate agents creates downstream parsing failures."
    Returns ``(ok, detail)``. When ``schema`` is falsy this is a no-op pass. We validate that the
    output parses as JSON (tolerating a ```json fenced block); structural key-presence is a
    best-effort check when the schema names top-level keys.
    """
    if not schema:
        return True, "no schema"
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text[:4].lower() == "json" else text
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"not valid JSON: {e}"
    # best-effort: if the schema names required keys (JSON object or comma list), check presence
    required: list[str] = []
    try:
        sj = json.loads(schema)
        if isinstance(sj, dict):
            required = list(sj.get("required") or sj.keys())
    except (json.JSONDecodeError, ValueError):
        required = [
            k.strip()
            for k in schema.replace("{", "").replace("}", "").split(",")
            if k.strip()
        ]
    if isinstance(parsed, dict) and required:
        missing = [k for k in required if k not in parsed]
        if missing:
            return False, f"missing keys: {missing}"
    return True, "ok"


def resolve_model_role(role: str) -> str:
    """SWARM-6: resolve an ``AgentSpec.model_role`` to a concrete ``provider:model`` id, or "".

    Heterogeneous swarm (Claw Groups) — different models per agent role (e.g. reasoning vs bulk vs
    local). Routes through the existing model-role registry; returns "" when unresolvable so the
    caller falls back to the manifest/default model.
    """
    if not role:
        return ""
    try:
        from ..rlm.roles import rlm_role_model

        resolved = rlm_role_model(role, fallback="")
        return str(resolved or "")
    except Exception:  # noqa: BLE001 - role routing is best-effort; caller falls back on ""
        return ""


# ── Parallel Engine ─────────────────────────────────────────────────


class ParallelEngine:
    """CONCEPT:ORCH-1.8 — Parallel Engine.

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
        # ONE canonical breaker per agent type (state machine shared with the
        # engine client / multiplexer children — see AgentTypeCircuitBreaker).
        self._breaker_threshold = int(getattr(config, "circuit_breaker_threshold", 3))
        self._agent_breakers: dict[str, AgentTypeCircuitBreaker] = {}
        # Repeated-failure escalation — absorbed from the dormant
        # AutoHealingEngine shell (strangler-then-delete): the per-agent
        # failure threshold survives; at threshold the failure enters the
        # LIVE propose-only remediation chain — a failure_gap Concept topic
        # instead of never-wired skill_evolver hooks (CONCEPT:AHE-3.18).
        self._agent_failure_counts: dict[str, int] = {}
        # CONCEPT:ORCH-1.32 — schedule metadata (critical-path, parallelism) captured per run
        self._schedule_meta: dict[str, Any] = {}
        # CONCEPT:ORCH-1.32 — previous run's MASS latent-state distribution, for W1 drift
        self._prev_social_states: list[float] = []

    def _agent_breaker(self, agent_id: str) -> AgentTypeCircuitBreaker:
        """The shared per-agent-type breaker (created on first use)."""
        breaker = self._agent_breakers.get(agent_id)
        if breaker is None:
            breaker = self._agent_breakers[agent_id] = AgentTypeCircuitBreaker(
                agent_id, threshold=self._breaker_threshold
            )
        return breaker

    # ── Public API ──────────────────────────────────────────────────

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. This is the **only** entry point.

        CONCEPT:ORCH-1.8 — Parallel Engine

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
            "[CONCEPT:ORCH-1.8] Executing manifest '%s' — %d agents, mode=%s, "
            "synthesis=%s, source=%s",
            resolved.name or resolved.manifest_id,
            resolved.agent_count,
            resolved.execution_mode,
            resolved.synthesis.strategy,
            resolved.source or "direct",
        )

        # 2. Build DAG and schedule waves
        waves = self._schedule_waves(resolved)

        # Generate Mermaid diagram representing the execution topography
        mermaid_code = None
        try:
            from agent_utilities.workflows.visualizer import WorkflowVisualizer

            mermaid_code = WorkflowVisualizer.generate(resolved, waves)
            logger.info(
                "\n" + "=" * 80 + "\n"
                "[VISUALIZER] Deterministically Generated Mermaid Topography:\n\n"
                f"```mermaid\n{mermaid_code}\n```\n" + "=" * 80 + "\n"
            )
        except Exception as vis_err:
            logger.warning("Failed to generate workflow Mermaid diagram: %s", vis_err)

        # 3. Select and apply coordination protocol
        protocol = self.coordination.select_protocol(
            agent_count=resolved.agent_count,
            execution_mode=resolved.execution_mode,
        )

        # Apply protocol to establish consensus/voting mechanics
        agent_ids = [a.agent_id for a in resolved.agents]
        coordination_result = self.coordination.apply_protocol(
            protocol=protocol,
            agent_ids=agent_ids,
            task=resolved.query,
            task_type=resolved.metadata.get("task_type", "general"),
        )
        self.coordination.log_coordination_trace(coordination_result)

        # 4. Execute waves with backpressure
        concurrency = resolved.max_concurrency
        if concurrency is None:
            concurrency = getattr(config, "max_parallel_agents", 60) or 60

        from ..core.cognitive_scheduler import CognitiveScheduler

        scheduler = CognitiveScheduler(
            max_concurrent=int(concurrency), engine=self.engine
        )

        wave_results: list[WaveResult] = []

        for wave_idx, wave_agents in enumerate(waves):
            logger.info(
                "[CONCEPT:ORCH-1.8] Wave %d/%d — %d agents",
                wave_idx + 1,
                len(waves),
                len(wave_agents),
            )

            wave_result = await self._execute_wave(
                wave_agents, wave_idx, scheduler, resolved, graph_deps, wave_results
            )
            wave_results.append(wave_result)

            logger.info(
                "[CONCEPT:ORCH-1.8] Wave %d complete — success_rate=%.1f%%, "
                "duration=%.0fms",
                wave_idx + 1,
                wave_result.success_rate * 100,
                wave_result.duration_ms,
            )

        # 4b. SWARM-2: verify leaves against success_criteria + bounded re-dispatch (the
        # planner→execute→verify loop). Gated by metadata["verify"]; only agents declaring
        # success_criteria are checked. Runs before synthesis so the deliverable is assembled from
        # verified outputs.
        verification: dict[str, Any] = {}
        if resolved.metadata.get("verify"):
            verification = await self._verify_and_redispatch(
                resolved, wave_results, graph_deps
            )
            logger.info("[CONCEPT:ORCH-1.32] Verification pass: %s", verification)

        # 5. Synthesize outputs (RLM-native)
        all_results = [r for w in wave_results for r in w.results]

        # 5a. CONCEPT:ORCH-1.2 — Global Workspace Attention: score the specialists'
        # outputs, select winners, and broadcast them to the KG. The broadcast is the
        # training signal `executor.get_attention_score` reads back as each
        # specialist's runtime standing. Runs only with a shared engine and ≥2
        # successful outputs (consensus is meaningless for one); non-fatal.
        self._broadcast_workspace_attention(all_results, resolved)

        # 5b. CONCEPT:ORCH-1.32 — model the wave as a Multi-Agent Social System and
        # snapshot swarm health (archetype heterogeneity, topology variance,
        # co-evolution slope, W1 drift vs the previous run); non-fatal telemetry.
        social_health = self._social_swarm_health(all_results, resolved)

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

        # SWARM-3 + SWARM-7: critical-path + per-wave telemetry
        critical_path = int(
            self._schedule_meta.get("critical_path_length", len(wave_results))
        )
        parallelism = float(self._schedule_meta.get("parallelism_ratio", 1.0))
        telemetry = {
            "waves": [
                {
                    "index": w.wave_index,
                    "agents": len(w.results),
                    "duration_ms": round(w.duration_ms, 1),
                    "success_rate": round(w.success_rate, 3),
                }
                for w in wave_results
            ],
            "critical_path_length": critical_path,
            "parallelism_ratio": parallelism,
            "total_agents": resolved.agent_count,
            "wave_count": len(wave_results),
            "max_concurrency": int(concurrency),
        }
        # CONCEPT:ORCH-1.2 — surface GWT loop health (write/read counters +
        # suspected engine-instance mismatch) for observability.
        try:
            from .workspace_attention import workspace_attention_telemetry

            telemetry["workspace_attention"] = workspace_attention_telemetry()
        except Exception:  # pragma: no cover - telemetry is best-effort
            pass
        if social_health:
            telemetry["social_system"] = social_health

        result = ExecutionResult(
            manifest_id=resolved.manifest_id,
            execution_id=execution_id,
            synthesis_output=synthesis_output,
            mermaid=mermaid_code,
            wave_results=wave_results,
            agent_count=resolved.agent_count,
            protocol=protocol.name,
            total_duration_ms=total_duration,
            synthesis_strategy=resolved.synthesis.strategy,
            success=all(r.success for r in all_results) if all_results else True,
            critical_path_length=critical_path,
            parallelism_ratio=parallelism,
            wave_count=len(wave_results),
            verification=verification,
            telemetry=telemetry,
        )

        logger.info(
            "[CONCEPT:ORCH-1.8] Execution complete — %d agents, %d waves, "
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

        CONCEPT:ORCH-1.8 — Parallel Engine

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

        CONCEPT:ORCH-1.8 — Parallel Engine

        Uses topological sort on the dependency graph to determine
        execution order, then groups agents by topological level
        into parallel waves.

        Args:
            manifest: Resolved execution manifest.

        Returns:
            List of waves, each containing agents that can run concurrently.
        """
        if manifest.execution_mode == "sequential":
            # Each agent is its own wave — critical path == agent count
            seq = [[a] for a in self._expand_partitions(manifest)]
            self._schedule_meta = {
                "critical_path_length": len(seq),
                "parallelism_ratio": 1.0,
            }
            return seq

        expanded = self._expand_partitions(manifest)

        if not manifest.has_dependencies:
            # No DAG — all agents are independent; critical path == 1 (one logical level).
            # Wave count may be >1 only because of batch_size, not dependency depth.
            b_size = manifest.batch_size
            if b_size is None:
                b_size = getattr(config, "parallel_batch_size", 25) or 25
            batch_size = int(b_size)
            waves = []
            for i in range(0, len(expanded), batch_size):
                waves.append(expanded[i : i + batch_size])
            self._schedule_meta = {
                "critical_path_length": 1,
                "parallelism_ratio": float(len(expanded)),
            }
            return waves

        # Build DAG from depends_on edges using graph primitives
        dag = rx.PyDiGraph()
        agent_map: dict[str, AgentSpec] = {}
        node_indices: dict[str, int] = {}

        valid_ids = {a.agent_id for a in expanded}
        for agent in expanded:
            idx = dag.add_node(agent.agent_id)
            node_indices[agent.agent_id] = idx
            agent_map[agent.agent_id] = agent

        for agent in expanded:
            for dep in agent.depends_on:
                if dep in valid_ids:
                    dag.add_edge(node_indices[dep], node_indices[agent.agent_id], None)

        # Group by topological generation (parallel levels)
        try:
            generations = list(rx.topological_generations(dag))
        except Exception:
            logger.warning(
                "[CONCEPT:ORCH-1.8] Dependency cycle detected — falling back "
                "to sequential execution"
            )
            self._schedule_meta = {
                "critical_path_length": len(expanded),
                "parallelism_ratio": 1.0,
            }
            return [[a] for a in expanded]

        # SWARM-3: the critical path is the number of dependency generations (the longest chain),
        # NOT the wave count (which batch-splitting can inflate). Wall-clock floor ≈ critical path.
        n_gen = len(generations)
        self._schedule_meta = {
            "critical_path_length": max(1, n_gen),
            "parallelism_ratio": round(len(expanded) / max(1, n_gen), 2),
        }

        topological_waves: list[list[AgentSpec]] = []
        b_size = manifest.batch_size
        if b_size is None:
            b_size = getattr(config, "parallel_batch_size", 25) or 25
        batch_size = int(b_size)

        for generation in generations:
            gen_agents = [
                agent_map[dag[nidx]] for nidx in generation if dag[nidx] in agent_map
            ]
            # Sub-batch within a generation if it exceeds batch_size
            for i in range(0, len(gen_agents), batch_size):
                topological_waves.append(gen_agents[i : i + batch_size])

        return topological_waves

    def _expand_partitions(self, manifest: ExecutionManifest) -> list[AgentSpec]:
        """Expand fan-out partitions into individual agent specs.

        CONCEPT:ORCH-1.8 — Parallel Engine

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
        scheduler: Any,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
    ) -> WaveResult:
        """Execute one wave of agents concurrently with semaphore backpressure.

        CONCEPT:ORCH-1.8 — Parallel Engine

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

        # SWARM-5: retries-with-exponential-backoff (per-agent override, else manifest default).
        # Distinct from the circuit breaker (which disables a chronically-failing agent across
        # waves) — this recovers a single agent from a transient failure within its wave.
        meta_retries = int(manifest.metadata.get("max_retries", 0) or 0)

        async def _run_one(agent: AgentSpec) -> AgentExecutionResult:
            try:
                self._agent_breaker(agent.agent_id).before_call()
            except AgentBreakerOpenError:
                return AgentExecutionResult(
                    agent_id=agent.agent_id,
                    role=agent.role,
                    success=False,
                    error=f"Circuit breaker open for {agent.agent_id}",
                )

            retries = agent.max_retries or meta_retries
            attempts = 0

            async def _attempt_agent_run() -> AgentExecutionResult:
                nonlocal attempts
                attempts += 1
                proc = await scheduler.submit(
                    agent_id=agent.agent_id,
                    task=agent.task_template or manifest.query,
                )
                await scheduler.wait_for_running(proc.id)
                try:
                    res = await self._execute_agent(
                        agent, manifest, graph_deps, wave_results, proc
                    )
                    await scheduler.complete(proc.id)
                except Exception as e:
                    await scheduler.fail(proc.id, str(e))
                    res = AgentExecutionResult(
                        agent_id=agent.agent_id,
                        role=agent.role,
                        success=False,
                        error=str(e),
                    )
                if not res.success:
                    raise AgentAttemptFailedError(res)
                return res

            # SWARM-5 backoff, declaratively (CONCEPT:ORCH-1.36): the
            # historical 0.5s, 1s, 2s, ... delays bounded at 8s.
            retry_policy = ResiliencePolicy(
                max_attempts=retries + 1,
                backoff_base_s=0.5,
                backoff_factor=2.0,
                max_backoff_s=8.0,
                jitter=False,
                retry_on=(AgentAttemptFailedError,),
                name=f"wave-agent:{agent.agent_id}",
            )
            try:
                res = await run_with_resilience(_attempt_agent_run, retry_policy)
            except AgentAttemptFailedError as exc:
                res = exc.result
            if attempts > 1:
                res.metadata["retries"] = attempts - 1
            return res

        tasks = [_run_one(a) for a in agents]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[AgentExecutionResult] = []
        for raw in raw_results:
            if isinstance(raw, AgentExecutionResult):
                results.append(raw)
                # Update circuit breaker
                if raw.success:
                    self._agent_breaker(raw.agent_id).record_success()
                else:
                    self._agent_breaker(raw.agent_id).record_failure()
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
        proc: Any = None,
    ) -> AgentExecutionResult:
        """Execute a single agent invocation with full capability wiring.

        CONCEPT:ORCH-1.8 — Parallel Engine

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

        # CONCEPT:OS-5.8 Context Paging
        if proc and hasattr(proc, "checkpoint_id") and proc.checkpoint_id:
            logger.info(
                "Paging context from checkpoint %s for agent %s",
                proc.checkpoint_id,
                agent.agent_id,
            )
            try:
                from ..capabilities.checkpointing import GraphCheckpointStore

                store = GraphCheckpointStore(engine=self.engine)
                ckpt_data = store.get(proc.checkpoint_id)
                if ckpt_data:
                    task = f"{task}\n\n## RESUMED CONTEXT (Paged from KG)\n{ckpt_data}"
            except Exception as e:
                logger.warning(
                    "Failed to page context from checkpoint %s: %s",
                    proc.checkpoint_id,
                    e,
                )

        # Determine model — SWARM-6: per-agent model_role (heterogeneous swarm / Claw Groups)
        # resolves before the manifest/default fallback so e.g. a "reasoning" agent can run on a
        # frontier model while bulk agents run on a cheaper tier.
        model_id = agent.model_id
        if not model_id and agent.model_role:
            model_id = resolve_model_role(agent.model_role)
        if not model_id and graph_deps:
            model_id = str(graph_deps.agent_model)
        if not model_id:
            model_id = "openai:gpt-4o-mini"  # Fallback

        system_prompt = agent.system_prompt or (
            f"You are a {agent.role or agent.agent_id} specialist agent. "
            f"Provide your best analysis and response."
        )
        # SWARM-4: structured-output contract — instruct the sub-agent to return only valid JSON
        # matching the schema (prose from intermediates breaks downstream synthesis).
        if agent.output_schema:
            system_prompt += (
                "\n\nSTRUCTURED OUTPUT CONTRACT: Return ONLY valid JSON matching this shape "
                f"(no prose, no markdown fences):\n{agent.output_schema}"
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

            # CONCEPT:ECO-4.0 — gap-fill the workflow's declared tools against what's
            # available (substitute by capability, or surface a precise gap). Defensive:
            # falls back to agent.tools unchanged when availability is undeterminable.
            from .tool_resolver import resolve_agent_tools

            _tool_res = resolve_agent_tools(self.engine, agent.tools)
            if _tool_res.filled or _tool_res.missing:
                logger.info(
                    "[CONCEPT:ECO-4.0] tool gap-fill for '%s': filled=%s missing=%s",
                    agent.agent_id,
                    _tool_res.filled,
                    _tool_res.missing,
                )

            # Wire up all 8 capabilities natively using factory
            llm_agent, _ = create_agent(
                provider=provider,
                model_id=prov_model,
                system_prompt=system_prompt,
                name=agent.agent_id,
                enable_skills=True,
                enable_universal_tools=True,
                mcp_config=metadata.get("mcp_config"),
                tool_tags=_tool_res.resolved or agent.tools,
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
                "[CONCEPT:ORCH-1.8] Agent %s (%s) completed in %.0fms — "
                "output=%d chars",
                agent.agent_id,
                agent.role,
                duration_ms,
                len(output),
            )

            # SWARM-4: enforce the structured-output contract. A schema violation is a soft failure
            # (success=False) so retry/verify handles it rather than feeding prose into synthesis.
            schema_ok, schema_detail = enforce_structured_output(
                output, agent.output_schema
            )
            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                partition=agent.partitions[0] if agent.partitions else "",
                output=output,
                success=schema_ok,
                error="" if schema_ok else f"schema violation: {schema_detail}",
                duration_ms=duration_ms,
                model_id=model_id,
                metadata={"schema_valid": schema_ok} if agent.output_schema else {},
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[CONCEPT:ORCH-1.8] Agent %s timed out after %.0fms",
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
                "[CONCEPT:ORCH-1.8] Agent %s failed: %s",
                agent.agent_id,
                e,
            )
            # Threshold-counted escalation into the failure_gap remediation chain
            try:
                self._escalate_repeated_failure(agent.agent_id, str(e))
            except Exception as ah_err:
                logger.debug("Failure escalation skipped: %s", ah_err)

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                model_id=model_id,
            )

    _FAILURE_ESCALATION_THRESHOLD = 3

    def _escalate_repeated_failure(self, agent_id: str, error_context: str) -> None:
        """File a ``failure_gap`` topic once an agent fails repeatedly.

        Strangled replacement for the dormant ``AutoHealingEngine`` shell: its
        useful bit (threshold-counted failure registry) is kept; the dead
        skill-synthesis hooks (never wired) are gone. At the threshold the
        recurring failure enters the shared gap-topic path the golden loop
        already remediates — propose-only, no LLM here (CONCEPT:AHE-3.18).
        """
        count = self._agent_failure_counts.get(agent_id, 0) + 1
        self._agent_failure_counts[agent_id] = count
        if count < self._FAILURE_ESCALATION_THRESHOLD or self.engine is None:
            return
        self._agent_failure_counts[agent_id] = 0
        from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
            ANOMALY_ERROR,
            FailurePattern,
            _normalize_detail,
            _sig,
            file_gap_topic,
        )

        pattern = FailurePattern(
            signature=_sig(
                agent_id, "agent_execution", _normalize_detail(error_context)
            ),
            name=agent_id,
            kind="agent_execution",
            anomaly_type=ANOMALY_ERROR,
            count=count,
            sample_detail=error_context[:500],
        )
        file_gap_topic(self.engine, pattern, source="parallel_engine")

    # ── Verification (SWARM-2: planner → execute → verify loop) ──────

    async def _judge_against_criteria(
        self,
        output: str,
        criteria: str,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> tuple[bool, str]:
        """Judge one leaf output against its ``success_criteria`` (CONCEPT:ORCH-1.32 SWARM-2).

        Returns ``(passed, feedback)``. When no model is available, degrades to *pass* so
        verification never blocks execution in model-less environments. Factored out so tests can
        monkeypatch it without a live LLM.
        """
        model_id = "openai:gpt-4o-mini"
        if graph_deps and getattr(graph_deps, "agent_model", None):
            model_id = str(graph_deps.agent_model)
        try:
            judge = Agent(
                model=model_id,
                system_prompt=(
                    "You verify whether an agent output satisfies its success criteria. "
                    "Reply on two lines:\nVERDICT: PASS or FAIL\nFEEDBACK: <specific gap if FAIL>"
                ),
            )
            res = await asyncio.wait_for(
                judge.run(
                    f"Task: {query}\n\nSuccess criteria: {criteria}\n\nOutput:\n{output}"
                ),
                timeout=60.0,
            )
            text = str(res.output)
        except Exception as e:  # pragma: no cover - exercised via monkeypatch
            logger.debug("verify judge unavailable, passing: %s", e)
            return True, ""
        passed = "FAIL" not in text.upper().split("FEEDBACK")[0]
        feedback = ""
        if "FEEDBACK:" in text:
            feedback = text.split("FEEDBACK:", 1)[1].strip()
        return passed, feedback

    async def _verify_and_redispatch(
        self,
        resolved: ExecutionManifest,
        wave_results: list[WaveResult],
        graph_deps: GraphDeps | None,
    ) -> dict[str, Any]:
        """Verify leaves with ``success_criteria`` and re-dispatch failures once (bounded).

        CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm.
        SWARM-2: the planner→execute→verify loop the articles say most "throw-more-agents" setups
        skip. Gated
        by ``metadata["verify"]``; only agents that declare ``success_criteria`` are checked. Returns
        a verification summary attached to the result.
        """
        spec_by_id = {a.agent_id: a for a in self._expand_partitions(resolved)}
        checked = passed = redispatched = 0
        for wave in wave_results:
            for res in wave.results:
                spec = spec_by_id.get(res.agent_id)
                if not spec or not spec.success_criteria or not res.success:
                    continue
                checked += 1
                ok, feedback = await self._judge_against_criteria(
                    res.output, spec.success_criteria, resolved.query, graph_deps
                )
                if ok:
                    passed += 1
                    continue
                # one bounded re-dispatch with the judge's feedback appended
                redispatched += 1
                retry_spec = spec.model_copy(deep=True)
                retry_spec.task_template = (
                    f"{spec.task_template or resolved.query}\n\n"
                    f"## PRIOR ATTEMPT FAILED VERIFICATION\nFix exactly this and satisfy the "
                    f"success criteria ({spec.success_criteria}):\n{feedback}"
                )
                new_res = await self._execute_agent(
                    retry_spec, resolved, graph_deps, wave_results
                )
                # replace the leaf in place
                for i, r in enumerate(wave.results):
                    if r.agent_id == res.agent_id:
                        new_res.metadata["reverified"] = True
                        wave.results[i] = new_res
                        if new_res.success:
                            passed += 1
                        break
        return {
            "checked": checked,
            "passed": passed,
            "failed": checked - passed,
            "redispatched": redispatched,
        }

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

    def _broadcast_workspace_attention(
        self, all_results: list[AgentExecutionResult], manifest: ExecutionManifest
    ) -> list[str]:
        """Score + broadcast specialist outputs through Global Workspace Attention.

        CONCEPT:ORCH-1.2. Builds ``{agent_id: output}`` from the successful results
        and runs :meth:`WorkspaceAttention.select_and_broadcast`, persisting the
        winning proposals to the shared engine so ``get_attention_score`` can read
        each specialist's standing on later runs. Best-effort: a missing engine,
        fewer than two outputs, or any error degrades to a no-op.

        Returns the broadcast specialist ids (empty when it no-ops).
        """
        if self.engine is None:
            return []
        outputs = {
            r.agent_id: r.output
            for r in all_results
            if r.success and r.output and r.agent_id
        }
        if len(outputs) < 2:
            return []
        try:
            from .workspace_attention import WorkspaceAttention

            wa = WorkspaceAttention(self.engine)
            winners = wa.select_and_broadcast(
                outputs, manifest.query, task_id=manifest.manifest_id
            )
            if winners:
                logger.info(
                    "[CONCEPT:ORCH-1.2] GWT broadcast %d/%d specialist proposals "
                    "(top: %s=%.3f)",
                    len(winners),
                    len(outputs),
                    winners[0].specialist_id,
                    winners[0].composite_score,
                )
                self._record_winners_to_memory(winners)
            return [w.specialist_id for w in winners]
        except Exception as e:  # pragma: no cover - non-fatal telemetry path
            logger.debug("WorkspaceAttention broadcast skipped: %s", e)
            return []

    def _record_winners_to_memory(self, winners: list[Any]) -> None:
        """Record GWT winners into the evolving memory store (CONCEPT:KG-2.1).

        The winning specialists are durable signal about *what works*; routing them
        through :class:`EvolvingMemoryStore` (INSIGHT bank, deduped per specialist so
        repeat wins reinforce) gives the self-model a live, unified record alongside
        the skill/insight entries written by the evolution engine. Best-effort.
        """
        try:
            from ..harness.evolving_memory import EvolvingMemoryStore, MemoryBank

            store = EvolvingMemoryStore(engine=self.engine)
            for w in winners:
                store.add(
                    MemoryBank.INSIGHT,
                    f"Specialist '{w.specialist_id}' won the global workspace "
                    f"(composite={w.composite_score:.3f}).",
                    signature=f"gwt-winner:{w.specialist_id}",
                    importance=float(w.composite_score),
                    metadata={
                        "specialist_id": w.specialist_id,
                        "composite_score": w.composite_score,
                        "source": "workspace_attention",
                    },
                )
        except Exception as e:  # pragma: no cover - non-fatal
            logger.debug("EvolvingMemoryStore winner recording skipped: %s", e)

    def _social_swarm_health(
        self, all_results: list[AgentExecutionResult], manifest: ExecutionManifest
    ) -> dict:
        """Snapshot Multi-Agent Social System health for the wave (CONCEPT:ORCH-1.32).

        Builds a MASS from the run: archetype = each agent's role, latent state = a
        success-weighted output magnitude, and the interaction graph ``G`` from the
        manifest's ``depends_on`` DAG edges. Returns the P1–P4 swarm-health snapshot
        (heterogeneity / topology variance / co-evolution slope / W1 drift vs the
        previous run). Best-effort: <2 agents or any error → ``{}``.
        """
        if len(all_results) < 2:
            return {}
        try:
            from .social_system import MultiAgentSocialSystem

            roles = {a.agent_id: (a.role or "worker") for a in manifest.agents}
            mass = MultiAgentSocialSystem()
            for r in all_results:
                # Latent state: output magnitude, zeroed on failure.
                state = float(len(r.output)) if r.success else 0.0
                mass.add_agent(
                    r.agent_id,
                    archetype=roles.get(r.agent_id, r.role or "worker"),
                    latent_state=state,
                )
            present = {r.agent_id for r in all_results}
            for a in manifest.agents:
                for dep in getattr(a, "depends_on", []) or []:
                    if a.agent_id in present and dep in present:
                        mass.add_edge(a.agent_id, dep)
            health = mass.swarm_health(prev_states=self._prev_social_states or None)
            self._prev_social_states = [
                float(len(r.output)) if r.success else 0.0 for r in all_results
            ]
            return health
        except Exception as e:  # pragma: no cover - non-fatal telemetry
            logger.debug("Social-system health snapshot skipped: %s", e)
            return {}

    def _persist_execution(
        self,
        manifest: ExecutionManifest,
        wave_results: list[WaveResult],
        synthesis_output: str,
    ) -> str:
        """Persist execution results to the Knowledge Graph with verbose hierarchy.

        CONCEPT:ORCH-1.8 — Parallel Engine

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
                "[CONCEPT:ORCH-1.8] Persisted execution hierarchy %s to KG "
                "(%d agents, %d waves, %d topology edges)",
                execution_id,
                manifest.agent_count,
                len(wave_results),
                sum(len(a.depends_on) for a in manifest.agents),
            )

        except Exception as e:
            logger.debug("ParallelEngine: KG persistence failed: %s", e)

        return execution_id
