"""CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine.

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

from agent_utilities.core.config import config
from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.core.event_loop import run_blocking_ordered
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
    from ..capabilities.checkpointing import CheckpointStore
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from .state import GraphDeps

logger = logging.getLogger(__name__)


def _governed_agent_model(model: Any) -> Any:
    """Resolve an id or wrap an injected model at the mandatory context boundary."""

    if not isinstance(model, str):
        from ..core.contextual_model import wrap_model_with_context

        return wrap_model_with_context(model)
    from ..core.model_factory import create_model

    provider, concrete_model = model.split(":", 1) if ":" in model else (None, model)
    return create_model(provider=provider, model_id=concrete_model)


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

    CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

    Subclass-parameterized exactly like the multiplexer's per-child breaker
    (CONCEPT:AU-ECO.mcp.profile-differences-from-client). Profile differences from the engine-client breaker:

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


# ── Swarm helpers — CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm KG-Governed Agent Swarm


def _strip_json_fence(output: str) -> str:
    """Strip an optional ```json fenced block from a sub-agent output (pure helper)."""
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text[:4].lower() == "json" else text
        text = text.strip()
    return text


def _required_keys_from_schema(schema: str) -> list[str]:
    """Best-effort required-key extraction from a JSON-object or comma-list schema string."""
    try:
        sj = json.loads(schema)
        if isinstance(sj, dict):
            return list(sj.get("required") or sj.keys())
        return []
    except (json.JSONDecodeError, ValueError):
        return [
            k.strip()
            for k in schema.replace("{", "").replace("}", "").split(",")
            if k.strip()
        ]


def enforce_structured_output(output: str, schema: str | None) -> tuple[bool, str]:
    """SWARM-4: validate a sub-agent output against an expected JSON shape (pure, testable).

    Kimi guardrail #3 — "prose from intermediate agents creates downstream parsing failures."
    Returns ``(ok, detail)``. When ``schema`` is falsy this is a no-op pass. We validate that the
    output parses as JSON (tolerating a ```json fenced block); structural key-presence is a
    best-effort check when the schema names top-level keys.
    """
    if not schema:
        return True, "no schema"
    text = _strip_json_fence(output)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"not valid JSON: {e}"
    # best-effort: if the schema names required keys (JSON object or comma list), check presence
    required = _required_keys_from_schema(schema)
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
    """CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine.

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
        # instead of never-wired skill_evolver hooks (CONCEPT:AU-AHE.harness.failure-evolution).
        self._agent_failure_counts: dict[str, int] = {}
        # CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm — schedule metadata (critical-path, parallelism) captured per run
        self._schedule_meta: dict[str, Any] = {}
        # CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm — previous run's MASS latent-state distribution, for W1 drift
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

    @staticmethod
    def _generate_mermaid_diagram(
        resolved: ExecutionManifest, waves: list[list[AgentSpec]]
    ) -> str | None:
        """Generate the Mermaid diagram for the execution topography (non-fatal)."""
        try:
            from agent_utilities.workflows.visualizer import WorkflowVisualizer

            mermaid_code = WorkflowVisualizer.generate(resolved, waves)
            logger.info(
                "\n" + "=" * 80 + "\n"
                "[VISUALIZER] Deterministically Generated Mermaid Topography:\n\n"
                f"```mermaid\n{mermaid_code}\n```\n" + "=" * 80 + "\n"
            )
            return mermaid_code
        except Exception as vis_err:
            logger.warning("Failed to generate workflow Mermaid diagram: %s", vis_err)
            return None

    async def _run_all_waves(
        self,
        waves: list[list[AgentSpec]],
        scheduler: Any,
        resolved: ExecutionManifest,
        graph_deps: GraphDeps | None,
    ) -> list[WaveResult]:
        """Execute every wave in order, accumulating ``WaveResult``s for dependency context."""
        wave_results: list[WaveResult] = []
        for wave_idx, wave_agents in enumerate(waves):
            logger.info(
                "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Wave %d/%d — %d agents",
                wave_idx + 1,
                len(waves),
                len(wave_agents),
            )

            wave_result = await self._execute_wave(
                wave_agents, wave_idx, scheduler, resolved, graph_deps, wave_results
            )
            wave_results.append(wave_result)

            logger.info(
                "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Wave %d complete — success_rate=%.1f%%, "
                "duration=%.0fms",
                wave_idx + 1,
                wave_result.success_rate * 100,
                wave_result.duration_ms,
            )
        return wave_results

    async def _run_adversarial_verification(
        self,
        resolved: ExecutionManifest,
        graph_deps: GraphDeps | None,
        synthesis_output: str,
    ) -> None:
        """Run the final adversarial verification pass, if enabled (non-fatal).

        Mutates ``resolved.metadata["adversarial_findings"]`` when vulnerabilities are found.
        """
        from ..capabilities.adversarial_verifier import ADVERSARIAL_ENABLED

        if not ADVERSARIAL_ENABLED:
            return
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
            inherited_model = getattr(graph_deps, "agent_model", None)
            model_id = resolved.synthesis.model_id or (
                str(inherited_model) if inherited_model else ""
            )
            m_deps = cast(
                GraphDeps,
                MockGraphDeps(model_id, graph_deps.event_queue if graph_deps else None),
            )

            logger.info(
                "[CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort] Running final adversarial verification pass..."
            )
            adv_res = await run_adversarial_pass(m_state, m_deps, synthesis_output)
            if adv_res and adv_res.vulnerabilities_found:
                logger.warning(
                    "[CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort] Adversarial pass found vulnerabilities: %s",
                    adv_res.findings,
                )
                # Attach findings to resolved metadata or final execution log
                resolved.metadata["adversarial_findings"] = adv_res.findings
        except Exception as adv_err:
            logger.warning("Adversarial pass failed (non-fatal): %s", adv_err)

    def _critical_path_and_parallelism(
        self, wave_results: list[WaveResult]
    ) -> tuple[int, float]:
        """SWARM-3: critical-path length + parallelism ratio from the last schedule pass."""
        critical_path = int(
            self._schedule_meta.get("critical_path_length", len(wave_results))
        )
        parallelism = float(self._schedule_meta.get("parallelism_ratio", 1.0))
        return critical_path, parallelism

    def _build_execution_telemetry(
        self,
        resolved: ExecutionManifest,
        wave_results: list[WaveResult],
        social_health: Any,
        concurrency: int,
        critical_path: int,
        parallelism: float,
    ) -> dict[str, Any]:
        """SWARM-3 + SWARM-7: assemble the critical-path + per-wave telemetry dict."""
        telemetry: dict[str, Any] = {
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
        # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — surface GWT loop health (write/read counters +
        # suspected engine-instance mismatch) for observability.
        try:
            from .workspace_attention import workspace_attention_telemetry

            telemetry["workspace_attention"] = workspace_attention_telemetry()
        except Exception:  # pragma: no cover - telemetry is best-effort
            pass
        if social_health:
            telemetry["social_system"] = social_health
        return telemetry

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. This is the **only** entry point.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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
            "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Executing manifest '%s' — %d agents, mode=%s, "
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
        mermaid_code = self._generate_mermaid_diagram(resolved, waves)

        # 3. Select and apply coordination protocol
        # Historical protocol selection and trace persistence both perform
        # synchronous graph I/O. Keep the whole ordered coordination phase on
        # one worker so GraphOS's shared event loop remains available and the
        # trace cannot race ahead of protocol application.
        protocol = await run_blocking_ordered(self._prepare_coordination, resolved)

        # 4. Execute waves with backpressure
        concurrency = resolved.max_concurrency
        if concurrency is None:
            concurrency = getattr(config, "max_parallel_agents", 60) or 60

        from ..core.cognitive_scheduler import CognitiveScheduler

        scheduler = CognitiveScheduler(
            max_concurrent=int(concurrency), engine=self.engine
        )

        wave_results = await self._run_all_waves(waves, scheduler, resolved, graph_deps)

        # 4b. SWARM-2: verify leaves against success_criteria + bounded re-dispatch (the
        # planner→execute→verify loop). Gated by metadata["verify"]; only agents declaring
        # success_criteria are checked. Runs before synthesis so the deliverable is assembled from
        # verified outputs.
        verification: dict[str, Any] = {}
        if resolved.metadata.get("verify"):
            verification = await self._verify_and_redispatch(
                resolved, wave_results, graph_deps
            )
            logger.info(
                "[CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm] Verification pass: %s",
                verification,
            )

        # 5. Synthesize outputs (RLM-native)
        all_results = [r for w in wave_results for r in w.results]

        # 5a. CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Global Workspace Attention: score the specialists'
        # outputs, select winners, and broadcast them to the KG. The broadcast is the
        # training signal `executor.get_attention_score` reads back as each
        # specialist's runtime standing. Runs only with a shared engine and ≥2
        # successful outputs (consensus is meaningless for one); non-fatal.
        # Workspace broadcast includes native graph writes and winner-memory
        # reinforcement. Treat it as one ordered worker phase so the memory
        # record cannot precede its broadcast and neither blocks liveness.
        await run_blocking_ordered(
            self._broadcast_workspace_attention, all_results, resolved
        )

        # 5b. CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm — model the wave as a Multi-Agent Social System and
        # snapshot swarm health (archetype heterogeneity, topology variance,
        # co-evolution slope, W1 drift vs the previous run); non-fatal telemetry.
        social_health = self._social_swarm_health(all_results, resolved)

        synthesis_output = await self._synthesize(
            all_results, resolved.synthesis, resolved.query, graph_deps
        )

        # Adversarial verification on final run synthesized output
        await self._run_adversarial_verification(resolved, graph_deps, synthesis_output)

        total_duration = (time.monotonic() - start_time) * 1000

        # 6. Persist to KG
        execution_id = await run_blocking_ordered(
            self._persist_execution, resolved, wave_results, synthesis_output
        )

        critical_path, parallelism = self._critical_path_and_parallelism(wave_results)
        telemetry = self._build_execution_telemetry(
            resolved,
            wave_results,
            social_health,
            concurrency,
            critical_path,
            parallelism,
        )

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
            "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Execution complete — %d agents, %d waves, "
            "%.0fms total, success=%s",
            result.agent_count,
            len(wave_results),
            total_duration,
            result.success,
        )

        return result

    def _prepare_coordination(self, manifest: ExecutionManifest) -> Any:
        """Select, apply, then persist one coordination protocol in order."""
        protocol = self.coordination.select_protocol(
            agent_count=manifest.agent_count,
            execution_mode=manifest.execution_mode,
        )
        coordination_result = self.coordination.apply_protocol(
            protocol=protocol,
            agent_ids=[a.agent_id for a in manifest.agents],
            task=manifest.query,
            task_type=manifest.metadata.get("task_type", "general"),
        )
        self.coordination.log_coordination_trace(coordination_result)
        return protocol

    # ── Manifest Resolution ─────────────────────────────────────────

    def _resolve_manifest(self, manifest: ExecutionManifest) -> ExecutionManifest:
        """Resolve ``auto`` fields based on agent count and complexity.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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

    def _schedule_waves_sequential(
        self, manifest: ExecutionManifest
    ) -> list[list[AgentSpec]]:
        """Each agent is its own wave — critical path == agent count."""
        seq = [[a] for a in self._expand_partitions(manifest)]
        self._schedule_meta = {
            "critical_path_length": len(seq),
            "parallelism_ratio": 1.0,
        }
        return seq

    @staticmethod
    def _batch_size_for(manifest: ExecutionManifest) -> int:
        b_size = manifest.batch_size
        if b_size is None:
            b_size = getattr(config, "parallel_batch_size", 25) or 25
        return int(b_size)

    def _schedule_waves_independent(
        self, manifest: ExecutionManifest, expanded: list[AgentSpec]
    ) -> list[list[AgentSpec]]:
        """No DAG — all agents are independent; critical path == 1 (one logical level).

        Wave count may be >1 only because of batch_size, not dependency depth.
        """
        batch_size = self._batch_size_for(manifest)
        waves = []
        for i in range(0, len(expanded), batch_size):
            waves.append(expanded[i : i + batch_size])
        self._schedule_meta = {
            "critical_path_length": 1,
            "parallelism_ratio": float(len(expanded)),
        }
        return waves

    @staticmethod
    def _build_dependency_dag(
        expanded: list[AgentSpec],
    ) -> tuple[Any, dict[str, AgentSpec]]:
        """Build the dependency DAG from ``depends_on`` edges using graph primitives."""
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
        return dag, agent_map

    def _schedule_waves_dag(
        self, manifest: ExecutionManifest, expanded: list[AgentSpec]
    ) -> list[list[AgentSpec]]:
        """Topological-generation scheduling for a manifest with real dependencies."""
        dag, agent_map = self._build_dependency_dag(expanded)

        # Group by topological generation (parallel levels)
        try:
            generations = list(rx.topological_generations(dag))
        except Exception:
            logger.warning(
                "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Dependency cycle detected — falling back "
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

        batch_size = self._batch_size_for(manifest)
        topological_waves: list[list[AgentSpec]] = []
        for generation in generations:
            gen_agents = [
                agent_map[dag[nidx]] for nidx in generation if dag[nidx] in agent_map
            ]
            # Sub-batch within a generation if it exceeds batch_size
            for i in range(0, len(gen_agents), batch_size):
                topological_waves.append(gen_agents[i : i + batch_size])

        return topological_waves

    def _schedule_waves(self, manifest: ExecutionManifest) -> list[list[AgentSpec]]:
        """Build a dependency DAG and schedule agents into execution waves.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

        Uses topological sort on the dependency graph to determine
        execution order, then groups agents by topological level
        into parallel waves.

        Args:
            manifest: Resolved execution manifest.

        Returns:
            List of waves, each containing agents that can run concurrently.
        """
        if manifest.execution_mode == "sequential":
            return self._schedule_waves_sequential(manifest)

        expanded = self._expand_partitions(manifest)

        if not manifest.has_dependencies:
            return self._schedule_waves_independent(manifest, expanded)

        return self._schedule_waves_dag(manifest, expanded)

    def _expand_partitions(self, manifest: ExecutionManifest) -> list[AgentSpec]:
        """Expand fan-out partitions into individual agent specs.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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

            # SWARM-5 backoff, declaratively (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating): the
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

    @staticmethod
    def _matching_dependency_results(
        dep_id: str, wave_results: list[WaveResult]
    ) -> list[AgentExecutionResult]:
        """All successful results across waves whose id matches ``dep_id`` (exact or partitioned)."""
        matches: list[AgentExecutionResult] = []
        for wave_res in wave_results:
            for agent_res in wave_res.results:
                if not agent_res.success:
                    continue
                if agent_res.agent_id == dep_id or agent_res.agent_id.startswith(
                    f"{dep_id}:"
                ):
                    matches.append(agent_res)
        return matches

    @staticmethod
    def _format_dependency_context(agent_res: AgentExecutionResult) -> str:
        """Render one dependency-output entry for the ``## DEPENDENCY OUTPUTS`` block."""
        role_str = f"Role: {agent_res.role}" if agent_res.role else ""
        part_str = f", Partition: {agent_res.partition}" if agent_res.partition else ""
        return (
            f"### Output from dependent agent '{agent_res.agent_id}' ({role_str}{part_str}):\n"
            f"{agent_res.output}"
        )

    @classmethod
    def _dependency_context_block(
        cls, agent: AgentSpec, wave_results: list[WaveResult]
    ) -> str:
        """Render the ``## DEPENDENCY OUTPUTS`` block for ``agent`` (or ``""``).

        Extracted from ``_execute_agent`` (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).
        Ingests dependency outputs (Fan-In / Fan-Out topological context flow).
        """
        if not agent.depends_on:
            return ""
        dependency_contexts: list[str] = []
        for dep_id in agent.depends_on:
            matches = cls._matching_dependency_results(dep_id, wave_results)
            dependency_contexts.extend(
                cls._format_dependency_context(r) for r in matches
            )
        if not dependency_contexts:
            return ""
        dep_text = "\n\n".join(dependency_contexts)
        return (
            f"\n\n## DEPENDENCY OUTPUTS\n"
            f"The following dependent upstream steps have completed successfully. "
            f"Use their outputs to complete your task:\n\n"
            f"{dep_text}"
        )

    async def _paged_checkpoint_block(self, proc: Any) -> str:
        """Render the ``## RESUMED CONTEXT`` block paged from a governed checkpoint (or ``""``).

        CONCEPT:AU-OS.scaling.epistemic-dynamic-priority-quota Context Paging.
        """
        if not (proc and hasattr(proc, "checkpoint_id") and proc.checkpoint_id):
            return ""
        logger.info("Paging agent context from governed checkpoint")
        try:
            from ..capabilities.checkpointing import GraphCheckpointStore

            store = GraphCheckpointStore(engine=self.engine)
            ckpt_data = store.get(proc.checkpoint_id)
            if ckpt_data:
                return f"\n\n## RESUMED CONTEXT (Paged from KG)\n{ckpt_data}"
        except Exception as e:
            logger.warning("Failed to page agent context (%s)", type(e).__name__)
        return ""

    @staticmethod
    def _resolve_agent_model_id(agent: AgentSpec, graph_deps: GraphDeps | None) -> str:
        """Resolve the model id for ``agent`` (SWARM-6 heterogeneous swarm / Claw Groups).

        Per-agent ``model_role`` resolves before the manifest/default fallback so e.g. a
        "reasoning" agent can run on a frontier model while bulk agents run on a cheaper tier.
        """
        model_id = agent.model_id
        if not model_id and agent.model_role:
            model_id = resolve_model_role(agent.model_role)
        inherited_model = getattr(graph_deps, "agent_model", None)
        if not model_id and inherited_model:
            model_id = str(inherited_model)
        if not model_id:
            raise ValueError(
                "agent model is not configured in the manifest or model registry"
            )
        return model_id

    @staticmethod
    def _build_agent_system_prompt(agent: AgentSpec) -> str:
        """Build the system prompt for ``agent``, with the SWARM-4 structured-output contract."""
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
        return system_prompt

    def _resolve_checkpoint_store(
        self, metadata: dict[str, Any]
    ) -> CheckpointStore | None:
        """Resolve the checkpoint store named by ``metadata['checkpoint_store']`` (or ``None``)."""
        if metadata.get("checkpoint_store") == "file":
            from ..capabilities.checkpointing import FileCheckpointStore

            return FileCheckpointStore(
                directory=metadata.get("checkpoint_dir", "./checkpoints")
            )
        if metadata.get("checkpoint_store") == "graph":
            from ..capabilities.checkpointing import GraphCheckpointStore

            return GraphCheckpointStore(engine=self.engine)
        return None

    def _create_agent_for_spec(
        self,
        agent: AgentSpec,
        model_id: str,
        system_prompt: str,
        metadata: dict[str, Any],
    ) -> Any:
        """Wire up all 8 capabilities natively using the agent factory for one ``AgentSpec``."""
        from ..agent.factory import create_agent

        provider = None
        prov_model = model_id
        if ":" in model_id:
            provider, prov_model = model_id.split(":", 1)

        checkpoint_store = self._resolve_checkpoint_store(metadata)

        # CONCEPT:AU-ECO.toolkit.workflow-gap-fill — gap-fill the workflow's declared tools against what's
        # available (substitute by capability, or surface a precise gap). Defensive:
        # falls back to agent.tools unchanged when availability is undeterminable.
        from .tool_resolver import resolve_agent_tools

        _tool_res = resolve_agent_tools(self.engine, agent.tools)
        if _tool_res.filled or _tool_res.missing:
            logger.info(
                "[CONCEPT:AU-ECO.toolkit.workflow-gap-fill] tool gap-fill "
                "filled_count=%d missing_count=%d",
                len(_tool_res.filled),
                len(_tool_res.missing),
            )

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
            eviction_threshold_chars=metadata.get("eviction_threshold_chars", 80_000),
            include_checkpoints=metadata.get("include_checkpoints", False),
            checkpoint_store=checkpoint_store,
            checkpoint_frequency=metadata.get("checkpoint_frequency", "every_tool"),
            include_teams=metadata.get("include_teams", False),
        )
        return llm_agent

    def _agent_timeout_result(
        self, agent: AgentSpec, model_id: str, duration_ms: float
    ) -> AgentExecutionResult:
        """Build the ``AgentExecutionResult`` for a timed-out agent invocation."""
        logger.warning(
            "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] "
            "Agent timed out after %.0fms",
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

    def _agent_failure_result(
        self, agent: AgentSpec, model_id: str, duration_ms: float, error: Exception
    ) -> AgentExecutionResult:
        """Build the ``AgentExecutionResult`` for a failed agent invocation.

        Also threshold-counts the failure into the ``failure_gap`` remediation chain.
        """
        logger.warning(
            "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Agent failed (%s)",
            type(error).__name__,
        )
        try:
            self._escalate_repeated_failure(agent.agent_id, str(error))
        except Exception as ah_err:
            logger.debug("Failure escalation skipped (%s)", type(ah_err).__name__)

        return AgentExecutionResult(
            agent_id=agent.agent_id,
            role=agent.role,
            success=False,
            error=str(error),
            duration_ms=duration_ms,
            model_id=model_id,
        )

    async def _build_agent_task(
        self,
        agent: AgentSpec,
        manifest: ExecutionManifest,
        wave_results: list[WaveResult],
        proc: Any,
    ) -> str:
        """Assemble the full task prompt: base task + dependency + manifest + checkpoint context."""
        task = agent.task_template or manifest.query
        task += self._dependency_context_block(agent, wave_results)
        if manifest.context:
            task = f"{task}\n\nContext:\n{manifest.context}"
        task += await self._paged_checkpoint_block(proc)
        return task

    async def _execute_agent(
        self,
        agent: AgentSpec,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
        proc: Any = None,
    ) -> AgentExecutionResult:
        """Execute a single agent invocation with full capability wiring.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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

        task = await self._build_agent_task(agent, manifest, wave_results, proc)
        model_id = self._resolve_agent_model_id(agent, graph_deps)
        system_prompt = self._build_agent_system_prompt(agent)

        try:
            metadata = manifest.metadata or {}
            llm_agent = self._create_agent_for_spec(
                agent, model_id, system_prompt, metadata
            )

            result = await asyncio.wait_for(
                llm_agent.run(task),
                timeout=timeout,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            output = result.output

            logger.debug(
                "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] "
                "Agent completed in %.0fms — output=%d chars",
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
            return self._agent_timeout_result(agent, model_id, duration_ms)

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return self._agent_failure_result(agent, model_id, duration_ms, e)

    _FAILURE_ESCALATION_THRESHOLD = 3

    def _escalate_repeated_failure(self, agent_id: str, error_context: str) -> None:
        """File a ``failure_gap`` topic once an agent fails repeatedly.

        Strangled replacement for the dormant ``AutoHealingEngine`` shell: its
        useful bit (threshold-counted failure registry) is kept; the dead
        skill-synthesis hooks (never wired) are gone. At the threshold the
        recurring failure enters the shared gap-topic path the golden loop
        already remediates — propose-only, no LLM here (CONCEPT:AU-AHE.harness.failure-evolution).
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
        """Judge one leaf output against its ``success_criteria`` (CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm SWARM-2).

        Returns ``(passed, feedback)``. When no model is available, degrades to *pass* so
        verification never blocks execution in model-less environments. Factored out so tests can
        monkeypatch it without a live LLM.
        """
        model: Any = ""
        if graph_deps and getattr(graph_deps, "agent_model", None):
            model = graph_deps.agent_model
        try:
            judge = create_context_agent(
                model=_governed_agent_model(model),
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
        except Exception as e:  # pragma: no cover - exercised via monkeypatch  # noqa: BLE001 — docstring documents this exact contract ("degrades to pass so verification never blocks execution"); SWARM-2 verification is advisory, not a release gate
            logger.debug("verify judge unavailable, passing: %s", e)
            return True, ""
        passed = "FAIL" not in text.upper().split("FEEDBACK")[0]
        feedback = ""
        if "FEEDBACK:" in text:
            feedback = text.split("FEEDBACK:", 1)[1].strip()
        return passed, feedback

    async def _verify_and_redispatch_leaf(
        self,
        wave: WaveResult,
        res: AgentExecutionResult,
        spec: AgentSpec,
        resolved: ExecutionManifest,
        wave_results: list[WaveResult],
        graph_deps: GraphDeps | None,
    ) -> tuple[bool, bool]:
        """Verify one leaf against ``success_criteria``; re-dispatch once on failure.

        Returns ``(passed, was_redispatched)``.
        """
        ok, feedback = await self._judge_against_criteria(
            res.output, spec.success_criteria, resolved.query, graph_deps
        )
        if ok:
            return True, False

        # one bounded re-dispatch with the judge's feedback appended
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
        passed = False
        for i, r in enumerate(wave.results):
            if r.agent_id == res.agent_id:
                new_res.metadata["reverified"] = True
                wave.results[i] = new_res
                passed = new_res.success
                break
        return passed, True

    async def _verify_and_redispatch(
        self,
        resolved: ExecutionManifest,
        wave_results: list[WaveResult],
        graph_deps: GraphDeps | None,
    ) -> dict[str, Any]:
        """Verify leaves with ``success_criteria`` and re-dispatch failures once (bounded).

        CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm — KG-Governed Agent Swarm.
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
                leaf_passed, was_redispatched = await self._verify_and_redispatch_leaf(
                    wave, res, spec, resolved, wave_results, graph_deps
                )
                if leaf_passed:
                    passed += 1
                if was_redispatched:
                    redispatched += 1
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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis

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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis
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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis

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
            "[CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling] Hierarchical synthesis: %d results → "
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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis

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
                "[CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling] RLM synthesis failed, falling back to "
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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis

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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis
        """
        model: Any = ""
        if graph_deps and getattr(graph_deps, "agent_model", None):
            model = graph_deps.agent_model

        combined = "\n\n---\n\n".join(
            f"**{r.role or r.agent_id}**: {r.output}" for r in results
        )

        try:
            synthesizer = create_context_agent(
                model=_governed_agent_model(model),
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
                "[CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling] Group synthesis failed, using flat: %s",
                e,
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

        CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling — RLM-Native Hierarchical Synthesis
        """
        model: Any = ""
        if graph_deps and getattr(graph_deps, "agent_model", None):
            model = graph_deps.agent_model

        try:
            merger = create_context_agent(
                model=_governed_agent_model(model),
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

        CONCEPT:AU-ORCH.adapter.hot-cache-invalidation. Builds ``{agent_id: output}`` from the successful results
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
                    "[CONCEPT:AU-ORCH.adapter.hot-cache-invalidation] GWT broadcast %d/%d specialist proposals "
                    "(top: %s=%.3f)",
                    len(winners),
                    len(outputs),
                    winners[0].specialist_id,
                    winners[0].composite_score,
                )
                self._record_winners_to_memory(winners)
            return [w.specialist_id for w in winners]
        except Exception as e:  # pragma: no cover - non-fatal telemetry path  # noqa: BLE001 — explicit empty-list fallback returned right below; callers treat an empty winners list as "no GWT broadcast this round", the underlying `outputs` results are computed and available elsewhere regardless
            logger.debug("WorkspaceAttention broadcast skipped: %s", e)
            return []

    def _record_winners_to_memory(self, winners: list[Any]) -> None:
        """Record GWT winners into the evolving memory store (CONCEPT:AU-KG.memory.tiered-memory-caching).

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
        except Exception as e:  # pragma: no cover - non-fatal  # noqa: BLE001 — docstring: "Best-effort"; winners were already selected and returned to the caller before this call, so a memory-store failure only loses INSIGHT-bank reinforcement, not the round's winner list
            logger.debug("EvolvingMemoryStore winner recording skipped: %s", e)

    @staticmethod
    def _add_social_agents(
        mass: Any, all_results: list[AgentExecutionResult], manifest: ExecutionManifest
    ) -> None:
        """Register each result as a MASS agent (archetype = role, latent state = output size)."""
        roles = {a.agent_id: (a.role or "worker") for a in manifest.agents}
        for r in all_results:
            # Latent state: output magnitude, zeroed on failure.
            state = float(len(r.output)) if r.success else 0.0
            mass.add_agent(
                r.agent_id,
                archetype=roles.get(r.agent_id, r.role or "worker"),
                latent_state=state,
            )

    @staticmethod
    def _add_social_edges(
        mass: Any, manifest: ExecutionManifest, present: set[str]
    ) -> None:
        """Wire MASS interaction edges from the manifest's ``depends_on`` DAG (present agents only)."""
        for a in manifest.agents:
            for dep in getattr(a, "depends_on", []) or []:
                if a.agent_id in present and dep in present:
                    mass.add_edge(a.agent_id, dep)

    @classmethod
    def _build_social_system(
        cls, all_results: list[AgentExecutionResult], manifest: ExecutionManifest
    ) -> Any:
        """Build the ``MultiAgentSocialSystem`` (archetypes, latent states, interaction edges)."""
        from .social_system import MultiAgentSocialSystem

        mass = MultiAgentSocialSystem()
        cls._add_social_agents(mass, all_results, manifest)
        present = {r.agent_id for r in all_results}
        cls._add_social_edges(mass, manifest, present)
        return mass

    def _social_swarm_health(
        self, all_results: list[AgentExecutionResult], manifest: ExecutionManifest
    ) -> dict:
        """Snapshot Multi-Agent Social System health for the wave (CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm).

        Builds a MASS from the run: archetype = each agent's role, latent state = a
        success-weighted output magnitude, and the interaction graph ``G`` from the
        manifest's ``depends_on`` DAG edges. Returns the P1–P4 swarm-health snapshot
        (heterogeneity / topology variance / co-evolution slope / W1 drift vs the
        previous run). Best-effort: <2 agents or any error → ``{}``.
        """
        if len(all_results) < 2:
            return {}
        try:
            mass = self._build_social_system(all_results, manifest)
            health = mass.swarm_health(prev_states=self._prev_social_states or None)
            self._prev_social_states = [
                float(len(r.output)) if r.success else 0.0 for r in all_results
            ]
            return health
        except Exception as e:  # pragma: no cover - non-fatal telemetry  # noqa: BLE001 — returns {} on failure, and the caller only merges it when truthy; a snapshot failure just omits the P1-P4 block from that wave's telemetry
            logger.debug("Social-system health snapshot skipped: %s", e)
            return {}

    @staticmethod
    def _build_execution_node_data(
        manifest: ExecutionManifest,
        execution_id: str,
        all_results: list[AgentExecutionResult],
        wave_results: list[WaveResult],
        synthesis_output: str,
    ) -> dict[str, Any]:
        """Build the ``ParallelExecution`` node payload for KG persistence."""
        total_duration = sum(w.duration_ms for w in wave_results)
        success_count = sum(1 for r in all_results if r.success)
        return {
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

    @staticmethod
    def _persist_agent_result_nodes(
        engine: IntelligenceGraphEngine,
        execution_id: str,
        all_results: list[AgentExecutionResult],
    ) -> dict[str, str]:
        """Persist per-agent ``AgentExecutionResult`` nodes + ``PART_OF_EXECUTION`` edges.

        Returns the ``agent_id -> kg node id`` map used to wire dependency edges. Takes
        ``engine`` explicitly (rather than reading ``self.engine``) so the caller's
        ``self.engine is not None`` narrowing survives the method boundary.
        """
        kg_node_map: dict[str, str] = {}
        for result in all_results:
            node_uuid = f"agent_exec_res:{uuid.uuid4().hex}"
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
            engine.graph.add_node(node_uuid, **res_data)
            engine.graph.add_edge(execution_id, node_uuid, type="PART_OF_EXECUTION")
            kg_node_map[result.agent_id] = node_uuid
        return kg_node_map

    @staticmethod
    def _persist_dependency_edges(
        engine: IntelligenceGraphEngine,
        manifest: ExecutionManifest,
        kg_node_map: dict[str, str],
    ) -> None:
        """Reconstruct and persist dependency topology edges (``DEPENDS_ON``) inside the KG."""
        for agent_spec in manifest.agents:
            for dep in agent_spec.depends_on:
                source_kg = kg_node_map.get(dep)
                target_kg = kg_node_map.get(agent_spec.agent_id)
                if source_kg and target_kg:
                    engine.graph.add_edge(source_kg, target_kg, type="DEPENDS_ON")

    def _persist_execution(
        self,
        manifest: ExecutionManifest,
        wave_results: list[WaveResult],
        synthesis_output: str,
    ) -> str:
        """Persist execution results to the Knowledge Graph with verbose hierarchy.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

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
        execution_id = f"pe:{uuid.uuid4().hex}"

        engine = self.engine
        if engine is None:
            return execution_id

        try:
            all_results = [r for w in wave_results for r in w.results]
            node_data = self._build_execution_node_data(
                manifest, execution_id, all_results, wave_results, synthesis_output
            )
            engine.graph.add_node(execution_id, **node_data)

            kg_node_map = self._persist_agent_result_nodes(
                engine, execution_id, all_results
            )
            self._persist_dependency_edges(engine, manifest, kg_node_map)

            logger.info(
                "[CONCEPT:AU-ORCH.execution.parallel-engine-visualizer] Persisted execution hierarchy %s to KG "
                "(%d agents, %d waves, %d topology edges)",
                execution_id,
                manifest.agent_count,
                len(wave_results),
                sum(len(a.depends_on) for a in manifest.agents),
            )

        except Exception as e:  # noqa: BLE001 — execution_id is generated up front and already returned unpersisted when self.engine is None, an existing supported no-engine contract; a persistence exception hits that same already-covered return path
            logger.debug("ParallelEngine: KG persistence failed: %s", e)

        return execution_id
