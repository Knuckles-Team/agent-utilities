import logging
import time
from typing import Any

from .types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, phases: list[PipelinePhase]):
        self.phases = {p.name: p for p in phases}
        self.sorted_phases = self._topological_sort()

    def _topological_sort(self) -> list[PipelinePhase]:
        """Kahn's algorithm for topological sorting."""
        in_degree = {name: 0 for name in self.phases}
        for phase in self.phases.values():
            for dep in phase.deps:
                if dep not in self.phases:
                    raise ValueError(
                        f"Phase {phase.name} depends on unknown phase {dep}"
                    )
                in_degree[phase.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_names = []

        while queue:
            node = queue.pop(0)
            sorted_names.append(node)
            for phase in self.phases.values():
                if node in phase.deps:
                    in_degree[phase.name] -= 1
                    if in_degree[phase.name] == 0:
                        queue.append(phase.name)

        if len(sorted_names) != len(self.phases):
            raise ValueError("Cycle detected in pipeline dependencies")

        return [self.phases[name] for name in sorted_names]

    async def run(self, ctx: PipelineContext) -> dict[str, PhaseResult]:
        STAGE_MAPPING = {
            "Stage 1: Context Hydration": [
                "memory",
                "scan",
                "workspace_sync",
                "registry",
            ],
            "Stage 2: Structural Extraction": ["parse", "resolve", "mro", "reference"],
            "Stage 3: Topological & Semantic Enrichment": [
                "communities",
                "centrality",
                "embedding",
            ],
            "Stage 4: Epistemic Consolidation": [
                "shacl_gate",
                "sync",
                "owl_reasoning",
                "external_graphs",
                "knowledge_base",
            ],
            "Stage 5: Governance & Evolution": [
                "validate",
                "experience_distillation",
                "decision_evolution",
            ],
        }

        phase_to_stage = {}
        for stage, p_list in STAGE_MAPPING.items():
            for p in p_list:
                phase_to_stage[p] = stage

        current_stage = None

        for phase in self.sorted_phases:
            stage = phase_to_stage.get(phase.name, "Stage X: Unknown")
            if stage != current_stage:
                logger.info(f"=== Entering {stage} ===")
                current_stage = stage

            print(f"!!! EXECUTING PHASE: {phase.name}", flush=True)
            start_time = time.time()

            # Filter deps for this phase
            phase_deps = {
                dep: ctx.results[dep] for dep in phase.deps if dep in ctx.results
            }

            try:
                output = await phase.execute(ctx, phase_deps)
                duration = (time.time() - start_time) * 1000
                result = PhaseResult(
                    name=phase.name, duration_ms=duration, output=output, success=True
                )
                ctx.results[phase.name] = result
                logger.info(f"Completed {phase.name} in {duration:.2f}ms")
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                result = PhaseResult(
                    name=phase.name,
                    duration_ms=duration,
                    output=None,
                    success=False,
                    error=str(e),
                )
                ctx.results[phase.name] = result
                logger.error(f"Phase {phase.name} failed: {e}")
                raise e

        # CONCEPT:ORCH-1.2 — Invalidate hot cache after pipeline completion
        from agent_utilities.core.config import invalidate_registry_cache

        invalidate_registry_cache()

        return ctx.results

    def get_status(self) -> dict[str, Any]:
        """Return dummy status for compatibility."""
        return {
            "status": "idle",
            "phases": {
                "memory": "complete",
                "scan": "complete",
                "parse": "complete",
                "registry": "complete",
            },
        }
