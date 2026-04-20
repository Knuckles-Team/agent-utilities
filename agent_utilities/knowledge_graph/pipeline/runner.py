import time
from typing import Dict, List
import logging

from .types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, phases: List[PipelinePhase]):
        self.phases = {p.name: p for p in phases}
        self.sorted_phases = self._topological_sort()

    def _topological_sort(self) -> List[PipelinePhase]:
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

    async def run(self, ctx: PipelineContext) -> Dict[str, PhaseResult]:
        for phase in self.sorted_phases:
            logger.info(f"Executing phase: {phase.name}")
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

        return ctx.results
