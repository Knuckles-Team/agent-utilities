import asyncio
import logging
import time
from typing import Any

from ..core.engine_tasks import _retryable_partial_materialization
from .types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)

# Bounded resume for a retryable PARTIAL_MATERIALIZATION signal from the engine.
# ``_retryable_partial_materialization`` (imported above, not reimplemented —
# it is the ONE parser for this wire payload; see its docstring in
# engine_tasks.py) tells us the phase can resume from ``completeness_cursor``
# instead of aborting. Mirrors engine_tasks.py's own fixed-constant discipline
# for exactly this kind of bound (see the ``_ENRICH_BATCH``/``_ENRICH_MAX_BATCHES``
# comment there): a hardcoded module constant, never an env knob (Configuration
# discipline — an env flag is a last resort), capping how many times ONE phase
# may resume before a cursor that stops advancing terminates loudly instead of
# retrying forever.
_MATERIALIZATION_MAX_ATTEMPTS = 8
_MATERIALIZATION_RETRY_DELAY_S = 1.0

# Sentinel distinguishing "no cursor/snapshot observed yet" (first attempt)
# from a legitimate falsy/None value reported by the engine.
_UNSET = object()


class PipelineRunner:
    def __init__(self, phases: list[PipelinePhase]):
        self.phases = {p.name: p for p in phases}
        self.sorted_phases = self._topological_sort()
        # Real per-phase outcomes from the current/most recent run, kept for
        # get_status() — never fabricated (see get_status's docstring).
        self._results: dict[str, PhaseResult] = {}
        self._status: str = "idle"

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
        self._status = "running"

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

            # Bounded resume state for THIS phase's own attempts (reset per
            # phase — a later phase's materialization has nothing to do with
            # an earlier one's cursor/snapshot).
            attempt = 0
            resume_snapshot_version: Any = _UNSET
            last_cursor: Any = _UNSET

            while True:
                attempt += 1
                try:
                    output = await phase.execute(ctx, phase_deps)
                    duration = (time.time() - start_time) * 1000
                    result = PhaseResult(
                        name=phase.name,
                        duration_ms=duration,
                        output=output,
                        success=True,
                    )
                    ctx.results[phase.name] = result
                    self._results[phase.name] = result
                    logger.info(f"Completed {phase.name} in {duration:.2f}ms")
                    break
                except Exception as e:
                    # Only the exact retryable wire payload
                    # (PARTIAL_MATERIALIZATION, retryable=True) resumes; every
                    # other exception — malformed, stale, or terminal — falls
                    # straight through to the existing failure handling below,
                    # unchanged. This is the SAME strictness
                    # ``_retryable_partial_materialization`` already enforces;
                    # it is not re-implemented or broadened here.
                    materialization = _retryable_partial_materialization(e)
                    effective_error: BaseException = e
                    if materialization is not None:
                        cursor = materialization.get("completeness_cursor")
                        snapshot_version = materialization.get(
                            "source_snapshot_version"
                        )
                        if resume_snapshot_version is _UNSET:
                            resume_snapshot_version = snapshot_version
                        if snapshot_version != resume_snapshot_version:
                            # A completeness_cursor is only valid against the
                            # snapshot it was issued for; the engine moved on
                            # to a different snapshot mid-resume, so the
                            # cursor no longer means what it did.
                            effective_error = RuntimeError(
                                f"Phase {phase.name} partial-materialization "
                                "resume aborted: source_snapshot_version "
                                f"changed from {resume_snapshot_version!r} to "
                                f"{snapshot_version!r} while resuming from "
                                f"cursor={cursor!r}; a completeness_cursor is "
                                "only a valid resume point against the "
                                "snapshot it was issued for."
                            )
                        elif last_cursor is not _UNSET and cursor == last_cursor:
                            effective_error = RuntimeError(
                                f"Phase {phase.name} partial-materialization "
                                f"cursor stopped advancing at {cursor!r} "
                                f"(snapshot={snapshot_version!r}) after "
                                f"{attempt} attempt(s); aborting instead of "
                                "retrying forever."
                            )
                        elif attempt >= _MATERIALIZATION_MAX_ATTEMPTS:
                            effective_error = RuntimeError(
                                f"Phase {phase.name} did not finish "
                                "materializing within "
                                f"{_MATERIALIZATION_MAX_ATTEMPTS} attempts "
                                f"(cursor={cursor!r}, "
                                f"snapshot={snapshot_version!r})."
                            )
                        else:
                            last_cursor = cursor
                            logger.info(
                                "Phase %s hit a retryable partial "
                                "materialization (cursor=%s snapshot=%s); "
                                "resuming (attempt %d/%d)",
                                phase.name,
                                cursor,
                                snapshot_version,
                                attempt,
                                _MATERIALIZATION_MAX_ATTEMPTS,
                            )
                            await asyncio.sleep(_MATERIALIZATION_RETRY_DELAY_S)
                            continue

                    duration = (time.time() - start_time) * 1000
                    result = PhaseResult(
                        name=phase.name,
                        duration_ms=duration,
                        output=None,
                        success=False,
                        error=str(effective_error),
                    )
                    ctx.results[phase.name] = result
                    self._results[phase.name] = result
                    self._status = "failed"
                    logger.error(f"Phase {phase.name} failed: {effective_error}")
                    if effective_error is e:
                        raise e
                    raise effective_error from e

        # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Invalidate hot cache after pipeline completion
        from agent_utilities.core.config import invalidate_registry_cache

        invalidate_registry_cache()

        self._status = "complete"
        return ctx.results

    def get_status(self) -> dict[str, Any]:
        """Report the real per-phase outcome of the current/most recent run.

        The previous version of this method returned a hardcoded
        ``"complete"`` for a fixed set of phase names unconditionally —
        including in production, where ``scan`` was failing every run
        (PARTIAL_MATERIALIZATION). AGENTS.md's *Fail closed* rule bars a
        component that cannot do its job from returning a value its caller
        reads as "all clear". ``phases`` now reflects only phases that have
        actually executed: ``"complete"`` for a succeeded
        :class:`~agent_utilities.models.knowledge_graph.PhaseResult`,
        ``"failed"`` for one that raised. A phase that has not run yet is
        simply absent — never fabricated.
        """
        phases = {
            name: ("complete" if result.success else "failed")
            for name, result in self._results.items()
        }
        return {"status": self._status, "phases": phases}
