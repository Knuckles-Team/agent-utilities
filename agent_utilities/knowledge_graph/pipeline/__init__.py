#!/usr/bin/python
"""Intelligence Pipeline Package."""

import logging
import time
from contextlib import nullcontext

from agent_utilities.core.config import setting
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

from ...models.knowledge_graph import PipelineConfig, RegistryGraphMetadata
from ..backends.base import GraphBackend
from ..core.session import current_session, use_session
from .runner import PipelineRunner
from .types import PipelineContext

logger = logging.getLogger(__name__)


class IntelligencePipeline:
    """Orchestrator for the Intelligence Pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        backend: GraphBackend | None = None,
        graph_name: str = "__commons__",
    ):
        self.config = config
        # An isolated tenant graph keeps a bulk-ingest subprocess's scratch
        # symbol graph off the shared "__commons__" tenant — avoids saturating the
        # single daemon when many repos ingest concurrently. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
        self.graph = GraphComputeEngine.get_or_create(graph_name=graph_name)
        self.graph_name = graph_name
        self.metadata = RegistryGraphMetadata()
        self.backend = backend

    async def run(self) -> RegistryGraphMetadata:
        """Execute the full unified intelligence pipeline."""
        import datetime

        start_time = time.time()
        run_start_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        logger.info("Starting Intelligence Pipeline...")

        ctx = PipelineContext(
            config=self.config, graph=self.graph, backend=self.backend
        )
        ctx.metadata["ingestion_timestamp"] = run_start_timestamp

        from .phases import select_phases

        _profile = setting("KG_INGEST_PROFILE")
        _phases = select_phases(_profile)
        if _profile:
            logger.info("Pipeline profile=%s (%d phases)", _profile, len(_phases))
        runner = PipelineRunner(_phases)

        # D-CDX-70: RegistryPipeline intentionally targets ITS OWN shared
        # ``self.graph_name`` graph (default "__commons__",
        # CONCEPT:AU-KG.query.vendor-agnostic-traversal) — deliberately isolated from
        # whatever graph a caller's ambient verified GraphSession happens to be
        # scoped to (e.g. a live delegation's tenant graph). Reusing that
        # mismatched ambient session against this fixed-graph client used to
        # raise "A graph-scoped view cannot retarget the verified GraphSession"
        # (graph_compute.GraphComputeEngine._send) partway through a scan and
        # abort the whole pipeline — a live production failure, not just a test
        # fixture variant (D-OTR-2 fixed the latter by rebinding fixtures).
        # Explicitly, authorizedly retarget the SAME verified actor/tenant into
        # this pipeline's own graph for the run's duration: ``with_graph`` only
        # ever changes the target graph field, never the actor/tenant/scopes,
        # so tenant isolation and the fail-closed "no session at all" guard in
        # ``resolve_session``/``_send`` are both unchanged. When there is no
        # ambient session (e.g. an unauthenticated bootstrap context) this is a
        # no-op and the existing SessionRequiredError behavior is preserved.
        _ambient_session = current_session()
        _session_cm = (
            use_session(_ambient_session.with_graph(self.graph_name))
            if _ambient_session is not None
            and _ambient_session.graph != self.graph_name
            else nullcontext()
        )

        # Temporarily pause background watcher to avoid database locks/deadlocks during active ingestion
        try:
            import agent_utilities.sdd.watcher as sdd_watcher

            sdd_watcher._WATCHER_PAUSED = True  # type: ignore
            logger.info("Paused background plan watcher during active ingestion.")
        except Exception as e:  # noqa: BLE001 — the pause is a lock-contention mitigation, not correctness-critical; a failed pause means the watcher keeps running unpaused during this ingestion (a potential perf/lock hit), it does not corrupt or skip any ingestion work
            logger.debug(f"Could not pause watcher: {e}")

        try:
            with _session_cm:
                results = await runner.run(ctx)

                # Update metadata from results
                self.metadata.node_count = len(self.graph.node_ids())
                self.metadata.edge_count = self.graph.number_of_edges()

            if "registry" in results and results["registry"].success:
                reg_out = results["registry"].output
                self.metadata.agent_count = reg_out.get("agents", 0)
                self.metadata.tool_count = reg_out.get("tools", 0)

            self.metadata.last_sync = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            duration = time.time() - start_time
            logger.info(
                f"Pipeline completed in {duration:.2f}s. Nodes: {self.metadata.node_count}, Edges: {self.metadata.edge_count}"
            )

            return self.metadata

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            try:
                import agent_utilities.sdd.watcher as sdd_watcher

                sdd_watcher._WATCHER_PAUSED = False  # type: ignore
                logger.info("Resumed background plan watcher after ingestion.")
            except Exception as e:  # noqa: BLE001 — this is a plain module-attribute assignment guarded only because the import above it could fail; the import already succeeded once in the pause block earlier in this same run, so this branch is effectively unreachable in practice, and any failure here would be a genuinely exceptional environment issue, not a normal best-effort miss
                logger.debug(f"Could not resume watcher: {e}")


RegistryPipeline = IntelligencePipeline
