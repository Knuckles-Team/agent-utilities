#!/usr/bin/python
"""Unified Intelligence Pipeline Package."""

import logging
import time

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

from ...models.knowledge_graph import PipelineConfig, RegistryGraphMetadata
from ..backends.base import GraphBackend
from .phases import PHASES
from .runner import PipelineRunner
from .types import PipelineContext

logger = logging.getLogger(__name__)


class IntelligencePipeline:
    """Orchestrator for the Unified Intelligence Pipeline."""

    def __init__(self, config: PipelineConfig, backend: GraphBackend | None = None):
        self.config = config
        self.graph = GraphComputeEngine()
        self.metadata = RegistryGraphMetadata()
        self.backend = backend

    async def run(self) -> RegistryGraphMetadata:
        """Execute the full unified intelligence pipeline."""
        import datetime

        start_time = time.time()
        run_start_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        logger.info("Starting Unified Intelligence Pipeline...")

        ctx = PipelineContext(
            config=self.config, graph=self.graph, backend=self.backend
        )
        ctx.metadata["ingestion_timestamp"] = run_start_timestamp

        runner = PipelineRunner(PHASES)

        # Temporarily pause background watcher to avoid database locks/deadlocks during active ingestion
        try:
            import agent_utilities.sdd.watcher as sdd_watcher

            sdd_watcher._WATCHER_PAUSED = True  # type: ignore
            logger.info("Paused background plan watcher during active ingestion.")
        except Exception as e:
            logger.debug(f"Could not pause watcher: {e}")

        try:
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
            except Exception as e:
                logger.debug(f"Could not resume watcher: {e}")


RegistryPipeline = IntelligencePipeline
