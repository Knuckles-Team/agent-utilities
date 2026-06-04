import asyncio
import logging

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.integrations.sparql_ingestor import (
    FederatedSparqlIngestor,
)

logger = logging.getLogger(__name__)


class EpistemicSyncWorkflow:
    """
    Background worker that runs periodically to sync external SPARQL
    endpoints with the local Epistemic Graph.
    Maintains the local database as the Operational Source of Truth while
    federating external definitions.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.engine = GraphComputeEngine()
        self.ingestor = FederatedSparqlIngestor(
            endpoints=self.config.sparql_endpoints, engine=self.engine
        )
        from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
            LadybugBackend,
        )

        self.backend = LadybugBackend()

    async def run_sync_cycle(self) -> None:
        """Executes a single synchronization cycle against external authoritative graphs."""
        logger.info(
            f"Starting Epistemic Sync cycle across {len(self.config.sparql_endpoints)} SPARQL endpoints..."
        )

        try:
            # 1. Pull authoritative changes
            ingested_count = self.ingestor.ingest_entities(limit=100)
            logger.info(
                f"Ingested {ingested_count} external entities into local schema mapping."
            )

            # Flush local AST mutations to LadybugDB
            flushed_count = self.engine.flush_ledger_to_backend(self.backend)
            if flushed_count > 0:
                logger.info(
                    f"Flushed {flushed_count} AST mutations from epistemic-graph to LadybugDB."
                )

            # 2. In future iterations, temporal drift and importance_score
            # will be evaluated here to flag 'knowledge_gap' nodes if
            # external operational data conflicts with local truths.

        except Exception as e:
            logger.error(f"Epistemic Sync cycle failed: {e}", exc_info=True)

    async def run_forever(self, interval_seconds: int = 3600) -> None:
        """Daemon loop to trigger the sync intermittently."""
        while True:
            await self.run_sync_cycle()
            await asyncio.sleep(interval_seconds)


def start_epistemic_sync_daemon() -> None:
    """Entrypoint for starting the sync worker safely in an asyncio event loop."""
    import threading

    def loop_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        workflow = EpistemicSyncWorkflow()
        loop.run_until_complete(workflow.run_forever())

    t = threading.Thread(target=loop_in_thread, daemon=True, name="EpistemicSyncWorker")
    t.start()
    logger.info("Epistemic Sync background daemon initialized successfully.")
