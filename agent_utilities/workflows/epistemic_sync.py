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
        self.engine = GraphComputeEngine.get_or_create()
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
    """Entrypoint for starting the sync worker safely in an asyncio event loop.

    BUG-061: this used to spawn a bare ``threading.Thread`` (the BUG-055
    shape) -- ``contextvars.ContextVar``s do not cross a
    ``threading.Thread`` boundary, so every write the daemon's recurring
    ``run_sync_cycle()`` makes (ingested entities, flushed AST mutations)
    would land with no bound actor. It has zero callers repo-wide today (the
    live ``epistemic_sync`` MCP action calls ``EpistemicSyncWorkflow.
    run_sync_cycle()`` directly on the request's own event loop instead, so
    it never takes this path), so it cannot fail yet -- but it would the
    moment anything calls it. Fixed rather than deleted: it is a plausible,
    apparently-intentional recurring-sync daemon entrypoint, not legacy code,
    and the fix is a direct application of the SAME builder
    ``knowledge_graph/ingest_worker.py`` already uses for its own long-lived
    background consumer threads -- capture the caller's already-verified
    session BEFORE spawning (``ContextVar``s don't cross the thread
    boundary), then bind it for the daemon thread's entire lifetime via
    ``_authorized_background_thread``.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _authorized_background_thread,
        _capture_verified_background_session,
    )

    session = _capture_verified_background_session()

    def loop_in_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        workflow = EpistemicSyncWorkflow()
        loop.run_until_complete(workflow.run_forever())

    t = _authorized_background_thread(
        session, loop_in_thread, name="EpistemicSyncWorker"
    )
    t.start()
    logger.info("Epistemic Sync background daemon initialized successfully.")
