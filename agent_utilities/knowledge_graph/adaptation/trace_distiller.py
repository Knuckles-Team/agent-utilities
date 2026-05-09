import asyncio
import logging
from typing import Any

from ..core.engine import IntelligenceGraphEngine
from .consolidation import ConsolidationEngine

logger = logging.getLogger(__name__)


class TraceDistiller:
    """CONCEPT:KG-2.4 — Offline/Async Knowledge Compression.

    Periodically runs Cognitive Consolidation to distill execution traces
    (episodes, decisions, trajectories) into higher-order principles and preferences.
    """

    def __init__(self, engine: IntelligenceGraphEngine, interval_seconds: int = 3600):
        self.engine = engine
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task[Any] | None = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._distillation_loop())
            logger.info("Trace Distiller background task started.")

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None
            logger.info("Trace Distiller background task stopped.")

    async def _distillation_loop(self):
        while True:
            try:
                await asyncio.sleep(self.interval_seconds)
                logger.info("Running periodic trace distillation / consolidation...")
                consolidation_engine = ConsolidationEngine(engine=self.engine)
                # Ensure rules are registered
                from .consolidation import (
                    DecisionToPrincipleRule,
                    EpisodeToPreferenceRule,
                )

                consolidation_engine.register(EpisodeToPreferenceRule())
                consolidation_engine.register(DecisionToPrincipleRule())

                # Run consolidation and automatically persist (dry_run=False)
                proposals = consolidation_engine.run(dry_run=False)
                logger.info(
                    f"Trace distillation produced {len(proposals)} new proposals."
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic trace distillation failed: {e}")
