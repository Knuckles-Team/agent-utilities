#!/usr/bin/python
# coding: utf-8
"""Unified Intelligence Pipeline Package."""

import logging
import time
from typing import Optional

import networkx as nx
from ...models.knowledge_graph import PipelineConfig, RegistryGraphMetadata
from ..backends.base import GraphBackend
from .types import PipelineContext
from .runner import PipelineRunner
from .phases import PHASES

logger = logging.getLogger(__name__)


class IntelligencePipeline:
    """Orchestrator for the Unified Intelligence Pipeline."""

    def __init__(self, config: PipelineConfig, backend: Optional[GraphBackend] = None):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.metadata = RegistryGraphMetadata()
        self.backend = backend

    async def run(self) -> RegistryGraphMetadata:
        """Execute the full unified intelligence pipeline."""
        start_time = time.time()
        logger.info("Starting Unified Intelligence Pipeline...")

        ctx = PipelineContext(
            config=self.config, nx_graph=self.graph, backend=self.backend
        )

        runner = PipelineRunner(PHASES)

        try:
            results = await runner.run(ctx)

            # Update metadata from results
            self.metadata.node_count = self.graph.number_of_nodes()
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


RegistryPipeline = IntelligencePipeline
