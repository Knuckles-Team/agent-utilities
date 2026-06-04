from __future__ import annotations

"""DSPy to Knowledge Graph Bridge.

CONCEPT:KG-2.2 — DSPy Integration
CONCEPT:ORCH-1.8 — RLM-GEPA OptimizationTrajectory

Provides instant MERGE operations for EvolvedPromptNodes and DSPyTraceNodes
to bypass filesystem bottlenecks, while asynchronously dispatching a git sync
event to preserve causal history.
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DSPyKGBridge:
    """Bridge for persisting DSPy compiled states directly to the Knowledge Graph.

    Aligns with OWL/epistemic-graph reasoning by emitting nodes that can be
    semantically reasoned over (e.g., TimeoutFailure, ToolHallucination).
    """

    def __init__(self, kg_engine: Any, workspace_path: str) -> None:
        self.kg_engine = kg_engine
        self.workspace_path = workspace_path

    async def ingest_evolved_prompt(
        self, file_path: str, blueprint: dict[str, Any], traces: list[Any] | None = None
    ) -> None:
        """Instantly ingest an evolved prompt blueprint into the KG."""
        version = blueprint.get("version", "unknown")
        task_name = blueprint.get("task", "unknown")
        compiled_state = blueprint.get("dspy_compiled_state", {})

        cypher = """
        MERGE (p:EvolvedPromptNode {id: $prompt_id})
        SET p.task = $task_name,
            p.version = $version,
            p.compiled_state = $compiled_state,
            p.last_updated = timestamp()
        """

        params = {
            "prompt_id": f"prompt_{task_name}_{version}",
            "task_name": task_name,
            "version": version,
            "compiled_state": json.dumps(compiled_state),
        }

        try:
            if hasattr(self.kg_engine, "execute_cypher"):
                # Fast path Cypher execution
                await self.kg_engine.execute_cypher(cypher, params)
                logger.info(
                    f"DSPyKGBridge: Instantly merged EvolvedPromptNode for {task_name}."
                )
        except Exception as e:
            logger.warning(f"DSPyKGBridge: Failed to merge prompt to KG: {e}")

        if traces:
            await self._ingest_traces(task_name, version, traces)

        # Dispatch background file sync so the fast-path is not blocked
        asyncio.create_task(self._async_git_sync(file_path))

    async def _ingest_traces(
        self, task_name: str, version: str, traces: list[Any]
    ) -> None:
        """Ingest DSPy traces as OptimizationTrajectoryNodes."""
        for i, trace in enumerate(traces):
            cypher = """
            MATCH (p:EvolvedPromptNode {id: $prompt_id})
            MERGE (t:DSPyTraceNode:OptimizationTrajectoryNode {id: $trace_id})
            SET t.context = $context,
                t.task_input = $task_input,
                t.response = $response
            MERGE (p)-[:HAS_TRACE]->(t)
            """
            params = {
                "prompt_id": f"prompt_{task_name}_{version}",
                "trace_id": f"trace_{task_name}_{version}_{i}",
                "context": getattr(trace, "context", ""),
                "task_input": getattr(trace, "task", ""),
                "response": getattr(trace, "response", ""),
            }
            try:
                if hasattr(self.kg_engine, "execute_cypher"):
                    await self.kg_engine.execute_cypher(cypher, params)
            except Exception as e:
                logger.debug(f"Failed to ingest trace {i}: {e}")

    async def _async_git_sync(self, file_path: str) -> None:
        """Silently commit the file changes to git to maintain causal history."""
        import subprocess

        try:
            subprocess.run(
                ["git", "add", file_path],
                cwd=self.workspace_path,
                capture_output=True,
                check=True,
            )
            msg = f"ahe(system_prompt): Background DSPy sync for {file_path}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.workspace_path,
                capture_output=True,
                check=True,
            )
            logger.info(f"DSPyKGBridge: Background git sync completed for {file_path}.")
        except Exception as e:
            logger.debug(f"DSPyKGBridge: Background git sync failed: {e}")
