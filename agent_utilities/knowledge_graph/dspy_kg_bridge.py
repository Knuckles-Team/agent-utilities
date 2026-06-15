from __future__ import annotations

"""DSPy to Knowledge Graph Bridge.

CONCEPT:KG-2.2 — DSPy Integration
CONCEPT:ORCH-1.8 — RLM-GEPA OptimizationTrajectory
CONCEPT:AHE-3.40 — generalized to persist *any* optimized component (prompt, tool
description, skill), not just prompts, so the unified DSPy target registry has one
durable, queryable optimization-trajectory sink.

Provides instant MERGE operations for Evolved*Node + DSPyTraceNodes to bypass
filesystem bottlenecks, while asynchronously dispatching a git sync event to preserve
causal history.
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

    async def ingest_evolved_component(
        self,
        *,
        kg_label: str,
        component_type: str,
        identifier: str,
        file_path: str,
        compiled_state: dict[str, Any] | None = None,
        version: str = "unknown",
        optimizer: str = "",
        demos: list[Any] | None = None,
        traces: list[Any] | None = None,
    ) -> None:
        """Persist any DSPy-optimized component to the KG (CONCEPT:AHE-3.40).

        Generalizes the prompt-only path: ``kg_label`` selects the node class
        (``EvolvedPromptNode`` / ``EvolvedToolDescriptionNode`` / ``EvolvedSkillNode``,
        from the target registry — never user input, so safe to interpolate), and the
        optimization demos/traces are attached as ``OptimizationTrajectoryNode``s. Then a
        background git sync preserves causal history. Best-effort; never raises.
        """
        node_id = f"{component_type}_{identifier}_{version}"
        cypher = f"""
        MERGE (c:{kg_label}:OptimizedComponentNode {{id: $node_id}})
        SET c.component_type = $component_type,
            c.identifier = $identifier,
            c.version = $version,
            c.optimizer = $optimizer,
            c.compiled_state = $compiled_state,
            c.demo_count = $demo_count,
            c.last_updated = timestamp()
        """
        params = {
            "node_id": node_id,
            "component_type": component_type,
            "identifier": identifier,
            "version": version,
            "optimizer": optimizer,
            "compiled_state": json.dumps(compiled_state or {}),
            "demo_count": len(demos or []),
        }
        try:
            if hasattr(self.kg_engine, "execute_cypher"):
                await self.kg_engine.execute_cypher(cypher, params)
                logger.info(
                    "DSPyKGBridge: merged %s for %s (%s).",
                    kg_label,
                    identifier,
                    component_type,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("DSPyKGBridge: failed to merge %s to KG: %s", kg_label, e)

        if demos or traces:
            await self._ingest_traces(kg_label, node_id, demos or traces or [])

        asyncio.create_task(self._async_git_sync(file_path))

    async def ingest_evolved_prompt(
        self, file_path: str, blueprint: dict[str, Any], traces: list[Any] | None = None
    ) -> None:
        """Ingest an evolved system-prompt blueprint (CONCEPT:KG-2.2).

        Thin convenience over :meth:`ingest_evolved_component` for the system-prompt case.
        """
        await self.ingest_evolved_component(
            kg_label="EvolvedPromptNode",
            component_type="system_prompt",
            identifier=str(blueprint.get("task", "unknown")),
            file_path=file_path,
            compiled_state=blueprint.get("dspy_compiled_state", {}),
            version=str(blueprint.get("version", "unknown")),
            traces=traces,
        )

    async def _ingest_traces(
        self, kg_label: str, node_id: str, items: list[Any]
    ) -> None:
        """Attach DSPy demos/traces to a component as OptimizationTrajectoryNodes."""
        for i, item in enumerate(items):
            cypher = f"""
            MATCH (p:{kg_label} {{id: $node_id}})
            MERGE (t:DSPyTraceNode:OptimizationTrajectoryNode {{id: $trace_id}})
            SET t.context = $context,
                t.task_input = $task_input,
                t.response = $response
            MERGE (p)-[:HAS_TRACE]->(t)
            """
            if isinstance(item, dict):
                ctx, tsk, resp = (
                    item.get("context", ""),
                    item.get("task", ""),
                    item.get("response", ""),
                )
            else:
                ctx, tsk, resp = (
                    getattr(item, "context", ""),
                    getattr(item, "task", ""),
                    getattr(item, "response", ""),
                )
            params = {
                "node_id": node_id,
                "trace_id": f"trace_{node_id}_{i}",
                "context": ctx,
                "task_input": tsk,
                "response": resp,
            }
            try:
                if hasattr(self.kg_engine, "execute_cypher"):
                    await self.kg_engine.execute_cypher(cypher, params)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to ingest trace %s: %s", i, e)

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
