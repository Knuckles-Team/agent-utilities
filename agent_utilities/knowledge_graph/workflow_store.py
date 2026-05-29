"""KG-Native Workflow Storage.

CONCEPT:ORCH-1.22 — Workflow Persistence & Replay

Provides persistent storage and retrieval of workflow definitions
within the Knowledge Graph. Successful agent executions can be saved
as reusable workflow templates for fast replay, and new workflows
can be defined from natural language and stored as KG subgraphs.

KG Schema::

    (:WorkflowDefinition {id, name, description, nl_spec, created_at,
                          last_used, use_count, avg_duration_ms, version})
      -[:HAS_STEP {order}]->
    (:WorkflowStep {node_id, role, system_prompt, tools_json, timeout,
                    refined_subtask, is_parallel, depends_on_json})
      -[:TRANSITION_TO {condition, priority}]->
    (:WorkflowStep)
      -[:REQUIRES_TOOL]->
    (:CallableResource)

    (:WorkflowDefinition)-[:DERIVED_FROM]->(:RunTrace)

Architecture::

    ┌─────────────────┐     save_workflow()     ┌───────────────────┐
    │   GraphPlan     │ ───────────────────────► │ WorkflowDefinition│
    │  (in-memory)    │                          │   (in KG)         │
    └─────────────────┘                          └───────────────────┘
           ▲                                              │
           │     load_workflow()                          │
           └──────────────────────────────────────────────┘

    ┌──────────────────┐  save_from_execution()  ┌──────────────────┐
    │  RunTrace +      │ ───────────────────────► │ WorkflowDefinition│
    │  GraphResponse   │                          │  (auto-cached)   │
    └──────────────────┘                          └──────────────────┘

Usage::

    from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

    store = WorkflowStore(engine)

    # Save a plan
    workflow_id = store.save_workflow("research_pipeline", plan, metadata={...})

    # Load and replay
    plan = store.load_workflow("research_pipeline")

    # Auto-save from successful execution
    store.save_from_execution(run_id, plan, result)

    # List available workflows
    workflows = store.list_workflows()

    # Find by semantic similarity
    matches = store.find_similar("analyze code quality")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan, GraphResponse

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class WorkflowStore:
    """KG-native workflow persistence and retrieval.

    CONCEPT:ORCH-1.22 — Workflow Persistence & Replay

    Stores workflow definitions as KG subgraphs and materializes
    them back into ``GraphPlan`` objects for execution. Supports
    semantic search for workflow discovery and automatic caching
    of successful executions.

    Attributes:
        engine: The IntelligenceGraphEngine for KG operations.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def save_workflow(
        self,
        name: str,
        plan: GraphPlan,
        description: str = "",
        nl_spec: str = "",
        metadata: dict[str, Any] | None = None,
        derived_from_run_id: str | None = None,
    ) -> str:
        """Persist a GraphPlan as a reusable workflow in the KG.

        CONCEPT:ORCH-1.22 — Workflow Save

        Creates a ``WorkflowDefinition`` node with connected
        ``WorkflowStep`` nodes and ``TRANSITION_TO`` edges.

        Args:
            name: Human-readable workflow name (used as lookup key).
            plan: The GraphPlan to persist.
            description: Optional description of what the workflow does.
            nl_spec: The natural language specification that generated this workflow.
            metadata: Optional metadata dict to attach.
            derived_from_run_id: Optional RunTrace ID this workflow was derived from.

        Returns:
            The workflow definition node ID.
        """
        workflow_id = (
            f"workflow:{name.lower().replace(' ', '_')}:{uuid.uuid4().hex[:8]}"
        )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Create the workflow definition node
        props: dict[str, Any] = {
            "name": name,
            "description": description,
            "nl_spec": nl_spec,
            "created_at": ts,
            "last_used": ts,
            "use_count": 0,
            "avg_duration_ms": 0.0,
            "version": 1,
            "step_count": len(plan.steps),
            "mermaid": plan.to_mermaid(title=name),
        }
        if metadata:
            props["metadata_json"] = json.dumps(metadata, default=str)
        if plan.metadata:
            props["plan_metadata_json"] = json.dumps(plan.metadata, default=str)

        self.engine.add_node(workflow_id, "WorkflowDefinition", properties=props)
        logger.info(
            "[ORCH-1.22] Saved workflow definition: id=%s, name=%s, steps=%d",
            workflow_id,
            name,
            len(plan.steps),
        )

        # Create step nodes and edges using engine API (LadybugDB-compatible)
        prev_step_id = None
        for i, step in enumerate(plan.steps):
            step_id = f"{workflow_id}:step:{i}"
            step_props: dict[str, Any] = {
                "node_id": step.node_id,
                "step_order": i,
                "is_parallel": step.is_parallel,
                "timeout": step.timeout,
                "status": "pending",
            }
            if step.refined_subtask:
                step_props["refined_subtask"] = step.refined_subtask
            if step.input_data:
                step_props["input_data_json"] = json.dumps(step.input_data, default=str)
            if step.depends_on:
                step_props["depends_on_json"] = json.dumps(step.depends_on)
            if step.access_list:
                step_props["access_list_json"] = json.dumps(step.access_list)

            self.engine.add_node(step_id, "WorkflowStep", properties=step_props)

            # Link workflow → step (via engine.add_edge for backend compatibility)
            self.engine.link_nodes(
                workflow_id,
                step_id,
                "HAS_STEP",
                properties={"step_order": i},
            )

            # Link step → previous step (sequential dependency)
            if prev_step_id and not step.is_parallel:
                self.engine.link_nodes(
                    prev_step_id,
                    step_id,
                    "TRANSITION_TO",
                    properties={"condition": "on_success", "priority": 1},
                )

            # Link step → required tools (best-effort)
            if self.engine.backend:
                try:
                    tool_rows = self.engine.backend.execute(
                        "MATCH (r:CallableResource) WHERE r.name = $node_id "
                        "RETURN r.id AS rid",
                        {"node_id": step.node_id},
                    )
                    for row in tool_rows:
                        rid = row.get("rid")
                        if rid:
                            self.engine.link_nodes(step_id, rid, "REQUIRES_TOOL")
                except Exception:
                    pass  # nosec — tool linking is best-effort

            prev_step_id = step_id

        # Link to source RunTrace if available (best-effort)
        if derived_from_run_id:
            trace_id = f"trace:{derived_from_run_id}"
            try:
                self.engine.link_nodes(workflow_id, trace_id, "DERIVED_FROM")
            except Exception:
                pass  # nosec — provenance linking is best-effort

        return workflow_id

    def load_workflow(self, name: str) -> GraphPlan | None:
        """Materialize a stored workflow back into a GraphPlan.

        CONCEPT:ORCH-1.22 — Workflow Load

        Queries the KG for a ``WorkflowDefinition`` by name and
        reconstructs the ``GraphPlan`` with all steps and dependencies.

        Args:
            name: The workflow name to look up.

        Returns:
            Reconstructed GraphPlan or None if not found.
        """
        if self.engine.backend:
            return self._load_workflow_backend(name)
        return self._load_workflow_nx(name)

    def _load_workflow_nx(self, name: str) -> GraphPlan | None:
        """Load workflow from graph compute engine (memory-only mode).

        CONCEPT:ORCH-1.22 — NX Fallback
        """
        graph = self.engine.graph

        # Find the WorkflowDefinition node by name
        wid = None
        for nid, data in graph.nodes(data=True):
            if data.get("type") == "WorkflowDefinition" and data.get("name") == name:
                wid = nid
                break

        if wid is None:
            logger.info("[ORCH-1.22] Workflow '%s' not found in NX graph", name)
            return None

        # Find connected WorkflowStep nodes via HAS_STEP edges
        step_nodes: list[tuple[int, str, dict[str, Any]]] = []
        for _, target, edge_data in graph.out_edges(wid, data=True):
            if edge_data.get("type") == "HAS_STEP":
                target_data = graph.nodes[target]
                step_order = edge_data.get("step_order", 0)
                step_nodes.append((step_order, target, target_data))

        step_nodes.sort(key=lambda x: x[0])

        steps = []
        for _, _, data in step_nodes:
            depends_on = []
            if data.get("depends_on_json"):
                try:
                    depends_on = json.loads(data["depends_on_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            access_list = []
            if data.get("access_list_json"):
                try:
                    access_list = json.loads(data["access_list_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            input_data = None
            if data.get("input_data_json"):
                try:
                    input_data = json.loads(data["input_data_json"])
                except (json.JSONDecodeError, TypeError):
                    input_data = data["input_data_json"]

            step = ExecutionStep(
                node_id=data.get("node_id", "unknown"),
                refined_subtask=data.get("refined_subtask"),
                input_data=input_data,
                is_parallel=bool(data.get("is_parallel", False)),
                timeout=float(data.get("timeout", 120.0)),
                depends_on=depends_on,
                access_list=access_list,
            )
            steps.append(step)

        wf_data = graph.nodes[wid]
        plan_metadata = {}
        if wf_data.get("plan_metadata_json"):
            try:
                plan_metadata = json.loads(wf_data["plan_metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        plan = GraphPlan(steps=steps, metadata=plan_metadata)
        logger.info(
            "[ORCH-1.22] Loaded workflow '%s' from NX: %d steps", name, len(steps)
        )
        return plan

    def _load_workflow_backend(self, name: str) -> GraphPlan | None:
        """Load workflow from persistent backend via Cypher.

        CONCEPT:ORCH-1.22 — Backend Load
        """

        # Find the workflow definition
        assert self.engine.backend is not None  # guarded by caller
        rows = self.engine.backend.execute(
            "MATCH (w:WorkflowDefinition) WHERE w.name = $name "
            "RETURN w.id AS wid, w.metadata_json AS meta, w.plan_metadata_json AS plan_meta "
            "ORDER BY w.version DESC LIMIT 1",
            {"name": name},
        )
        if not rows:
            logger.info("[ORCH-1.22] Workflow '%s' not found in KG", name)
            return None

        wid = rows[0]["wid"]

        # Fetch steps ordered by position
        step_rows = self.engine.backend.execute(
            "MATCH (w:WorkflowDefinition {id: $wid})-[r:HAS_STEP]->(s:WorkflowStep) "
            "RETURN s.node_id AS node_id, s.refined_subtask AS refined_subtask, "
            "s.input_data_json AS input_data, s.is_parallel AS is_parallel, "
            "s.timeout AS timeout, s.depends_on_json AS depends_on, "
            "s.access_list_json AS access_list, r.step_order AS step_order "
            "ORDER BY r.step_order",
            {"wid": wid},
        )

        steps = []
        for row in step_rows:
            input_data = None
            if row.get("input_data"):
                try:
                    input_data = json.loads(row["input_data"])
                except (json.JSONDecodeError, TypeError):
                    input_data = row["input_data"]

            depends_on = []
            if row.get("depends_on"):
                try:
                    depends_on = json.loads(row["depends_on"])
                except (json.JSONDecodeError, TypeError):
                    pass

            access_list = []
            if row.get("access_list"):
                try:
                    access_list = json.loads(row["access_list"])
                except (json.JSONDecodeError, TypeError):
                    pass

            step = ExecutionStep(
                node_id=row.get("node_id", "unknown"),
                refined_subtask=row.get("refined_subtask"),
                input_data=input_data,
                is_parallel=bool(row.get("is_parallel", False)),
                timeout=float(row.get("timeout", 120.0)),
                depends_on=depends_on,
                access_list=access_list,
            )
            steps.append(step)

        plan_metadata = {}
        if rows[0].get("plan_meta"):
            try:
                plan_metadata = json.loads(rows[0]["plan_meta"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Update usage statistics
        self._update_usage(wid)

        plan = GraphPlan(steps=steps, metadata=plan_metadata)
        logger.info("[ORCH-1.22] Loaded workflow '%s': %d steps", name, len(steps))
        return plan

    def list_workflows(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all stored workflow definitions.

        CONCEPT:ORCH-1.22 — Workflow Discovery

        Returns:
            List of workflow summary dicts with keys: id, name,
            description, step_count, use_count, last_used.
        """
        if not self.engine.backend:
            return []

        rows = self.engine.backend.execute(
            "MATCH (w:WorkflowDefinition) "
            "RETURN w.id AS id, w.name AS name, w.description AS description, "
            "w.step_count AS step_count, w.use_count AS use_count, "
            "w.last_used AS last_used, w.avg_duration_ms AS avg_duration_ms, "
            "w.nl_spec AS nl_spec, w.version AS version "
            "ORDER BY w.use_count DESC LIMIT $limit",
            {"limit": limit},
        )
        return [dict(row) for row in rows]

    def save_from_execution(
        self,
        run_id: str,
        plan: GraphPlan,
        result: GraphResponse | dict[str, Any],
        task: str = "",
        agent_name: str = "",
        duration_ms: float = 0.0,
    ) -> str | None:
        """Auto-save a successful execution as a reusable workflow template.

        CONCEPT:ORCH-1.22 — Execution-Driven Workflow Learning

        Only saves workflows from successful executions with 2+ steps
        (single-step executions are too trivial to cache).

        Args:
            run_id: The execution run ID.
            plan: The executed GraphPlan.
            result: The execution result (GraphResponse or dict).
            task: The original task description.
            agent_name: The agent that executed the workflow.
            duration_ms: Execution duration in milliseconds.

        Returns:
            Workflow ID if saved, None if skipped.
        """
        # Only cache non-trivial successful executions
        status = (
            result.status
            if isinstance(result, GraphResponse)
            else result.get("status", "")
        )
        if status not in ("completed", "success"):
            logger.debug(
                "[ORCH-1.22] Skipping workflow save — execution not successful: %s",
                status,
            )
            return None

        if len(plan.steps) < 2:
            logger.debug(
                "[ORCH-1.22] Skipping workflow save — too few steps: %d",
                len(plan.steps),
            )
            return None

        # Generate a name from the task
        name = (
            f"{agent_name}:{task[:50].replace(' ', '_').lower()}"
            if task
            else f"auto:{run_id[:8]}"
        )

        return self.save_workflow(
            name=name,
            plan=plan,
            description=f"Auto-captured from execution {run_id}",
            nl_spec=task,
            metadata={
                "agent_name": agent_name,
                "duration_ms": duration_ms,
                "run_id": run_id,
                "auto_captured": True,
            },
            derived_from_run_id=run_id,
        )

    def find_similar(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find workflows similar to a natural language query.

        CONCEPT:ORCH-1.22 — Semantic Workflow Matching

        Uses the KG's semantic search to find workflows whose
        descriptions or NL specs match the query.

        Args:
            query: Natural language description of desired workflow.
            top_k: Maximum results to return.

        Returns:
            List of matching workflow summaries with similarity scores.
        """
        if not self.engine:
            return []

        try:
            results = self.engine.search_hybrid(query, top_k=top_k)
            workflows = []
            for r in results:
                if r.get("type") == "WorkflowDefinition" or "workflow:" in r.get(
                    "id", ""
                ):
                    workflows.append(r)
            return workflows
        except Exception as e:
            logger.debug("[ORCH-1.22] Semantic workflow search failed: %s", e)
            return []

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and its steps from the KG.

        Args:
            workflow_id: The workflow definition node ID.

        Returns:
            True if deleted, False if not found.
        """
        if not self.engine.backend:
            return False

        try:
            # Delete steps first
            self.engine.backend.execute(
                "MATCH (w:WorkflowDefinition {id: $wid})-[:HAS_STEP]->(s:WorkflowStep) "
                "DETACH DELETE s",
                {"wid": workflow_id},
            )
            # Delete the workflow definition
            self.engine.backend.execute(
                "MATCH (w:WorkflowDefinition {id: $wid}) DETACH DELETE w",
                {"wid": workflow_id},
            )
            logger.info("[ORCH-1.22] Deleted workflow: %s", workflow_id)
            return True
        except Exception as e:
            logger.error("[ORCH-1.22] Failed to delete workflow %s: %s", workflow_id, e)
            return False

    def get_mermaid(self, name: str) -> str | None:
        """Get the mermaid diagram for a stored workflow.

        Args:
            name: Workflow name.

        Returns:
            Mermaid diagram string or None.
        """
        if not self.engine.backend:
            return None

        rows = self.engine.backend.execute(
            "MATCH (w:WorkflowDefinition) WHERE w.name = $name "
            "RETURN w.mermaid AS mermaid "
            "ORDER BY w.version DESC LIMIT 1",
            {"name": name},
        )
        if rows:
            return rows[0].get("mermaid")
        return None

    def _update_usage(self, workflow_id: str) -> None:
        """Increment usage counter and update last_used timestamp."""
        if not self.engine.backend:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            self.engine.backend.execute(
                "MATCH (w:WorkflowDefinition {id: $wid}) "
                "SET w.use_count = COALESCE(w.use_count, 0) + 1, w.last_used = $ts",
                {"wid": workflow_id, "ts": ts},
            )
        except Exception:
            pass  # nosec — usage tracking is best-effort
