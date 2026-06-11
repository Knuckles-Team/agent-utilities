"""BusinessProcess → Executable Workflow Compiler.

CONCEPT:ORCH-1.41 — Ontology→Workflow Bridge

Compiles a *descriptive* process — a harvested ``BusinessProcess`` node with
its step-level ``BusinessTask``/``FLOWS_TO`` subgraph (lifted from Camunda
BPMN XML by the KG-2.53 extractor, or federated from Egeria/ArchiMate) — into
an *executable* ``GraphPlan`` and persists it as a ``WorkflowDefinition``,
closing the gap between the two process worlds.

Pipeline::

    ┌────────────────────┐  load subgraph   ┌───────────────────────┐
    │  BusinessProcess    │ ───────────────► │ BusinessTask nodes +   │
    │  (descriptive, KG)  │                  │ FLOWS_TO sequence DAG  │
    └────────────────────┘                  └──────────┬────────────┘
                                                       │ collapse gateways,
                                                       │ reject cycles
                                            ┌──────────▼────────────┐
                                            │ depends_on derivation  │
                                            │ (branches → parallel   │
                                            │  steps, shared deps)   │
                                            └──────────┬────────────┘
                                                       │ semantic agent/tool
                                                       │ matching (reuses the
                                                       │ ORCH-1.23 compiler)
                                            ┌──────────▼────────────┐
                                            │ GraphPlan (executable) │
                                            └──────────┬────────────┘
                                                       │ store.save_workflow()
                                            ┌──────────▼────────────┐
                                            │ WorkflowDefinition     │
                                            │  -[:REALIZES]->        │
                                            │ BusinessProcess        │
                                            └───────────────────────┘

Semantics:
    - **Gateways** (BusinessTask nodes with ``is_gateway``/gateway
      ``task_type``) are structural, not executable: they are collapsed so a
      gateway's branches become parallel steps that share the gateway's
      predecessors as dependencies, and downstream joins depend on every
      branch that reaches them.
    - **Cycles** among executable tasks (BPMN loop-backs) cannot be expressed
      in a ``GraphPlan`` DAG → ``ProcessCompilationError`` naming the tasks
      on the cycle.
    - **Unresolvable tasks** (no KG-registered agent/tool matches) stay in the
      plan as explicit manual steps (``manual:`` id prefix, step metadata
      ``unresolved=True``) and are listed in ``plan.metadata`` — or, with
      ``require_resolved=True``, fail compilation with the unmatched list.
    - The ``(:WorkflowDefinition)-[:REALIZES]->(:BusinessProcess)`` edge
      (``:realizesProcess`` in the ontology) records the bridge for lineage
      close-out (ORCH-1.43) and provenance queries.

Usage::

    from agent_utilities.knowledge_graph.process_plan_compiler import (
        ProcessPlanCompiler,
    )

    compiler = ProcessPlanCompiler(engine)
    plan = await compiler.compile("bpmn_process:invoice:1:abc")
    report = await compiler.compile_and_store("bpmn_process:invoice:1:abc")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Sentinel the ORCH-1.23 matcher returns when no KG agent/tool matched.
_UNMATCHED_SENTINEL = "executor"


class ProcessCompilationError(ValueError):
    """A BusinessProcess subgraph cannot be compiled into an executable plan."""


def _slug(text: str) -> str:
    """Stable, id-safe slug for step ids and workflow names."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "step"


def _is_gateway(props: dict[str, Any]) -> bool:
    """A BusinessTask that is BPMN routing structure, not executable work."""
    if props.get("is_gateway"):
        return True
    return "gateway" in str(props.get("task_type", "")).lower()


class ProcessPlanCompiler:
    """Compile a descriptive BusinessProcess subgraph into a GraphPlan.

    CONCEPT:ORCH-1.41 — Ontology→Workflow Bridge

    Sibling of :class:`~agent_utilities.knowledge_graph.workflow_compiler.
    WorkflowCompiler` (natural language → plan); this compiler starts from KG
    process *structure* instead of free text, and reuses the NL compiler's
    semantic agent/tool matching for step resolution.

    Attributes:
        engine: The IntelligenceGraphEngine for KG traversal + persistence.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        # Reuse the ORCH-1.23 KG semantic matcher (one matching brain, two
        # compilers) and its lazily-built WorkflowStore.
        self._nl_compiler = WorkflowCompiler(engine)

    @property
    def store(self) -> Any:
        """The shared WorkflowStore (lazily created by the NL compiler)."""
        return self._nl_compiler.store

    # ------------------------------------------------------------------
    # Subgraph loading
    # ------------------------------------------------------------------

    def _node_props(self, node_id: str) -> dict[str, Any]:
        """Best-effort property fetch for a node (compute graph, then backend)."""
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                data = graph.nodes[node_id]
                if data:
                    return dict(data)
            except Exception:  # noqa: BLE001 — fall through to the backend
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (p) WHERE p.id = $pid RETURN p", {"pid": node_id}
                )
                if rows and isinstance(rows[0].get("p"), dict):
                    return dict(rows[0]["p"])
            except Exception:  # noqa: BLE001 — absent node handled by caller
                pass
        return {}

    def _edge_rel(self, edge_data: dict[str, Any]) -> str:
        return str(edge_data.get("type") or edge_data.get("rel_type") or "").upper()

    def _load_subgraph(
        self, process_id: str
    ) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, Any]]]:
        """Load the process's BusinessTask nodes and FLOWS_TO sequence edges.

        Reads the L1 compute graph first (always warm); falls back to backend
        Cypher when the compute mirror has no structure for this process.

        Returns:
            ``(tasks, flows)`` — ``tasks`` maps task node id → properties;
            ``flows`` is ``(source_id, target_id, condition)`` triples.
        """
        tasks: dict[str, dict[str, Any]] = {}
        flows: list[tuple[str, str, Any]] = []

        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                for src, _tgt, edata in graph.in_edges(process_id, data=True):
                    if self._edge_rel(edata or {}) != "PART_OF":
                        continue
                    props = self._node_props(src)
                    if str(props.get("type", "BusinessTask")) != "BusinessTask":
                        continue
                    tasks[src] = props
                for tid in list(tasks):
                    for _src, tgt, edata in graph.out_edges(tid, data=True):
                        if self._edge_rel(edata or {}) != "FLOWS_TO":
                            continue
                        flows.append((tid, tgt, (edata or {}).get("condition")))
            except Exception as exc:  # noqa: BLE001 — compute mirror unavailable
                logger.debug("[ORCH-1.41] compute-graph traversal failed: %s", exc)
                tasks, flows = {}, []

        backend = getattr(self.engine, "backend", None)
        if not tasks and backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (t)-[:PART_OF]->(p) WHERE p.id = $pid RETURN t",
                    {"pid": process_id},
                )
                for row in rows or []:
                    node = row.get("t")
                    if not isinstance(node, dict):
                        continue
                    nid = node.get("id")
                    if nid and str(node.get("type", "BusinessTask")) == "BusinessTask":
                        tasks[str(nid)] = dict(node)
                if tasks:
                    flow_rows = backend.execute(
                        "MATCH (a)-[f:FLOWS_TO]->(b) "
                        "RETURN a.id AS src, b.id AS tgt, f.condition AS condition",
                        {},
                    )
                    for row in flow_rows or []:
                        src, tgt = row.get("src"), row.get("tgt")
                        if src in tasks and tgt in tasks:
                            flows.append((str(src), str(tgt), row.get("condition")))
            except Exception as exc:  # noqa: BLE001 — backend traversal best-effort
                logger.debug("[ORCH-1.41] backend traversal failed: %s", exc)

        # Drop flow endpoints outside the loaded task set (cross-process noise).
        flows = [(s, t, c) for s, t, c in flows if s in tasks and t in tasks]
        return tasks, flows

    # ------------------------------------------------------------------
    # Structure → dependency DAG
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_gateways(
        tasks: dict[str, dict[str, Any]],
        flows: list[tuple[str, str, Any]],
    ) -> tuple[list[str], dict[str, set[str]], list[dict[str, Any]]]:
        """Collapse gateway nodes into a dependency DAG over executable tasks.

        Returns ``(executable_ids, deps, branch_conditions)`` where ``deps``
        maps each executable task to the executable tasks it depends on, and
        ``branch_conditions`` records the surviving conditional hops.

        Raises:
            ProcessCompilationError: when the collapsed graph has a cycle.
        """
        executable = [tid for tid, p in tasks.items() if not _is_gateway(p)]
        outgoing: dict[str, list[tuple[str, Any]]] = {}
        for src, tgt, condition in flows:
            outgoing.setdefault(src, []).append((tgt, condition))

        deps: dict[str, set[str]] = {tid: set() for tid in executable}
        branch_conditions: list[dict[str, Any]] = []
        for src in executable:
            # Walk forward through gateway nodes to executable successors.
            frontier = list(outgoing.get(src, []))
            visited: set[str] = {src}
            while frontier:
                tgt, condition = frontier.pop(0)
                if tgt in deps:
                    deps[tgt].add(src)
                    if condition:
                        branch_conditions.append(
                            {"from": src, "to": tgt, "condition": condition}
                        )
                    continue
                if tgt in visited:
                    continue
                visited.add(tgt)
                for nxt, nxt_condition in outgoing.get(tgt, []):
                    frontier.append((nxt, condition or nxt_condition))

        # Kahn topological sort — leftovers are on a cycle.
        order: list[str] = []
        remaining = {tid: set(d) for tid, d in deps.items()}
        while remaining:
            ready = sorted(tid for tid, d in remaining.items() if not d)
            if not ready:
                names = sorted(
                    str(tasks[tid].get("name") or tid) for tid in remaining
                )
                raise ProcessCompilationError(
                    "BusinessProcess contains a cycle among tasks "
                    f"{names} — BPMN loop-backs cannot be compiled into a "
                    "GraphPlan DAG. Remodel the loop (e.g. as a retrying step) "
                    "or compile the acyclic portion."
                )
            for tid in ready:
                order.append(tid)
                del remaining[tid]
            for d in remaining.values():
                d.difference_update(ready)

        return order, deps, branch_conditions

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    async def compile(
        self,
        process_id: str,
        domain: str = "general",
        require_resolved: bool = False,
    ) -> GraphPlan:
        """Compile a BusinessProcess node into an executable GraphPlan.

        CONCEPT:ORCH-1.41 — structure-driven compilation: dependencies come
        from the lifted BPMN sequence flows (not NL heuristics); agents come
        from the same KG semantic matching the NL compiler uses.

        Args:
            process_id: The ``BusinessProcess`` node id (e.g.
                ``bpmn_process:invoice:1:abc``).
            domain: Domain hint for agent matching.
            require_resolved: When True, any task without a KG agent/tool
                match fails compilation (instead of becoming a manual step).

        Returns:
            Executable ``GraphPlan`` with ``depends_on`` derived from the
            sequence flows.

        Raises:
            ProcessCompilationError: missing process/structure, cycles, or
                (with ``require_resolved``) unmatched tasks.
        """
        process_props = self._node_props(process_id)
        tasks, flows = self._load_subgraph(process_id)
        if not tasks:
            raise ProcessCompilationError(
                f"BusinessProcess {process_id!r} has no BusinessTask structure "
                "in the KG. Ingest the process with the BPMN step-level lift "
                "(camunda extractor with an XML-capable client, KG-2.53) first."
            )

        order, deps, branch_conditions = self._collapse_gateways(tasks, flows)
        if not order:
            raise ProcessCompilationError(
                f"BusinessProcess {process_id!r} has only gateway structure — "
                "no executable tasks to compile."
            )

        # Resolve each executable task to a KG-registered agent/tool.
        step_ids: dict[str, str] = {}
        resolutions: dict[str, tuple[str, list[str]]] = {}
        unresolved: list[str] = []
        used_ids: set[str] = set()
        for tid in order:
            props = tasks[tid]
            label = str(props.get("name") or props.get("element_id") or tid)
            agent_id, tools = self._nl_compiler._match_agent(label, domain)
            element = _slug(props.get("element_id") or label)
            if agent_id == _UNMATCHED_SENTINEL and not tools:
                unresolved.append(label)
                step_id = f"manual:{element}"
            else:
                step_id = agent_id if agent_id not in used_ids else (
                    f"{agent_id}:{element}"
                )
            used_ids.add(step_id)
            step_ids[tid] = step_id
            resolutions[tid] = (agent_id, tools)

        if unresolved and require_resolved:
            raise ProcessCompilationError(
                "No KG-registered agent/tool matched these BusinessTasks: "
                f"{sorted(unresolved)}. Register the missing capabilities "
                "(graph_configure register_mcp / agent synthesis) or compile "
                "without require_resolved to keep them as manual steps."
            )

        # Gateway branches → parallel steps: siblings sharing one dependency
        # frontier run concurrently.
        dep_groups: dict[frozenset[str], int] = {}
        for tid in order:
            key = frozenset(step_ids[d] for d in deps[tid])
            dep_groups[key] = dep_groups.get(key, 0) + 1

        steps: list[ExecutionStep] = []
        for tid in order:
            props = tasks[tid]
            agent_id, tools = resolutions[tid]
            label = str(props.get("name") or props.get("element_id") or tid)
            depends_on = sorted(step_ids[d] for d in deps[tid])
            is_unresolved = step_ids[tid].startswith("manual:")
            step = ExecutionStep(
                id=step_ids[tid],
                refined_subtask=label,
                parallel=dep_groups[frozenset(depends_on)] > 1,
                depends_on=depends_on,
                access_list=depends_on,
                timeout=120.0,
                metadata={
                    "business_task_id": tid,
                    "task_type": props.get("task_type"),
                    **({"unresolved": True} if is_unresolved else {}),
                    **({"tools": tools} if tools else {}),
                },
            )
            steps.append(step)

        plan = GraphPlan(
            steps=steps,
            metadata={
                "source": "process_plan_compiler",
                "process_id": process_id,
                "process_name": process_props.get("name"),
                "domain": domain,
                "step_count": len(steps),
                "unresolved_tasks": sorted(unresolved),
                "branch_conditions": branch_conditions,
            },
        )
        logger.info(
            "[ORCH-1.41] Compiled process %s: %d steps (%d unresolved), agents=%s",
            process_id,
            len(steps),
            len(unresolved),
            [s.id for s in steps],
        )
        return plan

    async def compile_and_store(
        self,
        process_id: str,
        name: str | None = None,
        domain: str = "general",
        require_resolved: bool = False,
    ) -> dict[str, Any]:
        """Compile a BusinessProcess and persist it as a WorkflowDefinition.

        CONCEPT:ORCH-1.41 — persists via the shared ``WorkflowStore`` and
        records the bridge with a ``REALIZES`` edge
        ``(:WorkflowDefinition)-[:REALIZES]->(:BusinessProcess)``
        (``:realizesProcess`` in the ontology) so lineage close-out
        (ORCH-1.43) and provenance queries can traverse it.

        Returns:
            Report dict: ``workflow_id``, ``name``, ``step_count``,
            ``unresolved_tasks``, ``process_id``.
        """
        plan = await self.compile(
            process_id, domain=domain, require_resolved=require_resolved
        )
        process_name = plan.metadata.get("process_name")
        workflow_name = name or f"process_{_slug(process_name or process_id)}"
        workflow_id = self.store.save_workflow(
            name=workflow_name,
            plan=plan,
            description=(
                f"Compiled from BusinessProcess {process_id}"
                + (f" ({process_name})" if process_name else "")
            ),
            metadata={
                "source": "process_plan_compiler",
                "process_id": process_id,
                "domain": domain,
            },
        )
        # The bridge edge — the workflow REALIZES the descriptive process.
        self.engine.link_nodes(workflow_id, process_id, "REALIZES")
        logger.info(
            "[ORCH-1.41] Stored workflow %s REALIZES %s", workflow_id, process_id
        )
        return {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "step_count": len(plan.steps),
            "unresolved_tasks": plan.metadata.get("unresolved_tasks", []),
            "process_id": process_id,
        }
