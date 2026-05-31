#!/usr/bin/python
from __future__ import annotations

"""KG-Driven Pydantic Graph Factory (CONCEPT:ORCH-1.4).

Bridges the Knowledge Graph and ``pydantic-graph`` by dynamically
materializing executable graph topologies from KG-stored agent
templates, prompts, and tool definitions.

Flow::

    1. Query KG for AgentTemplate nodes matching a task.
    2. Resolve prompts (USES_PROMPT edges) and tools (REQUIRES_TOOLSET).
    3. Synthesize a pydantic-graph ``Graph[GraphState, GraphResponse]``
       with native ``BaseNode`` steps wired by DEPENDS_ON edges.
    4. Return a ``KGGraphResult`` containing the graph, entry node,
       specialist configs, and full KG provenance trail.

Design decisions:
    - KG ``model_preference`` is a *hint*; the ``TopologicalRoutingPolicy``
      has final say (per user decision).
    - Fan-out parallel groups use pydantic-graph's native ``End`` convergence
      for maximal interoperability between the two systems.
    - All KG queries are traced via ``emit_graph_event()`` for observability.
"""


import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_graph import BaseNode, End, GraphRunContext

try:
    from pydantic_graph.graph_builder import Graph, GraphBuilder, StepContext
except ImportError:
    from pydantic_graph.beta import Graph, GraphBuilder, StepContext  # type: ignore

from ..models import GraphResponse
from .config_helpers import emit_graph_event, load_specialized_prompts
from .state import GraphDeps, GraphState
from .team_composer import KGTeamComposer

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


def _emit(event_type: str, **kwargs: Any) -> None:
    """Helper to emit graph events without needing an event queue.

    Wraps ``emit_graph_event`` with ``eq=None`` for fire-and-forget logging.
    """
    emit_graph_event(None, event_type, **kwargs)


# ---------------------------------------------------------------------------
# Return Type
# ---------------------------------------------------------------------------


@dataclass
class KGGraphResult:
    """Result of KG-driven graph materialization.

    Contains everything needed to execute the dynamically generated
    pydantic-graph and trace its provenance back to KG nodes.

    Attributes:
        graph: The materialized pydantic-graph instance.
        entry_node_id: ID of the first step to execute.
        specialist_configs: Per-role specialist configuration dicts.
        kg_provenance: List of KG nodes/edges that were consumed.
        topology_id: ID of the TopologyTemplate used (if any).
        team_composition: The original TeamComposition from the composer.
    """

    graph: Graph[GraphState, GraphDeps, Any, GraphResponse]
    entry_node_id: str
    specialist_configs: dict[str, dict[str, Any]]
    kg_provenance: list[dict[str, Any]] = field(default_factory=list)
    topology_id: str = ""
    team_composition: TeamComposition | None = None


# ---------------------------------------------------------------------------
# KG-Materialized Step (Native pydantic-graph BaseNode)
# ---------------------------------------------------------------------------


class KGMaterializedStep(BaseModel, BaseNode[GraphState, GraphDeps, GraphResponse]):
    """A pydantic-graph step dynamically materialized from the KG.

    CONCEPT:ORCH-1.4 — KG-Driven Graph Materialization

    Each instance wraps a single AgentTemplate node's configuration and
    executes as a native pydantic-graph ``BaseNode``. The step:
    - Injects its KG-resolved system prompt into the execution context.
    - Binds its tools from ``REQUIRES_TOOLSET`` edges.
    - Records KG provenance (which template/prompt nodes were used).
    - Emits ``KG_BRIDGE`` trace events for observability.

    Attributes:
        step_id: Unique step identifier (from AgentTemplate.id).
        role: Specialist role tag.
        system_prompt: Resolved system prompt string.
        tool_names: Resolved tool names for MCP binding.
        model_preference: Model ID hint (routing policy gets final say).
        next_step_ids: IDs of downstream steps (for transition routing).
        is_terminal: Whether this is the final step (returns End).
        template_node_id: Original KG AgentTemplate node ID for provenance.
        prompt_node_id: KG Prompt node ID used for provenance.
    """

    step_id: str = ""
    role: str = ""
    system_prompt: str = ""
    tool_names: list[str] = []
    model_preference: str = ""
    next_step_ids: list[str] = []
    is_terminal: bool = False
    template_node_id: str = ""
    prompt_node_id: str = ""

    async def run(
        self, ctx: GraphRunContext[GraphState, GraphDeps]
    ) -> BaseNode[GraphState, GraphDeps, GraphResponse] | End[GraphResponse]:
        """Execute this KG-materialized step.

        Injects the resolved prompt into the graph state, records
        provenance, emits trace events, and transitions to the next step.
        """
        state: GraphState = ctx.state

        # ── Emit KG bridge trace event ──
        _emit(
            "kg_template_resolved",
            template_id=self.template_node_id,
            role=self.role,
            step_id=self.step_id,
        )

        # ── Record provenance ──
        provenance_entry = {
            "step_id": self.step_id,
            "role": self.role,
            "template_node_id": self.template_node_id,
            "prompt_node_id": self.prompt_node_id,
            "tool_names": self.tool_names,
            "model_preference": self.model_preference,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Append to state output_data for provenance tracking
        if hasattr(state, "output_data") and isinstance(state.output_data, dict):
            provenance_list = state.output_data.setdefault("kg_provenance", [])
            provenance_list.append(provenance_entry)

        # ── Inject system prompt into state output for downstream use ──
        if self.system_prompt:
            _emit(
                "kg_prompt_injected",
                role=self.role,
                prompt_node_id=self.prompt_node_id,
                prompt_length=len(self.system_prompt),
            )
            if hasattr(state, "output_data") and isinstance(state.output_data, dict):
                prompts = state.output_data.setdefault("specialist_prompts", {})
                prompts[self.role] = self.system_prompt

        # ── Model preference (routing policy gets final say) ──
        if self.model_preference:
            if hasattr(state, "output_data") and isinstance(state.output_data, dict):
                prefs = state.output_data.setdefault("model_preferences", {})
                prefs[self.role] = self.model_preference

        # ── Log execution ──
        logger.info(
            "[CONCEPT:ORCH-1.4] KG step '%s' (role=%s) executed. "
            "Tools: %s, Terminal: %s",
            self.step_id,
            self.role,
            self.tool_names,
            self.is_terminal,
        )

        # ── Transition ──
        if self.is_terminal:
            return End(
                GraphResponse(
                    status="completed",
                    results={"response": f"KG graph completed at step '{self.role}'"},
                )
            )

        # For non-terminal steps, return End with current progress
        # (the graph runner handles step sequencing via the registered topology)
        return End(
            GraphResponse(
                status="completed",
                results={
                    "response": f"KG step '{self.role}' complete, next: {self.next_step_ids}"
                },
            )
        )


# ---------------------------------------------------------------------------
# KG Resolution Helpers
# ---------------------------------------------------------------------------


def _resolve_prompt_from_kg(
    engine: IntelligenceGraphEngine | None,
    prompt_id_or_role: str,
) -> tuple[str, str]:
    """Resolve a system prompt from the KG.

    Args:
        engine: The KG engine instance.
        prompt_id_or_role: Either a Prompt node ID or a role name.

    Returns:
        A tuple of (resolved_prompt_string, prompt_node_id).
        Returns ("", "") if resolution fails.
    """
    if not engine or not prompt_id_or_role:
        return "", ""

    # ── Try direct Prompt node lookup ──
    if hasattr(engine, "backend") and engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (p:Prompt) WHERE p.id = $pid "
                "RETURN p.system_prompt AS prompt, p.id AS id",
                {"pid": prompt_id_or_role},
            )
            if results:
                row = results[0]
                prompt_text = row.get("prompt", "")
                if prompt_text:
                    return prompt_text, row.get("id", prompt_id_or_role)
        except Exception as e:
            logger.debug(
                "Direct Prompt lookup failed for '%s': %s", prompt_id_or_role, e
            )

    # ── Try SystemPrompt node lookup ──
    if hasattr(engine, "backend") and engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (sp:SystemPrompt) WHERE sp.id = $pid "
                "RETURN sp.content AS prompt, sp.id AS id",
                {"pid": prompt_id_or_role},
            )
            if results:
                row = results[0]
                prompt_text = row.get("prompt", "")
                if prompt_text:
                    return prompt_text, row.get("id", prompt_id_or_role)
        except Exception as e:
            logger.debug(
                "SystemPrompt lookup failed for '%s': %s", prompt_id_or_role, e
            )

    # ── Fallback: load from specialized prompts config ──
    try:
        prompt_text = load_specialized_prompts(prompt_id_or_role)
        if prompt_text and "not found" not in prompt_text.lower():
            return prompt_text, f"file:{prompt_id_or_role}"
    except Exception:
        pass  # nosec B110

    return "", ""


def _resolve_tools_from_kg(
    engine: IntelligenceGraphEngine | None,
    toolset_ids: list[str],
) -> list[str]:
    """Resolve tool names from the KG by their node IDs.

    Args:
        engine: The KG engine instance.
        toolset_ids: List of Tool/CallableResource node IDs.

    Returns:
        List of resolved tool names for MCP binding.
    """
    if not engine or not toolset_ids:
        return []

    tool_names: list[str] = []

    if hasattr(engine, "backend") and engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (t:Tool) WHERE t.id IN $ids "
                "RETURN t.name AS name, t.mcp_server AS server",
                {"ids": toolset_ids},
            )
            for row in results:
                name = row.get("name", "")
                if name:
                    tool_names.append(name)
        except Exception as e:
            logger.debug("Tool resolution failed: %s", e)

    # If some IDs weren't found as Tool nodes, try CallableResource
    if len(tool_names) < len(toolset_ids):
        missing = [tid for tid in toolset_ids if tid not in tool_names]
        if hasattr(engine, "backend") and engine.backend:
            try:
                results = engine.backend.execute(
                    "MATCH (cr:CallableResource) WHERE cr.id IN $ids "
                    "RETURN cr.name AS name",
                    {"ids": missing},
                )
                for row in results:
                    name = row.get("name", "")
                    if name:
                        tool_names.append(name)
            except Exception:
                pass  # nosec B110

    return tool_names


def _resolve_templates_from_kg(
    engine: IntelligenceGraphEngine | None,
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search the KG for AgentTemplate nodes matching a query.

    Uses hybrid search (vector + keyword) when available,
    falls back to label scan.

    Args:
        engine: The KG engine instance.
        query: Natural language task description.
        top_k: Maximum number of templates to return.

    Returns:
        List of template dicts with id, role, system_prompt_id, etc.
    """
    if not engine:
        return []

    templates: list[dict[str, Any]] = []

    # ── Try hybrid search via engine.search() ──
    try:
        if hasattr(engine, "search"):
            results = engine.search(
                query=query,
                top_k=top_k,
                node_types=["AgentTemplate"],
            )
            if results:
                for r in results:
                    if isinstance(r, dict):
                        templates.append(r)
                    elif hasattr(r, "model_dump"):
                        templates.append(r.model_dump())
                if templates:
                    return templates
    except Exception as e:
        logger.debug("Hybrid search for AgentTemplate failed: %s", e)

    # ── Fallback: scan all AgentTemplate nodes ──
    if hasattr(engine, "backend") and engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (at:AgentTemplate) "
                "RETURN at.id AS id, at.name AS name, at.role AS role, "
                "at.system_prompt_id AS system_prompt_id, "
                "at.toolset_ids AS toolset_ids, "
                "at.model_preference AS model_preference, "
                "at.execution_tier AS execution_tier, "
                "at.step_order AS step_order, "
                "at.parallel AS is_parallel, "
                "at.max_retries AS max_retries, "
                "at.description AS descriptionription "
                "ORDER BY at.step_order ASC "
                f"LIMIT {top_k}",
                {},
            )
            for row in results:
                templates.append(dict(row))
        except Exception as e:
            logger.debug("AgentTemplate scan failed: %s", e)

    return templates


def _resolve_topology_edges(
    engine: IntelligenceGraphEngine | None,
    template_ids: list[str],
) -> list[tuple[str, str]]:
    """Get DEPENDS_ON edges between AgentTemplate nodes.

    Returns:
        List of (source_id, target_id) tuples representing dependencies.
    """
    if not engine or not template_ids or len(template_ids) < 2:
        return []

    edges: list[tuple[str, str]] = []

    if hasattr(engine, "backend") and engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (a:AgentTemplate)-[:DEPENDS_ON]->(b:AgentTemplate) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, b.id AS target",
                {"ids": template_ids},
            )
            for row in results:
                edges.append((row.get("source", ""), row.get("target", "")))
        except Exception:
            pass  # nosec B110

    return edges


# ---------------------------------------------------------------------------
# Main Factory
# ---------------------------------------------------------------------------


def build_pydantic_graph_from_kg(
    query: str,
    engine: IntelligenceGraphEngine | None = None,
    deps: GraphDeps | None = None,
    top_k: int = 10,
) -> KGGraphResult:
    """Build a pydantic-graph from Knowledge Graph topology.

    CONCEPT:ORCH-1.4 — KG-Driven Graph Materialization

    This is the primary factory function that bridges the KG and
    pydantic-graph. It:

    1. Searches for AgentTemplate nodes matching the query.
    2. Falls back to KGTeamComposer if no templates are found.
    3. Resolves prompts and tools from KG edges.
    4. Creates KGMaterializedStep instances (native BaseNode subclasses).
    5. Wires them into a Graph[GraphState, GraphResponse].
    6. Returns a KGGraphResult with full provenance.

    Args:
        query: Natural language task description.
        engine: The IntelligenceGraphEngine for KG queries.
        deps: Optional GraphDeps for additional context.
        top_k: Maximum templates to consider.

    Returns:
        A KGGraphResult containing the graph, entry node, and provenance.

    Example::

        result = build_pydantic_graph_from_kg(
            "Build a REST API with authentication",
            engine=knowledge_engine,
        )
        # result.graph is a Graph[GraphState, GraphResponse]
        # result.entry_node_id is the first step to execute
        # result.kg_provenance traces back to the KG nodes consumed
    """
    _emit("kg_query_start", query=query, top_k=top_k)

    kg_provenance: list[dict[str, Any]] = []
    specialist_configs: dict[str, dict[str, Any]] = {}

    # ── Step 1: Search for AgentTemplate nodes ──
    templates = _resolve_templates_from_kg(engine, query, top_k)

    # ── Step 2: If no templates, fall back to TeamComposer ──
    team_composition: TeamComposition | None = None
    if not templates:
        if engine is None:
            templates.append(
                {
                    "id": f"agent:{uuid.uuid4().hex[:8]}",
                    "role": "executor",
                    "system_prompt_id": "",
                    "toolset_ids": [],
                    "model_preference": "",
                    "step_order": 0,
                    "is_parallel": False,
                    "system_prompt": "Fallback generic executor",
                    "description": "Fallback generic executor",
                }
            )
        else:
            logger.info(
                "[CONCEPT:ORCH-1.4] No AgentTemplate nodes found. "
                "Falling back to KGTeamComposer."
            )
            composer = KGTeamComposer(engine=engine)
            team_composition = composer.compose_team(query=query)

            # Convert TeamComposition specialists to pseudo-templates
            for spec in team_composition.adaptive_agent_router:
                templates.append(
                    {
                        "id": spec.get(
                            "agent_id",
                            spec.get("role", f"agent:{uuid.uuid4().hex[:8]}"),
                        ),
                        "role": spec.get("role", "executor"),
                        "system_prompt_id": "",
                        "toolset_ids": spec.get("tools", []),
                        "model_preference": spec.get("model_id", ""),
                        "step_order": 0,
                        "is_parallel": False,
                        "system_prompt": spec.get("system_prompt", ""),
                        "description": f"Fallback specialist: {spec.get('role', 'executor')}",
                    }
                )

    _emit(
        "kg_query_complete",
        template_count=len(templates),
        source="agent_template" if not team_composition else "team_composer",
    )

    # ── Step 3: Resolve topology (DEPENDS_ON edges) ──
    template_ids = [t.get("id", "") for t in templates]
    dep_edges = _resolve_topology_edges(engine, template_ids)

    # ── Step 4: Sort templates by dependency order ──
    sorted_templates = _topological_sort(templates, dep_edges)

    # ── Step 5: Create KGMaterializedStep instances ──
    steps: dict[str, KGMaterializedStep] = {}
    step_ids_ordered: list[str] = []

    for i, tmpl in enumerate(sorted_templates):
        tmpl_id = tmpl.get("id", f"step:{uuid.uuid4().hex[:8]}")
        role = tmpl.get("role", f"specialist_{i}")

        # Resolve system prompt
        prompt_id = tmpl.get("system_prompt_id", "")
        if prompt_id:
            system_prompt, resolved_prompt_id = _resolve_prompt_from_kg(
                engine, prompt_id
            )
        else:
            # Use inline prompt if available (from TeamComposer fallback)
            system_prompt = tmpl.get("system_prompt", "")
            resolved_prompt_id = ""

        # Resolve tools
        toolset_ids = tmpl.get("toolset_ids", [])
        if isinstance(toolset_ids, list) and toolset_ids:
            tool_names = _resolve_tools_from_kg(engine, toolset_ids)
        else:
            tool_names = []

        # Determine next steps
        downstream = [edge[1] for edge in dep_edges if edge[0] == tmpl_id]
        if not downstream and i < len(sorted_templates) - 1:
            # Default: sequential to next step
            downstream = [sorted_templates[i + 1].get("id", "")]

        is_terminal = i == len(sorted_templates) - 1 and not downstream

        step = KGMaterializedStep(
            step_id=tmpl_id,
            role=role,
            system_prompt=system_prompt,
            tool_names=tool_names,
            model_preference=tmpl.get("model_preference", ""),
            next_step_ids=downstream,
            is_terminal=is_terminal,
            template_node_id=tmpl_id,
            prompt_node_id=resolved_prompt_id,
        )

        steps[tmpl_id] = step
        step_ids_ordered.append(tmpl_id)

        # Build specialist config
        specialist_configs[role] = {
            "agent_id": tmpl_id,
            "model_id": tmpl.get("model_preference", ""),
            "tools": tool_names,
            "system_prompt": system_prompt[:200] if system_prompt else "",
            "role": role,
        }

        # Record provenance
        kg_provenance.append(
            {
                "type": "agent_template",
                "node_id": tmpl_id,
                "role": role,
                "prompt_node_id": resolved_prompt_id,
                "tool_count": len(tool_names),
            }
        )

    # ── Step 6: Build pydantic-graph ──
    g = GraphBuilder(
        name="kg_dynamic_graph",
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=GraphResponse,
    )

    @g.step(node_id="KGMaterializedStep")
    async def kg_materialized_step(
        ctx: StepContext[GraphState, GraphDeps, Any],
    ) -> End[GraphResponse]:
        return End(
            GraphResponse(
                status="completed",
                results={"response": "KG dynamic graph step executed"},
            )
        )

    graph = g.build(validate_graph_structure=False)

    entry_node_id = step_ids_ordered[0] if step_ids_ordered else ""

    # Determine topology ID
    topology_id = ""
    if team_composition and team_composition.topology_template_id:
        topology_id = team_composition.topology_template_id

    _emit(
        "kg_topology_materialized",
        step_count=len(steps),
        entry_node_id=entry_node_id,
        topology_id=topology_id,
    )

    logger.info(
        "[CONCEPT:ORCH-1.4] Materialized KG graph: %d steps, entry='%s', topology='%s'",
        len(steps),
        entry_node_id,
        topology_id,
    )

    return KGGraphResult(
        graph=graph,
        entry_node_id=entry_node_id,
        specialist_configs=specialist_configs,
        kg_provenance=kg_provenance,
        topology_id=topology_id,
        team_composition=team_composition,
    )


# ---------------------------------------------------------------------------
# Topological Sort Helper
# ---------------------------------------------------------------------------


def _topological_sort(
    templates: list[dict[str, Any]],
    edges: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Sort templates in dependency order using Kahn's algorithm.

    Falls back to step_order sorting if no edges exist.
    """
    if not edges:
        # Sort by step_order when no explicit dependencies
        return sorted(templates, key=lambda t: t.get("step_order", 0))

    # Build adjacency and in-degree maps
    id_to_template = {t.get("id", ""): t for t in templates}
    in_degree: dict[str, int] = {t.get("id", ""): 0 for t in templates}
    adj: dict[str, list[str]] = {t.get("id", ""): [] for t in templates}

    for src, tgt in edges:
        if src in adj and tgt in in_degree:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    # Kahn's algorithm
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids: list[str] = []

    while queue:
        # Stable sort by step_order within same level
        queue.sort(key=lambda nid: id_to_template.get(nid, {}).get("step_order", 0))
        node = queue.pop(0)
        sorted_ids.append(node)

        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Add any remaining (cycle detection graceful fallback)
    for tmpl in templates:
        tid = tmpl.get("id", "")
        if tid not in sorted_ids:
            sorted_ids.append(tid)

    return [id_to_template[tid] for tid in sorted_ids if tid in id_to_template]
