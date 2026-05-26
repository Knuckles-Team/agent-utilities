#!/usr/bin/python
"""X Research & Knowledge Assimilation Workflow Definitions.

CONCEPT:ORCH-1.24 — Workflow Lifecycle Management
CONCEPT:ECO-4.0 — Social Content Ingestion
CONCEPT:KG-2.6 — Universal Knowledge Assimilation

Pre-built workflow definitions for X search/ingestion and universal
knowledge assimilation. These are registered in the KG via
``WorkflowStore.save_workflow()`` during engine initialization or
on first use.

Usage::

    from agent_utilities.knowledge_graph.kb.x_workflows import (
        register_x_workflows,
    )

    register_x_workflows(engine)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ...models.graph import ExecutionStep, GraphPlan

if TYPE_CHECKING:
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Workflow: x_research
# --------------------------------------------------------------------------- #


def _x_research_plan() -> GraphPlan:
    """Create the X Research & Ingestion workflow plan.

    Steps:
        0. x-search-agent — Search X posts or browse a specific URL
        1. graph-os — Classify and ingest results into the KG
    """
    return GraphPlan(
        steps=[
            ExecutionStep(
                node_id="x-search-agent",
                refined_subtask=(
                    "Search X (formerly Twitter) using x_search for the given "
                    "topic query, or browse a specific post URL using "
                    "browse_x_post with auto_ingest=True. Retrieve the text, "
                    "author metadata, engagement statistics, and any article "
                    "content. Target/Query: {{task}}"
                ),
                is_parallel=False,
                timeout=180,
            ),
            ExecutionStep(
                node_id="graph-os",
                refined_subtask=(
                    "Classify and ingest each X search result into the "
                    "Knowledge Graph. For each result:\n"
                    "1. Score using UniversalKnowledgeClassifier\n"
                    "2. Create SocialPost node with engagement metrics\n"
                    "3. Create Person node and CREATED_BY_PERSON edge\n"
                    "4. Extract concepts and create ABOUT edges\n"
                    "5. If evolution potential >= 0.6, create EvolutionCandidate\n"
                    "6. For X Articles, fetch full content and ingest as Article"
                ),
                depends_on=["x-search-agent"],
                is_parallel=False,
                timeout=120,
            ),
        ],
        metadata={
            "workflow_type": "x_research",
            "concepts": ["CONCEPT:ECO-4.0"],
            "domain": "social",
        },
    )


# --------------------------------------------------------------------------- #
# Workflow: knowledge_assimilation
# --------------------------------------------------------------------------- #


def _knowledge_assimilation_plan() -> GraphPlan:
    """Create the Universal Knowledge Assimilation workflow plan.

    Steps:
        0. scout — Discover content from X, ScholarX, GitHub
        1. classify — Score with UniversalKnowledgeClassifier
        2. ingest — Persist high-value content to KG
        3. analyze — Run comparative-analysis on evolution candidates
        4. plan — Generate SDD plans for actionable gaps
    """
    return GraphPlan(
        steps=[
            ExecutionStep(
                node_id="scout",
                refined_subtask=(
                    "Discover content from multiple sources in parallel:\n"
                    "1. X Search: Use x_search for trending AI/ML/agent topics\n"
                    "2. ScholarX: Use sx_search with action='recent', "
                    "categories='cs.AI,cs.MA,cs.CL,cs.SE', days=3\n"
                    "3. GitHub: Search repositories for ai-agents, "
                    "knowledge-graph, mcp-server topics\n"
                    "4. KG Memory: Query pending EvolutionCandidate nodes\n"
                    "Target: {{task}}"
                ),
                is_parallel=False,
                timeout=300,
            ),
            ExecutionStep(
                node_id="classifier",
                refined_subtask=(
                    "Score each discovered item using the "
                    "UniversalKnowledgeClassifier:\n"
                    "1. Gather existing KG topics for context matching\n"
                    "2. For X posts, use browse_x_post with auto_ingest=True\n"
                    "3. For papers, classify with source_type='research_paper'\n"
                    "4. For repos, classify with source_type='github_repo'\n"
                    "5. Route based on classification action: skip/decay/ingest/"
                    "ingest_and_evolve"
                ),
                depends_on=["scout"],
                is_parallel=False,
                timeout=180,
            ),
            ExecutionStep(
                node_id="ingester",
                refined_subtask=(
                    "Ingest high-value and evolution-candidate content into "
                    "the Knowledge Graph:\n"
                    "1. X Posts/Articles: via XIngestionBridge\n"
                    "2. Research Papers: download PDF and ingest via "
                    "graph_ingest\n"
                    "3. GitHub Repos: ingest repository via graph_ingest\n"
                    "4. Link to extracted concepts via ABOUT edges"
                ),
                depends_on=["classifier"],
                is_parallel=False,
                timeout=300,
            ),
            ExecutionStep(
                node_id="analyst",
                refined_subtask=(
                    "Run comparative analysis on evolution candidates:\n"
                    "1. Use graph_analyze relevance_sweep against agent-utilities\n"
                    "2. Run deep_extract for actionable feature gaps\n"
                    "3. Update EvolutionCandidate status to 'analyzed'"
                ),
                depends_on=["ingester"],
                is_parallel=False,
                timeout=300,
            ),
            ExecutionStep(
                node_id="planner",
                refined_subtask=(
                    "Generate SDD implementation plans for actionable gaps:\n"
                    "1. Include constitution-mandated artifacts\n"
                    "2. Cross-reference with existing SDD plans\n"
                    "3. Present plan for user review\n"
                    "4. Log EvolutionCycle node in KG"
                ),
                depends_on=["analyst"],
                is_parallel=False,
                timeout=120,
            ),
        ],
        metadata={
            "workflow_type": "knowledge_assimilation",
            "concepts": ["CONCEPT:KG-2.6", "CONCEPT:ECO-4.0"],
            "domain": "research",
        },
    )


# --------------------------------------------------------------------------- #
# Workflow: self_evolution_v2
# --------------------------------------------------------------------------- #


def _self_evolution_v2_plan() -> GraphPlan:
    """Create the Self-Evolution v2 workflow plan.

    A streamlined variant of knowledge_assimilation focused on
    processing pending EvolutionCandidateNode entries in the KG.
    """
    return GraphPlan(
        steps=[
            ExecutionStep(
                node_id="graph-os",
                refined_subtask=(
                    "Query the KG for pending EvolutionCandidate nodes:\n"
                    "MATCH (e:EvolutionCandidate {status: 'pending'}) "
                    "RETURN e ORDER BY e.evolution_score DESC LIMIT 5\n"
                    "For each candidate, run comparative-analysis against "
                    "agent-utilities and generate an SDD plan if actionable "
                    "gaps are found. Update candidate status to 'analyzed'."
                ),
                is_parallel=False,
                timeout=300,
            ),
        ],
        metadata={
            "workflow_type": "self_evolution_v2",
            "concepts": ["CONCEPT:KG-2.6", "CONCEPT:AHE-3.0"],
            "domain": "research",
            "description": (
                "Push-based evolution triggered by incoming high-potential content"
            ),
        },
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_WORKFLOW_REGISTRY: dict[str, tuple[str, str, Callable[[], GraphPlan]]] = {
    "x_research": (
        "X Research & Ingestion",
        "Search X posts, classify content value, and ingest into KG",
        _x_research_plan,
    ),
    "knowledge_assimilation": (
        "Universal Knowledge Assimilation",
        "Multi-source content discovery → classify → ingest → evolve pipeline",
        _knowledge_assimilation_plan,
    ),
    "self_evolution_v2": (
        "Self-Evolution v2",
        "Push-based evolution triggered by incoming high-potential content",
        _self_evolution_v2_plan,
    ),
}


def register_x_workflows(
    engine: "IntelligenceGraphEngine",
    force: bool = False,
) -> dict[str, str]:
    """Register all X/assimilation workflows in the KG.

    Args:
        engine: The IntelligenceGraphEngine instance.
        force: If True, re-register even if already present.

    Returns:
        Dict mapping workflow names to their KG node IDs.
    """
    from ...knowledge_graph.workflow_store import WorkflowStore

    store = WorkflowStore(engine)
    registered: dict[str, str] = {}

    for key, (name, description, plan_fn) in _WORKFLOW_REGISTRY.items():
        # Check if already registered
        if not force:
            existing = store.load_workflow(key)
            if existing is not None:
                logger.debug("Workflow '%s' already registered, skipping", key)
                registered[key] = f"workflow:{key}:existing"
                continue

        plan = plan_fn()
        workflow_id = store.save_workflow(
            name=key,
            plan=plan,
            description=description,
            nl_spec=description,
            metadata={
                "display_name": name,
                "registered_by": "x_workflows",
            },
        )
        registered[key] = workflow_id
        logger.info("Registered workflow: %s → %s", key, workflow_id)

    return registered


def get_workflow_plan(workflow_name: str) -> GraphPlan | None:
    """Get a workflow plan by name without requiring a KG engine.

    Useful for direct execution without KG persistence.

    Args:
        workflow_name: One of: x_research, knowledge_assimilation, self_evolution_v2

    Returns:
        The GraphPlan if found, None otherwise.
    """
    entry = _WORKFLOW_REGISTRY.get(workflow_name)
    if entry is None:
        return None
    return entry[2]()
