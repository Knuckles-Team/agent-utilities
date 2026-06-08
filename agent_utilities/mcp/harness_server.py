#!/usr/bin/python
"""Agentic Harness MCP Server — Exposes DSTDD and agent-native tools.

CONCEPT:ECO-4.1 — Harness MCP Exposure

Exposes the DSTDD design governance pipeline, agent self-model,
team composition, and evaluation history as MCP tools for external
agents and IDEs.

Architecture:
    Companion server to ``kg_server.py``. While the KG server exposes
    graph operations, this server exposes the agentic harness:
    design governance, agent capabilities, and orchestration triggers.

Usage:
    python -m agent_utilities.mcp.harness_server
    python -m agent_utilities.mcp.harness_server --transport streamable-http --port 8101
"""

from __future__ import annotations

import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)

_AGENT_ID = os.environ.get("AGENT_ID", f"harness-{uuid.uuid4().hex[:8]}")
_SESSION_ID = os.environ.get("SESSION_ID", uuid.uuid4().hex)


def _build_server():
    """Build the Harness MCP server with all tools registered."""
    from agent_utilities.mcp.server_factory import create_mcp_server

    args, mcp, middlewares = create_mcp_server(
        name="agent-utilities-harness",
        version="0.1.0",
        instructions=(
            "Agentic Harness MCP Server for agent-utilities. "
            "Provides DSTDD governance tools, agent self-model queries, "
            "team composition, and evaluation history. "
            "Use dstdd_* tools for design-first development governance."
        ),
    )

    # --- DSTDD Governance Tools ---

    @mcp.tool()
    def dstdd_create_design(
        feature_id: str,
        title: str,
        description: str = "",
        nearest_concepts: str = "[]",
        extension_strategy: str = "augment",
        extension_point: str = "",
    ) -> str:
        """Create a design document for a new feature (ORCH-1.6).

        The first step in the DSTDD pipeline. Creates a design document
        with KG analysis and persists it to `.specify/design/`.

        Args:
            feature_id: Unique identifier for the feature (e.g., 'enhanced-routing').
            title: Human-readable title for the design.
            description: Detailed description of the feature.
            nearest_concepts: JSON list of nearest concepts from KG search.
                Each: {"concept_id": "...", "name": "...", "similarity": 0.X, "pillar": "..."}
            extension_strategy: One of 'augment', 'compose', 'specialize', 'new'.
            extension_point: The CONCEPT:ID being extended (if any).

        Returns:
            JSON confirmation with design path and validation preview.
        """
        try:
            from agent_utilities.models.sdd import (
                ExtensionStrategy,
                KGAnalysis,
                NearestConcept,
            )
            from agent_utilities.sdd import SDDManager

            workspace = os.environ.get("WORKSPACE_PATH", os.getcwd())
            mgr = SDDManager(workspace)

            # Parse nearest concepts
            concepts = []
            for c in json.loads(nearest_concepts):
                concepts.append(
                    NearestConcept(
                        concept_id=c.get("concept_id", ""),
                        name=c.get("name", ""),
                        similarity=c.get("similarity", 0.0),
                        pillar=c.get("pillar", ""),
                    )
                )

            strategy = ExtensionStrategy(extension_strategy)
            kg_analysis = KGAnalysis(
                nearest_concepts=concepts,
                extension_point=extension_point or None,
                extension_strategy=strategy,
            )

            doc = mgr.create_design(feature_id, kg_analysis, title=title)
            return json.dumps(
                {
                    "status": "created",
                    "feature_id": doc.feature_id,
                    "title": doc.title,
                    "strategy": doc.kg_analysis.extension_strategy.value,
                    "path": str(mgr.specify_dir / "design" / feature_id / "design.md"),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def dstdd_validate_design(feature_id: str) -> str:
        """Validate a design document against Extend-Before-Invent rules (ORCH-1.6).

        Checks governance compliance:
        - High-similarity concepts MUST extend, not create new.
        - New concepts MUST have a proposal with pillar assignment.
        - Extension points MUST be valid.

        Args:
            feature_id: The feature ID to validate.

        Returns:
            JSON with validation result and any violations.
        """
        try:
            from agent_utilities.sdd import SDDManager

            workspace = os.environ.get("WORKSPACE_PATH", os.getcwd())
            mgr = SDDManager(workspace)
            violations = mgr.validate_design(feature_id)
            return json.dumps(
                {
                    "valid": len(violations) == 0,
                    "feature_id": feature_id,
                    "violation_count": len(violations),
                    "violations": violations,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def dstdd_design_to_spec(feature_id: str) -> str:
        """Generate a spec from a validated design document (ORCH-1.6).

        Converts a passing design into a specification with user stories
        and acceptance criteria. The design must pass validation first.

        Args:
            feature_id: The feature ID to convert.

        Returns:
            JSON with spec path and generated user stories.
        """
        try:
            from agent_utilities.sdd import SDDManager

            workspace = os.environ.get("WORKSPACE_PATH", os.getcwd())
            mgr = SDDManager(workspace)
            spec = mgr.design_to_spec(feature_id)
            return json.dumps(
                {
                    "status": "generated",
                    "feature_id": feature_id,
                    "title": spec.title,
                    "user_story_count": len(spec.user_stories),
                    "path": str(mgr.specify_dir / "specs" / feature_id / "spec.md"),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    # --- Agent Capability Tools ---

    @mcp.tool()
    def agent_list_tools() -> str:
        """List all registered tools and skills in the agent ecosystem.

        Discovers tools from the tool registry, MCP servers, and
        pydantic-ai function tools.

        Returns:
            JSON with tool categories and their descriptions.
        """
        try:
            tool_categories = {
                "workspace": [
                    "read_file",
                    "write_file",
                    "search_files",
                    "list_directory",
                ],
                "git": ["git_status", "git_log", "git_diff", "git_commit"],
                "a2a": ["delegate_to_agent", "list_available_agents"],
                "mcp": ["trigger_mcp_sync"],
                "developer": ["run_command", "install_package"],
                "self_improvement": [
                    "run_self_improvement_cycle",
                    "propose_skills_from_history",
                ],
                "kg_mcp": [
                    "kg_query",
                    "kg_search",
                    "kg_concept_search",
                    "kg_pillar_view",
                    "kg_get_stats",
                    "kg_analogy_search",
                    "kg_blast_radius",
                    "kg_memory_recall",
                    "kg_memory_store",
                    "kg_ingest_batch",
                    "kg_ontology_validate",
                    "kg_write_node",
                    "kg_link_nodes",
                    "kg_get_constitution",
                ],
                "harness_mcp": [
                    "dstdd_create_design",
                    "dstdd_validate_design",
                    "dstdd_design_to_spec",
                    "agent_list_tools",
                    "agent_self_model",
                    "agent_team_compose",
                    "agent_eval_history",
                ],
            }
            total = sum(len(v) for v in tool_categories.values())
            return json.dumps({"total_tools": total, "categories": tool_categories})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def agent_self_model(agent_name: str = "default") -> str:
        """Query an agent's self-assessed capabilities (AHE-3.4).

        Returns the agent's self-model: its strengths, weaknesses,
        recent performance scores, and capability ratings.

        Args:
            agent_name: The agent name to query (default: current agent).

        Returns:
            JSON with capability scores and performance metrics.
        """
        try:
            from agent_utilities.mcp.kg_server import _get_engine

            engine = _get_engine()
            results = engine.query_cypher(
                "MATCH (a) WHERE a.type = 'AgentNode' "
                "AND (a.name = $name OR a.id = $name) "
                "OPTIONAL MATCH (a)-[:HAS_CAPABILITY]->(c) "
                "RETURN a, collect(c) AS capabilities LIMIT 1",
                {"name": agent_name},
            )
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps(
                {
                    "agent": agent_name,
                    "status": "no_self_model",
                    "message": "No self-model found. Run an evaluation cycle first.",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def agent_team_compose(task_description: str, max_agents: int = 5) -> str:
        """Suggest an optimal team composition for a task (AHE-3.3).

        Uses the Knowledge Graph's team synergy data to recommend
        the best agent coalition for a given task.

        Args:
            task_description: Natural language description of the task.
            max_agents: Maximum number of agents in the team.

        Returns:
            JSON with recommended agents and their roles.
        """
        try:
            from agent_utilities.mcp.kg_server import _get_engine

            engine = _get_engine()
            # Search for agents with relevant capabilities
            results = engine.search_hybrid(
                f"agent capability for: {task_description}",
                top_k=max_agents,
            )
            team = []
            for r in results:
                team.append(
                    {
                        "agent_id": r.get("id", ""),
                        "name": r.get("name", ""),
                        "relevance": r.get("score", 0.0),
                        "type": r.get("type", ""),
                    }
                )
            return json.dumps(
                {
                    "task": task_description[:200],
                    "team_size": len(team),
                    "recommended_team": team,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def agent_eval_history(agent_name: str = "default", limit: int = 10) -> str:
        """Query past evaluation scores and trends (AHE-3.1).

        Returns recent evaluation results for an agent, including
        scores, strategies tested, and improvement trends.

        Args:
            agent_name: Agent to query evaluations for.
            limit: Maximum evaluation records to return.

        Returns:
            JSON with evaluation history and trend summary.
        """
        try:
            from agent_utilities.mcp.kg_server import _get_engine

            engine = _get_engine()
            results = engine.query_cypher(
                "MATCH (e) WHERE e.type = 'EvaluationNode' "
                "AND (e.agent_name = $name OR e.agent_id = $name) "
                "RETURN e ORDER BY e.timestamp DESC LIMIT $limit",
                {"name": agent_name, "limit": limit},
            )
            return json.dumps(
                {
                    "agent": agent_name,
                    "evaluation_count": len(results),
                    "evaluations": results,
                },
                default=str,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def eval_reliability(
        output_text: str,
        input_text: str = "",
        context_json: str = "",
    ) -> str:
        """Run the reliability / guardrail evaluation suite (AHE-3.1).

        Scores an agent output with the bundled reliability scorers
        (faithfulness, safety, topic coverage, tool-necessity, deception,
        citation quality, Brier skill, retrieval recall, and content-injection).
        Supply scorer inputs (evidence, gold topics, retrieved ids, etc.) via
        ``context_json`` — a JSON object merged into every scorer's context.

        Args:
            output_text: The agent output to evaluate.
            input_text: Optional originating prompt/question.
            context_json: Optional JSON object of scorer context (evidence,
                gold, gold_topics, cited_ids, gold_evidence, outcome,
                forecast_prob, retrieved_ids, gold_ids, content, ...).

        Returns:
            JSON-serialised :class:`AggregatedEvalResult`.
        """
        try:
            from agent_utilities.harness.reliability_scorers import (
                build_reliability_suite,
            )

            context = json.loads(context_json) if context_json.strip() else {}
            if not isinstance(context, dict):
                return json.dumps({"error": "context_json must be a JSON object"})
            suite = build_reliability_suite()
            aggregated = suite.evaluate(input_text, output_text, context)
            return aggregated.model_dump_json()
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid context_json: {e}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def red_team_catalog() -> str:
        """List the agentic red-team attack probe catalog (AHE-3.1).

        Returns the static catalog of adversarial probes (prompt injection,
        jailbreak, role override, data exfiltration, sandbagging) with their
        techniques, severities, and OWASP-LLM mappings. Run each probe's
        ``probe_text`` against a target system, then score the response with
        ``red_team_assess``.

        Returns:
            JSON list of attack probes.
        """
        try:
            from agent_utilities.harness.red_team import ATTACK_CATALOG

            return json.dumps([p.model_dump() for p in ATTACK_CATALOG])
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def red_team_assess(probe_id: str, response_text: str) -> str:
        """Score a target's response to a red-team probe (AHE-3.1).

        Judges whether the response was compromised by the attack — via the
        probe's success markers plus the safety and deception reliability
        scorers — and returns a per-probe verdict with reasons.

        Args:
            probe_id: The probe id from ``red_team_catalog`` (e.g. ``pi-001``).
            response_text: The target system's response to the probe.

        Returns:
            JSON-serialised ProbeResult, or an error if the id is unknown.
        """
        try:
            from agent_utilities.harness.red_team import ATTACK_CATALOG, RedTeamRunner

            probe = next((p for p in ATTACK_CATALOG if p.id == probe_id), None)
            if probe is None:
                return json.dumps({"error": f"unknown probe_id: {probe_id}"})
            result = RedTeamRunner().assess(probe, response_text)
            return result.model_dump_json()
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def provenance_check(
        answer: str,
        sources_json: str = "",
        tool_values_json: str = "",
        attempt: int = 0,
    ) -> str:
        """Provenance-completeness pre-emit gate (AHE-3.13).

        Deterministically checks that an answer's numeric and substantive claims
        trace to supplied sources/tool values, returning accept / revise /
        escalate with the ungrounded items.

        Args:
            answer: The answer text to gate.
            sources_json: JSON array (or object) of valid citation source ids.
            tool_values_json: JSON array of values numbers may trace to.
            attempt: Revise attempts already taken (drives escalation).

        Returns:
            JSON-serialised ProvenanceVerdict.
        """
        try:
            from agent_utilities.harness.provenance_gate import ProvenanceCriticGate

            sources = json.loads(sources_json) if sources_json.strip() else []
            tool_values = (
                json.loads(tool_values_json) if tool_values_json.strip() else []
            )
            verdict = ProvenanceCriticGate().evaluate(
                answer, sources=sources, tool_values=tool_values, attempt=attempt
            )
            return verdict.model_dump_json()
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json arg: {e}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return args, mcp, middlewares


def main():
    """Entry point for the Harness MCP server."""
    args, mcp, middlewares = _build_server()

    for middleware in middlewares:
        mcp.add_middleware(middleware)

    logger.info(
        "Starting Harness MCP Server (transport=%s, port=%s)",
        args.transport,
        args.port,
    )

    if args.transport == "stdio":
        from agent_utilities.mcp.server_factory import protect_stdio_jsonrpc

        protect_stdio_jsonrpc()
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
