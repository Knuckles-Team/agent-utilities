"""Focused graph-os evolution and enterprise-knowledge improvement operations."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_evolution_tools(mcp: Any) -> None:
    """Register on-demand evolution operations that share the daemon's native cores."""

    @mcp.tool(
        name="graph_evolution",
        description=(
            "Improve the graph and its executable knowledge. Actions: 'assimilate', "
            "'distill_skills', 'standardize', 'failure_ingest', 'optimize_component', "
            "and 'publish_proposal'. All proposal-producing actions remain review-gated."
        ),
        tags=["graph-os", "evolution", "optimization", "skills"],
    )
    def graph_evolution(
        action: str = Field(
            default="assimilate",
            description=(
                "assimilate | distill_skills | standardize | failure_ingest | "
                "optimize_component | publish_proposal"
            ),
        ),
        target: str = Field(
            default="",
            description=(
                "Proposal id, optimization component, or distillation proposal id. "
                "Use all/sweep for a full optimization sweep."
            ),
        ),
        data_json: str = Field(
            default="{}", description="JSON input data for component optimization."
        ),
        limit: int = Field(
            default=20, description="Bounded result/materialization limit."
        ),
        synthesize: bool = Field(
            default=False,
            description="Synthesize synergy proposals during assimilation.",
        ),
        force: bool = Field(
            default=False,
            description="Re-run assimilation even when its watermark is unchanged.",
        ),
        draft: bool = Field(
            default=False,
            description="Render reviewable skill drafts while distilling.",
        ),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action == "assimilate":
                from agent_utilities.knowledge_graph.research.loop_controller import (
                    run_assimilation_pass,
                )

                report = run_assimilation_pass(
                    engine,
                    synthesize=synthesize,
                    top_n=max(1, int(limit)),
                    force=force,
                )
                return json.dumps(report, indent=2, default=str)

            if action == "distill_skills":
                from agent_utilities.knowledge_graph.distillation.skill_synthesizer import (
                    ConnectorSkillDistiller,
                )

                distiller = ConnectorSkillDistiller(engine)
                if target:
                    return json.dumps(
                        distiller.materialize(target), indent=2, default=str
                    )
                report = distiller.run(draft=draft)
                return json.dumps(report.to_dict(), indent=2, default=str)

            if action == "standardize":
                from agent_utilities.knowledge_graph.standardization import (
                    run_standardization_pass,
                )

                return json.dumps(
                    run_standardization_pass(engine, top_n=max(1, int(limit))),
                    indent=2,
                    default=str,
                )

            if action == "failure_ingest":
                from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
                    run_failure_ingest,
                )

                return json.dumps(run_failure_ingest(engine), indent=2, default=str)

            if action == "optimize_component":
                from agent_utilities.harness.program_optimization import (
                    run_component_optimization,
                    run_optimization_sweep,
                )

                component = target.strip().lower()
                if component in {"", "all", "sweep"}:
                    report = run_optimization_sweep(engine)
                else:
                    data = json.loads(data_json) if data_json else {}
                    if not isinstance(data, dict):
                        raise ValueError("data_json must decode to an object")
                    report = run_component_optimization(component, data, engine=engine)
                return json.dumps(report, indent=2, default=str)

            if action == "publish_proposal":
                if not target:
                    raise ValueError("target proposal id is required")
                from agent_utilities.knowledge_graph.research.change_publisher import (
                    publish_proposal,
                )

                return json.dumps(
                    publish_proposal(engine, target), indent=2, default=str
                )

            return f"Error: Unknown graph_evolution action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_evolution"] = graph_evolution
    kg_server.ACTION_TOOL_ROUTES["graph_evolution"] = "/graph/evolution"
