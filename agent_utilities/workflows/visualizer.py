"""Workflow Visualizer — Programmatically generate beautiful Mermaid diagrams of parallel workflows.

CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine Visualizer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_utilities.observability.mermaid import FlowchartBuilder, MermaidTheme

if TYPE_CHECKING:
    from agent_utilities.models.execution_manifest import AgentSpec, ExecutionManifest

logger = logging.getLogger(__name__)


class WorkflowVisualizer:
    """Deterministic visualizer for parallel agent workflows.

    Generates premium themed Mermaid flowcharts showing wave groupings
    and execution dependencies.
    """

    @staticmethod
    def generate(manifest: ExecutionManifest, waves: list[list[AgentSpec]]) -> str:
        """Generate a Mermaid flowchart representation of the scheduled waves."""
        title = manifest.name or manifest.manifest_id or "Workflow Execution Flow"
        builder = FlowchartBuilder(title=title, direction="TD", theme=MermaidTheme.DARK)

        # 1. Map all agents to their waves and add wave subgraphs
        all_agent_ids = {a.agent_id for wave in waves for a in wave}

        for wave_idx, wave_agents in enumerate(waves):
            wave_nodes = []
            for agent in wave_agents:
                # Sanitize node ID for Mermaid
                node_id = agent.agent_id

                # Determine shape based on role
                shape = "round"
                if (
                    "synth" in node_id.lower()
                    or "fusion" in node_id.lower()
                    or "gate" in node_id.lower()
                ):
                    shape = "cylinder"
                elif agent.partitions:
                    shape = "diamond"

                # Create a description label
                label = f"{agent.agent_id}"
                if agent.role:
                    label += f"\\n(Role: {agent.role})"
                if agent.tools:
                    # Show up to 3 tools
                    tools_str = ", ".join(agent.tools[:3])
                    if len(agent.tools) > 3:
                        tools_str += "..."
                    label += f"\\n[Tools: {tools_str}]"

                # Choose CSS class for theme highlighting
                css_class = "active"
                if (
                    "synth" in node_id.lower()
                    or "fusion" in node_id.lower()
                    or "gate" in node_id.lower()
                ):
                    css_class = "success"
                elif agent.partitions:
                    css_class = "warning"

                builder.add_node(
                    node_id=node_id,
                    label=label,
                    shape=shape,
                    css_class=css_class,
                )

                # We need to use the sanitized version matching FlowchartBuilder
                safe_id = node_id.replace("-", "_").replace(":", "_").replace(".", "_")
                wave_nodes.append(safe_id)

            # Group into subgraph per wave
            builder.add_subgraph(
                title=f"Wave {wave_idx + 1} (Parallel Layer)",
                nodes=[
                    node.replace("_", "-") for node in wave_nodes
                ],  # builder will sanitize again
                direction="LR",
            )

        # 2. Add edges respecting dependencies
        for wave in waves:
            for agent in wave:
                for dep in agent.depends_on:
                    # Only add dependency if the dependent agent is part of this execution
                    if dep in all_agent_ids:
                        builder.add_edge(source=dep, target=agent.agent_id)

        # 3. Add custom styles matching premium dark mode theme
        builder.lines.append(
            "  classDef success fill:#1b5e20,stroke:#2e7d32,color:#ffffff,stroke-width:2px;"
        )
        builder.lines.append(
            "  classDef active fill:#0d47a1,stroke:#1565c0,color:#ffffff,stroke-width:2px;"
        )
        builder.lines.append(
            "  classDef warning fill:#e65100,stroke:#f57c00,color:#ffffff,stroke-width:2px;"
        )

        return builder.render()
