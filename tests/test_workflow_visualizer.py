"""Unit tests for the WorkflowVisualizer component.

CONCEPT:ORCH-1.8 — Parallel Engine Visualizer
"""

from __future__ import annotations

import os

from agent_utilities.models.execution_manifest import AgentSpec, ExecutionManifest
from agent_utilities.workflows.visualizer import WorkflowVisualizer

os.environ["OTEL_SDK_DISABLED"] = "true"


def test_workflow_visualizer_generation():
    """Verify that WorkflowVisualizer generates a beautifully themed Mermaid diagram."""
    manifest = ExecutionManifest(
        name="Test Swarm Flow",
        agents=[],
        query="Run analysis",
    )

    wave1 = [
        AgentSpec(
            agent_id="data-collector",
            role="data-gatherer",
            tools=["mcp_portainer_docker", "container_manager_list_containers"],
        ),
        AgentSpec(
            agent_id="network-auditor",
            role="network-inspector",
            tools=["tunnel_manager_list_tunnels"],
        ),
    ]
    wave2 = [
        AgentSpec(
            agent_id="sys-synth",
            role="systems-coordinator",
            depends_on=["data-collector", "network-auditor"],
        )
    ]

    waves = [wave1, wave2]

    mermaid_code = WorkflowVisualizer.generate(manifest, waves)

    assert "flowchart TD" in mermaid_code
    assert "subgraph Wave 1 &#40;Parallel Layer&#41;" in mermaid_code
    assert "subgraph Wave 2 &#40;Parallel Layer&#41;" in mermaid_code

    # Assert node sanitization and formatting
    assert "data_collector" in mermaid_code
    assert "sys_synth" in mermaid_code

    # Assert dependency edges
    assert "data_collector --> sys_synth" in mermaid_code
    assert "network_auditor --> sys_synth" in mermaid_code

    # Assert class highlights are added
    assert "classDef success" in mermaid_code
    assert "classDef active" in mermaid_code
