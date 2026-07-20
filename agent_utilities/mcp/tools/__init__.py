"""Grouped MCP tool registration modules for the graph-os server.

Each exposes register_<group>(mcp), called from kg_server._build_server —
the strangler split of the former ~5k-line monolithic builder.
"""

from agent_utilities.mcp.tools.agent_execution_tools import (
    register_agent_execution_tools,
)
from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
from agent_utilities.mcp.tools.analyze_suite import register_analyze_suite_tools
from agent_utilities.mcp.tools.argument_tools import register_argument_tools
from agent_utilities.mcp.tools.audit_tools import register_audit_tools
from agent_utilities.mcp.tools.bus_tools import register_bus_tools
from agent_utilities.mcp.tools.compliance_tools import register_compliance_tools
from agent_utilities.mcp.tools.domain_ops_tools import register_domain_ops_tools
from agent_utilities.mcp.tools.engine_surface_tools import (
    register_engine_surface_tools,
)
from agent_utilities.mcp.tools.engine_tools import register_engine_tools
from agent_utilities.mcp.tools.epistemic_tools import register_epistemic_tools
from agent_utilities.mcp.tools.evolution_tools import register_evolution_tools
from agent_utilities.mcp.tools.governance_tools import register_governance_tools
from agent_utilities.mcp.tools.graph_engineering_tools import (
    register_graph_engineering_tools,
)
from agent_utilities.mcp.tools.incident_tools import register_incident_tools
from agent_utilities.mcp.tools.job_tools import register_job_tools
from agent_utilities.mcp.tools.ontology_tools import register_ontology_tools
from agent_utilities.mcp.tools.ops_causal_tools import register_ops_causal_tools
from agent_utilities.mcp.tools.query_tools import register_query_tools
from agent_utilities.mcp.tools.reach_tools import register_reach_tools
from agent_utilities.mcp.tools.rlm_tools import register_rlm_tools
from agent_utilities.mcp.tools.secret_tools import register_secret_tools
from agent_utilities.mcp.tools.state_tools import register_state_tools
from agent_utilities.mcp.tools.workflow_tools import register_workflow_tools
from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools

__all__ = [
    "register_query_tools",
    "register_write_ingest_tools",
    "register_analysis_tools",
    "register_agent_execution_tools",
    "register_analyze_suite_tools",
    "register_state_tools",
    "register_ontology_tools",
    "register_reach_tools",
    "register_bus_tools",
    "register_secret_tools",
    "register_engine_tools",
    "register_domain_ops_tools",
    "register_engine_surface_tools",
    "register_evolution_tools",
    "register_governance_tools",
    "register_ops_causal_tools",
    "register_audit_tools",
    "register_epistemic_tools",
    "register_incident_tools",
    "register_job_tools",
    "register_compliance_tools",
    "register_rlm_tools",
    "register_workflow_tools",
    "register_argument_tools",
    "register_graph_engineering_tools",
]
