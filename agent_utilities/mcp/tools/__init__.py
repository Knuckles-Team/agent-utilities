"""Grouped MCP tool registration modules for the graph-os server.

Each exposes register_<group>(mcp), called from kg_server._build_server —
the strangler split of the former ~5k-line monolithic builder.
"""

from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
from agent_utilities.mcp.tools.ontology_tools import register_ontology_tools
from agent_utilities.mcp.tools.query_tools import register_query_tools
from agent_utilities.mcp.tools.reach_tools import register_reach_tools
from agent_utilities.mcp.tools.state_tools import register_state_tools
from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools

__all__ = [
    "register_query_tools",
    "register_write_ingest_tools",
    "register_analysis_tools",
    "register_state_tools",
    "register_ontology_tools",
    "register_reach_tools",
]
