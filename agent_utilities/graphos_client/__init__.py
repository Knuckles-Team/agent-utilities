"""Thin client of graph-os's served operations for the agent harness.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

agent-utilities is only the agent harness. graph-os owns every served
operation (MCP serving, the tool registry, the REST/A2A gateway). When a run
needs to call tools it does so as a CLIENT: it builds an MCP client toolset for
a remote endpoint (:mod:`.toolset_factory`), attaches outbound credentials
(:mod:`.client_credentials`, :mod:`.delegated_auth`), or binds tools that the
hosting graph-os process registered with :mod:`.host`. Nothing here serves,
registers or discloses a tool surface.
"""
