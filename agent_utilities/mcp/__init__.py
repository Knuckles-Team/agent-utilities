"""MCP (Model Context Protocol) subsystem for agent-utilities.

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces

This package provides:
- KG Server (graph-os) — Knowledge Graph MCP server exposing CRUD + search tools
- KG Coordinator — Centralized coordination for multi-backend KG operations
- MCP Multiplexer — Aggregates multiple child MCP servers into a single
  unified stdio endpoint with namespaced tool routing
- Config Loader — MCP configuration discovery and environment expansion
- Agent Manager — Agent lifecycle management over MCP
"""

from agent_utilities.mcp.oauth_log_hygiene import install_oauth_log_hygiene

# U-54: attach OAuth SDK log redaction as soon as anything under this package
# is imported — before any MCP transport (and therefore before any OAuth
# logger below could emit a record). Name-only (no mcp/fastmcp import), so
# this is unconditionally safe even when the `[mcp]` extra isn't installed.
install_oauth_log_hygiene()
