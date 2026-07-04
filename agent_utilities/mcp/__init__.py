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
