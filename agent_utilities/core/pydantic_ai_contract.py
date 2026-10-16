"""The one Pydantic-AI release the agent harness is built against.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

This literal is the single contract source the fleet parity checker
(``scripts/check_pydantic_ai_fleet_contract.py``) reads by AST: package
extras, locks and image inputs must agree with it. The harness carries no
protocol client of its own; graph-os builds the toolsets it hands in.
"""

from __future__ import annotations

_PYDANTIC_AI_CONTRACT_VERSION = "2.54.0"

__all__ = ["_PYDANTIC_AI_CONTRACT_VERSION"]
