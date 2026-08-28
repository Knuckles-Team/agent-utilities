"""Typed request/context value objects for the ``graph_bus`` action dispatcher.

Leaf module — imports NOTHING from ``agent_utilities`` (CX wD10-R-ARITY / BUG-CX-004 §9b).

``agent_utilities.mcp.tools`` is the single highest-leverage node in au's import SCC: its
``__init__.py`` eagerly imports all 35 tool-registration modules, so importing anything
*under* ``mcp.tools`` pulls the whole 405-edge cluster with it. These dataclasses are pure
data — no engine, no bus, no KG access — so any future caller (a REST handler, a test, another
tool) that needs to build or read a ``BusRequest`` can import this module alone without paying
that cost. ``agent_utilities/mcp/tools/bus_tools.py`` (the only current caller) imports and
re-exports these names so no import site has to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# NOTE: `fastmcp.Context` is a third-party type, not an `agent_utilities` module — importing
# it here does not add an edge into the SCC this file exists to stay out of.
from fastmcp import Context


@dataclass(frozen=True, slots=True)
class BusRequest:
    """One ``graph_bus`` call, typed. Internal only — no wire contract of its own.

    Mirrors ``graph_bus``'s parameter union exactly (it is built directly from those
    arguments), but every action handler only reads the 2-5 fields its own action actually
    uses instead of receiving all twenty as positional/keyword arguments.
    """

    action: str
    agent_id: str = ""
    sender: str = ""
    to: str = ""
    topic: str = ""
    payload: str = ""
    objective: str = ""
    kind: str = "develop"
    priority: str = "normal"
    provider: str = ""
    host: str = ""
    capabilities: str = ""
    session_id: str = ""
    since: int = 0
    online_only: bool = False
    reason: str = ""
    url: str = ""
    group: str = ""
    origin: str = ""
    scope: str = "commons"
    ctx: Context | None = None


@dataclass(frozen=True, slots=True)
class BusExecContext:
    """The two collaborators every action handler needs, bundled so handlers stay 2-arg."""

    bus: Any
    engine: Any
