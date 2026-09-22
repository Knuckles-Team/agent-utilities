"""Public session authority for AU application integrations.

Authentication middleware remains responsible for minting and binding a
verified session. This module exposes that one AU-owned session contract to
application consumers without making them depend on the knowledge-graph
implementation package.
"""

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    SessionExpiredError,
    SessionRequiredError,
    current_session,
    resolve_session,
    use_session,
)

__all__ = [
    "GraphSession",
    "ScopeError",
    "SessionExpiredError",
    "SessionRequiredError",
    "current_session",
    "resolve_session",
    "use_session",
]
