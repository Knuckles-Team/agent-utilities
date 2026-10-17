"""agent-utilities: the in-run agent harness.

The harness holds only the mechanism inside one agent run: applying edits,
checking tool contracts and constraints, pre-emit gates, the developer
workspace action/observation loop, the agent run loop
(``agent_utilities.graph``, ``agent_utilities.rlm``) and the ``host_ports`` protocols through
which the host (arbora) supplies tools, credentials, models and trace sinks.

Facts, storage, retrieval, decisions and learning live in epistemic-graph;
serving, admission and the registry live in arbora; connector contracts and
transports live in agent-connector-sdk.
"""

from typing import Any

from agent_utilities._version import __version__

_GRAPH_EXPORTS = frozenset(
    {"create_graph_agent", "execute_graph", "initialize_graph_from_workspace"}
)


def __getattr__(name: str) -> Any:
    """Load the run-loop entry points on first use (keeps ``import`` light)."""
    if name in _GRAPH_EXPORTS:
        from agent_utilities import graph

        return getattr(graph, name)
    raise AttributeError(f"module 'agent_utilities' has no attribute {name!r}")


__all__ = ["__version__", *sorted(_GRAPH_EXPORTS)]
