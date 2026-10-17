"""agent-utilities: the in-run agent harness.

The harness holds only the mechanism inside one agent run: applying edits,
checking tool contracts and constraints, pre-emit gates, the developer
workspace action/observation loop and the ``host_ports`` protocols through
which the host (graph-os) supplies tools, credentials, models and trace sinks.

Facts, storage, retrieval, decisions and learning live in epistemic-graph;
serving, admission and the registry live in graph-os; connector contracts and
transports live in agent-connector-sdk.
"""

from agent_utilities._version import __version__

__all__ = ["__version__"]
