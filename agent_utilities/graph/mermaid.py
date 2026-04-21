#!/usr/bin/python
"""Mermaid Visualization Module.

This module provides utilities for generating Mermaid diagrams from
pydantic-graph objects, allowing for visual representation of the
agentic workflow and state transitions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_graph_mermaid(
    graph, config: dict, title: str = "Graph", routed_domain: str | None = None
) -> str:
    """Generate a Mermaid diagram representation for the given graph.

    Args:
        graph: The Graph instance to visualize.
        config: Configuration dictionary for the graph agent.
        title: Title for the generated diagram.
        routed_domain: The domain tag that was activated (for labeling).

    Returns:
        A string containing the Mermaid flow diagram code.

    """
    if hasattr(graph, "mermaid_code"):
        mermaid = graph.mermaid_code()
    else:
        mermaid = graph.render()

    if title:
        if "---" in mermaid:
            import re

            mermaid = re.sub(r"title: .*", f"title: {title}", mermaid)
        else:
            mermaid = f"---\ntitle: {title}\n---\n{mermaid}"

    router_model = config.get("router_model") or "Master Router"
    if ":" in router_model:
        router_model = router_model.split(":")[-1]

    router_label = f"Router ({router_model})"
    domain_label = f"Domain Node ({routed_domain})" if routed_domain else "Domain Node"

    if "router" in mermaid:
        mermaid += f"\n  router : {router_label}"
    if "domain_execution" in mermaid:
        mermaid += f"\n  domain_execution : {domain_label}"

    return mermaid
