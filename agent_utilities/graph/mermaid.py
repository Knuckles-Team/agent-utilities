#!/usr/bin/python
"""Mermaid diagram generation for graph visualization"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_graph_mermaid(
    graph, config: dict, title: str = "Graph", routed_domain: str | None = None
) -> str:
    """Generate a Mermaid diagram for the graph.

    Args:
        graph: The Graph object.
        config: The config dict from create_graph_agent().
        title: Optional title for the diagram.
        routed_domain: Optional domain tag that was routed to.

    Returns:
        Mermaid diagram string.
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
