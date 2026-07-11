"""Ontology Stardog catalog is reachable on BOTH surfaces (Two-surfaces contract).

CONCEPT:AU-KG.ontology.stardog-catalog-overwrite / stardog-catalog-import — the overwrite-push and
import-back operations must be exposed as a ``graph_ontology`` MCP action AND a REST twin, so a
feature present on one surface and missing on the other is a build break.
"""

from __future__ import annotations


def _graph_ontology_actions() -> set[str]:
    import agent_utilities.mcp._graphos_action_manifest as m

    actions: set[str] = set()
    for v in vars(m).values():
        if isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and e.get("tool") == "graph_ontology":
                    a = e.get("action")
                    if a:
                        actions.add(a)
    return actions


def test_stardog_catalog_actions_are_registered_on_mcp():
    actions = _graph_ontology_actions()
    assert {"publish_stardog", "import_stardog"} <= actions


def test_stardog_catalog_rest_twins_exist():
    import agent_utilities.mcp.kg_server as k

    assert callable(k.graph_ontology_publish_stardog_endpoint)
    assert callable(k.graph_ontology_import_stardog_endpoint)
