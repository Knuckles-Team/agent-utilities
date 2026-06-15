"""Mealie source extractor (CONCEPT:KG-2.9).

Reads recipes/meal-plans/shopping-lists from a Mealie instance into canonical
wellness OWL nodes: recipes → :Recipe, meal plans → :MealPlan, shopping lists →
:ShoppingList. Stamped externalToolId + domain="mealie". Client
(``mealie_mcp.auth.get_client()``) injected; tolerant of Mealie's
``{"items": [...]}`` pagination shape (and bare lists / ``results``).
"""

from __future__ import annotations

from typing import Any

from ...core import owl_bridge
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "mealie"
_DOMAIN = "mealie"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _rows(res: Any) -> list[dict]:
    """Mealie ``{"items": [...]}`` / ``{"results": [...]}`` / bare list → list[dict]."""
    if isinstance(res, dict):
        res = res.get("items", res.get("results", res.get("data", [])))
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def _call(client: Any, name: str) -> Any:
    m = getattr(client, name, None)
    try:
        return m() if callable(m) else None
    except Exception:  # noqa: BLE001 - tolerant of unconfigured endpoints
        return None


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for r in _rows(_call(client, "get_recipes")):
        rid = r.get("id") or r.get("slug")
        if not rid:
            continue
        nodes.append(
            GraphNode(
                id=f"mealie:recipe:{rid}",
                type="Recipe",
                props={
                    "name": r.get("name"),
                    "slug": r.get("slug"),
                    "description": r.get("description"),
                    "externalToolId": str(rid),
                    "domain": _DOMAIN,
                },
            )
        )
    for mp in _rows(_call(client, "get_households_mealplans")):
        mid = mp.get("id")
        if mid is None:
            continue
        node_id = f"mealie:mealplan:{mid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="MealPlan",
                props={
                    "date": mp.get("date"),
                    "entry_type": mp.get("entryType"),
                    "title": mp.get("title"),
                    "externalToolId": str(mid),
                    "domain": _DOMAIN,
                },
            )
        )
        recipe = mp.get("recipeId") or (mp.get("recipe") or {}).get("id")
        if recipe:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"mealie:recipe:{recipe}",
                    rel_type="INCLUDES",
                )
            )
    for sl in _rows(_call(client, "get_households_shopping_lists")):
        sid = sl.get("id")
        if sid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"mealie:shoplist:{sid}",
                type="ShoppingList",
                props={
                    "name": sl.get("name"),
                    "externalToolId": str(sid),
                    "domain": _DOMAIN,
                },
            )
        )

    owl_bridge.register_promotable_node_types({n.type for n in nodes})
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Mealie (recipes/meal-plans/shopping) → KG"
)
