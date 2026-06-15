"""Mealie write-back sink (CONCEPT:KG-2.9).

Writes KG-derived meal plans and shopping lists back into Mealie (the planning
write twin of the Mealie extractor; fuses with wger for the wellness loop).
Standard tier, fail-closed (``MEALIE_ENABLE_WRITE``), dry-run-first.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


class MealieSink:
    domain = "mealie"
    enable_flag = "MEALIE_ENABLE_WRITE"
    risk_tier = "standard"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from mealie_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("mealie write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        for c in ops.get("creations") or []:
            ctype = (c.get("type") or "").lower()
            try:
                if ctype == "mealplan":
                    data = {
                        "date": c.get("date"),
                        "entryType": c.get("entry_type", "dinner"),
                    }
                    if c.get("recipe_id"):
                        data["recipeId"] = c["recipe_id"]
                    if c.get("title"):
                        data["title"] = c["title"]
                    if dry_run:
                        result.proposals.append({"op": "post_mealplan", **data})
                    else:
                        client.post_households_mealplans(data=data)
                        result.created += 1
                elif ctype == "shoppinglist":
                    data = {"name": c.get("name", "KG shopping list")}
                    if dry_run:
                        result.proposals.append({"op": "post_shopping_list", **data})
                    else:
                        client.post_households_shopping_lists(data=data)
                        result.created += 1
                elif ctype == "shoppingitem":
                    data = {"note": c.get("name"), "shoppingListId": c.get("list_id")}
                    if dry_run:
                        result.proposals.append({"op": "post_shopping_item", **data})
                    else:
                        client.post_households_shopping_items(data=data)
                        result.created += 1
                else:
                    result.skipped += 1
            except Exception:  # noqa: BLE001
                logger.debug("mealie write failed for %s", ctype, exc_info=True)
                result.errors += 1

        return result


register_sink(MealieSink())
