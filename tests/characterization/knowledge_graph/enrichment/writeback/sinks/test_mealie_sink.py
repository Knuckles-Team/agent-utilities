"""Characterization tests for MealieSink.run (CX-AU-06, pre-refactor CCN 15).

Pins OBSERVED behaviour: no-client live skip, the three ``creations`` type
branches (mealplan/shoppinglist/shoppingitem) each in dry-run and live form,
the unconditional ``else: result.skipped += 1`` for an unrecognised type (it
fires in BOTH dry-run and live mode -- no proposal is ever generated for an
unknown type), and live success/exception counting. No behaviour changed.

"No client" scenarios monkeypatch ``MealieSink._client`` directly for
environment independence (see the home_assistant/keycloak characterization
tests for the same pattern and the reason).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.mealie import MealieSink


class _FakeClient:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self.mealplans: list[dict] = []
        self.shopping_lists: list[dict] = []
        self.shopping_items: list[dict] = []
        self._raise_on = raise_on or set()

    def post_households_mealplans(self, *, data: dict) -> None:
        if "mealplan" in self._raise_on:
            raise RuntimeError("boom")
        self.mealplans.append(data)

    def post_households_shopping_lists(self, *, data: dict) -> None:
        if "shoppinglist" in self._raise_on:
            raise RuntimeError("boom")
        self.shopping_lists.append(data)

    def post_households_shopping_items(self, *, data: dict) -> None:
        if "shoppingitem" in self._raise_on:
            raise RuntimeError("boom")
        self.shopping_items.append(data)


def _fields(result: Any) -> tuple:
    return (result.created, result.errors, result.skipped, result.proposals)


def _sink_with_no_client(monkeypatch: pytest.MonkeyPatch) -> MealieSink:
    sink = MealieSink()
    monkeypatch.setattr(sink, "_client", lambda ops: None)
    return sink


def test_no_client_live_mode_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{"type": "mealplan"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 1, [])


def test_dry_run_mealplan_minimal() -> None:
    sink = MealieSink()
    ops: dict[str, Any] = {"creations": [{"type": "mealplan", "date": "2026-01-01"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "post_mealplan", "date": "2026-01-01", "entryType": "dinner"}
    ]


def test_dry_run_mealplan_with_recipe_and_title() -> None:
    sink = MealieSink()
    ops: dict[str, Any] = {
        "creations": [
            {
                "type": "mealplan",
                "date": "2026-01-01",
                "entry_type": "lunch",
                "recipe_id": "r1",
                "title": "Tacos",
            }
        ]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {
            "op": "post_mealplan",
            "date": "2026-01-01",
            "entryType": "lunch",
            "recipeId": "r1",
            "title": "Tacos",
        }
    ]


def test_dry_run_shoppinglist_default_name() -> None:
    sink = MealieSink()
    ops: dict[str, Any] = {"creations": [{"type": "shoppinglist"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "post_shopping_list", "name": "KG shopping list"}
    ]


def test_dry_run_shoppingitem() -> None:
    sink = MealieSink()
    ops: dict[str, Any] = {
        "creations": [{"type": "shoppingitem", "name": "Eggs", "list_id": "L1"}]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "post_shopping_item", "note": "Eggs", "shoppingListId": "L1"}
    ]


def test_unknown_type_skipped_even_in_dry_run() -> None:
    """OBSERVED: the else-branch skip fires unconditionally -- no proposal."""
    sink = MealieSink()
    ops: dict[str, Any] = {"creations": [{"type": "recipe"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 1, [])


def test_missing_type_defaults_to_empty_string_and_is_skipped() -> None:
    sink = MealieSink()
    result = sink.run(WritebackContext(), {"creations": [{}]}, dry_run=True)
    assert _fields(result) == (0, 0, 1, [])


def test_live_mealplan_success() -> None:
    client = _FakeClient()
    sink = MealieSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"type": "MealPlan", "date": "2026-01-01"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.mealplans) == (
        1,
        0,
        [{"date": "2026-01-01", "entryType": "dinner"}],
    )


def test_live_shoppinglist_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"shoppinglist"})
    sink = MealieSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"type": "shoppinglist"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_live_shoppingitem_success() -> None:
    client = _FakeClient()
    sink = MealieSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"type": "shoppingitem", "name": "Milk", "list_id": "L9"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.shopping_items) == (
        1,
        0,
        [{"note": "Milk", "shoppingListId": "L9"}],
    )


def test_multiple_creations_mixed_outcomes() -> None:
    client = _FakeClient(raise_on={"mealplan"})
    sink = MealieSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {"type": "mealplan", "date": "2026-01-01"},
            {"type": "shoppinglist"},
            {"type": "unknown"},
            {"type": "shoppingitem", "name": "Bread"},
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, result.skipped) == (2, 1, 1)
    assert client.shopping_lists == [{"name": "KG shopping list"}]
    assert client.shopping_items == [{"note": "Bread", "shoppingListId": None}]


def test_no_creations_key_returns_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])
