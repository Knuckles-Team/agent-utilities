"""Phase 4 — finance/legal/personal connectors (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

wger + mealie are standard (live once their flag is set); emerald + legal are
high_stakes / propose-only (never auto-execute — queued for approval).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import (
    emerald as em_ext,
    mealie as me_ext,
    wger as wg_ext,
)
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


# ── wger (standard, bidirectional) ───────────────────────────────────────────
class FakeWger:
    def __init__(self):
        self.weights = []
        self.sessions = []

    def get_weight_entries(self):
        return {"results": [{"id": 1, "date": "2026-06-14", "weight": 80.5}]}

    def get_measurements(self):
        return {
            "results": [{"id": 7, "category": 2, "date": "2026-06-14", "value": 40}]
        }

    def get_workout_sessions(self):
        return {
            "results": [
                {"id": 3, "date": "2026-06-14", "routine": 9, "impression": "4"}
            ]
        }

    def get_nutrition_plans(self):
        return {"results": [{"id": 5, "description": "cut"}]}

    def create_weight_entry(self, date, weight):
        self.weights.append((date, weight))

    def create_workout_session(self, routine, date, impression="3", notes=""):
        self.sessions.append((routine, date))


def test_wger_extract():
    batch = wg_ext.extract({"client": FakeWger()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["wger:weight:1"].type == "BodyMeasurement"
    assert by_id["wger:session:3"].type == "WorkoutSession"
    assert by_id["wger:nutplan:5"].type == "MealPlan"
    assert by_id["wger:weight:1"].props["domain"] == "wger"
    assert by_id["wger:weight:1"].props["externalToolId"] == "1"
    assert ("wger:session:3", "wger:routine:9", "PART_OF") in {
        (e.source, e.target, e.rel_type) for e in batch.edges
    }


def test_wger_sink_logs_when_enabled(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeWger()
    out = run_writeback(
        "wger",
        client=client,
        dry_run=False,
        creations=[
            {
                "type": "BodyMeasurement",
                "kind": "weight",
                "date": "2026-06-14",
                "value": 80.0,
            },
            {"type": "WorkoutSession", "routine": 9, "date": "2026-06-14"},
        ],
    )
    assert out["created"] == 2
    assert client.weights == [("2026-06-14", 80.0)]
    assert client.sessions == [(9, "2026-06-14")]


def test_wger_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "wger",
        client=FakeWger(),
        dry_run=False,
        creations=[{"type": "BodyMeasurement", "value": 1}],
    )
    assert out["status"] == "refused"


# ── mealie (standard, bidirectional) ─────────────────────────────────────────
class FakeMealie:
    def __init__(self):
        self.mealplans = []

    def get_recipes(self):
        return {"items": [{"id": "r1", "slug": "soup", "name": "Soup"}]}

    def get_households_mealplans(self):
        return {"items": [{"id": 11, "date": "2026-06-14", "recipeId": "r1"}]}

    def get_households_shopping_lists(self):
        return {"items": [{"id": "s1", "name": "Week"}]}

    def post_households_mealplans(self, data):
        self.mealplans.append(data)


def test_mealie_extract():
    batch = me_ext.extract({"client": FakeMealie()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["mealie:recipe:r1"].type == "Recipe"
    assert by_id["mealie:mealplan:11"].type == "MealPlan"
    assert by_id["mealie:shoplist:s1"].type == "ShoppingList"
    assert ("mealie:mealplan:11", "mealie:recipe:r1", "INCLUDES") in {
        (e.source, e.target, e.rel_type) for e in batch.edges
    }


def test_mealie_sink(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeMealie()
    out = run_writeback(
        "mealie",
        client=client,
        dry_run=False,
        creations=[{"type": "MealPlan", "date": "2026-06-14", "recipe_id": "r1"}],
    )
    assert out["created"] == 1
    assert client.mealplans[0]["recipeId"] == "r1"


# ── emerald (high_stakes, propose-only) ──────────────────────────────────────
class FakePosition:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = 100.0
        self.current_price = 110.0
        self.unrealized_pnl = 10.0
        self.side = "long"


class FakeAccount:
    equity = 1000.0
    cash = 500.0
    buying_power = 500.0
    currency = "USD"
    exchange = "paper"


class FakeExchange:
    def __init__(self):
        self.orders = []

    def get_account(self):
        return FakeAccount()

    def get_positions(self):
        return [FakePosition("BTC", 0.5)]

    def submit_order(self, symbol, side, qty, order_type=None, limit_price=None):
        self.orders.append((symbol, qty))
        return {"order_id": "o1"}


def test_emerald_extract():
    batch = em_ext.extract({"client": FakeExchange()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["emerald:portfolio:paper"].type == "Portfolio"
    assert by_id["emerald:account:paper"].type == "Account"
    pos = by_id["emerald:pos:paper:BTC"]
    assert pos.type == "Position" and pos.props["domain"] == "emerald"
    assert ("emerald:pos:paper:BTC", "emerald:portfolio:paper", "HELD_IN") in {
        (e.source, e.target, e.rel_type) for e in batch.edges
    }


def test_emerald_high_stakes_queues_never_executes(monkeypatch):
    """Enabled + live + unapproved → queued, client NEVER called."""
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeExchange()
    out = run_writeback(
        "emerald",
        client=client,
        dry_run=False,
        orders=[{"symbol": "BTC", "side": "buy", "qty": 0.1}],
    )
    assert out["status"] == "queued"
    assert out["proposal_id"]
    assert client.orders == []  # the safety spine: no auto-execute


def test_emerald_dry_run_proposes(monkeypatch):
    out = run_writeback(
        "emerald",
        client=FakeExchange(),
        dry_run=True,
        orders=[{"symbol": "BTC", "side": "buy", "qty": 0.1}],
    )
    assert out["proposals"][0]["op"] == "submit_order"


def test_emerald_approved_replay_executes(monkeypatch):
    """An approved replay (_approved) DOES submit the order live."""
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeExchange()
    out = run_writeback(
        "emerald",
        client=client,
        dry_run=False,
        _approved=True,
        orders=[{"symbol": "BTC", "side": "buy", "qty": 0.1}],
    )
    assert out["status"] == "completed"
    assert out["created"] == 1
    assert client.orders == [("BTC", 0.1)]


# ── legal (high_stakes, propose-only) ────────────────────────────────────────
class FakeLegal:
    def __init__(self):
        self.filed = []

    def draft_ein_form(self, **kw):
        self.filed.append(kw)
        return "drafted"


def test_legal_high_stakes_queues(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeLegal()
    out = run_writeback(
        "legal",
        client=client,
        dry_run=False,
        filings=[{"type": "EINApplication", "legal_name": "Acme LLC"}],
    )
    assert out["status"] == "queued"
    assert client.filed == []


def test_legal_dry_run_proposes():
    out = run_writeback(
        "legal",
        client=FakeLegal(),
        dry_run=True,
        filings=[{"type": "EINApplication", "legal_name": "Acme LLC"}],
    )
    assert out["proposals"][0]["op"] == "draft_ein"
