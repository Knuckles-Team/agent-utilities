"""Tests for CONCEPT:KG-2.63 — Versioned Order System."""

import pytest

from agent_utilities.domains.finance.versioned_orders import (
    OrderCommit,
    OrderHistory,
    OrderStage,
    OrderStatus,
    PreCommitGuard,
)


class TestOrderStage:
    def test_creation(self):
        stage = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
            price=150.0,
        )
        assert stage.order_id == "ord:001"
        assert stage.status == OrderStatus.STAGED
        assert stage.version == 1

    def test_immutability(self):
        stage = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        with pytest.raises(AttributeError):
            stage.quantity = 200  # type: ignore

    def test_content_hash_deterministic(self):
        s1 = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        s2 = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        assert s1.content_hash == s2.content_hash

    def test_content_hash_changes_with_version(self):
        s1 = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        s2 = OrderStage(
            order_id="ord:001",
            version=2,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        assert s1.content_hash != s2.content_hash


class TestPreCommitGuard:
    def test_no_guards(self):
        guard = PreCommitGuard()
        stage = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        passed, messages = guard.run_all(stage)
        assert passed is True

    def test_passing_guard(self):
        guard = PreCommitGuard()
        guard.register("size_check", lambda s: (True, "Size OK"))
        stage = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        passed, messages = guard.run_all(stage)
        assert passed is True
        assert any("PASS" in m for m in messages)

    def test_failing_guard(self):
        guard = PreCommitGuard()
        guard.register(
            "max_quantity", lambda s: (s.quantity <= 50, f"Qty {s.quantity} exceeds 50")
        )
        stage = OrderStage(
            order_id="ord:001",
            version=1,
            side="buy",
            instrument_id="AAPL",
            quantity=100,
        )
        passed, messages = guard.run_all(stage)
        assert passed is False
        assert any("FAIL" in m for m in messages)


class TestOrderHistory:
    def test_stage_creates_version(self):
        history = OrderHistory()
        stage = history.stage("ord:001", "buy", "AAPL", 100, price=150.0)
        assert stage.version == 1
        assert stage.status == OrderStatus.STAGED

    def test_multiple_versions(self):
        history = OrderHistory()
        history.stage("ord:001", "buy", "AAPL", 100)
        history.stage("ord:001", "buy", "AAPL", 150)  # Modify quantity
        stages = history.get_history("ord:001")
        assert len(stages) == 2
        assert stages[0].version == 1
        assert stages[1].version == 2
        assert stages[1].parent_hash == stages[0].content_hash

    def test_commit_success(self):
        history = OrderHistory()
        history.stage("ord:001", "buy", "AAPL", 100)
        commit = history.commit("ord:001")
        assert commit is not None
        assert commit.order_stage.status == OrderStatus.COMMITTED

    def test_commit_with_passing_guard(self):
        history = OrderHistory()
        history.register_guard("always_pass", lambda s: (True, "OK"))
        history.stage("ord:001", "buy", "AAPL", 100)
        commit = history.commit("ord:001")
        assert commit is not None

    def test_commit_with_failing_guard(self):
        history = OrderHistory()
        history.register_guard("block_all", lambda s: (False, "Blocked"))
        history.stage("ord:001", "buy", "AAPL", 100)
        commit = history.commit("ord:001")
        assert commit is None
        latest = history.get_latest("ord:001")
        assert latest and latest.status == OrderStatus.REJECTED

    def test_commit_nonexistent_order(self):
        history = OrderHistory()
        assert history.commit("nonexistent") is None

    def test_commit_log(self):
        history = OrderHistory()
        history.stage("ord:001", "buy", "AAPL", 100)
        history.stage("ord:002", "sell", "MSFT", 50)
        history.commit("ord:001")
        history.commit("ord:002")
        log = history.get_commit_log()
        assert len(log) == 2
        assert log[1].previous_hash == log[0].commit_hash

    def test_pending_count(self):
        history = OrderHistory()
        history.stage("ord:001", "buy", "AAPL", 100)
        history.stage("ord:002", "sell", "MSFT", 50)
        assert history.pending_count == 2
        history.commit("ord:001")
        assert history.pending_count == 1

    def test_metadata_preserved(self):
        history = OrderHistory()
        stage = history.stage(
            "ord:001",
            "buy",
            "AAPL",
            100,
            metadata={"strategy": "momentum", "signal_confidence": "0.95"},
        )
        assert ("strategy", "momentum") in stage.metadata
