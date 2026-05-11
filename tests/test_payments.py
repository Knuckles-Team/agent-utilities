"""Tests for CONCEPT:KG-2.6 — x402 AI Payment Protocol."""

import pytest

from agent_utilities.domains.finance.payments import (
    BudgetLimits,
    PaymentChallenge,
    PaymentGuard,
    PaymentProof,
    PaymentRecord,
    PaymentStatus,
    X402PaymentClient,
)


class TestPaymentChallenge:
    def test_creation(self):
        challenge = PaymentChallenge(
            resource_url="https://api.example.com/data",
            amount=0.50,
            currency="USDC",
            chain="base",
        )
        assert challenge.amount == 0.50
        assert challenge.currency == "USDC"


class TestPaymentProof:
    def test_creation_generates_hash(self):
        proof = PaymentProof(
            challenge_id="ch:001",
            transaction_hash="0xabc123",
            amount=0.50,
            currency="USDC",
            chain="base",
            sender_address="0xsender",
            recipient_address="0xrecipient",
        )
        assert proof.proof_hash != ""
        assert len(proof.proof_hash) == 32

    def test_deterministic_hash(self):
        p1 = PaymentProof(
            challenge_id="ch:001",
            transaction_hash="0xabc",
            amount=1.0,
            currency="USDC",
            chain="base",
            sender_address="s",
            recipient_address="r",
        )
        p2 = PaymentProof(
            challenge_id="ch:001",
            transaction_hash="0xabc",
            amount=1.0,
            currency="USDC",
            chain="base",
            sender_address="s",
            recipient_address="r",
        )
        assert p1.proof_hash == p2.proof_hash


class TestPaymentGuard:
    def test_approved_payment(self):
        guard = PaymentGuard()
        challenge = PaymentChallenge(resource_url="test", amount=5.0)
        approved, reason = guard.validate(challenge)
        assert approved is True

    def test_exceeds_single_limit(self):
        guard = PaymentGuard(BudgetLimits(max_single_payment=1.0))
        challenge = PaymentChallenge(resource_url="test", amount=5.0)
        approved, reason = guard.validate(challenge)
        assert approved is False
        assert "single payment limit" in reason

    def test_exceeds_daily_limit(self):
        guard = PaymentGuard(BudgetLimits(max_daily_spend=10.0))
        guard.record_spend(8.0)
        challenge = PaymentChallenge(resource_url="test", amount=5.0)
        approved, reason = guard.validate(challenge)
        assert approved is False
        assert "daily" in reason.lower()

    def test_disallowed_currency(self):
        guard = PaymentGuard(BudgetLimits(allowed_currencies=["USDC"]))
        challenge = PaymentChallenge(resource_url="test", amount=1.0, currency="BTC")
        approved, reason = guard.validate(challenge)
        assert approved is False
        assert "Currency" in reason

    def test_disallowed_chain(self):
        guard = PaymentGuard(BudgetLimits(allowed_chains=["base"]))
        challenge = PaymentChallenge(resource_url="test", amount=1.0, chain="polygon")
        approved, reason = guard.validate(challenge)
        assert approved is False

    def test_requires_approval_above_threshold(self):
        guard = PaymentGuard(
            BudgetLimits(require_approval_above=5.0, max_single_payment=100.0)
        )
        challenge = PaymentChallenge(resource_url="test", amount=10.0)
        approved, reason = guard.validate(challenge)
        assert approved is False
        assert "manual approval" in reason

    def test_remaining_budget(self):
        guard = PaymentGuard(BudgetLimits(max_daily_spend=100.0))
        guard.record_spend(30.0)
        assert guard.remaining_daily == 70.0

    def test_reset_daily(self):
        guard = PaymentGuard()
        guard.record_spend(50.0)
        guard.reset_daily()
        assert guard.remaining_daily == guard.limits.max_daily_spend


class TestX402PaymentClient:
    def test_handle_402(self):
        client = X402PaymentClient(wallet_address="0xwallet")
        headers = {
            "X-Payment-Amount": "0.50",
            "X-Payment-Currency": "USDC",
            "X-Payment-Recipient": "0xrecipient",
            "X-Payment-Chain": "base",
            "X-Payment-Challenge-Id": "ch:001",
        }
        challenge = client.handle_402(headers, "https://api.example.com/data")
        assert challenge.amount == 0.50
        assert challenge.chain == "base"

    def test_authorize_payment(self):
        client = X402PaymentClient()
        challenge = PaymentChallenge(resource_url="test", amount=1.0)
        approved, reason = client.authorize_payment(challenge)
        assert approved is True

    def test_create_proof(self):
        client = X402PaymentClient(wallet_address="0xwallet")
        challenge = PaymentChallenge(
            resource_url="test",
            amount=1.0,
            currency="USDC",
            chain="base",
            recipient_address="0xrecipient",
            challenge_id="ch:001",
        )
        proof = client.create_proof(challenge, "0xtxhash")
        assert proof.sender_address == "0xwallet"
        assert proof.transaction_hash == "0xtxhash"

    def test_record_payment(self):
        client = X402PaymentClient()
        challenge = PaymentChallenge(resource_url="test", amount=2.0)
        record = client.record_payment(challenge)
        assert record.status == PaymentStatus.CONFIRMED
        history = client.get_payment_history()
        assert len(history) == 1

    def test_get_proof_headers(self):
        client = X402PaymentClient()
        proof = PaymentProof(
            challenge_id="ch:001",
            transaction_hash="0xabc",
            amount=1.0,
            currency="USDC",
            chain="base",
            sender_address="s",
            recipient_address="r",
        )
        headers = client.get_proof_headers(proof)
        assert "Payment-Proof" in headers
        assert headers["X-Payment-Chain"] == "base"

    def test_budget_tracking(self):
        client = X402PaymentClient(budget_limits=BudgetLimits(max_daily_spend=5.0))
        challenge = PaymentChallenge(resource_url="test", amount=3.0)
        proof = client.create_proof(challenge, "0xtx1")
        client.record_payment(challenge, proof)
        assert client.guard.remaining_daily == 2.0
