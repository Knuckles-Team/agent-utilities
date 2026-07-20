"""
x402 AI Payment Protocol — CONCEPT:AU-KG.research.research-pipeline-runner

First-class module for autonomous AI agent payments using the x402
protocol (HTTP 402 challenge-response) with budget guards and KG provenance.

Sources: x402 Open Standard (Coinbase), AgentPay SDK
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class PaymentStatus(StrEnum):
    """Lifecycle states for a payment."""

    PENDING = "pending"
    CHALLENGED = "challenged"
    SIGNED = "signed"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED_BY_GUARD = "rejected_by_guard"


@dataclass
class PaymentChallenge:
    """
    An HTTP 402 challenge received from a paywall-protected resource.
    Contains the payment requirements to unlock the resource.
    """

    resource_url: str
    amount: float
    currency: str = "USDC"
    recipient_address: str = ""
    chain: str = "base"
    challenge_id: str = ""
    expires_at: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PaymentProof:
    """
    Proof of payment for inclusion in the request header.
    Submitted as 'Payment-Proof' HTTP header to unlock the resource.
    """

    challenge_id: str
    transaction_hash: str
    amount: float
    currency: str
    chain: str
    sender_address: str
    recipient_address: str
    signed_at: str = ""
    proof_hash: str = ""

    def __post_init__(self):
        if not self.signed_at:
            self.signed_at = datetime.now(UTC).isoformat()
        if not self.proof_hash:
            content = f"{self.challenge_id}:{self.transaction_hash}:{self.amount}:{self.chain}"
            self.proof_hash = hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class PaymentRecord:
    """Complete record of a payment transaction for KG persistence."""

    payment_id: str
    status: PaymentStatus
    challenge: PaymentChallenge
    proof: PaymentProof | None = None
    created_at: str = ""
    completed_at: str = ""
    error_message: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


@dataclass
class BudgetLimits:
    """Configurable budget limits for autonomous payments."""

    max_single_payment: float = 10.0
    max_daily_spend: float = 100.0
    max_monthly_spend: float = 1000.0
    allowed_currencies: list[str] = field(
        default_factory=lambda: ["USDC", "USDT", "DAI"]
    )
    allowed_chains: list[str] = field(
        default_factory=lambda: ["base", "ethereum", "solana"]
    )
    require_approval_above: float = 50.0


class PaymentGuard:
    """
    Budget enforcement and approval workflow for autonomous payments.
    Prevents runaway spending by AI agents.
    """

    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self._daily_spend: float = 0.0
        self._monthly_spend: float = 0.0
        self._payment_count: int = 0

    def validate(self, challenge: PaymentChallenge) -> tuple[bool, str]:
        """
        Validate a payment challenge against budget limits.

        Returns:
            (approved, reason) tuple.
        """
        # Currency check
        if challenge.currency not in self.limits.allowed_currencies:
            return False, f"Currency {challenge.currency} not in allowed list"

        # Chain check
        if challenge.chain not in self.limits.allowed_chains:
            return False, f"Chain {challenge.chain} not in allowed list"

        # Single payment limit
        if challenge.amount > self.limits.max_single_payment:
            return (
                False,
                f"Amount {challenge.amount} exceeds single payment limit {self.limits.max_single_payment}",
            )

        # Daily spend limit
        if self._daily_spend + challenge.amount > self.limits.max_daily_spend:
            return (
                False,
                f"Would exceed daily spend limit ({self._daily_spend + challenge.amount:.2f} > {self.limits.max_daily_spend})",
            )

        # Monthly spend limit
        if self._monthly_spend + challenge.amount > self.limits.max_monthly_spend:
            return False, "Would exceed monthly spend limit"

        # Approval required above threshold
        if challenge.amount > self.limits.require_approval_above:
            return (
                False,
                f"Amount {challenge.amount} requires manual approval (threshold: {self.limits.require_approval_above})",
            )

        return True, "Approved"

    def record_spend(self, amount: float):
        """Record a completed payment for budget tracking."""
        self._daily_spend += amount
        self._monthly_spend += amount
        self._payment_count += 1

    def reset_daily(self):
        """Reset daily spend counter (call at midnight)."""
        self._daily_spend = 0.0

    def reset_monthly(self):
        """Reset monthly spend counter (call at month start)."""
        self._monthly_spend = 0.0

    @property
    def remaining_daily(self) -> float:
        return max(0.0, self.limits.max_daily_spend - self._daily_spend)

    @property
    def remaining_monthly(self) -> float:
        return max(0.0, self.limits.max_monthly_spend - self._monthly_spend)


class X402PaymentClient:
    """
    HTTP 402 challenge-response handler for autonomous AI agent payments.

    Workflow:
    1. Agent requests a resource → receives HTTP 402 with PaymentChallenge
    2. PaymentGuard validates the challenge against budget limits
    3. Agent signs the payment transaction
    4. Agent resubmits request with Payment-Proof header
    5. PaymentRecord is created in KG with full PROV-O chain
    """

    def __init__(
        self,
        wallet_address: str = "",
        budget_limits: BudgetLimits | None = None,
    ):
        self.wallet_address = wallet_address
        self.guard = PaymentGuard(budget_limits)
        self._records: list[PaymentRecord] = []
        self._payment_counter = 0

    def handle_402(self, response_headers: dict, resource_url: str) -> PaymentChallenge:
        """
        Parse an HTTP 402 response into a PaymentChallenge.

        Args:
            response_headers: HTTP response headers containing payment info.
            resource_url: The URL that returned 402.
        """
        return PaymentChallenge(
            resource_url=resource_url,
            amount=float(response_headers.get("X-Payment-Amount", "0")),
            currency=response_headers.get("X-Payment-Currency", "USDC"),
            recipient_address=response_headers.get("X-Payment-Recipient", ""),
            chain=response_headers.get("X-Payment-Chain", "base"),
            challenge_id=response_headers.get("X-Payment-Challenge-Id", ""),
            expires_at=response_headers.get("X-Payment-Expires", ""),
        )

    def authorize_payment(self, challenge: PaymentChallenge) -> tuple[bool, str]:
        """Validate a payment challenge against budget guards."""
        return self.guard.validate(challenge)

    def create_proof(
        self, challenge: PaymentChallenge, transaction_hash: str
    ) -> PaymentProof:
        """
        Create a payment proof after signing a transaction.

        In production, the transaction_hash comes from the blockchain wallet.
        This method wraps the signed transaction into a PaymentProof for
        inclusion in the retry request's headers.
        """
        return PaymentProof(
            challenge_id=challenge.challenge_id,
            transaction_hash=transaction_hash,
            amount=challenge.amount,
            currency=challenge.currency,
            chain=challenge.chain,
            sender_address=self.wallet_address,
            recipient_address=challenge.recipient_address,
        )

    def record_payment(
        self,
        challenge: PaymentChallenge,
        proof: PaymentProof | None = None,
        status: PaymentStatus = PaymentStatus.CONFIRMED,
        error: str = "",
    ) -> PaymentRecord:
        """Record a payment for audit trail and KG persistence."""
        self._payment_counter += 1
        record = PaymentRecord(
            payment_id=f"pay:{self._payment_counter:06d}",
            status=status,
            challenge=challenge,
            proof=proof,
            error_message=error,
        )

        if status == PaymentStatus.CONFIRMED and proof:
            self.guard.record_spend(challenge.amount)

        self._records.append(record)
        logger.info(
            f"Payment {record.payment_id}: {status.value} — {challenge.amount} {challenge.currency}"
        )
        return record

    def get_payment_history(self) -> list[PaymentRecord]:
        """Get complete payment history."""
        return list(self._records)

    def get_proof_headers(self, proof: PaymentProof) -> dict[str, str]:
        """Generate HTTP headers for payment proof submission."""
        return {
            "Payment-Proof": proof.proof_hash,
            "X-Payment-Transaction": proof.transaction_hash,
            "X-Payment-Challenge-Id": proof.challenge_id,
            "X-Payment-Chain": proof.chain,
        }
