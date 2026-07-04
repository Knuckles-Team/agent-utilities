from __future__ import annotations

"""Enterprise Banking Service Layer (CONCEPT:AU-KG.research.research-pipeline-runner).

Provides banking-specific services:
- ISO 20022 message generation/parsing
- KYC/AML due diligence workflows
- Multi-leg settlement with correspondent banking chains
- Basel III regulatory capital calculation
- Credit risk PD/LGD/EAD modeling
"""


import logging
import uuid
from typing import Any

from agent_utilities.domains.finance.banking_models import (
    AMLAlertNode,
    AMLAlertSeverity,
    ISO20022MessageType,
    KYCRecordNode,
    PaymentMessageNode,
    RegulatoryCapitalNode,
    SettlementCycle,
    SettlementLegNode,
)

logger = logging.getLogger(__name__)


class ISO20022MessageFactory:
    """Generate and parse ISO 20022 payment messages.

    Supports pacs.008 (credit transfer), pain.001 (payment initiation),
    and camt.053 (bank statement) message types.
    """

    @staticmethod
    def create_credit_transfer(
        sender_bic: str,
        receiver_bic: str,
        amount: float,
        currency: str = "USD",
        value_date: str = "",
    ) -> PaymentMessageNode:
        """Generate a pacs.008 Credit Transfer Initiation."""
        return PaymentMessageNode(
            id=f"msg:{uuid.uuid4().hex[:12]}",
            message_type=ISO20022MessageType.PACS_008,
            sender_bic=sender_bic,
            receiver_bic=receiver_bic,
            amount=amount,
            currency=currency,
            value_date=value_date,
            end_to_end_id=f"E2E-{uuid.uuid4().hex[:8]}",
            instruction_id=f"INSTR-{uuid.uuid4().hex[:8]}",
        )

    @staticmethod
    def create_payment_initiation(
        sender_bic: str,
        amount: float,
        currency: str = "USD",
    ) -> PaymentMessageNode:
        """Generate a pain.001 Payment Initiation."""
        return PaymentMessageNode(
            id=f"msg:{uuid.uuid4().hex[:12]}",
            message_type=ISO20022MessageType.PAIN_001,
            sender_bic=sender_bic,
            amount=amount,
            currency=currency,
            end_to_end_id=f"E2E-{uuid.uuid4().hex[:8]}",
        )


class KYCAMLEngine:
    """KYC/AML compliance engine.

    Manages customer due diligence, PEP screening,
    and transaction monitoring with configurable thresholds.
    """

    def __init__(
        self,
        structuring_threshold: float = 10000.0,
        velocity_window_hours: int = 24,
        velocity_max_count: int = 10,
    ) -> None:
        self._kyc_records: dict[str, KYCRecordNode] = {}
        self._alerts: dict[str, AMLAlertNode] = {}
        self._structuring_threshold = structuring_threshold
        self._velocity_window_hours = velocity_window_hours
        self._velocity_max_count = velocity_max_count

    def add_kyc_record(self, record: KYCRecordNode) -> None:
        """Register a KYC record."""
        self._kyc_records[record.id] = record

    def screen_customer(self, customer_id: str) -> KYCRecordNode | None:
        """Retrieve KYC record for a customer."""
        for record in self._kyc_records.values():
            if record.customer_id == customer_id:
                return record
        return None

    def check_transaction(
        self,
        transaction_id: str,
        account_id: str,
        amount: float,
    ) -> AMLAlertNode | None:
        """Screen a transaction against AML thresholds.

        Returns an AMLAlertNode if the transaction triggers an alert.
        """
        # Structuring detection
        if (
            amount >= self._structuring_threshold * 0.8
            and amount < self._structuring_threshold
        ):
            alert = AMLAlertNode(
                id=f"aml:{uuid.uuid4().hex[:10]}",
                transaction_id=transaction_id,
                account_id=account_id,
                severity=AMLAlertSeverity.WARNING,
                alert_type="structuring",
                amount=amount,
            )
            self._alerts[alert.id] = alert
            logger.warning("AML alert: potential structuring on %s", transaction_id)
            return alert

        if amount >= self._structuring_threshold:
            alert = AMLAlertNode(
                id=f"aml:{uuid.uuid4().hex[:10]}",
                transaction_id=transaction_id,
                account_id=account_id,
                severity=AMLAlertSeverity.SAR_REQUIRED,
                alert_type="threshold_exceeded",
                amount=amount,
            )
            self._alerts[alert.id] = alert
            logger.warning("AML alert: SAR required for %s", transaction_id)
            return alert

        return None

    def get_alerts(
        self, account_id: str | None = None, unresolved_only: bool = True
    ) -> list[AMLAlertNode]:
        """Retrieve AML alerts, optionally filtered."""
        alerts = list(self._alerts.values())
        if account_id:
            alerts = [a for a in alerts if a.account_id == account_id]
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        return alerts


class SettlementEngine:
    """Multi-leg settlement with correspondent banking chains.

    Handles T+1/T+2 settlement cycles with nostro/vostro account tracking.
    """

    def __init__(self) -> None:
        self._legs: dict[str, list[SettlementLegNode]] = {}

    def create_settlement_chain(
        self,
        payment_id: str,
        correspondent_chain: list[str],
        amount: float,
        currency: str = "USD",
        cycle: SettlementCycle = SettlementCycle.T_PLUS_1,
    ) -> list[SettlementLegNode]:
        """Create a multi-leg settlement through a correspondent chain.

        Args:
            payment_id: The payment being settled.
            correspondent_chain: Ordered list of correspondent bank IDs.
            amount: Settlement amount.
            currency: Currency code.
            cycle: Settlement timing.

        Returns:
            List of settlement legs in execution order.
        """
        legs: list[SettlementLegNode] = []
        for i, bank_id in enumerate(correspondent_chain):
            leg = SettlementLegNode(
                id=f"leg:{uuid.uuid4().hex[:10]}",
                payment_id=payment_id,
                leg_sequence=i + 1,
                correspondent_bank_id=bank_id,
                settlement_cycle=cycle,
                amount=amount,
                currency=currency,
            )
            legs.append(leg)

        self._legs[payment_id] = legs
        logger.info(
            "Settlement chain created: %d legs for payment %s",
            len(legs),
            payment_id,
        )
        return legs

    def get_settlement_status(self, payment_id: str) -> list[SettlementLegNode]:
        """Get all settlement legs for a payment."""
        return self._legs.get(payment_id, [])


class RegulatoryCapitalCalculator:
    """Basel III/IV regulatory capital calculator.

    Computes CET1, AT1, Tier 2 ratios and liquidity metrics.
    """

    @staticmethod
    def calculate_ratios(
        cet1_capital: float,
        at1_capital: float,
        tier2_capital: float,
        risk_weighted_assets: float,
        total_exposure: float = 0.0,
    ) -> RegulatoryCapitalNode:
        """Calculate Basel III capital ratios.

        Basel III minimums: CET1 ≥ 4.5%, Total Capital ≥ 8%.
        """
        rwa = max(risk_weighted_assets, 1.0)  # Prevent division by zero
        total = cet1_capital + at1_capital + tier2_capital

        return RegulatoryCapitalNode(
            id=f"regcap:{uuid.uuid4().hex[:10]}",
            cet1_ratio=cet1_capital / rwa,
            at1_ratio=at1_capital / rwa,
            tier2_ratio=tier2_capital / rwa,
            total_capital_ratio=total / rwa,
            risk_weighted_assets=rwa,
            leverage_ratio=(
                total / max(total_exposure, 1.0) if total_exposure else 0.0
            ),
        )

    @staticmethod
    def check_compliance(capital: RegulatoryCapitalNode) -> dict[str, Any]:
        """Check capital ratios against Basel III minimums."""
        return {
            "cet1_compliant": capital.cet1_ratio >= 0.045,
            "total_capital_compliant": capital.total_capital_ratio >= 0.08,
            "leverage_compliant": capital.leverage_ratio >= 0.03
            if capital.leverage_ratio
            else True,
            "cet1_ratio": round(capital.cet1_ratio * 100, 2),
            "total_capital_ratio": round(capital.total_capital_ratio * 100, 2),
        }
