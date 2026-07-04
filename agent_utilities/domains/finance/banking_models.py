from __future__ import annotations

"""Enterprise Banking Pydantic Models (CONCEPT:AU-KG.research.research-pipeline-runner).

Models for ISO 20022 messaging, KYC/AML, correspondent banking,
regulatory capital (Basel III), and credit risk.
"""


from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ISO20022MessageType(StrEnum):
    """ISO 20022 message types."""

    PACS_008 = "pacs.008"  # Credit Transfer
    PAIN_001 = "pain.001"  # Payment Initiation
    CAMT_053 = "camt.053"  # Bank-to-Customer Statement
    CAMT_054 = "camt.054"  # Bank-to-Customer Debit/Credit Notification
    PACS_002 = "pacs.002"  # Payment Status Report


class SettlementCycle(StrEnum):
    """Settlement timing."""

    T_PLUS_0 = "T+0"
    T_PLUS_1 = "T+1"
    T_PLUS_2 = "T+2"


class KYCRiskLevel(StrEnum):
    """KYC customer risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


class AMLAlertSeverity(StrEnum):
    """AML alert severity."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SAR_REQUIRED = "sar_required"


class BankAccountNode(BaseModel):
    """A bank account with IBAN/SWIFT identifiers."""

    id: str
    account_name: str = ""
    iban: str = ""
    swift_code: str = ""
    currency: str = "USD"
    balance: float = 0.0
    account_type: str = ""  # checking, savings, nostro, vostro
    correspondent_bank_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaymentMessageNode(BaseModel):
    """An ISO 20022 payment message."""

    id: str
    message_type: ISO20022MessageType
    sender_bic: str = ""
    receiver_bic: str = ""
    amount: float = 0.0
    currency: str = "USD"
    value_date: str = ""
    end_to_end_id: str = ""
    instruction_id: str = ""
    raw_xml: str = ""
    status: str = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KYCRecordNode(BaseModel):
    """Know Your Customer due diligence record."""

    id: str
    customer_id: str
    customer_name: str = ""
    risk_classification: KYCRiskLevel = KYCRiskLevel.LOW
    pep_status: bool = False
    sanctions_clear: bool = True
    last_screening_date: str = ""
    identity_verified: bool = False
    source_of_funds: str = ""
    beneficial_owners: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AMLAlertNode(BaseModel):
    """Anti-Money Laundering alert."""

    id: str
    transaction_id: str
    account_id: str = ""
    severity: AMLAlertSeverity = AMLAlertSeverity.WARNING
    alert_type: str = ""  # structuring, velocity, geography, pep
    amount: float = 0.0
    triggered_at: str = ""
    resolved: bool = False
    resolution_notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreditRiskNode(BaseModel):
    """Credit risk assessment (PD/LGD/EAD)."""

    id: str
    account_id: str
    probability_of_default: float = Field(default=0.0, ge=0.0, le=1.0)
    loss_given_default: float = Field(default=0.0, ge=0.0, le=1.0)
    exposure_at_default: float = 0.0
    expected_loss: float = 0.0
    risk_rating: str = ""
    model_version: str = ""
    assessed_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SettlementLegNode(BaseModel):
    """A single leg in multi-leg settlement."""

    id: str
    payment_id: str
    leg_sequence: int = 1
    correspondent_bank_id: str = ""
    settlement_cycle: SettlementCycle = SettlementCycle.T_PLUS_1
    settlement_date: str = ""
    amount: float = 0.0
    currency: str = "USD"
    status: str = "pending"  # pending, settled, failed
    nostro_account_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegulatoryCapitalNode(BaseModel):
    """Basel III/IV regulatory capital."""

    id: str
    institution_id: str = ""
    cet1_ratio: float = 0.0
    at1_ratio: float = 0.0
    tier2_ratio: float = 0.0
    total_capital_ratio: float = 0.0
    risk_weighted_assets: float = 0.0
    leverage_ratio: float = 0.0
    lcr: float = 0.0  # Liquidity Coverage Ratio
    nsfr: float = 0.0  # Net Stable Funding Ratio
    reporting_date: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
