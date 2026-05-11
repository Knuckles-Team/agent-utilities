from __future__ import annotations

"""Government Domain Pydantic Models (CONCEPT:KG-2.0).

Aligned to NIST 800-53, FedRAMP, and NIEM standards.
"""


from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ClassificationLevel(StrEnum):
    UNCLASSIFIED = "unclassified"
    CUI = "cui"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"  # nosec
    TOP_SECRET = "top_secret"  # nosec


class ATOStatus(StrEnum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CONDITIONAL = "conditional"
    REVOKED = "revoked"


class FedRAMPImpact(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class SecurityControlNode(BaseModel):
    """A NIST 800-53 security control."""

    id: str
    control_id: str  # e.g., AC-2, AU-6
    family: str = ""  # AC, AU, CM, etc.
    title: str = ""
    description: str = ""
    baseline: str = ""  # low, moderate, high
    implementation_status: str = ""  # implemented, planned, not_applicable
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuthorizationToOperateNode(BaseModel):
    """A FedRAMP Authorization To Operate."""

    id: str
    system_name: str
    status: ATOStatus = ATOStatus.PENDING
    impact_level: FedRAMPImpact = FedRAMPImpact.MODERATE
    authorizing_official: str = ""
    authorized_date: str = ""
    expiry_date: str = ""
    controls_implemented: int = 0
    controls_total: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassifiedDocumentNode(BaseModel):
    """A document with security classification."""

    id: str
    title: str
    classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    originator: str = ""
    declassify_date: str = ""
    handling_caveats: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitizenServiceNode(BaseModel):
    """A public service with eligibility rules. Aligned to NIEM."""

    id: str
    service_name: str
    agency: str = ""
    eligibility_rules: list[str] = Field(default_factory=list)
    application_url: str = ""
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
