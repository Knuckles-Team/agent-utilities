from __future__ import annotations

"""Legal Domain Pydantic Models (CONCEPT:AU-KG.research.research-pipeline-runner).

Aligned to LKIF-Core, Akoma Ntoso, SALI/CLNR, EDRM.
"""


from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


class PrecedentStrength(StrEnum):
    BINDING = "binding"
    PERSUASIVE = "persuasive"
    DISTINGUISHABLE = "distinguishable"


class MatterStatus(StrEnum):
    INTAKE = "intake"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    CLOSED = "closed"
    ARCHIVED = "archived"


class CaseLawNode(BaseModel):
    """A judicial precedent. Aligned to LKIF-Core."""

    id: str
    case_name: str
    citation: str = ""
    jurisdiction: str = ""
    decided_date: str = ""
    precedent_strength: PrecedentStrength = PrecedentStrength.PERSUASIVE
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatuteNode(BaseModel):
    """A legislative enactment. Aligned to Akoma Ntoso."""

    id: str
    title: str
    jurisdiction: str = ""
    effective_date: str = ""
    section: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContractClauseNode(BaseModel):
    """A specific clause within a legal contract."""

    id: str
    contract_id: str
    clause_type: str = ""
    text: str = ""
    risk_level: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalMatterNode(BaseModel):
    """A client matter lifecycle."""

    id: str
    matter_name: str
    client_id: str = ""
    status: MatterStatus = MatterStatus.INTAKE
    assigned_attorneys: list[str] = Field(default_factory=list)
    billed_hours: float = 0.0
    opened_date: str = ""
    closed_date: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalTrustNode(RegistryNode):
    """Fiduciary trust entity tracking.
    Maps to OWL class :LegalTrust in ontology_legal.ttl.
    """

    type: RegistryNodeType = Field(default=RegistryNodeType.LEGAL_TRUST)
    trust_name: str = Field(..., description="Full legal name of the trust")
    trust_type: Literal[
        "revocable", "irrevocable", "asset_protection", "special_needs"
    ] = Field(..., description="Legal category of the trust")
    governing_law_state: str = Field(
        ..., description="US state of governing law (e.g., 'WY', 'NV')"
    )
    settlor_id: str = Field(..., description="Person node ID of the settlor")
    trustee_ids: list[str] = Field(
        default_factory=list, description="Person/Company IDs of trustees"
    )
    beneficiary_ids: list[str] = Field(
        default_factory=list, description="Person/Company IDs of beneficiaries"
    )
    is_funded: bool = Field(
        default=False, description="Whether assets have been assigned to Schedule A"
    )
    ein: str | None = Field(
        default=None,
        description="Employer Identification Number for irrevocable trusts",
    )


class LLCFormationFiling(RegistryNode):
    """State filing lifecycle tracker for an LLC.
    Maps to OWL class :RegulatoryFiling in ontology_legal.ttl.
    """

    type: RegistryNodeType = Field(default=RegistryNodeType.LLC_FORMATION_FILING)
    company_id: str = Field(..., description="CompanyProfile node ID")
    filing_state: str = Field(..., description="Filing state (e.g., 'DE', 'WY')")
    registered_agent_name: str = Field(..., description="Designated registered agent")
    registered_agent_address: str = Field(
        ..., description="Address of registered agent"
    )
    articles_of_organization_path: str = Field(
        ..., description="Path to generated articles document"
    )
    filing_status: Literal[
        "draft", "pending_submission", "submitted", "active", "rejected"
    ] = "draft"
