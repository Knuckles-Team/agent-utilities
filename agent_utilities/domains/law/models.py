from __future__ import annotations

"""Legal Domain Pydantic Models (CONCEPT:KG-2.95).

Aligned to LKIF-Core, Akoma Ntoso, SALI/CLNR, EDRM.
"""


from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
