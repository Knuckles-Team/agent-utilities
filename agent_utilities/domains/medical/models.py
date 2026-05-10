from __future__ import annotations

"""Medical Domain Pydantic Models (CONCEPT:KG-2.90).

Aligned to SNOMED-CT, ICD-10/11, LOINC, RxNorm, HL7 FHIR, CDISC.
"""


from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ConsentScope(StrEnum):
    TREATMENT = "treatment"
    RESEARCH = "research"
    DATA_SHARING = "data_sharing"


class TrialPhase(StrEnum):
    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    PHASE_III = "phase_iii"
    PHASE_IV = "phase_iv"


class PatientNode(BaseModel):
    """A patient receiving medical care. PHI-protected (HIPAA §164.312)."""

    id: str
    name: str = ""
    date_of_birth: str = ""
    medical_record_number: str = ""
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClinicalConditionNode(BaseModel):
    """A diagnosed medical condition. Aligned to SNOMED-CT."""

    id: str
    name: str
    snomed_id: str = ""
    icd_code: str = ""
    severity: str = ""
    onset_date: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MedicationNode(BaseModel):
    """A prescribed drug. Aligned to RxNorm."""

    id: str
    name: str
    rxnorm_cui: str = ""
    dosage: str = ""
    route: str = ""
    contraindications: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClinicalTrialNode(BaseModel):
    """A medical research trial. Aligned to CDISC/CDASH."""

    id: str
    title: str
    phase: TrialPhase = TrialPhase.PHASE_I
    sponsor: str = ""
    principal_investigator: str = ""
    enrollment_count: int = 0
    status: str = "recruiting"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsentRecordNode(BaseModel):
    """Granular patient consent. Aligned to HL7 FHIR Consent."""

    id: str
    patient_id: str
    scope: ConsentScope = ConsentScope.TREATMENT
    granted: bool = True
    valid_from: str = ""
    valid_until: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabResultNode(BaseModel):
    """A laboratory test result. Aligned to LOINC."""

    id: str
    patient_id: str
    test_name: str
    loinc_code: str = ""
    value: str = ""
    unit: str = ""
    reference_range: str = ""
    collected_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
