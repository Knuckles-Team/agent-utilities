"""Tests for Law Domain Pydantic Models.

CONCEPT:KG-2.6 — Law Domain
CONCEPT:KG-2.1 — Company Intelligence Graph

Validates all law domain Pydantic models, their field constraints,
and OWL ontology alignment.
"""

import pytest
from pydantic import ValidationError

from agent_utilities.domains.law.models import LegalTrustNode, LLCFormationFiling
from agent_utilities.models.knowledge_graph import RegistryNodeType


class TestLegalTrustNode:
    """Tests for LegalTrustNode model."""

    def test_create_trust(self):
        trust = LegalTrustNode(
            id="trust_asset_protection",
            name="Alpha Asset Protection Trust Node",
            trust_name="Alpha Asset Protection Trust",
            trust_type="asset_protection",
            governing_law_state="WY",
            settlor_id="person_alice",
            trustee_ids=["company_trustee_corp"],
            beneficiary_ids=["person_bob"],
            is_funded=True,
        )
        assert trust.type == RegistryNodeType.LEGAL_TRUST
        assert trust.trust_name == "Alpha Asset Protection Trust"
        assert trust.trust_type == "asset_protection"
        assert trust.governing_law_state == "WY"
        assert trust.settlor_id == "person_alice"
        assert trust.trustee_ids == ["company_trustee_corp"]
        assert trust.beneficiary_ids == ["person_bob"]
        assert trust.is_funded is True
        assert trust.ein is None

    def test_invalid_trust_type(self):
        with pytest.raises(ValidationError):
            # Invalid trust type (not in Literal)
            LegalTrustNode(
                id="trust_invalid",
                name="Invalid Trust Node",
                trust_name="Invalid Trust",
                trust_type="invalid_type",  # type: ignore
                governing_law_state="WY",
                settlor_id="person_alice",
            )


class TestLLCFormationFiling:
    """Tests for LLCFormationFiling model."""

    def test_create_filing(self):
        filing = LLCFormationFiling(
            id="filing_wyoming_llc",
            name="Wyoming Articles of Organization",
            company_id="company_acme",
            filing_state="WY",
            registered_agent_name="Wyoming Registered Agents LLC",
            registered_agent_address="123 Agent Way, Cheyenne, WY",
            articles_of_organization_path="/docs/articles_wy.pdf",
            filing_status="active",
        )
        assert filing.type == RegistryNodeType.LLC_FORMATION_FILING
        assert filing.company_id == "company_acme"
        assert filing.filing_state == "WY"
        assert filing.registered_agent_name == "Wyoming Registered Agents LLC"
        assert filing.articles_of_organization_path == "/docs/articles_wy.pdf"
        assert filing.filing_status == "active"

    def test_invalid_filing_status(self):
        with pytest.raises(ValidationError):
            LLCFormationFiling(
                id="filing_invalid",
                name="Invalid Filing",
                company_id="company_acme",
                filing_state="WY",
                registered_agent_name="Agent Inc",
                registered_agent_address="100 Pine St, SF, CA",
                articles_of_organization_path="/docs/articles_wy.pdf",
                filing_status="expired",  # type: ignore
            )
