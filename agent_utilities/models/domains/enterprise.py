from ...models.knowledge_graph import RegistryNode, RegistryNodeType


class BusinessUnitNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.BUSINESS_UNIT
    headcount: int = 0
    budget_allocated: float = 0.0
    cost_center_id: str = ""


class ValueStreamNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.VALUE_STREAM
    primary_kpi: str = ""
    target_value: float = 0.0


class EnterpriseResourceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.ENTERPRISE_RESOURCE
    resource_type: str = "general"
    availability_status: str = "available"
    capacity: float = 1.0


class DelegatedAuthorityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.DELEGATED_AUTHORITY
    authority_level: str = "standard"
    max_approval_amount: float = 0.0
    granted_by: str = ""


class LegalEntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.LEGAL_ENTITY
    jurisdiction: str = ""
    registration_number: str = ""
    entity_type: str = "corporation"


class RiskProfileNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.RISK_PROFILE
    risk_score: float = 0.0
    risk_tolerance: str = "medium"
    assessed_by: str = ""


class RegulatoryFrameworkNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.REGULATORY_FRAMEWORK
    jurisdiction: str = ""
    governing_body: str = ""
    version: str = "latest"


class ComplianceControlNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.COMPLIANCE_CONTROL
    control_type: str = "automated"
    status: str = "active"
    last_audited: str = ""


class SecurityClearanceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.SECURITY_CLEARANCE
    clearance_level: str = "unclassified"
    granted_date: str = ""
    expiry_date: str = ""


class PaymentBudgetNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PAYMENT_BUDGET
    total_budget: float = 0.0
    remaining_budget: float = 0.0
    currency: str = "USD"


class PaymentProofEntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PAYMENT_PROOF_ENTITY
    transaction_id: str = ""
    amount_paid: float = 0.0
    status: str = "pending"
