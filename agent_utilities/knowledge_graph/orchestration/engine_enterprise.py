from __future__ import annotations

"""CONCEPT:KG-2.8"""

import logging
import time
import typing
import uuid

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object

from ...models.domains.enterprise import (
    PaymentBudgetNode,
    RiskProfileNode,
    SecurityClearanceNode,
)

logger = logging.getLogger(__name__)


class EnterpriseEngineMixin(_Base):
    """Enterprise governance capabilities for the KG engine."""

    def allocate_budget(
        self, business_unit_id: str, amount: float, currency: str = "USD"
    ) -> str:
        """Allocate a new payment budget to a business unit."""
        budget_id = f"budget:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = PaymentBudgetNode(
            id=budget_id,
            name=f"Budget {amount} {currency}",
            total_budget=amount,
            remaining_budget=amount,
            currency=currency,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="PaymentBudget")
            self._upsert_node("PaymentBudget", budget_id, data)
            self.backend.execute(
                "MATCH (b:BusinessUnit {id: $bid}), (p:PaymentBudget {id: $pid}) "
                "MERGE (b)-[:HAS_ALLOCATION]->(p)",
                {"bid": business_unit_id, "pid": budget_id},
            )
        return budget_id

    def assess_risk(
        self, target_id: str, risk_score: float, risk_tolerance: str, assessed_by: str
    ) -> str:
        """Create and link a risk profile to a target node (e.g. strategy or unit)."""
        risk_id = f"risk:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = RiskProfileNode(
            id=risk_id,
            name=f"Risk Assessment for {target_id}",
            risk_score=risk_score,
            risk_tolerance=risk_tolerance,
            assessed_by=assessed_by,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="RiskProfile")
            self._upsert_node("RiskProfile", risk_id, data)
            self.backend.execute(
                "MATCH (t {id: $tid}), (r:RiskProfile {id: $rid}) "
                "MERGE (t)-[:ASSESSED_RISK]->(r)",
                {"tid": target_id, "rid": risk_id},
            )
        return risk_id

    def grant_security_clearance(
        self, agent_id: str, clearance_level: str, expiry_date: str
    ) -> str:
        """Grant a security clearance to an agent or human."""
        clearance_id = f"clearance:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = SecurityClearanceNode(
            id=clearance_id,
            name=f"Clearance {clearance_level}",
            clearance_level=clearance_level,
            granted_date=ts,
            expiry_date=expiry_date,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="SecurityClearance")
            self._upsert_node("SecurityClearance", clearance_id, data)
            self.backend.execute(
                "MATCH (a {id: $aid}), (c:SecurityClearance {id: $cid}) "
                "MERGE (a)-[:HAS_CLEARANCE]->(c)",
                {"aid": agent_id, "cid": clearance_id},
            )
        return clearance_id
