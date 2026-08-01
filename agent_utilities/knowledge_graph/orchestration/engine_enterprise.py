from __future__ import annotations

"""CONCEPT:AU-KG.research.research-pipeline-runner"""

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
        budget_id = f"budget:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = PaymentBudgetNode(
            id=budget_id,
            name=f"Budget {amount} {currency}",
            total_budget=amount,
            remaining_budget=amount,
            currency=currency,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **self._serialize_node(node))

        if self.backend:
            data = self._serialize_node(node, label="PaymentBudget")
            self._upsert_node("PaymentBudget", budget_id, data)
            # A comma-pattern MATCH plus an edge MERGE both exceed the
            # engine's native Cypher write subset (one leading MATCH, MERGE
            # on a single bare node only;
            # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184);
            # ``link_nodes`` dispatches through the typed engine API.
            self.link_nodes(business_unit_id, budget_id, "HAS_ALLOCATION")
        return budget_id

    def assess_risk(
        self, target_id: str, risk_score: float, risk_tolerance: str, assessed_by: str
    ) -> str:
        """Create and link a risk profile to a target node (e.g. strategy or unit)."""
        risk_id = f"risk:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = RiskProfileNode(
            id=risk_id,
            name=f"Risk Assessment for {target_id}",
            risk_score=risk_score,
            risk_tolerance=risk_tolerance,
            assessed_by=assessed_by,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **self._serialize_node(node))

        if self.backend:
            data = self._serialize_node(node, label="RiskProfile")
            self._upsert_node("RiskProfile", risk_id, data)
            # See allocate_budget above for why this is a typed link, not a
            # comma-pattern MATCH + edge MERGE.
            self.link_nodes(target_id, risk_id, "ASSESSED_RISK")
        return risk_id

    def grant_security_clearance(
        self, agent_id: str, clearance_level: str, expiry_date: str
    ) -> str:
        """Grant a security clearance to an agent or human."""
        clearance_id = f"clearance:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = SecurityClearanceNode(
            id=clearance_id,
            name=f"Clearance {clearance_level}",
            clearance_level=clearance_level,
            granted_date=ts,
            expiry_date=expiry_date,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **self._serialize_node(node))

        if self.backend:
            data = self._serialize_node(node, label="SecurityClearance")
            self._upsert_node("SecurityClearance", clearance_id, data)
            # See allocate_budget above for why this is a typed link, not a
            # comma-pattern MATCH + edge MERGE.
            self.link_nodes(agent_id, clearance_id, "HAS_CLEARANCE")
        return clearance_id
