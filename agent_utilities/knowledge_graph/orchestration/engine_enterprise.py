from __future__ import annotations

"""CONCEPT:AU-KG.research.research-pipeline-runner"""

import logging
import typing

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
from .engine_action_result import _persist_action_result

logger = logging.getLogger(__name__)


class EnterpriseEngineMixin(_Base):
    """Enterprise governance capabilities for the KG engine."""

    def allocate_budget(
        self, business_unit_id: str, amount: float, currency: str = "USD"
    ) -> str:
        """Allocate a new payment budget to a business unit."""
        return _persist_action_result(
            self,
            "budget",
            "PaymentBudget",
            PaymentBudgetNode,
            lambda _node_id, _timestamp: {
                "name": f"Budget {amount} {currency}",
                "total_budget": amount,
                "remaining_budget": amount,
                "currency": currency,
            },
            backend_links=((business_unit_id, None, "HAS_ALLOCATION"),),
        )

    def assess_risk(
        self, target_id: str, risk_score: float, risk_tolerance: str, assessed_by: str
    ) -> str:
        """Create and link a risk profile to a target node (e.g. strategy or unit)."""
        return _persist_action_result(
            self,
            "risk",
            "RiskProfile",
            RiskProfileNode,
            lambda _node_id, _timestamp: {
                "name": f"Risk Assessment for {target_id}",
                "risk_score": risk_score,
                "risk_tolerance": risk_tolerance,
                "assessed_by": assessed_by,
            },
            backend_links=((target_id, None, "ASSESSED_RISK"),),
        )

    def grant_security_clearance(
        self, agent_id: str, clearance_level: str, expiry_date: str
    ) -> str:
        """Grant a security clearance to an agent or human."""
        return _persist_action_result(
            self,
            "clearance",
            "SecurityClearance",
            SecurityClearanceNode,
            lambda _node_id, timestamp: {
                "name": f"Clearance {clearance_level}",
                "clearance_level": clearance_level,
                "granted_date": timestamp,
                "expiry_date": expiry_date,
            },
            backend_links=((agent_id, None, "HAS_CLEARANCE"),),
        )
