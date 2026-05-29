#!/usr/bin/env python3
"""Graph Governance Agent.

CONCEPT:KG-2.2 — Governance Agent

A background autonomous agent that manages continuous knowledge consolidation,
departmental orchestration, and reviews/approves ecosystem proposals
using the GovernanceWorkflow pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agent_utilities.ecosystem.governance_workflow import (
    ChangeProposal,
    GovernanceWorkflow,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GraphGovernanceAgent:
    """Autonomous background daemon for ecosystem governance.

    Coordinates the GovernanceWorkflow and background consolidation tasks.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine,
        workspace: str | Path = ".",
        interval_seconds: int = 300,
    ) -> None:
        self.engine = engine
        self.workspace = workspace
        self.interval_seconds = interval_seconds
        self.workflow = GovernanceWorkflow(engine=engine, workspace=workspace)
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background governance loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "GraphGovernanceAgent started with interval %d seconds",
            self.interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the background governance loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GraphGovernanceAgent stopped")

    async def _run_loop(self) -> None:
        """The main governance loop."""
        while self._running:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in governance loop: %s", e)

            # Wait for next interval
            await asyncio.sleep(self.interval_seconds)

    async def _run_once(self) -> None:
        """Execute one cycle of governance duties."""
        logger.debug("Running governance cycle...")

        # 1. Run the audit cycle (staleness + reflector proposals)
        audit_results = self.workflow.run_audit_cycle()
        logger.debug("Audit cycle results: %s", audit_results)

        # 2. Autonomous Proposal Review
        # The agent acts as an automated reviewer for pending proposals that
        # fall within its acceptable bounds, or escalates them.
        pending = self.workflow.list_pending()
        for proposal in pending:
            await self._evaluate_proposal(proposal)

        # 3. Trigger Semantic Consolidation
        # Call the ConsolidationEngine if available (from Phase 2)
        try:
            from agent_utilities.knowledge_graph.core.maintainer import (
                GraphMaintainer,
            )

            # Or whichever class handles consolidation
            # E.g. trigger compaction
            pass
        except ImportError:
            pass

    async def _evaluate_proposal(self, proposal: ChangeProposal) -> None:
        """Autonomously evaluate a pending proposal."""
        # Simple heuristic: If risk score is moderate, the governance agent can approve it.
        # If high, it requires a human.
        if proposal.risk_score <= 0.6:
            # The agent decides to approve
            self.workflow.approve(
                proposal_id=proposal.id,
                reviewer="governance_agent",
                conditions=["Autonomous approval based on acceptable risk score"],
            )
            logger.info(
                "GovernanceAgent autonomously approved proposal %s", proposal.id
            )
        else:
            # Escalated to human (remains pending)
            logger.info(
                "GovernanceAgent escalating proposal %s to human (risk %s)",
                proposal.id,
                proposal.risk_score,
            )
