"""Telemetry Engine — Synthesized Observability Facade.

CONCEPT:AU-OS.observability.telemetry-observability — Telemetry Engine

Provides a single entry point for all observability concerns:
- Token usage tracking (OS-5.5 via ``TokenTracker``)
- Audit logging (OS-5.6 via ``AuditLogger``)
- Deterministic replay (OS-5.6 via ``DistributedReplayEngine``)
- OpenTelemetry setup (OS-5.8 placeholder)

This facade wires the previously unwired AuditLogger and TokenTracker
into the main graph execution pipeline via ``on_graph_start()``,
``on_graph_end()``, and ``on_response()`` hooks.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryEngine:
    """Synthesized observability engine.

    CONCEPT:AU-OS.observability.telemetry-observability — Telemetry Engine

    Usage::

        telemetry = TelemetryEngine()

        # At graph start
        telemetry.on_graph_start(run_id="run-1", agent_id="agent-1", query="...")

        # After each LLM response
        telemetry.on_response(run_id="run-1", usage={"prompt": 100, "response": 50})

        # At graph end
        telemetry.on_graph_end(run_id="run-1", status="success")
    """

    def __init__(self, enable_audit: bool = True, enable_tokens: bool = True) -> None:
        self._audit_logger: Any = None
        self._token_tracker: Any = None
        self._enable_audit = enable_audit
        self._enable_tokens = enable_tokens
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize sub-engines to avoid import-time overhead."""
        if self._initialized:
            return
        self._initialized = True

        if self._enable_audit:
            try:
                from .audit_logger import AuditLogger

                self._audit_logger = AuditLogger()
            except Exception:
                logger.debug("AuditLogger not available, skipping audit logging")

        if self._enable_tokens:
            try:
                from .token_tracker import TokenUsageTracker

                self._token_tracker = TokenUsageTracker()
            except Exception:
                logger.debug("TokenTracker not available, skipping token tracking")

    def on_graph_start(
        self,
        run_id: str,
        agent_id: str = "",
        query: str = "",
        **metadata: Any,
    ) -> None:
        """Record the start of a graph execution."""
        self._lazy_init()
        if self._audit_logger:
            self._audit_logger.log(
                actor=agent_id or "system",
                action="graph.start",
                resource_type="graph",
                resource_id=run_id,
                details={"query_length": len(query), **metadata},
            )

    def on_response(
        self,
        run_id: str,
        usage: dict[str, int] | None = None,
        model: str = "",
        **metadata: Any,
    ) -> None:
        """Record token usage from an LLM response."""
        self._lazy_init()
        if self._token_tracker and usage:
            try:
                from .token_tracker import TokenUsageRecord

                record = TokenUsageRecord(
                    session_id=run_id,
                    model_name=model,
                    prompt_tokens=usage.get("prompt", 0),
                    response_tokens=usage.get("response", 0),
                    thoughts_tokens=usage.get("thoughts", 0),
                    tool_use_tokens=usage.get("tool_use", 0),
                )
                self._token_tracker.record(record)
            except Exception as e:
                logger.debug("Token recording failed: %s", e)

    def on_graph_end(
        self,
        run_id: str,
        status: str = "success",
        duration_ms: float = 0.0,
        **metadata: Any,
    ) -> None:
        """Record the end of a graph execution."""
        self._lazy_init()
        if self._audit_logger:
            self._audit_logger.log(
                actor="system",
                action="graph.end",
                resource_type="graph",
                resource_id=run_id,
                details={
                    "status": status,
                    "duration_ms": duration_ms,
                    **(metadata or {}),
                },
            )

    def get_token_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Get token usage summary, optionally filtered by run_id."""
        self._lazy_init()
        if self._token_tracker:
            if run_id:
                return self._token_tracker.get_session_totals(run_id).model_dump()
            return self._token_tracker.export_summary()
        return {}

    def get_audit_trail(
        self, limit: int = 100, action_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recent audit entries."""
        self._lazy_init()
        if self._audit_logger:
            records = self._audit_logger.query(action=action_filter, limit=limit)
            return [r.model_dump() for r in records]
        return []


# Replay Engine (OS-5.6) — Deterministic execution trace recording & replay
# HITL Escalation Matrix (OS-5.12) — formal value/risk → approver policy
from .escalation_matrix import (  # noqa: E402
    EscalationDecision,
    EscalationGate,
    EscalationMatrix,
    EscalationOutcome,
    EscalationRule,
    Fallback,
    RiskTier,
    ValueTier,
    classify_risk_tier,
    classify_value_tier,
    make_decision_provider,
)

# Langfuse exporter (ECO-4.24) — optional auto span/token/trace export
from .langfuse_exporter import (  # noqa: E402
    LangfuseExporter,
    get_langfuse_exporter,
)
from .replay_engine import (  # noqa: E402
    DistributedReplayEngine,
    InteractionRecord,
    ReplayManifest,
)

# Self-ingest telemetry (KG-2.304) — ship our OWN logs/RunTrace/ToolCall into the
# epistemic-graph engine obs store (dogfooding). Opt-in, default-off.
from .self_ingest import (  # noqa: E402
    SelfIngestConfig,
    SelfIngestLogHandler,
    SelfIngestSink,
    emit_run_trace,
    emit_tool_call,
    get_self_ingest_sink,
    install_self_ingest_logging,
    reset_self_ingest_sink,
    set_self_ingest_sink,
)

__all__ = [
    "TelemetryEngine",
    # Replay Engine (OS-5.6)
    "DistributedReplayEngine",
    "ReplayManifest",
    "InteractionRecord",
    # HITL Escalation Matrix (OS-5.12)
    "EscalationMatrix",
    "EscalationGate",
    "EscalationRule",
    "EscalationDecision",
    "EscalationOutcome",
    "RiskTier",
    "ValueTier",
    "Fallback",
    "classify_risk_tier",
    "classify_value_tier",
    "make_decision_provider",
    # Langfuse exporter (ECO-4.24)
    "LangfuseExporter",
    "get_langfuse_exporter",
    # Self-ingest telemetry (KG-2.304)
    "SelfIngestSink",
    "SelfIngestConfig",
    "SelfIngestLogHandler",
    "get_self_ingest_sink",
    "set_self_ingest_sink",
    "reset_self_ingest_sink",
    "install_self_ingest_logging",
    "emit_run_trace",
    "emit_tool_call",
]
