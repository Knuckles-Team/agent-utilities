#!/usr/bin/python
from __future__ import annotations

"""Audit Logger — Append-Only Compliance Logging (CONCEPT:AU-OS.config.secrets-authentication).

Ported from MATE's audit_service.py with KG-native persistence.
OWL: :AuditLog rdfs:subClassOf :Event
"""


import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Action constants (ported from MATE)
ACTION_AGENT_CREATE = "agent.create"
ACTION_AGENT_UPDATE = "agent.update"
ACTION_AGENT_DELETE = "agent.delete"
ACTION_AGENT_ACCESS = "agent.access"
ACTION_AGENT_ROLLBACK = "agent.rollback"
ACTION_USER_CREATE = "user.create"
ACTION_USER_UPDATE = "user.update"
ACTION_USER_DELETE = "user.delete"
ACTION_PROJECT_CREATE = "project.create"
ACTION_PROJECT_UPDATE = "project.update"
ACTION_PROJECT_DELETE = "project.delete"
ACTION_CONFIG_CHANGE = "config.change"
ACTION_CONFIG_VERSION = "config.version"
ACTION_RBAC_DENIAL = "rbac.denial"
ACTION_LOGIN = "auth.login"
ACTION_LOGOUT = "auth.logout"
ACTION_RATE_LIMIT_CREATE = "rate_limit.create"
ACTION_RATE_LIMIT_UPDATE = "rate_limit.update"
ACTION_RATE_LIMIT_DELETE = "rate_limit.delete"
ACTION_SERVER_START = "server.start"
ACTION_SERVER_STOP = "server.stop"
ACTION_SERVER_RESTART = "server.restart"
ACTION_TOOL_INVOKE = "tool.invoke"
ACTION_TOOL_BLOCK = "tool.block"
ACTION_EVAL_RUN = "eval.run"
ACTION_EVAL_FAIL = "eval.fail"
ACTION_GUARDRAIL_BLOCK = "guardrail.block"
ACTION_GUARDRAIL_WARN = "guardrail.warn"
ACTION_GUARDRAIL_REDACT = "guardrail.redact"
ACTION_SECURITY_SCAN = "security.scan"
ACTION_SECURITY_FINDING = "security.finding"
ACTION_TEMPLATE_IMPORT = "template.import"
ACTION_MEMORY_BLOCK_CREATE = "memory_block.create"
ACTION_MEMORY_BLOCK_UPDATE = "memory_block.update"
ACTION_MEMORY_BLOCK_DELETE = "memory_block.delete"

RESOURCE_AGENT = "agent"
RESOURCE_USER = "user"
RESOURCE_PROJECT = "project"
RESOURCE_TOOL = "tool"
RESOURCE_CONFIG = "config"
RESOURCE_RATE_LIMIT = "rate_limit"
RESOURCE_SERVER = "server"
RESOURCE_EVAL = "eval"
RESOURCE_GUARDRAIL = "guardrail"
RESOURCE_SECURITY = "security"
RESOURCE_TEMPLATE = "template"
RESOURCE_MEMORY_BLOCK = "memory_block"
RESOURCE_AUTH = "auth"


class AuditRecord(BaseModel):
    """A single append-only audit log entry. CONCEPT:AU-OS.config.secrets-authentication"""

    id: str = ""
    actor: str = "system"
    action: str
    resource_type: str
    resource_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str = ""
    session_id: str = ""
    timestamp: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"audit:{self.action}:{self.timestamp}"


class PiiRedactionFilter(logging.Filter):
    """Logging filter to redact PII (like SSNs, Tax IDs, and emails) from log lines.

    Concept: observability-governance
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        from agent_utilities.security.guardrails import PiiSanitizer

        self.sanitizer = PiiSanitizer()

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitizer.sanitize_text(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                self.sanitizer.sanitize_text(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                k: self.sanitizer.sanitize(v) for k, v in record.args.items()
            }
        return True


class AuditLogger:
    """Append-only audit logger. CONCEPT:AU-OS.config.secrets-authentication

    Ported from MATE's audit_service.py. Key principles:
    1. Append-only — no update/delete on records
    2. Never-raise — errors logged and swallowed
    3. Retention — time-based cleanup via run_retention(days)
    """

    def __init__(self, kg_engine: Any = None, max_in_memory: int = 10_000) -> None:
        self._engine = kg_engine
        self._max_in_memory = max_in_memory
        self._records: list[AuditRecord] = []

    @property
    def records(self) -> list[AuditRecord]:
        return list(self._records)

    def log(
        self,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        details: dict[str, Any] | None = None,
        ip_address: str = "",
        session_id: str = "",
    ) -> AuditRecord | None:
        """Append audit entry. Never raises."""
        try:
            from agent_utilities.security.guardrails import PiiSanitizer

            sanitizer = PiiSanitizer()

            clean_details = sanitizer.sanitize_dict(details or {})
            clean_resource_id = (
                sanitizer.sanitize_text(resource_id) if resource_id else ""
            )
            clean_actor = sanitizer.sanitize_text(actor or "system")
            clean_session_id = sanitizer.sanitize_text(session_id or "")

            record = AuditRecord(
                actor=clean_actor,
                action=action,
                resource_type=resource_type,
                resource_id=clean_resource_id,
                details=clean_details,
                ip_address=ip_address,
                session_id=clean_session_id,
            )
            self._records.append(record)
            if len(self._records) > self._max_in_memory:
                self._records = self._records[-self._max_in_memory :]
            logger.debug(
                "Audit: actor=%s action=%s resource=%s/%s",
                clean_actor,
                action,
                resource_type,
                clean_resource_id,
            )
            return record
        except Exception as exc:
            logger.warning("Audit log write failed: %s", exc)
            return None

    def run_retention(self, days: int = 0) -> dict[str, Any]:
        """Delete records older than days. 0 = keep all."""
        if days <= 0:
            return {
                "deleted_count": 0,
                "retention_days": 0,
                "message": "Retention disabled",
            }
        cutoff = time.time() - (days * 86400)
        original = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        deleted = original - len(self._records)
        return {"deleted_count": deleted, "retention_days": days, "cutoff": cutoff}

    def query(
        self,
        actor: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        session_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        """Query with optional filters, newest first."""
        results = self._records
        if actor:
            results = [r for r in results if r.actor == actor]
        if action:
            results = [r for r in results if r.action == action]
        if resource_type:
            results = [r for r in results if r.resource_type == resource_type]
        if resource_id:
            results = [r for r in results if r.resource_id == resource_id]
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        results = sorted(results, key=lambda r: r.timestamp, reverse=True)
        return results[:limit]

    def export(self, format: str = "json") -> list[dict[str, Any]]:
        """Export all records as dicts."""
        return [r.model_dump() for r in self._records]

    def summary(self) -> dict[str, Any]:
        """Summary of audit state."""
        action_counts: dict[str, int] = {}
        for r in self._records:
            action_counts[r.action] = action_counts.get(r.action, 0) + 1
        return {
            "total_records": len(self._records),
            "action_counts": action_counts,
            "unique_actors": len({r.actor for r in self._records}),
        }
