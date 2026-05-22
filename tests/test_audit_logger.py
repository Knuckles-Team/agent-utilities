"""Tests for AuditLogger — Compliance Logging (CONCEPT:OS-5.1).

@pytest.mark.concept("OS-5.7")
"""

import time

import pytest

from agent_utilities.observability.audit_logger import (
    ACTION_AGENT_CREATE,
    ACTION_AGENT_UPDATE,
    ACTION_CONFIG_CHANGE,
    ACTION_RBAC_DENIAL,
    RESOURCE_AGENT,
    RESOURCE_CONFIG,
    AuditLogger,
    AuditRecord,
)


@pytest.fixture
def audit_logger() -> AuditLogger:
    return AuditLogger(max_in_memory=100)


# ---------------------------------------------------------------------------
# AuditRecord model
# ---------------------------------------------------------------------------


class TestAuditRecord:
    def test_auto_id_generation(self):
        r = AuditRecord(action="test.action", resource_type="test")
        assert r.id.startswith("audit:test.action:")

    def test_timestamp_auto(self):
        r = AuditRecord(action="test", resource_type="test")
        assert r.timestamp > 0

    def test_default_actor(self):
        r = AuditRecord(action="test", resource_type="test")
        assert r.actor == "system"


# ---------------------------------------------------------------------------
# Append-only semantics
# ---------------------------------------------------------------------------


class TestAppendOnly:
    def test_log_appends(self, audit_logger):
        r1 = audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT, "agent-1")
        r2 = audit_logger.log("admin", ACTION_AGENT_UPDATE, RESOURCE_AGENT, "agent-1")
        assert r1 is not None
        assert r2 is not None
        assert len(audit_logger.records) == 2

    def test_no_delete_method(self, audit_logger):
        """AuditLogger should not have delete/update methods."""
        assert not hasattr(audit_logger, "delete")
        assert not hasattr(audit_logger, "update")

    def test_records_are_copies(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        records = audit_logger.records
        records.clear()
        # Original should be unchanged
        assert len(audit_logger.records) == 1


# ---------------------------------------------------------------------------
# Never-raise semantics
# ---------------------------------------------------------------------------


class TestNeverRaise:
    def test_log_never_raises(self, audit_logger):
        """Even with bad data, log should not raise."""
        result = audit_logger.log(
            actor=None,
            action="test",
            resource_type="test",
        )
        assert result is not None

    def test_log_with_empty_actor(self, audit_logger):
        result = audit_logger.log("", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        assert result is not None
        assert result.actor == "system"


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------


class TestFIFOEviction:
    def test_eviction_at_limit(self):
        audit = AuditLogger(max_in_memory=5)
        for i in range(10):
            audit.log("user", ACTION_AGENT_CREATE, RESOURCE_AGENT, f"agent-{i}")
        assert len(audit.records) == 5
        # Should have the last 5
        ids = [r.resource_id for r in audit.records]
        assert "agent-5" in ids
        assert "agent-9" in ids


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    def test_retention_deletes_old(self, audit_logger):
        # Create a record with old timestamp
        old_record = AuditRecord(
            action=ACTION_AGENT_CREATE,
            resource_type=RESOURCE_AGENT,
            timestamp=time.time() - 86400 * 10,  # 10 days ago
        )
        audit_logger._records.append(old_record)
        audit_logger.log("admin", ACTION_AGENT_UPDATE, RESOURCE_AGENT)

        result = audit_logger.run_retention(days=5)
        assert result["deleted_count"] == 1
        assert len(audit_logger.records) == 1

    def test_retention_zero_keeps_all(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        result = audit_logger.run_retention(days=0)
        assert result["deleted_count"] == 0
        assert result["message"] == "Retention disabled"

    def test_retention_negative_keeps_all(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        result = audit_logger.run_retention(days=-1)
        assert result["deleted_count"] == 0


# ---------------------------------------------------------------------------
# Query filtering
# ---------------------------------------------------------------------------


class TestQuery:
    def test_filter_by_actor(self, audit_logger):
        audit_logger.log("alice", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        audit_logger.log("bob", ACTION_AGENT_UPDATE, RESOURCE_AGENT)
        results = audit_logger.query(actor="alice")
        assert len(results) == 1
        assert results[0].actor == "alice"

    def test_filter_by_action(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        audit_logger.log("admin", ACTION_CONFIG_CHANGE, RESOURCE_CONFIG)
        results = audit_logger.query(action=ACTION_CONFIG_CHANGE)
        assert len(results) == 1

    def test_filter_by_resource_type(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        audit_logger.log("admin", ACTION_CONFIG_CHANGE, RESOURCE_CONFIG)
        results = audit_logger.query(resource_type=RESOURCE_AGENT)
        assert len(results) == 1

    def test_newest_first(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT, "first")
        time.sleep(0.01)
        audit_logger.log("admin", ACTION_AGENT_UPDATE, RESOURCE_AGENT, "second")
        results = audit_logger.query()
        assert results[0].resource_id == "second"

    def test_limit(self, audit_logger):
        for i in range(10):
            audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT, f"{i}")
        results = audit_logger.query(limit=3)
        assert len(results) == 3

    def test_filter_by_session(self, audit_logger):
        audit_logger.log("a", ACTION_AGENT_CREATE, RESOURCE_AGENT, session_id="s1")
        audit_logger.log("b", ACTION_AGENT_CREATE, RESOURCE_AGENT, session_id="s2")
        results = audit_logger.query(session_id="s1")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Export & summary
# ---------------------------------------------------------------------------


class TestExportAndSummary:
    def test_export_json(self, audit_logger):
        audit_logger.log("admin", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        exported = audit_logger.export()
        assert len(exported) == 1
        assert "action" in exported[0]

    def test_summary(self, audit_logger):
        audit_logger.log("alice", ACTION_AGENT_CREATE, RESOURCE_AGENT)
        audit_logger.log("bob", ACTION_RBAC_DENIAL, RESOURCE_AGENT)
        s = audit_logger.summary()
        assert s["total_records"] == 2
        assert s["unique_actors"] == 2
        assert ACTION_AGENT_CREATE in s["action_counts"]


# ---------------------------------------------------------------------------
# Action constants coverage
# ---------------------------------------------------------------------------


class TestActionConstants:
    def test_all_actions_are_strings(self):
        """All ACTION_* constants should be dotted strings."""
        import agent_utilities.observability.audit_logger as mod

        actions = [
            v
            for k, v in vars(mod).items()
            if k.startswith("ACTION_") and isinstance(v, str)
        ]
        assert len(actions) >= 30
        for a in actions:
            assert "." in a, f"Action '{a}' should be dotted (e.g., 'agent.create')"
