"""Regression: PostgreSQL lock-contention detection must not false-match identifiers.

A bare ``"lock" in str(exc)`` substring test misclassified a schema-drift error on
the ``idea_block`` table ("idea_b·lock·") as lock contention, so the write retried
the lock path and never reached the auto-DDL self-heal — every ``idea_block`` write
was silently dropped (CONCEPT:AU-KG.query.vendor-agnostic-traversal).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
)


def _err(msg: str, sqlstate: str | None = None) -> Exception:
    e = Exception(msg)
    e.sqlstate = sqlstate  # type: ignore[attr-defined]
    return e


_is_lock = PostgreSQLBackend._is_lock_contention


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
def test_idea_block_missing_column_is_not_lock():
    # The bug: "idea_block" contains the substring "lock".
    exc = _err(
        'column "label" of relation "idea_block" does not exist', sqlstate="42703"
    )
    assert _is_lock(exc) is False


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
@pytest.mark.parametrize(
    "msg",
    [
        'column "x" of relation "y" does not exist',
        'relation "deadlock_table" does not exist',  # name contains "deadlock"
        'column "unlock_ts" of relation "events" does not exist',  # contains "lock"
    ],
)
def test_schema_errors_are_not_lock(msg):
    assert _is_lock(_err(msg, sqlstate="42703")) is False


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
@pytest.mark.parametrize(
    ("msg", "sqlstate"),
    [
        ("deadlock detected", "40P01"),
        ("could not obtain lock on relation", "55P03"),
        ("canceling statement due to lock timeout", "55P03"),
        ("lock not available", "55P03"),
        # SQLSTATE alone classifies it even if the message lacks the phrase:
        ("some opaque driver wrapper", "40001"),
    ],
)
def test_real_lock_errors_are_detected(msg, sqlstate):
    assert _is_lock(_err(msg, sqlstate=sqlstate)) is True
