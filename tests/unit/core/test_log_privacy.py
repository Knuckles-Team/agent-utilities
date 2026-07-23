"""Regression tests for the process-wide logging privacy boundary."""

from __future__ import annotations

import logging

from agent_utilities.core.log_privacy import install_log_privacy_boundary


def test_log_record_redacts_runtime_locations_and_caller() -> None:
    install_log_privacy_boundary()
    record = logging.getLogger("agent_utilities.privacy-test").makeRecord(
        "agent_utilities.privacy-test",
        logging.ERROR,
        __file__,
        1,
        "failed endpoint=%s path=%s caller=%s",
        (
            "https://service.example.test:8443/private",
            "/runtime/private/config.json",
            "person@example.test",
        ),
        None,
    )

    message = record.getMessage()
    assert "service.example" not in message
    assert "runtime/private" not in message
    assert "person@example" not in message
    assert "<endpoint>" in message
    assert "<path>" in message
    assert "<caller>" in message


def test_log_record_drops_traceback_and_sanitizes_exception_message() -> None:
    # CONCEPT: the boundary sanitizes an interpolated exception's MESSAGE rather
    # than collapsing it to the bare type name — see the design note in
    # ``_sanitize_value`` (log_privacy.py): collapsing to the class name alone
    # destroyed the one piece of information the log existed to carry (e.g.
    # every ``logger.error("...(%s)", exc)`` became indistinguishable
    # "RuntimeError" noise). The message is still redacted by the same text
    # sanitizer as any other string (endpoints/paths/emails), and the
    # traceback/stack rendering (which embeds host filesystem paths) is
    # dropped entirely.
    install_log_privacy_boundary()
    error = RuntimeError("secret endpoint https://private.example.test")
    record = logging.getLogger("agent_utilities.privacy-test").makeRecord(
        "agent_utilities.privacy-test",
        logging.ERROR,
        __file__,
        1,
        "operation failed (%s)",
        (error,),
        (RuntimeError, error, None),
    )

    assert record.getMessage() == "operation failed (RuntimeError: secret endpoint <endpoint>)"
    assert record.exc_info is None
    assert record.stack_info is None
