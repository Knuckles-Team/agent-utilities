"""Tests for agent_utilities.http.redaction (CONCEPT:ECO-4.35).

Pins the promoted security_sanitizer patterns: scheme credentials, DSN
passwords, well-known token shapes, secret assignments, literal secrets,
and the LogRedactor logging filter.
"""

from __future__ import annotations

import logging

from agent_utilities.http.redaction import REDACTED, LogRedactor, redact_text

# --------------------------------------------------------------------------- #
# redact_text patterns
# --------------------------------------------------------------------------- #


def test_redacts_bearer_and_ssws_scheme_tokens():
    text = (
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig and SSWS 00abcDEF123456"
    )
    redacted = redact_text(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "00abcDEF123456" not in redacted
    assert f"Bearer {REDACTED}" in redacted
    assert f"SSWS {REDACTED}" in redacted


def test_redacts_dsn_credentials_keeping_structure():
    text = "connecting to postgresql://agent:hunter2pass@db.arpa:5432/kg"
    redacted = redact_text(text)
    assert "hunter2pass" not in redacted
    assert f"postgresql://agent:{REDACTED}@db.arpa:5432/kg" in redacted


def test_redacts_known_token_shapes():
    samples = [
        "ghp_" + "a1B2" * 10,
        "github_pat_" + "x9Y8" * 8,
        "glpat-" + "tok3nV4lue123",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "sk-" + "proj1234567890abcdefghij",
        "xoxb-" + "1234567890-abcdef",
    ]
    for sample in samples:
        assert sample not in redact_text(f"found {sample} in output"), sample


def test_redacts_secret_assignments():
    redacted = redact_text('api_key="superSecretValue99" password=topsecretvalue')
    assert "superSecretValue99" not in redacted
    assert "topsecretvalue" not in redacted


def test_redacts_literal_secrets():
    redacted = redact_text("the value is s3cr3t-v@lue!", secrets=["s3cr3t-v@lue!"])
    assert "s3cr3t-v@lue!" not in redacted
    assert REDACTED in redacted


def test_plain_text_passes_through_unchanged():
    text = "GET /api/stacks returned 200 with 14 items"
    assert redact_text(text) == text


# --------------------------------------------------------------------------- #
# LogRedactor filter
# --------------------------------------------------------------------------- #


def _capture_logger(name: str, redactor: LogRedactor) -> tuple[logging.Logger, list]:
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [_Collector()]
    logger.addFilter(redactor)
    logger.propagate = False
    return logger, records


def test_log_redactor_scrubs_records_including_args():
    redactor = LogRedactor()
    logger, records = _capture_logger("test.http.redactor", redactor)
    logger.info("auth header was Bearer %s", "abcdef123456token")
    assert len(records) == 1
    assert "abcdef123456token" not in records[0].getMessage()


def test_log_redactor_registered_secrets_and_provider():
    redactor = LogRedactor(secrets=["literal-one"])
    redactor.add_secret("literal-two")
    redactor.add_secrets_provider(lambda: ["from-provider"])
    logger, records = _capture_logger("test.http.redactor2", redactor)
    logger.info("got literal-one then literal-two then from-provider")
    message = records[0].getMessage()
    for secret in ("literal-one", "literal-two", "from-provider"):
        assert secret not in message
    assert message.count(REDACTED) == 3


def test_log_redactor_provider_errors_never_break_logging():
    def _broken():
        raise RuntimeError("provider exploded")

    redactor = LogRedactor(secrets_provider=_broken)
    logger, records = _capture_logger("test.http.redactor3", redactor)
    logger.info("still logs fine")
    assert records[0].getMessage() == "still logs fine"
