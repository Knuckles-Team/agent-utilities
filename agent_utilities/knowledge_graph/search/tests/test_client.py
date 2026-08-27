"""OpenSearchClient / OpenSearchClientConfig (CA-24-W02/W03).

The ``verify_certs`` env-parsing case is a REGRESSION for a real bug this
lane's own live proof caught against the deployed CA-50 cluster
(2026-08-26): passing ``cast=bool`` to ``core._env.setting`` calls Python's
bare ``bool(...)`` on the raw string, and ``bool("false")`` is ``True`` (any
non-empty string is truthy) — so ``OPENSEARCH_VERIFY_CERTS=false`` silently
did NOT disable certificate verification, and ``mcp_reindex`` failed live
with a real ``SSLError: CERTIFICATE_VERIFY_FAILED``.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.search.client import OpenSearchClientConfig


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("true", True),
        ("1", True),
    ],
)
def test_from_env_verify_certs_parses_boolean_strings_correctly(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("OPENSEARCH_VERIFY_CERTS", raw)
    cfg = OpenSearchClientConfig.from_env()
    assert cfg.verify_certs is expected


def test_from_env_verify_certs_defaults_true_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENSEARCH_VERIFY_CERTS", raising=False)
    cfg = OpenSearchClientConfig.from_env()
    assert cfg.verify_certs is True


def test_from_env_reads_endpoint_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSEARCH_URL", "https://opensearch.example")
    monkeypatch.setenv("OPENSEARCH_USER", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "secret")
    cfg = OpenSearchClientConfig.from_env()
    assert cfg.endpoint == "https://opensearch.example"
    assert cfg.username == "admin"
    assert cfg.password == "secret"


def test_default_endpoint_is_generic_not_a_deployment_hostname() -> None:
    """No internal hostname is baked into shipped source (privacy gate) --
    the default is OpenSearch's own generic localhost:9200; a real
    deployment (e.g. CA-50) is supplied via OPENSEARCH_URL."""
    assert OpenSearchClientConfig().endpoint == "http://localhost:9200"
