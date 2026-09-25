"""Agent configuration uses the connector SDK's outbound-host contract."""

import pytest

from agent_utilities.core.config import AgentConfig


def test_source_host_exceptions_are_normalized() -> None:
    config = AgentConfig(
        SOURCE_HTTP_ALLOWED_PRIVATE_HOSTS=["Example.Invalid.", "example.invalid"]
    )
    assert config.source_http_allowed_private_hosts == ["example.invalid"]


@pytest.mark.parametrize("host", ["*.example.invalid", "bad..invalid", "bäd.invalid"])
def test_source_host_exceptions_reject_non_exact_names(host: str) -> None:
    with pytest.raises(ValueError, match="HTTP host allow-lists"):
        AgentConfig(SOURCE_HTTP_ALLOWED_PRIVATE_HOSTS=[host])
