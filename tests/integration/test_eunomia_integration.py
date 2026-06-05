import pytest
from eunomia_mcp import create_eunomia_middleware
from eunomia_sdk import EunomiaClient

# Every test here talks to a real remote Eunomia server (http://eunomia.arpa) — a live
# external service, absent in CI/sandbox — so exclude from the default `-m "not live"` run.
pytestmark = pytest.mark.live


def test_eunomia_client_connection():
    """Verify that we can connect to the running Eunomia remote server and list policies."""
    client = EunomiaClient(endpoint="http://eunomia.arpa")
    policies = client.get_policies()
    assert len(policies) > 0, "No policies found in remote Eunomia server"

    # Check that at least one policy matches the standard naming convention
    mcp_policies = [p for p in policies if p.name.endswith("-mcp-policy")]
    assert len(mcp_policies) > 0, "No MCP policies found on Eunomia server"


def test_eunomia_middleware_creation():
    """Verify that the Eunomia middleware can be instantiated with remote config."""
    middleware = create_eunomia_middleware(
        policy_file=None,
        use_remote_eunomia=True,
        eunomia_endpoint="http://eunomia.arpa",
    )
    assert middleware is not None, "Failed to create remote Eunomia middleware"
