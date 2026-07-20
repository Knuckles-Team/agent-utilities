"""Opt-in live validation for an operator-configured Eunomia service."""

import pytest
from eunomia_core import schemas

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_runtime_configured_eunomia_decision_point():
    from agent_utilities.core.config import setting

    endpoint = setting("EUNOMIA_TEST_URL")
    if not endpoint:
        pytest.skip("EUNOMIA_TEST_URL is not configured")

    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    api_key_ref = (
        "env://EUNOMIA_TEST_API_KEY"
        if setting("EUNOMIA_TEST_API_KEY")
        else None
    )
    middleware = create_eunomia_middleware(
        use_remote_eunomia=True,
        eunomia_endpoint=endpoint,
        api_key_ref=api_key_ref,
        require_verified_principal=True,
    )
    response = await middleware._eunomia.check(
        schemas.CheckRequest(
            principal=schemas.PrincipalCheck(
                uri="agent:integration-probe", attributes={"jwt_verified": True}
            ),
            resource=schemas.ResourceCheck(
                uri="mcp:tool:integration-probe",
                attributes={"component_type": "tool", "name": "integration-probe"},
            ),
            action="list",
        )
    )
    assert isinstance(response, schemas.CheckResponse)
