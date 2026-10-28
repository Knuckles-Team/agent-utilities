"""Keep the certification authority owner decision and its source evidence."""

import re
from pathlib import Path

import pytest

PLAN = Path(__file__).resolve().parents[2] / "specs/au-boundary-deconstruction/plan.md"


@pytest.mark.spec("AU-BOUNDARY-R002.3.1")
def test_certification_owner_decision_names_the_pinned_owner_and_contract() -> None:
    plan = PLAN.read_text()
    section = plan.split("\n### AU-BOUNDARY-R002.3.1:", 1)[1].split("\n### ", 1)[0]
    source = re.search(
        r"https://github\.com/Knuckles-Team/arbora/blob/([0-9a-f]{40})/arbora/certification/oidc\.py",
        section,
    )
    assert source is not None
    assert "ARBORA-HOST-R027.1" in section
    assert (
        f"/blob/{source.group(1)}/specs/host-composition-boundary/requirements.md"
        in section
    )
    for symbol in (
        "EphemeralLoopbackOidcAuthority",
        "AUTHORITY_MODE",
        "DEFAULT_TOKEN_TTL_SECONDS",
        "validated_token_ttl_seconds",
    ):
        assert f"`{symbol}`" in section
    assert "https://github.com/Knuckles-Team/arbora/pull/526" in section


@pytest.mark.spec("AU-BOUNDARY-R010.3")
def test_value_helper_owner_decision_names_pinned_sdk_module() -> None:
    plan = PLAN.read_text()
    section = plan.split("\n### AU-BOUNDARY-R010.3:", 1)[1].split("\n### ", 1)[0]
    source = re.search(
        r"https://github\.com/Knuckles-Team/scion-sdk/blob/([0-9a-f]{40})/agent_connector_sdk/utilities\.py",
        section,
    )
    assert source is not None
    assert (
        f"/blob/{source.group(1)}/specs/SDK-CONNECTOR-CONTROL/requirements.md"
        in section
    )
    assert "SDK-CONNECTOR-CONTROL-R780.5" in section
    for symbol in ("to_boolean", "to_integer", "to_list", "to_dict"):
        assert f"`{symbol}`" in section
    assert "does not establish `AU-BOUNDARY-R010.4` parity" in section
