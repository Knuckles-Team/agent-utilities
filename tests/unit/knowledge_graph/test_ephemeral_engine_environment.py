"""Security contract for the real ephemeral test engine."""

from __future__ import annotations

import json

from tests._test_engine import (
    TEST_AGENT_ID,
    TEST_AUDIENCE,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_TENANT,
    strict_server_env,
)


def test_strict_server_env_retains_auth_and_only_opts_out_of_oidc() -> None:
    """The local fixture must not weaken transport or request authentication."""

    auth_secret = "synthetic-engine-auth-secret"
    state_dir = "/synthetic/security-state"

    env = strict_server_env(state_dir, auth_secret=auth_secret)

    assert env == {
        "GRAPH_SERVICE_AUTH_SECRET": auth_secret,
        "EPISTEMIC_GRAPH_AUDIENCE": TEST_AUDIENCE,
        "EPISTEMIC_GRAPH_TENANT": TEST_TENANT,
        "EPISTEMIC_GRAPH_POLICY_VERSION": TEST_POLICY_VERSION,
        "EPISTEMIC_GRAPH_REQUIRE_OIDC": "false",
        "EPISTEMIC_GRAPH_SECURITY_STATE_DIR": state_dir,
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON": json.dumps(
            {TEST_AGENT_ID: TEST_SIGNER_KEY}
        ),
    }
    assert json.loads(env["EPISTEMIC_GRAPH_SIGNER_KEYS_JSON"]) == {
        TEST_AGENT_ID: TEST_SIGNER_KEY
    }
    assert "EPISTEMIC_GRAPH_ALLOW_INSECURE" not in env
