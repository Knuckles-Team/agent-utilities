"""Deployment profile A — "tiny": zero external dependencies (Raspberry Pi 3).

The headline guarantee: agent-utilities + epistemic-graph cold-boot and *serve*
the Knowledge Graph + local OWL over the gateway with **no external services** —
no Kafka, no Postgres, no remote SPARQL/OWL server — using the self-contained
epistemic-graph authority and a local owlready2 reasoner.

This module is deliberately **not** ``@pytest.mark.live`` — it must pass in the
default PR suite as the continuously-enforced zero-dep contract. (The KG/engine
tests skip when the local epistemic-graph engine isn't running, e.g. a polyrepo
CI without the Rust source; the footprint guard always runs.)

Profile B (the full enterprise stack) lives in ``test_profile_enterprise_full.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

pytestmark = pytest.mark.integration

# External-service client libraries that the tiny profile must NEVER pull in.
# Their presence in a cold import means a heavyweight dependency leaked into the
# few-MB Pi-3 footprint.
_FORBIDDEN_DRIVERS = (
    "aiokafka",
    "confluent_kafka",
    "psycopg",
    "neo4j",
    "falkordb",
    "pystardog",
)


@pytest.fixture(autouse=True)
def _tiny_profile_env(monkeypatch, tmp_path):
    """Pin the process to the tiny (zero-dep) deployment profile for this module."""
    # The engine is the whole database for tiny; pin the host role for parity with
    # the singleton-host daemon path.
    monkeypatch.setenv("KG_DAEMON_ROLE", "host")
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "agent-data"))
    monkeypatch.setenv("OWL_BACKEND", "owlready2")
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "sqlite")
    for ext in (
        "GRAPH_DB_CONNECTION_PROFILE_REF",
        "STATE_DB_URI",
        "KAFKA_BOOTSTRAP_SERVERS",
    ):
        monkeypatch.delenv(ext, raising=False)
    # Rebuild engine/backend under the tiny env (root conftest also resets these).
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)


def test_cold_import_pulls_no_external_service_drivers():
    """The served stack imports with zero external-service client libraries.

    Runs in a clean subprocess (a shared pytest session would have other tests'
    imports polluting ``sys.modules``) so this is a true cold-boot footprint check.
    """
    probe = (
        "import os, sys, json\n"
        "os.environ.pop('GRAPH_DB_CONNECTION_PROFILE_REF', None)\n"
        "os.environ.pop('STATE_DB_URI', None)\n"
        "os.environ.pop('KAFKA_BOOTSTRAP_SERVERS', None)\n"
        # Import the served surface: the gateway route layer + the MCP/engine entry.
        "import agent_utilities.gateway.graph_api\n"
        "import agent_utilities.mcp.kg_server\n"
        "import agent_utilities.knowledge_graph.core.engine\n"
        f"forbidden = {list(_FORBIDDEN_DRIVERS)!r}\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "print(json.dumps(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"cold import failed:\n{result.stderr}"
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"tiny profile leaked external-service drivers: {leaked}"


def test_local_owl_reasoner_runs_in_process():
    """OWL runs locally (owlready2), not against a remote Stardog/Fuseki server."""
    pytest.importorskip("owlready2")
    from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
    from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
        Owlready2Backend,
    )

    owl = create_owl_backend()  # OWL_BACKEND=owlready2 from the profile env
    try:
        assert isinstance(owl, Owlready2Backend), (
            "tiny profile must use a local OWL reasoner"
        )
        # An in-process reasoner exposes live stats without any network call.
        stats = owl.get_stats()
        assert isinstance(stats, dict)
    finally:
        owl.close()


def test_tiny_profile_serves_kg_over_gateway_with_zero_containers(monkeypatch):
    """Write + read the KG through the local gateway REST surface, no containers."""
    if not os.environ.get("GRAPH_SERVICE_ENDPOINTS"):
        pytest.skip(
            "local epistemic-graph engine not running (GRAPH_SERVICE_ENDPOINTS unset)"
        )

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from agent_utilities.gateway.graph_api import register_graph_routes
    from agent_utilities.mcp import kg_server
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    kg_server.ensure_tools_registered()

    app = FastAPI()
    register_graph_routes(app, prefix="/api")

    # register_graph_routes always mounts ActorIdentityMiddleware
    # (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) -- "zero
    # external dependencies" (no Kafka/Postgres/Stardog) is an orthogonal
    # guarantee from "no identity required"; every route still needs a
    # verified Bearer identity. Mint one the same way
    # tests/integration/core/test_security_server.py's secure_client does,
    # without a real JWKS round-trip.
    from _test_engine import TEST_TENANT

    actor = ActorContext(
        actor_id="tiny-profile-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read", "kg:write"),
        # The shared epistemic-graph engine's placement catalog only knows
        # graphs the test-engine fixtures actually provisioned it for --
        # the same TEST_TENANT every other conftest fixture uses, not an
        # arbitrary string ("request context tenant does not match graph
        # tenant" / PlacementAuthorityError otherwise).
        tenant_id=TEST_TENANT,
        authenticated=True,
    )

    async def _actor_from_bearer_token(token: str) -> ActorContext:
        if token != "valid-token":
            raise PermissionError("invalid token")
        return actor

    import agent_utilities.core.config as config_mod
    import agent_utilities.security.request_identity as request_identity

    monkeypatch.setattr(
        config_mod.config, "auth_jwt_jwks_uri", "https://identity.example.test/jwks"
    )
    # mint_graph_session fails closed (PermissionError -> 403 "Verified
    # tenant claim required", the generic message the middleware maps every
    # PermissionError from mint_graph_session to) unless BOTH an audience and
    # a policy revision are configured -- neither has a test default. The
    # audience must additionally match what the shared epistemic-graph
    # engine this suite runs against was started with -- the same
    # TEST_AUDIENCE every other conftest fixture (engine_graph et al.) uses
    # -- or the native engine's placement layer rejects the request context
    # ("request context audience does not match deployment").
    from _test_engine import TEST_AUDIENCE, TEST_POLICY_VERSION

    monkeypatch.setattr(config_mod.config, "auth_jwt_audience", TEST_AUDIENCE)
    # Likewise, the active policy version must match what the shared engine
    # was started with, not an arbitrary string ("request context policy
    # version is not active" otherwise).
    monkeypatch.setattr(config_mod.config, "kg_policy_version", TEST_POLICY_VERSION)
    monkeypatch.setattr(
        request_identity, "actor_from_bearer_token", _actor_from_bearer_token
    )
    # mint_graph_session is left real (unlike test_security_server.py's
    # secure_client, which never exercises a live query_cypher round trip):
    # it derives the actual routing graph from actor.tenant_id, which the
    # write/read below need to land on the SAME graph the engine actually
    # serves. A hand-rolled static GraphSession here left the read's engine
    # resolution with no bound graph (AttributeError: 'NoneType' object has
    # no attribute 'query_cypher').
    auth_headers = {"Authorization": "Bearer valid-token"}

    # Force the process-active engine to exist (and be registered via
    # IntelligenceGraphEngine.set_active) BEFORE any request. The connection
    # registry's "default" target resolves ONLY to
    # IntelligenceGraphEngine.get_active() (get_connection_registry's own
    # docstring: "registry construction never creates or seeds a second
    # engine"), and _tiny_profile_env's autouse fixture just reset it to
    # None -- without an explicit warm-up here, resolution order between
    # the write and read requests is not guaranteed to both observe it.
    # Must run under the SAME actor as the requests below: with no ambient
    # actor, resolve_routing_graph(None) binds the engine to the default/
    # commons graph, while each authenticated request's minted GraphSession
    # binds to actor.tenant_id's OWN graph -- a mismatch the native engine's
    # placement layer rejects ("request context tenant does not match graph
    # tenant").
    from agent_utilities.security.brain_context import use_actor

    with use_actor(actor):
        kg_server._get_engine()

    node_id = f"tiny:{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        write = client.post(
            "/api/graph/write",
            headers=auth_headers,
            json={
                "action": "add_node",
                "node_id": node_id,
                "node_type": "TinyProfileNode",
                "properties": json.dumps({"served": True, "profile": "tiny"}),
            },
        )
        assert write.status_code == 200, write.text
        assert write.json().get("status") == "success", write.text

        # KNOWN ENVIRONMENTAL BLOCKER (not fixed by this pass): the read below
        # can still fail with PlacementAuthorityError ("ACCESS_DENIED:
        # verified request context lacks required scope
        # 'admin:cluster-read'") when GRAPH_SERVICE_ENDPOINTS resolves to a
        # multi-endpoint/route-configured engine (graph_compute.py only skips
        # placement resolution when `self._route_config is None or not
        # self._route_endpoints`). Per request_identity.py's own D-SP-1
        # docstring, that engine-side capability "no JWT claim can satisfy" --
        # it is enforced twice inside the compiled epistemic-graph engine
        # (dispatch.rs/access.rs), not resolvable from any Python-side
        # actor/session field. A genuinely single-node/tiny engine instance
        # (no route config, e.g. a fresh _session_engine-style ephemeral
        # engine with GRAPH_SERVICE_ENDPOINTS pointed at exactly one socket)
        # skips placement entirely and this passes; a shared multi-lane test
        # daemon configured with real route/placement config does not.
        read = client.post(
            "/api/graph/query",
            headers=auth_headers,
            json={
                "cypher": "MATCH (n:TinyProfileNode) WHERE n.id = $id RETURN n.id AS id",
                # graph_query expects ``params`` as a JSON string, mirroring the MCP tool.
                "params": json.dumps({"id": node_id}),
            },
        )
        assert read.status_code == 200, read.text
        payload = read.json()
        assert payload.get("status") == "success", payload
        assert node_id in json.dumps(payload["result"]), payload

    # The serving engine is backed by the zero-dep tiered stack — never Postgres.
    engine = kg_server._get_engine()
    assert type(engine.backend).__name__ != "PostgreSQLBackend"
