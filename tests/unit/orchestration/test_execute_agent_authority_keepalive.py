"""D-SNV-5 follow-up: ``Orchestrator.execute_agent`` renews a renewable
session's authority for the WHOLE delegation, not just while an MCP tool
dispatch is on the stack.

Root cause this closes: the original D-SNV-5 fix wired the background
keepalive into ``agent_utilities.mcp.kg_server._execute_tool``'s ``_guarded()``
— the MCP tool-dispatch path only. But every delegation entrypoint (MCP
``graph_orchestrate``, the agent-webui/REST gateway, the messaging router,
the autonomous ``agent_dispatch_worker``, ``org_runtime``, a governed dynamic
workflow, and the parallel engine) converges on ONE function,
``Orchestrator.execute_agent`` (``agent_utilities/orchestration/manager.py``).
At least six real callers reach that function WITHOUT ever going through
``_execute_tool``, so a long delegation started from any of them still died
mid-flight with ``SessionExpiredError``. The fix moves (additionally opens)
the SAME ``authority_keepalive_scope`` primitive around ``execute_agent``
itself so every surface inherits it.

These tests never touch ``_execute_tool``/MCP dispatch at all — they call
``execute_agent`` exactly the way the messaging router / dispatch worker /
org_runtime / governed dynamic workflow / parallel engine do, proving the
specific gap the deployment probe (``scripts/delegation_probe.py``, stage 7)
found is closed for a non-MCP entrypoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionExpiredError,
    current_session,
    use_session,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.orchestration.manager import Orchestrator
from agent_utilities.security.brain_context import (
    ActorContext,
    CredentialLease,
    use_actor,
)


def _verified_session(actor_id: str = "non-mcp-caller") -> GraphSession:
    """A server-minted, renewable session — the same shape the messaging
    router / dispatch worker / org_runtime / parallel engine already run
    under via the ambient process/network authority, never a caller-presented
    bearer JWT (which carries no ``credential_lease`` and is covered by the
    separate "never proactively renewed" tests in
    ``tests/unit/mcp/test_graphos_bootstrap_isolation.py``)."""
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.SYSTEM,
        roles=("system",),
        tenant_id="runtime-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        policy_version="current",
        audience="graph-runtime",
    )


def _expiring_session(actor_id: str, lease: CredentialLease) -> GraphSession:
    session = _verified_session(actor_id)
    return replace(
        session,
        actor=replace(
            session.actor,
            credential_expires_at=lease.expires_at,
            credential_lease=lease,
        ),
    )


@pytest.mark.asyncio
async def test_execute_agent_renews_authority_for_a_non_mcp_entrypoint() -> None:
    from agent_utilities.mcp import kg_server

    lease = CredentialLease(int(time.time()) + 2)
    session = _expiring_session("non-mcp-caller", lease)

    def renew(_session: GraphSession) -> GraphSession:
        lease.renew(int(time.time()) + 300)
        return _session

    deep_check_raised: list[BaseException] = []

    async def fake_run_agent(**_kwargs: Any) -> str:
        # Outlive the ORIGINAL lease's 2s expiry — a real long tool loop, with
        # nothing but execute_agent's own keepalive scope standing between
        # this and the SessionExpiredError the probe hit at 195.50s.
        await asyncio.sleep(2.5)
        # The real deep engine boundary check that actually raised in
        # production (GraphSession.ensure_authority_current at every engine
        # call, e.g. graph_compute.py::_invoke_at).
        try:
            current_session().ensure_authority_current()
        except SessionExpiredError as exc:
            deep_check_raised.append(exc)
            raise
        return "ok"

    orchestrator = Orchestrator(engine=object())

    with (
        use_session(session),
        use_actor(session.actor),
        patch.object(
            kg_server, "_refresh_process_authority", side_effect=renew
        ) as mock_renew,
        patch(
            "agent_utilities.orchestration.manager.run_agent",
            side_effect=fake_run_agent,
        ),
    ):
        result = await orchestrator.execute_agent(
            agent_name="agent-utilities-expert",
            task="git status && git log -5",
        )

    assert result == "ok"
    assert not deep_check_raised, (
        "execute_agent let a non-MCP-entrypoint delegation's session expire "
        "mid-flight instead of renewing it"
    )
    mock_renew.assert_called()
    assert lease.expires_at > int(time.time()) + 100


@pytest.mark.asyncio
async def test_execute_agent_would_fail_without_the_keepalive_scope() -> None:
    """Control proving the test above is not passing "by luck": with
    ``authority_keepalive_scope`` replaced by a no-op (simulating the state of
    the code before this fix), the identical scenario genuinely raises
    ``SessionExpiredError``."""
    from agent_utilities.mcp import kg_server

    @contextlib.asynccontextmanager
    async def _noop_scope(_session: Any = None):
        yield

    lease = CredentialLease(int(time.time()) + 2)
    session = _expiring_session("non-mcp-caller-control", lease)

    async def fake_run_agent(**_kwargs: Any) -> str:
        await asyncio.sleep(2.5)
        current_session().ensure_authority_current()
        return "ok"

    orchestrator = Orchestrator(engine=object())

    with (
        use_session(session),
        use_actor(session.actor),
        patch.object(kg_server, "authority_keepalive_scope", _noop_scope),
        patch.object(kg_server, "_refresh_process_authority") as mock_renew,
        patch(
            "agent_utilities.orchestration.manager.run_agent",
            side_effect=fake_run_agent,
        ),
        pytest.raises(SessionExpiredError),
    ):
        await orchestrator.execute_agent(
            agent_name="agent-utilities-expert",
            task="git status && git log -5",
        )

    mock_renew.assert_not_called()


@pytest.mark.asyncio
async def test_execute_agent_never_renews_a_caller_presented_bearer_session() -> None:
    """Security property carried over unchanged: a caller-presented bearer JWT
    (no server-held ``credential_lease``) must stay exactly as fail-closed on
    this entrypoint as it already was on the MCP dispatch path — this must
    never widen authority the server does not hold."""
    from agent_utilities.mcp import kg_server

    session = _verified_session("bearer-caller-non-mcp")
    session = replace(
        session,
        actor=replace(session.actor, credential_expires_at=int(time.time()) + 300),
    )
    assert session.actor.credential_lease is None

    async def fake_run_agent(**_kwargs: Any) -> str:
        return "ok"

    orchestrator = Orchestrator(engine=object())

    with (
        use_session(session),
        use_actor(session.actor),
        patch.object(kg_server, "_refresh_process_authority") as mock_renew,
        patch(
            "agent_utilities.orchestration.manager.run_agent",
            side_effect=fake_run_agent,
        ),
    ):
        result = await orchestrator.execute_agent(
            agent_name="agent-utilities-expert",
            task="git status && git log -5",
        )

    assert result == "ok"
    mock_renew.assert_not_called()
