from __future__ import annotations

"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

"""Coverage push for small tool modules.

Covers:
  * tool_guard.py: is_sensitive_tool, is_safe_tool, build_sensitive_tool_names,
    flag_mcp_tool_definitions, apply_tool_guard_approvals
  * middlewares.py: UserTokenMiddleware, JWTClaimsLoggingMiddleware
  * tools/memory_tools.py: read_agents_md
  * tools/team_tools.py: spawn_team, assign_team_task, message_teammate,
    list_team_tasks
  * tools/style_tools.py: set_output_style, list_output_styles
  * tools/onboarding_tools.py (partial)
  * tools/git_tools.py (partial)
  * sdd/orchestrator.py: SDDOrchestrator class basics
"""


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# tool_guard
# ---------------------------------------------------------------------------


def test_is_safe_tool_read() -> None:
    """Tools matching safe patterns are identified."""
    from agent_utilities.security.tool_guard import is_safe_tool

    assert is_safe_tool("read_file") is True
    assert is_safe_tool("list_directory") is True
    assert is_safe_tool("get_contents") is True
    assert is_safe_tool("search_code") is True
    assert is_safe_tool("inspect_server") is True
    assert is_safe_tool("view_file") is True
    assert is_safe_tool("show_tree") is True
    assert is_safe_tool("describe_resource") is True


def test_is_safe_tool_non_safe() -> None:
    """Non-safe names return False."""
    from agent_utilities.security.tool_guard import is_safe_tool

    assert is_safe_tool("write_file") is False
    assert is_safe_tool("delete_item") is False


def test_is_sensitive_tool_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """In strict mode, everything non-safe is sensitive."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "strict")
    assert tg.is_sensitive_tool("write_file") is True
    assert tg.is_sensitive_tool("read_file") is False


def test_is_sensitive_tool_normal_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal mode uses pattern matching."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    # Set up custom patterns
    monkeypatch.setattr(
        "agent_utilities.core.config.SENSITIVE_TOOL_PATTERNS", [r"^dangerous.*"]
    )
    assert tg.is_sensitive_tool("dangerous_op") is True
    assert tg.is_sensitive_tool("safe_op") is False


def test_build_sensitive_tool_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_sensitive_tool_names pulls from discovery registry."""
    import agent_utilities.security.tool_guard as tg
    from agent_utilities.models import MCPAgentRegistryModel, MCPToolInfo

    registry = MCPAgentRegistryModel(
        tools=[
            MCPToolInfo(
                name="DangerousTool",
                description="",
                mcp_server="s",
                requires_approval=True,
            ),
            MCPToolInfo(
                name="SafeTool",
                description="",
                mcp_server="s",
                requires_approval=False,
            ),
        ]
    )
    with patch(
        "agent_utilities.core.config.get_discovery_registry",
        return_value=registry,
    ):
        result = tg.build_sensitive_tool_names()
    assert "dangeroustool" in result
    assert "safetool" not in result


def test_build_sensitive_tool_names_handles_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception in registry fetch returns empty set."""
    import agent_utilities.security.tool_guard as tg

    with patch(
        "agent_utilities.core.config.get_discovery_registry",
        side_effect=RuntimeError("db down"),
    ):
        result = tg.build_sensitive_tool_names()
    assert result == set()


def test_flag_mcp_tool_definitions_cannot_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a stale off value cannot bypass mandatory MCP identity policy."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "off")
    toolsets = [MagicMock(spec=["list_tools"])]
    with pytest.raises(PermissionError, match="identity policy is required"):
        tg.flag_mcp_tool_definitions(
            toolsets,
            permissions_kernel=None,
            agent_identity=None,
        )


def test_flag_mcp_tool_definitions_wraps_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP-like toolsets are wrapped in ApprovalRequiredToolset."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    mcp_ts.list_tools = MagicMock()
    other_ts = MagicMock(spec=[])
    kernel = MagicMock()
    kernel.authorize_tool.return_value = "allow"

    result = tg.flag_mcp_tool_definitions(
        [mcp_ts, other_ts],
        permissions_kernel=kernel,
        agent_identity=object(),
    )
    # mcp_ts should be wrapped, other_ts passes through
    assert len(result) == 2
    assert result[1] is other_ts


def test_mcp_authorization_receives_declared_required_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    kernel = MagicMock()
    kernel.authorize_tool.return_value = "allow"
    identity = object()
    wrapped = tg.flag_mcp_tool_definitions(
        [mcp_ts], permissions_kernel=kernel, agent_identity=identity
    )[0]

    assert (
        wrapped.approval_required_func(
            MagicMock(),
            SimpleNamespace(
                name="knowledge_query",
                metadata={"required_capability": "kg.read"},
            ),
            {},
        )
        is False
    )
    kernel.authorize_tool.assert_called_once_with(
        identity,
        "knowledge_query",
        required_capability="kg.read",
    )


def test_flag_mcp_tool_definitions_requires_identity_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP toolset cannot bind without the current identity contract."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    with pytest.raises(PermissionError, match="identity policy is required"):
        tg.flag_mcp_tool_definitions(
            [mcp_ts], permissions_kernel=None, agent_identity=None
        )


def test_flag_mcp_tool_definitions_hard_deny_is_not_approvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RBAC DENY must never be downgraded to a human approval request."""
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    kernel = MagicMock()
    kernel.authorize_tool.return_value = "deny"
    wrapped = tg.flag_mcp_tool_definitions(
        [mcp_ts], permissions_kernel=kernel, agent_identity=object()
    )[0]

    with pytest.raises(PermissionError, match="identity policy"):
        wrapped.approval_required_func(
            MagicMock(), SimpleNamespace(name="read_file"), {}
        )


def test_flag_mcp_tool_definitions_authorization_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    kernel = MagicMock()
    kernel.authorize_tool.side_effect = RuntimeError("policy backend detail")
    wrapped = tg.flag_mcp_tool_definitions(
        [mcp_ts], permissions_kernel=kernel, agent_identity=object()
    )[0]

    with pytest.raises(PermissionError, match="authorization unavailable") as exc:
        wrapped.approval_required_func(
            MagicMock(), SimpleNamespace(name="read_file"), {}
        )
    assert "policy backend detail" not in str(exc.value)


def test_flag_mcp_tool_definitions_unknown_decision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.security.tool_guard as tg

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "on")
    mcp_ts = MagicMock(spec=["list_tools"])
    kernel = MagicMock()
    kernel.authorize_tool.return_value = "not-a-policy-decision"
    wrapped = tg.flag_mcp_tool_definitions(
        [mcp_ts], permissions_kernel=kernel, agent_identity=object()
    )[0]

    with pytest.raises(PermissionError, match="no valid decision"):
        wrapped.approval_required_func(
            MagicMock(), SimpleNamespace(name="read_file"), {}
        )


def test_apply_tool_guard_approvals_cannot_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale off value does not skip native function-tool inspection."""
    import agent_utilities.security.tool_guard as tg

    class TrackingAgent:
        accessed = False

        @property
        def toolsets(self):
            self.accessed = True
            return []

    monkeypatch.setattr("agent_utilities.core.config.TOOL_GUARD_MODE", "off")
    agent = TrackingAgent()
    tg.apply_tool_guard_approvals(agent)
    assert agent.accessed is True


# ---------------------------------------------------------------------------
# middlewares
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_token_middleware_delegation_disabled() -> None:
    """Delegation disabled -> passes through."""
    from agent_utilities.mcp.middlewares import UserTokenMiddleware

    mw = UserTokenMiddleware(config={"enable_delegation": False})
    ctx = MagicMock()
    ctx.message.headers = {}
    call_next = AsyncMock(return_value="ok")
    result = await mw.on_request(ctx, call_next)
    assert result == "ok"


@pytest.mark.asyncio
async def test_user_token_middleware_delegation_enabled() -> None:
    """Delegation uses only FastMCP's validated access token."""
    from agent_utilities.mcp.middlewares import UserTokenMiddleware

    mw = UserTokenMiddleware(config={"enable_delegation": True})
    ctx = MagicMock()
    call_next = AsyncMock(return_value="ok")
    access_token = SimpleNamespace(token="abc123", claims={"sub": "user1"})
    identity_tokens = (MagicMock(), MagicMock())
    with (
        patch(
            "fastmcp.server.dependencies.get_access_token",
            return_value=access_token,
        ),
        patch(
            "agent_utilities.mcp.delegated_auth._set_delegated_identity",
            return_value=identity_tokens,
        ) as set_identity,
        patch(
            "agent_utilities.mcp.delegated_auth._reset_delegated_identity"
        ) as reset_identity,
    ):
        result = await mw.on_call_tool(ctx, call_next)
    assert result == "ok"
    set_identity.assert_called_once_with("abc123", {"sub": "user1"})
    reset_identity.assert_called_once_with(identity_tokens)


@pytest.mark.asyncio
async def test_user_token_middleware_missing_header() -> None:
    """Delegation fails closed without a validated access token."""
    from agent_utilities.mcp.middlewares import UserTokenMiddleware

    mw = UserTokenMiddleware(config={"enable_delegation": True})
    ctx = MagicMock()
    call_next = AsyncMock(return_value="ok")
    with (
        patch("fastmcp.server.dependencies.get_access_token", return_value=None),
        pytest.raises(PermissionError, match="Validated user token required"),
    ):
        await mw.on_call_tool(ctx, call_next)
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_jwt_claims_logging_middleware() -> None:
    """JWT claims are logged on response (no return value)."""
    from agent_utilities.mcp.middlewares import JWTClaimsLoggingMiddleware

    mw = JWTClaimsLoggingMiddleware()
    ctx = MagicMock()
    ctx.auth.claims = {"sub": "u1", "client_id": "c1", "scope": "read"}
    call_next = AsyncMock(return_value="response-body")
    # The middleware does not return a value (returns implicit None)
    await mw.on_response(ctx, call_next)
    call_next.assert_awaited_once()


# ---------------------------------------------------------------------------
# tools/memory_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_agents_md_empty(tmp_path: Path) -> None:
    """Empty workspace -> 'No AGENTS.md found'."""
    from agent_utilities.tools.memory_tools import read_agents_md

    ctx = MagicMock()
    ctx.deps.workspace_path = str(tmp_path)
    result = await read_agents_md(ctx)
    assert "No project memory is configured" in result


@pytest.mark.asyncio
async def test_read_agents_md_with_file(tmp_path: Path) -> None:
    """Existing AGENTS.md is read."""
    from agent_utilities.tools.memory_tools import read_agents_md

    (tmp_path / "AGENTS.md").write_text("# Rules\ntdd")
    ctx = MagicMock()
    ctx.deps.workspace_path = str(tmp_path)
    result = await read_agents_md(ctx)
    assert "# Rules" in result


@pytest.mark.asyncio
async def test_read_agents_md_multiple_files(tmp_path: Path) -> None:
    """Multiple AGENTS.md files are concatenated."""
    from agent_utilities.tools.memory_tools import read_agents_md

    (tmp_path / "AGENTS.md").write_text("main content")
    (tmp_path / "AGENTS.local.md").write_text("local content")
    ctx = MagicMock()
    ctx.deps.workspace_path = str(tmp_path)
    result = await read_agents_md(ctx)
    assert "main content" in result
    assert "local content" in result


# ---------------------------------------------------------------------------
# tools/team_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_team_no_capability() -> None:
    """Missing capability auto-initializes one."""
    from agent_utilities.tools.team_tools import spawn_team

    ctx = MagicMock()
    # remove team_capability attr
    if hasattr(ctx, "team_capability"):
        delattr(ctx, "team_capability")
    ctx.deps = MagicMock(graph_engine=None)
    # Delete to simulate missing
    ctx.configure_mock(**{"team_capability": None})
    # getattr with default None will auto-init
    with patch("agent_utilities.tools.team_tools.TeamCapability") as MockCap:
        instance = MagicMock()
        instance.create_team = AsyncMock(return_value="team_1")
        MockCap.return_value = instance
        result = await spawn_team(ctx, "My Team", ["a1"])
    assert "team_1" in result


@pytest.mark.asyncio
async def test_assign_team_task_no_team() -> None:
    """No team capability -> 'No active team'."""
    from agent_utilities.tools.team_tools import assign_team_task

    ctx = MagicMock()
    ctx.configure_mock(**{"team_capability": None})
    result = await assign_team_task(ctx, "content")
    assert "No active team" in result


@pytest.mark.asyncio
async def test_message_teammate_no_team() -> None:
    """No team capability -> 'No active team'."""
    from agent_utilities.tools.team_tools import message_teammate

    ctx = MagicMock()
    ctx.configure_mock(**{"team_capability": None})
    result = await message_teammate(ctx, "member", "hi")
    assert "No active team" in result


@pytest.mark.asyncio
async def test_list_team_tasks_no_engine() -> None:
    """No graph engine -> error message."""
    from agent_utilities.tools.team_tools import list_team_tasks

    ctx = MagicMock()
    ctx.deps.graph_engine = None
    result = await list_team_tasks(ctx)
    assert "Knowledge graph not available" in result


@pytest.mark.asyncio
async def test_list_team_tasks_empty() -> None:
    """No tasks in graph -> 'No tasks found'."""
    from agent_utilities.tools.team_tools import list_team_tasks

    engine = MagicMock()
    engine.graph.nodes.return_value = []
    ctx = MagicMock()
    ctx.deps.graph_engine = engine
    result = await list_team_tasks(ctx)
    assert "No tasks found" in result


@pytest.mark.asyncio
async def test_list_team_tasks_with_tasks() -> None:
    """Tasks are enumerated."""
    from agent_utilities.tools.team_tools import list_team_tasks

    engine = MagicMock()
    engine.graph.nodes.return_value = [
        (
            "task1",
            {
                "type": "task",
                "status": "pending",
                "assigned_to": "agent_x",
                "content": "Do thing",
            },
        )
    ]
    ctx = MagicMock()
    ctx.deps.graph_engine = engine
    result = await list_team_tasks(ctx)
    assert "Do thing" in result
    assert "pending" in result


# ---------------------------------------------------------------------------
# tools/style_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_output_style_builtin() -> None:
    """Builtin style is set."""
    from agent_utilities.tools.style_tools import set_output_style

    ctx = MagicMock()
    ctx.deps.metadata = {}
    result = await set_output_style(ctx, "concise")
    assert "concise" in result
    assert ctx.deps.metadata["output_style"]


@pytest.mark.asyncio
async def test_set_output_style_case_insensitive() -> None:
    """Case-insensitive lookup."""
    from agent_utilities.tools.style_tools import set_output_style

    ctx = MagicMock()
    ctx.deps.metadata = {}
    result = await set_output_style(ctx, "CONCISE")
    assert "CONCISE" in result


@pytest.mark.asyncio
async def test_set_output_style_unknown() -> None:
    """Unknown style with no graph -> 'not found'."""
    from agent_utilities.tools.style_tools import set_output_style

    ctx = MagicMock()
    ctx.deps.metadata = {}
    ctx.deps.graph_engine = None
    result = await set_output_style(ctx, "mystyle")
    assert "not found" in result


@pytest.mark.asyncio
async def test_set_output_style_from_kb() -> None:
    """KB article with matching name provides the style."""
    from agent_utilities.tools.style_tools import set_output_style

    engine = MagicMock()
    engine.graph.nodes.return_value = [
        (
            "art:1",
            {"type": "article", "name": "mystyle", "content": "Be verbose."},
        )
    ]
    ctx = MagicMock()
    ctx.deps.metadata = {}
    ctx.deps.graph_engine = engine
    result = await set_output_style(ctx, "mystyle")
    assert "mystyle" in result


@pytest.mark.asyncio
async def test_list_output_styles_no_engine() -> None:
    """No engine -> lists only builtins."""
    from agent_utilities.tools.style_tools import (
        BUILTIN_STYLES,
        list_output_styles,
    )

    ctx = MagicMock()
    ctx.deps.graph_engine = None
    result = await list_output_styles(ctx)
    for name in BUILTIN_STYLES:
        assert name in result


@pytest.mark.asyncio
async def test_list_output_styles_with_kb() -> None:
    """KB articles tagged 'style' are included."""
    from agent_utilities.tools.style_tools import list_output_styles

    engine = MagicMock()
    engine.graph.nodes.return_value = [
        (
            "art:1",
            {"type": "article", "name": "sassy", "tags": ["style"]},
        )
    ]
    ctx = MagicMock()
    ctx.deps.graph_engine = engine
    result = await list_output_styles(ctx)
    assert "sassy" in result


# ---------------------------------------------------------------------------
# tools/scheduler_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_tools_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """list_tasks returns a formatted registry."""
    from agent_utilities.models.scheduling import CronRegistryModel, CronTaskModel
    from agent_utilities.tools import scheduler_tools

    registry = CronRegistryModel(
        tasks=[
            CronTaskModel(
                id="t1",
                name="Task One",
                interval_minutes=5,
                prompt="do",
            ),
        ]
    )
    monkeypatch.setattr(
        scheduler_tools,
        "list_scheduled_tasks_util",
        lambda: registry,
    )
    ctx = MagicMock()
    result = await scheduler_tools.list_tasks(ctx)
    assert result is registry


@pytest.mark.asyncio
async def test_scheduler_tools_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """schedule_task wraps scheduler.schedule_task."""
    from agent_utilities.tools import scheduler_tools

    monkeypatch.setattr(
        scheduler_tools,
        "schedule_task_util",
        lambda *args, **kw: "Scheduled OK",
    )
    ctx = MagicMock()
    result = await scheduler_tools.schedule_task(ctx, "t1", "Task", 5, "do")
    assert "Scheduled OK" in result


@pytest.mark.asyncio
async def test_scheduler_tools_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_task wraps scheduler.delete_scheduled_task."""
    from agent_utilities.tools import scheduler_tools

    monkeypatch.setattr(
        scheduler_tools,
        "delete_scheduled_task_util",
        lambda tid: "Deleted OK",
    )
    ctx = MagicMock()
    result = await scheduler_tools.delete_task(ctx, "t1")
    assert "Deleted" in result


@pytest.mark.asyncio
async def test_scheduler_tools_view_cron_log_missing_core_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """view_cron_log uses CORE_FILES['CRON_LOG'] which may not exist.

    This just verifies the function path doesn't crash on a missing key by
    monkeypatching CORE_FILES.
    """
    import agent_utilities.core.workspace as ws
    from agent_utilities.tools import scheduler_tools

    # Temporarily patch CORE_FILES to provide a CRON_LOG key
    orig_core = dict(ws.CORE_FILES)
    ws.CORE_FILES["CRON_LOG"] = "CRON_LOG.md"
    try:
        monkeypatch.setattr(
            ws,
            "read_md_file",
            lambda key: "\n".join(f"line{i}" for i in range(100)),
        )
        ctx = MagicMock()
        result = await scheduler_tools.view_cron_log(ctx, lines=5)
        assert "line99" in result
    finally:
        ws.CORE_FILES.clear()
        ws.CORE_FILES.update(orig_core)
