"""Focused RMDD-05 repository-task projection and route tests.

These tests deliberately stub the already-qualified RMDD-02 adapter boundary;
they do not replace WorkItem persistence or exercise a second task backend.
The exact FastMCP compatibility test remains in
``test_mcp_tasks_extension_compatibility.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_utilities.mcp.tasks_extension import (
    TASKS_EXTENSION_ID,
    TASKS_EXTENSION_REVISION,
    WorkItemTasksExtension,
    _GetTaskParams,
    _UpdateTaskParams,
)
from agent_utilities.orchestration.repository_work_item import (
    RepositoryJobState,
    RepositoryOperation,
    RepositoryWorkItemKind,
    RepositoryWorkItemView,
)

TASK_ID = "rmjob:00000000-0000-0000-0000-000000000001"
WORK_ITEM_ID = "workitem:repository_manager:00000000-0000-0000-0000-000000000001"


def _session(*, tenant: str = "tenant-a", owner: str = "owner-a") -> SimpleNamespace:
    return SimpleNamespace(
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        actor=SimpleNamespace(actor_id=owner),
    )


def _delegator(*scopes: str) -> SimpleNamespace:
    return SimpleNamespace(
        authenticated=True,
        issuer="https://issuer.example.test",
        audience="graph-os",
        service_principal="mcp-multiplexer",
        scopes=frozenset(scopes or ("mcp:delegate",)),
    )


def _view(
    state: RepositoryJobState,
    *,
    owner: str = "owner-a",
    error_ref: str | None = None,
    result_ref: str | None = "artifact:v1:build",
) -> RepositoryWorkItemView:
    return RepositoryWorkItemView(
        job_id=TASK_ID,
        work_item_id=WORK_ITEM_ID,
        request_id="request-1",
        operation=RepositoryOperation.BUILD,
        kind=RepositoryWorkItemKind.BUILD,
        state=state,
        repository_id="agent-utilities",
        tenant_id="tenant-a",
        owner_id=owner,
        session_id="session-1",
        base_ref="main",
        base_sha="a" * 40,
        target_kind="local",
        input_digest="b" * 64,
        attempt=1,
        max_attempts=3,
        result_ref=result_ref,
        error_ref=error_ref,
    )


def _raw_item(status: str = "succeeded") -> dict:
    return {
        "id": WORK_ITEM_ID,
        "tenant": "tenant-a",
        "status": status,
        "created_at": "2026-08-09T12:00:00Z",
        "updated_at": "2026-08-09T12:01:00Z",
        "metadata": {},
    }


def test_repository_completed_projection_preserves_domain_result_ref_and_route_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item

    view = _view(RepositoryJobState.SUCCEEDED)
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(work_item, "get_work_item", lambda *args, **kwargs: _raw_item())
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())

    result = extension._project(TASK_ID, session=_session())

    assert result.status == "completed"
    assert result.result["resultRef"] == "artifact:v1:build"
    assert result.result["jobId"] == TASK_ID
    assert result.meta == {
        TASKS_EXTENSION_ID: {
            "server": "repository-manager",
            "revision": TASKS_EXTENSION_REVISION,
            "caller": {
                "tenant": "tenant-a",
                "owner": "owner-a",
                "scopes": ["kg:read", "kg:write"],
            },
        }
    }


def test_repository_failed_projection_preserves_failure_and_refusal_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item

    view = _view(
        RepositoryJobState.FAILED,
        result_ref=None,
        error_ref="repository-error:v1::unsafe_ref:artifact:v1:failure",
    )
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(
        work_item, "get_work_item", lambda *args, **kwargs: _raw_item("failed")
    )
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())
    result = extension._project(TASK_ID, session=_session())

    assert result.status == "failed"
    assert result.error["failureClass"] is None
    assert result.error["refusalCode"] == "unsafe_ref"
    assert result.error["errorRef"] == "artifact:v1:failure"
    assert result.error["result"]["refusalCode"] == "unsafe_ref"


def test_repository_projection_rejects_wrong_owner_without_leaking_existence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter

    calls: list[dict] = []

    def _get(*args, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(repository_adapter, "get_repository_work_item", _get)
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())

    with pytest.raises(BaseException, match="Unknown task"):
        extension._project(TASK_ID, session=_session(owner="other-owner"))
    assert calls == [{"tenant": "tenant-a", "owner_id": "other-owner"}]


@pytest.mark.parametrize(
    "bad_row, message",
    [
        (None, "Repository WorkItem state is unavailable"),
        ({"id": WORK_ITEM_ID, "status": "succeeded"}, "state is corrupt"),
        (
            _raw_item() | {"created_at": "not-a-timestamp"},
            "Task timestamp was invalid",
        ),
    ],
)
def test_repository_projection_fails_closed_on_missing_or_corrupt_authoritative_row(
    monkeypatch: pytest.MonkeyPatch,
    bad_row: dict | None,
    message: str,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item

    view = _view(RepositoryJobState.SUCCEEDED)
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(work_item, "get_work_item", lambda *args, **kwargs: bad_row)
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())

    with pytest.raises(BaseException, match=message):
        extension._project(TASK_ID, session=_session())


def test_task_id_and_input_response_shapes_are_bounded_at_the_native_schema() -> None:
    with pytest.raises(Exception, match="must not be blank"):
        _GetTaskParams(taskId="   ")
    with pytest.raises(Exception, match="must be trimmed"):
        _GetTaskParams(taskId=" task-id")
    with pytest.raises(Exception, match="control character"):
        _GetTaskParams(taskId="task\n-id")
    with pytest.raises(Exception, match="size limit|at most 512"):
        _GetTaskParams(taskId="x" * 513)
    with pytest.raises(Exception, match="size limit"):
        _GetTaskParams(taskId="é" * 257)

    with pytest.raises(Exception, match="string exceeds|size limit"):
        _UpdateTaskParams(
            taskId=TASK_ID,
            inputResponses={"answer": "x" * (16 * 1024 + 1)},
        )
    with pytest.raises(Exception, match="size limit"):
        _UpdateTaskParams(
            taskId=TASK_ID,
            inputResponses={
                "first": "\\" * (16 * 1024),
                "second": "\\" * (16 * 1024),
                "third": "\\" * (16 * 1024),
            },
        )

    deep: dict[str, object] = {}
    for _ in range(26):
        deep = {"next": deep}
    with pytest.raises(Exception, match="nesting depth"):
        _UpdateTaskParams(taskId=TASK_ID, inputResponses={"request": deep})

    with pytest.raises(Exception, match="item count|at most 4096"):
        _UpdateTaskParams(
            taskId=TASK_ID,
            inputResponses={"answers": list(range(4_100))},
        )


def test_route_caller_spoof_is_rejected_even_when_server_is_local() -> None:
    extension = WorkItemTasksExtension(server_id="repository-manager")
    route = {
        "server": "repository-manager",
        "revision": TASKS_EXTENSION_REVISION,
        "caller": {"tenant": "tenant-a", "owner": "other-owner", "scopes": ["kg:read"]},
    }

    with pytest.raises(BaseException, match="does not match verified authority"):
        extension._validate_delegated_caller(route, _session())


@pytest.mark.asyncio
async def test_direct_tasks_get_handler_uses_repository_adapter_and_verified_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item

    view = _view(RepositoryJobState.READY, result_ref=None)
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(
        work_item, "get_work_item", lambda *args, **kwargs: _raw_item("ready")
    )
    extension = WorkItemTasksExtension(server_id="repository-manager")
    session = _session()
    monkeypatch.setattr(extension, "_engine", lambda: object())
    monkeypatch.setattr(extension, "_require_tasks_capability", lambda ctx: None)
    monkeypatch.setattr(extension, "_authorized_session", lambda ctx, scope: session)

    result = await extension._handle_get(
        SimpleNamespace(), _GetTaskParams(taskId=TASK_ID)
    )

    assert result.task_id == TASK_ID
    assert result.status == "working"
    assert result.meta[TASKS_EXTENSION_ID]["caller"]["owner"] == "owner-a"


@pytest.mark.asyncio
async def test_child_handler_accepts_only_the_signed_multiplexer_task_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real child handler path, not only token helper methods.

    A pooled child is authenticated as its service principal when the MCP
    connection opens.  The original graph caller therefore reaches this
    handler only through the bounded, signed delegation envelope emitted by
    the multiplexer.  This test runs the same route parsing, proof validation,
    owner-scoped adapter lookup, and response metadata path used by the
    FastMCP request handler.
    """
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item
    from agent_utilities.mcp.tasks_extension import (
        _channel_proof,
        _mint_delegation_token,
    )

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "handler-delegation-secret")
    channel_secret = "stdio-generation-secret-0123456789abcdef"
    monkeypatch.setenv("AGENT_UTILITIES_MCP_TASK_CHANNEL_SECRET", channel_secret)
    caller = {"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]}
    unsigned = _GetTaskParams(taskId=TASK_ID)
    token = _mint_delegation_token(
        "tasks/get",
        unsigned.model_dump(mode="json", by_alias=True, exclude_none=True),
        server="repository-manager",
        revision=TASKS_EXTENSION_REVISION,
        caller=caller,
    )
    params = _GetTaskParams.model_validate(
        {
            "taskId": TASK_ID,
            "_meta": {
                TASKS_EXTENSION_ID: {
                    "server": "repository-manager",
                    "revision": TASKS_EXTENSION_REVISION,
                    "caller": caller,
                    "delegation": {
                        "issuer": "mcp-multiplexer",
                        "token": token,
                        "channel": _channel_proof(channel_secret, token),
                    },
                }
            },
        }
    )
    view = _view(RepositoryJobState.READY, result_ref=None)
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(
        work_item, "get_work_item", lambda *args, **kwargs: _raw_item("ready")
    )
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())
    monkeypatch.setattr(extension, "_require_tasks_capability", lambda ctx: None)

    result = await extension._handle_get(SimpleNamespace(), params)

    assert result.task_id == TASK_ID
    assert result.status == "working"
    assert result.meta == {
        TASKS_EXTENSION_ID: {
            "server": "repository-manager",
            "revision": TASKS_EXTENSION_REVISION,
            "caller": caller,
        }
    }


def test_valid_hmac_from_a_non_delegator_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct client cannot use a valid shared-secret proof as authority."""
    from agent_utilities.mcp.tasks_extension import _mint_delegation_token

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "non-delegator-secret")
    caller = {"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]}
    params = _GetTaskParams(taskId=TASK_ID)
    token = _mint_delegation_token(
        "tasks/get",
        params.model_dump(mode="json", by_alias=True, exclude_none=True),
        server="repository-manager",
        revision=TASKS_EXTENSION_REVISION,
        caller=caller,
    )
    route = {
        "server": "repository-manager",
        "revision": TASKS_EXTENSION_REVISION,
        "caller": caller,
        "delegation": {"issuer": "mcp-multiplexer", "token": token},
    }
    extension = WorkItemTasksExtension(server_id="repository-manager")

    with pytest.raises(BaseException, match="service authority is required"):
        extension._delegated_session(
            route,
            method="tasks/get",
            params=params,
            scope="kg:read",
            service_authority=_delegator("kg:read"),
        )


def test_remote_valid_bearer_without_delegate_capability_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote HTTP children require the verified fleet-delegator capability."""
    from agent_utilities.core.config import config

    monkeypatch.delenv("AGENT_UTILITIES_MCP_TASK_CHANNEL_SECRET", raising=False)
    token = SimpleNamespace(
        claims={
            "iss": config.auth_jwt_issuer,
            "aud": config.auth_jwt_audience,
            "sub": "direct-client",
            "client_id": "direct-client",
            "scope": "kg:read",
        },
        scopes=("kg:read",),
        client_id="direct-client",
    )
    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", lambda: token)
    extension = WorkItemTasksExtension(server_id="repository-manager")

    with pytest.raises(BaseException, match="service authority is required"):
        extension._authorized_delegator(SimpleNamespace())


def test_signed_delegation_expiry_is_enforced_by_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp.tasks_extension import _delegation_binding
    from agent_utilities.security.run_token import mint_token

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "expiry-delegation-secret")
    caller = {"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]}
    params = _GetTaskParams(taskId=TASK_ID)
    binding = _delegation_binding(
        "tasks/get",
        params.model_dump(mode="json", by_alias=True, exclude_none=True),
        server="repository-manager",
        revision=TASKS_EXTENSION_REVISION,
        caller=caller,
    )
    token = mint_token(
        f"mcp-task:{binding}",
        project="repository-manager",
        endpoints=("repository-manager",),
        operations=("tasks/get",),
        ttl_seconds=-1.0,
        actor_id="owner-a",
        tenant_id="tenant-a",
    )
    route = {
        "server": "repository-manager",
        "revision": TASKS_EXTENSION_REVISION,
        "caller": caller,
        "delegation": {"issuer": "mcp-multiplexer", "token": token},
    }
    extension = WorkItemTasksExtension(server_id="repository-manager")

    with pytest.raises(BaseException, match="Invalid or expired delegated task proof"):
        extension._delegated_session(
            route,
            method="tasks/get",
            params=params,
            scope="kg:read",
            service_authority=_delegator(),
        )


@pytest.mark.asyncio
async def test_host_handler_forwards_through_bounded_runtime_to_child_handler(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover host extension -> multiplexer -> native child handler end to end."""
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item
    from agent_utilities.mcp.child_resilience import ChildRuntime
    from agent_utilities.mcp.multiplexer import MCPMultiplexer
    from agent_utilities.mcp.tasks_extension import _GetTaskResult

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "end-to-end-task-secret")
    channel_secret = "stdio-generation-secret-abcdef0123456789"
    monkeypatch.setenv("AGENT_UTILITIES_MCP_TASK_CHANNEL_SECRET", channel_secret)
    view = _view(RepositoryJobState.READY, result_ref=None)
    monkeypatch.setattr(
        repository_adapter, "get_repository_work_item", lambda *args, **kwargs: view
    )
    monkeypatch.setattr(
        work_item, "get_work_item", lambda *args, **kwargs: _raw_item("ready")
    )

    child = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(child, "_engine", lambda: object())
    monkeypatch.setattr(child, "_require_tasks_capability", lambda ctx: None)

    class _Session:
        initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                extensions={TASKS_EXTENSION_ID: {"revision": TASKS_EXTENSION_REVISION}}
            )
        )

        async def send_request(self, request, result_type):
            result = await child._handle_get(SimpleNamespace(), request.params)
            assert isinstance(result, result_type)
            return result

    runtime = ChildRuntime(
        "repository-manager", {"max_concurrency": 1, "queue_timeout": 0.5}
    )
    runtime._task_generation_secret = channel_secret
    runtime.adopt_sessions([_Session()])
    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {}}
    mux.children["repository-manager"] = runtime

    host = WorkItemTasksExtension(server_id="graph-os", task_router=mux)
    host_session = _session()
    monkeypatch.setattr(host, "_require_tasks_capability", lambda ctx: None)
    monkeypatch.setattr(host, "_authorized_session", lambda ctx, scope: host_session)
    params = _GetTaskParams.model_validate(
        {
            "taskId": TASK_ID,
            "_meta": {
                TASKS_EXTENSION_ID: {
                    "server": "repository-manager",
                    "revision": TASKS_EXTENSION_REVISION,
                }
            },
        }
    )

    result = await host._handle_get(SimpleNamespace(), params)

    assert isinstance(result, _GetTaskResult)
    assert result.status == "working"
    assert result.meta[TASKS_EXTENSION_ID]["server"] == "repository-manager"
    assert result.meta[TASKS_EXTENSION_ID]["caller"] == {
        "tenant": "tenant-a",
        "owner": "owner-a",
        "scopes": ["kg:read"],
    }
    await runtime.aclose()


@pytest.mark.asyncio
async def test_read_retry_rebuilds_stdio_proof_for_the_new_generation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconnecting stdio child receives a fresh channel MAC on retry."""
    from agent_utilities.mcp.child_resilience import ChildRuntime
    from agent_utilities.mcp.multiplexer import MCPMultiplexer
    from agent_utilities.mcp.tasks_extension import (
        WorkItemTasksExtension,
        _channel_proof,
        _GetTaskResult,
    )

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "retry-task-secret")
    old_secret = "old-generation-channel-secret-0123456789"
    new_secret = "new-generation-channel-secret-0123456789"
    generation_secrets = iter((old_secret, new_secret))
    sessions_seen: list[object] = []

    class _Session:
        initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                extensions={TASKS_EXTENSION_ID: {"revision": TASKS_EXTENSION_REVISION}}
            )
        )

        def __init__(self, secret: str, *, fails: bool) -> None:
            self.secret = secret
            self.fails = fails
            self.requests: list[object] = []

        async def send_request(self, request, result_type):
            self.requests.append(request)
            if self.fails:
                raise ConnectionResetError("synthetic stdio generation death")
            envelope = request.params.meta[TASKS_EXTENSION_ID]
            token = envelope["delegation"]["token"]
            assert envelope["delegation"]["channel"] == _channel_proof(
                self.secret, token
            )
            return _GetTaskResult(
                task_id=TASK_ID,
                status="completed",
                created_at="2026-08-09T12:00:00Z",
                last_updated_at="2026-08-09T12:01:00Z",
                result={"resultRef": "artifact:v1:retry"},
            )

    runtime: ChildRuntime

    async def connect(_stack):
        secret = next(generation_secrets)
        runtime._task_generation_secret = secret
        session = _Session(secret, fails=secret == old_secret)
        sessions_seen.append(session)
        return [session], []

    runtime = ChildRuntime(
        "repository-manager",
        {"max_concurrency": 1, "queue_timeout": 0.5},
        connect=connect,
        restart_backoff_base=0.001,
        restart_backoff_cap=0.001,
    )
    await runtime.start()
    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {"command": "repository-manager"}}
    mux.children["repository-manager"] = runtime

    result = await mux.forward_task_method(
        "tasks/get",
        {"taskId": TASK_ID},
        caller={"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]},
        route={"server": "repository-manager", "revision": TASKS_EXTENSION_REVISION},
    )

    assert result.result == {"resultRef": "artifact:v1:retry"}
    assert runtime.generation == 2
    old_request = sessions_seen[0].requests[0]
    new_request = sessions_seen[1].requests[0]
    old_route = old_request.params.meta[TASKS_EXTENSION_ID]
    new_route = new_request.params.meta[TASKS_EXTENSION_ID]
    child = WorkItemTasksExtension(server_id="repository-manager")
    new_authority = _delegator()
    new_authority.channel_secret = new_secret
    with pytest.raises(BaseException, match="Invalid delegated task proof"):
        child._delegated_session(
            old_route,
            method="tasks/get",
            params=old_request.params,
            scope="kg:read",
            service_authority=new_authority,
        )
    assert (
        child._delegated_session(
            new_route,
            method="tasks/get",
            params=new_request.params,
            scope="kg:read",
            service_authority=new_authority,
        ).actor.actor_id
        == "owner-a"
    )
    await runtime.aclose()


def test_cancel_race_reports_durable_terminal_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.orchestration.repository_work_item as repository_adapter
    import agent_utilities.orchestration.work_item as work_item

    terminal = _view(RepositoryJobState.SUCCEEDED)
    monkeypatch.setattr(
        WorkItemTasksExtension,
        "_repository_view",
        lambda self, task_id, session: terminal,
    )
    monkeypatch.setattr(
        repository_adapter,
        "cancel_repository_work_item",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(work_item, "get_work_item", lambda *args, **kwargs: _raw_item())
    extension = WorkItemTasksExtension(server_id="repository-manager")
    monkeypatch.setattr(extension, "_engine", lambda: object())

    result = extension._cancel_repository(TASK_ID, session=_session(), route=None)

    assert result.status == "completed"
    assert "lost a race" in (result.status_message or "")


def test_signed_delegation_reauthorizes_child_identity_and_binds_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp.tasks_extension import (
        _GetTaskParams,
        _mint_delegation_token,
    )

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "delegation-secret-a")
    caller = {"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]}
    params = _GetTaskParams(taskId=TASK_ID)
    token = _mint_delegation_token(
        "tasks/get",
        params.model_dump(mode="json", by_alias=True, exclude_none=True),
        server="repository-manager",
        revision=TASKS_EXTENSION_REVISION,
        caller=caller,
    )
    route = {
        "server": "repository-manager",
        "revision": TASKS_EXTENSION_REVISION,
        "caller": caller,
        "delegation": {"issuer": "mcp-multiplexer", "token": token},
    }
    extension = WorkItemTasksExtension(server_id="repository-manager")

    delegated = extension._delegated_session(
        route,
        method="tasks/get",
        params=params,
        scope="kg:read",
        service_authority=_delegator(),
    )

    assert delegated.tenant == "tenant-a"
    assert delegated.actor.actor_id == "owner-a"
    assert delegated.scopes == frozenset({"kg:read"})

    tampered = _GetTaskParams(taskId="rmjob:00000000-0000-0000-0000-000000000002")
    with pytest.raises(BaseException, match="binding is invalid"):
        extension._delegated_session(
            route,
            method="tasks/get",
            params=tampered,
            scope="kg:read",
            service_authority=_delegator(),
        )

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "delegation-secret-b")
    with pytest.raises(BaseException, match="expired delegated task proof|Invalid"):
        extension._delegated_session(
            route,
            method="tasks/get",
            params=params,
            scope="kg:read",
            service_authority=_delegator(),
        )


@pytest.mark.asyncio
async def test_multiplexer_denies_before_lazy_mount_when_fleet_capability_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.mcp.multiplexer as multiplexer

    mux = multiplexer.MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {"required_scopes": ["mcp:repo"]}}
    mount_called = False

    async def _mount(_server: str):
        nonlocal mount_called
        mount_called = True
        return []

    def _deny(*args, **kwargs):
        raise multiplexer.ToolError("MCP fleet delegate capability required")

    monkeypatch.setattr(mux, "mount_child", _mount)
    monkeypatch.setattr(multiplexer, "_require_fleet_capability", _deny)

    with pytest.raises(Exception, match="delegate capability required"):
        await mux.forward_task_method(
            "tasks/get",
            {"taskId": TASK_ID},
            caller={"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]},
            route={
                "server": "repository-manager",
                "revision": TASKS_EXTENSION_REVISION,
            },
        )
    assert mount_called is False


@pytest.mark.asyncio
async def test_multiplexer_forwards_exact_revision_route_and_verified_caller(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp.multiplexer import MCPMultiplexer
    from agent_utilities.mcp.tasks_extension import _GetTaskResult

    class _Session:
        initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                extensions={TASKS_EXTENSION_ID: {"revision": TASKS_EXTENSION_REVISION}}
            )
        )

        def __init__(self) -> None:
            self.requests = []

        async def send_request(self, request, result_type):
            self.requests.append(request)
            return _GetTaskResult(
                task_id=TASK_ID,
                status="completed",
                created_at="2026-08-09T12:00:00Z",
                last_updated_at="2026-08-09T12:01:00Z",
                result={"resultRef": "artifact:v1:build"},
            )

    class _Runtime:
        def __init__(self, live_session: _Session) -> None:
            self.primary_session = live_session
            self._sessions = [live_session]
            self.generation = 1
            self._task_generation_secret = "test-task-channel-secret-0123456789"

        async def call_request(
            self,
            request,
            result_type,
            *,
            before_send=None,
            request_factory=None,
            **_kwargs,
        ):
            if request_factory is not None:
                request = request_factory(self.generation, self._task_generation_secret)
            if before_send is not None:
                before_send(self.generation, self._task_generation_secret)
            return await self.primary_session.send_request(request, result_type)

    session = _Session()
    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "test-task-route-secret")
    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {}}
    mux.children["repository-manager"] = _Runtime(session)

    result = await mux.forward_task_method(
        "tasks/get",
        {"taskId": TASK_ID},
        route={"server": "repository-manager", "revision": TASKS_EXTENSION_REVISION},
        caller={"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:read"]},
    )

    assert isinstance(result, _GetTaskResult)
    assert result.task_id == TASK_ID
    sent_meta = session.requests[0].params.meta
    assert sent_meta[TASKS_EXTENSION_ID]["server"] == "repository-manager"
    assert sent_meta[TASKS_EXTENSION_ID]["revision"] == TASKS_EXTENSION_REVISION
    assert sent_meta[TASKS_EXTENSION_ID]["caller"] == {
        "tenant": "tenant-a",
        "owner": "owner-a",
        "scopes": ["kg:read"],
    }
    from agent_utilities.security.run_token import validate_token

    token = sent_meta[TASKS_EXTENSION_ID]["delegation"]["token"]
    assert len(sent_meta[TASKS_EXTENSION_ID]["delegation"]["channel"]) == 64
    decoded = validate_token(
        token, endpoint="repository-manager", operation="tasks/get"
    )
    assert decoded.actor_id == "owner-a"
    assert decoded.tenant_id == "tenant-a"
    assert result.meta is None or "delegation" not in result.meta


@pytest.mark.asyncio
async def test_mutating_task_forward_fails_before_retired_epoch_or_generation_send(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    class _Session:
        initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                extensions={TASKS_EXTENSION_ID: {"revision": TASKS_EXTENSION_REVISION}}
            )
        )

    class _Runtime:
        def __init__(self) -> None:
            self._sessions = [_Session()]
            self.generation = 1
            self._task_generation_secret = "mutation-channel-secret-0123456789"
            self.sent = False
            self.bump_catalog = True

        async def call_request(
            self,
            request,
            result_type,
            *,
            before_send=None,
            request_factory=None,
            **_kwargs,
        ):
            if self.bump_catalog:
                mux._catalog_epoch += 1
            else:
                self.generation += 1
            before_send(self.generation, self._task_generation_secret)
            self.sent = True
            raise AssertionError("retired mutation must not reach the child")

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "mutation-task-secret")
    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {"command": "repository-manager"}}
    runtime = _Runtime()
    mux.children["repository-manager"] = runtime
    route = {
        "server": "repository-manager",
        "revision": TASKS_EXTENSION_REVISION,
    }
    caller = {"tenant": "tenant-a", "owner": "owner-a", "scopes": ["kg:write"]}

    with pytest.raises(Exception, match="retired before the request was sent"):
        await mux.forward_task_method(
            "tasks/cancel",
            {"taskId": TASK_ID},
            caller=caller,
            route=route,
        )
    assert runtime.sent is False

    runtime.bump_catalog = False
    mux._catalog_epoch = 0
    runtime.generation = 1
    with pytest.raises(Exception, match="generation changed before send"):
        await mux.forward_task_method(
            "tasks/cancel",
            {"taskId": TASK_ID},
            caller=caller,
            route=route,
        )
    assert runtime.sent is False


@pytest.mark.asyncio
async def test_multiplexer_rejects_unqualified_or_wrong_revision_child(
    tmp_path,
) -> None:
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {"repository-manager": {}}
    incompatible = SimpleNamespace(
        capabilities=SimpleNamespace(
            extensions={TASKS_EXTENSION_ID: {"revision": "old-revision"}}
        )
    )

    class _Runtime:
        _sessions = [SimpleNamespace(initialize_result=incompatible)]
        generation = 1

        async def call_request(
            self, request, result_type, **_kwargs
        ):  # pragma: no cover - capability gate precedes call
            raise AssertionError("request should not be sent")

    mux.children["repository-manager"] = _Runtime()

    with pytest.raises(Exception, match="did not advertise native Tasks"):
        await mux.forward_task_method(
            "tasks/get",
            {"taskId": TASK_ID},
            route={
                "server": "repository-manager",
                "revision": TASKS_EXTENSION_REVISION,
            },
        )


def test_multiplexer_mixed_replica_pool_and_disconnect_fail_closed(tmp_path) -> None:
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    def _init(revision: str):
        return SimpleNamespace(
            capabilities=SimpleNamespace(
                extensions={TASKS_EXTENSION_ID: {"revision": revision}}
            )
        )

    capable = SimpleNamespace(initialize_result=_init(TASKS_EXTENSION_REVISION))
    incompatible = SimpleNamespace(initialize_result=_init("old-revision"))
    runtime = SimpleNamespace(_sessions=[capable, incompatible])
    mux = MCPMultiplexer(tmp_path / "mcp.json")

    assert not mux._tasks_runtime_capable("repository-manager", runtime)
    runtime._sessions = [capable, capable]
    assert mux._tasks_runtime_capable("repository-manager", runtime)
    runtime._sessions = []
    assert not mux._tasks_runtime_capable("repository-manager", runtime)


@pytest.mark.asyncio
async def test_retired_runtime_reconnect_cannot_repopulate_task_capability_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late reconnect from an old catalog epoch cannot overwrite new state."""
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    server_name = "repository-manager"
    mux = MCPMultiplexer(tmp_path / "mcp.json")
    mux._catalog = {server_name: {"command": "repository-manager"}}

    initialization = SimpleNamespace(
        capabilities=SimpleNamespace(
            extensions={TASKS_EXTENSION_ID: {"revision": TASKS_EXTENSION_REVISION}}
        )
    )

    class _Session:
        initialize_result = initialization

        async def list_tools(self):
            return SimpleNamespace(tools=[])

    opened = [_Session(), _Session()]

    async def _open(*_args):
        return opened.pop(0)

    open_one = monkeypatch.setattr(mux, "_open_one_session", _open)
    del open_one  # the async call count is tracked by the remaining generation state
    result = await mux._start_child(server_name, mux._catalog[server_name])
    assert result is not None
    _name, runtime, _tools, _cfg = result
    mux.children[server_name] = runtime

    # Install a replacement runtime before the old generation reconnects. The
    # late callback must not replace its live session or capability state.
    replacement_session = _Session()
    replacement = SimpleNamespace(
        _sessions=[replacement_session],
        generation=1,
        primary_session=replacement_session,
    )
    mux.children[server_name] = replacement
    mux.sessions[server_name] = replacement_session
    mux._catalog_epoch += 1
    runtime.restart_backoff_base = 0.001
    runtime.restart_backoff_cap = 0.001
    runtime.request_restart("catalog epoch retired")
    for _ in range(100):
        if not opened:
            break
        await asyncio.sleep(0.01)

    assert not opened
    assert mux.children[server_name] is replacement
    assert mux.sessions[server_name] is replacement_session
    assert mux._tasks_runtime_capable(server_name, replacement)
    await runtime.aclose()
