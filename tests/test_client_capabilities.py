"""Client tools run through the real agent loop without local effects."""

import asyncio
import json
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run
from agent_utilities.graph.state import HarnessConfig
from agent_utilities.harness.client_capabilities import (
    ClientCapabilityToolset,
    ClientToolLimits,
)


class Transport:
    def __init__(self, response="client reply", refusal=None):
        self.calls = []
        self.response = response
        self.refusal = refusal

    async def call(self, method, params):
        self.calls.append((method, params))
        if self.refusal:
            raise host_ports.ClientCapabilityRefused(self.refusal)
        return self.response


def scripted(tool, params, observed):
    async def model(messages, info):
        observed.append({tool.name for tool in info.function_tools})
        last = messages[-1].parts[-1]
        if getattr(last, "part_kind", "") == "tool-return":
            return ModelResponse(parts=[TextPart(str(last.content))])
        return ModelResponse(parts=[ToolCallPart(tool, params)])

    return FunctionModel(model)


@pytest.mark.spec("AU-BOUNDARY-R043.6")
@pytest.mark.parametrize(
    ("tool", "params", "method"),
    [
        ("read_file", {"path": "private.txt"}, "fs/read_text_file"),
        (
            "write_file",
            {"path": "private.txt", "content": "secret"},
            "fs/write_text_file",
        ),
        ("search_files", {"pattern": "secret"}, "fs/search_files"),
        ("apply_patch", {"patch": "private diff"}, "fs/apply_patch"),
        ("run_command", {"command": "private-command"}, "terminal/create"),
    ],
)
async def test_real_run_binds_only_selected_tool(
    tool, params, method, events, tmp_path
):
    transport, observed = Transport(), []
    plan = RunPlan(
        model=scripted(tool, params, observed),
        steps=("do it",),
        harness=HarnessConfig(client_tools=(tool,)),
        client_transport=transport,
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    result = await run(RunRequest(task="do it"))
    assert result.output == "client reply"
    assert observed == [{tool}, {tool}]
    assert len(transport.calls) == 1
    assert transport.calls[0][0] == method
    assert all(transport.calls[0][1][key] == value for key, value in params.items())
    assert list(tmp_path.iterdir()) == []
    trace = next(event for event in events if event["type"] == "client_capability")
    assert set(trace) == {
        "type",
        "run_id",
        "method",
        "request_bytes",
        "output_bytes",
        "outcome",
    }
    assert "private" not in json.dumps(trace) and "secret" not in json.dumps(trace)


@pytest.mark.spec("AU-BOUNDARY-R043.6")
@pytest.mark.parametrize(
    "reason", ["permission_required", "permission_denied", "client_disconnected"]
)
async def test_refusal_never_approves_or_retries(reason, events):
    transport = Transport(refusal=reason)
    tools = ClientCapabilityToolset(transport, ("write_file",), run_id="run")
    assert json.loads(await tools.write_file("path", "content"))["reason"] == reason
    assert len(transport.calls) == 1
    assert events[-1]["outcome"] == reason


@pytest.mark.spec("AU-BOUNDARY-R043.6")
@pytest.mark.parametrize(
    ("limits", "response", "reason", "calls"),
    [
        (ClientToolLimits(request_bytes=1), "ok", "request_limit", 0),
        (ClientToolLimits(output_bytes=1), "é", "output_limit", 1),
        (ClientToolLimits(), None, "invalid_response", 1),
    ],
)
async def test_byte_limits_and_invalid_response(limits, response, reason, calls):
    transport = Transport(response)
    tools = ClientCapabilityToolset(
        transport, ("read_file",), limits=limits, run_id="r"
    )
    assert json.loads(await tools.read_file("x"))["reason"] == reason
    assert len(transport.calls) == calls


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_transport_timeout_does_not_retry():
    class TimedOut(Transport):
        async def call(self, method, params):
            self.calls.append((method, params))
            raise TimeoutError

    transport = TimedOut()
    tools = ClientCapabilityToolset(transport, ("run_command",), run_id="r")
    assert json.loads(await tools.run_command("command"))["reason"] == "timeout"
    assert len(transport.calls) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.6")
@pytest.mark.parametrize(
    "changes",
    [{"output_bytes": 0}, {"request_bytes": True}, {"timeout_seconds": float("nan")}],
)
def test_invalid_limits(changes):
    with pytest.raises(ValueError):
        replace(ClientToolLimits(), **changes)


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_missing_transport_fails_real_run(events):
    plan = RunPlan(
        model=scripted("read_file", {}, []),
        steps=("q",),
        harness=HarnessConfig(client_tools=("read_file",)),
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    result = await run(RunRequest(task="q"))
    assert result.status == "failed"
    assert "client_disconnected" in result.error


@pytest.mark.spec("AU-BOUNDARY-R043.6")
@pytest.mark.parametrize("selected", [("unknown",), ("read_file", "read_file")])
def test_unknown_or_duplicate_selection(selected):
    with pytest.raises(ValueError):
        ClientCapabilityToolset(Transport(), selected, run_id="r")


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_wait_limit_cancels_request_without_retry():
    cancelled = asyncio.Event()

    class Waiting(Transport):
        async def call(self, method, params):
            self.calls.append((method, params))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    transport = Waiting()
    tools = ClientCapabilityToolset(
        transport,
        ("read_file",),
        run_id="r",
        limits=ClientToolLimits(timeout_seconds=0.001),
    )
    assert json.loads(await tools.read_file("x"))["reason"] == "timeout"
    assert cancelled.is_set() and len(transport.calls) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_transport_exception_has_no_private_detail(events):
    class Broken(Transport):
        async def call(self, method, params):
            raise RuntimeError("private transport credential")

    tools = ClientCapabilityToolset(Broken(), ("read_file",), run_id="r")
    result = await tools.read_file("private path")
    assert json.loads(result)["reason"] == "transport_error"
    assert "private" not in result + json.dumps(events)


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_public_hosted_run_binds_transport(events):
    from agent_utilities.harness.hosted_run import iterate_client_run

    transport = Transport()
    plan = RunPlan(
        model=scripted("read_file", {"path": "file"}, []),
        steps=("inspect",),
        harness=HarnessConfig(client_tools=("read_file",)),
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    emitted = [event async for event in iterate_client_run("inspect", transport)]
    assert emitted[-1]["type"] == "graph_complete"
    assert len(transport.calls) == 1
    assert any(event.get("output") == "client reply" for event in emitted)


@pytest.mark.spec("AU-BOUNDARY-R043.6")
async def test_public_hosted_run_rejects_server_tools(events):
    from agent_utilities.harness.hosted_run import iterate_client_run

    transport = Transport()
    plan = RunPlan(model=scripted("read_file", {}, []), tool_names=("server_tool",))
    host_ports.register_run_decision_provider(lambda request: plan)
    with pytest.raises(PermissionError, match="client-only"):
        _ = [event async for event in iterate_client_run("inspect", transport)]
    assert transport.calls == []
