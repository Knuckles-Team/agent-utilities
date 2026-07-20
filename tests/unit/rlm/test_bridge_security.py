"""Adversarial source tests for the isolated RLM helper bridge."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import struct

import pytest

from agent_utilities.rlm.sandboxes import _bridge


def test_bridge_inputs_are_private_bounded_and_helper_allowlisted(tmp_path):
    def final_var(name, value):  # noqa: ARG001
        return None

    _bridge.write_inputs(
        tmp_path,
        "x = 1",
        vars_payload={"seed": 1},
        tool_sources={},
        helpers={"FINAL_VAR": final_var, "arbitrary_host_callable": lambda: None},
        bridge_token="capability-" + "token",
        runner_data_dir=None,
    )
    context_path = tmp_path / "context.json"
    context = json.loads(context_path.read_text())
    assert context["bridge_token"] == "capability-token"
    assert "FINAL_VAR" in context["sync_helpers"]
    assert "arbitrary_host_callable" not in context["sync_helpers"]
    if os.name != "nt":
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert stat.S_IMODE(context_path.stat().st_mode) == 0o600


def test_bridge_rejects_excessive_json_depth():
    value: object = "leaf"
    for _ in range(_bridge.MAX_JSON_DEPTH + 2):
        value = [value]
    assert _bridge._bounded_json_shape(value) is False


@pytest.mark.skipif(not hasattr(asyncio, "start_unix_server"), reason="UDS required")
@pytest.mark.asyncio
async def test_bridge_rejects_unauthenticated_request_without_reflecting_details(
    tmp_path,
):
    socket_path = tmp_path / "bridge.sock"
    server = await _bridge.start_bridge(
        socket_path, {"FINAL_VAR": lambda *_args: None}, "expected-token"
    )
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        payload = json.dumps(
            {
                "token": "wrong-token",
                "helper": "FINAL_VAR",
                "args": [],
                "kwargs": {},
            }
        ).encode()
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
        size = struct.unpack(">I", await reader.readexactly(4))[0]
        response = json.loads(await reader.readexactly(size))
        assert response == {
            "ok": False,
            "error": "bridge_error:PermissionError",
        }
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks required")
def test_bridge_result_reader_rejects_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-result.json"
    outside.write_text('{"stdout":"secret","error":null}')
    (tmp_path / "result.json").symlink_to(outside)
    assert _bridge.read_result(tmp_path) == ("", None, False)
