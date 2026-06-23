"""Unit tests for the computer-use driver tier (CONCEPT:ECO-4.93 / ORCH-1.84).

Covers the governed action seam (capture read-only vs input gated), dispatcher
routing to the driver, the Null floor, and the ContainerExec driver's pure
mapping/parsing helpers — all without a real container.
"""

import base64
import struct
import zlib

import pytest

from agent_utilities.runtime.bridge import ActionDispatcher
from agent_utilities.runtime.computer_use_tier import (
    ContainerExecComputerUseDriver,
    NullComputerUseDriver,
    _png_dimensions,
    _xdotool_cmd,
)
from agent_utilities.runtime.events import (
    ComputerUseAction,
    ScreenObservation,
    mutating_action_name,
)


def _png(width: int, height: int) -> bytes:
    """Minimal valid PNG header (IHDR) for dimension parsing."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def test_capture_is_read_only_input_is_gated():
    # The capture op bypasses the policy gate; every input op is gated.
    assert mutating_action_name(ComputerUseAction(op="capture")) is None
    for op in ("click", "type", "key", "scroll", "drag", "double_click"):
        assert (
            mutating_action_name(ComputerUseAction(op=op)) == "workspace.computer_use"
        )


def test_xdotool_mapping():
    assert _xdotool_cmd(ComputerUseAction(op="click", x=10, y=20), 10, 20) == [
        "xdotool",
        "mousemove",
        "10",
        "20",
        "click",
        "1",
    ]
    assert _xdotool_cmd(ComputerUseAction(op="double_click"), 5, 6)[-3:] == [
        "--repeat",
        "2",
        "1",
    ]
    assert _xdotool_cmd(ComputerUseAction(op="right_click"), 5, 6)[-1] == "3"
    assert _xdotool_cmd(ComputerUseAction(op="type", text="hi"), None, None) == [
        "xdotool",
        "type",
        "--clearmodifiers",
        "hi",
    ]
    assert _xdotool_cmd(ComputerUseAction(op="key", keys="ctrl+l"), None, None) == [
        "xdotool",
        "key",
        "--clearmodifiers",
        "ctrl+l",
    ]


def test_png_dimensions():
    assert _png_dimensions(_png(1280, 800)) == (1280, 800)
    assert _png_dimensions(b"not a png") == (0, 0)


async def test_null_driver_advertises_unprovisioned():
    obs = await NullComputerUseDriver().run(ComputerUseAction(op="capture"))
    assert isinstance(obs, ScreenObservation)
    assert "not provisioned" in obs.error


async def test_dispatcher_routes_to_driver():
    class _Recorder:
        async def run(self, action):
            return ScreenObservation(image_b64="abc", session_id="s1")

    obs = await ActionDispatcher().dispatch(
        ComputerUseAction(op="capture"),
        backend=None,
        state=None,
        computer_use=_Recorder(),
    )
    assert isinstance(obs, ScreenObservation)
    assert obs.image_b64 == "abc"


class _FakeManager:
    """Stands in for container-manager's DockerManager.exec_in_container."""

    def __init__(self, png: bytes, a11y: str):
        self._png = png
        self._a11y = a11y
        self.calls = []

    def exec_in_container(self, container_id, command, detach=False, binary=False):
        self.calls.append((command, binary))
        if binary:  # maim screenshot path
            return {"exit_code": 0, "output_b64": base64.b64encode(self._png).decode()}
        if command[-1] == "a11y-dump":
            return {"exit_code": 0, "output": self._a11y}
        return {"exit_code": 0, "output": ""}


async def test_capture_builds_observation_and_grounding():
    a11y = (
        '{"elements": [{"role": "push button", "name": "Save", '
        '"x": 100, "y": 200, "w": 40, "h": 20}]}'
    )
    driver = ContainerExecComputerUseDriver("c1", session_id="sess")
    driver._manager = _FakeManager(_png(640, 480), a11y)

    obs = await driver.run(ComputerUseAction(op="capture"))
    assert isinstance(obs, ScreenObservation)
    assert obs.width == 640 and obs.height == 480
    assert obs.session_id == "sess"
    assert len(obs.elements) == 1
    assert obs.elements[0]["id"] == "el-0"
    # A click by element_id resolves to the element's center via the grounding cache.
    assert driver._grounding["el-0"] == (120, 210)
    x, y = driver._resolve_target(ComputerUseAction(op="click", element_id="el-0"))
    assert (x, y) == (120, 210)


async def test_input_op_execs_and_succeeds():
    driver = ContainerExecComputerUseDriver("c1")
    fake = _FakeManager(_png(10, 10), '{"elements": []}')
    driver._manager = fake
    obs = await driver.run(ComputerUseAction(op="click", x=3, y=4))
    assert isinstance(obs, ScreenObservation)
    assert not obs.error
    # The exec drops to the sandbox user and sets DISPLAY, and contains the xdotool call.
    cmd, _binary = fake.calls[-1]
    assert cmd[:3] == ["runuser", "-u", "sandbox"]
    assert "DISPLAY=:1" in cmd and "xdotool" in cmd


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"
