"""Unit tests for the computer-use agent tools (CONCEPT:ORCH-1.85)."""

import base64
import struct
import types

import pytest
from pydantic_ai import ToolReturn

from agent_utilities.runtime.events import ComputerUseAction, ScreenObservation
from agent_utilities.tools.computer_use_tools import capture_screen, gui_action


def _png(w: int, h: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr


class _FakeWorkspace:
    def __init__(self, obs):
        self._obs = obs
        self.acted = []

    async def act(self, action):
        self.acted.append(action)
        return self._obs


def _ctx(workspace):
    deps = types.SimpleNamespace(workspace=workspace)
    return types.SimpleNamespace(deps=deps)


async def test_capture_returns_image_and_som():
    img = base64.b64encode(_png(800, 600)).decode()
    obs = ScreenObservation(
        image_b64=img,
        width=800,
        height=600,
        elements=[
            {
                "id": "el-0",
                "role": "push button",
                "name": "Save",
                "x": 10,
                "y": 20,
                "w": 30,
                "h": 12,
            },
        ],
    )
    result = await capture_screen(_ctx(_FakeWorkspace(obs)))
    assert isinstance(result, ToolReturn)
    assert "el-0" in result.return_value and "Save" in result.return_value
    # The screenshot rides along as multimodal content for vision models.
    assert any(getattr(c, "media_type", "") == "image/png" for c in result.content)


async def test_capture_no_workspace():
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(workspace=None))
    assert "No developer workspace" in await capture_screen(ctx)


async def test_gui_action_validates_op():
    ws = _FakeWorkspace(ScreenObservation())
    out = await gui_action(_ctx(ws), op="frobnicate")
    assert "unknown op" in out
    assert not ws.acted  # invalid op never reaches the workspace


async def test_gui_action_dispatches_and_reports():
    ws = _FakeWorkspace(ScreenObservation())
    out = await gui_action(_ctx(ws), op="click", element_id="el-3")
    assert out == "click ok"
    assert isinstance(ws.acted[0], ComputerUseAction)
    assert ws.acted[0].op == "click" and ws.acted[0].element_id == "el-3"


async def test_gui_action_surfaces_error():
    ws = _FakeWorkspace(ScreenObservation(error="boom"))
    out = await gui_action(_ctx(ws), op="type", text="hi")
    assert "failed" in out and "boom" in out


def test_build_computer_use_agent_exposes_tools():
    from pydantic_ai.models.test import TestModel

    from agent_utilities.orchestration.computer_use_agent import (
        build_computer_use_agent,
    )

    agent = build_computer_use_agent(TestModel())
    assert agent.name == "computer-use-agent"
    names = set(agent._function_toolset.tools.keys())
    assert {"capture_screen", "gui_action"} <= names


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"
