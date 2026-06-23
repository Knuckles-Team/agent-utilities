"""CONCEPT:ORCH-1.85 — Computer-use tools: the agent's GUI action surface.

These Pydantic-AI tools translate the model's tool calls into typed
``ComputerUseAction``s executed inside the agent's :class:`DevWorkspace`
(``ctx.deps.workspace``) against an attached :class:`ComputerUseDriver` (ECO-4.93).
The agent inherits provenance (every action mirrored to the KG, KG-2.64) and
governance (``workspace.computer_use``, OS-5.57) for free.

The loop is Observe→Ground→Decide→Act: ``capture_screen`` observes (and returns the
screenshot + a numbered element list the model grounds against); ``gui_action``
acts, preferring ``element_id`` (resolved to coordinates by the driver) over raw
pixels. Two tools is the whole surface — grounding rides on the capture's element
list rather than a separate query tool.
"""

from __future__ import annotations

import logging

from pydantic_ai import BinaryContent, RunContext, ToolReturn

from agent_utilities.harness.tracing import trace

from ..models import AgentDeps
from ..runtime.events import ComputerUseAction
from .versioning import tool_version

logger = logging.getLogger(__name__)

_NO_WS = "No developer workspace is attached to this agent (deps.workspace is None)."


def _ws(ctx: RunContext[AgentDeps]):
    return getattr(ctx.deps, "workspace", None)


def _element_lines(elements: list[dict]) -> str:
    """Render the grounded element list as a numbered SOM the model can click by id."""
    lines = []
    for el in elements:
        name = el.get("name") or ""
        label = f" {name!r}" if name else ""
        lines.append(
            f"[{el['id']}] {el.get('role', '')}{label} "
            f"@({el.get('x', 0)},{el.get('y', 0)}) {el.get('w', 0)}x{el.get('h', 0)}"
        )
    return "\n".join(lines)


@trace(name="capture_screen", trace_type="TOOL")
@tool_version("1.0.0")
async def capture_screen(ctx: RunContext[AgentDeps]) -> object:
    """Capture the sandbox desktop: returns a screenshot plus a numbered list of
    interactable UI elements. Click an element by its ``[el-N]`` id with ``gui_action``."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(ComputerUseAction(op="capture"))
    if getattr(obs, "error", ""):
        return f"capture failed: {obs.error}"
    som = _element_lines(getattr(obs, "elements", []) or [])
    summary = (
        f"Captured {obs.width}x{obs.height} desktop with "
        f"{len(obs.elements)} interactable elements:\n{som}"
    )
    image_b64 = getattr(obs, "image_b64", "")
    if not image_b64:
        return summary
    import base64

    # Return the pixels (for vision models) alongside the text SOM (which also lets
    # non-vision models ground via element ids). ToolReturn keeps both in one result.
    return ToolReturn(
        return_value=summary,
        content=[
            summary,
            BinaryContent(data=base64.b64decode(image_b64), media_type="image/png"),
        ],
    )


@trace(name="gui_action", trace_type="TOOL")
@tool_version("1.0.0")
async def gui_action(
    ctx: RunContext[AgentDeps],
    op: str,
    element_id: str = "",
    x: int | None = None,
    y: int | None = None,
    text: str = "",
    keys: str = "",
    button: int = 1,
    dx: int = 0,
    dy: int = 0,
) -> str:
    """Perform a GUI action on the sandbox desktop.

    op: one of click, double_click, right_click, move, type, key, scroll, drag, wait.
    Prefer ``element_id`` (from the last capture_screen) as the target; otherwise pass
    absolute ``x``/``y`` pixels. ``text`` is for op=type; ``keys`` (e.g. "ctrl+l") for
    op=key; ``dx``/``dy`` for scroll/drag deltas. Re-run capture_screen to see the result.
    """
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    _ann = ComputerUseAction.model_fields["op"].annotation
    valid_ops = set(getattr(_ann, "__args__", ()))
    if op not in valid_ops:
        return f"unknown op {op!r}; valid: {', '.join(sorted(valid_ops))}"
    obs = await ws.act(
        ComputerUseAction(
            # op is validated against the Literal above, so the cast is safe.
            op=op,  # type: ignore[arg-type]
            element_id=element_id,
            x=x,
            y=y,
            text=text,
            keys=keys,
            button=button,
            dx=dx,
            dy=dy,
        )
    )
    err = getattr(obs, "error", "")
    return f"{op} failed: {err}" if err else f"{op} ok"


COMPUTER_USE_TOOLS = [capture_screen, gui_action]
