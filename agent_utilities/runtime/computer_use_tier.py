"""CONCEPT:ECO-4.93 / ORCH-1.84 — Computer-use driver tier for the workspace runtime.

A pluggable :class:`ComputerUseDriver` lets an agent drive a GUI desktop (a
``gui-sandbox`` container) the same way the optional :class:`BrowserDriver`
(ECO-4.44) lets it browse — without dragging GUI/native deps into core. The
default :class:`NullComputerUseDriver` advertises the tier but does nothing, so
the ``computer_use`` action surface exists everywhere; a real driver is attached
only where computer-use is provisioned.

:class:`ContainerExecComputerUseDriver` is the real actuator. It **reuses
container-manager-mcp's** manager abstraction (``create_manager`` +
``resolve_host_from_inventory``) rather than re-implementing exec-over-ssh, so it
drives a sandbox on *any* inventory host over the ssh:// docker/podman socket with
**no in-container daemon and no published port** — every action is a
``docker exec`` of native X11 tools (``xdotool``/``wmctrl``) and every observation a
``maim`` screenshot + ``a11y-dump`` accessibility tree. Because container-manager
abstracts Docker *and* Podman, podman-rootless sandboxes work for free.

Input ops are gated by ``ActionPolicy`` as ``workspace.computer_use`` (OS-5.57) at
the workspace ``act()`` seam; ``op="capture"`` is read-only and bypasses the gate.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .events import ComputerUseAction, Observation, ScreenObservation

if TYPE_CHECKING:
    pass


@runtime_checkable
class ComputerUseDriver(Protocol):
    async def run(self, action: ComputerUseAction) -> Observation: ...


class NullComputerUseDriver:
    """The always-available floor: advertises the tier but performs no actuation."""

    name = "null"

    async def run(self, action: ComputerUseAction) -> Observation:
        return ScreenObservation(
            error="computer-use tier not provisioned "
            "(attach a ComputerUseDriver to enable ECO-4.93)",
        )


def _xdotool_cmd(action: ComputerUseAction, x: int | None, y: int | None) -> list[str]:
    """Map a mutating computer-use op to a native X11 command (run under DISPLAY=:1)."""
    op = action.op
    move = ["mousemove", str(x), str(y)] if x is not None and y is not None else []
    if op == "move":
        return ["xdotool", *move]
    if op in ("click", "double_click", "right_click"):
        button = "3" if op == "right_click" else str(action.button)
        repeat = ["--repeat", "2"] if op == "double_click" else []
        return ["xdotool", *move, "click", *repeat, button]
    if op == "type":
        return ["xdotool", "type", "--clearmodifiers", action.text]
    if op == "key":
        # xdotool uses '+' for chords, matching the action's "ctrl+l" convention.
        return ["xdotool", "key", "--clearmodifiers", action.keys]
    if op == "scroll":
        # Button 4 = up, 5 = down; repeat by the magnitude of dy.
        button = "5" if action.dy >= 0 else "4"
        clicks = max(1, abs(action.dy))
        return ["xdotool", *move, "click", "--repeat", str(clicks), button]
    if op == "drag":
        return [
            "xdotool",
            *move,
            "mousedown",
            str(action.button),
            "mousemove",
            str((x or 0) + action.dx),
            str((y or 0) + action.dy),
            "mouseup",
            str(action.button),
        ]
    raise ValueError(f"unsupported computer-use op: {op}")


class ContainerExecComputerUseDriver:
    """Drive a ``gui-sandbox`` container via ``docker exec`` of native X11 tools.

    Reuses container-manager-mcp's manager (imported lazily so core never depends
    on it). ``container_id`` must be an already-running gui-sandbox; provisioning is
    the session/tool layer's job (``cm_container_operations action=run``).
    """

    name = "container-exec"

    def __init__(
        self,
        container_id: str,
        *,
        host: str | None = None,
        manager_type: str | None = None,
        session_id: str = "",
        engine: Any | None = None,
    ) -> None:
        self.container_id = container_id
        self.host = host
        self.manager_type = manager_type
        self.session_id = session_id
        self._engine = engine  # optional GraphComputeEngine for observe_screen ingest
        self._manager: Any | None = None
        # Grounding cache from the last capture: element_id -> (cx, cy) screen center.
        self._grounding: dict[str, tuple[int, int]] = {}

    def _manager_or_create(self) -> Any:
        if self._manager is None:
            try:
                from container_manager_mcp.container_manager import create_manager
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "computer-use requires the [computer-use] extra "
                    "(container-manager-mcp) to be installed"
                ) from exc
            self._manager = create_manager(self.manager_type, host=self.host)
        return self._manager

    async def _exec(self, command: list[str], binary: bool = False) -> dict:
        manager = self._manager_or_create()
        full = ["env", "DISPLAY=:1", *command]
        return await asyncio.to_thread(
            manager.exec_in_container, self.container_id, full, False, binary
        )

    async def run(self, action: ComputerUseAction) -> Observation:
        try:
            if action.op == "capture":
                return await self._capture()
            if action.op == "wait":
                await asyncio.sleep(max(0, action.duration_ms) / 1000.0)
                return ScreenObservation(session_id=self.session_id)
            x, y = self._resolve_target(action)
            cmd = _xdotool_cmd(action, x, y)
            res = await self._exec(cmd)
            if res.get("exit_code"):
                return ScreenObservation(
                    session_id=self.session_id,
                    error=f"{action.op} failed: {res.get('output')}",
                )
            # Input ops don't auto-screenshot; the agent captures when it wants to see.
            return ScreenObservation(session_id=self.session_id)
        except Exception as exc:  # noqa: BLE001 - surface as a typed observation
            return ScreenObservation(session_id=self.session_id, error=str(exc))

    def _resolve_target(
        self, action: ComputerUseAction
    ) -> tuple[int | None, int | None]:
        """Prefer ground-by-reference (element_id from the last capture) over raw x/y."""
        if action.element_id and action.element_id in self._grounding:
            return self._grounding[action.element_id]
        return action.x, action.y

    async def _capture(self) -> ScreenObservation:
        shot = await self._exec(["maim", "--format", "png"], binary=True)
        image_b64 = shot.get("output_b64") or ""
        tree = await self._exec(["a11y-dump"])
        elements = self._parse_elements(tree.get("output", ""))
        # Refresh the grounding cache so a later click(element_id=...) resolves.
        self._grounding = {
            el["id"]: (el["x"] + el["w"] // 2, el["y"] + el["h"] // 2)
            for el in elements
            if el.get("w") and el.get("h")
        }
        width = height = 0
        if image_b64:
            width, height = _png_dimensions(base64.b64decode(image_b64))
        obs = ScreenObservation(
            session_id=self.session_id,
            image_b64=image_b64,
            width=width,
            height=height,
            elements=elements,
        )
        await self._maybe_ingest(image_b64, elements)
        return obs

    @staticmethod
    def _parse_elements(raw: str) -> list[dict]:
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return []
        out = []
        for i, el in enumerate(data.get("elements", [])):
            out.append(
                {
                    "id": f"el-{i}",
                    "role": el.get("role", ""),
                    "name": el.get("name", ""),
                    "x": int(el.get("x", 0)),
                    "y": int(el.get("y", 0)),
                    "w": int(el.get("w", 0)),
                    "h": int(el.get("h", 0)),
                }
            )
        return out

    async def _maybe_ingest(self, image_b64: str, elements: list[dict]) -> None:
        """Feed the frame to the engine's observe_screen enrichment if available.

        Optional (KG-2.185): when an engine with ``observe_screen`` is wired, the
        PNG + accessibility tree are materialised as durable :UIElement nodes for
        cross-frame grounding. The driver works fully without it.
        """
        engine = self._engine
        if engine is None or not image_b64:
            return
        observe = getattr(engine, "observe_screen", None)
        if observe is None:
            return
        try:
            payload = json.dumps(
                {"session_id": self.session_id, "elements": elements}
            ).encode()
            await asyncio.to_thread(observe, base64.b64decode(image_b64), payload)
        except Exception:  # noqa: BLE001 - ingestion is best-effort, never blocks actuation
            return


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width/height from a PNG IHDR without pulling an image library."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    return 0, 0
