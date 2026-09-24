"""SandboxPort selection over the RLM backends and the real child-process launcher."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_utilities.layers.cli_process import SubprocessLauncher, confine
from agent_utilities.layers.contracts import HarnessNotInstalled, SandboxBoundaryError
from agent_utilities.layers.ports import SandboxLease, SandboxRequirements
from agent_utilities.layers.sandbox import RouterSandboxPort
from agent_utilities.rlm.sandboxes.base import (
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxRejected,
    SandboxResult,
)


def _caps(rank: int, **overrides: bool) -> SandboxCapabilities:
    fields = {
        "host_callbacks": False,
        "third_party_libs": False,
        "classes": True,
        "full_stdlib": True,
        "network": False,
        "isolated": True,
        **overrides,
    }
    return SandboxCapabilities(preference_rank=rank, **fields)


@dataclass
class _Backend(Sandbox):
    name: str
    capabilities: SandboxCapabilities
    available: bool = True

    def is_available(self) -> bool:
        return self.available

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        if "class" in code:
            raise SandboxRejected(self.name, "no classes here")
        return SandboxResult(updated_vars={"x": 1}, stdout=f"ran {code}")


def _port(tmp_path: Path) -> RouterSandboxPort:
    return RouterSandboxPort(
        str(tmp_path),
        backends=[
            _Backend("monty", _caps(0, classes=False)),
            _Backend("wasm", _caps(10)),
            _Backend("docker", _caps(20, third_party_libs=True, network=True)),
            _Backend("local", _caps(99, isolated=False)),
            _Backend("firecracker", _caps(25), available=False),
        ],
    )


def test_lease_picks_the_cheapest_capable_backend_and_records_why(tmp_path) -> None:
    lease = _port(tmp_path).lease(SandboxRequirements(classes=True))
    assert lease.backend == "wasm"
    assert Path(lease.workspace).is_dir()
    excluded = lease.reason["excluded"]
    assert excluded["monty"] == "no class support"
    assert excluded["local"] == "no isolation boundary"
    assert excluded["firecracker"] == "unavailable"
    assert excluded["docker"] == "network egress not allowed"


def test_pin_deny_and_egress_requirements(tmp_path) -> None:
    port = _port(tmp_path)
    egress = port.lease(SandboxRequirements(network="egress", third_party_libs=True))
    assert egress.backend == "docker"
    assert port.lease(SandboxRequirements(pin="wasm")).backend == "wasm"
    with pytest.raises(SandboxBoundaryError, match="no sandbox backend"):
        port.lease(SandboxRequirements(deny=frozenset({"monty", "wasm", "docker"})))


async def test_execute_release_and_health(tmp_path) -> None:
    port = _port(tmp_path)
    lease = port.lease(SandboxRequirements())
    assert (await port.execute(lease, "1 + 1"))["stdout"] == "ran 1 + 1"
    with pytest.raises(SandboxBoundaryError, match="rejected"):
        await port.execute(lease, "class A: pass")
    port.release(lease)
    assert not Path(lease.workspace).exists()
    assert port.health()["firecracker"] is False
    outside = SandboxLease(lease_id="x", backend="wasm", workspace="/etc")
    with pytest.raises(SandboxBoundaryError):
        port.release(outside)


def test_confine_refuses_escape(tmp_path) -> None:
    assert confine(str(tmp_path), tmp_path / "a") == (tmp_path / "a").resolve()
    with pytest.raises(SandboxBoundaryError):
        confine(str(tmp_path), tmp_path / ".." / "elsewhere")
    with pytest.raises(SandboxBoundaryError):
        confine(None, tmp_path)


async def test_subprocess_launcher_streams_lines_and_stderr(tmp_path) -> None:
    launcher = SubprocessLauncher()
    script = (
        "import sys\n"
        "data = sys.stdin.read()\n"
        "print('{\"echo\": \"' + data + '\"}')\n"
        "print('second')\n"
        "sys.stderr.write('warned')\n"
    )
    process = await launcher.launch(
        [sys.executable, "-c", script], cwd=str(tmp_path), env={}, stdin_text="hi"
    )
    lines = [line async for line in process.lines()]
    assert lines == ['{"echo": "hi"}', "second"]
    assert await process.wait() == 0
    assert process.stderr_tail() == "warned"


async def test_subprocess_launcher_terminates_a_hung_child(tmp_path) -> None:
    process = await SubprocessLauncher().launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(tmp_path),
        env={},
        stdin_text="",
    )
    await process.terminate()
    assert await process.wait() != 0


def test_missing_binary_is_typed() -> None:
    with pytest.raises(HarnessNotInstalled):
        SubprocessLauncher().resolve("definitely-not-a-harness-binary")
