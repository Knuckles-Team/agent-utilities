"""LANE H-driver: exit-code contract for ``scripts/full_validation_harness.py``.

The driver's whole job is to run its 7 stages in order and stop at the first
failure, returning the 1-based index of that stage (0 = every stage that ran
passed) — exactly ``scripts/delegation_probe.py``'s own contract. These tests
prove that contract with fully monkeypatched/faked stage callables; none of
them require a live cluster, a live KG engine, or a live graph-os pod.

Also covers the stdin-safe sibling-module import fallback
(``_import_harness_modules``), since a wrong fallback would only ever be
discovered live, inside a pod, which is exactly the failure mode this test
exists to catch before that happens.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "full_validation_harness.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full_validation_harness", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: object) -> argparse.Namespace:
    base = dict(
        base_url="http://127.0.0.1:8080",
        mcp_url="http://127.0.0.1:8004/mcp",
        tenant_slug="homelab",
        skill="",
        server="",
        tool="",
        transport="streamable-http",
        modules_dir="",
        delegation_probe_path="",
        delegation_probe_args=[],
        stop_after=None,
        timeout=15.0,
        json=False,
        traceback=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch_all_stages_pass(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Monkeypatch every stage callable to succeed; return the call-order log."""
    calls: list[str] = []

    def _sync(name: str):
        def _fn(_a: argparse.Namespace) -> str:
            calls.append(name)
            return f"{name} ok"

        return _fn

    def _async(name: str):
        async def _fn(_a: argparse.Namespace) -> str:
            calls.append(name)
            return f"{name} ok"

        return _fn

    monkeypatch.setattr(module, "_stage_mcp_handshake", _sync("mcp_handshake"))
    monkeypatch.setattr(module, "_stage_browser", _sync("browser"))
    monkeypatch.setattr(module, "_stage_api_gateway", _sync("api_gateway"))
    monkeypatch.setattr(module, "_stage_admission", _sync("admission"))
    monkeypatch.setattr(module, "_stage_delegation", _async("delegation"))
    monkeypatch.setattr(module, "_stage_mcp_tool_catalog", _async("mcp_tool_catalog"))
    monkeypatch.setattr(module, "_stage_kg_general", _async("kg_general"))
    return calls


# ---------------------------------------------------------------------------
# Exit-code contract
# ---------------------------------------------------------------------------
def test_all_stages_pass_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    calls = _patch_all_stages_pass(module, monkeypatch)

    rc = asyncio.run(module.run(_args()))

    assert rc == 0
    assert calls == list(module.STAGES)


@pytest.mark.parametrize(
    ("failing_stage", "expected_code"),
    [
        ("mcp_handshake", 1),
        ("browser", 2),
        ("api_gateway", 3),
        ("admission", 4),
        ("delegation", 5),
        ("mcp_tool_catalog", 6),
        ("kg_general", 7),
    ],
)
def test_first_failing_stage_sets_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, failing_stage: str, expected_code: int
) -> None:
    module = _module()
    calls = _patch_all_stages_pass(module, monkeypatch)

    is_async_stage = failing_stage in {"delegation", "mcp_tool_catalog", "kg_general"}
    attr = f"_stage_{failing_stage}"

    async def _fail_async(_a: argparse.Namespace) -> str:
        calls.append(failing_stage)
        raise RuntimeError(f"{failing_stage} deliberately failed")

    def _fail_sync(_a: argparse.Namespace) -> str:
        calls.append(failing_stage)
        raise RuntimeError(f"{failing_stage} deliberately failed")

    monkeypatch.setattr(module, attr, _fail_async if is_async_stage else _fail_sync)

    rc = asyncio.run(module.run(_args()))

    assert rc == expected_code
    # Every stage up to and including the failing one ran; nothing after it did.
    idx = module.STAGES.index(failing_stage)
    assert calls == list(module.STAGES[: idx + 1])


def test_stop_after_prevents_later_stages_from_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls = _patch_all_stages_pass(module, monkeypatch)

    rc = asyncio.run(module.run(_args(stop_after="api_gateway")))

    assert rc == 0
    assert calls == ["mcp_handshake", "browser", "api_gateway"]


def test_failure_chain_reports_the_causal_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_chain`` must surface both an exception's message AND its __cause__,
    mirroring delegation_probe.py's own honesty contract for error reporting.
    """
    module = _module()
    _patch_all_stages_pass(module, monkeypatch)

    async def _fail(_a: argparse.Namespace) -> str:
        try:
            raise ValueError("root cause")
        except ValueError as exc:
            raise RuntimeError("wrapper failure") from exc

    monkeypatch.setattr(module, "_stage_kg_general", _fail)

    rc = asyncio.run(module.run(_args()))

    assert rc == 7


# ---------------------------------------------------------------------------
# Engine acquisition sharing (stages 6-7 acquire the engine ONCE)
# ---------------------------------------------------------------------------
def test_engine_is_acquired_once_and_shared_across_kg_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    acquire_calls = 0

    async def _fake_acquire_engine(_a: argparse.Namespace) -> object:
        nonlocal acquire_calls
        acquire_calls += 1
        return SimpleNamespace(name="fake-engine")

    class _FakeHarnessKg:
        async def mcp_tool_catalog(self, engine: object) -> dict:
            assert engine is not None
            return {"detail": "catalog ok"}

        async def kg_general(self, engine: object) -> dict:
            assert engine is not None
            return {"detail": "general ok"}

    monkeypatch.setattr(module, "_acquire_engine", _fake_acquire_engine)
    monkeypatch.setattr(module, "_stage_mcp_handshake", lambda _a: "mcp ok")
    monkeypatch.setattr(module, "_stage_browser", lambda _a: "browser ok")
    monkeypatch.setattr(module, "_stage_api_gateway", lambda _a: "api ok")
    monkeypatch.setattr(module, "_stage_admission", lambda _a: "admission ok")

    async def _delegation_pass(_a: argparse.Namespace) -> str:
        return "delegation ok"

    monkeypatch.setattr(module, "_stage_delegation", _delegation_pass)
    module._STATE["harness_kg"] = _FakeHarnessKg()

    rc = asyncio.run(module.run(_args()))

    assert rc == 0
    assert acquire_calls == 1, "engine must be acquired ONCE and shared, not per-stage"


# ---------------------------------------------------------------------------
# Stdin-safe sibling-module import resolution
# ---------------------------------------------------------------------------
def test_has_harness_modules_detects_the_three_sibling_files() -> None:
    module = _module()

    assert module._has_harness_modules(_REPO_ROOT) is True
    assert module._has_harness_modules(_REPO_ROOT / "nonexistent-dir") is False


def test_plain_import_succeeds_when_repo_root_is_already_on_sys_path() -> None:
    """The production topology (fleet NFS-mounts /au with PYTHONPATH=/au, per
    AGENTS.md) and this test's own pytest.ini ``pythonpath = .`` both put the
    repo root on sys.path before the driver ever runs — the plain-import
    branch must succeed in that case without needing --modules-dir at all.
    """
    module = _module()

    harness_mcp, harness_browser, harness_kg, repo_root = (
        module._import_harness_modules(None)
    )

    assert hasattr(harness_mcp, "stage_mcp_handshake")
    assert hasattr(harness_browser, "stage_browser")
    assert hasattr(harness_browser, "stage_api_gateway")
    assert hasattr(harness_browser, "stage_admission")
    assert hasattr(harness_kg, "mcp_tool_catalog")
    assert hasattr(harness_kg, "kg_general")
    assert repo_root.resolve() == _REPO_ROOT.resolve()


def _evict_scripts_package_from_import_machinery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a Python process where ``scripts`` truly cannot resolve — the
    real "piped via stdin with an unseeded PYTHONPATH" failure mode.

    Two independent things make ``scripts`` importable in THIS dev venv, and
    both have to be neutralized or the plain-import branch spuriously
    succeeds and the fallback path is never actually exercised:

    1. The repo root on ``sys.path`` (via pytest.ini's ``pythonpath = .``, or
       an inherited PYTHONPATH) — a normal ``PathFinder`` lookup.
    2. This venv's ``pip install -e .`` for ``agent_utilities`` registers a
       ``_EditableFinder`` class directly on ``sys.meta_path``
       (``.venv/lib/*/site-packages/__editable__.agent_utilities-*.pth``)
       whose baked-in ``MAPPING`` includes ``'scripts': '<repo-root>/scripts'``
       — a ``MetaPathFinder`` that resolves ``scripts.*`` unconditionally,
       independent of ``sys.path`` entirely. This is also almost certainly
       why the plain-import branch works reliably in the production pod too
       (per the "Fleet editable-install state" memory: the k8s MCP fleet +
       graph-os run editable, source-over-site-packages) — but a hermetic,
       non-editable (wheel) install has no such finder, which is exactly the
       scenario ``--modules-dir`` exists to recover.
    """
    filtered_path = [
        p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT.resolve()
    ]
    monkeypatch.setattr(sys, "path", filtered_path)

    filtered_meta_path = [
        f for f in sys.meta_path if getattr(f, "__name__", "") != "_EditableFinder"
    ]
    monkeypatch.setattr(sys, "meta_path", filtered_meta_path)

    for name in (
        "scripts",
        "scripts._harness_mcp",
        "scripts._harness_browser_api",
        "scripts._harness_kg_queries",
        "scripts.validate_mcp_config",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_modules_dir_fallback_works_when_repo_root_is_not_on_sys_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the exact failure mode this module's docstring documents:
    piped via stdin with PYTHONPATH NOT seeded (i.e. the repo root is not on
    sys.path and none of the three sibling modules were imported yet), so the
    plain-import branch must fail and ``--modules-dir`` must recover it.
    """
    module = _module()
    _evict_scripts_package_from_import_machinery(monkeypatch)

    # Sanity: with the repo root evicted, the plain import really does fail —
    # otherwise this test would pass for the wrong reason.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scripts._harness_mcp")

    harness_mcp, harness_browser, harness_kg, repo_root = (
        module._import_harness_modules(str(_REPO_ROOT))
    )

    assert hasattr(harness_mcp, "stage_mcp_handshake")
    assert hasattr(harness_browser, "stage_browser")
    assert hasattr(harness_kg, "mcp_tool_catalog")
    assert repo_root == Path(_REPO_ROOT)


def test_import_harness_modules_raises_actionable_error_when_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _evict_scripts_package_from_import_machinery(monkeypatch)
    # Point the production default (/au) somewhere that cannot possibly exist
    # so both the explicit candidate and the default candidate fail, and the
    # final RuntimeError path is exercised.
    monkeypatch.setattr(module, "_DEFAULT_MODULES_DIR", "/nonexistent-au-mount")

    with pytest.raises(RuntimeError, match="--modules-dir"):
        module._import_harness_modules("/also-nonexistent")


# ---------------------------------------------------------------------------
# Stage 5 (delegation) subprocess exit-code propagation
# ---------------------------------------------------------------------------
class _FakeStreamReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeProcess:
    def __init__(self, returncode: int, lines: list[bytes]) -> None:
        self.stdout = _FakeStreamReader(lines)
        self._returncode = returncode

    async def wait(self) -> int:
        return self._returncode


@pytest.mark.parametrize("returncode", [0, 1, 4])
def test_stage_delegation_propagates_subprocess_exit_code(
    monkeypatch: pytest.MonkeyPatch, returncode: int, tmp_path: Path
) -> None:
    module = _module()
    (tmp_path / "scripts").mkdir()
    probe_path = tmp_path / "scripts" / "delegation_probe.py"
    probe_path.write_text("# stub\n")
    module._STATE["repo_root"] = tmp_path

    async def _fake_create_subprocess_exec(*_cmd, **_kwargs):
        return _FakeProcess(returncode, [b"  PASS  config   [0.01s]\n"])

    monkeypatch.setattr(
        module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    a = _args()
    if returncode == 0:
        detail = asyncio.run(module._stage_delegation(a))
        assert "exited 0" in detail
    else:
        with pytest.raises(RuntimeError, match=f"exited {returncode}"):
            asyncio.run(module._stage_delegation(a))


def test_stage_delegation_raises_when_probe_script_missing(tmp_path: Path) -> None:
    module = _module()
    module._STATE["repo_root"] = tmp_path  # no delegation_probe.py written here

    with pytest.raises(RuntimeError, match="delegation_probe.py not found"):
        asyncio.run(module._stage_delegation(_args()))


# ---------------------------------------------------------------------------
# FIX LANE 11 — DEFECT A: stage 1 (mcp_handshake) bearer acquisition/threading
# ---------------------------------------------------------------------------
def test_acquire_mcp_auth_token_returns_token_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda cfg: "fake-bearer-token",
    )

    assert module._acquire_mcp_auth_token() == "fake-bearer-token"


def test_acquire_mcp_auth_token_is_optional_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquisition failing must NOT raise — the stage falls back to an
    unauthenticated handshake so a deployment with no auth in front of /mcp
    keeps working; the live endpoint's own response is the real signal."""
    module = _module()

    def _boom(cfg: object) -> str:
        raise RuntimeError("neither KG_AUTH_TOKEN_REF nor KG_IDENTITY_OAUTH2 set")

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        _boom,
    )

    assert module._acquire_mcp_auth_token() is None


def test_stage_mcp_handshake_threads_the_acquired_token_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_stage_mcp_handshake` must pass whatever `_acquire_mcp_auth_token()`
    returns straight into `harness_mcp.stage_mcp_handshake(..., auth_token=)`."""
    module = _module()
    monkeypatch.setattr(module, "_acquire_mcp_auth_token", lambda: "threaded-token")

    captured: dict = {}

    class _FakeHarnessMcp:
        @staticmethod
        def stage_mcp_handshake(mcp_url: str, *, timeout: float, auth_token=None):
            captured["mcp_url"] = mcp_url
            captured["timeout"] = timeout
            captured["auth_token"] = auth_token
            return SimpleNamespace(
                tool_count=5,
                protocol_version="2025-06-18",
                server_name="fake",
                server_version="1",
                session_id="sess",
                tool_names_sample=("ask",),
            )

    module._STATE["harness_mcp"] = _FakeHarnessMcp()

    detail = module._stage_mcp_handshake(_args())

    assert captured["auth_token"] == "threaded-token"
    assert captured["mcp_url"] == "http://127.0.0.1:8004/mcp"
    assert "tools=5" in detail


def test_stage_mcp_handshake_passes_none_when_acquisition_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_acquire_mcp_auth_token", lambda: None)

    captured: dict = {}

    class _FakeHarnessMcp:
        @staticmethod
        def stage_mcp_handshake(mcp_url: str, *, timeout: float, auth_token=None):
            captured["auth_token"] = auth_token
            return SimpleNamespace(
                tool_count=1,
                protocol_version="2025-06-18",
                server_name="fake",
                server_version="1",
                session_id=None,
                tool_names_sample=("ask",),
            )

    module._STATE["harness_mcp"] = _FakeHarnessMcp()

    module._stage_mcp_handshake(_args())

    assert captured["auth_token"] is None
