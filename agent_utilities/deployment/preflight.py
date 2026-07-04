#!/usr/bin/python
"""Dependency preflight — *can this host run the deployment I picked?*

Where ``doctor`` inspects an already-installed deployment, ``preflight`` answers
the question that comes first: are the host's *tools and runtimes* in place for a
given profile and the optional UI components the operator chose. It is the gate
the one-link installer (``scripts/install.sh``) and the genesis skills run before
they touch anything.

Design notes that shaped the checks:

* **No Rust by default.** ``epistemic-graph`` ships as a prebuilt ``maturin``
  ``bindings="bin"`` wheel, so ``pip install agent-utilities`` drops the
  ``epistemic-graph-server`` binary straight into ``$VENV/bin``. Rust is only a
  *fallback* — needed when no prebuilt wheel exists for the host arch/libc, or in
  an air-gapped install. The engine check reflects that: missing binary → "install
  the wheel"; ``cargo`` presence is reported as informational, never required.
* **Profile-scoped.** Docker is only needed once you leave the ``tiny`` (zero-infra)
  profile. ``tiny`` needs nothing but Python.
* **Component-scoped.** Node+pnpm are only checked when ``agent-webui`` is selected;
  Qt system libs + a display only when ``geniusbot`` is selected. The core deploy
  never drags those in.

Reuses :func:`agent_utilities.deployment.doctor._result` so a preflight report has
the same shape (and same human/JSON renderers) as a doctor report.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

from .doctor import _RANK, _result

# Profiles that require a container runtime (everything above zero-infra tiny).
_DOCKER_PROFILES = {"single-node-prod", "enterprise"}
PROFILES = ("tiny", "single-node-prod", "enterprise")
COMPONENTS = ("agent-terminal-ui", "agent-webui", "geniusbot")


def _os_family() -> str:
    """Coarse OS family for per-OS remediation hints: ``windows`` / ``macos`` /
    ``linux`` / ``other`` (CONCEPT:AU-OS.deployment.cross-platform-locks-plus)."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def _per_os_install_hint(tool: str) -> str:
    """Per-OS package-manager hint for installing ``tool`` (best-effort, advisory).

    Keeps remediation platform-aware so a Windows/macOS operator isn't told to run
    ``apt-get``. Pure string formatting — no shelling out.
    """
    fam = _os_family()
    table = {
        "docker": {
            "linux": "install Docker Engine (`curl -fsSL https://get.docker.com | sh`); enterprise also needs Swarm",
            "macos": "install Docker Desktop (`brew install --cask docker`) and start it",
            "windows": "install Docker Desktop (`winget install Docker.DockerDesktop` or `choco install docker-desktop`) with the WSL2 backend",
        },
        "rust": {
            "linux": "install via rustup (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)",
            "macos": "install via rustup (`brew install rustup-init && rustup-init`) or the rustup script",
            "windows": "install via rustup (`winget install Rustlang.Rustup`); the MSVC build tools are also required",
        },
        "node": {
            "linux": "install Node>=18 (e.g. `nvm install 18` or your distro package) and `corepack enable`",
            "macos": "install Node>=18 (`brew install node`) and `corepack enable`",
            "windows": "install Node>=18 (`winget install OpenJS.NodeJS.LTS`) and `corepack enable`",
        },
    }
    return table.get(tool, {}).get(fam, f"install {tool} for your platform")


# ── individual checks (each returns one _result; never raises) ──────────────
def _check_python() -> dict[str, Any]:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 15)
    detail = f"Python {platform.python_version()} (need >=3.11,<3.15)"
    if ok:
        return _result("python", "ok", detail)
    return _result(
        "python",
        "fail",
        detail,
        remediation="install Python 3.11, 3.12, 3.13 or 3.14 (e.g. via pyenv/uv)",
    )


def _check_installer() -> dict[str, Any]:
    uv = shutil.which("uv")
    pip = shutil.which("pip") or shutil.which("pip3")
    if uv:
        return _result("installer", "ok", f"uv present ({uv})")
    if pip:
        return _result("installer", "ok", f"pip present ({pip}); uv recommended")
    return _result(
        "installer",
        "fail",
        "neither uv nor pip found",
        remediation="install uv (https://astral.sh/uv) or ensure pip is on PATH",
    )


def _engine_binary_path() -> str | None:
    candidate = Path(sys.executable).parent / "epistemic-graph-server"
    if candidate.exists():
        return str(candidate)
    return shutil.which("epistemic-graph-server")


def _engine_binary_tier(server_path: str) -> str:
    """Probe the engine binary's tier/capabilities (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision).

    A too-lean wheel may lack the supervised idle-shutdown contract the resolver
    relies on. Introspect ``--help`` once and report whether
    ``--idle-shutdown-secs`` is advertised so an older/leaner binary is flagged.
    Best-effort: returns ``"unknown"`` if the binary can't be introspected.
    """
    try:
        out = subprocess.run(  # nosec B603 — fixed argv, our own binary
            [server_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        haystack = f"{out.stdout}\n{out.stderr}"
    except Exception:  # noqa: BLE001
        return "unknown"
    if "--idle-shutdown-secs" in haystack:
        return "supervised"
    return "lean"


def _check_engine() -> dict[str, Any]:
    """Engine binary presence + tier — wheel-first, Rust only as a fallback."""
    found = _engine_binary_path()
    cargo = shutil.which("cargo")
    if found:
        tier = _engine_binary_tier(found)
        if tier == "lean":
            return _result(
                "engine_binary",
                "warn",
                f"epistemic-graph-server present ({found}) but does NOT advertise "
                "`--idle-shutdown-secs` — reference-counted idle shutdown "
                "(CONCEPT:AU-OS.deployment.engine-resolver-auto-provision) is unavailable; an autostarted engine will run "
                "persistently. Upgrade the wheel for supervised idle shutdown.",
                remediation="upgrade: `pip install -U agent-utilities` (lean/older engine binary)",
            )
        suffix = (
            "; supports supervised idle shutdown"
            if tier == "supervised"
            else " (tier not introspectable)"
        )
        return _result(
            "engine_binary",
            "ok",
            f"epistemic-graph-server present ({found}); no Rust needed{suffix}",
        )
    # Not installed yet — that's expected pre-install. The wheel provides it.
    cargo_note = (
        f"Rust toolchain present at {cargo} (fallback build available)"
        if cargo
        else f"Rust not installed — only needed if no prebuilt wheel exists for this platform ({_per_os_install_hint('rust')})"
    )
    return _result(
        "engine_binary",
        "warn",
        "epistemic-graph-server not found yet (provided by the agent-utilities wheel)",
        remediation=(
            "install the prebuilt wheel: `pip install agent-utilities` (or `uv tool install "
            f"agent-utilities`). {cargo_note}."
        ),
    )


def _check_docker(profile: str) -> dict[str, Any]:
    docker = shutil.which("docker")
    if profile not in _DOCKER_PROFILES:
        return _result(
            "docker", "skip", f"not required for profile {profile!r} (zero-infra)"
        )
    if docker:
        return _result("docker", "ok", f"docker present ({docker})")
    return _result(
        "docker",
        "fail",
        f"profile {profile!r} needs a container runtime but docker is not on PATH",
        remediation=_per_os_install_hint("docker"),
        skill="infrastructure-orchestrator",
    )


def _node_version() -> tuple[int, ...] | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return tuple(int(p) for p in out.lstrip("v").split(".") if p.isdigit())
    except Exception:  # noqa: BLE001
        return ()


def _check_webui() -> dict[str, Any]:
    ver = _node_version()
    pnpm = shutil.which("pnpm")
    if ver is None:
        return _result(
            "agent-webui",
            "fail",
            "Node.js not found (agent-webui builds with Vite on Node)",
            remediation="install Node.js >=18 and `corepack enable` / `npm i -g pnpm`",
        )
    node_ok = (not ver) or ver[0] >= 18
    if node_ok and pnpm:
        vtxt = ".".join(map(str, ver)) if ver else "present"
        return _result("agent-webui", "ok", f"Node {vtxt} + pnpm present")
    missing = []
    if not node_ok:
        missing.append(f"Node>=18 (have {'.'.join(map(str, ver))})")
    if not pnpm:
        missing.append("pnpm (the project's package manager)")
    return _result(
        "agent-webui",
        "warn",
        "Node present but " + ", ".join(missing),
        remediation="install Node>=18 and pnpm 10.x (`corepack enable`)",
    )


def _check_geniusbot() -> dict[str, Any]:
    """geniusbot is a PySide6/Qt desktop app — needs a display + Qt system libs."""
    has_display = bool(setting("DISPLAY", "") or setting("WAYLAND_DISPLAY", ""))
    # Best-effort probe for the Qt platform libs on Linux.
    libgl = None
    if sys.platform.startswith("linux"):
        ldconfig = shutil.which("ldconfig")
        if ldconfig:
            try:
                out = subprocess.run(
                    [ldconfig, "-p"], capture_output=True, text=True, timeout=5
                ).stdout
                libgl = "libGL.so" in out
            except Exception:  # noqa: BLE001
                libgl = None
    if not has_display:
        return _result(
            "geniusbot",
            "fail",
            "no X11/Wayland display ($DISPLAY/$WAYLAND_DISPLAY unset) — geniusbot is a desktop app",
            remediation=(
                "run on a desktop session, or skip geniusbot for headless hosts "
                "(use agent-webui/agent-terminal-ui instead)"
            ),
        )
    if libgl is False:
        return _result(
            "geniusbot",
            "warn",
            "display present but Qt libs (libGL) not detected",
            remediation="install Qt runtime libs: libGL, libxcb, libxkbcommon, fontconfig",
        )
    return _result("geniusbot", "ok", "display present; Qt desktop deploy viable")


def _check_terminal_ui() -> dict[str, Any]:
    # Thin httpx client — only needs the Python already checked above.
    return _result(
        "agent-terminal-ui", "ok", "thin client (Python only); containerizable"
    )


_COMPONENT_CHECKS: dict[str, Callable[[], dict[str, Any]]] = {
    "agent-webui": _check_webui,
    "geniusbot": _check_geniusbot,
    "agent-terminal-ui": _check_terminal_ui,
}


def run_preflight(
    profile: str = "tiny",
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Run the host dependency preflight for a profile + optional UI components.

    Args:
        profile: ``tiny`` | ``single-node-prod`` | ``enterprise``.
        components: any of ``agent-webui`` / ``geniusbot`` / ``agent-terminal-ui``.
    """
    components = components or []
    results: list[dict[str, Any]] = [
        _check_python(),
        _check_installer(),
        _check_engine(),
        _check_docker(profile),
    ]
    for comp in components:
        fn = _COMPONENT_CHECKS.get(comp)
        if fn is None:
            results.append(
                _result(comp, "skip", f"unknown component {comp!r} (no preflight)")
            )
            continue
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            results.append(_result(comp, "error", f"check raised: {exc}"))

    worst = max((_RANK[r["status"]] for r in results), default=0)
    overall = {0: "ready", 1: "warnings", 2: "blocked"}[worst]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r["name"] for r in results if r["status"] in ("warn", "fail", "error")]
    summary = (
        f"Host is ready for profile {profile!r}."
        if overall == "ready"
        else f"{overall}: attend to {bad} before deploying profile {profile!r}."
    )
    return {
        "status": overall,
        "profile": profile,
        "components": components,
        "counts": counts,
        "checks": results,
        "summary": summary,
    }
