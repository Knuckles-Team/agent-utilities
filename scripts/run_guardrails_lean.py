#!/usr/bin/env python3
"""Run the CI ``Guardrails`` gates inside a fresh CI-EQUIVALENT LEAN venv.

**Why this exists — closing the "passes-local / fails-CI" class.** The CI
``Guardrails`` workflow (``.github/workflows/guardrails.yml``) deliberately
installs a *lean* environment: ``pip install -e .`` plus only
``numpy pyyaml pytest rdflib pyshacl owlrl`` — it does **not** install the
``[agent]`` / ``[all]`` extras (which pull a circular ``universal-skills`` dep and
the heavy agent runtime). A full local dev install, by contrast, has everything.
So a guardrail gate that transitively imports an ``[agent]``-extra dependency
(e.g. ``pydantic_ai`` via ``agent_utilities.graph.__init__``) **passes locally but
dies in CI** with ``ModuleNotFoundError`` — exactly the failure mode the
Eval-corpus gate hit.

This runner reproduces CI's Guardrails environment **on the developer's machine**:
it builds a throwaway lean venv with the *exact* install from ``guardrails.yml``,
then executes the gate commands *derived from the same file* inside it. If a gate
would fail in CI's lean install, it fails here too — locally, before the push.

**DRY with ``guardrails.yml``.** The install package list and the gate command
list are both parsed out of ``.github/workflows/guardrails.yml`` at runtime, so
this script cannot drift from CI: add/remove/edit a gate in the workflow and this
runner follows automatically. Advisory steps (``continue-on-error: true``) are run
but never fail the run, mirroring CI semantics.

Wired into pre-commit as the ``guardrails-lean-parity`` hook
(``stages: [pre-push, manual]`` — it builds a venv, too slow for every commit)::

    pre-commit run guardrails-lean-parity --hook-stage manual --all-files
    # or, directly:
    python3 scripts/run_guardrails_lean.py

Exit 0 = every blocking gate passed in the lean env. 1 = a blocking gate failed
(or the env could not be built). Requires ``uv`` on PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "guardrails.yml"
_PY_VERSION = "3.12"  # matches actions/setup-python in guardrails.yml


@dataclass
class Gate:
    """A single guardrail step parsed from the workflow."""

    name: str
    command: str
    advisory: bool  # CI continue-on-error -> run but never fail the suite


def _load_workflow() -> dict:
    if not _WORKFLOW.is_file():
        sys.exit(f"ERROR: workflow not found: {_WORKFLOW}")
    with _WORKFLOW.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _steps(workflow: dict) -> list[dict]:
    jobs = workflow.get("jobs") or {}
    guardrails = jobs.get("guardrails") or {}
    return list(guardrails.get("steps") or [])


def _parse_install_targets(steps: list[dict]) -> list[str]:
    """Extract the ``pip install`` targets from the install step.

    Mirrors guardrails.yml's lean install EXACTLY (``-e .`` + the gate deps);
    skips the ``pip install --upgrade pip`` bootstrap line.
    """
    targets: list[str] = []
    for step in steps:
        run = step.get("run") or ""
        if "pip install" not in run:
            continue
        for line in run.splitlines():
            line = line.strip()
            if line.startswith("#") or not line.startswith("pip install"):
                continue
            args = line[len("pip install") :].split()
            if args == ["--upgrade", "pip"]:
                continue
            targets.extend(a for a in args if a != "--upgrade")
        if targets:
            break
    if not targets:
        sys.exit("ERROR: could not parse lean install targets from guardrails.yml")
    return targets


def _parse_gates(steps: list[dict]) -> list[Gate]:
    """Every run-step that invokes a gate script or pytest, in workflow order."""
    gates: list[Gate] = []
    for step in steps:
        run = (step.get("run") or "").strip()
        if not run or "pip install" in run:
            continue
        if not re.search(r"scripts/check_\w+\.py|-m pytest", run):
            continue
        gates.append(
            Gate(
                name=step.get("name", run.splitlines()[0]),
                command=run,
                advisory=bool(step.get("continue-on-error", False)),
            )
        )
    return gates


def _build_lean_venv(venv_dir: Path, targets: list[str]) -> Path:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("ERROR: `uv` is required on PATH to build the lean venv.")
    print(f"[lean] creating venv at {venv_dir} (python {_PY_VERSION})", flush=True)
    subprocess.run(
        [uv, "venv", "--python", _PY_VERSION, str(venv_dir)],
        check=True,
        cwd=_REPO_ROOT,
    )
    py = venv_dir / "bin" / "python"
    print(f"[lean] installing (mirrors guardrails.yml): {' '.join(targets)}", flush=True)
    subprocess.run(
        [uv, "pip", "install", "--python", str(py), *targets],
        check=True,
        cwd=_REPO_ROOT,
    )
    return py


def _gate_env(venv_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    # Drop any inherited interpreter pin so `python3` resolves to the lean venv.
    env.pop("PYTHONHOME", None)
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--venv",
        type=Path,
        default=None,
        help="venv path (default: a fresh temp dir, removed on exit).",
    )
    ap.add_argument(
        "--keep-venv",
        action="store_true",
        help="do not delete the venv on exit (debugging).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the derived install + gate list and exit (no venv built).",
    )
    args = ap.parse_args()

    workflow = _load_workflow()
    steps = _steps(workflow)
    targets = _parse_install_targets(steps)
    gates = _parse_gates(steps)

    if args.list:
        print("lean install targets:", " ".join(targets))
        print("gates (derived from guardrails.yml):")
        for g in gates:
            print(f"  - {'[advisory] ' if g.advisory else ''}{g.name}")
        return 0

    cleanup = False
    if args.venv is not None:
        venv_dir = args.venv
    else:
        venv_dir = Path(tempfile.mkdtemp(prefix="guardrails-lean-"))
        cleanup = not args.keep_venv

    failures: list[str] = []
    try:
        _build_lean_venv(venv_dir, targets)
        env = _gate_env(venv_dir)
        print(f"\n[lean] running {len(gates)} guardrail gate(s) in the lean env\n", flush=True)
        for gate in gates:
            tag = " (advisory)" if gate.advisory else ""
            print(f"::: {gate.name}{tag}", flush=True)
            result = subprocess.run(
                ["bash", "-c", gate.command],
                cwd=_REPO_ROOT,
                env=env,
            )
            if result.returncode != 0:
                if gate.advisory:
                    print(f"    WARN: advisory gate failed (ignored): {gate.name}", flush=True)
                else:
                    failures.append(gate.name)
                    print(f"    FAIL: {gate.name} (exit {result.returncode})", flush=True)
            else:
                print("    OK", flush=True)
    finally:
        if cleanup and venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

    print("\n" + "=" * 64)
    if failures:
        print(f"LEAN PARITY FAILED — {len(failures)} blocking gate(s) red in the lean env:")
        for name in failures:
            print(f"  - {name}")
        print("These would fail CI's Guardrails job. Make the heavy/agent-extra import")
        print("lazy + guarded (Dependency discipline) so the gate imports clean lean.")
        return 1
    print("LEAN PARITY OK — every blocking guardrail gate passes in the lean CI env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
