#!/usr/bin/env python3
"""Run a gate under the interpreter the repository actually declares.

Several guardrails (``check_cpd.py``, ``check_surface_parity.py``,
``gen_capability_power.py``) can only do their job by *importing the live MCP
server surface* — ``agent_utilities.mcp.kg_server`` and everything it pulls in.
That import is only satisfiable under this repository's declared dependency
floor (``pyproject.toml``: ``fastmcp>=4.0.0b1``), which lives in the
uv-managed ``<root>/.venv``.

``.pre-commit-config.yaml`` runs those hooks with ``language: system`` and a
bare ``python3``, so on a box whose ``PATH`` python is the host interpreter
(with, say, an old ``fastmcp`` in ``~/.local/lib``) the gates ran against a
dependency set the project does not support. The observable result was not a
clean failure but two *silently non-functional* gates:

* ``check_cpd.py`` aborted with an unhandled
  ``ModuleNotFoundError: No module named 'fastmcp.server.extensions'``
  traceback — a crashing gate enforces nothing at all; and
* ``check_surface_parity.py`` caught the same failure and folded it into its
  one "could not import kg_server surface" line, so its entire tool<->route
  drift half was checking nothing while still *looking* like it reported a
  finding.

``tests/conftest.py`` already declares the same contract for the test suite
(``sys.executable`` must live inside ``<root>/.venv``) and fails fast rather
than producing misleading results. This module extends that contract to the
guardrail scripts, but re-execs rather than merely refusing, so an ordinary
``pre-commit run`` keeps working without every caller having to know about
``scripts/uv_workspace.py``.

``AGENT_UTILITIES_ALLOW_ANY_INTERPRETER=1`` is the same explicit, deliberate
escape hatch ``tests/conftest.py`` honours — never the default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Set on the child so a re-exec can never recurse, however the venv resolves.
_REEXEC_MARKER = "AGENT_UTILITIES_GATE_REEXEC"


def _venv_python(root: Path) -> Path | None:
    for candidate in (
        root / ".venv" / "bin" / "python3",
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return candidate
    return None


def _inside(executable: str, venv: Path) -> bool:
    """Is *executable* the managed venv's own interpreter (or inside it)?

    Deliberately UNRESOLVED (``abspath``, not ``realpath``/``resolve``): a uv-
    managed venv's ``.venv/bin/python3`` is itself a SYMLINK to a shared base
    toolchain interpreter well outside any repo (e.g.
    ``~/.local/share/uv/python/cpython-.../bin/python3``), so resolving it
    walks OUT of ``.venv`` and this check would then find "not inside" on
    EVERY correctly-managed invocation -- unconditionally forcing the
    re-exec below on every run, mid-pytest-session when this module is
    imported by a test rather than run standalone (observed: the re-exec
    replaces the whole pytest process image via ``os.execve`` with pytest's
    OWN argv, killing the session outright). ``tests/conftest.py``'s
    equivalent guard (``_fail_fast_on_wrong_interpreter``) and
    ``tests/_test_engine.py``'s ``resolve_engine_binary`` both document and
    avoid this exact trap; this mirrors them instead of ``realpath``.
    """
    try:
        exe = os.path.abspath(executable)
        venv_abs = os.path.abspath(str(venv))
        return os.path.commonpath([exe, venv_abs]) == venv_abs
    except ValueError:  # different drives / unrelated roots
        return False


def require_project_interpreter(root: Path) -> None:
    """Re-exec the calling script under ``<root>/.venv`` when it is not already.

    No-op when the current interpreter already lives in the managed venv, when
    the deliberate escape hatch is set, or when this process *is* the re-exec'd
    child (so a mis-built venv surfaces as the gate's own error, not a loop).
    """

    if os.environ.get("AGENT_UTILITIES_ALLOW_ANY_INTERPRETER"):
        return
    if os.environ.get(_REEXEC_MARKER):
        return

    venv = root / ".venv"
    if venv.exists() and _inside(sys.executable, venv):
        return

    interpreter = _venv_python(root)
    if interpreter is None:
        # No managed venv to re-exec into. Do not pretend the gate is fine:
        # say plainly which interpreter is about to run it, so a downstream
        # import failure is attributable instead of mysterious.
        print(
            f"warning: {root / '.venv'} is absent; running this gate under "
            f"{sys.executable!r}, which is not the interpreter this repository "
            "declares. If it fails on an import, create the managed "
            "environment first: `python3 scripts/uv_workspace.py run "
            "--all-extras -- python -c pass`.",
            file=sys.stderr,
        )
        return

    env = dict(os.environ)
    env[_REEXEC_MARKER] = "1"
    os.execve(str(interpreter), [str(interpreter), *sys.argv], env)
