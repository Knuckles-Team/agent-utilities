"""Static regression gate for exception leakage at served boundaries.

**Root-cause history (this lane, W12-AU-EXCEPTIONS, 2026-08-16):** the
original detector treated ANY logging call (``logger.error("...: %s", exc)``,
``logger.exception(...)``, ``exc_info=True``) that referenced a caught
exception as an unsafe "leak", full stop -- with no notion of *where the
exception text ends up*. That is wrong: this repository already has an
established, tested, process-wide mechanism for exactly this problem --
``agent_utilities/core/log_privacy.py``'s ``install_log_privacy_boundary()``
(installed at package import time) sanitizes ``record.msg``/``record.args``
for any logger whose ``.name`` is ``"agent_utilities"`` or starts with
``"agent_utilities."``, and nulls ``exc_info``/``exc_text``/``stack_info``
outright. Passing a caught exception to ``logger.error(...)`` on one of
those loggers NEVER reaches a client -- it reaches a sanitized, server-side
log stream, which is precisely the "log the detail server-side" mitigation
this gate exists to encourage. The sibling gate
``scripts/security/check_log_exception_redaction.py`` already documents and
enforces this exact distinction (loggers INSIDE vs. OUTSIDE the privacy
boundary) -- its module docstring states plainly that flagging a
boundary-covered logger "would contradict an established, tested,
gate-enforced convention".

Proof the old detector was wrong on real code: of the 18 real findings it
raised against ``main`` (13 in ``kg_server.py``, 2 in ``client_credentials.py``,
1 in ``analysis_tools.py``, 2 in ``daemon.py``), the ``analysis_tools.py:4096``
site (the ``graph_configure`` MCP tool's error handler) was ALREADY doing
exactly the right thing -- it returns only ``type(exc).__name__`` to the
caller and logs the real exception server-side via
``logger.warning("...: %s", exc)`` on an ``agent_utilities.*``-covered
logger. The old detector flagged the correct, recommended fix shape as a
violation. The only two GENUINE leaks among the 18 were
``client_credentials.py:328,356`` -- that module's logger is
``fastmcp.utilities.logging.get_logger(name="MultiplexerClientAuth")``,
which always produces a ``"fastmcp.<name>"``-named logger, OUTSIDE the
``agent_utilities.*`` privacy boundary -- a raw token-path and exception
message really would reach an unsanitized log stream there. Both call sites
were fixed (this same commit) to the established remediation shape already
proven at ``agent_utilities/mcp/child_resilience.py``:
``type(exc).__name__`` plus ``redact_for_log(exc)``
(``agent_utilities/security/log_redaction.py``).

This detector is now boundary-aware: a logging call
(``logger.<level>(...)``) whose receiver resolves, by simple local
``<name> = logging.getLogger(<arg>)`` / ``<name> = get_logger(<arg>)``
assignment tracking, to a KNOWN ``agent_utilities.*``-covered logger is
exempt -- its arguments (including any nested ``str(exc)``/``repr(exc)``/
f-string embed, ``exc_info=True``, or a bare ``.exception()`` call) are not
flagged, because ``install_log_privacy_boundary`` already sanitizes them
before they leave the process. Every OTHER unsafe use of the caught
exception is still flagged exactly as before -- a receiver this gate cannot
prove is covered (an unresolved receiver such as ``self.logger``, an
imported logger, or a call to ``get_logger()`` which always yields a
``fastmcp.*``-named, uncovered logger) stays unsafe by default, and any
render of the exception OUTSIDE a logging call at all (embedded directly in
a ``return``/``raise``/response payload) is unaffected by this change and
still flagged unconditionally -- that is the actual client-facing leak this
gate exists to catch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent_utilities

_LOG_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warn",
    "warning",
}

#: The one prefix ``install_log_privacy_boundary`` filters on
#: (``agent_utilities/core/log_privacy.py``).
_PRIVACY_BOUNDARY_PREFIX = "agent_utilities"


def _module_logger_coverage(tree: ast.Module) -> dict[str, bool]:
    """Map local logger variable name -> ``True`` iff it is provably INSIDE
    the ``agent_utilities.*`` privacy boundary.

    Recognizes two constructors, module-wide (any scope), by simple
    ``<name> = <constructor>(<arg>)`` assignment tracking -- no dataflow,
    matching every other ``check_*``/``test_*_static_gate.py`` heuristic in
    this repo:

    * ``logging.getLogger(...)`` / bare ``getLogger(...)`` -- covered iff the
      first positional arg is the literal ``__name__`` (every candidate file
      lives under the ``agent_utilities`` package, so its ``__name__`` always
      starts with ``"agent_utilities."``) or a string constant equal to
      ``"agent_utilities"``/starting with ``"agent_utilities."``. No args
      (the bare root logger) or an unresolvable arg -> NOT covered.
    * ``get_logger(...)`` -- ``fastmcp.utilities.logging.get_logger`` is the
      only caller of this bare name in this codebase, and it ALWAYS produces
      a ``"fastmcp.<name>"``-prefixed logger -- never ``agent_utilities.*``.
      Always NOT covered.

    An assignment shape this function does not recognize (``self.logger =
    ...``, a logger imported from elsewhere, ``logging.getLogger(f"...")``)
    leaves that name absent from the returned mapping -- treated as NOT
    provably covered by the caller, the same fail-closed default
    ``scripts/security/check_log_exception_redaction.py`` uses.
    """
    coverage: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        else:
            continue
        if func_name == "getLogger":
            if not call.args:
                coverage[target.id] = False  # bare root logger
                continue
            arg = call.args[0]
            if isinstance(arg, ast.Name) and arg.id == "__name__":
                coverage[target.id] = True
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                coverage[target.id] = arg.value == _PRIVACY_BOUNDARY_PREFIX or (
                    arg.value.startswith(_PRIVACY_BOUNDARY_PREFIX + ".")
                )
            else:
                coverage[target.id] = False  # unresolvable -- conservative
        elif func_name == "get_logger":
            # fastmcp.utilities.logging.get_logger: always "fastmcp.<name>".
            coverage[target.id] = False
    return coverage


def _covered_log_call_safe_ids(body: ast.AST, coverage: dict[str, bool]) -> set[int]:
    """Node ids (call + full argument subtree) of every logging call whose
    receiver resolves to a KNOWN ``agent_utilities.*``-covered logger -- the
    privacy boundary already sanitizes these before they leave the process,
    so nothing inside them is a leak."""
    safe_ids: set[int] = set()
    for node in ast.walk(body):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _LOG_METHODS:
            continue
        receiver = node.func.value
        if not isinstance(receiver, ast.Name):
            continue  # unresolved receiver (e.g. self.logger) -- not exempt
        if coverage.get(receiver.id) is not True:
            continue  # uncovered, or this gate cannot prove coverage
        safe_ids.add(id(node))
        for value in list(node.args) + [kw.value for kw in node.keywords]:
            for inner in ast.walk(value):
                safe_ids.add(id(inner))
    return safe_ids


def _unsafe_exception_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    coverage = _module_logger_coverage(tree)
    findings: list[str] = []
    for handler in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.name
    ):
        exception_name = handler.name
        body = ast.Module(body=handler.body, type_ignores=[])
        safe_ids = _covered_log_call_safe_ids(body, coverage)
        for node in ast.walk(body):
            if id(node) in safe_ids:
                continue
            unsafe = False
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    expression = (
                        value.value if isinstance(value, ast.FormattedValue) else None
                    )
                    if isinstance(expression, ast.Name):
                        unsafe |= expression.id == exception_name
                    elif (
                        isinstance(expression, ast.Call)
                        and isinstance(expression.func, ast.Name)
                        and expression.func.id in {"str", "repr"}
                    ):
                        unsafe |= any(
                            isinstance(arg, ast.Name) and arg.id == exception_name
                            for arg in expression.args
                        )
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"str", "repr"}:
                    unsafe |= any(
                        isinstance(arg, ast.Name) and arg.id == exception_name
                        for arg in node.args
                    )
                if isinstance(node.func, ast.Attribute):
                    unsafe |= node.func.attr in {
                        "format_exc",
                        "format_exception",
                        "print_exc",
                    }
                    unsafe |= node.func.attr == "exception"
                    unsafe |= node.func.attr in _LOG_METHODS and any(
                        isinstance(arg, ast.Name) and arg.id == exception_name
                        for arg in node.args
                    )
                    unsafe |= node.func.attr in _LOG_METHODS and any(
                        isinstance(keyword.value, ast.Name)
                        and keyword.value.id == exception_name
                        for keyword in node.keywords
                    )
                    unsafe |= any(
                        keyword.arg == "exc_info"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
            if unsafe:
                findings.append(f"{path.name}:{getattr(node, 'lineno', 0)}")
    return findings


def test_served_packages_do_not_expose_raw_exception_text() -> None:
    package_root = Path(agent_utilities.__file__).resolve().parent
    findings: list[str] = []
    for relative_root in ("mcp", "gateway", "server"):
        for path in (package_root / relative_root).rglob("*.py"):
            findings.extend(_unsafe_exception_uses(path))
    assert findings == []


# ---------------------------------------------------------------------------
# Known-bad / known-good proofs (D-W12-AU-EXCEPTIONS-1): prove the
# boundary-aware rewrite still catches a genuine leak, and no longer flags
# the safe, boundary-covered idiom it used to false-positive on.
# ---------------------------------------------------------------------------

_COVERED_LOGGER_FSTRING = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception as exc:
        logger.error("op failed: %s", exc)
"""

_COVERED_LOGGER_EXCEPTION_METHOD = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception:
        logger.exception("op failed")
"""

_COVERED_LOGGER_EXC_INFO_TRUE = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception as exc:
        logger.error("op failed", exc_info=True)
"""

_UNCOVERED_GET_LOGGER_LEAK = """
from fastmcp.utilities.logging import get_logger
logger = get_logger(name="Whatever")

def f():
    try:
        risky()
    except Exception as exc:
        logger.warning("op failed: %s", exc)
"""

_UNCOVERED_ROOT_LOGGER_LEAK = """
import logging
logger = logging.getLogger()

def f():
    try:
        risky()
    except Exception as exc:
        logger.error(f"op failed: {exc}")
"""

_NON_LOGGING_RETURN_LEAK = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception as exc:
        logger.error("op failed: %s", exc)
        return {"error": str(exc)}
"""

_NON_LOGGING_RAISE_LEAK = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception as exc:
        raise RuntimeError(f"op failed: {exc}") from exc
"""


def _findings_for_source(tmp_path: Path, name: str, source: str) -> list[str]:
    fixture_path = tmp_path / f"{name}.py"
    fixture_path.write_text(source, encoding="utf-8")
    return _unsafe_exception_uses(fixture_path)


def test_covered_logger_raw_exception_is_not_flagged(tmp_path: Path) -> None:
    """The false positive this lane fixed: a boundary-covered
    ``logging.getLogger(__name__)`` logger passing the caught exception via
    ``%s`` formatting is the recommended "log the detail server-side"
    mitigation, not a leak -- ``install_log_privacy_boundary`` sanitizes it."""
    assert _findings_for_source(tmp_path, "covered_fstring", _COVERED_LOGGER_FSTRING) == []


def test_covered_logger_exception_method_is_not_flagged(tmp_path: Path) -> None:
    assert (
        _findings_for_source(
            tmp_path, "covered_exception_method", _COVERED_LOGGER_EXCEPTION_METHOD
        )
        == []
    )


def test_covered_logger_exc_info_true_is_not_flagged(tmp_path: Path) -> None:
    assert (
        _findings_for_source(tmp_path, "covered_exc_info", _COVERED_LOGGER_EXC_INFO_TRUE)
        == []
    )


def test_get_logger_helper_leak_is_still_flagged(tmp_path: Path) -> None:
    """``fastmcp.utilities.logging.get_logger`` always names the logger
    ``fastmcp.<name>`` -- never inside the ``agent_utilities.*`` boundary.
    This is the exact real shape client_credentials.py had (fixed this same
    commit); the gate must still catch it if it recurs."""
    assert (
        _findings_for_source(tmp_path, "uncovered_get_logger", _UNCOVERED_GET_LOGGER_LEAK)
        != []
    )


def test_bare_root_logger_leak_is_still_flagged(tmp_path: Path) -> None:
    assert (
        _findings_for_source(
            tmp_path, "uncovered_root_logger", _UNCOVERED_ROOT_LOGGER_LEAK
        )
        != []
    )


def test_non_logging_return_leak_is_still_flagged(tmp_path: Path) -> None:
    """A covered logger elsewhere in the same handler must not blind the
    gate to a SEPARATE, genuine leak: the raw exception text embedded
    directly in a value returned to the caller."""
    assert _findings_for_source(tmp_path, "return_leak", _NON_LOGGING_RETURN_LEAK) != []


def test_non_logging_raise_leak_is_still_flagged(tmp_path: Path) -> None:
    assert _findings_for_source(tmp_path, "raise_leak", _NON_LOGGING_RAISE_LEAK) != []
