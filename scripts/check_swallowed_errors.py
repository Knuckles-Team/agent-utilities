#!/usr/bin/env python3
"""Swallowed-error guardrail gate (CONCEPT:AU-AHE.evaluation — diagnosability).

The single highest-leverage defect class found across a two-day debugging
session (2026-07-22/23): a handler that discards *why* an operation failed —
``except Exception: pass``, a bare ``except:``, an ``except ...: return`` with
no log, or a handler that logs only ``type(exc).__name__`` (the class name,
message dropped). Every major blocker in that session hid behind one of these
shapes; fixing the log line to include the real cause turned an
undiagnosable failure into a five-minute fix. A gate that catches this
pattern is worth more than any individual bug fix — this is that gate.

This gate is a **static heuristic**, not a dataflow prover (matching every
other ``scripts/check_*.py`` gate in this repo). For each ``except`` handler
that does not re-raise, it asks: does this handler ALREADY have a trail back
to the real cause? Two independent ways to have one:

1. **The justified ``# noqa: BLE001 — <reason>`` convention** — a comment on
   the ``except`` line documenting why swallowing here is deliberate and safe
   (already used at ~2000 sites in this codebase, e.g. "telemetry must never
   block startup", "duplicate registration tolerated"). Any non-empty reason
   after the ``BLE001`` marker (``-``/``—``/``:`` separator) counts.
2. **Cause-preserving logging** — the handler logs the bound exception itself
   (or ``str(exc)``/``repr(exc)``), or calls ``logger.exception(...)`` (which
   always attaches the current exception's traceback), rather than only
   ``type(exc).__name__``/``exc.__class__.__name__`` or nothing at all.
   NOTE: ``exc_info=True`` is NOT recognized as cause-preserving in this
   codebase specifically — ``core/log_privacy.py``'s process-wide LogRecord
   factory unconditionally nulls ``record.exc_info``/``exc_text``/
   ``stack_info`` for every ``agent_utilities.*`` logger (tracebacks embed
   host filesystem paths), so passing the exception object as a ``%s``/f-string
   argument is the only calling convention that actually survives to the log
   output — ``core/log_privacy.py``'s ``_sanitize_value`` renders it as
   ``f"{type(exc).__name__}: {sanitize_log_text(str(exc))}"``, preserving the
   message while still redacting endpoints/paths/emails.

A handler with NEITHER of those is a **violation**: a genuinely undocumented,
cause-dropping swallow. Flagged shapes (matching the exact patterns named in
the task this gate was built for):

* a bare ``except:`` (catches even ``SystemExit``/``KeyboardInterrupt``);
* ``except <Type>: pass`` (or ``pass`` as the sole statement);
* ``except <Type>: return`` with no value and no log call anywhere in the body;
* ``except ... as exc:`` whose only reference to ``exc`` in a log call is
  ``type(exc).__name__``/``exc.__class__.__name__`` (message dropped).

Deliberately NOT flagged: a handler that recovers with a **typed, narrow**
exception class doing ordinary control flow (``except ValueError: x = default``,
``except FileExistsError: ...``) — those aren't the "swallow a broad Exception
silently" antipattern this gate targets, and flagging every typed fallback
would swamp the signal in normal, idiomatic Python. A handler that re-raises
(anywhere in its body) is never a swallow at all.

Because the codebase already carries a large number of un-annotated
`except Exception`-shaped handlers this gate's own author has not
individually triaged (the DEBT-2 sweep fixed every genuinely high-value site
it found — the public error-surface boundary, the MCP/graph-os boot chain,
the multiplexer child-lifecycle log lines, engine mutation paths — see the
sweep's final report for the full verdict table), this is a **ratchet**: the
current set is frozen in ``scripts/swallowed_error_baseline.txt`` and the
gate fails only on *newly introduced* cause-dropping handlers. Fixing a
baselined site (adding a `# noqa: BLE001 — <reason>` or cause-preserving log,
or re-raising) always shrinks the baseline on the next ``--update-baseline``.

NOTE on the baseline key: entries are keyed by (file, line number). Unlike the
env-sprawl gate's (file, ENV_KEY) key — which survives any unrelated edit
because env var names are content-stable — an exception handler has no
comparable natural name, so line number is the least-bad stable-ish key
available. An unrelated edit earlier in the same file WILL shift line numbers
and require a baseline refresh (`--update-baseline`) even though nothing
about the flagged handlers themselves changed; this is a known, accepted
tradeoff (the same one most line-based lint-baseline tools make), not a
design flaw to work around here.

Usage:
  python3 scripts/check_swallowed_errors.py                  # check (exit 1 on new)
  python3 scripts/check_swallowed_errors.py --update-baseline # freeze current set
  python3 scripts/check_swallowed_errors.py ROOT              # scan ROOT, no baseline
                                                               # (e.g. a test fixture dir)

Exit 0 = no new cause-dropping handler, 1 = at least one found.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "agent_utilities"
BASELINE = ROOT / "scripts" / "swallowed_error_baseline.txt"
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}

# The justified-convention marker: a "noqa: BLE001"-style comment followed by
# a non-empty reason after a -/—/: separator. A bare marker with NO reason is
# deliberately NOT accepted — the convention this codebase established is
# "document why", not just "silence the lint".
_NOQA_BLE001_WITH_REASON_RE = re.compile(r"#\s*noqa:\s*BLE001\s*[-—:]\s*\S")

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "warn"}


def _is_log_call(call: ast.Call) -> tuple[bool, str | None]:
    """(is_a_log_call, method_name) for ``<something with 'log' in its name>.<level>(...)``."""
    fn = call.func
    if not isinstance(fn, ast.Attribute) or fn.attr not in _LOG_METHODS:
        return False, None
    base = fn.value
    base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
    if base_name and "log" in base_name.lower():
        return True, fn.attr
    return False, None


def _expr_src(source_lines: list[str], node: ast.expr) -> str:
    try:
        return ast.get_source_segment("\n".join(source_lines), node) or ""
    except Exception:  # noqa: BLE001 - best-effort snippet only, never fatal
        return ""


def _has_raise(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(n, ast.Raise)
        for n in ast.walk(ast.Module(body=body, type_ignores=[]))
    )


def _is_cause_preserving(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> bool:
    """True if the handler's logging already carries the real cause forward."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        is_log, method = _is_log_call(node)
        if not is_log:
            continue
        if method == "exception":
            # logger.exception(...) always attaches the current exception's
            # traceback/message regardless of what args are passed.
            return True
        if bound_name is None:
            continue
        args_src = [_expr_src(source_lines, a) for a in node.args]
        kw_src = [_expr_src(source_lines, kw.value) for kw in node.keywords]
        joined = " ".join(args_src + kw_src)
        # The bound name appearing WITHOUT being wrapped in type(...)/
        # .__class__ is cause-preserving (log_privacy.py's _sanitize_value
        # renders a raw BaseException as "Type: message", and str(exc)/
        # repr(exc) obviously carry the message too).
        bare_ref = re.search(rf"\b{re.escape(bound_name)}\b", joined)
        type_wrapped = re.search(
            rf"type\(\s*{re.escape(bound_name)}\s*\)|"
            rf"{re.escape(bound_name)}\.__class__",
            joined,
        )
        if bare_ref and not (
            type_wrapped
            and not re.search(
                rf"str\(\s*{re.escape(bound_name)}\s*\)|repr\(\s*{re.escape(bound_name)}\s*\)",
                joined,
            )
        ):
            return True
    return False


def _shape(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> str | None:
    """Returns a violation shape name, or None if this handler isn't a target shape."""
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return "pass"
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is None:
        has_log = any(
            _is_log_call(n)[0]
            for n in ast.walk(ast.Module(body=body, type_ignores=[]))
            if isinstance(n, ast.Call)
        )
        return None if has_log else "return_none"
    # log-type-name-only: at least one log call exists, all of them only
    # reference type(bound_name).__name__ / bound_name.__class__.__name__.
    log_calls = [
        n
        for n in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(n, ast.Call) and _is_log_call(n)[0]
    ]
    if log_calls and bound_name:
        any_type_only = False
        for call in log_calls:
            args_src = [_expr_src(source_lines, a) for a in call.args]
            kw_src = [_expr_src(source_lines, kw.value) for kw in call.keywords]
            joined = " ".join(args_src + kw_src)
            if re.search(
                rf"type\(\s*{re.escape(bound_name)}\s*\)\.__name__|"
                rf"{re.escape(bound_name)}\.__class__\.__name__",
                joined,
            ):
                any_type_only = True
        if any_type_only:
            return "log_type_name_only"
    return None


def _find_violations(
    rel: str, source: str, tree: ast.Module
) -> list[tuple[int, str, str]]:
    """Returns (line, shape, except_line_text) for every unbaselined-candidate site."""
    lines = source.splitlines()
    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        except_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
        if _NOQA_BLE001_WITH_REASON_RE.search(except_line):
            continue  # justified convention — already-fine
        if node.type is None:
            violations.append((node.lineno, "bare_except", except_line.strip()))
            continue
        if _has_raise(node.body):
            continue  # re-raises — not a swallow at all
        if _is_cause_preserving(node.body, node.name, lines):
            continue
        shape = _shape(node.body, node.name, lines)
        if shape:
            violations.append((node.lineno, shape, except_line.strip()))
    return violations


def scan(
    target: Path = PKG, *, display_root: Path = ROOT
) -> dict[tuple[str, int], tuple[str, str]]:
    """Returns {(relpath, line): (shape, except_line_text)} for every current site.

    ``target``/``display_root`` default to the real package/repo root but are
    overridable (e.g. from a test's ``tmp_path`` fixture) so this gate's logic
    can be proven to actually trip on a broken fixture — see
    ``tests/gates/test_swallowed_errors_gate.py`` ("a gate that cannot fail is
    not a gate").
    """
    found: dict[tuple[str, int], tuple[str, str]] = {}
    for py in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        rel = py.relative_to(display_root).as_posix()
        try:
            source = py.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for lineno, shape, text in _find_violations(rel, source, tree):
            found[(rel, lineno)] = (shape, text)
    return found


def _load_baseline() -> set[tuple[str, int]]:
    if not BASELINE.exists():
        return set()
    out: set[tuple[str, int]] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, lineno = line.rpartition(":")
        if rel and lineno.isdigit():
            out.add((rel, int(lineno)))
    return out


def _write_baseline(entries: dict[tuple[str, int], tuple[str, str]]) -> None:
    body = "\n".join(f"{rel}:{lineno}" for rel, lineno in sorted(entries))
    BASELINE.write_text(
        "# Frozen baseline of cause-dropping exception handlers (ratchet — burn-down\n"
        "# toward zero). Keyed by file:line — an unrelated edit earlier in a baselined\n"
        "# file WILL shift line numbers and require --update-baseline even though the\n"
        "# flagged handler itself is unchanged (see the script's module docstring).\n"
        "# New entries fail scripts/check_swallowed_errors.py — either (a) add cause-\n"
        "# preserving logging (pass the exception itself, not type(exc).__name__) while\n"
        "# staying best-effort, (b) re-raise where swallowing hides a genuine failure, or\n"
        "# (c) document a deliberate best-effort swallow with `# noqa: BLE001 — <reason>`.\n"
        + body
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        help=(
            "Scan this path instead of agent_utilities/, with NO baseline "
            "comparison (every finding is reported) — for scanning an "
            "arbitrary fixture/test directory, mirroring "
            "check_identifier_interpolation.py's explicit-ROOT mode."
        ),
    )
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    if args.root:
        explicit_root = Path(args.root)
        current = scan(explicit_root, display_root=explicit_root)
        if current:
            print("Cause-dropping exception handler(s) found:\n")
            for (rel, lineno), (shape, text) in sorted(current.items()):
                print(f"  {rel}:{lineno} [{shape}] {text}")
            return 1
        print(f"OK — no swallowed-error sites under {explicit_root}")
        return 0

    current = scan()
    if args.update_baseline:
        _write_baseline(current)
        print(f"Baseline updated: {len(current)} entries → {BASELINE.name}")
        return 0

    baseline = _load_baseline()
    new = sorted(k for k in current if k not in baseline)
    if new:
        print("New cause-dropping exception handler(s) found:\n")
        for rel, lineno in new:
            shape, text = current[(rel, lineno)]
            print(f"  {rel}:{lineno} [{shape}] {text}")
        print(
            "\nEach must either (a) log the real cause (pass the exception itself, not "
            "type(exc).__name__, e.g. logger.warning('...: %s', exc)) while staying "
            "best-effort, (b) re-raise where swallowing hides a genuine failure, or "
            "(c) document a deliberate best-effort swallow with "
            "`# noqa: BLE001 — <reason>`. See AGENTS.md and this task's final report."
        )
        return 1
    removed = len(baseline) - len(set(current) & baseline)
    msg = f"OK — no new swallowed-error sites ({len(current)} baselined"
    print(msg + (f", {removed} fixed since baseline)." if removed else ")."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
