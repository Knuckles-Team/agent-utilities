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
  ``type(exc).__name__``/``exc.__class__.__name__`` (message dropped);
* ``except ... as exc:`` whose ONLY cause-preserving log call is at
  ``logger.debug(...)`` level, with no ``# noqa: BLE001 — <reason>``
  (``debug_only_swallow`` — see "Why DEBUG alone doesn't count" below,
  CONCEPT:AU-AHE.evaluation.debug-swallow-justification).

Deliberately NOT flagged: a handler that recovers with a **typed, narrow**
exception class doing ordinary control flow (``except ValueError: x = default``,
``except FileExistsError: ...``) — those aren't the "swallow a broad Exception
silently" antipattern this gate targets, and flagging every typed fallback
would swamp the signal in normal, idiomatic Python. A handler that re-raises
(anywhere in its body) is never a swallow at all.

Why DEBUG alone doesn't count (D-SWG-2, CONCEPT:AU-AHE.evaluation.
debug-swallow-justification): ``core/log_privacy.py``'s sanitizer makes a
handler's exception object the only calling convention that survives to the
log output at all — but *surviving to the log* and *surviving to a log
level anyone is actually watching in production* are different claims.
``agent_runner.py``'s ``:RunTrace`` write failure (D-DG-7) was
``except Exception as e: logger.debug(...)``: the exception object WAS
passed, cause-preservation as originally defined was satisfied, the gate
was green, and the failure was still invisible — DEBUG is off by default in
every production deployment this codebase ships. A handler whose *only*
cause-preserving log call is at DEBUG is therefore treated the same as an
undocumented swallow (`debug_only_swallow`) UNLESS it carries the existing
``# noqa: BLE001 — <reason>`` justification. This is deliberately not a
blanket ban on ``logger.debug``: a handler that also (or instead) logs at
``warning``/``error``/``exception``/``critical``/``info`` is unaffected, and
a handler with a real justification (e.g. "telemetry span close, expected to
fail when tracing is disabled, checked hourly by /health instead") stays
green by writing it down — the same bar every other swallow in this gate
already has to clear. The line is drawn on *loudness*, not on *level as a
lint target*: this keeps the common legitimate best-effort DEBUG swallow
quiet (one comment) while making the load-bearing one loud (a failing gate)
without re-flagging the ~39 ``log_type_name_only`` sites that have nothing
to do with level.

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

NOTE on the baseline key (D-SWG-1, CONCEPT:AU-AHE.evaluation.
swallow-baseline-stable-key): entries used to be keyed by (file, line
number). With ~20 concurrent lanes editing these files, ANY unrelated edit
earlier in a baselined file shifts every line below it and manufactures a
phantom "new" entry — this happened repeatedly in practice (see git history:
"re-key swallowed baseline for kg_server.py line shift", "...for
loop_controller line shift") and eventually rotted the gate permanently red,
which enforces nothing. The key is now (file, enclosing symbol, exception
type text, violation shape, ordinal) — computed by walking the AST with an
explicit scope stack (module/class/function qualname, e.g.
``ClientFactory.create``), reading the caught exception type(s) via
``ast.unparse``, and disambiguating multiple identical violations in the same
symbol with a stable traversal-order ordinal. None of those components move
when unrelated code shifts lines; they only change when the handler itself
is edited (its exception type, its shape, or its enclosing function is
renamed/moved) — genuinely new information, not noise. Reordering two
identical violations within the same function can still swap their ordinals;
that is accepted as out of scope (equivalent-shaped violations are
interchangeable for ratchet purposes) rather than solved with something
heavier (e.g. hashing handler body text, which would itself break on
harmless reformatting).

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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "agent_utilities"
BASELINE = ROOT / "scripts" / "swallowed_error_baseline.txt"
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared violation. Falls back to a filesystem walk only when ``target``
    is not inside a git working tree (e.g. a synthetic test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(target), "ls-files", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [target / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sorted(target.rglob("*.py"))

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


def _call_preserves_cause(
    call: ast.Call, bound_name: str, source_lines: list[str]
) -> bool:
    """True if this one log ``call`` references the bound exception itself
    (not just its type name)."""
    args_src = [_expr_src(source_lines, a) for a in call.args]
    kw_src = [_expr_src(source_lines, kw.value) for kw in call.keywords]
    joined = " ".join(args_src + kw_src)
    # The bound name appearing WITHOUT being wrapped in type(...)/.__class__
    # is cause-preserving (log_privacy.py's _sanitize_value renders a raw
    # BaseException as "Type: message", and str(exc)/repr(exc) obviously
    # carry the message too).
    bare_ref = re.search(rf"\b{re.escape(bound_name)}\b", joined)
    type_wrapped = re.search(
        rf"type\(\s*{re.escape(bound_name)}\s*\)|"
        rf"{re.escape(bound_name)}\.__class__",
        joined,
    )
    return bool(
        bare_ref
        and not (
            type_wrapped
            and not re.search(
                rf"str\(\s*{re.escape(bound_name)}\s*\)|repr\(\s*{re.escape(bound_name)}\s*\)",
                joined,
            )
        )
    )


def _cause_preservation_level(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> str:
    """ "loud" (a non-DEBUG log call already carries the real cause forward —
    fully justified, no violation), "debug_only" (the ONLY cause-preserving
    log call(s) are at DEBUG level — D-SWG-2: invisible in production unless
    explicitly justified via ``# noqa: BLE001 — <reason>``), or "none" (no
    cause-preserving log call at all — handled by ``_shape`` instead).
    """
    saw_debug_only = False
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        is_log, method = _is_log_call(node)
        if not is_log:
            continue
        if method == "exception":
            # logger.exception(...) always attaches the current exception's
            # traceback/message regardless of what args are passed, and is
            # never a quiet level (it logs at ERROR).
            return "loud"
        if bound_name is None:
            continue
        if _call_preserves_cause(node, bound_name, source_lines):
            if method == "debug":
                saw_debug_only = True
            else:
                return "loud"
    return "debug_only" if saw_debug_only else "none"


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
        type_name_expr = re.compile(
            rf"type\(\s*{re.escape(bound_name)}\s*\)\.__name__|"
            rf"{re.escape(bound_name)}\.__class__\.__name__"
        )
        bare_exc = re.compile(rf"(?<![\w.]){re.escape(bound_name)}(?![\w(])")
        type_only_calls = 0
        for call in log_calls:
            args_src = [_expr_src(source_lines, a) for a in call.args]
            kw_src = [_expr_src(source_lines, kw.value) for kw in call.keywords]
            joined = " ".join(args_src + kw_src)
            if not type_name_expr.search(joined):
                continue
            # A call that logs the type name AND the exception itself is
            # cause-preserving — "failed (%s: %s)", type(e).__name__, e carries
            # the real message. Only a call that reduces the exception to its
            # CLASS NAME drops the cause, so the bare binding must be absent
            # once the type-name expressions themselves are removed.
            if bare_exc.search(type_name_expr.sub("", joined)):
                continue
            type_only_calls += 1
        # Matches this branch's documented intent ("all of them only reference
        # type(bound_name).__name__"): if even one log call carries the real
        # cause, the handler is not cause-dropping.
        if type_only_calls and type_only_calls == len(log_calls):
            return "log_type_name_only"
    return None


def _iter_except_handlers_with_scope(
    node: ast.AST, scope_stack: tuple[str, ...] = ()
) -> "list[tuple[ast.ExceptHandler, str]]":
    """Yields (handler, enclosing qualname) for every ``ExceptHandler`` under
    ``node``, e.g. ``"ClientFactory.create"`` or ``"<module>"``.

    ``ast.walk()`` gives no parent information, so this is a manual
    pre-order DFS that tracks the nearest enclosing function/class chain —
    the stable component of D-SWG-1's re-key (CONCEPT:AU-AHE.evaluation.
    swallow-baseline-stable-key). Traversal is deterministic given the AST
    (source order of sibling nodes), so it does not depend on line numbers.
    """
    out: list[tuple[ast.ExceptHandler, str]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.extend(
                _iter_except_handlers_with_scope(child, scope_stack + (child.name,))
            )
            continue
        if isinstance(child, ast.ExceptHandler):
            out.append((child, ".".join(scope_stack) if scope_stack else "<module>"))
        out.extend(_iter_except_handlers_with_scope(child, scope_stack))
    return out


def _exception_type_text(handler: ast.ExceptHandler) -> str:
    """A normalized, formatting-insensitive rendering of the caught type(s),
    e.g. ``"(ValueError, TypeError)"`` or ``"<bare>"`` for a bare ``except:``.
    Stable across reformatting because ``ast.unparse`` reconstructs from the
    parsed AST, not from source text."""
    if handler.type is None:
        return "<bare>"
    return re.sub(r"\s+", " ", ast.unparse(handler.type)).strip()


# HandlerKey: (file, enclosing symbol, exception type text, shape, ordinal).
# Stable under pure line motion (unrelated edits elsewhere in the file);
# changes only when the handler's own type/shape/enclosing-symbol/relative
# order among identically-shaped siblings actually changes. See the module
# docstring's "NOTE on the baseline key" for the full rationale.
HandlerKey = tuple[str, str, str, str, int]

_KEY_SEP = "\t"


def _key_to_str(key: HandlerKey) -> str:
    rel, symbol, exc_types, shape, ordinal = key
    return _KEY_SEP.join([rel, symbol, exc_types, shape, str(ordinal)])


def _key_from_str(line: str) -> HandlerKey | None:
    fields = line.split(_KEY_SEP)
    if len(fields) < 5 or not fields[4].isdigit():
        return None
    rel, symbol, exc_types, shape = fields[0], fields[1], fields[2], fields[3]
    return (rel, symbol, exc_types, shape, int(fields[4]))


def _find_violations(
    rel: str, source: str, tree: ast.Module
) -> list[tuple[HandlerKey, int, str, str]]:
    """Returns (key, line, shape, except_line_text) for every
    unbaselined-candidate site in this file."""
    lines = source.splitlines()
    violations: list[tuple[HandlerKey, int, str, str]] = []
    ordinals: dict[tuple[str, str, str], int] = {}
    for node, symbol in _iter_except_handlers_with_scope(tree):
        except_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
        if _NOQA_BLE001_WITH_REASON_RE.search(except_line):
            continue  # justified convention — already-fine
        exc_types = _exception_type_text(node)
        shape: str | None
        if node.type is None:
            shape = "bare_except"
        elif _has_raise(node.body):
            continue  # re-raises — not a swallow at all
        else:
            level = _cause_preservation_level(node.body, node.name, lines)
            if level == "loud":
                continue
            if level == "debug_only":
                shape = "debug_only_swallow"
            else:
                shape = _shape(node.body, node.name, lines)
        if not shape:
            continue
        group = (symbol, exc_types, shape)
        ordinal = ordinals.get(group, 0)
        ordinals[group] = ordinal + 1
        key: HandlerKey = (rel, symbol, exc_types, shape, ordinal)
        violations.append((key, node.lineno, shape, except_line.strip()))
    return violations


def scan(
    target: Path = PKG, *, display_root: Path = ROOT
) -> dict[HandlerKey, tuple[int, str, str]]:
    """Returns {key: (line, shape, except_line_text)} for every current site.

    ``target``/``display_root`` default to the real package/repo root but are
    overridable (e.g. from a test's ``tmp_path`` fixture) so this gate's logic
    can be proven to actually trip on a broken fixture — see
    ``tests/gates/test_swallowed_errors_gate.py`` ("a gate that cannot fail is
    not a gate").
    """
    found: dict[HandlerKey, tuple[int, str, str]] = {}
    for py in _tracked_or_walked_py_files(target):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        if not py.is_file():
            continue
        rel = py.relative_to(display_root).as_posix()
        try:
            source = py.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for key, lineno, shape, text in _find_violations(rel, source, tree):
            found[key] = (lineno, shape, text)
    return found


def _load_baseline() -> set[HandlerKey]:
    if not BASELINE.exists():
        return set()
    out: set[HandlerKey] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = _key_from_str(line)
        if key is not None:
            out.add(key)
    return out


def _write_baseline(entries: dict[HandlerKey, tuple[int, str, str]]) -> None:
    def _sort_key(item: tuple[HandlerKey, tuple[int, str, str]]) -> tuple:
        key, (lineno, _shape, _text) = item
        rel, symbol, exc_types, shape, ordinal = key
        return (rel, lineno, symbol, exc_types, shape, ordinal)

    lines = []
    for key, (lineno, shape, text) in sorted(entries.items(), key=_sort_key):
        rel = key[0]
        lines.append(f"{_key_to_str(key)}\t# {rel}:{lineno} {text}")
    body = "\n".join(lines)
    BASELINE.write_text(
        "# Frozen baseline of cause-dropping exception handlers (ratchet — burn-down\n"
        "# toward zero). Keyed by TAB-separated (file, enclosing symbol, exception\n"
        "# type text, shape, ordinal) — stable under pure line motion; see D-SWG-1 /\n"
        "# CONCEPT:AU-AHE.evaluation.swallow-baseline-stable-key in the script's module\n"
        "# docstring. Everything after the 5 TAB-separated fields (a 6th tab then\n"
        "# '# ...') is a\n"
        "# human-readable comment (current file:line + source text) and is NOT parsed —\n"
        "# it will drift as the codebase changes; that's expected and harmless.\n"
        "# New entries fail scripts/check_swallowed_errors.py — either (a) add cause-\n"
        "# preserving logging at a level someone actually watches (pass the exception\n"
        "# itself, not type(exc).__name__, and not ONLY at logger.debug) while staying\n"
        "# best-effort, (b) re-raise where swallowing hides a genuine failure, or\n"
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
            for key, (lineno, shape, text) in sorted(
                current.items(), key=lambda kv: (kv[0][0], kv[1][0])
            ):
                print(f"  {key[0]}:{lineno} [{shape}] {text}")
            return 1
        print(f"OK — no swallowed-error sites under {explicit_root}")
        return 0

    current = scan()
    if args.update_baseline:
        _write_baseline(current)
        print(f"Baseline updated: {len(current)} entries → {BASELINE.name}")
        return 0

    baseline = _load_baseline()
    new = sorted(
        (k for k in current if k not in baseline),
        key=lambda k: (k[0], current[k][0]),
    )
    if new:
        print("New cause-dropping exception handler(s) found:\n")
        for key in new:
            lineno, shape, text = current[key]
            print(f"  {key[0]}:{lineno} [{shape}] {text}  (symbol={key[1]})")
        print(
            "\nEach must either (a) log the real cause at a level someone actually "
            "watches — pass the exception itself, not type(exc).__name__, and not "
            "ONLY at logger.debug (e.g. logger.warning('...: %s', exc)) — while "
            "staying best-effort, (b) re-raise where swallowing hides a genuine "
            "failure, or (c) document a deliberate best-effort swallow with "
            "`# noqa: BLE001 — <reason>`. See AGENTS.md and this task's final report."
        )
        return 1
    removed = len(baseline) - len(set(current) & baseline)
    msg = f"OK — no new swallowed-error sites ({len(current)} baselined"
    print(msg + (f", {removed} fixed since baseline)." if removed else ")."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
