#!/usr/bin/env python3
"""httpx / httpx2 duality boundary gate (D-MTT-1).

This environment installs **two** structurally-identical but distinct HTTP
client packages side by side: ``httpx`` 0.28.1 (this repo's own base
dependency) and ``httpx2`` 2.7.0 (a real, separately published PyPI package —
pulled in transitively by ``fastmcp-slim[client]``/``mcp`` because this repo
deliberately adopted ``fastmcp>=4.0.0b1`` early — see
``agent_utilities/mcp/httpx_boundary.py``'s module docstring for the full
diagnosis). fastmcp's client transports and the underlying ``mcp`` SDK v2
client functions type their ``auth`` parameter as ``httpx2.Auth``; this
repo's own outbound child/service auth
(``agent_utilities.mcp.client_credentials.child_auth``) builds a LOCAL
``httpx.Auth``. Handing that straight to one of the four fastmcp/mcp SDK v2
client-constructor call sites below raises ``TypeError: Invalid "auth"
argument`` — verified against this environment's installed httpx2 2.7.0
source. Confirmed real, not hypothetical: mypy caught the ``multiplexer.py``
instance directly; ``toolset_factory.py``'s ``SSETransport`` path carried the
identical shape, reachable from three call sites
(``deployment/skill_validation_assets.py``, ``skills/runtime_validation.py``,
``orchestration/agent_runner.py``'s ``_spawn_auth()``). Both were fixed by
routing through ``coerce_httpx2_auth()``.

**What this gate checks.** A call to one of ``sse_client``,
``streamable_http_client``, ``SSETransport``, ``StreamableHttpTransport`` — the
four fastmcp/mcp SDK v2 constructors typed against ``httpx2.Auth`` — whose
``auth=`` keyword argument is (directly, or via a local variable this
function can trace back to a single assignment) a call to a KNOWN
local-``httpx.Auth``-producing function (``child_auth`` today; extend
``_RISKY_PRODUCERS`` as new ones are added) that is NOT wrapped in
``coerce_httpx2_auth(...)`` (or one of its known aliases). This is a
targeted heuristic scoped to the exact reproduced defect shape, not a
general type checker — matching every other ``scripts/security/check_*.py``
gate in this repo (see ``check_cypher_write_subset.py``'s module docstring
for the same tradeoff, spelled out there in more detail). A value this gate
cannot classify (an unrecognized call, an attribute access, a function
parameter) is NOT flagged — false negatives are preferred over false
positives for a heuristic this narrow; mypy remains the broader, if noisier,
backstop for shapes outside this gate's specific vocabulary.

**Fail-closed vs. an honest absence** (the same codified distinction
``check_cypher_write_subset.py`` and ``run_contract_checks`` in
``agent_utilities/governance/merge_queue.py`` use): this gate raises
``HttpxDualityGateError`` — and ``main()`` exits 1 — if it cannot read the
tree it was asked to check (the target directory is absent, or a candidate
file can't be read/decoded/parsed). It exits 0, reporting the count, when the
scan completes and finds zero violations: a clean scan is success, not
something to be suspicious of.

Usage:
  python3 scripts/security/check_httpx_duality.py
  python3 scripts/security/check_httpx_duality.py --repository-root DIR
  python3 scripts/security/check_httpx_duality.py --self-check

Exit 0 = no violation (or self-check passed too), 1 = a violation was found
or the gate could not read the tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The four fastmcp/mcp SDK v2 client constructors whose ``auth`` parameter
#: is typed ``httpx2.Auth | None`` in the installed fastmcp 4.0.0b1 / mcp
#: 2.0.0 source (verified directly against
#: ``site-packages/{fastmcp/client/transports,mcp/client}/{http,sse}.py``).
_TARGET_CALLS = {
    "sse_client",
    "streamable_http_client",
    "SSETransport",
    "StreamableHttpTransport",
}

#: Local functions/callables known to return a genuine ``httpx.Auth``
#: instance built from THIS repo's own ``httpx``. Extend this set the moment
#: a new one is added — that is exactly the "reintroduction" this gate
#: exists to catch.
_RISKY_PRODUCERS = {"child_auth"}

#: Names that neutralize a risky producer's output before it reaches a
#: target call — ``coerce_httpx2_auth`` (this repo's adapter,
#: ``agent_utilities/mcp/httpx_boundary.py``) is the one today.
_SAFE_WRAPPERS = {"coerce_httpx2_auth"}


class HttpxDualityGateError(RuntimeError):
    """The gate could not read the tree it was asked to check (fail-closed)."""


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    target: str
    producer: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line} auth= passed to {self.target}() from "
            f"unwrapped {self.producer}() — wrap it in coerce_httpx2_auth(...) "
            "(see agent_utilities/mcp/httpx_boundary.py)"
        )


def _callee_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _risky_producer_name(expr: ast.expr) -> str | None:
    """If ``expr`` is a direct call to a known risky producer, its name."""
    if not isinstance(expr, ast.Call):
        return None
    name = _callee_name(expr.func)
    return name if name in _RISKY_PRODUCERS else None


def _is_safely_wrapped(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    return _callee_name(expr.func) in _SAFE_WRAPPERS


def _classify_auth_arg(expr: ast.expr, local_producers: dict[str, str]) -> str | None:
    """Return the risky producer's name if ``expr`` resolves to its
    unwrapped output, else ``None`` (safe, or simply not classifiable —
    see the module docstring on why "not classifiable" is never flagged)."""
    if _is_safely_wrapped(expr):
        return None
    direct = _risky_producer_name(expr)
    if direct is not None:
        return direct
    if isinstance(expr, ast.Name):
        return local_producers.get(expr.id)
    return None


def _ordered_descendants(node: ast.AST) -> list[ast.AST]:
    """Depth-first, source-order traversal — same rationale as the
    equivalent helper in ``check_cypher_write_subset.py``: an assignment
    must be seen before a later call that references it, and a nested
    function/async-function gets its own independent scope (handled by
    ``_scope_units``), so this never descends into one."""
    ordered: list[ast.AST] = [node]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ordered.extend(_ordered_descendants(child))
    return ordered


def _scope_units(tree: ast.Module) -> list[list[ast.stmt]]:
    units: list[list[ast.stmt]] = [tree.body]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(node.body)
    return units


def _find_violations(rel: str, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for body in _scope_units(tree):
        local_producers: dict[str, str] = {}
        ordered: list[ast.AST] = []
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            ordered.extend(_ordered_descendants(stmt))
        for node in ordered:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    producer = _risky_producer_name(node.value)
                    if producer is not None:
                        local_producers[target.id] = producer
                    elif _is_safely_wrapped(node.value) or not isinstance(
                        node.value, ast.Call
                    ):
                        # Reassigning the name to something safe (or to a
                        # non-call) clears any earlier risky binding — a
                        # stale entry would be a false positive, not a
                        # false negative, so this err on precision.
                        local_producers.pop(target.id, None)
                continue
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node.func)
            if callee not in _TARGET_CALLS:
                continue
            for kw in node.keywords:
                if kw.arg != "auth":
                    continue
                producer = _classify_auth_arg(kw.value, local_producers)
                if producer is not None:
                    violations.append(Violation(rel, node.lineno, callee, producer))
    return violations


def _candidate_files(root: Path) -> list[Path]:
    """Files under ``root`` worth AST-parsing: only those referencing one of
    the four target constructors at all — same grep-first performance
    rationale as ``check_cypher_write_subset.py``'s ``_candidate_files``."""
    pattern = r"|".join(rf"\b{name}\(" for name in sorted(_TARGET_CALLS))
    try:
        proc = subprocess.run(
            ["grep", "-rlE", pattern, str(root), "--include=*.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HttpxDualityGateError(
            f"could not enumerate candidate files under {root}: {exc}"
        ) from exc
    if proc.returncode not in (0, 1):  # 1 == grep found nothing, not an error
        raise HttpxDualityGateError(
            f"grep over {root} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return sorted(Path(line) for line in proc.stdout.splitlines() if line)


def scan(root: Path) -> list[Violation]:
    """Scan ``root`` for httpx/httpx2 auth-boundary violations.

    Fail-closed: raises :class:`HttpxDualityGateError` if ``root`` doesn't
    exist, or a candidate file can't be read/decoded/parsed.
    """
    if not root.is_dir():
        raise HttpxDualityGateError(f"repository root is not a directory: {root}")

    violations: list[Violation] = []
    for path in _candidate_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HttpxDualityGateError(
                f"could not read candidate file {path}: {exc}"
            ) from exc
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise HttpxDualityGateError(
                f"could not parse candidate file {path}: {exc}"
            ) from exc
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        violations.extend(_find_violations(rel, tree))
    return sorted(violations, key=lambda v: (v.path, v.line))


def check(root: Path) -> dict[str, Any]:
    violations = scan(root)
    if violations:
        raise HttpxDualityGateError(
            "httpx/httpx2 duality violation(s) found:\n"
            + "\n".join(f"  {v.render()}" for v in violations)
        )
    return {"ok": True, "violations": 0}


# ---------------------------------------------------------------------------
# Self-check: prove the gate catches the known-bad shape (a reconstruction of
# the actual multiplexer.py:1475 / toolset_factory.py defect this lane fixed)
# and does not flag the fixed shape or unrelated auth usage.
# ---------------------------------------------------------------------------

_GOOD_FIXTURE = """
from agent_utilities.mcp.httpx_boundary import coerce_httpx2_auth
from agent_utilities.mcp.client_credentials import child_auth


def connect_direct(url, headers):
    transport = sse_client(
        url,
        headers=headers,
        auth=coerce_httpx2_auth(child_auth(headers)),
        httpx_client_factory=factory,
    )
    return transport


def connect_via_local(url, headers):
    _svc_auth = child_auth(headers)
    transport = sse_client(
        url,
        headers=headers,
        auth=coerce_httpx2_auth(_svc_auth),
        httpx_client_factory=factory,
    )
    return transport


def unrelated_auth_usage(url, headers, auth):
    # auth= to a target call, but not from a risky producer at all -- must
    # not be flagged (the gate is scoped to KNOWN local-httpx.Auth
    # producers, not every auth= keyword).
    return sse_client(url, headers=headers, auth=auth)


def own_httpx_client_is_fine(headers, svc_auth):
    # child_auth() feeding OUR OWN httpx.AsyncClient (not a target call) is
    # correct and must never be flagged.
    return create_async_http_client(headers=headers, auth=svc_auth)
"""

_BAD_FIXTURES: dict[str, str] = {
    "direct_call_unwrapped": """
from agent_utilities.mcp.client_credentials import child_auth


def connect(url, headers):
    transport = sse_client(
        url,
        headers=headers,
        auth=child_auth(headers),
        httpx_client_factory=factory,
    )
    return transport
""",
    "local_variable_unwrapped": """
from agent_utilities.mcp.client_credentials import child_auth


def connect(url, headers):
    _svc_auth = child_auth(headers)
    transport = sse_client(
        url,
        headers=headers,
        auth=_svc_auth,
        httpx_client_factory=factory,
    )
    return transport
""",
    "streamable_http_transport_unwrapped": """
from agent_utilities.mcp.client_credentials import child_auth


def build(url, headers, auth):
    return StreamableHttpTransport(
        url,
        headers=headers or None,
        auth=child_auth(headers),
        httpx_client_factory=factory,
    )
""",
}


def self_check() -> None:
    """Prove: (1) each known-bad shape is caught, (2) the fixed/coerced shape
    and unrelated auth usage are NOT flagged, (3) a missing root fails
    closed."""
    good_tree = ast.parse(_GOOD_FIXTURE)
    good_hits = _find_violations("fixture_good.py", good_tree)
    if good_hits:
        raise HttpxDualityGateError(
            "self-check: the coerced/unrelated shapes were flagged as a "
            f"violation: {[v.render() for v in good_hits]}"
        )

    for shape, fixture in _BAD_FIXTURES.items():
        tree = ast.parse(fixture)
        hits = _find_violations(f"fixture_{shape}.py", tree)
        if not hits:
            raise HttpxDualityGateError(
                f"self-check: known-bad shape {shape!r} was NOT caught"
            )

    missing = Path("/nonexistent-httpx-duality-gate-fixture-root")
    try:
        scan(missing)
    except HttpxDualityGateError:
        pass
    else:
        raise HttpxDualityGateError(
            "self-check: scanning a nonexistent root did not fail closed"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-httpx-duality")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            self_check()
        report = check(args.repository_root)
        if args.self_check:
            report["selfCheck"] = True
    except HttpxDualityGateError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    except Exception as exc:  # noqa: BLE001 — gate boundary: any unexpected failure must fail closed (exit 1), never be swallowed into a false "clean scan"
        print(
            json.dumps(
                {"ok": False, "error": f"unexpected {type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
