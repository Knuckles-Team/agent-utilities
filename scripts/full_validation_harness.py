#!/usr/bin/env python3
"""LANE H-driver — the assembled, committed end-to-end validation harness.

Runs seven stages, in dependency order, and STOPS at the first failure — the
same "later failures are meaningless once an earlier one is broken" contract
as ``scripts/delegation_probe.py``:

    1 mcp_handshake     graph-os's own MCP streamable-http endpoint answers
                         initialize -> notifications/initialized -> tools/list
                         with a non-empty tool surface.
    2 browser           the unauthenticated, user-visible agent-webui surface
                         (/, /auth/login, /graph, /skills) is up.
    3 api_gateway       the service-token REST surface answers with a real
                         bearer, acquired the way graph-os itself acquires one.
    4 admission         the tenant-admission RPC a signed-in principal
                         triggers on every request — a SURROGATE, see below.
    5 delegation        scripts/delegation_probe.py's own 9-stage in-process
                         delegation probe, run as a subprocess and propagated.
    6 mcp_tool_catalog  the durable MCP tool/CallableResource catalog is
                         actually reachable in the live KG.
    7 kg_general        broad KG health: node/label counts, RunTrace/ToolCall/
                         Concept/MCPServer/OutcomeEvaluation presence, and the
                         RunTrace-USED_TOOL-ToolCall edge.

WHAT EACH STAGE PROVES — AND WHAT IT DOES NOT
------------------------------------------------
1 mcp_handshake    Proves: the MCP transport is alive and serves >=1 tool.
                   Does NOT prove: any individual tool executes correctly, or
                   that non-streamable-http transports (stdio) work.
2 browser          Proves: agent-webui itself is serving traffic to an
                   unauthenticated client (200/302/303/401 all count as "up";
                   5xx or a connection failure does not).
                   Does NOT prove: a signed-in user can do anything past the
                   login screen — see stage 4's honesty requirement, which
                   applies transitively to any downstream "chat works" claim.
3 api_gateway      Proves: the API gateway honors a legitimate SERVICE bearer,
                   acquired the same way graph-os acquires its own.
                   Does NOT prove: a user-scoped (browser session) token
                   behaves identically — that path is only reachable through
                   the real OIDC authorization-code + PKCE flow (stage 4).
4 admission        *** SURROGATE ONLY — NOT PROOF OF HUMAN SIGN-IN. ***
                   Mints THIS POD'S OWN graph process identity and calls the
                   exact ``run_tenant_admission()`` RPC agent-webui's
                   ``ensure_tenant_admission()`` calls on every authenticated
                   request. agent-webui's real login is OIDC
                   authorization-code + PKCE via Keycloak; no service
                   credential can perform that browser flow, so a PASS here
                   proves only that the admission RPC itself grants
                   successfully for a verified identity — never read it as
                   "a human can sign in". (This also means any future "does
                   chat work" check built on top of this harness inherits the
                   same gap: chat additionally depends on the browser
                   session and the webui's streaming/render path, neither of
                   which leaves a server-side signal this harness — or any
                   service-credential probe — can read back. No such check is
                   added here, deliberately, so as not to manufacture a green
                   for something never actually proven.)
5 delegation       Proves: the delegation core (config, identity, engine,
                   grounding, model, skill resolution, toolset binding,
                   execute_agent/execute_capability, and provenance
                   read-back) works end-to-end for the given --skill/--server/
                   --tool, exactly as delegation_probe.py measures it.
                   Does NOT prove: the NL/grounding planning path works in
                   general. The measured baseline (2026-08-24) is that
                   default grounding='required' delegation FAILS closed with
                   "could not establish compiled evidence (timeout)" — this
                   driver does not attempt to paper over that; it propagates
                   delegation_probe.py's real exit code untouched.
6 mcp_tool_catalog Proves: :Tool and :CallableResource are both non-empty and
                   the :CallableResource set is reachable via
                   (:Server)-[:PROVIDES]->, i.e. delegation has something to
                   bind to.
                   Does NOT prove: every individual tool is invocable, or
                   that the catalog is fully up to date with the live fleet.
7 kg_general       Proves: the KG backend answers ordinary Cypher/SQL under
                   this dialect and the graph is populated broadly (not just
                   one label). Baseline observed 2026-08-24: ~56,853 total
                   nodes, 2,941 :Tool, 361 :CallableResource, 72 :RunTrace —
                   this driver does not hard-assert those exact numbers (they
                   drift with fleet activity); it asserts non-zero presence,
                   matching what ``scripts/_harness_kg_queries.py`` itself
                   checks.
                   Does NOT prove: data freshness or per-record correctness
                   beyond gross, non-zero counts.

EXIT-CODE CONTRACT
-------------------
Exit code = the 1-based index (1..7, per the numbered list above) of the
FIRST stage that failed. Exit code 0 means every stage that ran (respecting
``--stop-after``) PASSED. This is exactly ``scripts/delegation_probe.py``'s
own contract, so the two compose without a new mental model — including
stage 5 here, which does not attempt to renumber delegation_probe.py's own
internal 1..9 stage failures into this driver's 1..7 numbering: ANY non-zero
delegation_probe.py exit is reported, verbatim, as this driver's stage 5
FAIL, and delegation_probe.py's own stage output (which already states which
of ITS 9 stages failed) is streamed through live so no detail is lost.

STDIN / IN-POD IMPORT NOTE — READ THIS BEFORE ASSUMING A ModuleNotFoundError
-------------------------------------------------------------------------------
This file is designed to be piped into ``python3 -`` inside the graph-os pod,
exactly like ``scripts/delegation_probe.py``::

    kubectl -n platform exec -i <pod> -c graph-os -- python3 - \\
        --skill ... --server ... --tool ... \\
        < scripts/full_validation_harness.py

When Python reads a script from stdin this way, ``__file__`` is the literal
string ``'<stdin>'`` (verified empirically: ``python3 - <<<'print(__file__)'``
prints ``<stdin>``) and therefore CANNOT be used to locate sibling files on
disk. ``delegation_probe.py`` never hits this problem because it imports only
the INSTALLED ``agent_utilities`` package, which is resolvable from any cwd
via site-packages/PYTHONPATH regardless of how the driver script itself was
loaded. This driver is different: it also needs three DEV-ONLY sibling
modules that are not part of the installed wheel —
``scripts._harness_mcp``, ``scripts._harness_browser_api``,
``scripts._harness_kg_queries`` — plus a subprocess path to
``scripts/delegation_probe.py`` itself.

``_import_harness_modules()`` below resolves this with a documented fallback
chain, in order:

  1. Plain ``import scripts._harness_mcp`` (etc). This succeeds UNMODIFIED
     whenever the repo root is already importable as a package root — true
     in the documented production pod topology, where the fleet NFS-mounts
     the canonical checkout at ``/au`` with ``PYTHONPATH=/au`` (see
     ``AGENTS.md`` "Merging is not deploying"), and ALSO true whenever the
     process's cwd IS the repo root, because ``python3 -`` puts ``''``
     (cwd) at ``sys.path[0]``.
  2. If that raises ``ModuleNotFoundError``, try an explicit
     ``--modules-dir DIR`` — the repo-root directory that contains
     ``scripts/`` — inserted at ``sys.path[0]``, then retry the plain
     import. This is the fallback an operator reaches for when running from
     a worktree that is NOT ``/au`` (e.g. an integration worktree) and
     PYTHONPATH was not pre-seeded.
  3. If ``--modules-dir`` was not given, additionally try the production
     default, ``/au``, before giving up.
  4. If still unimportable, raise a clear, actionable ``RuntimeError`` —
     never a bare ``ModuleNotFoundError`` — naming exactly what to pass.

The same resolved directory is reused to locate ``scripts/delegation_probe.py``
for the subprocess in stage 5 (``--delegation-probe-path`` overrides this
directly if the two ever need to diverge).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

STAGES = [
    "mcp_handshake",
    "browser",
    "api_gateway",
    "admission",
    "delegation",
    "mcp_tool_catalog",
    "kg_general",
]

#: Same production mount documented in AGENTS.md ("Merging is not deploying"):
#: the fleet NFS-mounts the canonical checkout at /au with PYTHONPATH=/au.
_DEFAULT_MODULES_DIR = "/au"

# Populated by the engine-acquisition stage and reused by stages 6-7, exactly
# like delegation_probe.py's module-level _STATE dict.
_STATE: dict[str, Any] = {}


def _emit(stage: str, ok: bool, detail: str = "", elapsed: float | None = None) -> None:
    """Same reporting convention as scripts/delegation_probe.py, verbatim,
    so operators reading either tool's output see one shape."""
    mark = "PASS" if ok else "FAIL"
    t = f" [{elapsed:6.2f}s]" if elapsed is not None else ""
    print(f"  {mark:4s} {stage:17s}{t} {detail}"[:600], flush=True)


def _chain(exc: BaseException) -> str:
    """Full causal chain — the thing an opaque one-line error throws away.
    Identical to scripts/delegation_probe.py's ``_chain``."""
    out: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n      caused by ".join(out)


# ---------------------------------------------------------------------------
# stdin-safe sibling-module resolution (see module docstring)
# ---------------------------------------------------------------------------
def _has_harness_modules(root: Path) -> bool:
    scripts_dir = root / "scripts"
    return (
        (scripts_dir / "_harness_mcp.py").is_file()
        and (scripts_dir / "_harness_browser_api.py").is_file()
        and (scripts_dir / "_harness_kg_queries.py").is_file()
    )


def _import_harness_modules(
    modules_dir: str | None,
) -> tuple[ModuleType, ModuleType, ModuleType, Path]:
    """Return (harness_mcp, harness_browser_api, harness_kg_queries, repo_root).

    See the module docstring's "STDIN / IN-POD IMPORT NOTE" for the full
    resolution chain and why it exists.
    """

    def _try_plain() -> tuple[ModuleType, ModuleType, ModuleType] | None:
        try:
            mcp_mod = importlib.import_module("scripts._harness_mcp")
            browser_mod = importlib.import_module("scripts._harness_browser_api")
            kg_mod = importlib.import_module("scripts._harness_kg_queries")
        except ModuleNotFoundError:
            return None
        return mcp_mod, browser_mod, kg_mod

    # 1. plain import — works whenever the repo root is already on sys.path
    #    (production /au mount with PYTHONPATH=/au, or cwd == repo root
    #    because `python3 -` seeds sys.path[0] with '').
    result = _try_plain()
    if result is not None:
        mcp_mod, browser_mod, kg_mod = result
        # Best-effort repo_root for the delegation_probe.py subprocess path:
        # derive it from wherever scripts._harness_mcp actually loaded from.
        mcp_file = getattr(mcp_mod, "__file__", None)
        root = (
            Path(mcp_file).resolve().parents[1]
            if mcp_file
            else Path(_DEFAULT_MODULES_DIR)
        )
        return mcp_mod, browser_mod, kg_mod, root

    # 2/3. explicit --modules-dir, then the production default /au.
    candidates = [c for c in (modules_dir, _DEFAULT_MODULES_DIR) if c]
    last_exc: Exception | None = None
    for candidate in candidates:
        root = Path(candidate)
        if not _has_harness_modules(root):
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        # Drop any partial import left behind by a prior failed candidate so
        # this retry is not silently served a stale module.
        for name in (
            "scripts",
            "scripts._harness_mcp",
            "scripts._harness_browser_api",
            "scripts._harness_kg_queries",
        ):
            sys.modules.pop(name, None)
        result = _try_plain()
        if result is not None:
            mcp_mod, browser_mod, kg_mod = result
            return mcp_mod, browser_mod, kg_mod, root
        last_exc = ModuleNotFoundError(
            f"import still failed with {candidate!r} on sys.path"
        )

    raise RuntimeError(
        "could not import scripts._harness_mcp / scripts._harness_browser_api / "
        "scripts._harness_kg_queries. This almost always means the driver was "
        "piped via stdin (`python3 - < full_validation_harness.py`), where "
        "__file__ == '<stdin>' and cannot locate sibling files, AND the repo "
        "root was not already on sys.path/PYTHONPATH. Fix: pass "
        "--modules-dir <repo-root-containing-scripts/> explicitly (in the "
        f"documented production pod topology this is {_DEFAULT_MODULES_DIR!r}, "
        "the fleet's NFS-mounted canonical checkout — see AGENTS.md 'Merging is "
        "not deploying'), or invoke this file as a real path instead of over "
        "stdin, or export PYTHONPATH to include the repo root before running."
    ) from last_exc


def _delegation_probe_path(repo_root: Path, override: str) -> Path:
    if override:
        return Path(override)
    return repo_root / "scripts" / "delegation_probe.py"


# ---------------------------------------------------------------------------
# Stage 1 — mcp_handshake (sync stage function, run via asyncio.to_thread)
# ---------------------------------------------------------------------------
def _stage_mcp_handshake(a: argparse.Namespace) -> str:
    harness_mcp = _STATE["harness_mcp"]
    result = harness_mcp.stage_mcp_handshake(a.mcp_url, timeout=a.timeout)
    return (
        f"tools={result.tool_count} protocol={result.protocol_version} "
        f"server={result.server_name}/{result.server_version} "
        f"session={result.session_id!r} sample={list(result.tool_names_sample)}"
    )


# ---------------------------------------------------------------------------
# Stage 2 — browser
# ---------------------------------------------------------------------------
def _stage_browser(a: argparse.Namespace) -> str:
    harness_browser = _STATE["harness_browser"]
    report = harness_browser.stage_browser(a.base_url, timeout=a.timeout)
    return report.detail


# ---------------------------------------------------------------------------
# Stage 3 — api_gateway
# ---------------------------------------------------------------------------
def _stage_api_gateway(a: argparse.Namespace) -> str:
    harness_browser = _STATE["harness_browser"]
    report = harness_browser.stage_api_gateway(a.base_url, timeout=max(a.timeout, 30.0))
    return report.detail


# ---------------------------------------------------------------------------
# Stage 4 — admission (SURROGATE — see module docstring)
# ---------------------------------------------------------------------------
def _stage_admission(a: argparse.Namespace) -> str:
    harness_browser = _STATE["harness_browser"]
    report = harness_browser.stage_admission(tenant_slug=a.tenant_slug)
    return (
        report.detail
        + " [SURROGATE — NOT proof of human sign-in; see module docstring]"
    )


# ---------------------------------------------------------------------------
# Stage 5 — delegation (shell out to scripts/delegation_probe.py)
# ---------------------------------------------------------------------------
async def _stage_delegation(a: argparse.Namespace) -> str:
    probe_path = _delegation_probe_path(_STATE["repo_root"], a.delegation_probe_path)
    if not probe_path.is_file():
        raise RuntimeError(
            f"delegation_probe.py not found at {probe_path} — pass "
            "--delegation-probe-path explicitly if it lives elsewhere"
        )
    cmd = [
        sys.executable,
        str(probe_path),
        "--skill",
        a.skill,
        "--server",
        a.server,
        "--tool",
        a.tool,
    ]
    if a.delegation_probe_args:
        cmd.extend(a.delegation_probe_args)
    print(f"  ...  delegation      $ {' '.join(cmd)}", flush=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace")
        print("    [delegation_probe] " + line, end="", flush=True)
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    rc = await proc.wait()
    _STATE["delegation_probe_exit_code"] = rc
    if rc != 0:
        raise RuntimeError(
            f"scripts/delegation_probe.py exited {rc} (its own stage {rc} of 9 failed; "
            "see the streamed [delegation_probe] output above for the exact stage and "
            "root-cause chain). tail:\n" + "".join(tail[-15:])
        )
    return (
        "delegation_probe.py exited 0 (all 9 of its stages passed). tail:\n"
        + "".join(tail[-8:])
    )


# ---------------------------------------------------------------------------
# Engine acquisition for stages 6-7 — mirrors delegation_probe.py stages 2-3
# EXACTLY (same identity mint, same "prefer the already-running engine"
# reasoning), acquired ONCE and shared across both stages.
# ---------------------------------------------------------------------------
async def _acquire_engine(a: argparse.Namespace) -> Any:
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.security.brain_context import set_actor

    # Same reasoning as delegation_probe.py's _stage_identity: the engine is
    # fail-closed on IdentityRequiredError/SessionRequiredError, and both are
    # features, not bugs to route around. Minting via _mint_process_session()
    # is the literal function the served graph-os process calls at boot
    # (kg_server.py:4218), so a failure here means identity/session binding
    # is broken in production too — it cannot be masked.
    session = await asyncio.to_thread(_mint_process_session, a.transport)
    session.engine_verified_context()
    set_actor(session.actor)
    set_session(session)

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # Prefer the ALREADY-RUNNING engine over constructing a second one:
    # a private engine would not be the one delegation/production actually
    # uses, and could mask a live misbinding (same reasoning as
    # delegation_probe.py's _stage_engine).
    eng = IntelligenceGraphEngine.get_active()
    if eng is None:
        eng = IntelligenceGraphEngine.get_or_create(defer_background_start=True)
    return eng


# ---------------------------------------------------------------------------
# Stage 6 — mcp_tool_catalog
# ---------------------------------------------------------------------------
async def _stage_mcp_tool_catalog(a: argparse.Namespace) -> str:
    harness_kg = _STATE["harness_kg"]
    if "engine" not in _STATE:
        _STATE["engine"] = await _acquire_engine(a)
    result = await harness_kg.mcp_tool_catalog(_STATE["engine"])
    return str(result.get("detail", result))


# ---------------------------------------------------------------------------
# Stage 7 — kg_general
# ---------------------------------------------------------------------------
async def _stage_kg_general(a: argparse.Namespace) -> str:
    harness_kg = _STATE["harness_kg"]
    if "engine" not in _STATE:
        _STATE["engine"] = await _acquire_engine(a)
    result = await harness_kg.kg_general(_STATE["engine"])
    return str(result.get("detail", result))


# ---------------------------------------------------------------------------
# Driver — mixed sync/async, one asyncio.run() for the whole process, sync
# stages dispatched via asyncio.to_thread so no second competing event loop
# is ever created.
# ---------------------------------------------------------------------------
async def run(a: argparse.Namespace) -> int:
    print(
        f"\nfull validation harness — base_url={a.base_url!r} mcp_url={a.mcp_url!r} "
        f"skill={a.skill!r} server={a.server!r} tool={a.tool!r}\n"
    )
    timings: list[tuple[str, float]] = []
    for n, stage in enumerate(STAGES, 1):
        if a.stop_after and STAGES.index(a.stop_after) + 1 < n:
            break
        t0 = time.monotonic()
        try:
            if stage == "mcp_handshake":
                d = await asyncio.to_thread(_stage_mcp_handshake, a)
            elif stage == "browser":
                d = await asyncio.to_thread(_stage_browser, a)
            elif stage == "api_gateway":
                d = await asyncio.to_thread(_stage_api_gateway, a)
            elif stage == "admission":
                d = await asyncio.to_thread(_stage_admission, a)
            elif stage == "delegation":
                d = await _stage_delegation(a)
            elif stage == "mcp_tool_catalog":
                d = await _stage_mcp_tool_catalog(a)
            else:
                d = await _stage_kg_general(a)
            dt = time.monotonic() - t0
            timings.append((stage, dt))
            _emit(stage, True, d, dt)
        except Exception as exc:  # report the full chain, never swallow
            dt = time.monotonic() - t0
            timings.append((stage, dt))
            _emit(stage, False, "", dt)
            print(
                f"\n  ROOT CAUSE at stage {n} ({stage}):\n      {_chain(exc)}\n",
                flush=True,
            )
            if a.traceback:
                traceback.print_exc()
            _print_timings(timings)
            print(
                f"\n  SUMMARY: FAIL at stage {n}/{len(STAGES)} ({stage}) -> exit code {n}\n",
                flush=True,
            )
            return n
    print("\n  all stages passed\n")
    _print_timings(timings)
    print(
        f"\n  SUMMARY: all {len(timings)} stage(s) PASSED -> exit code 0\n", flush=True
    )
    return 0


def _print_timings(timings: list[tuple[str, float]]) -> None:
    total = sum(t for _, t in timings) or 1e-9
    print("  === wall-clock by stage ===", flush=True)
    for stage, dt in timings:
        bar = "#" * int(40 * dt / total)
        print(f"    {stage:17s} {dt:7.2f}s {100 * dt / total:5.1f}%  {bar}", flush=True)
    print(f"    {'TOTAL':17s} {total:7.2f}s", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="agent-webui base URL for the browser/api_gateway/admission stages",
    )
    p.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8004/mcp",
        help="graph-os MCP streamable-http endpoint for the mcp_handshake stage",
    )
    p.add_argument(
        "--tenant-slug", default="homelab", help="passed to the admission stage"
    )
    p.add_argument("--skill", default="", help="passed through to delegation_probe.py")
    p.add_argument("--server", default="", help="passed through to delegation_probe.py")
    p.add_argument("--tool", default="", help="passed through to delegation_probe.py")
    p.add_argument(
        "--transport",
        default="streamable-http",
        help="transport whose process-authority path to mint for stages 6-7's engine",
    )
    p.add_argument(
        "--modules-dir",
        default="",
        help=(
            "repo-root directory containing scripts/ — required only as a fallback "
            "when this driver was piped via stdin AND the repo root is not already "
            f"on sys.path/PYTHONPATH (production default: {_DEFAULT_MODULES_DIR!r})"
        ),
    )
    p.add_argument(
        "--delegation-probe-path",
        default="",
        help="override the path to scripts/delegation_probe.py (default: derived "
        "from --modules-dir / the resolved repo root)",
    )
    p.add_argument(
        "--delegation-probe-arg",
        dest="delegation_probe_args",
        action="append",
        default=[],
        help=(
            "extra raw argument to pass through to delegation_probe.py verbatim "
            "(repeatable), e.g. --delegation-probe-arg=--require-tool"
        ),
    )
    p.add_argument("--stop-after", choices=STAGES, help="run only up to this stage")
    p.add_argument(
        "--timeout", type=float, default=15.0, help="per-HTTP-request timeout (s)"
    )
    p.add_argument(
        "--json", action="store_true", help="also emit a JSON summary at the end"
    )
    p.add_argument("--traceback", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _build_parser().parse_args(argv)

    harness_mcp, harness_browser, harness_kg, repo_root = _import_harness_modules(
        a.modules_dir or None
    )
    _STATE["harness_mcp"] = harness_mcp
    _STATE["harness_browser"] = harness_browser
    _STATE["harness_kg"] = harness_kg
    _STATE["repo_root"] = repo_root

    rc = asyncio.run(run(a))

    if a.json:
        import json as _json

        print(
            _json.dumps(
                {
                    "exit_code": rc,
                    "stages": STAGES,
                    "failed_stage": STAGES[rc - 1] if 0 < rc <= len(STAGES) else None,
                },
                default=str,
            )
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
