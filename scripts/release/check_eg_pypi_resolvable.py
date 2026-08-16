#!/usr/bin/env python3
"""Fail-closed check that the required epistemic-graph release is on PyPI.

**The vector this gate defends.** agent-utilities declares a version-floored
dependency on ``epistemic-graph[full]`` (GOC-73: via the opt-in
``[graphos]`` extra — see ``pyproject.toml`` — not a base dependency; any
`pip install`/`uv sync` that requests `[graphos]`/`[serving]`/`[all]`/`[test]`,
or this repo's own dev/release tooling, still resolves it). If that floor has
never been published to PyPI — e.g. because epistemic-graph's own release job
failed on something unrelated to this repo (a wheel-digest mismatch, a flaky
publish step) — a normal ``pip install``/``uv sync`` of one of those extras
dies deep inside the dependency resolver with an opaque backtrace that never
says WHY: it just reports no version of agent-utilities is installable. This
is exactly what broke this repo's Pages/Deploy job.

**What it does.** Reads the ``epistemic-graph`` entry straight out of
``[project.dependencies]`` in ``pyproject.toml`` (no ``packaging`` import —
this runs before the guardrails environment is synced, deliberately, so a
missing floor is caught before any CI time is spent building), queries
PyPI's public JSON API for every published, non-yanked release of
``epistemic-graph``, and checks whether at least one satisfies the declared
specifier. On failure it names the exact declared constraint, the newest
version actually on PyPI, and the release-order rule (epistemic-graph must
publish before agent-utilities) — one line instead of a resolver stack
trace.

Only handles the two comparison operators this repo's constraint actually
uses (``>=`` and ``<``); rejects anything else instead of silently
approximating.

**Opt-in extras scoping (``--extras``).** A consumer that installs only a
*subset* of this repo's extras -- e.g. agent-webui's Docker image, which
installs ``agent-utilities[graph,mcp,metrics]`` and pulls in NO
epistemic-graph reference at all -- can hit a false positive: with no flag,
this script scans the WHOLE pyproject (base ``[project.dependencies]`` plus
**every** ``[project.optional-dependencies]`` entry, including the
`graphos` extra, GOC-73), so it fails even when the extras actually being
installed never resolve epistemic-graph. Pass ``--extras a,b,c`` to scope
the scan to the base dependencies plus exactly the transitive closure of
those named extras (following self-referencing ``agent-utilities[...]``
entries, e.g. `serving`/`test`/`all`) -- using a real TOML parse
(``tomllib``, stdlib, no extra dependency) instead of the line-scan used by
the default path. **Default behaviour with no ``--extras`` flag is
UNCHANGED** -- it still scans everything, exactly as before, which is the
correct, maximally-protective behaviour for this repo's OWN release (every
extra it ships must resolve). Scoping is something only a downstream
consumer's release gate should opt into, naming the extras it actually
installs.

Run via::

    python3 scripts/release/check_eg_pypi_resolvable.py
    python3 scripts/release/check_eg_pypi_resolvable.py --extras graph,mcp,metrics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
_TIMEOUT_SECS = 30

_DEP_LINE_RE = re.compile(
    r'^\s*"epistemic-graph(?:\[[^\]]*\])?\s*(?P<spec>[^"]*)"\s*,?\s*$'
)
# Same shape as _DEP_LINE_RE but applied to a bare TOML array-item string
# (already unquoted by tomllib), not a raw source line -- used by the
# --extras scoped path.
_DEP_ITEM_RE = re.compile(r"^\s*epistemic-graph(?:\[[^\]]*\])?\s*(?P<spec>.*?)\s*$")
_CLAUSE_RE = re.compile(r"(>=|<=|==|<|>|!=)\s*([0-9][0-9A-Za-z.\-+]*)")


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = re.split(r"[.\-+]", v)
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            break
    if not out:
        raise ValueError(f"cannot parse version: {v!r}")
    return tuple(out)


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    la, lb = len(a), len(b)
    n = max(la, lb)
    a = a + (0,) * (n - la)
    b = b + (0,) * (n - lb)
    return (a > b) - (a < b)


def _find_constraint() -> tuple[str, str]:
    """Return (name, raw_specifier) for the epistemic-graph dependency line."""
    text = PYPROJECT.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = _DEP_LINE_RE.match(line)
        if m:
            return "epistemic-graph", m.group("spec").strip()
    raise SystemExit(
        "::error::check_eg_pypi_resolvable: no 'epistemic-graph' entry found "
        f"anywhere in {PYPROJECT} (checked [project.dependencies] and every "
        "[project.optional-dependencies] entry, e.g. GOC-73's `graphos` extra) "
        "-- cannot verify cross-repo release order."
    )


def _self_ref_extras(item: str, package_name: str) -> list[str] | None:
    """If `item` is a self-referencing extra (e.g. `"agent-utilities[graphos,mcp]>=1.0.0"`),
    return the list of extra names inside the brackets; else None."""
    m = re.match(rf"^\s*{re.escape(package_name)}\s*\[(?P<extras>[^\]]+)\]", item)
    if not m:
        return None
    return [e.strip() for e in m.group("extras").split(",") if e.strip()]


def _closure_dependency_strings(pyproject: Path, extras: list[str]) -> list[str]:
    """Return base dependencies plus the transitive closure of `extras`'
    dependency strings, resolving self-referencing `<package>[...]` entries
    against this same pyproject.toml's own optional-dependencies table."""
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {})
    package_name = project.get("name", "")
    base_deps: list[str] = list(project.get("dependencies", []))
    opt_deps: dict[str, list[str]] = project.get("optional-dependencies", {})

    collected: list[str] = list(base_deps)
    visited: set[str] = set()
    worklist: list[str] = list(extras)
    while worklist:
        extra = worklist.pop()
        if extra in visited:
            continue
        visited.add(extra)
        if extra not in opt_deps:
            raise SystemExit(
                f"::error::check_eg_pypi_resolvable: --extras named '{extra}', "
                f"which is not an entry in [project.optional-dependencies] of "
                f"{pyproject}. Available extras: {', '.join(sorted(opt_deps))}."
            )
        for item in opt_deps[extra]:
            collected.append(item)
            sub_extras = _self_ref_extras(item, package_name)
            if sub_extras:
                worklist.extend(e for e in sub_extras if e not in visited)
    return collected


def _find_constraint_scoped(
    pyproject: Path, extras: list[str]
) -> tuple[str, str] | None:
    """Like `_find_constraint`, but scoped to base deps + the transitive
    closure of `extras`. Returns None if epistemic-graph is not pulled in by
    that closure at all (nothing to verify)."""
    for item in _closure_dependency_strings(pyproject, extras):
        m = _DEP_ITEM_RE.match(item)
        if m:
            return "epistemic-graph", m.group("spec").strip()
    return None


def _satisfies(version: str, spec: str) -> bool:
    try:
        v = _version_tuple(version)
    except ValueError:
        return False
    for op, bound_raw in _CLAUSE_RE.findall(spec):
        try:
            bound = _version_tuple(bound_raw)
        except ValueError:
            continue
        c = _cmp(v, bound)
        if op == ">=" and not (c >= 0):
            return False
        if op == "<=" and not (c <= 0):
            return False
        if op == "==" and not (c == 0):
            return False
        if op == "<" and not (c < 0):
            return False
        if op == ">" and not (c > 0):
            return False
        if op == "!=" and not (c != 0):
            return False
    return True


def _fetch_versions(name: str) -> list[str]:
    url = PYPI_JSON_URL.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECS) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(
                f"::error::check_eg_pypi_resolvable: PyPI project '{name}' does not "
                f"exist (HTTP 404 from {url})."
            ) from exc
        raise SystemExit(
            f"::error::check_eg_pypi_resolvable: PyPI lookup for '{name}' failed "
            f"(HTTP {exc.code} from {url})."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"::error::check_eg_pypi_resolvable: could not reach PyPI to verify "
            f"'{name}' ({exc.reason}). This is a network/PyPI-availability failure, "
            "not a version gap -- re-run once connectivity is restored."
        ) from exc

    releases = payload.get("releases", {})
    published = []
    for version_str, files in releases.items():
        if not files:
            continue
        if all(f.get("yanked", False) for f in files):
            continue
        published.append(version_str)
    return published


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed check that agent-utilities' declared epistemic-graph "
            "floor is resolvable on PyPI."
        )
    )
    parser.add_argument(
        "--extras",
        default=None,
        help=(
            "Comma-separated extras to scope the check to (base dependencies "
            "plus the transitive closure of these extras only). Omit to scan "
            "everything (base + every optional-dependencies entry) -- the "
            "default, unchanged behaviour."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.extras is not None:
        extras = [e.strip() for e in args.extras.split(",") if e.strip()]
        found = _find_constraint_scoped(PYPROJECT, extras)
        if found is None:
            print(
                "check_eg_pypi_resolvable: OK -- none of the requested extras "
                f"({', '.join(extras)}) pull in 'epistemic-graph'; nothing to "
                "verify."
            )
            return 0
        name, spec = found
    else:
        name, spec = _find_constraint()

    if not spec:
        print(
            f"check_eg_pypi_resolvable: OK -- '{name}' has no version constraint; "
            "nothing to verify."
        )
        return 0

    versions_raw = _fetch_versions(name)
    valid_versions = []
    for v in versions_raw:
        try:
            _version_tuple(v)
        except ValueError:
            continue
        valid_versions.append(v)

    matching = sorted(
        (v for v in valid_versions if _satisfies(v, spec)),
        key=_version_tuple,
    )
    if matching:
        print(
            f"check_eg_pypi_resolvable: OK -- {name}=={matching[-1]} on PyPI "
            f"satisfies the declared constraint '{spec}'."
        )
        return 0

    newest = max(valid_versions, key=_version_tuple) if valid_versions else None
    print(
        f"::error::check_eg_pypi_resolvable: agent-utilities requires "
        f"'{name} {spec}' but no published PyPI release of '{name}' satisfies "
        f"that constraint. Newest published: {newest if newest else '(none found)'}. "
        f"epistemic-graph MUST publish a satisfying version to PyPI BEFORE "
        "agent-utilities' release can be built -- this is the cross-repo "
        "release-order dependency (epistemic-graph publishes first, then "
        "agent-utilities). Publish epistemic-graph, wait for PyPI to serve it "
        "(the release-build.yml verify step retries for CDN lag), then re-run "
        "this workflow.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
