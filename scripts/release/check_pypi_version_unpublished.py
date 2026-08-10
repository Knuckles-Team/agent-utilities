#!/usr/bin/env python3
"""Skip a redundant PyPI publish when this exact version is already released.

**The problem this fixes (BUG-072).** ``uv publish`` fails hard with PyPI's
``400 File already exists`` when the current ``pyproject.toml`` version has
already been published and the wheel content has since changed (e.g. commits
landed on ``main`` -- and a fresh release tag was pushed -- without a version
bump). That is a no-op, not a CI failure: there is nothing to publish that
isn't already there. This script runs before ``uv publish`` and tells the
workflow whether to skip it.

**What it does.** Reads the version straight out of the wheel filename this
job just built (``agent_utilities-<version>-*.whl``), queries PyPI's public
JSON API for ``agent-utilities``, and writes ``already_published=true|false``
to ``$GITHUB_OUTPUT`` (falls back to stdout when run outside CI).

**Fails OPEN, deliberately.** This is a usability skip, not a security gate --
unlike the release-tag / environment gates (D-ORC-10), a wrong answer here
only ever costs an extra (harmless) publish attempt, never a bypassed
protection. So on any inability to find/parse the wheel or reach PyPI, it
writes ``already_published=false`` and exits 0: the real ``uv publish`` step
still runs and reports the actual outcome. A network hiccup in this check
must never silently block a legitimate release.

Run via::

    python3 scripts/release/check_pypi_version_unpublished.py dist/agent_utilities-*.whl
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
_TIMEOUT_SECS = 30
_WHEEL_RE = re.compile(r"^agent_utilities-(?P<version>[^-]+)-.+\.whl$")


def _emit(already_published: bool, message: str) -> None:
    print(message)
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"already_published={'true' if already_published else 'false'}\n")


def main(argv: list[str]) -> int:
    patterns = argv[1:] or ["dist/agent_utilities-*.whl"]
    wheels: list[str] = []
    for pattern in patterns:
        wheels.extend(glob.glob(pattern))
    if not wheels:
        _emit(False, f"no wheel matched {patterns!r} -- proceeding with normal publish")
        return 0

    match = _WHEEL_RE.match(os.path.basename(sorted(wheels)[0]))
    if not match:
        _emit(
            False,
            f"could not parse a version out of {wheels[0]!r} -- proceeding with normal publish",
        )
        return 0
    version = match.group("version")

    try:
        with urllib.request.urlopen(
            PYPI_JSON_URL.format(name="agent-utilities"), timeout=_TIMEOUT_SECS
        ) as resp:
            data = json.load(resp)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        _emit(False, f"could not query PyPI ({exc}) -- proceeding with normal publish")
        return 0

    released = set(data.get("releases", {}) or {})
    if version in released:
        _emit(
            True,
            f"agent-utilities {version} is already published on PyPI -- "
            "skipping publish (no-op, not a failure). Bump the version to "
            "release the new commits.",
        )
    else:
        _emit(False, f"agent-utilities {version} is not yet on PyPI -- publishing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
