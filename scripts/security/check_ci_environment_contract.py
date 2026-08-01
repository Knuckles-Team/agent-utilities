#!/usr/bin/env python3
"""Replay every CI workflow's *environment build* against a tracked-only tree.

CONCEPT:AU-OS.governance.tiered-merge-gate — D-CIP-1.

**The vector this gate defends.** Every guardrail in
``.github/workflows/guardrails.yml`` and ``.github/workflows/security.yml``
runs *after* a ``uv sync --frozen`` step that builds the job's environment.
If that sync fails, **not one gate runs** and the whole workflow reds out
with zero contract signal. That is not hypothetical: on 2026-07-28 all three
push workflows failed at exactly that step::

    error: Failed to determine installation plan
      Caused by: Distribution not found at:
        file:///home/runner/work/agent-utilities/agent-utilities/.uv-workspace-siblings/epistemic-graph

``uv.lock`` pins ``epistemic-graph`` and ``langfuse-agent`` to editable
workspace members under ``.uv-workspace-siblings/`` (see
``[tool.uv.sources]`` in ``pyproject.toml`` and ``scripts/uv_workspace.py``,
which materializes that directory from the *local workspace's* sibling
repos). That directory is **untracked** — it does not exist in a fresh
``git clone``, which is all a GitHub runner ever has. A sync that does not
exclude those two packages therefore cannot resolve, on any runner, ever.

**Why no existing local gate catches it.** Every developer machine has
``.uv-workspace-siblings/`` materialized, or a warm ``.venv``, so the sync
that is structurally impossible on a runner succeeds locally every time. The
``uv-lock`` pre-commit hook *regenerates* the lock rather than proving the
committed lock is installable from a bare checkout, and no hook exports the
tracked tree. The failure is invisible to ``pre-commit`` by construction —
the precise "CI can fail where local cannot" defect this gate closes.

**What it does.** Discovers the environment-build commands from the
workflows themselves (never an enumerated copy, which would drift the moment
a workflow changes), exports the repository's **tracked files only** into a
throwaway tree — byte-for-byte what ``actions/checkout`` produces — and
replays each command there with ``--dry-run`` appended. ``--dry-run`` stops
before downloading or installing anything but still performs the full
resolution and installation-plan step, which is exactly the step that fails.
It costs well under a second per command.

**Fail-closed.** Discovering *zero* environment builds is a refusal, not a
pass: a gate whose discovery silently stops matching is the failure mode this
repository has already been burned by (D-MW-9, where the merge queue's
``scripts/security/check_*.py`` glob never matched the 7 real checks living
at ``scripts/check_*.py``). Likewise an un-exportable tree is refused rather
than assumed clean.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")

#: Commands that build a job's environment. ``uv sync`` is enforced (it is
#: fully offline-resolvable from ``uv.lock`` and is what has actually broken);
#: ``pip install -e .`` is *reported* but not replayed, because resolving it
#: reaches PyPI for the whole dependency closure and cannot fit the merge
#: queue's per-check budget. Reporting it keeps the omission visible instead
#: of letting it look covered.
ENFORCED = re.compile(r"\buv\s+sync\b")
REPORTED = re.compile(r"\bpip\s+install\s+(?:-e\s+\.|--editable\s+\.)")

#: uv flags that take a value, so a replay can drop/keep them coherently.
_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class EnvBuild:
    """One environment-build command lifted from a workflow's ``run:`` block."""

    workflow: str
    command: str
    enforced: bool

    @property
    def argv(self) -> list[str]:
        return shlex.split(self.command)


def _run_blocks(text: str) -> list[str]:
    """Every shell body under a ``run:`` key in *text*.

    Deliberately a scanner rather than a YAML load: the workflows use block
    scalars (``run: |`` and ``run: >-``) whose bodies are plain shell, and a
    scanner cannot be broken by an unrelated schema change elsewhere in the
    file. Continuation backslashes are folded so a command split across lines
    is recovered whole -- which is how *both* real sync steps are written.
    """
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)-?\s*run:\s*(\|-?|>-?|)\s*(.*)$", line)
        if not m:
            i += 1
            continue
        indent, style, inline = m.group(1), m.group(2), m.group(3)
        if not style and inline:
            blocks.append(inline)
            i += 1
            continue
        body: list[str] = []
        base = len(indent)
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= base:
                break
            body.append(nxt.strip())
            i += 1
        # Fold shell line-continuations back into single commands.
        folded: list[str] = []
        acc = ""
        for raw in body:
            if raw.endswith("\\"):
                acc += raw[:-1].rstrip() + " "
                continue
            folded.append((acc + raw).strip())
            acc = ""
        if acc.strip():
            folded.append(acc.strip())
        # A `>-` folded scalar joins every line with spaces, not newlines.
        if style.startswith(">"):
            blocks.append(" ".join(folded))
        else:
            blocks.extend(folded)
    return blocks


def discover(repo_root: Path) -> list[EnvBuild]:
    """Environment-build commands across every workflow, discovered not listed."""
    found: list[EnvBuild] = []
    wf_dir = repo_root / WORKFLOW_DIR
    for path in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for block in _run_blocks(text):
            cmd = block.strip()
            if ENFORCED.search(cmd):
                found.append(EnvBuild(path.name, cmd, enforced=True))
            elif REPORTED.search(cmd):
                found.append(EnvBuild(path.name, cmd, enforced=False))
    return found


def export_tracked_tree(repo_root: Path, dest: Path) -> None:
    """Materialize *repo_root*'s tracked files at HEAD into *dest*.

    This is the whole point: an untracked file (a materialized
    ``.uv-workspace-siblings/``, a warm ``.venv``) must not be able to make a
    command succeed here that cannot succeed on a runner.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(  # noqa: S603
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )
    if archive.returncode != 0:
        raise RuntimeError(
            "could not export the tracked tree at HEAD "
            f"({archive.stderr.decode('utf-8', 'replace').strip()}) — an "
            "unverifiable environment contract is REFUSED, never assumed clean"
        )
    extract = subprocess.run(  # noqa: S603
        ["tar", "-x", "-C", str(dest)],
        input=archive.stdout,
        capture_output=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )
    if extract.returncode != 0:
        raise RuntimeError(
            "could not unpack the exported tracked tree: "
            f"{extract.stderr.decode('utf-8', 'replace').strip()}"
        )


def _replay_argv(argv: list[str]) -> list[str]:
    """*argv* with ``--dry-run`` added, so resolution + the installation plan
    run (the step that fails) without downloading or installing anything."""
    return [*argv, "--dry-run"] if "--dry-run" not in argv else list(argv)


def replay(tree: Path, build: EnvBuild) -> tuple[bool, str]:
    """Run *build* inside *tree*; return ``(ok, detail)``."""
    try:
        proc = subprocess.run(  # noqa: S603
            _replay_argv(build.argv),
            cwd=str(tree),
            capture_output=True,
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run: {exc}"
    if proc.returncode == 0:
        return True, "resolved"
    tail = (proc.stderr or proc.stdout).decode("utf-8", "replace").strip()
    return False, "\n".join(tail.splitlines()[-4:])


def check(repo_root: Path) -> tuple[int, dict]:
    builds = discover(repo_root)
    enforced = [b for b in builds if b.enforced]
    reported = [b for b in builds if not b.enforced]

    if not enforced:
        return 1, {
            "ok": False,
            "error": (
                "discovered ZERO enforceable `uv sync` environment builds under "
                f"{WORKFLOW_DIR} — a gate that finds nothing to check must "
                "refuse, not pass (D-MW-9 class: silent discovery drift)"
            ),
            "workflowsScanned": sorted(
                p.name for p in (repo_root / WORKFLOW_DIR).glob("*.y*ml")
            ),
        }

    with tempfile.TemporaryDirectory(prefix="ci-env-contract-") as tmp:
        tree = Path(tmp) / "tracked"
        export_tracked_tree(repo_root, tree)
        results = []
        failures = 0
        for build in enforced:
            ok, detail = replay(tree, build)
            failures += 0 if ok else 1
            results.append(
                {
                    "workflow": build.workflow,
                    "command": build.command,
                    "ok": ok,
                    "detail": detail,
                }
            )

    payload = {
        "ok": failures == 0,
        "enforced": results,
        "notReplayed": [
            {
                "workflow": b.workflow,
                "command": b.command,
                "reason": (
                    "pip resolves the full dependency closure over the network; "
                    "replaying it does not fit the merge-queue budget. Declared "
                    "here so the omission stays visible rather than looking covered."
                ),
            }
            for b in reported
        ],
    }
    if failures:
        payload["error"] = (
            f"{failures} CI environment build(s) cannot resolve from a "
            "tracked-only checkout — every gate in the affected workflow(s) "
            "would be skipped and the workflow would red out before running "
            "a single contract check"
        )
    return (1 if failures else 0), payload


def self_check(repo_root: Path) -> None:
    """Prove this gate trips on a known-bad input rather than merely existing.

    1. discovery finds the real workflows' sync steps (not zero);
    2. a sync command that cannot resolve is reported ``ok=False`` — the
       2026-07-28 failure, reconstructed; and
    3. zero discovery fails closed instead of passing vacuously.
    """
    builds = discover(repo_root)
    enforced = [b for b in builds if b.enforced]
    if not enforced:
        raise AssertionError(
            "self-check: discovery found no `uv sync` steps in the real "
            f"{WORKFLOW_DIR} — discovery has drifted and this gate is blind"
        )

    with tempfile.TemporaryDirectory(prefix="ci-env-selfcheck-") as tmp:
        tree = Path(tmp) / "tracked"
        export_tracked_tree(repo_root, tree)
        broken = EnvBuild(
            "synthetic.yml",
            "uv sync --frozen --group definitely-no-such-group",
            enforced=True,
        )
        ok, _detail = replay(tree, broken)
        if ok:
            raise AssertionError(
                "self-check: an unresolvable `uv sync` was reported as passing "
                "— this gate would not have caught the 2026-07-28 CI failure"
            )

    empty = tempfile.TemporaryDirectory(prefix="ci-env-empty-")
    try:
        (Path(empty.name) / WORKFLOW_DIR).mkdir(parents=True)
        rc, payload = check(Path(empty.name))
        if rc == 0 or payload.get("ok"):
            raise AssertionError(
                "self-check: a repository with NO discoverable environment "
                "builds was reported as passing — discovery must fail closed"
            )
    finally:
        empty.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="prove the gate trips on a known-bad input, then run normally",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent.parent

    if args.self_check:
        try:
            self_check(repo_root)
        except AssertionError as exc:
            print(json.dumps({"ok": False, "selfCheck": "FAILED", "error": str(exc)}))
            return 1

    try:
        rc, payload = check(repo_root)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    payload["selfCheck"] = args.self_check
    print(json.dumps(payload, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
