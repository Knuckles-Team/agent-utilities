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
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github/workflows")

#: The composite action every job MUST run before a `uv sync` that does not
#: `--no-install-package agent-connector-sdk` (it is a hard base dependency,
#: never excludable — see `agent-connector-sdk` in `[tool.uv.sources]`,
#: pyproject.toml). Unlike epistemic-graph/langfuse-agent (excluded from
#: resolution entirely via `--no-install-package`, so their untracked sibling
#: directory is never needed), this package's editable path source MUST exist
#: on disk for `uv sync --frozen` to resolve at all — real CI provisions it
#: with this action; a tracked-files-only export never has it materialized.
SDK_CHECKOUT_ACTION = "./.github/actions/checkout-agent-connector-sdk"
SDK_SIBLING_DIR = Path(".uv-workspace-siblings/agent-connector-sdk")

#: Commands that build a job's environment. ``uv sync`` is enforced (it is
#: fully offline-resolvable from ``uv.lock`` and is what has actually broken);
#: ``pip install -e .`` is *reported* but not replayed, because resolving it
#: reaches PyPI for the whole dependency closure and cannot fit the merge
#: queue's per-check budget. Reporting it keeps the omission visible instead
#: of letting it look covered.
ENFORCED = re.compile(r"\buv\s+sync\b")
REPORTED = re.compile(r"\bpip\s+install\s+(?:-e\s+\.|--editable\s+\.)")

_TIMEOUT_SECONDS = 45

#: Frozen list of env builds already known not to resolve — a ratchet, not an
#: exemption. See the file's own header and :func:`load_baseline`.
BASELINE_PATH = Path("scripts/security/ci_environment_contract_baseline.txt")


@dataclass(frozen=True)
class EnvBuild:
    """One environment-build command lifted from a workflow's ``run:`` block."""

    workflow: str
    command: str
    enforced: bool
    job: str = ""

    @property
    def argv(self) -> list[str]:
        return shlex.split(self.command)


def _job_of_line(text: str) -> list[str]:
    """The enclosing top-level job name for every line index in *text*.

    A minimal companion scan to :func:`_run_blocks`: job keys are exactly
    2-space-indented under a top-level ``jobs:`` key in every workflow this
    repository writes. Kept as its own tiny regex pass rather than folded
    into the YAML load below, so a schema change elsewhere still cannot break
    command extraction -- only the job *label* attached to it degrades to
    ``""``, which is handled explicitly by callers.
    """
    lines = text.splitlines()
    result: list[str] = [""] * len(lines)
    in_jobs = False
    current = ""
    for i, line in enumerate(lines):
        if re.match(r"^jobs:\s*(#.*)?$", line):
            in_jobs = True
            current = ""
        elif in_jobs:
            m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*(#.*)?$", line)
            if m:
                current = m.group(1)
        result[i] = current
    return result


_RUN_KEY = re.compile(r"^(\s*)-?\s*run:\s*(\|-?|>-?|)\s*(.*)$")


def _block_body(lines: list[str], start: int, base: int) -> tuple[list[str], int]:
    """Stripped lines of the block scalar starting at *start*, and the next index."""
    body: list[str] = []
    i = start
    while i < len(lines):
        nxt = lines[i]
        if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= base:
            break
        body.append(nxt.strip())
        i += 1
    return body, i


def _fold_continuations(body: list[str]) -> list[str]:
    """Fold shell line-continuations back into single commands."""
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
    return folded


def _block_commands(style: str, body: list[str]) -> list[str]:
    """Commands of one block scalar; a ``>`` folded scalar joins with spaces."""
    folded = _fold_continuations(body)
    if style.startswith(">"):
        return [" ".join(folded)]
    return folded


def _run_blocks(text: str) -> list[tuple[str, str]]:
    """Every ``(job, shell body)`` pair under a ``run:`` key in *text*.

    Deliberately a scanner rather than a YAML load: the workflows use block
    scalars (``run: |`` and ``run: >-``) whose bodies are plain shell, and a
    scanner cannot be broken by an unrelated schema change elsewhere in the
    file. Continuation backslashes are folded so a command split across lines
    is recovered whole -- which is how *both* real sync steps are written.
    """
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    job_of_line = _job_of_line(text)
    i = 0
    while i < len(lines):
        m = _RUN_KEY.match(lines[i])
        if not m:
            i += 1
            continue
        job = job_of_line[i]
        indent, style, inline = m.group(1), m.group(2), m.group(3)
        if not style and inline:
            blocks.append((job, inline))
            i += 1
            continue
        body, i = _block_body(lines, i + 1, len(indent))
        blocks.extend((job, command) for command in _block_commands(style, body))
    return blocks


def sdk_checkout_jobs(text: str) -> set[str]:
    """Job names in *text* whose steps run :data:`SDK_CHECKOUT_ACTION`.

    A real (alias-resolving) YAML load, deliberately scoped to the ``uses:``
    field alone -- the one place a full parse is safe, since GitHub Actions
    jobs commonly share this exact step via a YAML anchor/alias (``&sdk-
    checkout`` / ``*sdk-checkout``) that a plain-text scan cannot follow.
    Malformed/unparseable YAML degrades to "no job provisions it", which
    makes every enforced build in that file MORE strict, never less.
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return set()
    if not isinstance(doc, dict):
        return set()
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict):
        return set()
    provisioned: set[str] = set()
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict) and step.get("uses") == SDK_CHECKOUT_ACTION:
                provisioned.add(name)
                break
    return provisioned


def discover(repo_root: Path) -> list[EnvBuild]:
    """Environment-build commands across every workflow, discovered not listed."""
    found: list[EnvBuild] = []
    wf_dir = repo_root / WORKFLOW_DIR
    for path in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for job, block in _run_blocks(text):
            cmd = block.strip()
            if ENFORCED.search(cmd):
                found.append(EnvBuild(path.name, cmd, enforced=True, job=job))
            elif REPORTED.search(cmd):
                found.append(EnvBuild(path.name, cmd, enforced=False, job=job))
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


def _provisioned_jobs_by_workflow(repo_root: Path) -> dict[str, set[str]]:
    """``{workflow filename: {job names that run SDK_CHECKOUT_ACTION}}``."""
    wf_dir = repo_root / WORKFLOW_DIR
    out: dict[str, set[str]] = {}
    for path in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        out[path.name] = sdk_checkout_jobs(path.read_text(encoding="utf-8"))
    return out


#: Just enough of a `pyproject.toml`/package for `uv sync --frozen --dry-run`
#: to resolve a path-source dependency -- NOT a stand-in for the real SDK's
#: content (this gate has no business validating agent-connector-sdk's own
#: code; that is that repository's contract, not this one's). It models
#: exactly the one real-CI-observable fact this gate can assert offline: the
#: job provisions *some* directory there via SDK_CHECKOUT_ACTION before `uv
#: sync` runs, so a tracked-files-only export must do the same to reproduce
#: CI's actual resolvable state instead of manufacturing a failure CI never
#: has.
_SDK_STUB_PYPROJECT = """\
[project]
name = "agent-connector-sdk"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
"""


def _set_sdk_stub_present(tree: Path, present: bool) -> None:
    """Materialize or remove the placeholder SDK sibling directory in *tree*.

    Toggled per-build (not once for the whole tree) because whether it should
    exist is itself the thing under test: a job that omits
    SDK_CHECKOUT_ACTION must still see "Distribution not found" here, exactly
    as its real CI run would.
    """
    sibling = tree / SDK_SIBLING_DIR
    if present:
        if sibling.is_dir():
            return
        pkg = sibling / "agent_connector_sdk"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()
        (sibling / "pyproject.toml").write_text(_SDK_STUB_PYPROJECT, encoding="utf-8")
    elif sibling.is_dir():
        shutil.rmtree(sibling)


#: A workflow flag that EXCLUDES a package from resolution. It names what CI
#: deliberately leaves out, not which build step this is, so it is not part of
#: a step's baseline identity (entries predating an added exclusion still name
#: the same step).
_EXCLUSION_FLAG = "--no-install-package"


def command_key(command: str) -> tuple[str, ...]:
    """Whitespace- and exclusion-insensitive identity of one sync command.

    Tokens are compared after ``shlex`` splitting, so spacing, line folding
    and quoting differences never make a baseline entry silently miss its
    build. ``--no-install-package <pkg>`` pairs are dropped (see
    :data:`_EXCLUSION_FLAG`); every other token, in order, is identity.
    """
    tokens = shlex.split(command)
    kept: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
        elif token == _EXCLUSION_FLAG:
            skip_next = True
        elif not token.startswith(f"{_EXCLUSION_FLAG}="):
            kept.append(token)
    return tuple(kept)


def load_baseline(repo_root: Path) -> set[tuple[str, str]]:
    """``{(workflow, command)}`` already known not to resolve.

    A missing baseline file is an empty baseline — i.e. *stricter*, every
    failure is new. That direction is deliberate: a lost or mistyped baseline
    path must never silently excuse a real regression.
    """
    path = repo_root / BASELINE_PATH
    if not path.is_file():
        return set()
    entries: set[tuple[str, str]] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        workflow, _, command = line.partition("\t")
        if command:
            entries.add((workflow.strip(), command.strip()))
    return entries


def _baseline_index(
    baseline: set[tuple[str, str]],
) -> dict[tuple[str, tuple[str, ...]], tuple[str, str]]:
    """Baseline entries keyed by ``(workflow, command_key)``."""
    return {
        (workflow, command_key(command)): (workflow, command)
        for workflow, command in baseline
    }


#: ``(ok, baselined) -> (status, counts as a failure, is a stale entry)``.
_RATCHET: dict[tuple[bool, bool], tuple[str, bool, bool]] = {
    (False, True): ("known-broken (baselined)", False, False),
    (False, False): ("NEW FAILURE", True, False),
    (True, True): ("FIXED — remove from baseline", True, True),
    (True, False): ("resolved", False, False),
}


def orphaned_baseline_entries(
    baseline: set[tuple[str, str]], builds: list[EnvBuild]
) -> list[tuple[str, str]]:
    """Baseline entries that match no discovered build (workflow/step gone)."""
    live = {(build.workflow, command_key(build.command)) for build in builds}
    return sorted(
        entry for key, entry in _baseline_index(baseline).items() if key not in live
    )


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

    baseline = load_baseline(repo_root)
    index = _baseline_index(baseline)
    provisioned = _provisioned_jobs_by_workflow(repo_root)
    with tempfile.TemporaryDirectory(prefix="ci-env-contract-") as tmp:
        tree = Path(tmp) / "tracked"
        export_tracked_tree(repo_root, tree)
        results = []
        failures = 0
        stale: list[dict] = []
        for build in enforced:
            _set_sdk_stub_present(
                tree, build.job in provisioned.get(build.workflow, set())
            )
            ok, detail = replay(tree, build)
            baselined = (build.workflow, command_key(build.command)) in index
            status, failed, is_stale = _RATCHET[(ok, baselined)]
            failures += int(failed)
            if is_stale:
                stale.append({"workflow": build.workflow, "command": build.command})
            results.append(
                {
                    "workflow": build.workflow,
                    "command": build.command,
                    "ok": ok,
                    "baselined": baselined,
                    "status": status,
                    "detail": detail,
                }
            )

    orphaned = orphaned_baseline_entries(baseline, enforced)
    failures += len(orphaned)
    stale.extend({"workflow": w, "command": c} for w, c in orphaned)
    payload = {
        "ok": failures == 0,
        "baselinedKnownBroken": sorted(f"{w}: {c}" for w, c in baseline),
        "staleBaselineEntries": stale,
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
            f"{failures} CI environment build(s) regressed against "
            f"{BASELINE_PATH}. A build that newly fails to resolve means every "
            "gate in that workflow would be skipped and the workflow would red "
            "out before running a single contract check. A build listed as "
            "known-broken that now RESOLVES must be removed from the baseline "
            "— the ratchet only shrinks."
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

    _self_check_ratchet()


def _self_check_ratchet() -> None:
    """Steps 4-6: the ratchet moves in BOTH directions and the matcher is exact.

    Proven on synthetic entries so the proof does not depend on the real
    baseline still carrying debt -- a fully burned-down (empty) baseline is
    the goal state, not a broken gate.
    """
    if _RATCHET[(False, True)][1] or not _RATCHET[(False, False)][1]:
        raise AssertionError(
            "self-check: a known-broken build must be excused and a new one must fail"
        )
    if not (_RATCHET[(True, True)][1] and _RATCHET[(True, True)][2]):
        raise AssertionError(
            "self-check: a baselined build that resolves must fail as stale"
        )
    entry = ("synthetic.yml", "uv sync --frozen --group g")
    folded = EnvBuild(
        "synthetic.yml",
        "uv  sync --frozen --group g --no-install-package epistemic-graph",
        enforced=True,
    )
    if orphaned_baseline_entries({entry}, [folded]):
        raise AssertionError(
            "self-check: the matcher missed an exclusion/whitespace variant"
        )
    other = EnvBuild("synthetic.yml", "uv sync --frozen --group other", enforced=True)
    if orphaned_baseline_entries({entry}, [other]) != [entry]:
        raise AssertionError(
            "self-check: a baseline entry naming no live build was not flagged"
        )


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
