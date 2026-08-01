#!/usr/bin/env python3
"""Reject host-specific data and local identities from tracked public artifacts.

Identifiers are derived in memory from the current account, home, checkout and
host. Findings report only ``file:line`` and a category; the sensitive matched
value is never written or printed. Machine paths and persisted path fields are
checked independently, so the gate remains useful in clean CI environments.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import pwd
import re
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_TEXT_SUFFIXES = frozenset({".md", ".json", ".yaml", ".yml", ".toml"})
_SOURCE_SUFFIXES = frozenset(
    {".js", ".md", ".ps1", ".py", ".rs", ".sh", ".ts", ".yaml", ".yml"}
)
_GENERIC_IDENTIFIERS = frozenset(
    {
        "admin",
        "agent",
        "apps",
        "build",
        "developer",
        "genius",
        "home",
        "localhost",
        "maintainer",
        "maintainers",
        "root",
        "runner",
        "service",
        "user",
        "workspace",
    }
)
_HOME_PATH_PATTERN = (
    r"(?:(?<![A-Za-z0-9_.-])/home/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"(?<![A-Za-z0-9_.-])/Users/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"(?<![A-Za-z0-9_.-])/mnt/[A-Za-z]/Users/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+(?:[\\/]|\b))"
)
_HOME_PATH_RE = re.compile(_HOME_PATH_PATTERN, re.IGNORECASE)
_PERSISTED_FIELD_RE = re.compile(
    r"[\"']?(?P<field>workspace_path|source_path|skill_path|local_path|source_file|"
    r"eg_ledger_path)[\"']?\s*[:=]\s*(?P<value>.+)",
    re.IGNORECASE,
)
_NEUTRAL_URI_RE = re.compile(
    r"^[\s\"']*(?:repo|skill|connector|design)://", re.IGNORECASE
)
_INTERNAL_ENDPOINT_RE = re.compile(
    r"(?i)\b(?:[A-Za-z0-9-]+\.)+(?:arpa|internal)\b|"
    r"\b(?:[A-Za-z0-9-]+\.)*svc\.cluster\.local\b|"
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_SOURCE_INTERNAL_URL_RE = re.compile(
    r"(?i)https?://[^\s\"'<>]*(?:\.(?:arpa|internal|corp|lan)\b|"
    r"\.svc\.cluster\.local\b)"
)
_PRIVATE_KEY_LINE_RE = re.compile(r"^\s*-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----\s*$")
_CREDENTIAL_URI_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@")
_HOST_IDENTITY_RE = re.compile(r"(?i)\bssh://(?!\$\{)[^\s/@]+@")
_MACHINE_HOST_ID_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:rw?|host)[0-9]{3,}(?![a-z0-9])"
)
_NEUTRAL_AUTHOR_NAME = "repository maintainers"
_NEUTRAL_AUTHOR_EMAIL_SUFFIX = "@example.invalid"
_SCAN_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".acp-sessions",
        ".benchmarks",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".pytest_tmp",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site",
        "target",
        "venv",
        "workspace",
    }
)
_MAX_SCAN_FILES = 500_000


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    category: str
    content_hash: str
    ordinal: int

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.category}"


def _content_hash(text: str) -> str:
    """Irreversible fingerprint of a line's content, never the value itself.

    A SHA-256 digest cannot be inverted back to the sensitive substring that
    produced it, so it is safe to persist in the baseline file even though
    ``render()``/every printed message still withholds the matched value.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _identifier_from_path(value: str) -> set[str]:
    identifiers: set[str] = set()
    normalized = value.replace("\\", "/")
    for pattern in (r"/home/([^/]+)", r"/Users/([^/]+)"):
        identifiers.update(re.findall(pattern, normalized, flags=re.IGNORECASE))
    return identifiers


def derive_local_identifiers(root: Path = ROOT) -> frozenset[str]:
    candidates = {
        getpass.getuser(),
        pwd.getpwuid(os.getuid()).pw_name,
        socket.gethostname(),
        socket.gethostname().split(".", 1)[0],
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
        os.environ.get("USERNAME", ""),
        Path.home().name,
    }
    candidates.update(_identifier_from_path(str(Path.home())))
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidates.update(_identifier_from_path(result.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    for command in (
        ["git", "config", "--get", "user.name"],
        ["git", "config", "--get", "user.email"],
        ["git", "log", "-1", "--format=%an%n%ae"],
    ):
        try:
            result = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            for value in result.stdout.splitlines():
                candidates.add(value.strip())
        except OSError:
            pass
    return frozenset(
        value.casefold()
        for value in candidates
        if value
        and len(value) >= 4
        and value.casefold() not in _GENERIC_IDENTIFIERS
    )


def _is_deployment_doc(path: Path) -> bool:
    value = path.as_posix().casefold()
    return value.startswith("docs/recipes/") or any(
        marker in value
        for marker in (
            "deploy",
            "runbook",
            "configuration",
            "workspace-config",
            "mcp_auth",
            "secrets-auth",
        )
    )


def classify_line(
    line: str,
    *,
    identifiers: frozenset[str],
    deployment_doc: bool,
) -> frozenset[str]:
    categories: set[str] = set()
    persisted = _PERSISTED_FIELD_RE.search(line)
    if persisted and not _NEUTRAL_URI_RE.search(persisted.group("value")):
        value = persisted.group("value").strip(" \t,;)}]\"'").casefold()
        field = persisted.group("field")
        # A shell/template interpolation placeholder (``${WORKSPACE_ROOT}``, closing
        # brace already stripped above) is resolved at runtime by whatever consumes
        # the file, never a baked-in machine path — safe regardless of the field's
        # own casing, same reasoning as the existing uppercase-field exemption.
        is_template_placeholder = value.startswith("${")
        runtime_relative = (
            field.isupper() or is_template_placeholder
        ) and not re.match(r"^(?:[a-z]:|[/\\]|~)", value, re.IGNORECASE)
        if value not in {"", "none", "null", "unset"} and not runtime_relative:
            categories.add("persisted machine path")
    if "persisted machine path" not in categories and _HOME_PATH_RE.search(line):
        categories.add("machine-specific home path")
    folded = line.casefold()
    if any(re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", folded) for value in identifiers):
        categories.add("local account or host identifier")
    if _MACHINE_HOST_ID_RE.search(line):
        categories.add("machine-specific host identifier")
    if deployment_doc and _INTERNAL_ENDPOINT_RE.search(line):
        categories.add("hard-coded internal endpoint")
    if deployment_doc and _CREDENTIAL_URI_RE.search(line):
        categories.add("credential-bearing URI")
    if deployment_doc and _HOST_IDENTITY_RE.search(line):
        categories.add("hard-coded remote account")
    return frozenset(categories)


def classify_runtime_source_line(
    line: str, *, identifiers: frozenset[str]
) -> frozenset[str]:
    """Classify runtime/deployment source without applying public-doc path heuristics.

    Source code legitimately manipulates path-shaped values, so the generic
    ``source_path = ...`` rule would be noisy here. Concrete account paths,
    environment endpoints, credential-bearing URLs, and local identities are
    never legitimate package defaults and are checked for every shipped runtime
    and deployment file instead.

    D-CIP-10: the categories below used to be suffixed ``in changed source``,
    from when :func:`_runtime_source_artifacts` only looked at ``git diff``.
    That scope was the bug; the label is now ``in runtime source`` so the gate's
    own output cannot imply a narrower guarantee than it actually gives.
    """

    categories: set[str] = set()
    if _HOME_PATH_RE.search(line):
        categories.add("machine-specific home path in runtime source")
    folded = line.casefold()
    if any(
        re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", folded)
        for value in identifiers
    ):
        categories.add("local account or host identifier in runtime source")
    if _SOURCE_INTERNAL_URL_RE.search(line):
        categories.add("hard-coded internal endpoint in runtime source")
    if _CREDENTIAL_URI_RE.search(line):
        categories.add("credential-bearing URI in runtime source")
    if _PRIVATE_KEY_LINE_RE.fullmatch(line):
        categories.add("private key material in runtime source")
    return frozenset(categories)


def _is_public_artifact(name: str) -> bool:
    path = Path(name)
    if path.suffix.casefold() not in _TEXT_SUFFIXES:
        return False
    if "skills" in path.parts:
        return False
    return (
        path.parts[0] == "docs"
        or path.parts[0] == ".github"
        or len(path.parts) == 1
        or path.suffix.casefold() == ".toml"
    )


def _filesystem_files(root: Path) -> list[Path]:
    """Enumerate a bounded no-Git source snapshot without following links."""

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True):
        current = Path(directory)
        traversable: list[str] = []
        for name in sorted(directory_names):
            if name in _SCAN_EXCLUDED_DIRECTORIES or name.endswith(".egg-info"):
                continue
            metadata = (current / name).lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_ISDIR(metadata.st_mode):
                traversable.append(name)
        directory_names[:] = traversable
        for name in sorted(file_names):
            path = current / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            files.append(path)
            if len(files) > _MAX_SCAN_FILES:
                raise RuntimeError("privacy source inventory exceeds its file bound")
    return files


def _git_file_names(root: Path, command: list[str]) -> list[str] | None:
    """Return Git inventory names, or ``None`` for an immutable no-Git snapshot."""

    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [name for name in result.stdout.splitlines() if name]


def _tracked_artifacts(root: Path) -> list[Path]:
    names = _git_file_names(
        root,
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    )
    candidates = (
        _filesystem_files(root) if names is None else [root / name for name in names]
    )
    return [
        path
        for path in candidates
        if _is_public_artifact(path.relative_to(root).as_posix())
    ]


def _runtime_source_artifacts(root: Path) -> list[Path]:
    """Every **tracked** runtime/deployment source path, not merely the changed ones.

    D-CIP-10. This pass used to be scoped to ``git diff``/untracked files, which
    made the gate structurally blind to its own back-catalogue: a machine path
    that landed in an earlier commit was never re-examined, so it stayed public
    forever. Combined with :func:`_is_public_artifact`'s *location* filter — which
    admits only ``docs/``, ``.github/``, top-level and ``*.toml`` — the two passes
    left a hole exactly where the real leaks live. ``docker/*.yaml`` passes the
    suffix test and fails the location test, so nothing scanned it; the gate
    reported 7 lines in 2 ``docs/`` files while 11 tracked files carried machine
    paths, two of them a **personal** account name (D-PCC-1).

    Scanning the whole tracked tree here closes that hole without inventing a
    heuristic: the classification (:func:`classify_changed_source_line`) and the
    scope (:func:`_is_runtime_source_path`, which already lists ``docker``) were
    both already correct and already noise-tuned. Only the *diff* restriction was
    wrong.

    A public GitHub repository publishes its history, not just its tip, so
    "changed in this commit" was never the right boundary for a privacy gate.
    """

    names = _git_file_names(
        root,
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    )
    candidates = (
        _filesystem_files(root) if names is None else [root / name for name in names]
    )
    return sorted(
        (
            path
            for path in candidates
            if path.suffix.casefold() in _SOURCE_SUFFIXES
            and _is_runtime_source_path(path.relative_to(root))
        ),
        key=lambda path: path.as_posix(),
    )


def _is_runtime_source_path(path: Path) -> bool:
    """Scope the changed-source gate to shipped runtime/deployment material.

    Adversarial tests and the privacy scanners themselves intentionally contain
    synthetic bad values. Public docs are already scanned in full by
    ``_tracked_artifacts``; bundled skills live under the runtime package and are
    included here.
    """

    if not path.parts:
        return False
    return path.parts[0].casefold() in {
        "agent_utilities",
        "deploy",
        "docker",
        "helm",
        "k8s",
    }


def _is_bundled_connector_profile(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return (
        parts[:4]
        == (
            "agent_utilities",
            "protocols",
            "source_connectors",
            "profiles",
        )
        and path.suffix.casefold() in {".py", ".json", ".yaml", ".yml"}
    )


def _author_metadata_lines(path: Path, lines: list[str]) -> list[int]:
    """Return non-neutral package-author lines without returning their values."""
    if path.suffix.casefold() != ".toml":
        return []
    violations: list[int] = []
    in_project_authors = False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "[[project.authors]]":
            in_project_authors = True
            continue
        if in_project_authors and stripped.startswith("["):
            in_project_authors = False
        folded = stripped.casefold()
        if re.match(r"authors\s*=", folded):
            if (
                _NEUTRAL_AUTHOR_NAME not in folded
                or _NEUTRAL_AUTHOR_EMAIL_SUFFIX not in folded
            ):
                violations.append(number)
        elif in_project_authors and re.match(r"name\s*=", folded):
            if _NEUTRAL_AUTHOR_NAME not in folded:
                violations.append(number)
        elif in_project_authors and re.match(r"email\s*=", folded):
            if _NEUTRAL_AUTHOR_EMAIL_SUFFIX not in folded:
                violations.append(number)
    return violations


def _next_ordinal(counts: dict[tuple[str, str, str], int], group: tuple[str, str, str]) -> int:
    ordinal = counts.get(group, 0)
    counts[group] = ordinal + 1
    return ordinal


def scan(root: Path = ROOT) -> list[Violation]:
    identifiers = derive_local_identifiers(root)
    violations: list[Violation] = []
    # (path, category, content_hash) -> count so far, i.e. an ordinal
    # disambiguating two genuinely-identical lines in the same file/category —
    # never the line number, which drifts under unrelated edits and would
    # otherwise report a moved (not new) leak as a phantom NEW finding.
    ordinals: dict[tuple[str, str, str], int] = {}
    for path in _tracked_artifacts(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rel_str = relative.as_posix()
        deployment_doc = _is_deployment_doc(relative)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number in _author_metadata_lines(path, lines):
            category = "non-neutral package author identity"
            content_hash = _content_hash(lines[number - 1])
            ordinal = _next_ordinal(ordinals, (rel_str, category, content_hash))
            violations.append(
                Violation(rel_str, number, category, content_hash, ordinal)
            )
        for number, line in enumerate(lines, 1):
            for category in classify_line(
                line,
                identifiers=identifiers,
                deployment_doc=deployment_doc,
            ):
                content_hash = _content_hash(line)
                ordinal = _next_ordinal(ordinals, (rel_str, category, content_hash))
                violations.append(
                    Violation(rel_str, number, category, content_hash, ordinal)
                )
    for path in _runtime_source_artifacts(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rel_str = relative.as_posix()
        if _is_bundled_connector_profile(relative):
            category = "bundled environment-specific connector profile"
            # Whole-file finding, not line-anchored: hash the path itself so
            # it stays stable regardless of the file's own line churn.
            content_hash = _content_hash(rel_str)
            ordinal = _next_ordinal(ordinals, (rel_str, category, content_hash))
            violations.append(
                Violation(rel_str, 1, category, content_hash, ordinal)
            )
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            for category in classify_runtime_source_line(
                line, identifiers=identifiers
            ):
                content_hash = _content_hash(line)
                ordinal = _next_ordinal(ordinals, (rel_str, category, content_hash))
                violations.append(
                    Violation(rel_str, number, category, content_hash, ordinal)
                )
    return violations


BASELINE = Path(__file__).resolve().parent / "tracked_privacy_baseline.txt"

_BASELINE_SEP = "\t"


# BaselineKey: (path, category, content_hash, ordinal) — stable under pure
# line motion elsewhere in the file. ``content_hash`` fingerprints the
# offending line itself (irreversibly — see ``_content_hash``), so a leak
# that merely shifted line number because unrelated code was inserted above
# it keys identically before and after the shift. ``ordinal`` only
# disambiguates two genuinely-identical lines in the same file/category.
# Same scheme as ``check_swallowed_errors.py``'s ``HandlerKey`` (D-SWG-1) and
# ``check_wiring.py``'s ``_finding_key`` (D-OP-11) — canonical across every
# gate baseline in this repo, not a bespoke third scheme.
BaselineKey = tuple[str, str, str, int]


def _baseline_key(violation: Violation) -> BaselineKey:
    return (violation.path, violation.category, violation.content_hash, violation.ordinal)


def _load_baseline() -> set[BaselineKey]:
    """Pre-existing leaks this gate newly *sees* but did not previously report.

    A missing baseline file is an EMPTY baseline — stricter, never laxer — so a
    lost or renamed path can only turn known debt back into hard failures, it
    can never silently excuse a real leak.
    """
    if not BASELINE.exists():
        return set()
    out: set[BaselineKey] = set()
    for raw in BASELINE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(_BASELINE_SEP)
        if len(fields) < 5 or not fields[4].isdigit():
            continue
        # fields: path, line(informational), category, content_hash, ordinal
        out.add((fields[0], fields[2], fields[3], int(fields[4])))
    return out


def _write_baseline(violations: list[Violation]) -> None:
    lines = [
        f"{v.path}{_BASELINE_SEP}{v.line}{_BASELINE_SEP}{v.category}"
        f"{_BASELINE_SEP}{v.content_hash}{_BASELINE_SEP}{v.ordinal}"
        for v in sorted(
            violations,
            key=lambda v: (v.path, v.category, v.content_hash, v.ordinal),
        )
    ]
    BASELINE.write_text(
        "# Frozen baseline of pre-existing privacy leaks that D-CIP-10's scope\n"
        "# fix made VISIBLE for the first time (a ratchet — burn down toward\n"
        "# zero, same convention as scripts/security/cypher_write_subset_baseline.txt).\n"
        "#\n"
        "# Until D-CIP-10, check_tracked_privacy had two passes with a hole\n"
        "# between them: the whole-tree pass was location-filtered to docs/,\n"
        "# .github/, top-level and *.toml, and the unrestricted pass was scoped\n"
        "# to `git diff`. Anything committed earlier under docker/ or\n"
        "# agent_utilities/ was scanned by neither, so these leaks were public\n"
        "# and invisible. They are NOT new — only newly seen.\n"
        "#\n"
        "# These entries are tracked debt, not exemptions. Two are the personal\n"
        "# account name in live kaniko Job manifests (D-PCC-1, owned by\n"
        "# lane-release-0801); the rest are internal endpoints and shared-account\n"
        "# paths. GitHub is public-facing and gets STRICT standards, so this file\n"
        "# must reach zero before the repository is pushed.\n"
        "#\n"
        "# TAB-separated: path\\tline\\tcategory\\tcontent_hash\\tordinal. The KEY is\n"
        "# (path, category, content_hash, ordinal) -- content_hash is an\n"
        "# irreversible fingerprint of the offending line, so the entry is stable\n"
        "# under pure line motion elsewhere in the file (D-W2P: this file used to\n"
        "# key on line number alone and reported moved-not-new leaks as phantom\n"
        "# NEW findings). `line` is informational only, for a human to locate the\n"
        "# site; it is never compared. A NEW leak (not listed here) FAILS the\n"
        "# gate. Fixing a listed site shrinks this file on the next\n"
        "# --update-baseline; never hand-add an entry to make a real leak\n"
        "# disappear.\n"
        + "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="check-tracked-privacy")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="re-anchor the frozen baseline to the current violations",
    )
    args = parser.parse_args()

    violations = scan()
    if args.update_baseline:
        _write_baseline(violations)
        print(f"Wrote {BASELINE.name}: {len(violations)} baselined leak(s).")
        return 0

    baseline = _load_baseline()
    new = [v for v in violations if _baseline_key(v) not in baseline]
    current = {_baseline_key(v) for v in violations}
    fixed = len(baseline - current)

    if new:
        print("Tracked artifact privacy gate FAILED:")
        for violation in new:
            print(f"  - {violation.render()}")
        print("Matched values are intentionally suppressed.")
        print(
            f"{len(new)} NEW leak(s) not in {BASELINE.name}. "
            f"({len(baseline)} pre-existing leak(s) remain baselined — "
            "burn them down, never grow the file.)"
        )
        return 1
    if fixed:
        print(
            f"Tracked artifact privacy gate passed. {fixed} baselined leak(s) "
            f"are now fixed — re-run with --update-baseline to shrink "
            f"{BASELINE.name}."
        )
        return 0
    print("Tracked artifact privacy gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
