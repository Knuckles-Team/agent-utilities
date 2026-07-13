#!/usr/bin/env python3
"""Deterministic structural + bug-class scanner for the `ecosystem-validation` skill.

Runs the parts of validation that need no LLM: resolve a target (a fleet agent
package under `agent-packages/agents/*`, the `agent-utilities` package itself,
or a universal-skill / agent-utilities skill directory), then run

- ``structural``: pytest (packages, via the tier4_fleet_regression.sh pattern) or
  frontmatter/reference-file checks (skills).
- ``bug_hunt``: regex scan for the known fleet bug-classes (see BUG_CLASSES below).

The ``delegation`` (grounded execute_agent probe) and ``evolution`` (persisting
:ValidationFinding nodes via graph_write/graph_feedback) phases are NOT done by
this script — they need graph-os/an LLM and are performed by the calling agent
per SKILL.md. This script only emits the deterministic, script-checkable half of
the report as JSON so the delegated phases have something concrete to attach to.

Usage:
    python validate_target.py --target agents/gitlab-api --mode full
    python validate_target.py --target agent-utilities --mode structural
    python validate_target.py --target kg-write --mode bug_hunt
    python validate_target.py --target ecosystem-standardizer --mode full --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except (
    ImportError
):  # pragma: no cover - PyYAML is a transitive dep everywhere in this repo
    yaml = None  # type: ignore[assignment]

WORKSPACE_ROOT = Path("/home/apps/workspace/agent-packages")
AGENTS_ROOT = WORKSPACE_ROOT / "agents"
AU_ROOT = WORKSPACE_ROOT / "agent-utilities"
AU_SKILLS_ROOT = AU_ROOT / "agent_utilities" / "skills"
UNIVERSAL_SKILLS_ROOT = (
    WORKSPACE_ROOT / "skills" / "universal-skills" / "universal_skills"
)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}
MULTIPLEXER_PREFIXES = ("gith__", "go__", "cm__", "tm__")


@dataclass
class Finding:
    category: str
    severity: str  # low | medium | high | critical
    file: str
    detail: str
    line: int | None = None


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    target: str
    resolved_path: str
    kind: str  # package | skill | unresolved
    mode: str
    checks: list[CheckResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall_pass"] = (
            all(c["passed"] for c in d["checks"]) if d["checks"] else None
        )
        return d


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #


def resolve_target(target: str) -> tuple[Path | None, str]:
    """Return (path, kind) where kind is 'package', 'skill', or 'unresolved'."""
    t = target.strip()

    # literal agent-utilities
    if t.lower() in {"agent-utilities", "agent_utilities"}:
        return AU_ROOT, "package"

    # an absolute/relative path that exists
    p = Path(t)
    if not p.is_absolute():
        candidates = [WORKSPACE_ROOT / t, AGENTS_ROOT / t]
    else:
        candidates = [p]
    for c in candidates:
        if c.is_dir():
            return c, (
                "package"
                if (c / "pyproject.toml").exists()
                else "skill"
                if (c / "SKILL.md").exists()
                else "package"
            )

    # a bare fleet-agent package name
    fleet_candidate = AGENTS_ROOT / t
    if fleet_candidate.is_dir():
        return fleet_candidate, "package"

    # an agent-utilities own kg-* skill or universal-skill name — search by dir basename
    for root in (AU_SKILLS_ROOT, UNIVERSAL_SKILLS_ROOT):
        if not root.is_dir():
            continue
        for skill_dir in root.rglob(t):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                return skill_dir, "skill"

    return None, "unresolved"


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


def _load_frontmatter(skill_md: Path) -> tuple[dict, str]:
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    fm_text, body = m.group(1), m.group(2)
    if yaml is not None:
        try:
            fm = yaml.safe_load(fm_text) or {}
        except Exception:
            fm = {}
    else:  # degraded regex fallback — enough to check name/skill_type/description presence
        fm = {}
        for key in ("name", "skill_type", "description", "domain"):
            mm = re.search(rf"^{key}:\s*(.+)$", fm_text, re.MULTILINE)
            if mm:
                fm[key] = mm.group(1).strip().strip(">-").strip()
    return fm, body


def structural_check_skill(skill_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [CheckResult("skill_md_exists", False, f"no SKILL.md in {skill_dir}")]
    checks.append(CheckResult("skill_md_exists", True, str(skill_md)))

    fm, body = _load_frontmatter(skill_md)
    checks.append(
        CheckResult(
            "frontmatter_parses",
            bool(fm),
            "frontmatter parsed" if fm else "frontmatter missing/unparsable",
        )
    )

    name_ok = fm.get("name") == skill_dir.name
    checks.append(
        CheckResult(
            "name_matches_dir",
            name_ok,
            f"name={fm.get('name')!r} dir={skill_dir.name!r}",
        )
    )

    has_skill_type = "skill_type" in fm
    checks.append(
        CheckResult(
            "declares_skill_type",
            has_skill_type,
            f"skill_type={fm.get('skill_type')!r}",
        )
    )

    has_description = bool(fm.get("description"))
    checks.append(
        CheckResult(
            "has_description",
            has_description,
            "description present" if has_description else "missing description",
        )
    )

    # every referenced scripts/ or references/ file actually exists
    referenced = set(re.findall(r"\b((?:scripts|references)/[\w./-]+\.\w+)", body))
    missing = [r for r in referenced if not (skill_dir / r).exists()]
    checks.append(
        CheckResult(
            "referenced_files_exist",
            not missing,
            "all referenced files exist"
            if not missing
            else f"missing: {sorted(missing)}",
        )
    )

    # multiplexer-prefix naming drift without a dual-context note
    prefixed = [p for p in MULTIPLEXER_PREFIXES if p in body]
    if (
        prefixed
        and "dual-context" not in body.lower()
        and "dual context" not in body.lower()
    ):
        checks.append(
            CheckResult(
                "no_unexplained_multiplexer_prefix",
                False,
                f"found prefixes {prefixed} with no dual-context note",
            )
        )
    else:
        checks.append(
            CheckResult(
                "no_unexplained_multiplexer_prefix", True, "no unexplained prefix drift"
            )
        )

    return checks


def structural_check_package(pkg_dir: Path, timeout_s: int) -> list[CheckResult]:
    checks: list[CheckResult] = []
    test_files = [f for f in pkg_dir.rglob("test_*.py") if "__pycache__" not in f.parts]
    if not test_files:
        checks.append(CheckResult("has_tests", False, "no test_*.py files found"))
        return checks
    checks.append(CheckResult("has_tests", True, f"{len(test_files)} test files"))

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        passed = proc.returncode == 0
        checks.append(CheckResult("pytest", passed, tail or f"exit={proc.returncode}"))
    except subprocess.TimeoutExpired:
        checks.append(CheckResult("pytest", False, f"TIMEOUT after {timeout_s}s"))
    return checks


# --------------------------------------------------------------------------- #
# Bug-class hunt — the KNOWN fleet bug-classes we've hit
# --------------------------------------------------------------------------- #


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in _walk(root):
        for fn in filenames:
            if fn.endswith(suffixes):
                out.append(Path(dirpath) / fn)
    return out


def _walk(root: Path):
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        yield dirpath, dirnames, filenames


def _grep(files: list[Path], pattern: re.Pattern) -> list[tuple[Path, int, str]]:
    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((f, i, line.strip()[:200]))
    return hits


def bug_hunt_unbounded_object_set(root: Path) -> list[Finding]:
    """Unbounded object_set(of_type=...) / unbounded graph scans (OOM risk)."""
    py = _iter_files(root, (".py",))
    pattern = re.compile(r"object_set\s*\(\s*of_type\s*=")
    findings = []
    for f, ln, line in _grep(py, pattern):
        if not re.search(r"\b(limit|page_size|max_results|top_k)\s*=", line):
            findings.append(
                Finding(
                    "unbounded_object_set",
                    "high",
                    str(f.relative_to(root)),
                    "object_set(of_type=...) call with no visible limit/page_size — risk of unbounded graph scan (OOM)",
                    ln,
                )
            )
    return findings


def bug_hunt_fake_success_stub(root: Path) -> list[Finding]:
    """Fake-success stubs: returns a canned success string without doing real work."""
    py = _iter_files(root, (".py",))
    pattern = re.compile(
        r"return\s+.*(executed successfully|completed successfully|done successfully)",
        re.IGNORECASE,
    )
    findings = []
    for f, ln, line in _grep(py, pattern):
        findings.append(
            Finding(
                "fake_success_stub",
                "critical",
                str(f.relative_to(root)),
                f"canned success string returned — verify real work precedes it: {line}",
                ln,
            )
        )
    return findings


def bug_hunt_nl_query_event_loop(root: Path) -> list[Finding]:
    """nl_query event-loop-already-running fallback (asyncio.get_event_loop misuse)."""
    py = _iter_files(root, (".py",))
    pattern = re.compile(r"get_event_loop\(\)|already running")
    findings = []
    for f, ln, line in _grep(py, pattern):
        if (
            "nl_query" in f.name.lower()
            or "nl_query" in line.lower()
            or "asyncio" in line.lower()
        ):
            findings.append(
                Finding(
                    "nl_query_event_loop_fallback",
                    "medium",
                    str(f.relative_to(root)),
                    f"possible event-loop-already-running fallback pattern: {line}",
                    ln,
                )
            )
    return findings


def bug_hunt_multiplexer_prefix_drift(root: Path) -> list[Finding]:
    """Multiplexer-prefix naming drift in skills (gith__/go__/cm__/tm__ with no dual-context note)."""
    skill_mds = _iter_files(root, ("SKILL.md",))
    findings = []
    pattern = re.compile(r"\b(gith__|go__|cm__|tm__)")
    for f in skill_mds:
        text = f.read_text(encoding="utf-8", errors="replace")
        hits = set(pattern.findall(text))
        if (
            hits
            and "dual-context" not in text.lower()
            and "dual context" not in text.lower()
        ):
            findings.append(
                Finding(
                    "multiplexer_prefix_drift",
                    "low",
                    str(f.relative_to(root)),
                    f"multiplexer prefixes {sorted(hits)} referenced with no dual-context note",
                )
            )
    return findings


def bug_hunt_missing_skill_files(root: Path) -> list[Finding]:
    """Missing skill script/reference files referenced from a SKILL.md body."""
    skill_mds = _iter_files(root, ("SKILL.md",))
    findings = []
    ref_pattern = re.compile(r"\b((?:scripts|references)/[\w./-]+\.\w+)")
    for f in skill_mds:
        text = f.read_text(encoding="utf-8", errors="replace")
        for ref in set(ref_pattern.findall(text)):
            if not (f.parent / ref).exists():
                findings.append(
                    Finding(
                        "missing_skill_reference_file",
                        "medium",
                        str(f.relative_to(root)),
                        f"referenced but missing: {ref}",
                    )
                )
    return findings


def bug_hunt_stale_swarm_hostnames(root: Path) -> list[Finding]:
    """Stale swarm hostnames in env (`*_*:port` service-name references, not *.arpa/k8s Service DNS)."""
    files = _iter_files(
        root, (".env", ".env.example", "compose.yml", "compose.yaml", "mcp_config.json")
    )
    pattern = re.compile(r"://([a-z0-9][a-z0-9_-]*_[a-z0-9_-]+):(\d{2,5})")
    findings = []
    for f, ln, line in _grep(files, pattern):
        if ".arpa" in line or "localhost" in line:
            continue
        findings.append(
            Finding(
                "stale_swarm_hostname",
                "medium",
                str(f.relative_to(root)),
                f"looks like a Swarm-style service-name:port host (k8s Service DNS/.arpa expected instead): {line}",
                ln,
            )
        )
    return findings


def bug_hunt_aggressive_k8s_probes(root: Path) -> list[Finding]:
    """Aggressive `timeoutSeconds: 1` k8s probes."""
    yml = _iter_files(root, (".yaml", ".yml"))
    pattern = re.compile(r"timeoutSeconds:\s*1\b")
    findings = []
    for f, ln, line in _grep(yml, pattern):
        findings.append(
            Finding(
                "aggressive_k8s_probe_timeout",
                "medium",
                str(f.relative_to(root)),
                f"timeoutSeconds: 1 is too aggressive for a probe under load: {line}",
                ln,
            )
        )
    return findings


def bug_hunt_unpinned_ci_python(root: Path) -> list[Finding]:
    """Unpinned CI Python (allows 3.14+ → native-build breaks)."""
    workflows = (
        _iter_files(root / ".github" / "workflows", (".yml", ".yaml"))
        if (root / ".github" / "workflows").is_dir()
        else []
    )
    pattern = re.compile(r"python-version:\s*\[?['\"]?(3\.\d+|3\.x|3)['\"]?\]?")
    findings = []
    for f, ln, line in _grep(workflows, pattern):
        mm = pattern.search(line)
        version = mm.group(1) if mm else ""
        if version in {"3.x", "3"} or not re.search(r"3\.1[0-3]", version):
            findings.append(
                Finding(
                    "unpinned_ci_python",
                    "medium",
                    str(f.relative_to(root)),
                    f"CI python-version not pinned to a known-good 3.10-3.13 floor/ceiling: {line}",
                    ln,
                )
            )
    return findings


BUG_CLASSES = {
    "unbounded_object_set": bug_hunt_unbounded_object_set,
    "fake_success_stub": bug_hunt_fake_success_stub,
    "nl_query_event_loop_fallback": bug_hunt_nl_query_event_loop,
    "multiplexer_prefix_drift": bug_hunt_multiplexer_prefix_drift,
    "missing_skill_reference_file": bug_hunt_missing_skill_files,
    "stale_swarm_hostname": bug_hunt_stale_swarm_hostnames,
    "aggressive_k8s_probe_timeout": bug_hunt_aggressive_k8s_probes,
    "unpinned_ci_python": bug_hunt_unpinned_ci_python,
}


def run_bug_hunt(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for scanner in BUG_CLASSES.values():
        findings.extend(scanner(root))
    return findings


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--target",
        required=True,
        help="agents/<pkg>, 'agent-utilities', or a skill name",
    )
    ap.add_argument(
        "--mode", choices=["structural", "bug_hunt", "full"], default="full"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="pytest timeout in seconds (packages only)",
    )
    ap.add_argument(
        "--json", action="store_true", help="print JSON instead of a markdown summary"
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="write JSON report to this path"
    )
    args = ap.parse_args()

    path, kind = resolve_target(args.target)
    if path is None:
        report = Report(
            target=args.target, resolved_path="", kind="unresolved", mode=args.mode
        )
        report.checks.append(
            CheckResult(
                "resolve_target",
                False,
                "could not resolve target to a package or skill directory",
            )
        )
        _emit(report, args)
        return 1

    report = Report(
        target=args.target, resolved_path=str(path), kind=kind, mode=args.mode
    )

    if args.mode in ("structural", "full"):
        if kind == "package":
            report.checks.extend(structural_check_package(path, args.timeout))
        else:
            report.checks.extend(structural_check_skill(path))

    if args.mode in ("bug_hunt", "full"):
        report.findings.extend(run_bug_hunt(path))

    _emit(report, args)
    return (
        0
        if all(c.passed for c in report.checks)
        and not any(f.severity == "critical" for f in report.findings)
        else 1
    )


def _emit(report: Report, args: argparse.Namespace) -> None:
    data = report.to_dict()
    if args.output:
        args.output.write_text(json.dumps(data, indent=2))
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print(f"# ecosystem-validation report — {report.target}")
    print(
        f"\nresolved: `{report.resolved_path}` (kind={report.kind}, mode={report.mode})\n"
    )
    if report.checks:
        print("## Structural checks\n")
        print("| check | result | detail |")
        print("|---|---|---|")
        for c in report.checks:
            print(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    if report.mode in ("bug_hunt", "full"):
        print("\n## Bug-hunt findings\n")
        if not report.findings:
            print("None found.")
        else:
            print("| category | severity | file:line | detail |")
            print("|---|---|---|---|")
            for f in report.findings:
                loc = f"{f.file}:{f.line}" if f.line else f.file
                print(f"| {f.category} | {f.severity} | {loc} | {f.detail} |")


if __name__ == "__main__":
    sys.exit(main())
