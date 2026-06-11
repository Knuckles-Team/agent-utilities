#!/usr/bin/python
from __future__ import annotations

"""Proposal → concrete change-set synthesis (CONCEPT:AHE-3.21).

The evolution→branch bridge's first half: given a *promoted* golden-loop
proposal (a ``SpecDraft``/``TeamSpec``/dict spec or the raw KG proposal node's
properties), materialize a concrete, reviewable :class:`ChangeSet`.

This module deliberately contains **no LLM calls** — generation happens in the
golden loop's synthesize/distill stages (KG-2.7/KG-2.10). Here we only
materialize what the proposal already carries:

* **Code-bearing proposals** — a proposal that embeds explicit file-level
  artifacts (a ``files`` attribute/key, or a ``files_json`` property on the KG
  node: ``[{"path": ..., "content": ...}, ...]``) becomes a ``kind="code"``
  change set. Its Python files are validated through the tiered RLM sandbox
  (CONCEPT:ORCH-1.38) — syntax compile + best-effort import — before any
  publisher may turn them into a branch. Proposal-named tests (``tests`` /
  ``tests_json``) ride along on the change set; they are executed later, in the
  publisher's worktree, where the full repository context exists (a snippet
  sandbox cannot run repo-relative pytest — an honest v1 limit, see
  ``docs/guides/autonomous-evolution.md``).
* **Prose-only proposals** — most SpecDrafts/TeamSpecs carry only prose
  (title/problem/approach or name/goal/description). For those the reviewable
  artifact IS an SDD plan skeleton: ``.specify/specs/<topic>/spec.md`` +
  ``tasks.md`` following the repository's ``.specify/specs/_template.md``
  conventions (``kind="sdd_plan"``).

Paths are repo-relative and sanitized — an embedded artifact can never escape
the publisher's worktree.
"""

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Upper bound on embedded files per proposal — a proposal is a focused change,
#: not a repository import.
MAX_EMBEDDED_FILES = 50

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


# ── data model ───────────────────────────────────────────────────────


@dataclass
class FileChange:
    """One repo-relative file the change set writes (full new content)."""

    path: str
    content: str


@dataclass
class SandboxCheck:
    """One named validation outcome inside a :class:`ValidationReport`."""

    name: str
    passed: bool
    reason: str = ""


@dataclass
class ValidationReport:
    """The RLM-sandbox validation verdict for a code-bearing change set."""

    ok: bool
    backend: str = ""
    checks: list[SandboxCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeSet:
    """A concrete, publishable change derived from one proposal.

    ``kind`` is ``"code"`` (the proposal embedded explicit file artifacts) or
    ``"sdd_plan"`` (prose proposal → SDD plan skeleton). ``tests`` are
    proposal-named pytest targets the publisher runs in its worktree.
    """

    proposal_id: str
    title: str
    kind: str
    files: list[FileChange] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    summary: str = ""
    validation: ValidationReport | None = None

    @property
    def publishable(self) -> bool:
        """A change set may be published unless sandbox validation failed."""
        return bool(self.files) and (self.validation is None or self.validation.ok)


# ── proposal field extraction ────────────────────────────────────────


def _proposal_field(proposal: Any, *names: str) -> Any:
    """First non-empty attribute/key among ``names`` on the proposal."""
    for name in names:
        val = getattr(proposal, name, None)
        if val is None and isinstance(proposal, dict):
            val = proposal.get(name)
        if val:
            return val
    return None


def proposal_id_of(proposal: Any) -> str:
    """The proposal's stable id (shared with the auto-merge audit trail)."""
    from .auto_merge import GovernedAutoMerger

    return GovernedAutoMerger._spec_id(proposal)


def _safe_rel_path(raw: Any) -> str | None:
    """Sanitize an embedded artifact path to a safe repo-relative POSIX path."""
    path = str(raw or "").strip().replace("\\", "/")
    if not path or path.startswith("/") or ":" in path.split("/", 1)[0]:
        return None
    segments = [s for s in path.split("/") if s and s != "."]
    if not segments or any(s == ".." or not _SAFE_SEGMENT.match(s) for s in segments):
        return None
    return "/".join(segments)


def _json_list(value: Any) -> list[Any]:
    """Coerce a list field that may arrive JSON-encoded (KG node property)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    return value if isinstance(value, list) else []


def extract_embedded_files(proposal: Any) -> list[FileChange]:
    """Explicit file-level artifacts the proposal carries (sanitized).

    Sources, in order: a ``files`` attribute/key (list of
    ``{"path", "content"}``), or a ``files_json`` property (the KG-node
    serialized form). Unsafe paths are dropped with a warning.
    """
    raw = _proposal_field(proposal, "files") or _proposal_field(proposal, "files_json")
    out: list[FileChange] = []
    for item in _json_list(raw)[:MAX_EMBEDDED_FILES]:
        if not isinstance(item, dict):
            continue
        path = _safe_rel_path(item.get("path"))
        if path is None:
            logger.warning(
                "change_synthesis: dropping unsafe embedded path %r",
                item.get("path"),
            )
            continue
        out.append(FileChange(path=path, content=str(item.get("content", ""))))
    return out


def extract_named_tests(proposal: Any) -> list[str]:
    """Proposal-named pytest targets (``tests`` / ``tests_json``), sanitized.

    Targets may carry a ``::`` node selector (``tests/test_x.py::TestY``); the
    file part must still be a safe repo-relative path.
    """
    raw = _proposal_field(proposal, "tests") or _proposal_field(proposal, "tests_json")
    out: list[str] = []
    for item in _json_list(raw):
        target = str(item or "").strip()
        file_part, _, selector = target.partition("::")
        path = _safe_rel_path(file_part)
        if path is None:
            continue
        out.append(f"{path}::{selector}" if selector else path)
    return out


def _concept_ids(proposal: Any) -> list[str]:
    raw = _proposal_field(proposal, "concept_ids") or _proposal_field(
        proposal, "concept_ids_json"
    )
    return [str(c) for c in _json_list(raw) if c]


# ── RLM-sandbox validation (code-bearing change sets) ────────────────


def _validation_program(py_files: list[FileChange]) -> str:
    """Build the self-contained validation program run inside a sandbox.

    The program embeds the changed Python sources as literals, syntax-compiles
    each, then materializes them into a temp dir and import-smoke-tests each
    module. A ``ModuleNotFoundError`` for a module *outside* the change set is
    non-fatal (repo context lives in the publisher's worktree, not the
    sandbox); a missing module *inside* the change set, a syntax error, or any
    other import-time crash fails validation. The verdict is printed as the
    final JSON line (parsed from stdout — uniform across sandbox tiers).
    """
    files_literal = json.dumps({f.path: f.content for f in py_files})
    return f"""
import importlib, json, os, sys, tempfile

_files = json.loads({files_literal!r})
_checks = []
_ok = True

for _path, _src in sorted(_files.items()):
    try:
        compile(_src, _path, "exec")
        _checks.append({{"name": "syntax:" + _path, "passed": True, "reason": "compiles"}})
    except SyntaxError as _e:
        _ok = False
        _checks.append({{"name": "syntax:" + _path, "passed": False, "reason": str(_e)}})

if _ok:
    _root = tempfile.mkdtemp(prefix="evolution-validate-")
    for _path, _src in _files.items():
        _dest = os.path.join(_root, *_path.split("/"))
        os.makedirs(os.path.dirname(_dest), exist_ok=True)
        with open(_dest, "w", encoding="utf-8") as _fh:
            _fh.write(_src)
    sys.path.insert(0, _root)
    _tops = {{_p.split("/")[0].removesuffix(".py") for _p in _files}}
    for _path in sorted(_files):
        _mod = _path.removesuffix(".py").replace("/", ".")
        _name = "import:" + _path
        try:
            importlib.import_module(_mod)
            _checks.append({{"name": _name, "passed": True, "reason": "imports"}})
        except ModuleNotFoundError as _e:
            _missing = (_e.name or "").split(".")[0]
            if _missing in _tops:
                _ok = False
                _checks.append({{"name": _name, "passed": False, "reason": str(_e)}})
            else:
                _checks.append({{
                    "name": _name, "passed": True,
                    "reason": "external dependency %r deferred to worktree tests" % _missing,
                }})
        except Exception as _e:
            _ok = False
            _checks.append({{"name": _name, "passed": False, "reason": repr(_e)}})

print(json.dumps({{"ok": _ok, "checks": _checks}}))
"""


def _run_in_sandbox(program: str) -> tuple[str, str]:
    """Execute ``program`` through the tiered sandbox chain; return (backend, stdout).

    Mirrors the RLM REPL's escalate-on-reject loop (CONCEPT:ORCH-1.38): the
    router picks the cheapest capable tier; a :class:`SandboxRejected` escalates
    to the next; the ``local`` floor anchors the chain.
    """
    from agent_utilities.rlm.sandboxes.base import SandboxEnv, SandboxRejected
    from agent_utilities.rlm.sandboxes.registry import default_sandboxes
    from agent_utilities.rlm.sandboxes.router import SandboxRouter

    async def _run() -> tuple[str, str]:
        chain = SandboxRouter(default_sandboxes()).select(program)
        last_error = ""
        for backend in chain:
            try:
                result = await backend.execute(program, SandboxEnv(vars={}))
            except SandboxRejected as rej:
                last_error = rej.reason
                continue
            if result.error:
                last_error = result.error
                continue
            return backend.name, result.stdout
        return "none", json.dumps(
            {
                "ok": False,
                "checks": [
                    {
                        "name": "sandbox",
                        "passed": False,
                        "reason": last_error or "no backend accepted the program",
                    }
                ],
            }
        )

    return asyncio.run(_run())


def validate_in_sandbox(files: list[FileChange]) -> ValidationReport:
    """Validate a change set's Python files in the tiered RLM sandbox.

    Non-Python change sets (docs, specs, configs) conform vacuously — there is
    nothing to compile. Any sandbox-infrastructure error fails closed.
    """
    py_files = [f for f in files if f.path.endswith(".py")]
    if not py_files:
        return ValidationReport(
            ok=True,
            backend="none",
            checks=[SandboxCheck("no_python", True, "no Python files to validate")],
        )
    try:
        backend, stdout = _run_in_sandbox(_validation_program(py_files))
        verdict = json.loads(stdout.strip().splitlines()[-1])
        return ValidationReport(
            ok=bool(verdict.get("ok")),
            backend=backend,
            checks=[
                SandboxCheck(
                    name=str(c.get("name", "?")),
                    passed=bool(c.get("passed")),
                    reason=str(c.get("reason", "")),
                )
                for c in verdict.get("checks", [])
                if isinstance(c, dict)
            ],
        )
    except Exception as exc:  # noqa: BLE001 — cannot prove validity ⇒ fail closed
        logger.warning("change_synthesis: sandbox validation error: %s", exc)
        return ValidationReport(
            ok=False,
            backend="error",
            checks=[SandboxCheck("sandbox", False, f"validation error: {exc}")],
        )


# ── SDD plan skeleton (prose proposals) ──────────────────────────────


def _sdd_spec_markdown(
    title: str,
    proposal_id: str,
    problem: str,
    approach: str,
    value: str,
    concept_ids: list[str],
) -> str:
    concepts = ", ".join(concept_ids) or "n/a"
    return (
        f"# Spec: {title}\n\n"
        f"> Materialized by the evolution→branch bridge (CONCEPT:AHE-3.21) from "
        f"promoted proposal `{proposal_id}`. Concepts: {concepts}\n\n"
        "## Pre-Flight Checklist (Mandatory — DSTDD)\n\n"
        "- [ ] **KG search completed** — design doc exists\n"
        "- [ ] **Extension point identified** — or New Concept Proposal approved\n"
        "- [ ] **C4 diagram created** — showing integration into pillar topology\n"
        "- [ ] **No new CONCEPT: tag** without pillar reference\n"
        "- [ ] **Design validation passes**\n\n"
        f"## Problem\n\n{problem or 'TBD — refine from the proposal.'}\n\n"
        f"## Approach\n\n{approach or 'TBD — refine from the proposal.'}\n\n"
        f"## Value\n\n{value or 'TBD — refine from the proposal.'}\n\n"
        "## Non-Functional Requirements\n\n"
        "- [ ] All existing tests continue to pass (zero regression)\n"
        "- [ ] Pre-commit hooks pass cleanly\n"
        "- [ ] Documentation updated if pillar topology changes\n"
    )


def _sdd_tasks_markdown(title: str, proposal_id: str) -> str:
    return (
        f"# Tasks: {title}\n\n"
        f"> Skeleton for proposal `{proposal_id}` — break the approach into "
        "verifiable, test-first tasks before implementation.\n\n"
        "- [ ] T1: Write the failing test that captures the acceptance criteria\n"
        "- [ ] T2: Implement the minimal change that makes it pass\n"
        "- [ ] T3: Wire the change into its extension point + docs\n"
        "- [ ] T4: Run the full suite + pre-commit; fix all findings\n"
    )


# ── the synthesis entry point ────────────────────────────────────────


def synthesize_change_set(proposal: Any, *, validate: bool = True) -> ChangeSet:
    """Materialize a promoted proposal into a concrete :class:`ChangeSet`.

    Code-bearing proposals (embedded ``files``) become ``kind="code"`` change
    sets, sandbox-validated when ``validate`` (the default). Prose proposals
    become a ``kind="sdd_plan"`` SDD skeleton under ``.specify/specs/<topic>/``
    — that skeleton IS the reviewable artifact for a prose proposal.
    """
    from ..enrichment.extractors.document import slug

    proposal_id = proposal_id_of(proposal)
    title = str(_proposal_field(proposal, "title", "name") or proposal_id).strip()
    summary = str(
        _proposal_field(proposal, "summary", "goal", "problem", "description") or ""
    ).strip()
    concept_ids = _concept_ids(proposal)

    files = extract_embedded_files(proposal)
    if files:
        change = ChangeSet(
            proposal_id=proposal_id,
            title=title,
            kind="code",
            files=files,
            tests=extract_named_tests(proposal),
            concept_ids=concept_ids,
            summary=summary,
        )
        if validate:
            change.validation = validate_in_sandbox(files)
        return change

    # Prose proposal → the SDD plan skeleton is the change.
    topic = slug(title) or "proposal"
    base = f".specify/specs/{topic}"
    problem = str(_proposal_field(proposal, "problem", "goal") or "").strip()
    approach = str(_proposal_field(proposal, "approach", "description") or "").strip()
    value = str(_proposal_field(proposal, "value") or "").strip()
    return ChangeSet(
        proposal_id=proposal_id,
        title=title,
        kind="sdd_plan",
        files=[
            FileChange(
                path=f"{base}/spec.md",
                content=_sdd_spec_markdown(
                    title, proposal_id, problem, approach, value, concept_ids
                ),
            ),
            FileChange(
                path=f"{base}/tasks.md",
                content=_sdd_tasks_markdown(title, proposal_id),
            ),
        ],
        concept_ids=concept_ids,
        summary=summary,
    )
