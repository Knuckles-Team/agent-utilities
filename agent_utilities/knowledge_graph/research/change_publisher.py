#!/usr/bin/python
from __future__ import annotations

"""Change publication seam — proposals become reviewable git branches.

CONCEPT:AU-AHE.harness.evolution-branch-bridge — Evolution-to-branch bridge publishing promoted proposals as regression-gated reviewable local git branches via change synthesis, RLM-sandbox validation and a governed ChangePublisher seam

Second half of the evolution→branch bridge: a :class:`ChangePublisher` turns a
:class:`~agent_utilities.knowledge_graph.research.change_synthesis.ChangeSet`
into something a human can review. agent-utilities deliberately takes NO hard
dependency on the ecosystem's repository tooling (repository-manager's
``rm_git``/``rm_worktree`` MCP tools) — publication is a protocol, exactly like
the fleet-actuation seam (CONCEPT:AU-OS.config.desired-state-fleet-reconciler):

* :class:`ChangePublisher` — ``publish(change_set, metadata) -> PublishResult``.
* :class:`LocalBranchPublisher` — the DEFAULT: plain ``git`` subprocess (git is
  universally available). Creates a **fresh worktree** off the target repo's
  default branch under a configurable root (``EVOLUTION_WORKTREE_ROOT``, default
  ``data_dir()/evolution_worktrees``) — it never writes into the canonical
  checkout's working tree — applies the change set, commits with opaque
  attribution, optionally delegates bounded test targets to an injected
  governed sandbox runner, runs the injected regression gate, and records only
  opaque references. **No pushes, no merges to main**: the output is a local
  branch a human (or a deployment-wired publisher) takes forward.
* A deployment wires an MCP-backed publisher (e.g. over repository-manager) by
  registering its own implementation via :func:`set_change_publisher` — see
  ``docs/guides/autonomous-evolution.md`` for the wiring recipe.

The governed entry points (:func:`governed_publish`, :func:`publish_proposal`)
consult the operational :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
(CONCEPT:AU-OS.deployment.fleet-lifecycle-control) under the reserved ``merge_promotion`` action kind before any
publisher runs. The shipped policy makes that tier ``approval_required``, so by
default publication *queues an approval*; a granted approval (via
``POST /api/fleet/approvals/grant``) lets the one-shot
``graph_evolution(action="publish_proposal")`` action proceed.
"""

import json
import logging
import os
import re
import secrets
import subprocess  # nosec B404 — argv-only git calls, no shell
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_utilities.security.persistence_privacy import (
    persistence_reference,
    sanitize_for_persistence,
)

from .change_synthesis import ChangeSet, proposal_id_of, synthesize_change_set

logger = logging.getLogger(__name__)

AUDIT_PUBLISH = "loop_engine.publish_proposal"

#: Commit identity used when the host has no git identity configured.
_GIT_IDENTITY = ("governed-evolution", "governed-evolution@localhost")

_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_MAX_CHANGED_FILES = 256
_MAX_CHANGED_FILE_BYTES = 2 * 1024 * 1024
_MAX_CHANGESET_BYTES = 16 * 1024 * 1024


def _reference(kind: str, value: Any) -> str:
    return persistence_reference(kind, value, namespace="change-publisher")


def _safe_persistent_text(value: Any, *, limit: int = 4096) -> str:
    rendered = str(value or "")
    if "\x00" in rendered or len(rendered.encode("utf-8")) > limit:
        raise ValueError("publication text exceeds its safety boundary")
    clean, _report = sanitize_for_persistence(rendered)
    if clean != rendered:
        raise ValueError("publication text contains prohibited identifying material")
    return rendered


def _atomic_private_text(target: Path, content: str) -> None:
    payload = content.encode("utf-8")
    if len(payload) > _MAX_CHANGED_FILE_BYTES:
        raise ValueError("published file exceeds its size boundary")
    if target.is_symlink():
        raise PermissionError("symbolic-link publication target is not permitted")
    temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("publication write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── result + protocol ────────────────────────────────────────────────


@dataclass
class PublishResult:
    """What one publication produced: the reviewable branch + its verdicts."""

    ok: bool
    proposal_id: str = ""
    branch: str = ""
    commit_sha: str = ""
    repo_path: str = ""
    worktree_path: str = ""
    gate_result: str = "not_run"  # pass | hold | not_run
    tests_passed: bool | None = None  # None = no governed test execution
    test_report: dict[str, Any] | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a persistence-safe/public view; raw local paths never escape."""

        return {
            "ok": self.ok,
            "proposal_ref": _reference("proposal", self.proposal_id),
            "branch_ref": _reference("branch", self.branch) if self.branch else "",
            "commit_ref": (
                _reference("commit", self.commit_sha) if self.commit_sha else ""
            ),
            "repository_ref": (
                _reference("repository", self.repo_path) if self.repo_path else ""
            ),
            "worktree_ref": (
                _reference("worktree", self.worktree_path) if self.worktree_path else ""
            ),
            "gate_result": self.gate_result,
            "tests_passed": self.tests_passed,
            "test_report": self.test_report,
            "detail": self.detail,
        }


@runtime_checkable
class ChangePublisher(Protocol):
    """Anything that can turn a change set into a reviewable artifact."""

    name: str

    def publish(
        self, change_set: ChangeSet, metadata: dict[str, Any] | None = None
    ) -> PublishResult:
        """Publish ``change_set``; never raises — failures land in the result."""
        ...  # ABSTRACT-OK


# ── the default publisher: plain git, local branch only ─────────────


def default_target_repo() -> Path | None:
    """The agent-utilities checkout this package runs from (when it is one).

    Walks up from the installed package looking for a ``.git`` — an installed
    wheel has none, in which case the caller must supply ``repo_path``.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def resolve_worktree_root() -> Path:
    """The root all evolution worktrees live under (never a checkout's tree).

    ``EVOLUTION_WORKTREE_ROOT`` (typed ``AgentConfig.evolution_worktree_root``)
    when set; otherwise ``data_dir()/evolution_worktrees``.
    """
    try:
        from agent_utilities.core.config import AgentConfig

        configured = AgentConfig().evolution_worktree_root
        if configured:
            return Path(configured).expanduser()
    except Exception as exc:  # noqa: BLE001 — config must never block publication
        logger.debug(
            "Publication root configuration unavailable: error_type=%s",
            type(exc).__name__,
        )
    from agent_utilities.core.paths import data_dir

    return data_dir() / "evolution_worktrees"


class LocalBranchPublisher:
    """Publish a change set as a local git branch in a fresh worktree.

    Parameters
    ----------
    engine:
        Optional KG engine; when present the publication (branch, sha, gate
        verdict) is recorded as a ``ProposalPublication`` node linked
        ``PUBLISHED_AS`` from the proposal, and the proposal node is stamped.
    repo_path:
        Trusted target repository supplied by deployment configuration.
        Proposal metadata cannot select a repository. The default is the
        agent-utilities checkout itself when running from a checkout.
    worktree_root:
        Where fresh worktrees are created (default: :func:`resolve_worktree_root`).
    regression_check:
        Optional ``(spec) -> bool`` gate (e.g. the failure analyzer's
        ``make_regression_check`` product, CONCEPT:AU-AHE.harness.failure-evolution). ``True`` = no
        regression. The verdict is recorded; a ``hold`` does not delete the
        branch — the branch is the review artifact either way.
    test_runner:
        Optional governed sandbox runner. Host-side proposal-selected test
        execution is not supported.
    """

    name = "local_branch"

    def __init__(
        self,
        engine: Any = None,
        *,
        repo_path: str | Path | None = None,
        worktree_root: str | Path | None = None,
        regression_check: Any = None,
        run_tests: bool = False,
        test_runner: Any = None,
        test_timeout: float = 600.0,
    ) -> None:
        self.engine = engine
        self.repo_path = Path(repo_path) if repo_path else None
        self.worktree_root = Path(worktree_root) if worktree_root else None
        self.regression_check = regression_check
        self.run_tests = run_tests
        self.test_runner = test_runner
        self.test_timeout = max(1.0, min(float(test_timeout), 600.0))

    # ── git plumbing (argv-only) ────────────────────────────────────
    @staticmethod
    def _git(*args: str, cwd: Path | None = None) -> tuple[bool, str]:
        try:
            child_env = {
                key: value
                for key in (
                    "PATH",
                    "HOME",
                    "USERPROFILE",
                    "SYSTEMROOT",
                    "TMP",
                    "TEMP",
                    "LANG",
                    "LC_ALL",
                )
                if (value := os.environ.get(key))
            }
            child_env.update(
                {
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_EXTERNAL_DIFF": "",
                }
            )
            proc = subprocess.run(  # nosec B603 B607 — fixed binary, validated argv
                [
                    "git",
                    "-c",
                    f"core.hooksPath={os.devnull}",
                    "-c",
                    "core.fsmonitor=false",
                    *args,
                ],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=120.0,
                env=child_env,
                stdin=subprocess.DEVNULL,
            )
            out = (proc.stdout or proc.stderr or "")[:65536].strip()
            clean, report = sanitize_for_persistence(out)
            if report.changed:
                clean = ""
            return proc.returncode == 0, str(clean)
        except Exception as exc:  # noqa: BLE001 — publishers never raise
            return False, type(exc).__name__

    def _base_ref(self, repo: Path) -> str:
        for ref in ("main", "master"):
            ok, _ = self._git("rev-parse", "--verify", "--quiet", ref, cwd=repo)
            if ok:
                return ref
        return "HEAD"

    @staticmethod
    def _branch_name(change_set: ChangeSet) -> str:
        del change_set
        return f"evolution/proposal-{uuid.uuid4().hex}"

    @staticmethod
    def _repository(configured: str | Path | None) -> Path | None:
        if configured is None:
            return None
        raw = Path(configured).expanduser()
        if raw.is_symlink():
            return None
        try:
            resolved = raw.resolve(strict=True)
        except OSError:
            return None
        git_dir = resolved / ".git"
        if not resolved.is_dir() or git_dir.is_symlink() or not git_dir.is_dir():
            return None
        return resolved

    @staticmethod
    def _private_worktree_root(configured: Path, repo: Path) -> Path:
        # Prove lexical disjointness and reject pre-existing symlink components
        # before creating anything. A rejected configuration must not mutate the
        # filesystem or redirect worktrees through a parent symlink.
        candidate = Path(os.path.abspath(configured))
        for component in (candidate, *candidate.parents):
            if component.exists() and component.is_symlink():
                raise PermissionError("worktree root cannot traverse a symbolic link")
        for first, second in ((candidate, repo), (repo, candidate)):
            try:
                first.relative_to(second)
            except ValueError:
                continue
            raise PermissionError("worktree root and repository must be disjoint")
        candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(candidate, 0o700)
        root = candidate.resolve(strict=True)
        for first, second in ((root, repo), (repo, root)):
            try:
                first.relative_to(second)
            except ValueError:
                continue
            raise PermissionError("worktree root and repository must be disjoint")
        return root

    @staticmethod
    def _destination(worktree: Path, raw_path: str) -> Path:
        rendered = str(raw_path or "")
        if (
            not rendered
            or len(rendered.encode("utf-8")) > 1024
            or "\x00" in rendered
            or "\\" in rendered
        ):
            raise ValueError("publication path is malformed")
        relative = Path(rendered)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise PermissionError("publication path is not confined")
        target = worktree / relative
        cursor = worktree
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise PermissionError("publication path traverses a symbolic link")
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=True)
        resolved_parent.relative_to(worktree.resolve(strict=True))
        return resolved_parent / target.name

    # ── the publication ─────────────────────────────────────────────
    def publish(
        self, change_set: ChangeSet, metadata: dict[str, Any] | None = None
    ) -> PublishResult:
        metadata = dict(metadata or {})
        result = PublishResult(ok=False, proposal_id=change_set.proposal_id)

        if not change_set.publishable:
            result.detail = "change set is not publishable"
            self._record(change_set, result)
            return result

        configured = self.repo_path or default_target_repo()
        repo = self._repository(configured)
        if repo is None:
            result.detail = "configured publication repository is unavailable"
            self._record(change_set, result)
            return result
        result.repo_path = str(repo)

        try:
            root = self._private_worktree_root(
                (self.worktree_root or resolve_worktree_root()).expanduser(), repo
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Publication worktree root rejected: error_type=%s",
                type(exc).__name__,
            )
            result.detail = "configured publication worktree root is unavailable"
            self._record(change_set, result)
            return result
        branch = self._branch_name(change_set)
        worktree = root / branch.replace("/", "--")
        if worktree.exists() or worktree.is_symlink():
            result.detail = "publication worktree collision"
            self._record(change_set, result)
            return result
        base = self._base_ref(repo)

        ok, out = self._git(
            "worktree", "add", "-b", branch, str(worktree), base, cwd=repo
        )
        if not ok:
            result.detail = "git worktree creation failed"
            self._record(change_set, result)
            return result
        result.branch = branch
        result.worktree_path = str(worktree)

        try:
            if len(change_set.files) > _MAX_CHANGED_FILES:
                raise ValueError("change set has too many files")
            total_bytes = 0
            for change in change_set.files:
                content = _safe_persistent_text(
                    change.content, limit=_MAX_CHANGED_FILE_BYTES
                )
                total_bytes += len(content.encode("utf-8"))
                if total_bytes > _MAX_CHANGESET_BYTES:
                    raise ValueError("change set exceeds its aggregate size boundary")
                dest = self._destination(worktree, change.path)
                _atomic_private_text(dest, content)

            if self.run_tests and change_set.tests:
                result.test_report = self._run_targeted_tests(
                    worktree, change_set.tests
                )
                result.tests_passed = bool(result.test_report.get("passed"))

            result.gate_result = self._run_gate(metadata.get("proposal", change_set))

            ok, out = self._git("add", "-A", cwd=worktree)
            if ok:
                ok, out = self._git(
                    "-c",
                    f"user.name={_GIT_IDENTITY[0]}",
                    "-c",
                    f"user.email={_GIT_IDENTITY[1]}",
                    "commit",
                    "--no-verify",
                    "-m",
                    self._commit_message(change_set, result),
                    cwd=worktree,
                )
            if not ok:
                raise RuntimeError("git commit failed")
            ok, sha = self._git("rev-parse", "HEAD", cwd=worktree)
            result.commit_sha = sha if ok else ""
            result.ok = True
            result.detail = (
                "published a local review branch; no push or merge performed"
            )
        except Exception as exc:  # noqa: BLE001 — publishers never raise
            logger.warning("Publication rejected: error_type=%s", type(exc).__name__)
            result.detail = "publication rejected by the artifact safety boundary"
            self._git("worktree", "remove", "--force", str(worktree), cwd=repo)
            self._git("branch", "-D", branch, cwd=repo)
            result.branch = ""
            result.worktree_path = ""

        self._record(change_set, result)
        return result

    def _commit_message(self, change_set: ChangeSet, result: PublishResult) -> str:
        kind = _safe_persistent_text(change_set.kind, limit=64)
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", kind):
            kind = "proposal"
        tests_line = (
            "not named by the proposal"
            if result.tests_passed is None
            else ("pass" if result.tests_passed else "FAIL — review required")
        )
        return (
            f"evolution: governed {kind} proposal\n\n"
            "Materialized from an approved proposal.\n\n"
            f"Proposal reference: {_reference('proposal', change_set.proposal_id)}\n"
            "Concept: AU-AHE.harness.evolution-branch-bridge\n"
            f"Regression gate: {result.gate_result}; targeted tests: {tests_line}\n"
            "Generated by the evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge)."
        )

    def _run_targeted_tests(self, worktree: Path, tests: list[str]) -> dict[str, Any]:
        """Delegate bounded tests to an injected governed sandbox runner."""

        if self.test_runner is None:
            return {
                "passed": False,
                "status": "sandbox_runner_required",
                "target_count": min(len(tests), 256),
            }
        if len(tests) > 256 or any(
            not isinstance(target, str)
            or not target
            or len(target.encode("utf-8")) > 1024
            or target.startswith("-")
            or "\x00" in target
            for target in tests
        ):
            return {
                "passed": False,
                "status": "invalid_targets",
                "target_count": min(len(tests), 256),
            }
        try:
            raw = self.test_runner(
                worktree=worktree,
                targets=list(tests),
                timeout=self.test_timeout,
            )
            if isinstance(raw, dict):
                passed = bool(raw.get("passed"))
                return_code = raw.get("returncode")
            else:
                passed = bool(raw)
                return_code = None
            return {
                "passed": passed,
                "status": "completed",
                "returncode": return_code if isinstance(return_code, int) else None,
                "target_count": len(tests),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sandbox test runner failed: error_type=%s", type(exc).__name__
            )
            return {
                "passed": False,
                "status": "runner_failed",
                "error_type": type(exc).__name__,
                "target_count": len(tests),
            }

    def _run_gate(self, spec: Any) -> str:
        """Run the injected regression gate; verdicts: pass | hold | not_run."""
        if self.regression_check is None:
            return "not_run"
        try:
            return "pass" if bool(self.regression_check(spec)) else "hold"
        except Exception as exc:  # noqa: BLE001 — an erroring gate holds
            logger.debug("Regression gate failed: error_type=%s", type(exc).__name__)
            return "hold"

    def _record(self, change_set: ChangeSet, result: PublishResult) -> None:
        """Persist the publication on the graph (node + edge + proposal stamp)."""
        if self.engine is None:
            return
        publication_id = f"proposal_publication:{uuid.uuid4().hex}"
        try:
            kind = _safe_persistent_text(change_set.kind, limit=64)
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", kind):
                kind = "proposal"
            self.engine.add_node(
                publication_id,
                "ProposalPublication",
                properties={
                    "proposal_ref": _reference("proposal", change_set.proposal_id),
                    "kind": kind,
                    "ok": result.ok,
                    "branch_ref": _reference("branch", result.branch)
                    if result.branch
                    else "",
                    "commit_ref": _reference("commit", result.commit_sha)
                    if result.commit_sha
                    else "",
                    "repository_ref": _reference("repository", result.repo_path)
                    if result.repo_path
                    else "",
                    "worktree_ref": _reference("worktree", result.worktree_path)
                    if result.worktree_path
                    else "",
                    "gate_result": result.gate_result,
                    "tests_passed": (
                        "" if result.tests_passed is None else str(result.tests_passed)
                    ),
                    "detail": _safe_persistent_text(result.detail, limit=500),
                    "published_at": _now_iso(),
                },
            )
            link = getattr(self.engine, "link_nodes", None)
            if callable(link):
                link(change_set.proposal_id, publication_id, "PUBLISHED_AS")
        except Exception as exc:  # noqa: BLE001 — recording never blocks publication
            logger.debug("Publication record failed: error_type=%s", type(exc).__name__)
        # Stamp the proposal node itself so its branch/sha/verdict are one
        # node-read away (best-effort; the ProposalPublication node is durable).
        try:
            self.engine.backend.execute(
                "MATCH (p) WHERE p.id = $id SET p.publish_branch_ref = $branch, "
                "p.publish_commit_ref = $sha, p.publish_gate = $gate",
                {
                    "id": change_set.proposal_id,
                    "branch": _reference("branch", result.branch)
                    if result.branch
                    else "",
                    "sha": _reference("commit", result.commit_sha)
                    if result.commit_sha
                    else "",
                    "gate": result.gate_result,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Proposal stamp failed: error_type=%s", type(exc).__name__)


# ── registry (deployment injection point) ───────────────────────────

_PUBLISHER: ChangePublisher | None = None


def set_change_publisher(publisher: ChangePublisher | None) -> None:
    """Register the process-wide publisher (``None`` resets to the default)."""
    global _PUBLISHER
    _PUBLISHER = publisher


def get_change_publisher(
    engine: Any = None, *, regression_check: Any = None
) -> ChangePublisher:
    """Resolve the active publisher: injected > default :class:`LocalBranchPublisher`."""
    if _PUBLISHER is not None:
        return _PUBLISHER
    return LocalBranchPublisher(engine, regression_check=regression_check)


# ── governed entry points ────────────────────────────────────────────


def _find_granted_approval(engine: Any, proposal_id: str) -> str | None:
    """A human-granted ``merge_promotion`` approval for this proposal, if any."""
    if engine is None:
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (a:ActionApproval {status: 'approved'}) RETURN a LIMIT 100"
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Approval scan failed: error_type=%s", type(exc).__name__)
        return None
    for row in rows or []:
        props = row.get("a") if isinstance(row, dict) else None
        if (
            isinstance(props, dict)
            and props.get("kind") == "merge_promotion"
            and props.get("target") == proposal_id
            and props.get("id")
        ):
            return str(props["id"])
    return None


def _stamp_approval(engine: Any, approval_id: str, status: str) -> None:
    try:
        engine.backend.execute(
            "MATCH (a:ActionApproval {id: $id}) "
            "SET a.status = $status, a.executed_at = $ts",
            {"id": approval_id, "status": status, "ts": _now_iso()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Approval stamp failed: error_type=%s", type(exc).__name__)


def _record_execution(
    engine: Any,
    proposal_id: str,
    publisher_name: str,
    result: PublishResult,
    source: str,
) -> str | None:
    """Stamp the ``ActionExecution`` audit node (mirrors fleet actuation's)."""
    if engine is None:
        return None
    execution_id = f"action_execution:{uuid.uuid4().hex}"
    try:
        engine.add_node(
            execution_id,
            "ActionExecution",
            properties={
                "kind": "merge_promotion",
                "target_ref": _reference("proposal", proposal_id),
                "params_json": json.dumps(
                    {
                        "branch_ref": _reference("branch", result.branch)
                        if result.branch
                        else "",
                        "commit_ref": _reference("commit", result.commit_sha)
                        if result.commit_sha
                        else "",
                    }
                ),
                "source_ref": _reference("source", source),
                "actuator_ref": _reference("publisher", publisher_name),
                "ok": bool(result.ok),
                "dry_run": False,
                "detail": _safe_persistent_text(result.detail, limit=500),
                "executed_at": _now_iso(),
                "executed_unix": time.time(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Execution record failed: error_type=%s", type(exc).__name__)
        return None
    return execution_id


def _abandon_branch(repo_path: str, worktree_path: str, branch: str) -> None:
    """Remove a published worktree and delete its local branch (AHE-3.23 rollback).

    Used when the capability ratchet rejects a just-published change: the local
    branch was never pushed, so a clean removal fully undoes the publication.
    """
    if not repo_path:
        return
    repo = Path(repo_path)
    if repo.is_symlink() or not (repo / ".git").exists():
        return
    if branch and not _SAFE_REF.fullmatch(branch):
        return

    def _run(*args: str) -> None:
        try:
            subprocess.run(  # nosec B603 — fixed git args, no shell
                ["git", "-c", f"core.hooksPath={os.devnull}", *args],
                cwd=str(repo),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=60,
                env={
                    key: value
                    for key in ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT")
                    if (value := os.environ.get(key))
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.debug("Branch cleanup failed: error_type=%s", type(exc).__name__)

    if worktree_path:
        _run("worktree", "remove", "--force", worktree_path)
    if branch:
        _run("branch", "-D", branch)


def governed_publish(
    engine: Any,
    proposal: Any,
    *,
    publisher: ChangePublisher | None = None,
    regression_check: Any = None,
    source: str = "loop_engine",
    action_policy: Any = None,
    code_synthesizer: Any = None,
    capability_ratchet: Any = None,
) -> dict[str, Any]:
    """Publish a promoted proposal through the ActionPolicy gate (CONCEPT:AU-AHE.harness.evolution-branch-bridge).

    Decision order:

    1. A pre-granted ``merge_promotion`` approval for this proposal is consumed
       and publication proceeds.
    2. Otherwise the OS-5.24 :class:`ActionPolicy` decides. The shipped default
       tier is ``approval_required`` → an approval is queued and this returns
       ``{"status": "approval_queued", "approval_id": ...}`` — a human grants
       it (``POST /api/fleet/approvals/grant``) and triggers the one-shot
       ``publish_proposal`` action.
    3. Allowed (granted, or a deployment relaxed the tier): synthesize the
       change set, refuse sandbox-invalid code, publish via the resolved
       :class:`ChangePublisher`, and record an ``ActionExecution`` audit node.

    Never raises; every outcome is a JSON-able report.
    """
    proposal_id = proposal_id_of(proposal)
    report: dict[str, Any] = {
        "proposal_ref": _reference("proposal", proposal_id),
        "source_ref": _reference("source", source),
    }

    granted_id = _find_granted_approval(engine, proposal_id)
    if granted_id is None:
        try:
            from agent_utilities.orchestration.action_policy import (
                ActionRequest,
                get_action_policy,
            )

            policy = action_policy or get_action_policy(engine)
            decision = policy.decide(
                ActionRequest(
                    kind="merge_promotion",
                    target=proposal_id,
                    params={"proposal_id": proposal_id},
                    source=source,
                    reason="publish promoted evolution proposal as a reviewable branch",
                )
            )
        except Exception as exc:  # noqa: BLE001 — gate failure ⇒ fail closed
            logger.warning("Action policy failed: error_type=%s", type(exc).__name__)
            report["status"] = "denied"
            report["detail"] = "action policy unavailable; publication denied"
            return report
        report["decision"] = decision.decision
        report["approval_id"] = decision.approval_id
        if not decision.allowed:
            report["status"] = "approval_queued" if decision.approval_id else "denied"
            report["detail"] = (
                "approval is required" if decision.approval_id else "publication denied"
            )
            return report
    else:
        report["decision"] = "approved"
        report["approval_id"] = granted_id

    # CONCEPT:AU-AHE.harness.single-file-code-synthesis — autonomous code-synthesis stage. For an attributed
    # proposal that carries no embedded files, generate a single-file edit so the
    # promotion emits real code; an un-attributed proposal yields None and falls
    # through to the prose SDD skeleton exactly as before. Generation failure is
    # never fatal — the prose path still runs.
    extra_files = None
    try:
        from .code_synthesis import synthesize_code

        extra_files = synthesize_code(proposal, synthesizer=code_synthesizer)
        if extra_files:
            report["code_synthesis"] = {"file_count": len(extra_files)}
    except Exception as exc:  # noqa: BLE001 — generation failure ⇒ prose fallback
        logger.debug("Code synthesis skipped: error_type=%s", type(exc).__name__)

    try:
        change_set = synthesize_change_set(proposal, extra_files=extra_files)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Change synthesis failed: error_type=%s", type(exc).__name__)
        report["status"] = "synthesis_failed"
        report["detail"] = "change synthesis failed"
        return report
    change_kind = str(change_set.kind or "")
    report["change_kind"] = (
        change_kind
        if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", change_kind)
        else "proposal"
    )
    if change_set.validation is not None:
        checks = list(change_set.validation.checks or [])
        report["validation"] = {
            "passed": all(check.passed for check in checks),
            "check_count": len(checks),
            "failed_count": sum(not check.passed for check in checks),
        }
    if not change_set.publishable:
        report["status"] = "validation_failed"
        report["detail"] = "sandbox validation rejected the change set"
        if granted_id:
            _stamp_approval(engine, granted_id, "failed")
        return report

    pub = publisher or get_change_publisher(engine, regression_check=regression_check)
    result = pub.publish(change_set, {"proposal": proposal})
    report["publish"] = result.to_dict()
    report["status"] = "published" if result.ok else "publish_failed"

    # CONCEPT:AU-AHE.harness.capability-ratchet / AHE-3.24 — verified apply→verify→rollback + capability
    # ratchet. Re-measure the published worktree against the persisted baseline; a
    # measured regression (ManifestVerifier *_revert recommendation, or any tracked
    # capability dropping below baseline) abandons the branch rather than leaving it
    # for review. A worktree with no capability probes is "not measured" → no block.
    if result.ok and result.worktree_path:
        try:
            from .capability_ratchet import CapabilityRatchet

            ratchet = capability_ratchet or CapabilityRatchet(engine)
            verdict = ratchet.evaluate(
                result.worktree_path, change_set=change_set, proposal_id=proposal_id
            )
            report["capability_ratchet"] = {"passed": bool(verdict.passed)}
            if not verdict.passed:
                _abandon_branch(result.repo_path, result.worktree_path, result.branch)
                result.ok = False
                report["publish"] = result.to_dict()
                report["status"] = "reverted"
                report["detail"] = "capability regression detected; branch removed"
        except Exception as exc:  # noqa: BLE001 — ratchet failure must not wedge publish
            logger.debug(
                "Capability ratchet skipped: error_type=%s", type(exc).__name__
            )

    execution_id = _record_execution(
        engine, proposal_id, getattr(pub, "name", "?"), result, source
    )
    report["execution_ref"] = (
        _reference("action_execution", execution_id) if execution_id else None
    )
    if granted_id:
        _stamp_approval(engine, granted_id, "executed" if result.ok else "failed")
    _audit_publish(proposal_id, report)
    return report


def load_proposal(engine: Any, proposal_id: str) -> dict[str, Any] | None:
    """Load a proposal node's properties by id (the spec-shaped dict)."""
    if engine is None or not proposal_id:
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (n) WHERE n.id = $id RETURN n LIMIT 1", {"id": proposal_id}
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Proposal lookup failed: error_type=%s", type(exc).__name__)
        rows = []
    for row in rows or []:
        props = row.get("n") if isinstance(row, dict) else None
        if isinstance(props, dict):
            return {**props, "id": props.get("id") or proposal_id}
    return None


def publish_proposal(
    engine: Any,
    proposal_id: str,
    *,
    publisher: ChangePublisher | None = None,
) -> dict[str, Any]:
    """One-shot, human-triggered publication of a promoted proposal by node id.

    The ``graph_evolution(action="publish_proposal")`` / REST entry point:
    loads the proposal node, then runs :func:`governed_publish` — so the
    ActionPolicy gate still applies (a granted approval, or an explicitly
    relaxed ``merge_promotion`` tier, lets it proceed; otherwise it queues).
    Governance validation already happened at promotion time (AHE-3.20); this
    step only materializes + publishes what was promoted.
    """
    proposal = load_proposal(engine, proposal_id)
    if proposal is None:
        return {
            "proposal_ref": _reference("proposal", proposal_id),
            "status": "not_found",
            "detail": "proposal was not found",
        }
    return governed_publish(
        engine, proposal, publisher=publisher, source="publish_proposal"
    )


def _audit_publish(proposal_id: str, report: dict[str, Any]) -> None:
    """Mirror the publication outcome into the audit log (best-effort)."""
    try:
        from agent_utilities.observability.audit_logger import AuditLogger

        AuditLogger().log(
            actor="evolution_bridge",
            action=AUDIT_PUBLISH,
            resource_type="proposal",
            resource_id=_reference("proposal", proposal_id),
            details={
                "status": report.get("status"),
                "branch_ref": (report.get("publish") or {}).get("branch_ref", ""),
                "commit_ref": (report.get("publish") or {}).get("commit_ref", ""),
                "gate_result": (report.get("publish") or {}).get("gate_result", ""),
                "decision": report.get("decision", ""),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Publication audit failed: error_type=%s", type(exc).__name__)
