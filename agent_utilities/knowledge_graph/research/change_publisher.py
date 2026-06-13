#!/usr/bin/python
from __future__ import annotations

"""Change publication seam — proposals become reviewable git branches.

CONCEPT:AHE-3.21 — Evolution-to-branch bridge publishing promoted proposals as regression-gated reviewable local git branches via change synthesis, RLM-sandbox validation and a governed ChangePublisher seam

Second half of the evolution→branch bridge: a :class:`ChangePublisher` turns a
:class:`~agent_utilities.knowledge_graph.research.change_synthesis.ChangeSet`
into something a human can review. agent-utilities deliberately takes NO hard
dependency on the ecosystem's repository tooling (repository-manager's
``rm_git``/``rm_worktree`` MCP tools) — publication is a protocol, exactly like
the fleet-actuation seam (CONCEPT:OS-5.25):

* :class:`ChangePublisher` — ``publish(change_set, metadata) -> PublishResult``.
* :class:`LocalBranchPublisher` — the DEFAULT: plain ``git`` subprocess (git is
  universally available). Creates a **fresh worktree** off the target repo's
  default branch under a configurable root (``EVOLUTION_WORKTREE_ROOT``, default
  ``data_dir()/evolution_worktrees``) — it never writes into the canonical
  checkout's working tree — applies the change set, commits citing the proposal
  + concept ids, runs proposal-named tests and the injected regression gate,
  and records everything. **No pushes, no merges to main**: the output is a
  local branch a human (or a deployment-wired publisher) takes forward.
* A deployment wires an MCP-backed publisher (e.g. over repository-manager) by
  registering its own implementation via :func:`set_change_publisher` — see
  ``docs/guides/autonomous-evolution.md`` for the wiring recipe.

The governed entry points (:func:`governed_publish`, :func:`publish_proposal`)
consult the operational :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
(CONCEPT:OS-5.24) under the reserved ``merge_promotion`` action kind before any
publisher runs. The shipped policy makes that tier ``approval_required``, so by
default publication *queues an approval*; a granted approval (via
``POST /api/fleet/approvals/grant``) lets the one-shot
``graph_orchestrate(action="publish_proposal")`` action proceed.
"""

import json
import logging
import re
import subprocess  # nosec B404 — argv-only git calls, no shell
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .change_synthesis import ChangeSet, proposal_id_of, synthesize_change_set

logger = logging.getLogger(__name__)

AUDIT_PUBLISH = "golden_loop.publish_proposal"

#: Commit identity used when the host has no git identity configured.
_GIT_IDENTITY = ("evolution-bridge", "evolution-bridge@agent-utilities.local")

_BRANCH_SAFE = re.compile(r"[^A-Za-z0-9._/-]+")


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
    tests_passed: bool | None = None  # None = no proposal-named tests
    test_report: dict[str, Any] | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        logger.debug("change_publisher: config root unavailable: %s", exc)
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
        Target repository (a proposal's ``repo_path`` metadata overrides;
        default = the agent-utilities checkout itself).
    worktree_root:
        Where fresh worktrees are created (default: :func:`resolve_worktree_root`).
    regression_check:
        Optional ``(spec) -> bool`` gate (e.g. the failure analyzer's
        ``make_regression_check`` product, CONCEPT:AHE-3.18). ``True`` = no
        regression. The verdict is recorded; a ``hold`` does not delete the
        branch — the branch is the review artifact either way.
    run_tests:
        Run proposal-named tests in the worktree before committing the verdict
        (``python -m pytest <targets>``; bounded by ``test_timeout`` seconds).
    """

    name = "local_branch"

    def __init__(
        self,
        engine: Any = None,
        *,
        repo_path: str | Path | None = None,
        worktree_root: str | Path | None = None,
        regression_check: Any = None,
        run_tests: bool = True,
        test_timeout: float = 600.0,
    ) -> None:
        self.engine = engine
        self.repo_path = Path(repo_path) if repo_path else None
        self.worktree_root = Path(worktree_root) if worktree_root else None
        self.regression_check = regression_check
        self.run_tests = run_tests
        self.test_timeout = test_timeout

    # ── git plumbing (argv-only) ────────────────────────────────────
    @staticmethod
    def _git(*args: str, cwd: Path | None = None) -> tuple[bool, str]:
        try:
            proc = subprocess.run(  # nosec B603 B607 — fixed binary, validated argv
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            return proc.returncode == 0, out
        except Exception as exc:  # noqa: BLE001 — publishers never raise
            return False, str(exc)

    def _base_ref(self, repo: Path) -> str:
        for ref in ("main", "master"):
            ok, _ = self._git("rev-parse", "--verify", "--quiet", ref, cwd=repo)
            if ok:
                return ref
        return "HEAD"

    @staticmethod
    def _branch_name(change_set: ChangeSet) -> str:
        from ..enrichment.extractors.document import slug

        stem = slug(change_set.title)[:48] or "proposal"
        return _BRANCH_SAFE.sub("-", f"evolution/{stem}-{uuid.uuid4().hex[:8]}")

    # ── the publication ─────────────────────────────────────────────
    def publish(
        self, change_set: ChangeSet, metadata: dict[str, Any] | None = None
    ) -> PublishResult:
        metadata = dict(metadata or {})
        result = PublishResult(ok=False, proposal_id=change_set.proposal_id)

        if not change_set.publishable:
            failures = [
                f"{c.name}: {c.reason}"
                for c in (change_set.validation.checks if change_set.validation else [])
                if not c.passed
            ]
            result.detail = "change set is not publishable: " + (
                "; ".join(failures) or "no files to publish"
            )
            self._record(change_set, result)
            return result

        configured = metadata.get("repo_path") or self.repo_path
        repo = Path(configured) if configured else default_target_repo()
        if repo is None or not (Path(repo) / ".git").exists():
            result.detail = f"target is not a git repository: {repo}"
            self._record(change_set, result)
            return result
        repo = Path(repo).resolve()
        result.repo_path = str(repo)

        root = (self.worktree_root or resolve_worktree_root()).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        branch = self._branch_name(change_set)
        worktree = root / branch.replace("/", "--")
        base = str(metadata.get("base_ref") or self._base_ref(repo))

        ok, out = self._git(
            "worktree", "add", "-b", branch, str(worktree), base, cwd=repo
        )
        if not ok:
            result.detail = f"git worktree add failed: {out}"
            self._record(change_set, result)
            return result
        result.branch = branch
        result.worktree_path = str(worktree)

        try:
            for change in change_set.files:
                dest = worktree / change.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(change.content, encoding="utf-8")

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
                    "-m",
                    self._commit_message(change_set, result),
                    cwd=worktree,
                )
            if not ok:
                result.detail = f"git commit failed: {out}"
                self._record(change_set, result)
                return result
            ok, sha = self._git("rev-parse", "HEAD", cwd=worktree)
            result.commit_sha = sha if ok else ""
            result.ok = True
            result.detail = (
                f"published local branch {branch} (NOT pushed — review, then "
                "merge through the normal release flow)"
            )
        except Exception as exc:  # noqa: BLE001 — publishers never raise
            result.detail = f"publication error: {exc}"
            self._git("worktree", "remove", "--force", str(worktree), cwd=repo)
            result.worktree_path = ""

        self._record(change_set, result)
        return result

    def _commit_message(self, change_set: ChangeSet, result: PublishResult) -> str:
        concepts = ", ".join(dict.fromkeys([*change_set.concept_ids, "AHE-3.21"]))
        tests_line = (
            "not named by the proposal"
            if result.tests_passed is None
            else ("pass" if result.tests_passed else "FAIL — review required")
        )
        return (
            f"evolution: {change_set.kind} change for {change_set.title}\n\n"
            f"{change_set.summary or 'Materialized from the promoted proposal.'}\n\n"
            f"Proposal: {change_set.proposal_id}\n"
            f"Concepts: {concepts}\n"
            f"Regression gate: {result.gate_result}; targeted tests: {tests_line}\n"
            "Generated by the evolution→branch bridge (CONCEPT:AHE-3.21)."
        )

    def _run_targeted_tests(self, worktree: Path, tests: list[str]) -> dict[str, Any]:
        """Run proposal-named pytest targets inside the fresh worktree."""
        import sys

        argv = [sys.executable, "-m", "pytest", "-q", "--no-header", *tests]
        try:
            proc = subprocess.run(  # nosec B603 — fixed interpreter, sanitized targets
                argv,
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
            )
            return {
                "passed": proc.returncode == 0,
                "returncode": proc.returncode,
                "targets": tests,
                "tail": (proc.stdout or proc.stderr or "")[-2000:],
            }
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "targets": tests, "tail": str(exc)}

    def _run_gate(self, spec: Any) -> str:
        """Run the injected regression gate; verdicts: pass | hold | not_run."""
        if self.regression_check is None:
            return "not_run"
        try:
            return "pass" if bool(self.regression_check(spec)) else "hold"
        except Exception as exc:  # noqa: BLE001 — an erroring gate holds
            logger.debug("change_publisher: regression gate error: %s", exc)
            return "hold"

    def _record(self, change_set: ChangeSet, result: PublishResult) -> None:
        """Persist the publication on the graph (node + edge + proposal stamp)."""
        if self.engine is None:
            return
        publication_id = f"proposal_publication:{uuid.uuid4().hex[:12]}"
        try:
            self.engine.add_node(
                publication_id,
                "ProposalPublication",
                properties={
                    "proposal_id": change_set.proposal_id,
                    "kind": change_set.kind,
                    "ok": result.ok,
                    "branch": result.branch,
                    "commit_sha": result.commit_sha,
                    "repo_path": result.repo_path,
                    "worktree_path": result.worktree_path,
                    "gate_result": result.gate_result,
                    "tests_passed": (
                        "" if result.tests_passed is None else str(result.tests_passed)
                    ),
                    "detail": result.detail[:500],
                    "published_at": _now_iso(),
                },
            )
            link = getattr(self.engine, "link_nodes", None)
            if callable(link):
                link(change_set.proposal_id, publication_id, "PUBLISHED_AS")
        except Exception as exc:  # noqa: BLE001 — recording never blocks publication
            logger.debug("change_publisher: publication record failed: %s", exc)
        # Stamp the proposal node itself so its branch/sha/verdict are one
        # node-read away (best-effort; the ProposalPublication node is durable).
        try:
            self.engine.backend.execute(
                "MATCH (p) WHERE p.id = $id SET p.publish_branch = $branch, "
                "p.publish_commit = $sha, p.publish_gate = $gate",
                {
                    "id": change_set.proposal_id,
                    "branch": result.branch,
                    "sha": result.commit_sha,
                    "gate": result.gate_result,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("change_publisher: proposal stamp failed: %s", exc)


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
        logger.debug("change_publisher: approval scan failed: %s", exc)
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
        logger.debug("change_publisher: approval stamp failed: %s", exc)


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
    execution_id = f"action_execution:{uuid.uuid4().hex[:12]}"
    try:
        engine.add_node(
            execution_id,
            "ActionExecution",
            properties={
                "kind": "merge_promotion",
                "target": proposal_id,
                "params_json": json.dumps(
                    {"branch": result.branch, "commit_sha": result.commit_sha}
                ),
                "source": source,
                "actuator": publisher_name,
                "ok": bool(result.ok),
                "dry_run": False,
                "detail": result.detail[:500],
                "executed_at": _now_iso(),
                "executed_unix": time.time(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("change_publisher: execution record failed: %s", exc)
        return None
    return execution_id


def _abandon_branch(repo_path: str, worktree_path: str, branch: str) -> None:
    """Remove a published worktree and delete its local branch (AHE-3.23 rollback).

    Used when the capability ratchet rejects a just-published change: the local
    branch was never pushed, so a clean removal fully undoes the publication.
    """
    if not repo_path:
        return

    def _run(*args: str) -> None:
        try:
            subprocess.run(  # nosec B603 — fixed git args, no shell
                ["git", *args], cwd=repo_path, capture_output=True, text=True, timeout=60
            )
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.debug("change_publisher: abandon-branch step failed: %s", exc)

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
    source: str = "golden_loop",
    action_policy: Any = None,
    code_synthesizer: Any = None,
    capability_ratchet: Any = None,
) -> dict[str, Any]:
    """Publish a promoted proposal through the ActionPolicy gate (CONCEPT:AHE-3.21).

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
    report: dict[str, Any] = {"proposal_id": proposal_id, "source": source}

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
            report["status"] = "denied"
            report["detail"] = f"action policy unavailable (fail closed): {exc}"
            return report
        report["decision"] = decision.decision
        report["approval_id"] = decision.approval_id
        if not decision.allowed:
            report["status"] = "approval_queued" if decision.approval_id else "denied"
            report["detail"] = decision.reason
            return report
    else:
        report["decision"] = "approved"
        report["approval_id"] = granted_id

    # CONCEPT:AHE-3.22 — autonomous code-synthesis stage. For an attributed
    # proposal that carries no embedded files, generate a single-file edit so the
    # promotion emits real code; an un-attributed proposal yields None and falls
    # through to the prose SDD skeleton exactly as before. Generation failure is
    # never fatal — the prose path still runs.
    extra_files = None
    try:
        from .code_synthesis import synthesize_code

        extra_files = synthesize_code(proposal, synthesizer=code_synthesizer)
        if extra_files:
            report["code_synthesis"] = {"files": [f.path for f in extra_files]}
    except Exception as exc:  # noqa: BLE001 — generation failure ⇒ prose fallback
        logger.debug("[AHE-3.22] code synthesis skipped: %s", exc)

    try:
        change_set = synthesize_change_set(proposal, extra_files=extra_files)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "synthesis_failed"
        report["detail"] = str(exc)
        return report
    report["change_kind"] = change_set.kind
    if change_set.validation is not None:
        report["validation"] = change_set.validation.to_dict()
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

    # CONCEPT:AHE-3.23 / AHE-3.24 — verified apply→verify→rollback + capability
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
            report["capability_ratchet"] = verdict.to_dict()
            if not verdict.passed:
                _abandon_branch(result.repo_path, result.worktree_path, result.branch)
                result.ok = False
                report["publish"] = result.to_dict()
                report["status"] = "reverted"
                report["detail"] = verdict.reason
        except Exception as exc:  # noqa: BLE001 — ratchet failure must not wedge publish
            logger.debug("[AHE-3.23] capability ratchet skipped: %s", exc)

    report["execution_id"] = _record_execution(
        engine, proposal_id, getattr(pub, "name", "?"), result, source
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
        logger.debug("change_publisher: proposal lookup failed: %s", exc)
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

    The ``graph_orchestrate(action="publish_proposal")`` / REST entry point:
    loads the proposal node, then runs :func:`governed_publish` — so the
    ActionPolicy gate still applies (a granted approval, or an explicitly
    relaxed ``merge_promotion`` tier, lets it proceed; otherwise it queues).
    Governance validation already happened at promotion time (AHE-3.20); this
    step only materializes + publishes what was promoted.
    """
    proposal = load_proposal(engine, proposal_id)
    if proposal is None:
        return {
            "proposal_id": proposal_id,
            "status": "not_found",
            "detail": f"no proposal node with id {proposal_id!r}",
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
            resource_id=proposal_id,
            details={
                "status": report.get("status"),
                "branch": (report.get("publish") or {}).get("branch", ""),
                "commit_sha": (report.get("publish") or {}).get("commit_sha", ""),
                "gate_result": (report.get("publish") or {}).get("gate_result", ""),
                "decision": report.get("decision", ""),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("change_publisher: audit failed: %s", exc)
