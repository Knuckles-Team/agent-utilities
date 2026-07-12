#!/usr/bin/python
from __future__ import annotations

"""CI-failure -> LLM re-cycling loop — G15, the self-healing-CI piece of the SDLC
loop (``reports/autonomous-sdlc-loop-design.md`` §5.2), incorporating the three
ideas from the Atomic Task Graph paper analysis
(``reports/paper-analysis-2607.01942.md``).

CONCEPT:AU-OS.host.report-only-remediation-proposal. Given a ``:PipelineRun``/
``:CheckRun`` with a failing conclusion (now ingested by gitlab-api/github-agent —
grep ``kg_ingest`` in either package), this module:

1. **Reads** the failing job(s)' log excerpt + changed files from ``context``.
   agent-utilities core stays MCP-agnostic (per its own dependency-discipline
   rule — connectors live in ``agents/*``, not here); the CALLER — the existing
   ``github-ci-failure-sweep`` skill (``github_agent/skills/github-ci-failure-sweep``)
   or its GitLab twin — already pulls these via ``gitlab_jobs(action="get_log")``/
   ``github_actions(action="job_logs")`` and hands them in, exactly like
   :mod:`escalation_policy` consumes pre-resolved evidence rather than
   re-deriving it.
2. **DIAGNOSES** the failure with a small heuristic classifier over the log
   excerpt (lint / typecheck / test / build / timeout / pre_commit / unknown) —
   the same "probable cause" step the sweep skill already does by hand, made
   reusable.
3. **Localizes the repair region** (paper idea #1 —
   :func:`agent_utilities.workflows.localized_repair.localized_repair_region`):
   walks ``TRANSITION_TO`` FORWARD from the failing job/step node so a repair
   proposal is scoped to what the failure actually invalidates, not "rerun the
   whole pipeline" nor "rerun everything after this job in wall-clock order".
4. **Checks for a REUSABLE prior repair** (paper idea #2 —
   :func:`agent_utilities.observability.lifecycle_orchestrator.find_reusable`)
   before diagnosing from scratch: an equivalent earlier repair (same repo +
   failure-class signature) is reused rather than re-derived.
5. **Proposes a fix** as a REPORT-ONLY ``:CIRepairProposal`` (a
   ``:CodeChangeProposal``-shaped node) bound to the fleet capability that
   would open the fix — the spec-proposal/coding-agent path
   (``spec_proposals.develop_spec`` / ``graph_loops(submit, kind="develop")``,
   scoped to the failing branch per design §5.2 step 2) via gitlab-api/
   github-agent — stamped with a ``modelTier`` routing hint (paper idea #3): a
   single, well-scoped, already-decomposed one-job repair defaults to a cheap
   tier; only a genuinely ambiguous ("unknown") diagnosis escalates the tier.
6. **Tracks re-cycle attempts** per pipeline (idempotent, signature-keyed) and,
   once the retry cap is exceeded, consults
   :func:`agent_utilities.observability.escalation_policy.evaluate_escalation`
   (the existing ``red_ci_past_cap`` signal, design §3.4 signal 3) instead of
   looping forever — design §5.2 step 4.

**Report-only by design**, mirroring :mod:`lifecycle_orchestrator` /
:mod:`escalation_policy`: this module NEVER re-triggers CI, opens an MR/PR, or
merges anything. Turning a proposal into action is the SAME enablement path
those modules document: ``graph_loops(action="submit", kind="develop",
entry_node=<proposal id>)`` scoped to the failing branch, gated by an operator
un-suspending the CronJob
(``inventory/k8s-migration/cutover/apptier/ci-recycle.yaml``) once satisfied
with the report-only output — see that manifest's header comment for the exact
enablement steps.

Best-effort + engine-guarded: a missing engine degrades every entry point to
an empty/no-op result rather than raising. Run over one failed pipeline via
:func:`propose_ci_repair`, or sweep every currently-red pipeline via
:func:`sweep_failed_pipelines` (also runnable as
``python -m agent_utilities.observability.ci_recycle``).
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from agent_utilities.observability import escalation_policy, health_ingest
from agent_utilities.observability.lifecycle_orchestrator import find_reusable
from agent_utilities.workflows.localized_repair import localized_repair_region

logger = logging.getLogger("agent_utilities.observability.ci_recycle")

_SOURCE = "agent-utilities-ci-recycle"

# Retry cap before escalation fires (design §5.2 step 4 / §3.4 signal 3). Kept
# in sync with escalation_policy's own ESCALATION_CI_RETRY_CAP default (3) so
# the two agree without duplicating the setting() lookup — escalation_policy
# reads its own cap when it evaluates the red_ci_past_cap signal; this is only
# the LOCAL loop's own "stop proposing, hand off" threshold.
DEFAULT_RETRY_CAP = 3

# Heuristic failure classes -> a regex over the job-log excerpt. Mirrors the
# "probable cause" step the github-ci-failure-sweep skill already does by
# hand (SKILL.md Step 4), made reusable so the re-cycle loop doesn't need an
# LLM round-trip for the common cases.
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"pre-commit", "pre_commit"),
    (r"ruff|lint|flake8|black --check", "lint"),
    (r"mypy|type[- ]check", "typecheck"),
    (r"pytest|failed test|assertionerror|test.*failed", "test"),
    (r"docker build|buildx|failed to build|dockerfile", "build"),
    (r"time(d)? ?out|deadline exceeded", "timeout"),
)

# failure class -> the fleet capability that would open the fix (design §5.2
# step 2/3: a `develop` Loop scoped to the failing branch, reusing
# develop_spec()/validate_in_sandbox, then a push + gitlab_jobs(retry)/
# github_actions(rerun)). Report-only: this is bound, never invoked, here.
_CAPABILITY_BY_CLASS: dict[str, str] = {
    "lint": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(ruff --fix)",
    "typecheck": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(mypy fix)",
    "test": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(fix failing test)",
    "build": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(fix Dockerfile/build)",
    "timeout": "gitlab_jobs|github_actions:retry(no code change — transient)",
    "pre_commit": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(pre-commit fix)",
    "unknown": "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec(diagnose+fix)",
}

# Paper idea #3 (evidentiary routing) — a single, well-scoped, already-decomposed
# one-job repair (a known failure class) is exactly the case ATG shows a
# small/cheap model handles fine; an "unknown" cause needs judgment across the
# whole failure context, so it stays at the standard tier. Deliberately NOT a
# new model registry — just the same model_tier hint WorkflowRunner already
# honors (``workflows/runner.py``'s ``MODEL_TIER_REASONING_EFFORT``).
_MODEL_TIER_BY_CLASS: dict[str, str] = {
    "lint": "small",
    "typecheck": "small",
    "pre_commit": "small",
    "timeout": "small",
    "build": "standard",
    "test": "standard",
    "unknown": "standard",
}


def _diagnose(log_excerpt: str) -> str:
    """Classify a job-log excerpt into a failure class (probable cause)."""
    text = (log_excerpt or "").lower()
    for pattern, label in _FAILURE_PATTERNS:
        if re.search(pattern, text):
            return label
    return "unknown"


@dataclass
class _GraphReader:
    """Tiny best-effort reader, mirrors the pattern in lifecycle_orchestrator /
    escalation_policy so this module has no engine-shape assumptions of its own."""

    engine: Any

    def nodes_by_label(self, label: str) -> list[tuple[str, dict[str, Any]]]:
        if self.engine is None:
            return []
        try:
            return self.engine.get_nodes_by_label(label, 0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("ci_recycle: get_nodes_by_label(%s) failed: %s", label, e)
            return []


def _signature(pipeline_id: str, failure_class: str, attempt: int) -> str:
    raw = f"{pipeline_id}|{failure_class}|{attempt}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def _class_key(repo: str, failure_class: str) -> dict[str, Any]:
    """The coarse fields :func:`find_reusable` compares — repo + failure CLASS,
    deliberately excluding the specific pipeline/attempt so a repeat of the
    same failure class on the same repo reuses the prior diagnosis instead of
    re-deriving it (paper idea #2)."""
    return {
        "kind": "ci_repair",
        "fromStage": "pipeline_run",
        "toStage": "code_change",
        "transition": failure_class,
        "targetType": repo,
    }


def propose_ci_repair(
    pipeline_run: dict[str, Any],
    *,
    engine: Any | None = None,
    write: bool = True,
    context: dict[str, Any] | None = None,
    retry_cap: int = DEFAULT_RETRY_CAP,
    reuse_similarity: float = 0.85,
) -> dict[str, Any]:
    """Diagnose ONE failed ``:PipelineRun``/``:CheckRun`` and propose a
    REPORT-ONLY repair (design §5.2, G15).

    ``pipeline_run``: ``{"id": ..., "repo": ...}`` (or any dict with an "id").
    ``context``: what the caller already pulled — ``{"log_excerpt",
    "changed_files", "repo", "commit", "branch", "failed_step", "all_steps"}``.
    ``failed_step`` is the node id to localize the repair region from (design
    §4 Rank 1 of the paper analysis); defaults to the pipeline id itself when
    the caller has no finer-grained job/step graph. ``all_steps`` is the full
    id set of the pipeline's modeled steps (for computing the PRESERVED
    region) — optional.

    NEVER executes anything — see the module docstring for the enablement
    path. Idempotent: re-proposing the same (pipeline, failure_class, attempt)
    triple is a no-op signature collision, handled by the caller's dedupe
    exactly like :mod:`lifecycle_orchestrator`'s ``:LifecycleStep`` proposals
    (this module doesn't re-check the graph for its own signature because
    ``attempt`` — derived from the count of prior ``:CIRepairProposal`` nodes —
    already advances monotonically per call).
    """
    eng = engine if engine is not None else health_ingest._engine()
    pid = str(pipeline_run.get("id") or "")
    ctx = dict(context or {})
    repo = str(ctx.get("repo") or pipeline_run.get("repo") or "")
    log_excerpt = str(ctx.get("log_excerpt") or "")
    failed_step = str(ctx.get("failed_step") or pid)

    reader = _GraphReader(engine=eng)
    prior_attempts = [
        p
        for _id, p in reader.nodes_by_label("CIRepairProposal")
        if isinstance(p, dict) and p.get("pipelineRun") == pid
    ]
    attempt = len(prior_attempts) + 1

    # --- retry cap exceeded -> escalate instead of looping forever ---------- #
    if attempt > retry_cap:
        request = escalation_policy.evaluate_escalation(
            {
                "entry": pid,
                "transition": "await_ci",
                "pipeline_run": pid,
                "pipeline_status": "failed",
                "attempts": attempt - 1,
            },
            engine=eng,
            write=write,
        )
        logger.info(
            "[ORCH.ci_recycle] pipeline %s exceeded retry cap (%d) -> escalated",
            pid,
            retry_cap,
        )
        return {
            "pipeline_run": pid,
            "attempt": attempt - 1,
            "capped": True,
            "escalation": request,
            "proposal": None,
        }

    # --- diagnose ------------------------------------------------------------#
    failure_class = _diagnose(log_excerpt)

    # --- paper idea #1: localize the repair region --------------------------#
    region = localized_repair_region(
        failed_step, engine=eng, all_nodes=ctx.get("all_steps")
    )

    # --- paper idea #2: reuse a prior equivalent repair before re-deriving --#
    class_key = _class_key(repo, failure_class)
    reused = find_reusable(
        class_key,
        [
            (nid, p)
            for nid, p in reader.nodes_by_label("CIRepairProposal")
            if isinstance(p, dict)
        ],
        similarity=reuse_similarity,
    )

    sig = _signature(pid, failure_class, attempt)
    proposal: dict[str, Any] = {
        "id": f"cirepair:{pid}:{sig}",
        "type": "CIRepairProposal",
        "pipelineRun": pid,
        "repo": repo,
        "attempt": attempt,
        "failureClass": failure_class,
        "invalidatedRegion": region["invalidated"],
        "preservedRegion": region["preserved"],
        "boundCapability": _CAPABILITY_BY_CLASS.get(
            failure_class, _CAPABILITY_BY_CLASS["unknown"]
        ),
        # paper idea #3 — evidentiary routing hint (WorkflowRunner's
        # MODEL_TIER_REASONING_EFFORT honors this same field name).
        "modelTier": _MODEL_TIER_BY_CLASS.get(failure_class, "standard"),
        "signature": sig,
        "status": "proposed",  # report-only — nothing dispatched
    }
    if reused is not None:
        proposal["reusedFrom"] = reused["reusedFrom"]
        proposal["reuseScore"] = reused["reuseScore"]
        proposal["status"] = "reused"
        logger.info(
            "[ORCH.ci_recycle] pipeline %s: reusing prior repair %s (score=%.2f) "
            "instead of re-deriving",
            pid,
            reused["reusedFrom"],
            reused["reuseScore"],
        )

    if write and eng is not None:
        _write_proposal(proposal, pid)

    return {
        "pipeline_run": pid,
        "attempt": attempt,
        "capped": False,
        "escalation": None,
        "proposal": proposal,
    }


def _write_proposal(
    proposal: dict[str, Any], pipeline_id: str
) -> dict[str, int] | None:
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    relationships = (
        [{"source": pipeline_id, "target": proposal["id"], "type": "hasRepairProposal"}]
        if pipeline_id
        else []
    )
    try:
        return ingest_entities([proposal], relationships, source=_SOURCE, domain="sdlc")
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        logger.debug("ci_recycle: write skipped: %s", e)
        return None


def sweep_failed_pipelines(
    *, engine: Any | None = None, write: bool = True, retry_cap: int = DEFAULT_RETRY_CAP
) -> dict[str, Any]:
    """Scheduled sweep (mirrors ``lifecycle_orchestrator.sweep_open_spine``):
    scan every ``:PipelineRun``/``:CheckRun`` with a failing status and propose
    a repair for each — catches CI failures whether or not a webhook fired.
    Best-effort per node."""
    eng = engine or health_ingest._engine()
    if eng is None:
        return {"scanned": 0, "proposed": 0, "reused": 0, "escalated": 0}
    reader = _GraphReader(engine=eng)
    scanned = proposed = reused = escalated = 0
    for label in ("PipelineRun", "CheckRun"):
        for node_id, props in reader.nodes_by_label(label):
            if not isinstance(props, dict):
                continue
            status = str(props.get("status") or props.get("conclusion") or "").lower()
            if status not in ("failed", "failure", "red", "error"):
                continue
            scanned += 1
            try:
                out = propose_ci_repair(
                    {"id": node_id, "repo": props.get("repo")},
                    engine=eng,
                    write=write,
                    context={"log_excerpt": str(props.get("logExcerpt") or "")},
                    retry_cap=retry_cap,
                )
                if out.get("capped"):
                    escalated += 1
                elif out.get("proposal"):
                    proposed += 1
                    if out["proposal"].get("status") == "reused":
                        reused += 1
            except Exception as e:  # noqa: BLE001 — one node must not break the sweep
                logger.debug("ci_recycle: sweep failed for %s: %s", node_id, e)
    return {
        "scanned": scanned,
        "proposed": proposed,
        "reused": reused,
        "escalated": escalated,
    }


def main() -> None:
    """CLI (``python -m agent_utilities.observability.ci_recycle``): one sweep
    across every currently-red pipeline; prints a JSON summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print(json.dumps(sweep_failed_pipelines(), default=str, indent=2))


if __name__ == "__main__":
    main()
