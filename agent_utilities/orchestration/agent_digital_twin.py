#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-ORCH.twin.agent-digital-twin — Agent Digital Twin + deterministic replay (Codex X-8).

Stores, for one agent run: the agent definition + the exact model/prompt/tool/skill
VERSIONS it executed under, its policy + budget, the run graph, every tool call + its
evidence, the decisions it made, and its outcome — so a stored run can be
deterministically replayed against a historical KG snapshot for regression testing,
incident investigation, counterfactual policy evaluation, and safe-evolution proposals.

This module is deliberately thin. Every piece of provenance it touches already exists
elsewhere in this codebase and is REUSED, never duplicated:

* **The run graph** — :class:`~agent_utilities.orchestration.work_item.WorkItemStatus`
  ids (``orchestration/work_item.py``): the ``WorkItem`` DAG (``depends_on``/
  ``downstream_ids``) a run executed as. A twin stores the ids, not a second copy of
  the DAG.
* **Tool calls + evidence** — the SAME ``:ToolCall`` shape
  (``orchestration/tool_provenance.py``'s ``extract_tool_calls``/``sanitize_tool_args``,
  persisted by ``agent_runner._persist_tool_calls``): ``{tool_name, args, result,
  error}``. A twin mirrors each one as a ``tool_call:<name>`` declare/capture pair in a
  :class:`~agent_utilities.runtime.run_vcs.kernel.RunEventLog` — the SAME content-
  addressed event kernel a live run appends to — so replay can mock it (never
  re-execute a real side effect), exactly per :mod:`agent_utilities.runtime.run_vcs.replay`'s
  existing contract.
* **Decisions** — :class:`~agent_utilities.models.knowledge_graph.AgentPolicyDecisionNode`
  (the named object over ``action_policy.ActionPolicy.decide()``'s ``ActionDecision``
  audit). A twin records each recorded ``(request, decision)`` pair so a counterfactual
  policy swap can genuinely re-invoke ``ActionPolicy.decide()`` (a pure function, no
  I/O side effect) against the SAME recorded request and diff the outcome.
* **Version pins** — :class:`VersionPins` reads the version stamps that already exist:
  ``GenerationNode.model``/``prompt_version_id``
  (:mod:`agent_utilities.models.knowledge_graph`), ``tool_version``
  (:mod:`agent_utilities.tools.versioning`), a skill's own ``version`` frontmatter, the
  policy file's content digest (:mod:`agent_utilities.orchestration.action_policy`),
  and :attr:`agent_utilities.knowledge_graph.core.session.GraphSession.catalog_epoch`
  (the historical-snapshot pin).
* **Deterministic replay** — :func:`agent_utilities.runtime.run_vcs.replay.replay_run`
  is reused UNCHANGED for the regression check; :func:`counterfactual_replay` in this
  module extends the same event log + :class:`~agent_utilities.runtime.run_vcs.replay.ReplayModel`
  machinery to swap a version and observe the delta, still without ever calling a live
  model or re-executing a real tool.

See :func:`capture_twin` (build one from explicit run data — this is what a live run or
a test uses), :func:`capture_twin_from_kg` (best-effort hydration from an already-running
KG's existing nodes), :func:`replay_twin` (regression), :func:`counterfactual_replay`
(policy/model/prompt swap), and :func:`twin_incident_steps` (step-through investigation).
"""

import contextlib
import json
import logging
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.orchestration.action_policy import (
    ActionDecision,
    ActionPolicy,
    ActionRequest,
)
from agent_utilities.orchestration.tool_provenance import sanitize_tool_args
from agent_utilities.orchestration.work_item import WorkItemStatus
from agent_utilities.runtime.run_vcs.kernel import RunEventLog, content_digest
from agent_utilities.runtime.run_vcs.replay import (
    MODEL_EXCHANGE,
    ReplayModel,
    ReplayResult,
    capture_of,
    replay_run,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VersionPins",
    "AgentDigitalTwin",
    "TwinReplayReport",
    "capture_twin",
    "capture_twin_from_kg",
    "persist_twin",
    "replay_twin",
    "counterfactual_replay",
    "twin_incident_steps",
]

#: run-VCS ``schema_ref`` prefix for a mirrored ``:ToolCall`` (per-tool-name so a
#: replay divergence names the exact tool, mirroring MODEL_EXCHANGE's own single
#: well-known schema_ref for model calls).
TOOL_CALL_SCHEMA_PREFIX = "tool_call:"

#: run-VCS ``schema_ref`` for a mirrored policy decision (request -> ActionDecision).
POLICY_DECISION_SCHEMA = "policy_decision"

_VERSION_PIN_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_provider",
    "prompt_version_id",
    "tool_versions",
    "skill_versions",
    "policy_version",
    "policy_digest",
    "catalog_epoch",
)


# ---------------------------------------------------------------------------
# VersionPins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionPins:
    """The exact version pins one agent run executed under (Codex X-8).

    Every field is a version identifier for one axis of "what could regress if it
    changed" — never a new versioning scheme: each is read straight from an existing
    stamp elsewhere in the codebase (see the module docstring). ``catalog_epoch`` pins
    the KG schema/snapshot generation
    (:attr:`~agent_utilities.knowledge_graph.core.session.GraphSession.catalog_epoch`)
    so replay can be run against the SAME historical snapshot the run actually saw.
    """

    model_id: str = ""
    model_provider: str = ""
    prompt_version_id: str | None = None
    tool_versions: dict[str, str] = field(default_factory=dict)
    skill_versions: dict[str, str] = field(default_factory=dict)
    policy_version: str = ""
    policy_digest: str = ""
    catalog_epoch: int | None = None

    def digest(self) -> str:
        """A single content digest identifying this exact version combination.

        Two twins with identical pins share this digest — the version-pin analogue of
        the run-VCS kernel's ``record_id == digest`` identity.
        """
        return content_digest(
            "version_pins",
            "declaration",
            {
                "model_id": self.model_id,
                "model_provider": self.model_provider,
                "prompt_version_id": self.prompt_version_id,
                "tool_versions": dict(sorted(self.tool_versions.items())),
                "skill_versions": dict(sorted(self.skill_versions.items())),
                "policy_version": self.policy_version,
                "policy_digest": self.policy_digest,
                "catalog_epoch": self.catalog_epoch,
            },
            (),
        )

    def diff(self, other: VersionPins) -> dict[str, tuple[Any, Any]]:
        """Field-by-field diff against ``other``; empty ⇒ identical pins."""
        out: dict[str, tuple[Any, Any]] = {}
        for f in _VERSION_PIN_FIELDS:
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out[f] = (a, b)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_provider": self.model_provider,
            "prompt_version_id": self.prompt_version_id,
            "tool_versions": dict(self.tool_versions),
            "skill_versions": dict(self.skill_versions),
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "catalog_epoch": self.catalog_epoch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VersionPins:
        return cls(
            model_id=str(data.get("model_id") or ""),
            model_provider=str(data.get("model_provider") or ""),
            prompt_version_id=data.get("prompt_version_id"),
            tool_versions=dict(data.get("tool_versions") or {}),
            skill_versions=dict(data.get("skill_versions") or {}),
            policy_version=str(data.get("policy_version") or ""),
            policy_digest=str(data.get("policy_digest") or ""),
            catalog_epoch=data.get("catalog_epoch"),
        )


# ---------------------------------------------------------------------------
# ActionDecision <-> plain-dict normalization (so a twin can serialize/replay one)
# ---------------------------------------------------------------------------


def _decision_to_dict(decision: ActionDecision | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an ``ActionDecision`` (or an already-dict one) to a plain, JSON-safe dict."""
    if isinstance(decision, Mapping):
        req = decision.get("request") or {}
        request = (
            dict(req) if isinstance(req, Mapping) else _request_to_dict(req)  # type: ignore[arg-type]
        )
        return {
            "request": request,
            "decision": str(decision.get("decision") or ""),
            "tier": str(decision.get("tier") or ""),
            "reason": str(decision.get("reason") or ""),
            "rule_origin": str(decision.get("rule_origin") or "default"),
            "approval_id": decision.get("approval_id"),
            "audit_id": decision.get("audit_id"),
        }
    return {
        "request": _request_to_dict(decision.request),
        "decision": decision.decision,
        "tier": decision.tier,
        "reason": decision.reason,
        "rule_origin": decision.rule_origin,
        "approval_id": decision.approval_id,
        "audit_id": decision.audit_id,
    }


def _request_to_dict(request: ActionRequest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(request, Mapping):
        return {
            "kind": str(request.get("kind") or ""),
            "target": str(request.get("target") or ""),
            "params": dict(request.get("params") or {}),
            "source": str(request.get("source") or "manual"),
            "reason": str(request.get("reason") or ""),
            "actor_id": str(request.get("actor_id") or ""),
        }
    return {
        "kind": request.kind,
        "target": request.target,
        "params": dict(request.params),
        "source": request.source,
        "reason": request.reason,
        "actor_id": request.actor_id,
    }


def _evidence_to_dict(evidence: Any) -> dict[str, Any]:
    """Normalize an ``EvidenceBundle`` (or a dict) to a plain dict, never fabricating fields."""
    if isinstance(evidence, Mapping):
        return dict(evidence)
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    return dict(evidence)  # best-effort — raises loudly on a truly unusable shape


# ---------------------------------------------------------------------------
# AgentDigitalTwin
# ---------------------------------------------------------------------------


@dataclass
class AgentDigitalTwin:
    """A durable, replayable projection of one agent run (Codex X-8).

    See the module docstring for what this reuses. ``event_log`` is the run-VCS
    :class:`~agent_utilities.runtime.run_vcs.kernel.RunEventLog` this twin's tool calls
    and decisions were mirrored into — the deterministic-replay substrate.
    """

    twin_id: str
    run_id: str
    agent_name: str
    task: str
    versions: VersionPins
    budget: dict[str, Any]
    work_item_ids: list[str]
    tool_call_ids: list[str]
    decision_ids: list[str]
    policy_decisions: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    outcome: str
    created_at: float
    event_log: RunEventLog

    # ---- queryable run-graph / decisions / evidence ------------------------
    def run_graph(self) -> list[str]:
        """The WorkItem ids forming this run's DAG (query surface, not a copy of the DAG)."""
        return list(self.work_item_ids)

    def decisions(self) -> list[dict[str, Any]]:
        """The recorded ``(request, decision)`` pairs this run's policy gate produced."""
        return [dict(d) for d in self.policy_decisions]

    def evidence_bundles(self) -> list[dict[str, Any]]:
        """The recorded evidence backing this run's decisions/outcome."""
        return [dict(e) for e in self.evidence]

    # ---- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "twin_id": self.twin_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "versions": self.versions.to_dict(),
            "budget": dict(self.budget),
            "work_item_ids": list(self.work_item_ids),
            "tool_call_ids": list(self.tool_call_ids),
            "decision_ids": list(self.decision_ids),
            "policy_decisions": [dict(d) for d in self.policy_decisions],
            "evidence": [dict(e) for e in self.evidence],
            "outcome": self.outcome,
            "created_at": self.created_at,
            "events": [e.to_dict() for e in self.event_log.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentDigitalTwin:
        run_id = str(data["run_id"])
        log = RunEventLog.from_records(run_id, list(data.get("events") or []))
        return cls(
            twin_id=str(data["twin_id"]),
            run_id=run_id,
            agent_name=str(data.get("agent_name") or ""),
            task=str(data.get("task") or ""),
            versions=VersionPins.from_dict(data.get("versions") or {}),
            budget=dict(data.get("budget") or {}),
            work_item_ids=list(data.get("work_item_ids") or []),
            tool_call_ids=list(data.get("tool_call_ids") or []),
            decision_ids=list(data.get("decision_ids") or []),
            policy_decisions=[dict(d) for d in data.get("policy_decisions") or []],
            evidence=[dict(e) for e in data.get("evidence") or []],
            outcome=str(data.get("outcome") or ""),
            created_at=float(data.get("created_at") or 0.0),
            event_log=log,
        )

    def to_node_properties(self) -> dict[str, Any]:
        """The property row for the persisted ``:AgentDigitalTwin`` KG node (see :func:`persist_twin`)."""
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "task": self.task[:500],
            "versions_digest": self.versions.digest(),
            "versions_json": json.dumps(self.versions.to_dict(), default=str)[:4000],
            "budget": dict(self.budget),
            "work_item_ids": list(self.work_item_ids),
            "tool_call_ids": list(self.tool_call_ids),
            "decision_ids": list(self.decision_ids),
            "outcome": self.outcome,
            "event_count": len(self.event_log.events),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_twin(
    *,
    agent_name: str,
    task: str = "",
    versions: VersionPins,
    run_id: str | None = None,
    budget: Mapping[str, Any] | None = None,
    work_item_ids: Sequence[str] = (),
    tool_calls: Sequence[Mapping[str, Any]] = (),
    model_exchanges: Sequence[Mapping[str, Any]] = (),
    policy_decisions: Sequence[ActionDecision | Mapping[str, Any]] = (),
    evidence: Sequence[Any] = (),
    outcome: str = WorkItemStatus.SUCCEEDED.value,
    engine: Any | None = None,
) -> AgentDigitalTwin:
    """Capture an :class:`AgentDigitalTwin` from explicit run data.

    This is the path a live run (or a test standing in for one) uses: pass whatever
    the run actually produced — the ``:ToolCall``-shaped ``tool_calls``
    (``{tool_name, args, result, error}``, the exact shape
    :func:`agent_utilities.orchestration.tool_provenance.extract_tool_calls` returns),
    any ``model_exchanges`` (``{request, response}``), the ``ActionDecision``s its
    policy gate produced, and any ``EvidenceBundle``s backing the outcome. Every tool
    call / model exchange is mirrored into a fresh
    :class:`~agent_utilities.runtime.run_vcs.kernel.RunEventLog` as a declare/capture
    pair, so :func:`replay_twin`/:func:`counterfactual_replay` can replay them without
    ever re-executing a real side effect. ``engine`` is optional and best-effort (KG
    mirroring only) — capture never requires a live KG.
    """
    run_id = run_id or f"twinrun:{uuid.uuid4().hex}"
    log = RunEventLog(run_id, engine=engine)

    # Model exchanges causally precede whatever they drove.
    for exch in model_exchanges:
        decl = log.declare(MODEL_EXCHANGE, {"request": exch.get("request")})
        log.capture(MODEL_EXCHANGE, {"response": exch.get("response")}, of=decl)

    tool_call_ids: list[str] = []
    run_suffix = run_id.split(":", 1)[-1]
    for i, tc in enumerate(tool_calls):
        tool_name = str(tc.get("tool_name") or "")
        schema_ref = f"{TOOL_CALL_SCHEMA_PREFIX}{tool_name}"
        decl = log.declare(
            schema_ref,
            {"tool_name": tool_name, "args": sanitize_tool_args(tc.get("args"))},
        )
        log.capture(
            schema_ref,
            {"result": tc.get("result", ""), "error": tc.get("error", "")},
            of=decl,
        )
        tool_call_ids.append(f"toolcall:{run_suffix}:{i}")

    decision_ids: list[str] = []
    normalized_decisions: list[dict[str, Any]] = []
    for i, pd in enumerate(policy_decisions):
        d = _decision_to_dict(pd)
        decl = log.declare(POLICY_DECISION_SCHEMA, {"request": d["request"]})
        log.capture(POLICY_DECISION_SCHEMA, {"decision": d}, of=decl)
        decision_ids.append(d.get("audit_id") or f"action_decision:{run_id}:{i}")
        normalized_decisions.append(d)

    return AgentDigitalTwin(
        twin_id=f"twin:{uuid.uuid4().hex}",
        run_id=run_id,
        agent_name=agent_name,
        task=task,
        versions=versions,
        budget=dict(budget or {}),
        work_item_ids=list(work_item_ids),
        tool_call_ids=tool_call_ids,
        decision_ids=decision_ids,
        policy_decisions=normalized_decisions,
        evidence=[_evidence_to_dict(e) for e in evidence],
        outcome=outcome,
        created_at=time.time(),
        event_log=log,
    )


def capture_twin_from_kg(
    engine: Any,
    run_id: str,
    *,
    agent_name: str = "",
    task: str = "",
    versions: VersionPins,
    outcome: str = "",
    policy_decisions: Sequence[ActionDecision | Mapping[str, Any]] = (),
    evidence: Sequence[Any] = (),
) -> AgentDigitalTwin:
    """Best-effort hydration of a twin from an ALREADY-RUNNING KG's existing nodes.

    Reads (never duplicates) what ``agent_runner``/``work_item`` already wrote:

    * the run's ``:ToolCall`` children via the ``MADE_TOOL_CALL`` edge off
      ``trace:<run_id>`` (the SAME edge ``agent_runner._persist_tool_calls`` writes);
    * any ``WorkItem`` rows whose ``correlation_id`` equals this run id.

    ``policy_decisions``/``evidence`` are NOT auto-discovered — today's
    ``AgentPolicyDecisionNode``/``ActionDecision`` audit rows carry no edge back to the
    ``RunTrace`` that produced them (a follow-up gap this twin surfaces rather than
    papering over with a fabricated join); pass them explicitly when known. Degrades to
    empty lists on a cold/partial graph rather than raising — capture must never break
    a run.
    """
    tool_calls: list[dict[str, Any]] = []
    work_item_ids: list[str] = []
    trace_id = f"trace:{run_id}"

    query = getattr(engine, "query_cypher", None)
    if callable(query):
        with contextlib.suppress(Exception):
            rows = query(
                "MATCH (:RunTrace {id: $tid})-[:MADE_TOOL_CALL]->(tc:ToolCall) "
                "RETURN tc.id AS id, tc.tool_name AS tool_name, tc.args AS args, "
                "tc.result_preview AS result, tc.error AS error "
                "ORDER BY tc.sequence",
                {"tid": trace_id},
            )
            for row in rows or []:
                tool_calls.append(
                    {
                        "tool_name": row.get("tool_name") or "",
                        "args": row.get("args") or "",
                        "result": row.get("result") or "",
                        "error": row.get("error") or "",
                    }
                )
        with contextlib.suppress(Exception):
            rows = query(
                "MATCH (w:WorkItem {correlation_id: $cid}) RETURN w.id AS id",
                {"cid": run_id},
            )
            work_item_ids = [
                str(row["id"]) for row in (rows or []) if row.get("id") is not None
            ]

    return capture_twin(
        agent_name=agent_name,
        task=task,
        versions=versions,
        run_id=run_id,
        work_item_ids=work_item_ids,
        tool_calls=tool_calls,
        policy_decisions=policy_decisions,
        evidence=evidence,
        outcome=outcome or WorkItemStatus.SUCCEEDED.value,
        engine=engine,
    )


def persist_twin(engine: Any, twin: AgentDigitalTwin) -> str | None:
    """Best-effort: write ``twin`` as a durable ``:AgentDigitalTwin`` KG node.

    Mirrors :func:`agent_utilities.orchestration.agent_runner._record_execution_trace`'s
    pattern: one node write plus reference edges, all best-effort so a cold/absent KG
    never breaks capture. Returns the node id, or ``None`` if ``engine`` is falsy or the
    write failed.
    """
    if not engine:
        return None
    node_id = (
        twin.twin_id if twin.twin_id.startswith("twin:") else f"twin:{twin.twin_id}"
    )
    try:
        engine.add_node(
            node_id, "AgentDigitalTwin", properties=twin.to_node_properties()
        )
        linker = getattr(engine, "link_nodes", None) or getattr(
            engine, "add_edge", None
        )
        if callable(linker):
            linker(node_id, f"trace:{twin.run_id}", "TWIN_OF")
            for wid in twin.work_item_ids:
                linker(node_id, wid, "REFERENCES")
            for tid in twin.tool_call_ids:
                linker(node_id, tid, "REFERENCES")
            for did in twin.decision_ids:
                linker(node_id, did, "REFERENCES")
        return node_id
    except Exception as exc:  # noqa: BLE001 — a provenance write must never break the run
        logger.debug("agent_digital_twin: persist failed for %s: %s", node_id, exc)
        return None


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


@dataclass
class TwinReplayReport:
    """The outcome of replaying an :class:`AgentDigitalTwin`.

    ``regression`` is always populated (:func:`agent_utilities.runtime.run_vcs.replay.replay_run`'s
    own :class:`~agent_utilities.runtime.run_vcs.replay.ReplayResult`); ``version_delta``/
    ``decision_delta`` are only non-empty for a :func:`counterfactual_replay` call.
    """

    run_id: str
    twin_id: str
    regression: ReplayResult
    version_delta: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    decision_delta: list[dict[str, Any]] = field(default_factory=list)
    diverged: bool = False

    @property
    def deterministic(self) -> bool:
        """True iff the regression replay reproduced the original trace exactly."""
        return self.regression.deterministic

    @property
    def has_counterfactual_delta(self) -> bool:
        """True iff a version swap produced an observable difference (policy or stream)."""
        return self.diverged


def replay_twin(twin: AgentDigitalTwin) -> TwinReplayReport:
    """Regression replay: re-drive ``twin``'s recorded run graph verbatim.

    Reuses :func:`agent_utilities.runtime.run_vcs.replay.replay_run` UNCHANGED — every
    tool call / model exchange is answered from the record (never re-executed), so
    identical pinned versions + identical recorded log ⇒ an identical reconstructed
    output stream. ``report.deterministic`` is ``True`` iff the reconstruction digest
    equals the original.
    """
    result = replay_run(twin.event_log)
    return TwinReplayReport(run_id=twin.run_id, twin_id=twin.twin_id, regression=result)


def counterfactual_replay(
    twin: AgentDigitalTwin,
    *,
    versions: VersionPins | None = None,
    policy_overrides: Mapping[str, Any] | None = None,
    model_responses: Mapping[Any, Any] | None = None,
) -> TwinReplayReport:
    """Re-drive ``twin``'s recorded run graph under a swapped model/prompt/policy version.

    Never calls a live model or re-executes a real tool — that would make "replay" a
    fresh, non-deterministic run instead of a counterfactual over the SAME recorded
    history:

    * **Policy swap** — ``policy_overrides`` (a policy YAML's own ``{version, defaults,
      rules}`` shape) is written to a throwaway file and a fresh
      :class:`~agent_utilities.orchestration.action_policy.ActionPolicy` is built
      against it (``engine=None`` — no KG audit write). ``ActionPolicy.decide()`` is a
      pure function of ``(request, loaded rules)``, so re-invoking it against every
      recorded :class:`~agent_utilities.orchestration.action_policy.ActionRequest` is
      NOT a side effect — it genuinely recomputes what the swapped policy would have
      decided, and the result is diffed against the twin's recorded decision.
    * **Model/prompt swap** — since no live model call is safe to make deterministically,
      the caller supplies ``model_responses`` (``{request: alternate_response}``): what
      a different model/prompt version WOULD have produced for a given recorded
      request. This substitutes the alternate response into the replay in place of the
      recorded capture, surfacing the resulting stream divergence — the same mechanism
      an incident root-cause A/B uses.

    ``versions`` (if given) is diffed against ``twin.versions`` purely for reporting
    (``report.version_delta``) — it does not by itself change replay behavior; pass
    ``policy_overrides``/``model_responses`` to actually drive a different outcome.
    """
    versions = versions or twin.versions
    version_delta = twin.versions.diff(versions)

    # ---- model/prompt counterfactual -------------------------------------
    model = ReplayModel.from_log(twin.event_log)
    if model_responses:
        for request, response in model_responses.items():
            model.override(request, response)

    events = twin.event_log.events
    reconstructed: list[Any] = []
    for ev in events:
        if ev.mode != "declaration":
            continue
        if ev.schema_ref == MODEL_EXCHANGE:
            reconstructed.append(model.respond(ev.payload.get("request")))
        else:
            capture = capture_of(ev, events)
            reconstructed.append(capture.payload if capture is not None else None)

    baseline = replay_run(twin.event_log)
    replay_digest = content_digest("replay", "capture", {"stream": reconstructed}, ())
    regression = ReplayResult(
        run_id=twin.run_id,
        steps=baseline.steps,
        model_calls=model.calls,
        reconstructed=reconstructed,
        original_digest=baseline.original_digest,
        replay_digest=replay_digest,
    )

    # ---- policy counterfactual: genuinely recompute each recorded decision --
    decision_delta: list[dict[str, Any]] = []
    if policy_overrides is not None:
        with _policy_from_overrides(policy_overrides) as policy:
            for d in twin.policy_decisions:
                req = ActionRequest(**d["request"])
                recomputed = policy.decide(req)
                if recomputed.decision != d["decision"] or recomputed.tier != d["tier"]:
                    decision_delta.append(
                        {
                            "request": d["request"],
                            "original": {
                                "decision": d["decision"],
                                "tier": d["tier"],
                                "reason": d["reason"],
                            },
                            "counterfactual": {
                                "decision": recomputed.decision,
                                "tier": recomputed.tier,
                                "reason": recomputed.reason,
                            },
                        }
                    )

    diverged = (
        replay_digest != baseline.original_digest
        or bool(decision_delta)
        or bool(version_delta and (model_responses or policy_overrides))
    )
    return TwinReplayReport(
        run_id=twin.run_id,
        twin_id=twin.twin_id,
        regression=regression,
        version_delta=version_delta,
        decision_delta=decision_delta,
        diverged=diverged,
    )


@contextlib.contextmanager
def _policy_from_overrides(policy_overrides: Mapping[str, Any]) -> Any:
    """Yield a throwaway, engine-less :class:`ActionPolicy` over an inline rule-set.

    Writes ``policy_overrides`` to a temporary YAML file (the only ruleset input
    ``ActionPolicy`` accepts) so the counterfactual policy version is a real,
    independently-loaded rule-set — not a monkeypatch of the recorded policy. The file
    must outlive every ``decide()`` call made against the yielded policy (``decide()``
    lazily re-reads its ``policy_path`` on each call), so cleanup happens on scope exit,
    not immediately after construction.
    """
    import yaml

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(dict(policy_overrides), tmp)
        tmp.flush()
        tmp.close()
        yield ActionPolicy(engine=None, policy_path=Path(tmp.name))
    finally:
        with contextlib.suppress(OSError):
            Path(tmp.name).unlink()


# ---------------------------------------------------------------------------
# Incident investigation
# ---------------------------------------------------------------------------


def twin_incident_steps(twin: AgentDigitalTwin) -> list[dict[str, Any]]:
    """Ordered, human-inspectable steps for incident investigation (Codex X-8c).

    Walks the recorded event log in causal order, projecting each declaration/capture
    pair into one step: what was proposed (a tool call / model exchange / policy
    decision) and what was recorded as its outcome. This is read-only — it steps
    THROUGH the recorded decisions/evidence, it never re-runs anything.
    """
    events = twin.event_log.events
    steps: list[dict[str, Any]] = []
    for ev in events:
        if ev.mode != "declaration":
            continue
        capture = capture_of(ev, events)
        steps.append(
            {
                "ordinal": ev.ordinal,
                "schema_ref": ev.schema_ref,
                "declaration": dict(ev.payload),
                "capture": dict(capture.payload) if capture is not None else None,
                "record_id": ev.record_id,
            }
        )
    return steps
