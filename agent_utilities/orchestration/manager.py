import asyncio
import json
import logging
import uuid
from typing import Any, TypedDict, cast

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler
from agent_utilities.observability.trace_ontology import (
    TRACE_USED_TOOL_EDGE,
)
from agent_utilities.observability.trace_ontology import (
    trace_id as canonical_trace_id,
)
from agent_utilities.orchestration.agent_runner import ProgressSink, run_agent
from agent_utilities.orchestration.execution_contract import (
    ExecutionMode,
    validate_execution_mode,
    validate_pydantic_graph_contract,
    validate_tool_contract,
)
from agent_utilities.orchestration.response_format import (
    ResponseFormat,
    validate_response_format,
)
from agent_utilities.security.threat_defense_engine import PromptInjectionScanner

logger = logging.getLogger(__name__)

# Aliased from the capability contract so the delegate a resolved Tool binds
# through cannot drift between the two modules.
from agent_utilities.core.capability_contract import (  # noqa: E402
    DEFAULT_TOOL_DELEGATE as _DEFAULT_DELEGATE,
)
_GATEWAY_OUTPUT_LIMIT = 12_000
_GATEWAY_MERMAID_LIMIT = 8_000
_GATEWAY_TRACE_TOOL_LIMIT = 32


class DynamicWorkflowExecutionPayload(TypedDict, total=False):
    """Serialized result contract for the GraphOS dynamic-workflow surface."""

    status: str
    output: Any
    workflow_run_id: str
    run_id: str
    session_id: str
    trace_ref: str
    backend: str
    upstream_version: str
    script_evidence: list[dict[str, Any]]
    child_runs: list[dict[str, Any]]
    usage: dict[str, int]
    fallback_used: bool
    fallback_reason: str


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…[truncated {len(text) - limit} characters]"


def _search_hit_properties(hit: dict[str, Any]) -> dict[str, Any]:
    node = hit.get("node")
    if not isinstance(node, dict):
        return dict(hit)
    return {**node, **hit}


def _search_hit_kind(hit: dict[str, Any]) -> str:
    """Classify one ``search_hybrid`` hit's capability kind.

    Delegates to the shared, table-driven
    :func:`~agent_utilities.core.capability_contract.capability_kind_from_node`
    (CONCEPT:AU-KG.retrieval.unified-capability-contract) — the same
    classifier ``find``/``find_tools`` use — so a ``Tool`` node resolves to
    ``"tool"`` here too instead of being silently dropped as unclassified.
    """
    from agent_utilities.core.capability_contract import capability_kind_from_node

    props = _search_hit_properties(hit)
    node_type = str(
        props.get("node_type")
        or props.get("type")
        or props.get("label")
        or props.get("kind")
        or ""
    )
    resource_type = str(props.get("resource_type") or "")
    node_id = str(props.get("id") or "")
    return capability_kind_from_node(
        node_type=node_type, resource_type=resource_type, node_id=node_id
    )


def _search_hit_score(hit: dict[str, Any], rank: int) -> float:
    for key in ("score", "_score", "similarity", "confidence"):
        try:
            value = hit.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 1.0 / (rank + 1)


def _approval_request(payload: Any) -> dict[str, Any] | None:
    """Extract the bounded approval handle from a delegated result."""

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None

    status = str(payload.get("status") or "").casefold()
    approval_id = payload.get("approval_id")
    required = bool(payload.get("approval_required")) or bool(approval_id)
    if status in {"blocked_on_approval", "suspended", "pending_approval"}:
        required = True
    if required:
        return {
            "required": True,
            "approval_id": str(approval_id or "") or None,
            "status": status or "pending",
            "reason": _bounded_text(
                payload.get("reason") or payload.get("error") or "", 500
            )
            or None,
        }

    for value in payload.values():
        if isinstance(value, dict | list):
            nested = _approval_request_from_collection(value)
            if nested is not None:
                return nested
    return None


def _approval_request_from_collection(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _approval_request(value)
    if isinstance(value, list):
        for item in value[:_GATEWAY_TRACE_TOOL_LIMIT]:
            found = _approval_request(item)
            if found is not None:
                return found
    return None


class Orchestrator:
    """Centralized Orchestration Manager.

    Provides dispatch, execution, compilation, and security capabilities for
    Graph-OS agent orchestration, replacing scattered scripts and wrappers.
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine
        self.compiler = WorkflowCompiler(self.engine)
        self.scanner = PromptInjectionScanner()

    def _scan_task(self, task: str) -> None:
        """Scan a task for prompt injection or malicious intent.

        Uses the scanner's real ``scan_text`` API (pure regex, microseconds). The
        previous ``hasattr(self.scanner, "analyze")`` guard always evaluated False —
        ``PromptInjectionScanner`` exposes ``scan_text``/``scan_conversation``, never
        ``analyze`` — so this gate silently never fired (dead code).
        """
        result = self.scanner.scan_text(task)
        if result.is_malicious:
            raise ValueError(
                "Security Alert: Task rejected due to detected prompt "
                f"injection/threat. Details: {result.explanation}"
            )

    async def dispatch_task(
        self, task: str, dependencies: list[str] | None = None
    ) -> str:
        """Submit an orchestrator assignment to the sole WorkItem authority."""
        self._scan_task(task)
        job_id = f"orch-{uuid.uuid4().hex}"
        from agent_utilities.knowledge_graph.core.session import resolve_session
        from agent_utilities.orchestration.work_item import (
            submit_orchestrator_work_item,
        )

        session = resolve_session(required_scope="kg:write")
        submit_orchestrator_work_item(
            getattr(self.engine, "_work_item_engine", self.engine),
            job_id,
            tenant=session.tenant,
            description=task,
            depends_on=dependencies or [],
        )
        logger.info(f"Dispatched task {job_id}")
        return job_id

    def get_task_status(self, job_id: str) -> dict[str, Any]:
        """Read the authoritative WorkItem for a dispatched assignment."""
        from agent_utilities.orchestration import work_item as _wi

        view = getattr(self.engine, "_work_item_engine", self.engine)
        item = _wi.get_work_item(view, _wi.orchestrator_work_item_id(job_id))
        if item is None:
            return {"status": "not_found", "error": f"Job {job_id} not found"}
        return item

    def get_run_trace(self, run_id: str) -> dict[str, Any]:
        """Fetch the REAL ``:RunTrace`` + its ``:ToolCall`` provenance for a delegated run.

        CONCEPT:AU-ORCH.execution.run-trace-status-tool — a delegated
        ``execute_agent``/``execute_workflow`` run's provenance is a ``:RunTrace``
        node (``agent_runner._record_execution_trace``, ORCH-1.21) plus ``:ToolCall``
        children linked by the canonical ``USED_TOOL`` edge
        (``agent_runner._persist_tool_calls``,
        KG-2.296) — a privacy-safe id namespace (``trace:pref_run_<digest>``) that
        :meth:`get_task_status` never looked at. So a caller holding the ``run_id`` the
        MCP ``execute_agent``/``execute_workflow`` response hands back (ORCH-1.97's
        ``run_id``/``session_id`` handle) had NO way to query what that run actually
        did: ``status`` reported ``not_found`` for a run that really executed, with
        real output and tool calls already sitting in the graph. This reads the
        RunTrace node directly (by ``run_id`` or its opaque canonical trace id) and
        every ``ToolCall`` it made, in call order, so the caller sees the run's true
        status/output/duration AND each tool call's name/args/result/status — not an
        empty shell. Native Pydantic Graph runs also return their complete persisted
        topology/version/runtime/transition/checkpoint evidence; the stored transition
        JSON is decoded to the public structured-list form.
        """
        trace_id = canonical_trace_id(run_id)
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            return {
                "status": "not_found",
                "run_id": run_id,
                "error": "no KG backend active",
            }
        try:
            rows = backend.execute(
                "MATCH (t:RunTrace {id: $tid}) RETURN t.status AS status, "
                "t.attribution_ref AS attribution_ref, t.task AS task, t.timestamp AS timestamp, "
                "t.duration_ms AS duration_ms, t.result_preview AS result_preview, "
                "t.error AS error, t.execution_mode AS execution_mode, "
                "t.graph_evidence_schema_version AS graph_evidence_schema_version, "
                "t.graph_topology AS graph_topology, "
                "t.graph_topology_digest AS graph_topology_digest, "
                "t.graph_version_digest AS graph_version_digest, "
                "t.graph_runtime_version AS graph_runtime_version, "
                "t.graph_node_sequence AS graph_node_sequence, "
                "t.graph_transition_sequence AS graph_transition_sequence, "
                "t.graph_transition_count AS graph_transition_count, "
                "t.graph_checkpoint_ids AS graph_checkpoint_ids, "
                "t.graph_resume_supported AS graph_resume_supported, "
                "t.skill_ref AS skill_ref, "
                "t.server_ref AS server_ref, t.model_ref AS model_ref, "
                "t.model_class AS model_class, "
                "t.skill_instruction_digest AS skill_instruction_digest, "
                "t.event_sequence AS event_sequence",
                {"tid": trace_id},
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "not_found",
                "run_id": run_id,
                "error": f"RunTrace query failed: {exc}",
            }
        if not rows:
            return {"status": "not_found", "run_id": run_id}
        trace: dict[str, Any] = dict(rows[0])
        transition_sequence = trace.get("graph_transition_sequence")
        if isinstance(transition_sequence, str):
            decoded_transitions: Any = None
            try:
                decoded_transitions = json.loads(transition_sequence)
            except (TypeError, ValueError):
                decoded_transitions = None
            if isinstance(decoded_transitions, list):
                trace["graph_transition_sequence"] = decoded_transitions
        try:
            tc_rows = backend.execute(
                # NOTE: the node carrying the ``{id: $tid}`` property-map filter MUST be
                # bound to a variable (``t:RunTrace``, not the anonymous ``:RunTrace``) —
                # the epistemic-graph backend's fast-path Cypher parser silently
                # under-matches (returns zero rows, no error/warning) an anonymous node
                # with an inline property map, even though the identical pattern with a
                # bound-but-unused variable name matches correctly. Verified live; this
                # bit a pre-existing query in ``agent_digital_twin.py`` too (fixed
                # alongside this one).
                f"MATCH (t:RunTrace {{id: $tid}})-[:{TRACE_USED_TOOL_EDGE}]->(tc:ToolCall) "
                "RETURN tc.sequence AS sequence, tc.tool_name AS tool_name, "
                "tc.args AS args, tc.result AS result_preview, "
                "tc.status AS status, tc.error AS error "
                "ORDER BY tc.sequence ASC",
                {"tid": trace_id},
            )
        except Exception:  # noqa: BLE001 — tool-call listing is best-effort
            tc_rows = []
        tool_calls = [dict(r) for r in (tc_rows or [])]
        trace["run_id"] = run_id
        trace["trace_id"] = trace_id
        trace["tool_calls"] = tool_calls
        trace["tool_call_count"] = len(tool_calls)
        return trace

    def get_tool_calls_for_target(self, target_id: str) -> dict[str, Any]:
        """Entity-anchored reverse-index: every ``:ToolCall`` that acted on ``target_id``.

        CONCEPT:AU-KG.audit.tool-call-acted-on-reverse-index (G23, audit-trail closure) —
        :meth:`get_run_trace` answers "what did run X do", organized by run; this answers
        the complementary "what happened to entity X", organized by target. It walks the
        ``:ToolCall -[:ACTED_ON]-> <target>`` edge :func:`agent_utilities.orchestration.
        agent_runner._persist_tool_calls` writes (best-effort) whenever a tool call's
        sanitized args carry a recognizable id key (``node_id``/``id``/``entity_id``/
        ``target_id``/``ticket_id``/``incident_id``/``spec_id``) that resolves to an
        existing graph node — so an auditor can reconstruct the full step-by-step history
        against any entity (an ``:Incident``, ``:SpecProposal``, ``:Ticket``, a governance
        gate node, …), in call order, each carrying its agent/tool/args/result/status.

        Also attaches a best-effort :meth:`~agent_utilities.knowledge_graph.core.
        graph_compute.GraphComputeEngine.audit_verify` snapshot (``audit`` key) so the
        reconstructed history comes with the engine's cryptographic tamper-evidence
        guarantee, not just the graph read — ``None`` when the engine build/config
        doesn't support it (see :meth:`GraphComputeEngine.audit_verify`).
        """
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            return {
                "target_id": target_id,
                "error": "no KG backend active",
                "tool_calls": [],
            }
        try:
            rows = backend.execute(
                "MATCH (tc:ToolCall)-[:ACTED_ON]->(target {id: $tid}) "
                "RETURN tc.id AS id, tc.run_id AS run_id, tc.agent_name AS agent_name, "
                "tc.server AS server, tc.tool_name AS tool_name, tc.args AS args, "
                "tc.result_preview AS result_preview, tc.error AS error, "
                "tc.status AS status, tc.sequence AS sequence, tc.timestamp AS timestamp "
                "ORDER BY tc.timestamp ASC, tc.sequence ASC",
                {"tid": target_id},
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "target_id": target_id,
                "error": f"reverse-index query failed: {exc}",
                "tool_calls": [],
            }
        tool_calls = [dict(r) for r in (rows or [])]
        audit: dict[str, Any] | None = None
        graph_client = getattr(self.engine, "graph", None)
        if graph_client is not None:
            try:
                audit = graph_client.audit_verify()
            except Exception:  # noqa: BLE001 — verification is a best-effort add-on
                audit = None
        return {
            "target_id": target_id,
            "tool_call_count": len(tool_calls),
            "tool_calls": tool_calls,
            "audit": audit,
        }

    def get_session_runs(self, session_id: str) -> dict[str, Any]:
        """Fetch every ``:RunTrace`` anchored to a ``:Session`` (a multi-step delegation).

        CONCEPT:AU-ORCH.execution.run-trace-status-tool — a compiled ``execute_workflow`` run
        (or any multi-turn ``session_id``-scoped delegation) spans several ``run_agent``
        calls, each recording its OWN ``:RunTrace``, anchored to one ``:Session`` node via
        ``HAS_RUN`` (ORCH-1.97 / session-anchored-collections-native). This aggregates them —
        the workflow/session-level twin of :meth:`get_run_trace` — so a caller holding a
        workflow's ``run_id`` (its ``session_id``) can see every step's real trace + tool
        calls, not just a top-level "completed"/"failed" flag.
        """
        from agent_utilities.security.persistence_privacy import persistence_reference

        session_value = session_id.removeprefix("session:")
        session_ref = (
            session_value
            if session_value.startswith("pref_session_")
            else persistence_reference(
                "session", session_value, namespace="session-continuity"
            )
        )
        sid = f"session:{session_ref}"
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            return {
                "status": "not_found",
                "session_id": session_id,
                "error": "no KG backend active",
            }
        try:
            rows = backend.execute(
                "MATCH (s:Session {id: $sid})-[:HAS_RUN]->(t:RunTrace) "
                "RETURN t.id AS tid",
                {"sid": sid},
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "not_found",
                "session_id": session_id,
                "error": f"Session query failed: {exc}",
            }
        run_ids = [str(r["tid"]) for r in (rows or []) if r.get("tid")]
        runs = [self.get_run_trace(tid) for tid in run_ids]
        return {
            "session_id": session_id,
            "run_count": len(runs),
            "runs": runs,
        }

    def grant_approval(self, job_id: str, approval_status: str) -> str:
        """Grant or deny approval for a pending job."""
        if job_id not in self.engine.graph.nodes:
            return f"Error: job {job_id} not found"
        self.engine.graph.nodes[job_id]["approval_status"] = approval_status
        return f"Job {job_id} approval updated to: {approval_status}"

    async def execute_agent(
        self,
        agent_name: str,
        task: str,
        max_steps: int = 30,
        return_mermaid: bool = False,
        context: str | None = None,
        budget_tokens: int | None = None,
        context_ref: str | None = None,
        allowed_tools: list[str] | None = None,
        cred_ref: str | None = None,
        session_id: str | None = None,
        open_channel: bool = False,
        memento_source: str | None = None,
        execution_profile: str | None = None,
        reasoning_effort: str | None = None,
        model_class: str = "standard",
        response_format: ResponseFormat = "text",
        run_id: str | None = None,
        include_run_summary: bool = False,
        progress_sink: ProgressSink | None = None,
        required_tools: list[str] | None = None,
        skill_name: str | None = None,
        tool_server: str | None = None,
        execution_mode: ExecutionMode = "auto",
        grounding: str = "required",
    ) -> str:
        """Execute a single agent against a task.

        CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid — ``return_mermaid`` forwards to :func:`run_agent` so the MCP
        layer can surface the routed-graph diagram (off by default for internal callers).
        CONCEPT:AU-ORCH.session.invoker-agent-handoff — ``context`` is the invoking agent's curated context, threaded to
        the spawned agent's prompt (budgeted to the model window).
        CONCEPT:AU-ECO.messaging.universal-graph-agent — ``memento_source`` scopes which compressed-memory stream primes
        the run (defaults to ``agent_name``); a session-scoped caller passes its session key
        so successive turns of one conversation share continuity through the core memory.
        CONCEPT:AU-ORCH.execution.chat-profile-timeouts — ``execution_profile`` ("chat" vs the default "task") selects the
        per-node timeout budget. A chat-budget profile bounds each LLM round to tens of
        seconds (not 300 s) so a slow/degraded backend fails fast inside the chat budget;
        the messaging reply path passes ``"chat"``.
        CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — ``run_id``/
        ``include_run_summary`` forward to :func:`run_agent` verbatim: a caller (the messaging
        router) that wants a translated route/outcome/failure summary — and a ``trace_ref``
        that survives even a hard cancellation of this call — opts in via these two.
        ``progress_sink`` forwards the SAME way: an optional async sink that receives the
        checkpoint-by-checkpoint :class:`~agent_utilities.orchestration.agent_runner.ProgressEvent`
        stream so a long delegation is transparent step-by-step. Default ``None`` is a strict
        no-op (the run is byte-for-byte unchanged); every emission is fire-and-forget and
        exception-isolated, so a slow/failing sink can never stall or fail the run.
        CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — ``grounding`` scopes the
        mandatory evidence-compilation policy for every model call this run makes
        (``"required"`` — the default — fails the run closed on a compile timeout/error
        or a retrieval-quality-gate failure rather than silently answering ungrounded;
        ``"best_effort"``/``"none"`` opt into degraded operation, marked explicitly in
        the model messages, the RunTrace, and the OTel span).
        """
        response_format = validate_response_format(response_format)
        execution_mode = validate_execution_mode(execution_mode)
        allowed_tools, required_tools = validate_tool_contract(
            allowed_tools, required_tools
        )
        self._scan_task(task)
        logger.info(f"Executing agent {agent_name} for task: {task[:50]}...")
        # CONCEPT:AU-ORCH.scheduling.resource-priority-edict — a delegation is
        # ORCHESTRATION-tier work (rank below a live interactive read, above background
        # ingestion). Tag the run so run_agent's tool loop and its RAG reads carry the Orch
        # QoS class — which reserves the very top INTERACTIVE lane for the caller's OWN direct
        # reads (a live-client graph_orchestrate call entered INTERACTIVE via the MCP dispatch
        # is downgraded to ORCHESTRATION here). The ONE exception: a delegation spawned BY a
        # background job (the autonomous loop, ambient BACKGROUND_INGESTION) is NEVER upgraded —
        # it must keep yielding — so its class passes through unchanged.
        from agent_utilities.core.resource_priority import (
            PriorityClass,
            current_priority,
            priority_scope,
        )

        _delegation_prio = (
            PriorityClass.BACKGROUND_INGESTION
            if current_priority() is PriorityClass.BACKGROUND_INGESTION
            else PriorityClass.ORCHESTRATION
        )
        from agent_utilities.core.contextual_model import use_grounding_policy

        with priority_scope(_delegation_prio), use_grounding_policy(grounding):
            result = await run_agent(
                agent_name=agent_name,
                task=task,
                engine=self.engine,
                max_steps=max_steps,
                return_mermaid=return_mermaid,
                context=context,
                budget_tokens=budget_tokens,
                context_ref=context_ref,
                allowed_tools=allowed_tools,
                required_tools=required_tools,
                skill_name=skill_name,
                tool_server=tool_server,
                execution_mode=execution_mode,
                cred_ref=cred_ref,
                session_id=session_id,
                open_channel=open_channel,
                memento_source=memento_source,
                execution_profile=execution_profile,
                reasoning_effort=reasoning_effort,
                model_class=model_class,
                response_format=response_format,
                run_id=run_id,
                include_run_summary=include_run_summary,
                progress_sink=progress_sink,
            )
        return result

    def resolve_capability(self, task: str, agent_name: str = "") -> dict[str, Any]:
        """Resolve a task to an ingested skill/workflow, or the default expert.

        CONCEPT:AU-ORCH.execution.execution-seam-closure — the GraphOS gateway
        resolves against the KG's hybrid index before local-vLLM execution. An
        explicit agent remains authoritative; an unresolved task routes to the
        KG-bound ``agent-utilities-expert`` instead of exposing skill bodies or
        fleet tool schemas to the calling harness.
        """
        if agent_name.strip():
            return {
                "kind": "agent",
                "name": agent_name.strip(),
                "id": "",
                "score": 1.0,
                "source": "caller",
                "alternatives": [],
            }

        try:
            hits = list(self.engine.search_hybrid(task, top_k=24) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphOS capability resolution degraded: %s", exc)
            hits = []

        candidates: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits):
            if not isinstance(hit, dict):
                continue
            kind = _search_hit_kind(hit)
            if not kind:
                continue
            props = _search_hit_properties(hit)
            if bool(props.get("disabled")):
                continue
            name = str(props.get("name") or "").strip()
            if not name:
                continue
            candidates.append(
                {
                    "kind": kind,
                    "name": name,
                    "id": str(props.get("id") or ""),
                    "score": _search_hit_score(hit, rank),
                    "source": "kg_hybrid",
                    # Owning MCP server for a "tool" (or a fleet-served
                    # "skill") kind — "" for a purely local skill/agent.
                    # Carried so a resolved capability can be bound without
                    # the caller re-deriving it (CONCEPT:AU-KG.retrieval.unified-capability-contract).
                    "server": str(props.get("mcp_server") or ""),
                }
            )

        if candidates:
            from agent_utilities.knowledge_graph.core.secured_reads import permit

            candidate_ids = [
                str(candidate["id"]) for candidate in candidates if candidate["id"]
            ]
            try:
                permitted_ids = set(permit(candidate_ids))
            except PermissionError as exc:
                logger.warning("GraphOS capability permission filter denied: %s", exc)
                permitted_ids = set()
            candidates = [
                candidate
                for candidate in candidates
                if candidate["id"] and candidate["id"] in permitted_ids
            ]

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        if candidates:
            chosen = dict(candidates[0])
            chosen["alternatives"] = [
                {
                    "kind": candidate["kind"],
                    "name": candidate["name"],
                    "score": candidate["score"],
                }
                for candidate in candidates[1:4]
            ]
            return chosen

        return {
            "kind": "agent",
            "name": _DEFAULT_DELEGATE,
            "id": "",
            "score": 0.0,
            "source": "default",
            "alternatives": [],
        }

    def _run_provenance(self, run_id: str) -> dict[str, Any]:
        trace = self.get_run_trace(run_id)
        calls = trace.get("tool_calls") if isinstance(trace, dict) else []
        if not isinstance(calls, list):
            calls = []
        return {
            "run_id": run_id,
            "trace_ref": trace.get("trace_id") if isinstance(trace, dict) else None,
            "status": trace.get("status") if isinstance(trace, dict) else None,
            "execution_mode": trace.get("execution_mode")
            if isinstance(trace, dict)
            else None,
            "duration_ms": trace.get("duration_ms")
            if isinstance(trace, dict)
            else None,
            "skill_ref": trace.get("skill_ref") if isinstance(trace, dict) else None,
            "server_ref": trace.get("server_ref") if isinstance(trace, dict) else None,
            "model_ref": trace.get("model_ref") if isinstance(trace, dict) else None,
            "model_class": trace.get("model_class")
            if isinstance(trace, dict)
            else None,
            "tool_call_count": int(
                trace.get("tool_call_count", len(calls))
                if isinstance(trace, dict)
                else len(calls)
            ),
            "tool_calls": [
                {
                    "sequence": call.get("sequence"),
                    "tool_name": call.get("tool_name"),
                    "status": call.get("status"),
                }
                for call in calls[:_GATEWAY_TRACE_TOOL_LIMIT]
                if isinstance(call, dict)
            ],
        }

    def _workflow_provenance(self, session_id: str) -> dict[str, Any]:
        session = self.get_session_runs(session_id)
        runs = session.get("runs") if isinstance(session, dict) else []
        if not isinstance(runs, list):
            runs = []
        compact_runs = []
        total_tool_calls = 0
        for run in runs[:_GATEWAY_TRACE_TOOL_LIMIT]:
            if not isinstance(run, dict):
                continue
            total_tool_calls += int(run.get("tool_call_count") or 0)
            compact_runs.append(
                {
                    "run_id": run.get("run_id"),
                    "trace_ref": run.get("trace_id"),
                    "status": run.get("status"),
                    "tool_call_count": run.get("tool_call_count", 0),
                }
            )
        return {
            "session_id": session_id,
            "run_count": int(
                session.get("run_count", len(runs))
                if isinstance(session, dict)
                else len(runs)
            ),
            "tool_call_count": total_tool_calls,
            "runs": compact_runs,
        }

    async def execute_capability(
        self,
        *,
        task: str,
        agent_name: str = "",
        skill_name: str = "",
        tool_server: str = "",
        execution_mode: ExecutionMode = "auto",
        max_steps: int = 30,
        context: str | None = None,
        budget_tokens: int | None = None,
        context_ref: str | None = None,
        allowed_tools: list[str] | None = None,
        required_tools: list[str] | None = None,
        cred_ref: str | None = None,
        open_channel: bool = False,
        reasoning_effort: str | None = None,
        model_class: str = "standard",
        response_format: ResponseFormat = "text",
        grounding: str = "required",
    ) -> dict[str, Any]:
        """Resolve and execute one task through the bounded GraphOS skill gateway.

        ``grounding`` forwards to :meth:`execute_agent` (CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract)
        for the ``agent``/``skill`` resolution path; a resolved ``workflow`` runs
        through :meth:`execute_workflow` instead, which is unaffected (each step's own
        model call still defaults to ``"required"`` — the process-wide default — since
        no scope is opened for it here).
        """
        execution_mode = validate_execution_mode(execution_mode)
        allowed_tools, required_tools = validate_tool_contract(
            allowed_tools, required_tools
        )
        if agent_name.strip() and skill_name.strip():
            raise ValueError("agent_name and skill_name are mutually exclusive")
        if tool_server.strip() and not skill_name.strip():
            raise ValueError("tool_server requires skill_name")
        validate_pydantic_graph_contract(
            execution_mode,
            skill_name=skill_name,
            tool_server=tool_server,
            allowed_tools=allowed_tools,
        )
        self._scan_task(task)
        target = await asyncio.to_thread(
            self.resolve_capability, task, agent_name=skill_name or agent_name
        )
        if skill_name:
            target = {
                **target,
                "kind": "skill",
                "name": skill_name,
                "source": "caller_skill",
            }

        if target["kind"] == "workflow":
            from agent_utilities.knowledge_graph.core.workflow_gate import (
                gate_workflow_execution,
            )

            gate = await asyncio.to_thread(
                gate_workflow_execution, self.engine, target["name"]
            )
            if gate.get("allowed") is not True:
                return {
                    "output": "Workflow execution was refused by the ontology/ACL gate.",
                    "run_id": None,
                    "mermaid": None,
                    "resolution": target,
                    "provenance": {
                        "workflow_id": gate.get("workflow_id"),
                        "violations": gate.get("violations", [])[:10],
                    },
                    "approval_request": None,
                }
            result = await self.execute_workflow(
                workflow_id=target["name"],
                task=task,
                max_steps=max_steps,
            )
            run_id = str(result.get("run_id") or result.get("session_id") or "")
            provenance = await asyncio.to_thread(self._workflow_provenance, run_id)
            return {
                "output": _bounded_text(
                    json.dumps(result, default=str), _GATEWAY_OUTPUT_LIMIT
                ),
                "run_id": run_id or None,
                "mermaid": _bounded_text(
                    result.get("mermaid") or "", _GATEWAY_MERMAID_LIMIT
                )
                or None,
                "resolution": target,
                "provenance": provenance,
                "approval_request": _approval_request(result),
            }

        # A ``target`` resolved (not caller-supplied — an explicit skill_name
        # was already folded into ``target["kind"] == "skill"`` above) to a
        # bare ``Tool`` node binds through the SAME Capability contract a
        # ranked ``find``/``find_tools`` result would (CONCEPT:AU-KG.retrieval.unified-capability-contract):
        # run the default expert scoped to just that one fleet tool, rather
        # than the wrong-shaped ``agent_name=<tool name>``.
        call_agent_name = target["name"]
        call_tool_server = tool_server or None
        call_allowed_tools = allowed_tools
        call_skill_name = skill_name or None
        if target["kind"] == "tool":
            from agent_utilities.core.capability_contract import Capability

            binding = Capability(
                kind="tool",
                id=str(target.get("id") or ""),
                name=target["name"],
                server=str(target.get("server") or "") or None,
            ).to_binding()
            call_agent_name = _DEFAULT_DELEGATE
            call_tool_server = binding.get("tool_server") or call_tool_server
            call_allowed_tools = binding.get("allowed_tools") or call_allowed_tools
            # ``run_agent`` enforces both "tool_server requires skill_name" AND
            # "skill_name must match the dispatched agent_name", so the delegate
            # has to be named on BOTH keywords. Leaving skill_name empty here
            # made every auto-resolved Tool raise ValueError inside run_agent --
            # and every fleet Tool node carries a non-empty mcp_server, so this
            # was every real resolution of this kind, not an edge case.
            call_skill_name = _DEFAULT_DELEGATE

        raw = await self.execute_agent(
            agent_name=call_agent_name,
            skill_name=call_skill_name,
            tool_server=call_tool_server,
            execution_mode=execution_mode,
            task=task,
            max_steps=max_steps,
            return_mermaid=True,
            context=context,
            budget_tokens=budget_tokens,
            context_ref=context_ref,
            allowed_tools=call_allowed_tools,
            required_tools=required_tools,
            cred_ref=cred_ref,
            open_channel=open_channel,
            reasoning_effort=reasoning_effort,
            model_class=model_class,
            response_format=response_format,
            include_run_summary=True,
            grounding=grounding,
        )
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = {"output": str(raw)}
        if not isinstance(payload, dict):
            payload = {"output": str(payload)}
        run_id = str(payload.get("run_id") or "")
        provenance = (
            await asyncio.to_thread(self._run_provenance, run_id) if run_id else {}
        )
        # Defense in depth for the public gateway envelope. ``run_agent`` rejects
        # ungrounded tool-required executions before recording the trace, but a
        # malformed/legacy runner response must not be re-labelled as success here.
        if (allowed_tools or required_tools) and int(
            provenance.get("tool_call_count") or 0
        ) == 0:
            refusal = (
                "Tool-required execution produced no recorded ToolCall provenance; "
                "refusing an ungrounded result."
            )
            payload["output"] = refusal
            summary = payload.get("run_summary")
            payload["run_summary"] = {
                **(summary if isinstance(summary, dict) else {}),
                "outcome": "degraded",
                "failure": {"category": "ungrounded_tool_execution", "raw": refusal},
            }
            provenance = {**provenance, "status": "degraded"}
        return {
            "output": _bounded_text(payload.get("output"), _GATEWAY_OUTPUT_LIMIT),
            "run_id": run_id or None,
            "mermaid": _bounded_text(
                payload.get("mermaid") or "", _GATEWAY_MERMAID_LIMIT
            )
            or None,
            "channel_id": payload.get("channel_id"),
            "run_summary": payload.get("run_summary"),
            "execution_evidence": payload.get("execution_evidence"),
            "resolution": target,
            "provenance": provenance,
            "approval_request": _approval_request(payload),
        }

    async def compile_workflow(self, name: str, task: str) -> str:
        """Compile a workflow topology from a natural language task."""
        self._scan_task(task)
        logger.info(f"Compiling workflow {name} for task: {task[:50]}...")
        # WorkflowCompiler.compile_and_store generally returns the workflow/topology ID
        try:
            workflow_id = await self.compiler.compile_and_store(
                name=name, description=task
            )
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to compile workflow: {e}")
            raise

    async def execute_workflow(
        self, workflow_id: str, task: str = "", max_steps: int = 30
    ) -> dict[str, Any]:
        """Execute a compiled workflow by running its STORED step-DAG.

        CONCEPT:AU-ORCH.execution.execution-seam-closure — close the execution seam. This previously constructed a
        generic ``AgentOrchestrationEngine`` whose no-completion-state path ran ONE
        ``dynamic_worker`` agent and never loaded the ingested
        ``WorkflowDefinition``/``WorkflowStep`` DAG — so a stored/ingested workflow
        (the KG-2.97 ``WorkflowStore`` shape) was dispatchable but never executed.

        It now routes to the real :class:`WorkflowRunner` (ORCH-1.24), which
        ``load_workflow(name)`` → builds dependency waves → runs each step on the
        local LLM. The SHACL+ACL ontology gate (ORCH-1.42) still runs upstream in
        the ``graph_workflows`` handler before this is called, so governance stays
        in the path. Returns the ``WorkflowResult`` as a dict carrying the ``run_id``
        handle (the session id) so a delegated workflow run is trackable (ORCH-1.97).
        """
        if task:
            self._scan_task(task)

        logger.info(f"Executing workflow {workflow_id} via WorkflowRunner...")
        from agent_utilities.workflows.runner import WorkflowRunner

        runner = WorkflowRunner(max_steps_per_agent=max_steps)
        result = await runner.execute_by_name(
            workflow_name=workflow_id,
            engine=self.engine,
            task=task or None,
        )
        payload = result.to_dict()
        # ORCH-1.97 — surface a stable run handle for the delegated workflow run.
        payload["run_id"] = result.session_id
        return payload

    async def execute_dynamic_workflow(
        self,
        workflow_id: str,
        task: str = "",
        max_steps: int = 30,
        *,
        max_agent_calls: int = 50,
        max_concurrency: int = 8,
        budget_tokens: int | None = None,
        model_class: str = "standard",
        unavailable_fallback: str = "error",
        orchestrator_model: Any | None = None,
    ) -> DynamicWorkflowExecutionPayload:
        """Execute a stored catalog with upstream Harness DynamicWorkflow.

        The Harness parent owns only its sandboxed ``run_workflow`` tool. Every
        catalog call re-enters :meth:`execute_agent`, so the same KG resolution,
        tenant policy, skill/tool contract, model-class router, budgets, and
        RunTrace/ToolCall persistence used by ``graph_orchestrate`` remain
        authoritative.

        ``unavailable_fallback`` is explicit: ``"error"`` fails when the optional
        Harness extra or a required upstream seam is unavailable;
        ``"stored_dag"`` runs the ordinary :class:`WorkflowRunner` instead. A
        runtime/model/tool failure never silently changes engines.
        """

        if task:
            self._scan_task(task)
        if unavailable_fallback not in {"error", "stored_dag"}:
            raise ValueError("unavailable_fallback must be 'error' or 'stored_dag'")
        if model_class not in {"economy", "standard"}:
            raise ValueError("model_class must be economy or standard")

        from agent_utilities.capabilities.governed_dynamic_workflow import (
            DynamicWorkflowUnavailableError,
            GovernedDynamicWorkflow,
        )
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        try:
            plan = await asyncio.to_thread(
                WorkflowStore(self.engine).load_workflow, workflow_id
            )
            if plan is None:
                raise ValueError(f"Workflow '{workflow_id}' not found in KG or catalog")
            workflow = GovernedDynamicWorkflow.from_graph_plan(
                plan,
                name=workflow_id,
                query=task or f"Execute the governed workflow {workflow_id}",
                max_agent_calls=max_agent_calls,
                max_concurrency=max_concurrency,
                budget_tokens=budget_tokens,
                max_steps=max_steps,
            )
            # Fail before constructing a provider model when the optional
            # Harness runtime is absent. This also validates the pinned API.
            workflow.build_upstream_capability(self)

            if orchestrator_model is None:
                from agent_utilities.core.model_factory import create_model
                from agent_utilities.orchestration.agent_runner import (
                    _configured_model_for_class,
                )

                selected = _configured_model_for_class(model_class)
                orchestrator_model = create_model(model_id=selected.id)
            result = await workflow.execute(
                self,
                orchestrator_model=orchestrator_model,
            )
        except DynamicWorkflowUnavailableError as exc:
            if unavailable_fallback != "stored_dag":
                raise
            fallback = await self.execute_workflow(
                workflow_id=workflow_id,
                task=task,
                max_steps=max_steps,
            )
            return cast(
                DynamicWorkflowExecutionPayload,
                {
                    **fallback,
                    "backend": "stored_dag",
                    "fallback_used": True,
                    "fallback_reason": type(exc).__name__,
                },
            )
        return cast(
            DynamicWorkflowExecutionPayload,
            result.model_dump(mode="json"),
        )
