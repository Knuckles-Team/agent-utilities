import logging
import uuid
from typing import Any

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler
from agent_utilities.orchestration.agent_runner import run_agent
from agent_utilities.security.threat_defense_engine import PromptInjectionScanner

logger = logging.getLogger(__name__)


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
        """Dispatch an asynchronous task for execution."""
        self._scan_task(task)
        job_id = f"orch-{uuid.uuid4().hex[:8]}"
        self.engine.add_node(
            job_id,
            "Task",
            properties={
                "status": "pending",
                "description": task,
                "dependencies": dependencies or [],
            },
        )
        logger.info(f"Dispatched task {job_id}")
        return job_id

    def get_task_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a dispatched task."""
        if job_id not in self.engine.graph.nodes:
            return {"status": "not_found", "error": f"Job {job_id} not found"}
        return self.engine.graph.nodes[job_id]

    def get_run_trace(self, run_id: str) -> dict[str, Any]:
        """Fetch the REAL ``:RunTrace`` + its ``:ToolCall`` provenance for a delegated run.

        CONCEPT:AU-ORCH.execution.run-trace-status-tool — ``graph_orchestrate(action="status")``
        previously only read ``:Task`` nodes written by :meth:`dispatch_task`. A delegated
        ``execute_agent``/``execute_workflow`` run's provenance is instead a ``:RunTrace``
        node (``agent_runner._record_execution_trace``, ORCH-1.21) plus ``:ToolCall``
        children linked by ``MADE_TOOL_CALL`` (``agent_runner._persist_tool_calls``,
        KG-2.296) — a completely different id namespace (``trace:run:<hex>``) that
        :meth:`get_task_status` never looked at. So a caller holding the ``run_id`` the
        MCP ``execute_agent``/``execute_workflow`` response hands back (ORCH-1.97's
        ``run_id``/``session_id`` handle) had NO way to query what that run actually
        did: ``status`` reported ``not_found`` for a run that really executed, with
        real output and tool calls already sitting in the graph. This reads the
        RunTrace node directly (by ``run_id`` or its ``trace:<run_id>`` node id) and
        every ``ToolCall`` it made, in call order, so the caller sees the run's true
        status/output/duration AND each tool call's name/args/result/status — not an
        empty shell.
        """
        trace_id = run_id if run_id.startswith("trace:") else f"trace:{run_id}"
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
                "t.agent_name AS agent_name, t.task AS task, t.timestamp AS timestamp, "
                "t.duration_ms AS duration_ms, t.result_preview AS result_preview, "
                "t.error AS error, t.skill_used AS skill_used, "
                "t.bound_server AS bound_server",
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
                "MATCH (t:RunTrace {id: $tid})-[:MADE_TOOL_CALL]->(tc:ToolCall) "
                "RETURN tc.sequence AS sequence, tc.tool_name AS tool_name, "
                "tc.args AS args, tc.result_preview AS result_preview, "
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
        sid = (
            session_id if session_id.startswith("session:") else f"session:{session_id}"
        )
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
        """
        self._scan_task(task)
        logger.info(f"Executing agent {agent_name} for task: {task[:50]}...")
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
            cred_ref=cred_ref,
            session_id=session_id,
            open_channel=open_channel,
            memento_source=memento_source,
            execution_profile=execution_profile,
            reasoning_effort=reasoning_effort,
        )
        return result

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
        the ``graph_orchestrate`` handler before this is called, so governance stays
        in the path. Returns the ``WorkflowResult`` as a dict carrying the ``run_id``
        handle (the session id) so a delegated workflow run is trackable (ORCH-1.97).
        """
        if task:
            self._scan_task(task)

        logger.info(f"Executing workflow {workflow_id} via WorkflowRunner...")
        from agent_utilities.workflows.runner import WorkflowRunner

        runner = WorkflowRunner()
        result = await runner.execute_by_name(
            workflow_name=workflow_id,
            engine=self.engine,
            task=task or None,
        )
        payload = result.to_dict()
        # ORCH-1.97 — surface a stable run handle for the delegated workflow run.
        payload["run_id"] = result.session_id
        return payload
