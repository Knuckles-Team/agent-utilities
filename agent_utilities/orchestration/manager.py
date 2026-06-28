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
    ) -> str:
        """Execute a single agent against a task.

        CONCEPT:ORCH-1.37 — ``return_mermaid`` forwards to :func:`run_agent` so the MCP
        layer can surface the routed-graph diagram (off by default for internal callers).
        CONCEPT:ORCH-1.39 — ``context`` is the invoking agent's curated context, threaded to
        the spawned agent's prompt (budgeted to the model window).
        CONCEPT:ECO-4.78 — ``memento_source`` scopes which compressed-memory stream primes
        the run (defaults to ``agent_name``); a session-scoped caller passes its session key
        so successive turns of one conversation share continuity through the core memory.
        CONCEPT:ORCH-1.62 — ``execution_profile`` ("chat" vs the default "task") selects the
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

        CONCEPT:ORCH-1.95 — close the execution seam. This previously constructed a
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
