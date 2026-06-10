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
        """Scan a task for prompt injection or malicious intent."""
        # Check if analyze method exists and returns a dict/object indicating threat
        if hasattr(self.scanner, "analyze"):
            result = self.scanner.analyze(task)
            # Handle possible variations of the result (dict vs object)
            if isinstance(result, dict) and result.get("is_threat"):
                raise ValueError(
                    f"Security Alert: Task rejected due to detected prompt injection/threat. Details: {result.get('reason')}"
                )
            elif hasattr(result, "is_threat") and result.is_threat:
                raise ValueError(
                    f"Security Alert: Task rejected due to detected prompt injection/threat. Details: {getattr(result, 'reason', 'Unknown')}"
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
    ) -> str:
        """Execute a single agent against a task.

        CONCEPT:ORCH-1.37 — ``return_mermaid`` forwards to :func:`run_agent` so the MCP
        layer can surface the routed-graph diagram (off by default for internal callers).
        """
        self._scan_task(task)
        logger.info(f"Executing agent {agent_name} for task: {task[:50]}...")
        result = await run_agent(
            agent_name=agent_name,
            task=task,
            engine=self.engine,
            max_steps=max_steps,
            return_mermaid=return_mermaid,
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
    ) -> str:
        """Execute a compiled workflow."""
        if task:
            self._scan_task(task)

        logger.info(f"Executing workflow {workflow_id}...")
        try:
            from agent_utilities.orchestration import AgentOrchestrationEngine

            runner = AgentOrchestrationEngine()
            result = await runner.execute_workflow(workflow_id=workflow_id)
            return str(result)
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise
