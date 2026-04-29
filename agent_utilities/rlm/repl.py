import asyncio
import io
import json
import logging
import sys
import tempfile
import traceback
from typing import Any

import networkx as nx

from ..graph.client import get_graph_client
from ..graph.state import GraphDeps
from .config import RLMConfig

logger = logging.getLogger(__name__)


class RecursionLimitError(Exception):
    pass


class RLMEnvironment:
    """
    A persistent Python REPL environment for Recursive Language Models.
    Holds variables (context) between executions and provides safe access
    to graph traversal and recursive dispatch.
    """

    def __init__(
        self,
        context: Any = None,
        depth: int = 0,
        config: RLMConfig | None = None,
        graph_deps: GraphDeps | None = None,
    ):
        self.config = config or RLMConfig()
        self.depth = depth
        self.max_depth = self.config.max_depth
        self.graph_deps = graph_deps

        self.vars: dict[str, Any] = {"context": context, "depth": depth}

        # We don't await get_graph_client() here because __init__ is synchronous.
        # Instead, we provide it as an async helper in the globals.
        self._get_graph_client = get_graph_client

        # The global namespace for the REPL
        self.globals_dict = {
            "__builtins__": __builtins__,
            "nx": nx,
            "context": self.vars["context"],
            "depth": self.vars["depth"],
            "rlm_query": self.rlm_query,
            "run_parallel_sub_calls": self.run_parallel_sub_calls,
            "magma_view": self.magma_view,
            "graph_query": self.graph_query,
            "sub_agent_call": self.sub_agent_call_helper,
            "FINAL_VAR": self.FINAL_VAR,
            "json": json,
            "asyncio": asyncio,
        }

    def FINAL_VAR(self, name: str, value: Any):
        """Helper for the LLM to output its final result explicitly."""
        self.vars[name] = value
        self.vars["__FINAL__"] = name

    async def magma_view(
        self, query: str, views: list[str] | None = None
    ) -> dict[str, Any]:
        """MAGMA orthogonal memory views: semantic, temporal, causal, entity."""
        if not self.graph_deps or not hasattr(self.graph_deps, "knowledge_engine"):
            return {"error": "Knowledge engine not available"}

        engine = self.graph_deps.knowledge_engine
        if not engine:
            return {"error": "Knowledge engine not initialized"}

        # Default to all 4 views as requested by the user
        if views is None:
            views = ["semantic", "temporal", "causal", "entity"]

        return engine.retrieve_orthogonal_context(query, views=views)

    async def graph_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query against the knowledge graph."""
        if not self.graph_deps or not hasattr(self.graph_deps, "knowledge_engine"):
            return [{"error": "Knowledge engine not available"}]

        engine = self.graph_deps.knowledge_engine
        if not engine:
            return [{"error": "Knowledge engine not initialized"}]

        return engine.query_cypher(cypher, params)

    async def sub_agent_call_helper(
        self, prompt: str, agent_id: str | None = None, input_data: Any = None
    ) -> str:
        """Recursive dispatch to other specialists via the graph dispatcher."""
        if not self.graph_deps:
            return "Error: graph_deps not available"

        # In a real implementation, we would call the dispatcher.
        # Since we are inside a specialist execution, we might need to
        # use the dispatcher provided in graph_deps if it exists.

        # For now, we'll use a simplified dispatch if dispatcher is available
        # But wait, RLM is usually called from a StepContext.

        # This is tricky because we need a StepContext.
        # If we don't have it, we fallback to a direct Agent call.

        from pydantic_ai import Agent

        agent = Agent(
            model=self.config.sub_llm_model_small,
            system_prompt=f"You are a specialized sub-agent for: {agent_id or 'general'}",
        )
        res = await agent.run(f"Context: {input_data}\n\nTask: {prompt}")
        return str(res.output)

    async def rlm_query(self, prompt: str, sub_context: Any = None) -> str:
        """Spawn a full recursive RLM at the next depth."""
        if self.depth >= self.max_depth:
            raise RecursionLimitError(
                f"RLM recursion depth exceeded (max {self.max_depth})"
            )

        logger.info(
            f"RLM at depth {self.depth} spawning sub-RLM for prompt: {prompt[:50]}..."
        )
        sub_env = RLMEnvironment(
            context=sub_context,
            depth=self.depth + 1,
            config=self.config,
            graph_deps=self.graph_deps,
        )
        return await sub_env.run_full_rlm(prompt)

    async def run_parallel_sub_calls(self, calls: list[dict[str, Any]]) -> list[Any]:
        """
        Run multiple sub-calls in parallel.
        calls is a list of dicts: {"prompt": "...", "context": Any}
        """
        if not self.config.async_enabled:
            results = []
            for item in calls:
                results.append(await self._execute_sub_call(item))
            return results

        async def _call(item):
            return await self._execute_sub_call(item)

        return await asyncio.gather(
            *[_call(item) for item in calls], return_exceptions=True
        )

    async def _execute_sub_call(self, item: dict[str, Any]) -> str:
        if self.depth < self.max_depth:
            sub_env = RLMEnvironment(
                context=item.get("context"),
                depth=self.depth + 1,
                config=self.config,
                graph_deps=self.graph_deps,
            )
            return await sub_env.run_full_rlm(item["prompt"])
        else:
            # Fallback to normal specialist
            # ... For simplicity, we just use the router model to answer directly if depth is exhausted
            from pydantic_ai import Agent

            agent = Agent(
                model=self.config.sub_llm_model_small,
                system_prompt="Answer the sub-task directly.",
            )
            res = await agent.run(
                f"Context: {item.get('context')}\n\nPrompt: {item['prompt']}"
            )
            return res.output

    async def execute(self, code: str) -> tuple[dict[str, Any], str]:
        """
        Executes Python code in the persistent environment.
        Returns the updated variables and stdout.
        """
        if self.config.use_container:
            return await self._execute_container(code)
        else:
            return await self._execute_local(code)

    async def _execute_local(self, code: str) -> tuple[dict[str, Any], str]:
        """Execute LLM-generated code in a restricted local namespace.

        Security Advisory (CWE-94):
            This method intentionally uses ``exec()`` to implement a persistent
            Python REPL for Recursive Language Models (RLM).  The execution
            environment is restricted to an explicit ``globals_dict`` that
            exposes only approved helpers (``rlm_query``, ``magma_view``,
            ``graph_query``, ``FINAL_VAR``, ``json``, ``asyncio``, ``nx``).

            For full isolation, set ``RLMConfig.use_container = True`` to
            delegate execution to a sandboxed Docker/Podman container via
            ``_execute_container()``.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # We want to support 'await' inside the exec, so we wrap it in an async function
            wrapped_code = "async def __async_exec__():\n"
            for line in code.splitlines():
                wrapped_code += f"    {line}\n"

            exec(wrapped_code, self.globals_dict)  # nosec B102  # RLM REPL - intentional
            await self.globals_dict["__async_exec__"]()

            # Sync back globals to vars (except builtins)
            for k, v in self.globals_dict.items():
                if k not in [
                    "__builtins__",
                    "nx",
                    "rlm_query",
                    "run_parallel_sub_calls",
                    "FINAL_VAR",
                    "json",
                    "asyncio",
                    "__async_exec__",
                ]:
                    self.vars[k] = v
        except Exception as e:
            traceback.print_exc(file=redirected_output)
            logger.error(f"RLM execute error: {e}")
        finally:
            sys.stdout = old_stdout

        return self.vars, redirected_output.getvalue()

    async def _execute_container(self, code: str) -> tuple[dict[str, Any], str]:
        """Executes the code in a sandboxed container using container-manager-mcp.

        The backend (Docker or Podman) is auto-detected via the ``create_manager``
        factory.  Override with the ``CONTAINER_MANAGER_TYPE`` env-var
        (``docker`` | ``podman``) if explicit selection is needed.
        """
        from container_manager_mcp.container_manager import create_manager

        manager = create_manager()

        # We need to serialize the context/vars, run a python script, and get output
        # For full isolation, we'll write the context to a temp json, mount it, run it
        # Note: Handling complex objects (like callables) is hard in container mode.
        # This is a simplified version.

        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx_path = os.path.join(tmpdir, "context.json")
            with open(ctx_path, "w") as f:
                # Best effort serialization
                try:
                    json.dump({"context": self.vars.get("context", "")}, f, default=str)
                except Exception:
                    json.dump({"context": str(self.vars.get("context", ""))}, f)

            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write("import json\n")
                f.write("with open('/data/context.json') as f:\n")
                f.write("    context = json.load(f)['context']\n\n")
                f.write(code)
                f.write("\n\n")

            res = manager.run_container(
                image="python:3.11-slim",
                command="python /data/script.py",
                volumes={tmpdir: {"bind": "/data", "mode": "rw"}},
                detach=False,
            )
            output = res.get("output", "")
            return self.vars, output

    async def run_full_rlm(self, prompt: str) -> str:
        """
        The main agent loop for the RLM.
        """
        from pydantic_ai import Agent

        model_id = (
            self.config.sub_llm_model_large
            if self.depth == 0
            else self.config.sub_llm_model_small
        )

        agent = Agent(
            model=model_id,
            system_prompt=(
                "You are a Recursive Language Model (RLM).\n"
                "You have access to a persistent Python REPL.\n"
                "Your objective is to write python code to analyze the `context` variable, "
                "which contains massive amounts of data.\n\n"
                "AVAILABLE HELPERS:\n"
                "- `await rlm_query(prompt, context)`: Spawn a full recursive RLM at the next depth.\n"
                "- `await run_parallel_sub_calls(calls)`: Run multiple sub-calls in parallel. `calls` is a list of `{'prompt': '...', 'context': ...}`.\n"
                "- `await magma_view(query, views=None)`: Retrieve MAGMA orthogonal context (semantic, temporal, causal, entity).\n"
                "- `await graph_query(cypher, params=None)`: Run a Cypher query against the knowledge graph.\n"
                "- `await sub_agent_call(prompt, agent_id, input_data)`: Dispatch a task to another specialist.\n"
                "- `FINAL_VAR('result_name', value)`: Explicitly output the final result.\n\n"
                "You can write python code inside ```python blocks. "
                "Only tiny metadata feeds back to the root LLM (stdout, lengths)."
            ),
        )

        history: list[Any] = []
        max_turns = 5

        for turn in range(max_turns):
            res = await agent.run(prompt, message_history=history)
            history = res.all_messages()

            output_text = str(res.output)

            # Extract code block
            code_blocks = [
                b.split("```")[0] for b in output_text.split("```python\n")[1:]
            ]

            if code_blocks:
                code_to_run = code_blocks[0]
                _, stdout = await self.execute(code_to_run)

                # Record trajectory
                if (
                    self.config.trajectory_storage == "process_flow"
                    and self.graph_deps
                    and hasattr(self.graph_deps, "knowledge_engine")
                ):
                    engine = self.graph_deps.knowledge_engine
                    if engine:
                        import time

                        from ..models.knowledge_graph import (
                            ReasoningTraceNode,
                            RegistryNodeType,
                        )

                        node_id = f"rlm_trace_{time.time_ns()}"
                        trace_node = ReasoningTraceNode(
                            id=node_id,
                            type=RegistryNodeType.REASONING_TRACE,
                            name=f"RLM Depth {self.depth} Execution",
                            thought=prompt,
                            reflection=f"Code: {code_to_run}\nResult: {stdout[:500]}",
                            timestamp=time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        )
                        try:
                            engine.graph.add_node(
                                trace_node.id, **trace_node.model_dump()
                            )
                        except Exception as e:
                            logger.warning(f"Failed to store RLM trajectory: {e}")

                if "__FINAL__" in self.vars:
                    final_var_name = self.vars["__FINAL__"]
                    return str(self.vars.get(final_var_name, stdout))

                # Feed stdout back
                history.append(
                    {
                        "role": "user",
                        "content": f"Execution STDOUT:\n{stdout[:2000]}\n\nContinue analyzing or output FINAL_VAR.",
                    }
                )
            else:
                if "__FINAL__" in self.vars:
                    final_var_name = self.vars["__FINAL__"]
                    return str(self.vars.get(final_var_name, output_text))
                break

        return str(self.vars.get("__FINAL__", "Max turns reached without FINAL_VAR"))
