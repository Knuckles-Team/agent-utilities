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
    """A persistent Python REPL environment for Recursive Language Models.

    CONCEPT:AU-007 — RLM Execution

    Implements Algorithm 1 from Zhang et al. (2025): the user prompt is
    loaded as a variable inside the REPL — the root LLM receives only
    constant-size metadata (length, prefix, type) and writes code to
    programmatically examine, decompose, and recursively call itself
    over slices of the prompt.

    Available helpers in the REPL namespace:
        - ``rlm_query(prompt, context)`` — Spawn a recursive sub-RLM
        - ``run_parallel_sub_calls(calls)`` — Parallel sub-call dispatch
        - ``magma_view(query, views)`` — MAGMA orthogonal memory views
        - ``graph_query(cypher, params)`` — Cypher against LPG
        - ``owl_query(sparql)`` — SPARQL against OWL reasoner
        - ``kg_bulk_export(node_type, limit)`` — Bulk KG node export
        - ``sub_agent_call(prompt, agent_id, data)`` — Specialist dispatch
        - ``FINAL_VAR(name, value)`` — Output the final result

    Args:
        context: The (potentially massive) data to analyze.
        depth: Current recursion depth (0 = root).
        config: RLM configuration.
        graph_deps: Graph dependencies for KG/OWL access.
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
        self._stdout_counter = 0

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
            "owl_query": self.owl_query,
            "kg_bulk_export": self.kg_bulk_export,
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

    async def owl_query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query against the OWL reasoner backend.

        Enables the RLM to leverage transitive reasoning chains
        (e.g., ``wasDerivedFrom``, ``escalatedTo``, SKOS hierarchies)
        over KG subgraphs without loading raw triples into the context window.

        Args:
            sparql: A SPARQL SELECT query string.

        Returns:
            List of result bindings as dicts.
        """
        if not self.graph_deps or not hasattr(self.graph_deps, "knowledge_engine"):
            return [{"error": "Knowledge engine not available"}]

        engine = self.graph_deps.knowledge_engine
        if not engine:
            return [{"error": "Knowledge engine not initialized"}]

        # Delegate to OWL bridge if available
        if hasattr(engine, "owl_bridge") and engine.owl_bridge:
            try:
                return engine.owl_bridge.query_sparql(sparql)
            except Exception as e:
                return [{"error": f"SPARQL query failed: {e}"}]

        return [{"error": "OWL bridge not configured"}]

    async def kg_bulk_export(
        self, node_type: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Export a batch of KG nodes as JSON dicts for programmatic analysis.

        The LLM can write Python code to aggregate, filter, and
        cross-reference these nodes without ever loading them into
        the context window — a key RLM advantage over vanilla agents.

        Args:
            node_type: The node type to export (e.g., 'memory', 'task', 'evidence').
            limit: Maximum number of nodes to return.

        Returns:
            List of node dicts with id, name, type, and metadata.
        """
        if not self.graph_deps or not hasattr(self.graph_deps, "knowledge_engine"):
            return [{"error": "Knowledge engine not available"}]

        engine = self.graph_deps.knowledge_engine
        if not engine:
            return [{"error": "Knowledge engine not initialized"}]

        try:
            nodes = []
            graph = engine.graph
            count = 0
            for node_id, data in graph.nodes(data=True):
                if data.get("type") == node_type or node_type == "*":
                    nodes.append({"id": node_id, **data})
                    count += 1
                    if count >= limit:
                        break
            return nodes
        except Exception as e:
            return [{"error": f"KG bulk export failed: {e}"}]

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

            # Sync back globals to vars (except builtins and helper functions)
            for k, v in self.globals_dict.items():
                if k not in [
                    "__builtins__",
                    "nx",
                    "rlm_query",
                    "run_parallel_sub_calls",
                    "magma_view",
                    "graph_query",
                    "owl_query",
                    "kg_bulk_export",
                    "sub_agent_call",
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

    # ── Whitepaper Alignment: Metadata Helpers ──

    @staticmethod
    def _infer_context_type(context: Any) -> str:
        """Infer the type of the context variable for metadata."""
        ctx_str = str(context)
        if ctx_str.lstrip().startswith("{") or ctx_str.lstrip().startswith("["):
            return "json"
        if "," in ctx_str[:500] and "\n" in ctx_str[:500]:
            return "csv"
        if "<" in ctx_str[:200] and ">" in ctx_str[:200]:
            return "xml/html"
        return "text"

    def _build_context_metadata(self) -> str:
        """Build a metadata-only description of the context variable.

        Implements Algorithm 1 from Zhang et al. — the root LLM receives
        only constant-size metadata about the prompt, not the prompt itself.

        Returns:
            A metadata string with length, prefix, type, and access
            instructions for the ``context`` variable.
        """
        ctx = self.vars.get("context", "")
        ctx_str = str(ctx)
        ctx_type = self._infer_context_type(ctx)
        prefix = ctx_str[:200].replace("\n", " ")
        return (
            f"CONTEXT METADATA:\n"
            f"  type: {ctx_type}\n"
            f"  length: {len(ctx_str):,} characters\n"
            f'  prefix: "{prefix}..."\n'
            f"ACCESS INSTRUCTIONS:\n"
            f"  - The full context is in the `context` variable.\n"
            f"  - Peek at slices: `context[start:end]`\n"
            f"  - Get length: `len(context)`\n"
            f"  - Parse JSON: `json.loads(context)`\n"
            f"  - Split lines: `context.splitlines()`\n"
            f"  - Use `await rlm_query(prompt, sub_context)` to recursively analyze sub-slices.\n"
            f"  - Use `await run_parallel_sub_calls(calls)` for parallel decomposition."
        )

    def _build_stdout_metadata(self, stdout: str, turn: int) -> str:
        """Build metadata-only feedback for stdout from a REPL turn.

        Stores full stdout in a numbered variable and returns only
        a constant-size metadata summary to the root LLM.

        Args:
            stdout: The full stdout string from the REPL execution.
            turn: The current turn number.

        Returns:
            A metadata string referencing the stored variable.
        """
        self._stdout_counter += 1
        var_name = f"_stdout_{self._stdout_counter}"
        self.vars[var_name] = stdout
        self.globals_dict[var_name] = stdout

        prefix = stdout[:200].replace("\n", " ")
        return (
            f"EXECUTION RESULT (turn {turn + 1}):\n"
            f"  stdout_length: {len(stdout):,} characters\n"
            f'  stdout_prefix: "{prefix}..."\n'
            f"  Full output stored in `{var_name}`. Access with `{var_name}[start:end]`.\n"
            f"  Continue analyzing or output FINAL_VAR('result', value)."
        )

    async def run_full_rlm(self, prompt: str) -> str:
        """The main RLM agent loop (Algorithm 1, Zhang et al. 2025).

        The root LLM receives only metadata about the context and generates
        Python code to programmatically examine, decompose, and recursively
        process the data. Each iteration:

            1. LLM generates a response (potentially containing ```python blocks)
            2. Code blocks are extracted and executed via ``execute()``
            3. Stdout metadata is fed back (not raw stdout)
            4. If ``FINAL_VAR`` was called, the result is returned
            5. Otherwise, the loop continues (up to ``max_turns=5``)

        Args:
            prompt: The analytical task to perform on the context.

        Returns:
            The final result string from ``FINAL_VAR``.
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
                "- `await run_parallel_sub_calls(calls)`: Run multiple sub-calls in parallel. "
                "`calls` is a list of `{'prompt': '...', 'context': ...}`.\n"
                "- `await magma_view(query, views=None)`: Retrieve MAGMA orthogonal context "
                "(semantic, temporal, causal, entity).\n"
                "- `await graph_query(cypher, params=None)`: Run a Cypher query against the knowledge graph.\n"
                "- `await owl_query(sparql)`: Run a SPARQL query against the OWL reasoner "
                "for transitive reasoning (wasDerivedFrom chains, SKOS hierarchies, escalation paths).\n"
                "- `await kg_bulk_export(node_type, limit=500)`: Export KG nodes as JSON for bulk analysis.\n"
                "- `await sub_agent_call(prompt, agent_id, input_data)`: Dispatch a task to another specialist.\n"
                "- `FINAL_VAR('result_name', value)`: Explicitly output the final result.\n\n"
                "IMPORTANT: You do NOT have the full context in your window. "
                "Access it programmatically via the `context` variable. "
                "Write Python code inside ```python blocks."
            ),
        )

        history: list[Any] = []
        max_turns = 5

        # Build the initial prompt — metadata-only or full depending on config
        if self.config.metadata_only_root and self.depth == 0:
            context_info = self._build_context_metadata()
            initial_prompt = f"{prompt}\n\n{context_info}"
        else:
            initial_prompt = prompt

        for turn in range(max_turns):
            run_prompt = initial_prompt if turn == 0 else None
            if run_prompt:
                res = await agent.run(run_prompt, message_history=history)
            else:
                # Subsequent turns use the history with stdout metadata appended
                res = await agent.run("Continue.", message_history=history)
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

                # Feed stdout back as metadata (Algorithm 1 alignment)
                if self.config.metadata_only_root and self.depth == 0:
                    stdout_feedback = self._build_stdout_metadata(stdout, turn)
                else:
                    stdout_feedback = (
                        f"Execution STDOUT:\n{stdout[:2000]}\n\n"
                        f"Continue analyzing or output FINAL_VAR."
                    )
                history.append(
                    {
                        "role": "user",
                        "content": stdout_feedback,
                    }
                )
            else:
                if "__FINAL__" in self.vars:
                    final_var_name = self.vars["__FINAL__"]
                    return str(self.vars.get(final_var_name, output_text))
                break

        return str(self.vars.get("__FINAL__", "Max turns reached without FINAL_VAR"))
