import asyncio
import json
import logging
from typing import Any

from pydantic_ai.messages import ModelRequest, UserPromptPart

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

from ..graph.state import GraphDeps
from ..graph.topology_engine import (
    ElasticTopologyAdmission,
    TopologyAdmissionError,
)
from .config import RLMConfig
from .prompts import (
    build_system_prompt,  # CONCEPT:AU-ORCH.execution.drop-rlm-completion-client
)
from .sandboxes import (  # CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox
    HELPER_NAMES,
    SandboxEnv,
    SandboxRejected,
)
from .sandboxes.registry import default_sandboxes
from .sandboxes.router import SandboxRouter
from .schema import (
    SchemaContract,  # CONCEPT:AU-ORCH.session.structured-subagent-contracts — structured subagent contracts
)
from .telemetry import (  # CONCEPT:AU-ORCH.execution.typed-failure-classification
    RunTrace,
    SandboxFatalError,
    classify_failure,
)

logger = logging.getLogger(__name__)


def _model_usage_tokens(res: Any) -> tuple[int, int]:
    """Read version-tolerant request/response token counts from a model result."""
    try:
        u = res.usage() if callable(getattr(res, "usage", None)) else None
    except Exception:  # noqa: BLE001 — usage is telemetry, never fatal
        u = None
    if u is None:
        return 0, 0
    prompt = next(
        (
            getattr(u, a)
            for a in ("request_tokens", "input_tokens")
            if isinstance(getattr(u, a, None), int)
        ),
        0,
    )
    completion = next(
        (
            getattr(u, a)
            for a in ("response_tokens", "output_tokens")
            if isinstance(getattr(u, a, None), int)
        ),
        0,
    )
    return prompt, completion


def _accumulate_root_usage(usage: Any, res: Any) -> int:
    """Fold one pydantic-ai run's token usage into the RunTrace ``usage`` (CONCEPT:AU-AHE.rlm.long-context-benchmark).

    Best-effort and version-tolerant: maps request/input tokens → ``prompt_tokens`` and
    response/output tokens → ``completion_tokens``. A missing usage object is a no-op.
    """
    prompt, completion = _model_usage_tokens(res)
    usage.prompt_tokens += prompt
    usage.completion_tokens += completion
    return prompt + completion


class RecursionLimitError(Exception):
    pass


class RLMEnvironment:
    """A persistent Python REPL environment for Recursive Language Models.

    CONCEPT:AU-ORCH.execution.rlm-execution — RLM Execution

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
        signature: Any = None,
        inputs_keys: list[str] | None = None,
        outputs_keys: list[str] | None = None,
        tool_sources: dict[str, str] | None = None,
        output_contract: Any = None,
        admission: ElasticTopologyAdmission | None = None,
        work_item_id: str = "",
        _usage_state: dict[str, int] | None = None,
    ):
        self.config = config or RLMConfig()
        self.depth = depth
        self.max_depth = self.config.max_depth
        self.max_turns = self.config.max_turns
        self.admission = admission
        self.work_item_id = str(work_item_id or "")
        self._usage_state = (
            _usage_state if _usage_state is not None else {"tokens": 0, "nodes": 1}
        )
        if self.admission is not None:
            self.admission.require_capabilities(("rlm.execute",))
            if self.depth > self.admission.max_depth:
                raise TopologyAdmissionError(
                    f"RLM depth exceeds admission ({self.depth} > {self.admission.max_depth})"
                )
            if self.max_depth > self.admission.max_depth:
                raise TopologyAdmissionError(
                    f"RLM max_depth exceeds admission ({self.max_depth} > {self.admission.max_depth})"
                )
        self.graph_deps = graph_deps
        self.signature = signature
        self.inputs_keys = inputs_keys or []
        self.outputs_keys = outputs_keys or []
        self.tool_sources = tool_sources or {}
        # CONCEPT:AU-ORCH.session.structured-subagent-contracts (subagent fan-out) — a single-value structured-output
        # contract for this (sub)agent. When set, FINAL is validated/coerced
        # against it and the JSON schema is shown to the model at REPL startup.
        self.output_contract = output_contract
        self._stdout_counter = 0
        # CONCEPT:AU-ORCH.execution.typed-failure-classification — the most recent RunTrace (set when run_full_rlm runs). Holds token
        # usage (root + folded-in sub-call usage) so callers can surface cost (CONCEPT:AU-AHE.rlm.long-context-benchmark).
        self.last_run_trace: Any = None

        self.vars: dict[str, Any] = {"context": context, "depth": depth}
        if self.admission is not None:
            self.admission.require_payload(context, label="RLM context")

        # The global namespace for the REPL
        self.globals_dict = {
            "__builtins__": __builtins__,
            "GraphComputeEngine": GraphComputeEngine,
            "context": self.vars["context"],
            "depth": self.vars["depth"],
            "rlm_query": self.rlm_query,
            "run_parallel_sub_calls": self.run_parallel_sub_calls,
            "magma_view": self.magma_view,
            "graph_query": self.graph_query,
            "owl_query": self.owl_query,
            "kg_bulk_export": self.kg_bulk_export,
            "ephemeral_graph_query": self.ephemeral_graph_query,
            "sub_agent_call": self.sub_agent_call_helper,
            "FINAL_VAR": self.FINAL_VAR,
            "json": json,
            "asyncio": asyncio,
        }

    def FINAL_VAR(self, name: str, value: Any):
        """Helper for the LLM to output its final result explicitly."""
        self.vars[name] = value
        self.vars["__FINAL__"] = name

    def _check_admission(self, *, payload: Any = None, label: str = "payload") -> None:
        if self.admission is None:
            return
        self.admission.remaining_seconds()
        if payload is not None:
            self.admission.require_payload(payload, label=label)
        if self._usage_state["nodes"] > self.admission.max_nodes:
            raise TopologyAdmissionError(
                "RLM node budget exhausted before another recursive call"
            )

    def _register_node(self) -> None:
        self._check_admission()
        self._usage_state["nodes"] += 1
        if (
            self.admission is not None
            and self._usage_state["nodes"] > self.admission.max_nodes
        ):
            self._usage_state["nodes"] -= 1
            raise TopologyAdmissionError(
                f"RLM node count exceeds admission ({self._usage_state['nodes'] + 1} > {self.admission.max_nodes})"
            )

    def _record_usage(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self._usage_state["tokens"] += int(tokens)
        if (
            self.admission is not None
            and self._usage_state["tokens"] > self.admission.max_tokens
        ):
            raise TopologyAdmissionError(
                f"RLM token budget exceeded ({self._usage_state['tokens']} > {self.admission.max_tokens})"
            )

    async def _run_model(self, agent: Any, prompt: str, **kwargs: Any) -> Any:
        """Run one model call inside the immutable deadline and token contract."""
        self._check_admission(payload=prompt, label="RLM model prompt")
        if self.admission is not None and kwargs.get("message_history") is not None:
            try:
                history_size = len(repr(kwargs["message_history"]).encode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - an unbounded history fails closed
                raise TopologyAdmissionError(
                    "RLM message history is not measurable"
                ) from exc
            if history_size > self.admission.max_payload_bytes:
                raise TopologyAdmissionError(
                    "RLM message history exceeds admission payload limit"
                )
        if self.admission is None:
            return await agent.run(prompt, **kwargs)
        remaining = self.admission.remaining_seconds()
        try:
            return await asyncio.wait_for(
                agent.run(prompt, **kwargs), timeout=remaining
            )
        except TimeoutError as exc:
            raise TopologyAdmissionError(
                "RLM model call exceeded its admission deadline"
            ) from exc

    def _record_direct_model_usage(self, result: Any) -> None:
        """Account model calls that do not run through ``run_full_rlm``."""
        prompt, completion = _model_usage_tokens(result)
        tokens = prompt + completion
        self._record_usage(tokens)
        if self.last_run_trace is not None:
            self.last_run_trace.usage.sub_lm_tokens += tokens

    def retire(self, *, reason: str = "rlm-retired") -> bool:
        """Retire this run through the engine-native WorkItem cancel verb."""
        if not self.work_item_id or not self.graph_deps:
            return False
        engine = getattr(self.graph_deps, "knowledge_engine", None)
        if engine is None:
            return False
        if self.admission is None:
            raise TopologyAdmissionError(
                "live RLM retirement requires an immutable admission"
            )
        from ..knowledge_graph.core.work_durability import (
            cancel_work_item,
            get_work_item,
        )

        item = get_work_item(engine, self.work_item_id)
        self.admission.require_work_item(item)

        return bool(cancel_work_item(engine, self.work_item_id, reason=reason))

    def _absorb_sub_usage(self, sub_env: "RLMEnvironment") -> None:
        """Fold a recursive sub-call's total token usage into this run's ``sub_lm_tokens``.

        CONCEPT:AU-AHE.rlm.long-context-benchmark — lets the root surface a complete cost figure (root + recursion) so the
        benchmark can compare against the paper's per-query cost. No-op if either trace is absent.
        """
        parent = self.last_run_trace
        child = getattr(sub_env, "last_run_trace", None)
        if parent is not None and child is not None:
            parent.usage.sub_lm_tokens += child.usage.total

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

    async def ephemeral_graph_query(
        self, cypher: str, namespace: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query against a specific ephemeral graph namespace."""
        try:
            # A graph-scoped view shares the process transport; OWLBridge has
            # already hydrated this namespace.
            ephemeral_engine = GraphComputeEngine.get_or_create(namespace)
            if hasattr(ephemeral_engine._client, "cypher"):
                result = ephemeral_engine._client.cypher.query(cypher, params or {})
                import json

                return json.loads(result) if isinstance(result, str) else result
            return [{"error": "Ephemeral graph client does not support direct cypher."}]
        except Exception as e:
            return [{"error": f"Ephemeral graph query failed: {e}"}]

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
            List of node dicts with id, name, node_type, and metadata.
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
            for node_id in graph.node_ids():
                data = graph._get_node_properties(node_id)
                if data.get("node_type") == node_type or node_type == "*":
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
        """Recursive dispatch to other adaptive_agent_router via the graph dispatcher."""
        self._check_admission(
            payload={"prompt": prompt, "input": input_data}, label="sub-agent payload"
        )
        if not self.graph_deps:
            return "Error: graph_deps not available"

        # In a real implementation, we would call the dispatcher.
        # Since we are inside a specialist execution, we might need to
        # use the dispatcher provided in graph_deps if it exists.

        # For now, we'll use a simplified dispatch if dispatcher is available
        # But wait, RLM is usually called from a StepContext.

        # This is tricky because we need a StepContext.
        # If we don't have it, we fallback to a direct Agent call.

        from agent_utilities.core.contextual_model import create_context_agent

        agent = create_context_agent(
            model=self.config.sub_llm_model_small,
            system_prompt=f"You are a specialized sub-agent for: {agent_id or 'general'}",
        )
        res = await self._run_model(agent, f"Context: {input_data}\n\nTask: {prompt}")
        self._record_direct_model_usage(res)
        return res.output

    async def rlm_query(
        self, prompt: str, sub_context: Any = None, schema: Any = None
    ) -> Any:
        """Spawn a full recursive RLM at the next depth.

        When ``schema`` is given (a Pydantic model, a primitive/generic type, or a
        raw JSON-Schema dict), the sub-RLM's ``FINAL`` is validated and coerced
        against it and a *typed* value is returned — not a free-form string. This
        lets the parent route on a clean structured value instead of re-parsing
        prose (CONCEPT:AU-ORCH.session.structured-subagent-contracts, structured subagent contracts).
        """
        self._check_admission(
            payload={"prompt": prompt, "context": sub_context},
            label="RLM child payload",
        )
        if self.depth >= self.max_depth:
            raise RecursionLimitError(
                f"RLM recursion depth exceeded (max {self.max_depth})"
            )
        self._register_node()

        logger.info(
            f"RLM at depth {self.depth} spawning sub-RLM for prompt: {prompt[:50]}..."
        )
        sub_env = RLMEnvironment(
            context=sub_context,
            depth=self.depth + 1,
            config=self.config,
            graph_deps=self.graph_deps,
            output_contract=SchemaContract.from_spec(schema)
            if schema is not None
            else None,
            admission=self.admission,
            work_item_id=self.work_item_id,
            _usage_state=self._usage_state,
        )
        result = await sub_env.run_full_rlm(prompt)
        self._absorb_sub_usage(
            sub_env
        )  # CONCEPT:AU-AHE.rlm.long-context-benchmark — fold sub-call tokens into our trace
        return result

    async def run_parallel_sub_calls(self, calls: list[dict[str, Any]]) -> list[Any]:
        """
        Run multiple sub-calls in parallel.
        calls is a list of dicts: ``{"prompt": "...", "context": Any, "schema": Any}``.
        ``schema`` is optional and per-call — when present, that sub-agent must
        return a value conforming to it (structured fan-out, CONCEPT:AU-ORCH.session.structured-subagent-contracts).
        """
        if self.admission is not None:
            self._check_admission(payload=calls, label="RLM fan-out payload")
            if len(calls) > self.admission.max_fan_out:
                raise TopologyAdmissionError(
                    f"RLM fan-out exceeds admission ({len(calls)} > {self.admission.max_fan_out})"
                )

        if not self.config.async_enabled:
            results = []
            for item in calls:
                results.append(await self._execute_sub_call(item))
            return results

        parallelism = len(calls)
        if self.admission is not None:
            parallelism = min(parallelism, self.admission.max_parallelism)
        semaphore = asyncio.Semaphore(max(1, parallelism))

        async def _call(item):
            async with semaphore:
                return await self._execute_sub_call(item)

        results = await asyncio.gather(
            *[_call(item) for item in calls], return_exceptions=True
        )
        for result in results:
            if isinstance(result, TopologyAdmissionError):
                raise result
        return results

    async def _execute_sub_call(self, item: dict[str, Any]) -> Any:
        if not isinstance(item, dict):
            raise TopologyAdmissionError("RLM fan-out entries must be mappings")
        self._check_admission(payload=item, label="RLM child call")
        schema = item.get("schema")
        contract = SchemaContract.from_spec(schema) if schema is not None else None
        if self.depth < self.max_depth:
            self._register_node()
            sub_env = RLMEnvironment(
                context=item.get("context"),
                depth=self.depth + 1,
                config=self.config,
                graph_deps=self.graph_deps,
                output_contract=contract,
                admission=self.admission,
                work_item_id=self.work_item_id,
                _usage_state=self._usage_state,
            )
            result = await sub_env.run_full_rlm(item["prompt"])
            self._absorb_sub_usage(
                sub_env
            )  # CONCEPT:AU-AHE.rlm.long-context-benchmark — fold sub-call tokens in
            return result
        else:
            # Fallback to normal specialist at the recursion floor. The contract
            # still holds — pydantic_ai enforces it natively via ``output_type``
            # when the contract is backed by a real Pydantic model; a
            # primitive/generic (TypeAdapter) or raw JSON-Schema contract has no
            # single Python type to hand pydantic-ai, so it's conveyed as prompt
            # text instead (the caller still gets back whatever the model wrote —
            # there's no post-hoc ``.validate()`` at this recursion floor).
            self._register_node()
            from agent_utilities.core.contextual_model import create_context_agent

            model_type = contract.model_type if contract else None
            if model_type is not None:
                agent = create_context_agent(
                    model=self.config.sub_llm_model_small,
                    system_prompt="Answer the sub-task directly.",
                    output_type=model_type,
                )
            elif contract is not None:
                agent = create_context_agent(
                    model=self.config.sub_llm_model_small,
                    system_prompt=(
                        "Answer the sub-task directly. Respond with JSON matching "
                        f"this schema:\n{contract.json_schema_str}"
                    ),
                )
            else:
                agent = create_context_agent(
                    model=self.config.sub_llm_model_small,
                    system_prompt="Answer the sub-task directly.",
                )
            res = await self._run_model(
                agent,
                f"Context: {item.get('context')}\n\nPrompt: {item['prompt']}",
            )
            self._record_direct_model_usage(res)
            return res.output

    def _build_sandbox_env(self) -> SandboxEnv:
        """Snapshot the REPL state into the cross-backend :class:`SandboxEnv` (ORCH-1.38).

        ``helpers`` is the host-callback subset of the REPL namespace (the ``HELPER_NAMES``
        bound methods). Confined backends wire ``helpers`` through their governed callback
        boundary (monty's ``external_functions`` or Docker's UDS bridge).
        """
        if self.admission is not None:
            self.admission.require_payload(
                self.tool_sources, label="RLM tool-source payload"
            )
        helpers = {
            name: self.globals_dict[name]
            for name in HELPER_NAMES
            if name in self.globals_dict
        }
        return SandboxEnv(
            vars=self.vars,
            tool_sources=self.tool_sources,
            helpers=helpers,
            admission=self.admission,
        )

    def _get_sandbox_router(self) -> SandboxRouter:
        """Lazily build (and cache) this environment's router over the available backends."""
        router = getattr(self, "_sandbox_router", None)
        if router is None:
            # CONCEPT:AU-ORCH.sandbox.rung-reward-ema — feed the per-rung reward EMA so a persistently failing rung is
            # routed around (bounded; steady-state order is unchanged).
            from .sandboxes.reward import SandboxRewardTracker

            router = SandboxRouter(
                default_sandboxes(admission=self.admission),
                reward_fn=SandboxRewardTracker.get().reward,
            )
            self._sandbox_router = router
        return router

    async def execute(self, code: str) -> tuple[dict[str, Any], str]:
        """Execute LLM-generated code via the tiered sandbox router (CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox).

        Builds the escalation chain for this snippet (cheapest capable backend first), then
        runs each in order:

        * :class:`SandboxRejected` — this backend can't run the snippet; escalate to the next
          tier. Parse-level rejections (classes, unsupported syntax) happen before any host
          helper fires, so escalation has no side effects.
        * :class:`SandboxFatalError` — irreversible infra death; propagate to fast-fail the run
          (ORCH-1.29 semantics, unchanged — deliberately not caught here).
        * success — sync the namespace back into ``self.vars`` and return ``(vars, stdout)``.

        Secure/default routing fails closed when no isolated backend can accept the
        code. Unsafe local execution requires an explicit configuration opt-in.
        """
        self._check_admission(payload=code, label="RLM source payload")
        env = self._build_sandbox_env()
        forced = self.config.sandbox
        chain = self._get_sandbox_router().select(
            code, force=None if forced == "auto" else forced
        )

        from .sandboxes.reward import SandboxRewardTracker

        rewards = SandboxRewardTracker.get()
        last_reject: SandboxRejected | None = None
        for backend in chain:
            try:
                result = await backend.execute(code, env)
            except SandboxRejected as rej:
                # A rejection is a capability mismatch, NOT a backend failure — don't penalise
                # the rung's reward; just escalate to the next tier.
                last_reject = rej
                logger.info(
                    "Sandbox %s rejected snippet (%s); escalating.",
                    backend.name,
                    rej.reason,
                )
                continue
            except SandboxFatalError:
                # Irreversible infra death on this rung (CONCEPT:AU-ORCH.sandbox.rung-reward-ema): penalise its reward
                # so the router prefers a healthier capable rung next time, then fast-fail the run
                # (ORCH-1.29 semantics unchanged — still not swallowed).
                rewards.record(backend.name, success=False)
                raise
            # Success: reward the rung, sync the namespace back, return.
            rewards.record(backend.name, success=True)
            self.vars.update(result.updated_vars)
            if backend is not chain[0]:
                logger.debug("RLM snippet ran on escalated backend %s", backend.name)
            return self.vars, result.stdout

        # Every approved backend rejected the snippet. Do not silently cross the
        # isolation boundary by evaluating model code in this process.
        reason = last_reject.reason if last_reject else "no available backend"
        raise SandboxFatalError(
            f"no approved RLM sandbox accepted the snippet ({reason})"
        )

    def _validate_outputs(self) -> str | None:
        """Validate the agent's output(s). Returns an error string if invalid, None if valid.

        Two modes:
          * ``output_contract`` (subagent fan-out) — validate the single ``FINAL``
            value against the structured-output contract and store the *coerced*
            value back into ``self.vars`` so the parent receives a typed value.
          * ``signature`` (root Predict-RLM) — validate gathered output variables
            against the full Pydantic signature.
        """
        if self.output_contract is not None:
            return self._validate_contract_output()

        if not self.signature:
            return None

        return self._validate_signature_outputs()

    def _validate_contract_output(self) -> str | None:
        """Validate and coerce the single output-contract value, when present."""
        final_name = self.vars.get("__FINAL__")
        if final_name is None:
            return None
        ok, coerced, err = self.output_contract.validate(self.vars.get(final_name))
        if not ok:
            return (
                f"FINAL value failed schema validation.\n"
                f"Required JSON Schema:\n{self.output_contract.json_schema_str}\n\n"
                f"Validation errors:\n{err}\n\n"
                f"Fix the value and call FINAL_VAR again."
            )
        # Persist the coerced (type-correct) value for the parent to consume.
        self.vars[final_name] = coerced
        return None

    def _signature_output_data(self) -> dict[str, Any]:
        """Collect signature outputs from REPL vars, FINAL_VAR, or globals."""
        output_data = {}
        for name in self.outputs_keys:
            if name in self.vars:
                output_data[name] = self.vars[name]
            elif "__FINAL__" in self.vars and self.vars["__FINAL__"] == name:
                output_data[name] = self.vars[name]
            elif name in self.globals_dict:
                output_data[name] = self.globals_dict[name]
        return output_data

    def _validate_signature_outputs(self) -> str | None:
        """Validate collected output values against the configured signature."""
        try:
            output_data = self._signature_output_data()
            context = self.vars.get("context", {})
            inputs = context if isinstance(context, dict) else {}
            full_data = {**inputs, **output_data}
            self.signature(**full_data)
            return None
        except Exception as e:
            return f"Validation Error for outputs:\n{str(e)}\nPlease correct the variables and output again."

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
            f"  prefix: {prefix!r}...\n"
            f"ACCESS INSTRUCTIONS:\n"
            f"  - The full context is in the `context` variable.\n"
            f"  - Peek at slices: `context[start:end]`\n"
            f"  - Get length: `len(context)`\n"
            f"  - Parse JSON: `json.loads(context)`\n"
            f"  - Split lines: `context.splitlines()`\n"
            f"  - Use `await rlm_query(prompt, sub_context, schema=...)` to recursively analyze "
            f"sub-slices (pass `schema` for a typed answer).\n"
            f"  - Use `await run_parallel_sub_calls(calls)` for parallel decomposition; give each "
            f"call a `schema` (e.g. a boolean flag) so you filter on typed results, not prose."
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

    @staticmethod
    def _extract_rlm_code(output_text: str) -> str | None:
        """Extract the first Python fenced block from a model response."""
        code_blocks = [
            block.split("```")[0] for block in output_text.split("```python\n")[1:]
        ]
        return code_blocks[0] if code_blocks else None

    def _build_rlm_initial_prompt(self, prompt: str) -> str:
        """Build the first-turn prompt, including metadata and output contracts."""
        if self.config.metadata_only_root and self.depth == 0:
            initial_prompt = f"{prompt}\n\n{self._build_context_metadata()}"
        else:
            initial_prompt = prompt

        # CONCEPT:AU-ORCH.session.structured-subagent-contracts (structured subagent contracts) — show the output
        # contract before any code is written, so the model knows the exact
        # shape it must return via FINAL_VAR.
        if self.output_contract is not None:
            initial_prompt = (
                f"{initial_prompt}\n\n"
                f"REQUIRED OUTPUT CONTRACT:\n"
                f"You MUST call `FINAL_VAR('result', value)` with a value conforming to "
                f"this JSON Schema:\n{self.output_contract.json_schema_str}"
            )
        return initial_prompt

    def _build_rlm_turn_feedback(self, stdout: str, turn: int) -> str:
        """Build the bounded feedback sent to the model after code execution."""
        if self.config.metadata_only_root and self.depth == 0:
            return self._build_stdout_metadata(stdout, turn)
        return (
            f"Execution STDOUT:\n{stdout[:2000]}\n\n"
            f"Continue analyzing or output FINAL_VAR."
        )

    async def _execute_rlm_code(
        self,
        code: str,
        response: Any,
        prompt: str,
        run_trace: RunTrace,
    ) -> str:
        """Execute one generated block and record its trace and trajectory."""
        # CONCEPT:AU-ORCH.execution.typed-failure-classification — record this iteration; classify + re-raise on failure (fatal
        # sandbox death still fast-fails) so the RunTrace captures the failure class.
        try:
            _, stdout = await self.execute(code)
        except SandboxFatalError as exc:
            run_trace.add_step(code=code, failure_class=classify_failure(exc))
            run_trace.final_status = "failure"
            raise
        except Exception as exc:  # noqa: BLE001
            run_trace.add_step(code=code, failure_class=classify_failure(exc))
            run_trace.final_status = "failure"
            raise

        run_trace.add_step(
            code=code,
            output=str(stdout)[:2000],
            finish_reason=str(getattr(response, "finish_reason", "") or "stop"),
        )
        self._persist_rlm_trajectory(prompt, code, stdout)
        return stdout

    def _persist_rlm_trajectory(self, prompt: str, code: str, stdout: str) -> None:
        """Persist one successful RLM step to the configured graph stores."""
        if self.config.trajectory_storage != "process_flow" or not self.graph_deps:
            return

        import time

        from ..graph.client import create_or_merge_node
        from ..graph.models import GraphNode
        from ..models.knowledge_graph import ReasoningTraceNode, RegistryNodeType

        node_id = f"rlm_trace_{time.time_ns()}"
        trace_node = ReasoningTraceNode(
            id=node_id,
            type=RegistryNodeType.REASONING_TRACE,
            name=f"RLM Depth {self.depth} Execution",
            thought=prompt,
            reflection=f"Code: {code}\nResult: {stdout[:500]}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # 1. In-Memory Graph persistence if engine exists
        if (
            hasattr(self.graph_deps, "knowledge_engine")
            and self.graph_deps.knowledge_engine
        ):
            try:
                self.graph_deps.knowledge_engine.graph.add_node(
                    trace_node.id, **trace_node.to_graph_properties()
                )
            except Exception as exc:
                logger.warning(f"Failed to store RLM trajectory in-memory: {exc}")

        # 2. Asynchronous/Persistent Graph DB persistence
        try:
            g_node = GraphNode(
                id=trace_node.id,
                labels=["ReasoningTrace"],
                properties=trace_node.to_graph_properties(exclude_none=True),
            )
            # Schedule background/async write to persistent backend
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(create_or_merge_node(g_node))
        except Exception as exc:
            logger.warning(f"Failed to store RLM trajectory in DB backend: {exc}")

    async def _run_rlm_turn(
        self,
        agent: Any,
        initial_prompt: str,
        history: list[Any],
        turn: int,
        model_settings: Any,
        run_trace: RunTrace,
        prompt: str,
    ) -> tuple[list[Any], str, str, bool]:
        """Run one model turn, execute its optional code, and return bounded output."""
        run_prompt = initial_prompt if turn == 0 else None
        if not run_prompt:
            run_prompt = "Continue."
        response = await self._run_model(
            agent,
            run_prompt,
            message_history=history,
            model_settings=model_settings,
        )
        next_history = response.all_messages()
        self._record_usage(
            _accumulate_root_usage(run_trace.usage, response)
        )  # CONCEPT:AU-AHE.rlm.long-context-benchmark cost capture
        output_text = response.output
        code = self._extract_rlm_code(output_text)
        if code is None:
            # ``code_executed`` is the authority for whether stdout is meaningful;
            # keep the value typed as text so callers do not carry a second,
            # redundant optional state through the turn loop.
            return next_history, output_text, "", False
        stdout = await self._execute_rlm_code(code, response, prompt, run_trace)
        return next_history, output_text, stdout, True

    def _finalize_rlm_output(
        self,
        fallback: str,
        stdout: str,
        run_trace: RunTrace,
        code_executed: bool,
    ) -> tuple[bool, Any]:
        """Validate and return a FINAL value, or return feedback for a retry."""
        if "__FINAL__" not in self.vars:
            return False, None

        final_var_name = self.vars["__FINAL__"]
        validation_err = self._validate_outputs()
        if validation_err:
            # Clear FINAL_VAR so the LLM has to try again.
            del self.vars["__FINAL__"]
            if not code_executed:
                return False, f"CRITICAL: {validation_err}"
            return (
                False,
                f"Execution STDOUT:\n{stdout[:2000]}\n\nCRITICAL: {validation_err}",
            )

        if code_executed:
            run_trace.final_status = (
                "success"  # CONCEPT:AU-ORCH.execution.typed-failure-classification
            )
        return True, self._final_value(final_var_name, fallback)

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
        from agent_utilities.core.contextual_model import create_context_agent

        model_id = (
            self.config.sub_llm_model_large
            if self.depth == 0
            else self.config.sub_llm_model_small
        )

        # CONCEPT:AU-ORCH.execution.drop-rlm-completion-client — family-aware system prompt (the paper's "one prompt fails across
        # model families" failure mode); 'auto' infers the family from the root model id.
        repl_system_prompt = build_system_prompt(self.config.prompt_family, model_id)
        agent = create_context_agent(model=model_id, system_prompt=repl_system_prompt)

        # CONCEPT:AU-ORCH.routing.depth-tiered-sampling — depth-tiered sampling. The root is the strong reasoner
        # (higher temperature for exploration); recursive sub-calls are deterministic executors
        # writing/running code (low temp + tight top_k). Mirrors the model_id depth split above.
        from agent_utilities.agent.sampling_profile import resolve_sampling_profile

        profile_role = "rlm-root" if self.depth == 0 else "rlm-executor"
        profile_settings = resolve_sampling_profile(
            role=profile_role
        ).to_model_settings({})
        try:
            # D-54c-4 — the RLM REPL loop calls agent.run() directly with an explicit
            # model_settings, bypassing attach_profile_resolver's prompt-cache fold
            # (CONCEPT:AU-ORCH.optimization.provider-prompt-cache). Fold it here too — the
            # system prompt is stable per (depth, prompt_family, model_id), which is exactly
            # the repeated-prefix shape prompt caching benefits from across turns/recursion.
            from agent_utilities.caching.prompt_cache import fold_prompt_cache_hint

            profile_settings = fold_prompt_cache_hint(
                profile_settings,
                system_prompt=repl_system_prompt,
                model_identity=model_id,
            )
        except Exception:  # noqa: BLE001 - prompt-cache hint is best-effort
            pass

        history: list[Any] = []

        # CONCEPT:AU-ORCH.execution.typed-failure-classification — populate a
        # structured RunTrace as the live loop runs for canonical outcome analysis.
        run_trace = RunTrace()
        self.last_run_trace = run_trace
        initial_prompt = self._build_rlm_initial_prompt(prompt)
        self._check_admission(payload=initial_prompt, label="RLM initial prompt")

        for turn in range(self.max_turns):
            self._register_node()
            history, output_text, stdout, code_executed = await self._run_rlm_turn(
                agent,
                initial_prompt,
                history,
                turn,
                profile_settings,
                run_trace,
                prompt,
            )
            finalized, feedback = self._finalize_rlm_output(
                stdout if code_executed else output_text,
                stdout,
                run_trace,
                code_executed,
            )
            if finalized:
                return feedback
            if feedback is not None:
                history.append(ModelRequest(parts=[UserPromptPart(content=feedback)]))
                continue
            if not code_executed:
                break
            history.append(
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=self._build_rlm_turn_feedback(stdout, turn)
                        )
                    ]
                )
            )

        return str(self.vars.get("__FINAL__", "Max turns reached without FINAL_VAR"))

    def _final_value(self, final_var_name: str, fallback: str) -> Any:
        """Return the FINAL result — the coerced typed value when a structured
        ``output_contract`` is set (so the parent gets a real bool/model/list),
        otherwise the string form for the free-text path (back-compat)."""
        value = self.vars.get(final_var_name, fallback)
        if self.output_contract is not None:
            return value
        return str(value)
