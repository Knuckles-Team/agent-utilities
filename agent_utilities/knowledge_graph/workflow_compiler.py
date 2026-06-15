"""Natural Language → Workflow Compiler.

CONCEPT:ORCH-1.23 — NL Workflow Compilation

Compiles natural language workflow descriptions into executable
``GraphPlan`` objects and persists them in the Knowledge Graph.

Pipeline::

    ┌───────────────┐   parse_intent()    ┌──────────────────┐
    │  Natural       │ ─────────────────► │  Intent + Steps   │
    │  Language      │                     │  (structured)     │
    └───────────────┘                     └────────┬─────────┘
                                                   │
                                          match_agents()
                                                   │
                                          ┌────────▼─────────┐
                                          │  KG Agent/Tool    │
                                          │  Resolution       │
                                          └────────┬─────────┘
                                                   │
                                          build_dag()
                                                   │
                                          ┌────────▼─────────┐
                                          │  GraphPlan        │
                                          │  (executable)     │
                                          └────────┬─────────┘
                                                   │
                                          store.save_workflow()
                                                   │
                                          ┌────────▼─────────┐
                                          │  KG Persistence   │
                                          │  (WorkflowDef)    │
                                          └──────────────────┘

The compiler handles:
    - Intent extraction from free-text descriptions
    - Agent/tool matching via KG semantic search
    - Dependency inference between steps
    - Parallel group detection
    - Automatic mermaid diagram generation
    - KG persistence via ``WorkflowStore``

Usage::

    from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

    compiler = WorkflowCompiler(engine)

    # Compile from natural language
    plan = await compiler.compile(
        "Search for recent AI papers, then summarize the top 3, "
        "and finally create a presentation with the findings."
    )

    # Compile and persist in one call
    workflow_id = await compiler.compile_and_store(
        name="research_to_presentation",
        description="Search papers → summarize → present",
    )

    # Find and replay a stored workflow
    plan = compiler.find_and_load("summarize papers")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Common workflow step patterns for NL parsing
_STEP_PATTERNS = [
    # Explicit ordering: "first...then...finally..."
    r"(?:first|1\.?\s*)\s*(.+?)(?:\.\s*|,\s*(?:then|next|2))",
    r"(?:then|next|2\.?\s*)\s*(.+?)(?:\.\s*|,\s*(?:then|finally|3))",
    r"(?:finally|lastly|3\.?\s*)\s*(.+?)(?:\.\s*|$)",
    # Sequential: "step 1: ... step 2: ..."
    r"step\s+\d+[:\s]+(.+?)(?=step\s+\d+|$)",
    # Comma-separated: "do A, do B, and do C"
    r"(?:^|,\s*(?:and\s+)?)\s*(.+?)(?=,|$)",
]

# Keywords that indicate parallel execution
_PARALLEL_KEYWORDS = [
    "simultaneously",
    "concurrently",
    "in parallel",
    "at the same time",
    "while also",
    "along with",
]

# Keywords that indicate sequential dependencies
_SEQUENTIAL_KEYWORDS = [
    "then",
    "after",
    "once",
    "when done",
    "next",
    "finally",
    "followed by",
    "based on",
    "using the results",
]

# Sentinel: the per-compiler embedding fn has not been built yet (distinct from
# ``None``, which means "built and unavailable").
_UNSET: Any = object()

# Node types a workflow step may resolve to during semantic agent matching.
_AGENT_MATCH_TYPES = frozenset(
    {
        "AGENT_SKILL",
        "Server",
        "MCP_TOOL",
        "Agent",
        "Tool",
        "Skill",
        "CallableResource",
    }
)


class WorkflowCompiler:
    """Compiles natural language descriptions into GraphPlan flows.

    CONCEPT:ORCH-1.23 — NL Workflow Compilation

    Uses the KG's agent/tool registry and semantic search to match
    natural language step descriptions to registered capabilities,
    then builds an executable ``GraphPlan`` with proper dependencies.

    Attributes:
        engine: The IntelligenceGraphEngine for KG queries.
        store: WorkflowStore for persistence (lazily created).
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine
        self._store: Any = None
        # Per-compiler embedding fn, built once on first semantic match and reused
        # across steps. ``_UNSET`` = not built; ``None`` = embeddings unavailable;
        # callable = ready. The fn is bounded per-call (``_embed``) so a slow/down
        # endpoint degrades agent matching in seconds instead of stalling.
        self._embed_fn: Any = _UNSET

    @property
    def store(self) -> Any:
        """Lazy-load the WorkflowStore."""
        if self._store is None:
            from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

            self._store = WorkflowStore(self.engine)
        return self._store

    async def compile(
        self,
        description: str,
        domain: str = "general",
    ) -> GraphPlan:
        """Compile a natural language workflow description into a GraphPlan.

        CONCEPT:ORCH-1.23 — NL Compilation Pipeline

        Steps:
            1. Parse the NL description into discrete step intents
            2. Detect parallel vs sequential relationships
            3. Match each step to a KG-registered agent/tool
            4. Build the dependency DAG
            5. Create the GraphPlan

        Args:
            description: Free-text workflow description.
            domain: Optional domain hint for agent matching.

        Returns:
            Executable GraphPlan with matched agents and dependencies.
        """
        logger.info("[ORCH-1.23] Compiling workflow from NL: %.80s...", description)

        # Step 1: Parse intent into discrete steps
        raw_steps = self._parse_steps(description)
        if not raw_steps:
            # Fallback: treat entire description as a single step
            raw_steps = [{"text": description, "is_parallel": False, "order": 0}]

        logger.debug("[ORCH-1.23] Parsed %d raw steps", len(raw_steps))

        # Step 2: Match each step to KG agents/tools
        matched_steps: list[ExecutionStep] = []
        for i, raw in enumerate(raw_steps):
            step_text = raw["text"]
            is_parallel = raw.get("is_parallel", False)

            # Find the best matching agent/tool from KG
            agent_id, tools = self._match_agent(step_text, domain)

            # Build dependencies
            depends_on = []
            if not is_parallel and i > 0:
                # Sequential: depends on previous step
                prev_step = raw_steps[i - 1]
                prev_agent, _ = self._match_agent(prev_step["text"], domain)
                depends_on = [prev_agent]

            step = ExecutionStep(
                id=agent_id,
                refined_subtask=step_text,
                parallel=is_parallel,
                depends_on=depends_on,
                access_list=depends_on if depends_on else [],
                timeout=120.0,
            )
            matched_steps.append(step)

        plan = GraphPlan(
            steps=matched_steps,
            metadata={
                "source": "nl_compiler",
                "original_description": description,
                "domain": domain,
                "step_count": len(matched_steps),
            },
        )

        logger.info(
            "[ORCH-1.23] Compiled workflow: %d steps, agents=%s",
            len(matched_steps),
            [s.id for s in matched_steps],
        )
        return plan

    async def compile_and_store(
        self,
        name: str,
        description: str,
        domain: str = "general",
    ) -> str:
        """Compile and persist a workflow in one step.

        CONCEPT:ORCH-1.23 — Compile + Store Pipeline

        Args:
            name: Human-readable workflow name.
            description: Natural language workflow description.
            domain: Optional domain hint.

        Returns:
            The stored workflow definition ID in the KG.
        """
        plan = await self.compile(description, domain)
        workflow_id = self.store.save_workflow(
            name=name,
            plan=plan,
            description=description,
            nl_spec=description,
            metadata={"domain": domain, "source": "nl_compiler"},
        )
        logger.info(
            "[ORCH-1.23] Workflow compiled and stored: id=%s, name=%s",
            workflow_id,
            name,
        )
        return workflow_id

    def find_and_load(self, query: str) -> GraphPlan | None:
        """Find a stored workflow by semantic similarity and load it.

        CONCEPT:ORCH-1.23 — Semantic Workflow Retrieval

        Searches stored workflows by description similarity and
        returns the best match as a GraphPlan.

        Args:
            query: Natural language description of desired workflow.

        Returns:
            Loaded GraphPlan or None if no match found.
        """
        matches = self.store.find_similar(query, top_k=1)
        if matches:
            name = matches[0].get("name", "")
            if name:
                return self.store.load_workflow(name)
        return None

    # -----------------------------------------------------------------------
    # Internal: NL Parsing
    # -----------------------------------------------------------------------

    def _parse_steps(self, description: str) -> list[dict[str, Any]]:
        """Parse a natural language description into discrete step intents.

        CONCEPT:ORCH-1.23 — Step Extraction

        Uses heuristic patterns and keyword detection to split
        free-text into ordered steps with parallel/sequential flags.

        Args:
            description: The raw NL description.

        Returns:
            List of dicts with keys: text, is_parallel, order.
        """
        steps: list[dict[str, Any]] = []

        # Strategy 1: Split on explicit numbering ("1. ... 2. ... 3. ...")
        numbered = re.split(r"\d+\.\s+", description)
        numbered = [s.strip() for s in numbered if s.strip()]
        if len(numbered) >= 2:
            for i, text in enumerate(numbered):
                steps.append(
                    {
                        "text": text.rstrip(".").strip(),
                        "is_parallel": self._is_parallel_step(text),
                        "order": i,
                    }
                )
            return steps

        # Strategy 2: Split on sequential keywords
        parts = re.split(
            r"\b(?:then|next|after that|finally|followed by)\b",
            description,
            flags=re.IGNORECASE,
        )
        parts = [p.strip().rstrip(",").strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            for i, text in enumerate(parts):
                steps.append(
                    {
                        "text": text.rstrip(".").strip(),
                        "is_parallel": self._is_parallel_step(text),
                        "order": i,
                    }
                )
            return steps

        # Strategy 3: Split on commas + "and"
        parts = re.split(r",\s*(?:and\s+)?", description)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
        if len(parts) >= 2:
            for i, text in enumerate(parts):
                steps.append(
                    {
                        "text": text.rstrip(".").strip(),
                        "is_parallel": self._is_parallel_step(text),
                        "order": i,
                    }
                )
            return steps

        return []

    def _is_parallel_step(self, text: str) -> bool:
        """Detect if a step description indicates parallel execution."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in _PARALLEL_KEYWORDS)

    # -----------------------------------------------------------------------
    # Internal: Agent Matching
    # -----------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        """Bounded embedding of ``text`` for semantic agent matching.

        Builds the embedding fn once per compiler (reused across steps) and bounds
        each embed via the same ``bounded_embed`` helper the Loop engine uses, so a
        slow/down endpoint (e.g. the GB10/vLLM power fault) degrades matching in
        seconds rather than stalling. Returns the vector, or ``None`` when
        embeddings are unavailable (caller falls back to the generic executor).
        """
        if self._embed_fn is _UNSET:
            try:
                from .enrichment.semantic import make_embed_fn

                self._embed_fn = make_embed_fn()
            except Exception:  # noqa: BLE001 — embedder unavailable ⇒ structural only
                self._embed_fn = None
                logger.info(
                    "[ORCH-1.23] embedding endpoint unavailable — workflow "
                    "compilation falls back to structural agent matching only"
                )
        if not self._embed_fn:
            return None
        from .research.search import _ACQUIRE_TIMEOUT_S, bounded_embed

        return bounded_embed(self._embed_fn, text, _ACQUIRE_TIMEOUT_S)

    def _match_agent(
        self,
        step_text: str,
        domain: str,
    ) -> tuple[str, list[str]]:
        """Match a step description to a KG-registered agent or tool.

        CONCEPT:ORCH-1.23 — Agent Resolution

        Uses the KG's semantic search and direct queries to find
        the best agent for a given step description.

        Args:
            step_text: The step's natural language description.
            domain: Domain hint for prioritization.

        Returns:
            Tuple of (agent_id, list_of_tool_names).
        """
        # Try KG semantic search first
        if self.engine and self.engine.backend:
            try:
                # Search for matching Server nodes (MCP servers)
                results = self.engine.backend.execute(
                    "MATCH (s:Server)-[:PROVIDES]->(r:CallableResource) "
                    "WHERE toLower($text) CONTAINS toLower(s.name) "
                    "OR toLower($text) CONTAINS toLower(r.name) "
                    "RETURN s.name AS server, COLLECT(r.name) AS tools "
                    "LIMIT 1",
                    {"text": step_text},
                )
                if results:
                    return results[0]["server"], results[0].get("tools", [])
            except Exception:
                pass  # nosec B110 — tool matching is best-effort

            # Bounded semantic fallback: rank candidate agent/tool nodes via the
            # engine's vector index (HNSW top-k + k property reads) instead of
            # ``search_hybrid``. The hybrid retriever materializes the WHOLE graph
            # (a per-label full-node scan, once per search table) which is O(graph)
            # and OOM-crashes the process on a large KG — pathological for
            # resolving a single step to an agent. ``backend.semantic_search`` is
            # bounded, and a bounded embed means a down embedder degrades in
            # seconds. (Fixes the compile_workflow child-crash root cause.)
            vec = self._embed(step_text)
            if vec is not None:
                search = getattr(self.engine.backend, "semantic_search", None)
                if callable(search):
                    try:
                        for r in search(vec, 5) or []:
                            if not isinstance(r, dict):
                                continue
                            rtype = str(
                                r.get("type")
                                or r.get("node_type")
                                or r.get("resource_type")
                                or r.get("_table_label")
                                or ""
                            )
                            if rtype in _AGENT_MATCH_TYPES:
                                name = r.get("name") or r.get("id")
                                if name:
                                    return str(name), []
                    except Exception:
                        pass  # nosec B110 — best-effort; falls through to generic executor

        # Last resort: use a generic executor
        return "executor", []
