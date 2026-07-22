"""The Loop engine controller — one hot path for every long-running objective.

CONCEPT:AU-KG.query.vendor-agnostic-traversal / KG-2.10 / KG-2.78 — research assimilation + orchestration synthesis,
generalized to advance **any** active :class:`~..research.loops.Loop` (research /
develop / skill) through ONE cycle. Formerly the "golden loop"; renamed because goals,
research topics, failure gaps and skill executions all collapse into the single Loop
unit the controller advances — there is no separate goal-runner or research-runner.

The research path composes existing primitives into one propose-only cycle that makes
the KG self-improving WITHOUT auto-merging anything:

    intake  → active Loops (research topics with no ``ADDRESSED_BY``; KG-2.78)
    acquire → semantically related sources for each topic (research/search)
    resolve → ``ADDRESSES`` edges source→topic so the loop converges
    reason  → OWL/RDF reasoning over the ecosystem, harvest extrapolations (KG-2.79)
    distill → ``SpecDraft`` markdown into ``.specify/specs/kg-distilled/`` (gated)
    synth   → a ``TeamSpec``/``AgentSpec`` proposal persisted to the KG

Every research artifact is a DRAFT/proposal: spec markdown under ``.specify/`` and KG
proposal nodes. No code execution, no PR merge, no edits outside ``.specify``.
Exposed on-demand (the ``graph_loops`` / ``graph_evolution`` MCP tools and the REST
twin) and via a throttled daemon tick.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from typing import Any

from agent_utilities.core.config import setting

from ..adaptation.topic_resolver import mark_addressed
from .loops import active_loops
from .search import acquire_for_topic_perspectival

logger = logging.getLogger(__name__)

# Node types whose (id, status, content_hash) define the assimilation input state —
# the cycle watermark. If unchanged since the last cycle, the graph-compute middle
# is skipped (idempotent: cost grows with the delta, not the corpus).
_WATERMARK_TYPES = {
    "sdd_feature",
    "capability",
    "article",
    "requirement",
    "decision",
    "concept",
    # Enterprise standardization inputs (CONCEPT:AU-KG.ontology.populated-at-import-real-3): new harvested assets or
    # edited standards re-trigger the standardize stage.
    "enterprise_resource",
    "enterprise_standard",
}
_WATERMARK_NODE = "assimilation:watermark"


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from this sync cycle, loop-running or not.

    The cycle is sync (daemon tick / MCP), but the research-intake mechanism is
    async. When no loop is running we ``asyncio.run``; when one is (an async MCP
    handler) we run it on a worker thread with its own loop so we never reenter a
    running loop. (CONCEPT:AU-KG.research.research-intelligence-loop)
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


class LoopController:
    """Advance the active Loops one propose-only cycle over the KG (CONCEPT:AU-KG.research.these-properties-carry)."""

    def __init__(
        self,
        engine: Any,
        *,
        codebase_root: str | None = None,
        propose_only: bool = True,
        auto_merge: bool | None = None,
        regression_check: Any = None,
        develop_runner: Any = None,
        skill_runner: Any = None,
        event_probes: dict[str, Callable[[], bool]] | None = None,
        skill_eval_targets_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.engine = engine
        self.codebase_root = codebase_root or setting("WORKSPACE_PATH") or "."
        # Execution backends for the non-research Loop kinds (CONCEPT:AU-KG.research.these-properties-carry L3),
        # injectable so the develop/skill stages are unit-testable without a real
        # subprocess / workflow engine. Defaults are wired lazily on first use.
        self._develop_runner = develop_runner
        self._skill_runner = skill_runner
        # SkillOpt-native ReflACT skill evolution (CONCEPT:AU-AHE.optimization.skillopt-native-reflact) —
        # discovery hook for which skills to evolve this cycle, injectable so the
        # stage is unit-testable without a real ``:SkillEvalSuite`` KG type existing
        # yet (see ``_discover_skill_evolution_targets``'s docstring).
        self._skill_eval_targets_provider = skill_eval_targets_provider
        # exit 8 EXTERNAL EVENT — registry of named real-world signal probes an
        # ``external_event`` Loop resolves by ``event_ref`` (e.g. "pr:owner/repo#42
        # merged"). Injectable so the exit is unit-testable without a live GitHub /
        # ticketing poll; ``run_loop(event_probe=...)`` overrides this per-run.
        self._event_probes: dict[str, Callable[[], bool]] = dict(event_probes or {})
        # Live-beacon + cycle id (CONCEPT:AU-KG.research.evolutionstate-live-surface-per) — set per run_one_cycle.
        self._beacon: Any = None
        self._cycle_id: str = ""
        # propose_only is always True in v1 — kept explicit so a future
        # human-approved apply path is a deliberate flip, never accidental.
        self.propose_only = propose_only
        # Governed auto-merge (CONCEPT:AU-AHE.assimilation.research-auto-merge) — OFF by default. Enabled
        # explicitly (auto_merge=True) or via KG_GOLDEN_AUTO_MERGE=1; only then
        # do high-quality, governance-valid proposals promote proposal→active.
        # ``regression_check`` gates failure-remediation merges (CONCEPT:AU-AHE.harness.failure-evolution)
        # against the originally observed failures — the failure-ingest tick passes
        # one so a remediation only auto-merges when it does not coincide with a
        # regression.
        from .auto_merge import GovernedAutoMerger, MergePolicy

        self._merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy.from_env(auto_merge),
            regression_check=regression_check,
        )

    # ------------------------------------------------------------------
    def _capability_search(self):
        """Build a ``(query, top_k) -> list[dict]`` capability search fn."""
        backend = getattr(self.engine, "backend", None)
        search = getattr(backend, "semantic_search", None)
        if not callable(search):
            return None
        from ..enrichment.semantic import make_embed_fn

        embed = make_embed_fn()

        def _fn(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            try:
                return search(embed([query])[0], top_k) or []
            except Exception:  # noqa: BLE001
                return []

        return _fn

    def run_one_cycle(
        self,
        *,
        max_topics: int = 5,
        distill: bool | None = None,
        synthesize: bool = True,
        assimilate: bool = True,
        breadth: bool | None = None,
        force_assimilate: bool = False,
        standardize: bool | None = None,
        topics: list[dict[str, Any]] | None = None,
        synthesize_search: bool = False,
        discover: bool | None = None,
        papers: list[dict[str, Any]] | None = None,
        reason: bool = True,
        tri_evolution: bool = False,
        focus_query: str = "",
        mine_discovery: bool | None = None,
        belief_revision: bool | None = None,
        insight_validation: bool | None = None,
        trace_mining: bool | None = None,
        skill_evolution: bool | None = None,
    ) -> dict[str, Any]:
        """Execute one cycle. Returns a structured, JSON-able report.

        Stages (each best-effort + timed; one failing stage never aborts the cycle):
        ``breadth`` (env ``KG_LOOP_BREADTH`` — ingest the OSS/repos/docs corpus,
        idempotent/content-addressed) → ``assimilate`` (dedup→gap→synergy→rank,
        idempotent via the state watermark) → ``reason`` → ``mine_discovery`` (env
        ``KG_LOOP_MINE_DISCOVERY``, default ON — the discovery-flywheel mining pass,
        CONCEPT:AU-KG.evolution.mining-flywheel) → ``trace_mining`` (env
        ``KG_LOOP_TRACE_MINING``, default ON — workstream C6, closed-loop agent
        mining: mines RunTrace/OutcomeEvaluation/ToolCall provenance for repeated
        FAILURE tool-call sequences and runs each through the SAME C4
        CandidateInsight→Claim→Validation→Action-gate pipeline; SAFETY-CRITICAL,
        see ``_run_trace_mining``'s docstring) → ``placement_control`` (typed
        ``PLACEMENT_CONTROL_LOOP_ENABLED``, default OFF — mines placement evidence
        and enters the approval-gated measured canary) → ``insight_validation`` (env
        ``KG_LOOP_INSIGHT_VALIDATION``, default ON — workstream C4, the Insight
        Engine closed loop: mined findings above a confidence floor become
        reviewable ``ClaimNode``s, gated by ``action_policy.decide()``) →
        ``belief_revision`` (env ``KG_LOOP_BELIEF_REVISION``, default ON —
        confidence propagation + light TMS over ``Belief`` nodes
        (CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision))
        → intake/acquire/resolve → ``distill`` (env ``KG_LOOP_DISTILL``) →
        ``synthesize``. The report carries
        a ``metrics`` block (per-stage timings + error count) and is persisted as an
        ``EvolutionCycle`` node for monitoring.
        """
        import time

        from agent_utilities.core.config import config

        if distill is None:
            distill = config.kg_loop_distill
        if breadth is None:
            breadth = config.kg_loop_breadth
        if standardize is None:
            standardize = config.kg_loop_standardize
        if discover is None:
            discover = config.kg_loop_discover
        if mine_discovery is None:
            mine_discovery = config.kg_loop_mine_discovery
        if belief_revision is None:
            belief_revision = config.kg_loop_belief_revision
        if insight_validation is None:
            insight_validation = config.kg_loop_insight_validation
        if trace_mining is None:
            trace_mining = config.kg_loop_trace_mining
        if skill_evolution is None:
            skill_evolution = config.kg_loop_skill_evolution

        report: dict[str, Any] = {
            "propose_only": self.propose_only,
            "topics_intake": 0,
            "topics_resolved": 0,
            "sources_linked": 0,
            "intake_papers": None,
            "breadth": None,
            "archivebox": None,
            "assimilate": None,
            "reason": None,
            "mine_discovery": None,
            "insight_validation": None,
            "trace_mining": None,
            "skill_evolution": None,
            "placement_control": None,
            "belief_revision": None,
            "standardize": None,
            "skill_proposals": None,
            "executed": None,
            "spec_drafts": [],
            "team": None,
            "search_tasks": None,
            "tri_evolution": None,
            "errors": [],
            "metrics": {"stage_ms": {}},
        }
        cycle_start = time.monotonic()

        # CONCEPT:AU-KG.research.evolutionstate-live-surface-per — live per-stage progress beacon. A single mutable node
        # updated at every stage boundary so the cycle is legible MID-FLIGHT (not only
        # at finalize): graph_loops(action="state") reports the current stage + why.
        import uuid as _uuid

        from .evolution_state import StageBeacon

        self._cycle_id = (
            f"evo_cycle_{time.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex}"
        )
        self._beacon = StageBeacon(
            self.engine,
            cycle_id=self._cycle_id,
            why=(focus_query or "").strip() or "advance active loops + mine open gaps",
        )
        self._beacon.enter("start")

        def _stage(name: str, fn):
            """Run a stage best-effort, capture timing + any error."""
            t0 = time.monotonic()
            self._beacon.enter(name)
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"{name}: {e}")
                logger.warning("golden-loop stage %s failed: %s", name, e)
                return None
            finally:
                report["metrics"]["stage_ms"][name] = round(
                    (time.monotonic() - t0) * 1000, 1
                )

        # -2. INTAKE PAPERS — discover + ingest research (scholarx → tiered KB
        # ingest → LLM concept/fact extraction) so the cycle is a research-pipeline
        # runner: the assimilate stage then matches the fresh papers against the
        # ecosystem. Opt-in (external calls) via KG_LOOP_DISCOVER; caller-supplied
        # ``papers`` always run. (CONCEPT:AU-KG.research.research-intelligence-loop)
        if discover or papers:
            report["intake_papers"] = _stage(
                "intake_papers", lambda: self._run_intake_papers(papers)
            )

        # -1. BREADTH — ingest the OSS/repos/docs corpus (idempotent; opt-in).
        if breadth:
            report["breadth"] = _stage("breadth", self._run_breadth)

        # -0.5 ARCHIVEBOX — pull preserved snapshots (delta) when an ArchiveBox
        # instance is wired. Default ON when configured (Native by default); the
        # URL's presence is the on-signal, the watermark keeps it idempotent.
        if (setting("ARCHIVEBOX_URL", default="") or "").strip():
            report["archivebox"] = _stage("archivebox", self._run_archivebox_intake)

        # -0.4 FRESHRSS — pull curated RSS items through the world-model relevance
        # gate (delta) when a FreshRSS instance is wired. Default ON when configured;
        # the URL's presence is the on-signal, the watermark keeps it idempotent, and
        # the gate keeps it selective (only KG-relevant/novel items land). (KG-2.116)
        if (setting("FRESHRSS_URL", default="") or "").strip():
            report["freshrss"] = _stage("freshrss", self._run_freshrss_intake)

        # 0. ASSIMILATE — graph-compute middle (dedup/gap/synergy/rank), idempotent.
        if assimilate:
            report["assimilate"] = _stage(
                "assimilate", lambda: self._run_assimilate(force=force_assimilate)
            )

        # 0a. REASON — OWL/RDF reasoning over the ONE ecosystem ontology; harvest the
        # extrapolated relationships and surface cross-domain inferences as fresh
        # research Loops (CONCEPT:AU-KG.research.best-effort-lightweight-never). Best-effort + lightweight; never blocks.
        if reason:
            report["reason"] = _stage("reason", self._run_reason)

        # 0a1. MINE DISCOVERY — the discovery-flywheel mining pass (CONCEPT:AU-KG.evolution.mining-flywheel):
        # association-rule mining over Capability/Concept co-occurrence, an anomaly
        # pass over capability-coverage divergence, and graph_learn link-prediction
        # over Concept relations — each writes back typed KG nodes (:AssociationRule/
        # :Anomaly/:PredictedEdge) for a human/agent to review (propose-only; never
        # auto-merges). Placed after ``reason`` so mining sees OWL-inferred edges
        # already materialized, and before ``synthesize`` so the team/spec synthesis
        # stage can eventually read the freshly-mined nodes. Best-effort + gated
        # (KG_LOOP_MINE_DISCOVERY, default ON — sub-steps degrade to empty/no-op on a
        # no-mining engine build, so it's safe to leave on everywhere).
        if mine_discovery:
            report["mine_discovery"] = _stage(
                "mine_discovery", self._run_mine_discovery
            )

        # 0a1.4 INSIGHT VALIDATION — the Insight Engine closed loop (workstream C4,
        # CONCEPT:AU-KG.evolution.insight-engine-closed-loop): Mine → CandidateInsight →
        # EvidenceBundle → Claim → Validation (REUSES promotion_governance +
        # capability_ratchet as-is) → Action gate (REUSES action_policy.decide(),
        # kind="promote_mined_claim", shipped default approval_required —
        # SAFETY-CRITICAL, see ``_run_insight_validation``'s docstring). Only runs
        # when mine_discovery actually produced a report this cycle (it consumes
        # that report's mined findings — there's nothing to validate otherwise).
        # Best-effort + gated (KG_LOOP_INSIGHT_VALIDATION, default ON — the stage
        # is itself propose-only regardless of the flag; see the config field's
        # docstring in ``core/config.py``).
        if insight_validation and report.get("mine_discovery"):
            report["insight_validation"] = _stage(
                "insight_validation",
                lambda: self._run_insight_validation(report["mine_discovery"]),
            )

        # 0a1.45 TRACE MINING — closed-loop agent mining (workstream C6,
        # CONCEPT:AU-KG.evolution.insight-engine-closed-loop): mines RunTrace/
        # OutcomeEvaluation/ToolCall provenance for repeated FAILURE tool-call
        # sequences (``trace_pattern_miner``) and runs each mined pattern through
        # the SAME C4 CandidateInsight→Claim→Validation→Action-gate pipeline —
        # REUSES ``action_policy.decide(kind="route_policy_update")``, shipped
        # default approval_required. SAFETY-CRITICAL: see ``_run_trace_mining``'s
        # docstring — ``OutcomeRouter.record()`` is NEVER called before that
        # ``decide()`` call on any path (tests/unit/knowledge_graph/
        # test_trace_pattern_miner.py::test_gate_runs_before_any_outcome_record).
        # Best-effort + gated (KG_LOOP_TRACE_MINING, default ON — the stage is
        # itself propose-only regardless of the flag; see the config field's
        # docstring in ``core/config.py``).
        if trace_mining:
            report["trace_mining"] = _stage("trace_mining", self._run_trace_mining)

        # 0a1.46 PLACEMENT CONTROL — workload-aware placement mining (X-5).
        # This is the one automatic caller governed by the typed opt-in. The
        # controller itself reuses the same propose -> ActionPolicy -> measured
        # canary -> promote/rollback spine as the explicit graph_loops action;
        # approval_required remains the shipped mutation posture.
        if config.placement_control_loop_enabled:
            from .placement_mining import placement_control_loop

            report["placement_control"] = _stage(
                "placement_control",
                lambda: placement_control_loop(self.engine, enabled=True),
            )

        # 0a1.5 BELIEF REVISION — confidence propagation + light TMS (CONCEPT:AU-KG.
        # adaptation.confidence-propagation-belief-revision, workstream C2): recomputes
        # every ``Belief`` node's confidence from its support/contradiction
        # neighborhood (fresh ``ContradictionDetector`` friction + already-recorded
        # edges), persisting each outcome as a ``:BeliefRevisionProposal`` — never a
        # mutation of the live belief (propose-only; the Critic flags, it does not
        # arbitrate). Placed right after ``mine_discovery`` so revision sees any
        # newly-mined/reasoned structure first. Best-effort + gated
        # (KG_LOOP_BELIEF_REVISION, default ON — degrades to a no-op ``skipped``
        # result with fewer than 2 Belief nodes, so it's safe to leave on everywhere).
        if belief_revision:
            report["belief_revision"] = _stage(
                "belief_revision", self._run_belief_revision
            )

        # 0a2. DISTILL SKILLS — turn the mapped processes of ALL connected systems
        # (egeria/leanix/aris/camunda) into propose-only atomic-skill and
        # skill-workflow PROPOSALS (CONCEPT:AU-KG.ontology.connector-agnostic-proposal/2.83). Connector-agnostic over
        # the ontology, default-ON, propose-only (nothing lands in any repo). Best-
        # effort: a failing stage never aborts the cycle.
        report["skill_proposals"] = _stage("distill_skills", self._distill_skills)

        # 0a3. SKILL EVOLUTION — the SkillOpt-native ReflACT cycle
        # (CONCEPT:AU-AHE.optimization.skillopt-native-reflact): Rollout->Reflect->
        # Aggregate/Select/Update->Evaluate over any registered skill-eval target,
        # gated onto "active" only by beating the incumbent on a held-out benchmark
        # AND action_policy.decide(kind="promote_skill_version") (shipped default
        # approval_required — a benchmark win alone never auto-promotes). Sibling to
        # distill_skills (which proposes brand-new skills from connector-mapped
        # processes); this stage evolves EXISTING skills. Best-effort + gated
        # (KG_LOOP_SKILL_EVOLUTION, default ON — degrades to a clean no-op with zero
        # registered skill-eval targets, mirroring belief_revision's <2-node no-op).
        if skill_evolution:
            report["skill_evolution"] = _stage(
                "skill_evolution", self._run_skill_evolution
            )

        # 0b. STANDARDIZE — enterprise standardization + consolidation (CONCEPT:AU-KG.ontology.populated-at-import-real-3),
        # propose-only. Gated (KG_LOOP_STANDARDIZE) since it requires a harvested
        # enterprise estate; idempotent (CONFORMS_TO/ABSORBED_INTO cleared on re-write).
        if standardize:
            report["standardize"] = _stage("standardize", self._run_standardize)

        # 1. INTAKE — every active Loop the engine should advance (CONCEPT:AU-KG.research.these-properties-carry):
        # research/develop/skill objectives + autonomous gaps, each carrying its
        # ``kind`` so later stages dispatch correctly. Caller-supplied ``topics``
        # (e.g. the failure-ingest tick's just-materialized failure_gap loops)
        # bypass the generic ``active_loops`` scan so a brand-new gap is addressed
        # deterministically instead of competing for a slot. (CONCEPT:AU-AHE.harness.failure-evolution)
        if topics is not None:
            topics = topics[:max_topics] if max_topics else list(topics)
            report["metrics"]["stage_ms"]["intake"] = 0.0
        else:
            topics = (
                _stage("intake", lambda: active_loops(self.engine, max_topics)) or []
            )
        # Focus-query biasing: a caller-supplied query becomes a prioritized research
        # topic for this cycle so acquire/resolve converges on it first (CONCEPT:AU-KG.research.research-intelligence-loop).
        fq = (focus_query or "").strip()
        if fq:
            topics = [
                {"id": f"focus:{fq}", "name": fq, "kind": "research"},
                *topics,
            ]
            if max_topics:
                topics = topics[:max_topics]
        report["topics_intake"] = len(topics)

        # 1b. EXECUTE — advance develop/skill Loops one step through the SAME hot
        # path (CONCEPT:AU-KG.research.these-properties-carry L3): develop runs act→validate, skill runs its
        # skill/skill-workflow. Research loops fall through to acquire_resolve below.
        exec_loops = [t for t in topics if t.get("kind", "research") != "research"]
        if exec_loops:
            report["executed"] = _stage(
                "execute", lambda: self._run_execute_loops(exec_loops)
            )

        if topics:
            # 2–3. ACQUIRE related sources + RESOLVE (ADDRESSES) so the loop converges.
            def _acquire_resolve():
                from ..enrichment.semantic import make_embed_fn
                from .search import _ACQUIRE_TIMEOUT_S, bounded_embed

                # Build the embedder ONCE per cycle (not per topic), then a single
                # bounded probe: if embeddings are down, skip the whole stage in
                # seconds instead of paying the per-topic timeout for every topic.
                embed_fn = make_embed_fn()
                if bounded_embed(embed_fn, "ping", _ACQUIRE_TIMEOUT_S) is None:
                    report["errors"].append(
                        "acquire_resolve:embedding endpoint unavailable — stage skipped"
                    )
                    return
                for t in topics:
                    # Only RESEARCH loops are resolved by acquiring sources; develop/
                    # skill loops are advanced by their own stages (CONCEPT:AU-KG.research.these-properties-carry,
                    # L3) and must NOT be marked addressed by semantic sources here.
                    if t.get("kind", "research") != "research":
                        continue
                    srcs = acquire_for_topic_perspectival(
                        self.engine, t, embed_fn=embed_fn
                    )
                    if srcs:
                        n = mark_addressed(
                            self.engine, t["id"], srcs, source="loop_engine"
                        )
                        if n:
                            report["topics_resolved"] += 1
                            report["sources_linked"] += n

            _stage("acquire_resolve", _acquire_resolve)

            # 4. DISTILL spec drafts (gated; propose-only → .specify/).
            if distill:
                report["spec_drafts"] = (
                    _stage("distill", lambda: self._distill_specs(topics)) or []
                )

            # 5. SYNTHESIZE a team proposal for the open topics (propose-only).
            if synthesize:
                report["team"] = _stage(
                    "synthesize", lambda: self._synthesize_team(topics)
                )

        # 6. SELF-PLAY SEARCH-TASK SYNTHESIS (CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72) — build
        # shortcut-resistant deep-search tasks from the evidence graph and draft a
        # training corpus (propose-only). Opt-in: it does not depend on open
        # topics and is skipped by default to keep the zero-infra cycle cheap.
        if synthesize_search:
            report["search_tasks"] = _stage(
                "synthesize_search", self._synthesize_search_tasks
            )

        # 7. HYBRID TRI-EVOLUTION (CONCEPT:AU-AHE.harness.co-evolve-research) — co-evolve the research
        # proposer/solver/judge and report the ablation that proves co-evolution
        # beats solo (HOTE arXiv:2606.13710). Opt-in (off by default): the CPU
        # ablation harness runs without LLMs; the LLM-backed integration of the
        # real OntologyReasoningDriver/ARA/ConceptMatcher is the production path.
        if tri_evolution:
            report["tri_evolution"] = _stage("tri_evolution", self._run_tri_evolution)

        self._finalize_metrics(report, cycle_start)
        return report

    # ------------------------------------------------------------------
    def _run_tri_evolution(self, *, rounds: int = 20) -> dict[str, Any]:
        """Run the HOTE co-evolution ablation harness (CONCEPT:AU-AHE.harness.co-evolve-research).

        Returns the joint-vs-solo final skills, the indispensability verdict, and
        the marginal adaptation-speed gain of joint co-evolution. CPU-only and
        deterministic; the real LLM-backed proposer/solver/judge plug into
        ``HybridTriEvolutionController`` via its injectable hooks.
        """
        from agent_utilities.harness.hote_tri_evolution import (
            HybridTriEvolutionController,
        )

        return HybridTriEvolutionController().run_ablation(rounds=rounds)

    # ------------------------------------------------------------------
    def _cheap_input_count(self) -> int | None:
        """Input-scoped node count via Cypher — cheap (no embedding transfer).

        Returns the count of assimilation *input* node types (so the cycle's own
        outputs — proposals/plans/watermark/cycle nodes — don't perturb it), or
        ``None`` when ``query_cypher`` is unavailable (→ caller falls back to the
        full hash). Optimization: the unchanged-graph skip path avoids fetching all
        ~5k embedded nodes (~10s) — it just runs one count query.

        Caveat: a pure in-place content update with no node-count change is not
        detected by the count alone; use ``force`` to override when needed.
        """
        q = getattr(self.engine, "query_cypher", None)
        if not callable(q):
            return None
        casings: set[str] = set()
        for t in _WATERMARK_TYPES:
            casings.update({t, t.upper(), t.capitalize(), t.title()})
        # Inline the type literals (controlled enum casings — no user input) since
        # this backend does not reliably bind list params.
        type_list = ", ".join("'" + t.replace("'", "") + "'" for t in sorted(casings))
        try:
            rows = q(
                f"MATCH (n) WHERE n.node_type IN [{type_list}] RETURN count(n) AS c"
            )
            if not rows:
                return None
            row = rows[0]
            if isinstance(row, dict):
                val = row.get("c")
            elif isinstance(row, list | tuple):
                val = row[0] if row else None
            else:
                val = row
            return int(val) if val is not None else None
        except Exception:  # noqa: BLE001 - fall back to the full hash
            return None

    def _state_watermark(self) -> str:
        """Watermark of the assimilation input state. Unchanged ⇒ nothing to do.

        Fast path: an input-scoped count via Cypher (no embedding fetch). Fallback:
        hash of ((id, status, content_hash)) over the input node types, fetched via
        the engine's BOUNDED per-label index (``iter_typed_nodes``, the same helper
        ``dedup_features`` uses) — never an unscoped ``graph.nodes(data=True)``,
        which at ecosystem scale is a whole-graph ``GetNodes`` dump the response
        guard refuses (``RESULT_TOO_LARGE``, CONCEPT:EG-KG.ingest.resets-socket-so-assimilation).
        """
        c = self._cheap_input_count()
        if c is not None:
            return f"count:{c}"
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return ""
        from ..assimilation.dedup import iter_typed_nodes

        try:
            node_iter = iter_typed_nodes(graph, tuple(_WATERMARK_TYPES))
        except Exception:  # noqa: BLE001 - defensive; keep the watermark best-effort
            return ""
        items = sorted(
            (nid, str(d.get("status", "")), str(d.get("content_hash", "")))
            for nid, d in node_iter
            if isinstance(d, dict)
        )
        return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]

    def _load_watermark(self) -> str | None:
        """Read the persisted watermark hash — a BOUNDED single-id lookup.

        This runs on every ``assimilate`` call (before dedup/gap/synergy/rank even
        start), so it must never fall back to an unscoped ``graph.nodes(data=True)``
        on a live engine: at ecosystem scale that is a whole-graph ``GetNodes`` dump
        that the response guard refuses outright (``RESULT_TOO_LARGE``,
        CONCEPT:EG-KG.ingest.resets-socket-so-assimilation) — reproduced live at
        139,657 nodes > the 50,000 cap. Try the same bounded single-row Cypher
        id-match already used elsewhere for this exact shape (e.g.
        ``change_publisher.py``, ``durable_outcome_store.py``:
        ``MATCH (n) WHERE n.id = $id RETURN ... LIMIT 1``). A successful call is
        trusted even when it returns no row — "no watermark persisted yet" (the
        first cycle) is a real, valid answer here, unlike a whole-graph scan there
        is no cheaper way to get. Fall back to the ``graph.nodes()`` scan only when
        ``query_cypher`` itself is unavailable or raises (e.g. a minimal test
        double with no Cypher support at all).
        """
        q = getattr(self.engine, "query_cypher", None)
        if callable(q):
            try:
                rows = q(
                    "MATCH (n) WHERE n.id = $id RETURN n.hash AS hash LIMIT 1",
                    {"id": _WATERMARK_NODE},
                )
            except Exception:  # noqa: BLE001 - fall back to the full scan
                rows = None
            else:
                if not rows:
                    return None
                row = rows[0]
                if isinstance(row, dict):
                    return row.get("hash")
                if isinstance(row, list | tuple) and row:
                    return row[0]
                return None
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return None
        try:
            for nid, d in graph.nodes(data=True):
                if nid == _WATERMARK_NODE and isinstance(d, dict):
                    return d.get("hash")
        except TypeError:
            return None
        return None

    def _run_assimilate(
        self,
        *,
        force: bool = False,
        restrict_to: set[str] | None = None,
        matrix_node_id: str = "feature_matrix:latest",
    ) -> dict[str, Any]:
        """Run dedup → auto-satisfy → synergy → rank over the feature graph.

        Idempotent: if the input watermark is unchanged since the last cycle (and
        not ``force``), skip the work. The ranked gaps are exclusion-filtered to
        ``open_features`` (satisfied/superseded/implemented features are never
        re-proposed). CONCEPT:AU-KG.query.vendor-agnostic-traversal.

        ``restrict_to`` scopes satisfy/synergy/rank/matrix to a feature set (e.g. a
        research cohort's sources, CONCEPT:AU-KG.ingest.fetch-only-requested-ids) so per-cohort synthesis is
        O(cohort), and the matrix is materialized to ``matrix_node_id`` (a cohort
        gets its own node instead of overwriting the ecosystem-wide one).
        """
        pre = self._state_watermark()
        # The watermark guards the WHOLE-graph cycle; a scoped (cohort) pass always
        # runs — its delta isn't reflected in the global watermark.
        if not force and restrict_to is None and pre and pre == self._load_watermark():
            return {"skipped": True, "reason": "unchanged", "watermark": pre}

        from ..assimilation import (
            ConceptMatcher,
            dedup_features,
            enrich_concepts,
            rank_features,
            synergy_bundles,
        )
        from ..assimilation.gap_analysis import _CONCEPT_TYPES, _FEATURE_TYPES
        from ..core.ingest_profile import stage as _pstage  # OS-5.70 per-stage timing

        # Feature dedup is a WHOLE-GRAPH ecosystem op (SUPERSEDES clustering); skip it
        # for a SCOPED (cohort) pass (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) so finalize stays O(cohort) —
        # a cohort's matrix doesn't need ecosystem-wide dedup.
        with _pstage("dedup"):
            dedup = None if restrict_to is not None else dedup_features(self.engine)
        # Ensure the ecosystem Concept registry is embedded so the matcher's
        # retrieval stage has vectors (idempotent; skips already-embedded). Then
        # the robust ConceptMatcher (id + embedding-recall + LLM-judge) decides
        # covered (SATISFIED_BY) vs related-novel (RELATES_TO) — replacing the old
        # single-cosine auto_satisfy that recognised 0/21. (CONCEPT:AU-KG.ingest.world-model-gate)
        # WHOLE-GRAPH op (iterates every node to find vectorless concepts): skip it
        # for a SCOPED (cohort) pass (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) — the registry is embedded
        # ecosystem-wide once, and the matcher recalls from the engine HNSW; a cohort
        # finalize must not re-scan the whole graph (which resets the socket at scale).
        if restrict_to is None:
            with _pstage("enrich_concepts"):
                enrich_concepts(self.engine)
        with _pstage("satisfy"):
            gap = ConceptMatcher().satisfy(
                self.engine,
                feature_types=_FEATURE_TYPES,
                concept_types=_CONCEPT_TYPES,
                restrict_to=restrict_to,
            )
        with _pstage("synergy_rank"):
            syn = synergy_bundles(self.engine, restrict_to=restrict_to)
            ranked = rank_features(
                self.engine,
                feature_ids=(list(restrict_to) if restrict_to is not None else None),
            )

        # Materialize the comparative feature/innovation matrix from the now-
        # assimilated graph (CONCEPT:AU-KG.research.default-so-every-cycle) — default-ON so every cycle emits the
        # deliverable: coverage rows, leverage-ranked novel gaps, and the cross-source
        # synergy bundles (the combine-to-surpass candidates).
        matrix_summary: dict[str, Any] = {}
        try:
            from datetime import UTC, datetime

            from ..assimilation.feature_matrix import build_feature_matrix, materialize

            with _pstage("matrix"):
                matrix = build_feature_matrix(
                    self.engine,
                    generated_at=datetime.now(UTC).isoformat(),
                    restrict_to=restrict_to,
                )
                matrix_summary = materialize(
                    self.engine, matrix, node_id=matrix_node_id
                )
        except Exception as e:  # noqa: BLE001 — best-effort, never fails the cycle
            logger.debug("feature matrix materialize failed: %s", e)

        watermark = self._state_watermark()
        try:
            self.engine.add_node(
                _WATERMARK_NODE,
                "assimilation_watermark",
                properties={"hash": watermark},
            )
        except Exception as e:  # noqa: BLE001 - watermark persistence is best-effort
            logger.debug("watermark persist failed: %s", e)

        return {
            "skipped": False,
            "duplicates_superseded": dedup.duplicates_superseded if dedup else 0,
            "auto_satisfied": gap.satisfied,
            "related": gap.related,
            "used_llm": gap.used_llm,
            "synergy_bundles": len(syn.bundles),
            "open_gaps": len(ranked),
            # exclusion-filtered, leverage-ranked work-list for plan synthesis (VU-8)
            "ranked_gaps": [
                {
                    "feature_id": r.feature_id,
                    "score": r.score,
                    "sources": r.source_count,
                }
                for r in ranked[:20]
            ],
            "feature_matrix": matrix_summary,
            "watermark": watermark,
        }

    def _run_intake_papers(self, papers: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Discover + ingest research papers as the cycle's front stage.

        Delegates to the ``ResearchPipelineRunner`` intake mechanism (scholarx
        discovery → tiered KB ingest → LLM concept/fact extraction → OWL enrich),
        so the unified cycle is a research-pipeline runner: the ``assimilate`` stage
        then matches the freshly-ingested papers against the ecosystem Concept
        registry via the ConceptMatcher. (CONCEPT:AU-KG.research.research-intelligence-loop)
        """
        from agent_utilities.automation.research_pipeline import ResearchPipelineRunner

        runner = ResearchPipelineRunner(engine=self.engine)
        rep = _run_coro(runner.run_daily_pipeline(papers=papers))
        return {
            "papers_discovered": rep.papers_discovered,
            "papers_relevant": rep.papers_relevant,
            "papers_marginal": rep.papers_marginal,
            "papers_already_known": rep.papers_already_known,
            "owl_inferences": rep.owl_inferences,
            "errors": rep.errors[:5],
        }

    def _run_reason(self) -> dict[str, Any]:
        """Run OWL/RDF reasoning over the ecosystem; harvest the extrapolation.

        The ontology-driven research engine (CONCEPT:AU-KG.research.best-effort-lightweight-never): reason over the one
        ecosystem knowledge-graph and turn the newly-inferred cross-domain
        relationships into fresh research Loops for subsequent cycles — so research
        compounds on what reasoning discovers, not just what was ingested.
        """
        from .ara.reasoning_driver import OntologyReasoningDriver

        harvest = OntologyReasoningDriver(self.engine).extrapolate()
        return {
            "inferred": len(harvest.inferred_edges),
            "new_topics": len(harvest.new_topics),
            "stats": harvest.stats,
            "error": harvest.error,
        }

    # -- discovery-flywheel mining (CONCEPT:AU-KG.evolution.mining-flywheel) ---------------- #
    def _run_mine_discovery(self) -> dict[str, Any]:
        """Discovery-flywheel mining pass over the KG (CONCEPT:AU-KG.evolution.mining-flywheel).

        The agent-utilities-evolution flywheel's data-mining stage (see
        ``plans/data-mining-capabilities-plan.md`` §5): three independent,
        best-effort passes over the engine's ``graph_mine``/``graph_learn``
        surfaces, called through the SAME ``_invoke`` helper the ``graph_mine``/
        ``graph_learn`` MCP tools use (:func:`agent_utilities.mcp.tools.
        engine_surface_tools._invoke`) — so this degrades exactly like those
        tools on a no-mining engine build (empty/no-op, never raises):

        1. **association** — mines each ``Capability`` node's full outbound
           neighborhood (``SATISFIED_BY``/``RELATES_TO`` concept edges,
           ``DERIVED_FROM_RESEARCH`` article edges, ``HAS_SYNERGY_WITH`` sibling-
           capability edges — whatever it points to) as one "transaction",
           discovering concept(+article+capability) co-occurrence rules that
           auto-suggest implementations (plan §5: "papers citing X + concept Y
           ⇒ capability Z is usually implemented"). ``writeback=True`` ⇒
           ``:AssociationRule`` nodes, feeding evolution-skill step 5 (SDD Plan
           Generation).
           SIMPLIFICATION vs. the ``docs/mining.md`` toy example: this
           production schema links features→concepts via ``SATISFIED_BY``/
           ``RELATES_TO`` (see ``assimilation/gap_analysis.py``/``concept_matcher.py``)
           rather than the doc's direct ``Paper --TOUCHES--> Concept`` /
           ``Paper --IMPLEMENTS--> Capability`` edges — there is no direct
           paper→capability edge in this schema — so we mine a ``Capability``
           node's outbound superset rather than a literal ``Paper`` neighborhood.
        2. **anomaly** — a simplified coverage-divergence proxy: for each
           ``Capability`` node, feature = count of outbound ``SATISFIED_BY``
           edges (how many concepts it actually satisfies). Capabilities whose
           covered-concept count is a statistical outlier (usually near-zero)
           are flagged as divergent/under-implemented. This is a defensible
           simplification of "concepts/tests touching each capability" (no
           test-coverage edge exists in this schema yet) rather than a literal
           test-touch count. ``writeback=True`` ⇒ ``:Anomaly`` nodes.
        3. **predicted_edges** — ``graph_learn`` fit→predict over
           ``Concept``-labeled nodes (concept↔concept link prediction).
           SCOPE CUT: ``graph_learn.fit`` builds one homogeneous ``node_label``
           vertex set, so a true cross-type concept↔capability predictor isn't
           expressible in this call shape yet; we predict missing concept↔
           concept relations instead, which still serves plan §5's "suggests
           missing concept relations." ``writeback=True`` ⇒ ``:PredictedEdge``
           nodes.

        Each sub-step is wrapped in its own try/except (mirroring the ``_stage``
        best-effort philosophy at the sub-step level, since ``_stage`` only wraps
        the whole method) so one failing pass never blocks the others. Returns a
        compact, JSON-able summary (counts + a handful of examples) — never the
        raw mining payloads. Propose-only: write-back only materializes
        descriptive KG facts for later review (``graph_query`` / a future review
        stage); it never triggers SDD plan creation, code edits, or a merge.
        """
        errors: list[str] = []
        association = self._mine_association_rules(errors)
        anomalies = self._mine_capability_anomalies(errors)
        predicted = self._mine_predicted_edges(errors)
        return {
            "association_rules": association,
            "anomalies": anomalies,
            "predicted_edges": predicted,
            "errors": errors,
        }

    @staticmethod
    def _mining_ok(payload: Any) -> bool:
        """``True`` when an ``_invoke`` JSON payload is a live (non-degraded) result."""
        return isinstance(payload, dict) and "error" not in payload

    def _mine_association_rules(self, errors: list[str]) -> dict[str, Any]:
        """Association-rule mining over ``Capability`` neighborhoods (see class docstring)."""
        import json as _json

        from agent_utilities.mcp.tools.engine_surface_tools import _invoke

        try:
            raw = _invoke(
                surface="mining",
                action="associate",
                graph="",
                candidates=(("mining", "associate"),),
                params={
                    "source": {"node_label": "Capability", "direction": "out"},
                    "min_support": 0.1,
                    "min_confidence": 0.6,
                    "algorithm": "fpgrowth",
                    "writeback": True,
                },
            )
            payload = _json.loads(raw)
        except Exception as e:  # noqa: BLE001 — never let mining break the cycle
            errors.append(f"mine_association: {e}")
            return {"count": 0, "examples": []}
        if not self._mining_ok(payload):
            errors.append(f"mine_association: {payload.get('error') or payload}")
            return {"count": 0, "examples": []}
        result = payload.get("result") or {}
        rules = result.get("rules") or []
        examples = [
            {
                "antecedent": r.get("antecedent"),
                "consequent": r.get("consequent"),
                "confidence": r.get("confidence"),
                "lift": r.get("lift"),
            }
            for r in rules[:5]
        ]
        return {
            "count": len(rules),
            "examples": examples,
            "written_back": result.get("written_back"),
        }

    def _mine_capability_anomalies(self, errors: list[str]) -> dict[str, Any]:
        """Coverage-divergence anomaly pass over ``Capability`` nodes (see class docstring)."""
        import json as _json

        from agent_utilities.mcp.tools.engine_surface_tools import _invoke

        try:
            rows = (
                self.engine.query_cypher(
                    "MATCH (cap:Capability) "
                    "OPTIONAL MATCH (cap)-[:SATISFIED_BY]->(c:Concept) "
                    "RETURN cap.id AS id, count(c) AS covered LIMIT $limit",
                    {"limit": 200},
                )
                or []
            )
        except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
            errors.append(f"mine_anomaly:query: {e}")
            return {"count": 0, "examples": []}
        ids = [r["id"] for r in rows if isinstance(r, dict) and r.get("id")]
        values = [
            float(r.get("covered") or 0)
            for r in rows
            if isinstance(r, dict) and r.get("id")
        ]
        if len(values) < 3:
            # Not enough population for a meaningful outlier pass — empty, not an error.
            return {"count": 0, "examples": []}
        try:
            raw = _invoke(
                surface="mining",
                action="anomaly",
                graph="",
                candidates=(("mining", "anomaly"),),
                params={"values": values, "algorithm": "zscore", "writeback": True},
            )
            payload = _json.loads(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"mine_anomaly:invoke: {e}")
            return {"count": 0, "examples": []}
        if not self._mining_ok(payload):
            errors.append(f"mine_anomaly: {payload.get('error') or payload}")
            return {"count": 0, "examples": []}
        result = payload.get("result") or {}
        rows_out = result.get("rows") or []
        examples: list[dict[str, Any]] = []
        for idx, row in enumerate(rows_out):
            if not (isinstance(row, dict) and row.get("is_anomaly")):
                continue
            examples.append(
                {
                    "capability": ids[idx] if idx < len(ids) else row.get("id"),
                    "covered_concepts": values[idx] if idx < len(values) else None,
                    "anomaly_score": row.get("anomaly_score"),
                }
            )
            if len(examples) >= 5:
                break
        return {
            "count": int(result.get("n_anomalies") or len(examples)),
            "examples": examples,
        }

    def _mine_predicted_edges(self, errors: list[str]) -> dict[str, Any]:
        """``graph_learn`` fit→predict link prediction over ``Concept`` nodes (see class docstring)."""
        import json as _json

        from agent_utilities.mcp.tools.engine_surface_tools import _invoke

        try:
            raw = _invoke(
                surface="graphlearn",
                action="fit",
                graph="",
                candidates=(("graphlearn", "fit"),),
                params={
                    "node_label": "Concept",
                    "direction": "any",
                    "epochs": 50,
                    "writeback": False,
                },
            )
            fit_payload = _json.loads(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"mine_predict:fit: {e}")
            return {"count": 0, "examples": []}
        if not self._mining_ok(fit_payload):
            errors.append(
                f"mine_predict:fit: {fit_payload.get('error') or fit_payload}"
            )
            return {"count": 0, "examples": []}
        model = (fit_payload.get("result") or {}).get("model")
        if not model:
            errors.append("mine_predict:fit: no model returned")
            return {"count": 0, "examples": []}
        try:
            raw = _invoke(
                surface="graphlearn",
                action="predict",
                graph="",
                candidates=(("graphlearn", "predict"),),
                params={
                    "model": model,
                    "node_label": "Concept",
                    "top_k": 10,
                    "writeback": True,
                },
            )
            predict_payload = _json.loads(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"mine_predict:predict: {e}")
            return {"count": 0, "examples": []}
        if not self._mining_ok(predict_payload):
            errors.append(
                f"mine_predict:predict: {predict_payload.get('error') or predict_payload}"
            )
            return {"count": 0, "examples": []}
        result = predict_payload.get("result") or {}
        predicted = result.get("predicted") or []
        return {"count": len(predicted), "examples": predicted[:5]}

    # -- X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance) -- #
    def _register_derived_claim(
        self, claim: Any, errors: list[str], context: str
    ) -> None:
        """Thin ``self.engine``-bound adapter over the shared ``candidate_insight.
        register_claim_materialization`` writeback seam — see that function's
        docstring for the full contract. ``_run_insight_validation`` (association/
        anomaly/predicted-edge findings) and ``_run_trace_mining`` (mined
        sequential-pattern findings) both call this so a new mining family gets
        the same reversible-derived-data coverage for free; ``placement_mining.
        run_placement_mining_cycle`` (a free function, not a ``LoopController``
        method) calls the shared function directly with its own ``engine``.
        """
        from .candidate_insight import register_claim_materialization

        register_claim_materialization(self.engine, claim, errors, context=context)

    # -- Insight Engine closed loop (CONCEPT:AU-KG.evolution.insight-engine-closed-loop, workstream C4) -- #
    def _run_insight_validation(self, mine_result: dict[str, Any]) -> dict[str, Any]:
        """Mine → CandidateInsight → EvidenceBundle → Claim → Validation → Action gate.

        Workstream C4 — the Insight Engine closed loop. ``mine_discovery`` already
        writes back typed, descriptive ``:AssociationRule``/``:Anomaly``/
        ``:PredictedEdge`` nodes; nothing turns one into something the rest of the
        epistemic substrate (C1 ``EvidenceBundle``, C2 belief-revision, the AHE-3.20
        promotion-governance stack) can reason about. This stage closes that loop:

        1. **CandidateInsight** (:mod:`.candidate_insight`) extracts each mined
           finding's real confidence signal (never fabricated — see that module's
           docstring) and drops anything below :data:`~.candidate_insight.
           CONFIDENCE_FLOOR` — a below-floor finding is counted but NEVER
           materialized as a ``ClaimNode``.
        2. **EvidenceBundle** (C1) packages the raw finding as the claim's
           evidence trail (audit-visible, nothing silently dropped).
        3. **ClaimNode** — persisted as a KG ``Claim`` with ``status="proposal"``,
           ``is_verified=False`` — ALWAYS, unconditionally, regardless of
           ``KG_INSIGHT_AUTONOMY``. This is the propose-only floor every other
           stage in this controller already guarantees.
        4. **Validation** — REUSES :class:`~.promotion_governance.
           PromotionGovernanceValidator` AS-IS (SHACL shapes, the recorded
           capability-ratchet verdict, the recorded regression-gate verdict,
           constitution/forbid rules, MergePolicy thresholds) — no reimplemented
           governance logic.
        5. **Action gate** — SAFETY-CRITICAL: every above-floor claim, autonomy on
           or off, is run through ``action_policy.decide()`` under the reserved
           ``kind="promote_mined_claim"`` BEFORE any promotion is even considered.
           The shipped default tier for this kind is ``approval_required`` (see
           ``deploy/action-policy.default.yml`` / ``action_policy.DEFAULT_POLICY``)
           — a mined claim is NEVER promoted without this call on the path, and the
           shipped default never allows it to happen automatically.

        X3 — opt-in autonomy tier (``KG_INSIGHT_AUTONOMY``, default OFF): only
        when explicitly enabled AND governance is valid AND the action-policy
        decision above independently allows (``allow``/``allow_notify`` — which
        requires an operator to have ALSO relaxed the shipped ``promote_mined_claim``
        tier) does this reuse the EXISTING :class:`~.auto_merge.GovernedAutoMerger`
        (which re-consults its OWN ``merge_promotion`` action-policy kind and the
        SAME governance validator — belt-and-suspenders, not a bypass) to flip the
        claim ``proposal → active`` via an injected claim-specific promoter (the
        default TeamSpec/AgentSpec promoter doesn't apply to a bare Claim). The
        cycle's own ``_finalize_metrics`` already records an ``ImprovementVelocity``
        node every cycle (:mod:`.improvement_ledger`) over whatever
        ``CapabilityRatchetResult``/``ProposalPublication`` nodes this stage (or
        any other) wrote — so the audit trail is reused, not duplicated here.

        Every sub-step is independently best-effort (mirroring the ``_mine_*``/
        belief-revision sub-step tolerance) so one bad candidate never blocks the
        rest. Best-effort + gated (``KG_LOOP_INSIGHT_VALIDATION``, default ON —
        this stage is itself propose-only regardless of the flag).
        """
        from agent_utilities.core.config import config as _cfg
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )

        from .auto_merge import GovernedAutoMerger, MergePolicy
        from .candidate_insight import candidates_from_mine_discovery
        from .claim_flywheel import ClaimFlywheel
        from .promotion_governance import PromotionGovernanceValidator

        errors: list[str] = []
        candidates = candidates_from_mine_discovery(mine_result)
        below_floor = [c for c in candidates if not c.clears_floor]
        eligible = [c for c in candidates if c.clears_floor]

        validator = PromotionGovernanceValidator(self.engine)
        action_policy = get_action_policy(self.engine)
        # X3 — the epistemic mining flywheel's lifecycle overlay (CONCEPT:AU-KG.
        # evolution.mining-flywheel). One instance per cycle so its in-process
        # cache keeps a single candidate's propose→validate→accept sequence
        # correct even against a minimal engine double; cross-cycle
        # retracted-memory (a claim never re-proposed) additionally depends on
        # the engine's own query_cypher reflecting prior writes.
        flywheel = ClaimFlywheel(self.engine)
        autonomy_on = bool(_cfg.kg_insight_autonomy)

        persisted = 0
        promoted = 0
        examples: list[dict[str, Any]] = []

        for cand in eligible:
            claim = cand.to_claim_node()
            bundle = cand.to_evidence_bundle()

            # -- X3: a retracted claim is never re-proposed, even when re-mining
            # produces the identical content-addressed finding id again. --
            if flywheel.is_retracted(claim.id):
                if len(examples) < 5:
                    examples.append(
                        {
                            "claim_id": claim.id,
                            "finding_type": cand.finding_type,
                            "confidence": round(cand.confidence, 4),
                            "skipped": "retracted",
                        }
                    )
                continue

            spec = {
                "id": claim.id,
                "name": claim.name,
                "goal": claim.claim_text,
                "description": claim.claim_text,
                "quality_score": claim.confidence,
                "type": "Claim",
            }

            # -- persist the proposal (ALWAYS; propose-only is the floor every
            # other stage in this controller already guarantees). ``type`` is
            # excluded from the dump: ClaimNode's own ``type`` field (the
            # RegistryNodeType enum value, e.g. "claim") would otherwise collide
            # with the ``"Claim"`` node-label positional arg once merged into
            # ``properties``. --
            try:
                self.engine.add_node(
                    claim.id,
                    "Claim",
                    properties={
                        **claim.model_dump(mode="json", exclude={"type"}),
                        "status": "proposal",
                        "evidence_bundle_json": bundle.model_dump_json(),
                    },
                )
                persisted += 1
            except Exception as e:  # noqa: BLE001 — persistence is best-effort
                errors.append(f"insight_validation:persist {claim.id}: {e}")
                continue

            # -- X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): the shared
            # writeback seam — see ``_register_derived_claim`` docstring. --
            self._register_derived_claim(claim, errors, "insight_validation")

            # -- X3: record the flywheel's PROPOSED event (best-effort audit
            # overlay; never gates the pipeline above). --
            try:
                flywheel.propose(claim.id, reason=f"mined {cand.finding_type} finding")
            except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                errors.append(f"insight_validation:flywheel_propose {claim.id}: {e}")

            # -- validation: REUSE promotion_governance (which itself reuses the
            # capability ratchet's recorded verdict) as-is, never reimplemented. --
            try:
                verdict = validator.validate(spec)
            except Exception as e:  # noqa: BLE001 — a validator error holds, never crashes
                errors.append(f"insight_validation:validate {claim.id}: {e}")
                continue

            try:
                flywheel.validate(
                    claim.id, verdict.valid, reason="; ".join(verdict.failures)
                )
            except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                errors.append(f"insight_validation:flywheel_validate {claim.id}: {e}")

            # -- action gate: SAFETY-CRITICAL, unconditional. A mined claim is
            # NEVER promoted without this call, autonomy on or off (see docstring). --
            try:
                decision = action_policy.decide(
                    ActionRequest(
                        kind="promote_mined_claim",
                        target=claim.id,
                        params={
                            "finding_type": cand.finding_type,
                            "confidence": cand.confidence,
                            "governance_valid": verdict.valid,
                        },
                        source="loop_engine",
                        reason=(
                            f"promote mined {cand.finding_type} finding to a "
                            "verified claim"
                        ),
                    )
                )
            except Exception as e:  # noqa: BLE001 — fail closed, never crash
                errors.append(f"insight_validation:action_policy {claim.id}: {e}")
                continue

            if decision.decision == "deny":
                try:
                    flywheel.reject(
                        claim.id,
                        reason=f"action_policy denied: {decision.reason}",
                        action_decision=decision.decision,
                    )
                except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                    errors.append(f"insight_validation:flywheel_reject {claim.id}: {e}")

            record: dict[str, Any] = {
                "claim_id": claim.id,
                "finding_type": cand.finding_type,
                "confidence": round(cand.confidence, 4),
                "governance_valid": verdict.valid,
                "action_decision": decision.decision,
                "promoted": False,
            }

            # -- X3: opt-in autonomy tier (KG_INSIGHT_AUTONOMY, default OFF). Both
            # the action-policy gate above AND governance validity must
            # independently allow before the EXISTING GovernedAutoMerger is even
            # consulted; the merger applies its OWN merge_promotion action-policy
            # check + the same governance validator on top (belt-and-suspenders). --
            if autonomy_on and verdict.valid and decision.allowed:
                # quality_threshold=0.0: this stage's own confidence floor
                # (CandidateInsight.clears_floor) already gated eligibility above;
                # the merger's quality check is redundant here — governance
                # validity + the action-policy decision already gathered are what
                # matter for this reused evaluate()/consider() call.
                merger = GovernedAutoMerger(
                    self.engine,
                    policy=MergePolicy(enabled=True, quality_threshold=0.0),
                    governance_validator=validator,
                    promoter=self._claim_promoter(claim, bundle, errors),
                )
                try:
                    evaluation = merger.consider(spec)
                    record["promoted"] = bool(evaluation.merged)
                    record["merge_reason"] = evaluation.reason
                    if evaluation.merged:
                        promoted += 1
                        self._accept_mined_claim(
                            cand, claim, decision.decision, flywheel, record, errors
                        )
                except Exception as e:  # noqa: BLE001 — never crash the cycle
                    errors.append(f"insight_validation:merge {claim.id}: {e}")

            if len(examples) < 5:
                examples.append(record)

        return {
            "candidates": len(candidates),
            "below_floor": len(below_floor),
            "eligible": len(eligible),
            "persisted_claims": persisted,
            "promoted": promoted,
            "autonomy_enabled": autonomy_on,
            "examples": examples,
            "errors": errors,
        }

    def _accept_mined_claim(
        self,
        cand: Any,
        claim: Any,
        action_decision: str,
        flywheel: Any,
        record: dict[str, Any],
        errors: list[str],
    ) -> None:
        """X3 flywheel LOOP 1 — ontology-gap (``PredictedEdge``) → accept → materialize.

        Called ONLY after the EXISTING ``GovernedAutoMerger`` already flipped
        the claim proposal → active (``evaluation.merged``); this never makes
        its own promotion decision. Records the flywheel's VALIDATED→ACCEPTED
        transition, and — for a ``PredictedEdge`` finding specifically — closes
        the ontology-gap loop: materializes the predicted relation as a REAL
        edge in the KG (the accepted claim finally lands as graph structure,
        not just a claim about one) and captures that acceptance as an
        observation fed back through the durable bandit spine
        (:meth:`~.claim_flywheel.ClaimFlywheel.record_outcome`, CONCEPT:AU-P1-3).
        Best-effort throughout — a failure here never unwinds the promotion
        that already happened.
        """
        try:
            flywheel.accept(
                claim.id,
                reason=f"promoted mined {cand.finding_type} finding",
                action_decision=action_decision,
            )
        except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
            errors.append(f"insight_validation:flywheel_accept {claim.id}: {e}")

        if cand.finding_type != "PredictedEdge":
            return

        payload = cand.payload or {}
        src = payload.get("source") or payload.get("src") or payload.get("from")
        dst = payload.get("target") or payload.get("dst") or payload.get("to")
        if src and dst:
            try:
                self.engine.add_edge(
                    str(src),
                    str(dst),
                    "PREDICTED_RELATION",
                    confidence=cand.confidence,
                    claim_id=claim.id,
                )
                record["materialized"] = True
            except Exception as e:  # noqa: BLE001 — materialization is best-effort
                errors.append(f"insight_validation:materialize {claim.id}: {e}")

        try:
            outcome = flywheel.record_outcome(
                claim.id,
                reward=cand.confidence,
                note="ontology-gap claim accepted and materialized",
            )
            record["outcome"] = outcome
        except Exception as e:  # noqa: BLE001 — outcome feedback is best-effort
            errors.append(f"insight_validation:outcome {claim.id}: {e}")

    def _claim_promoter(
        self, claim: Any, bundle: Any, errors: list[str]
    ) -> Callable[[Any], bool]:
        """Build a ``GovernedAutoMerger`` promoter that flips a Claim proposal → active.

        Injected instead of the merger's default ``persist_synthesis``-based
        promoter (built for ``TeamSpec``/``AgentSpec``/``PromptSpec`` artifacts) —
        a mined ``ClaimNode`` has its own simple proposal→active lifecycle
        (``is_verified``) rather than a synthesized team/agent/prompt.
        """

        def _promote(_spec: Any) -> bool:
            try:
                self.engine.add_node(
                    claim.id,
                    "Claim",
                    properties={
                        **claim.model_dump(mode="json", exclude={"type"}),
                        "status": "active",
                        "is_verified": True,
                        "evidence_bundle_json": bundle.model_dump_json(),
                    },
                )
                return True
            except Exception as e:  # noqa: BLE001 — promotion failure degrades, never raises
                errors.append(f"insight_validation:promote {claim.id}: {e}")
                return False

        return _promote

    # -- closed-loop agent mining (CONCEPT:AU-KG.evolution.insight-engine-closed-loop, workstream C6) -- #
    def _run_trace_mining(self) -> dict[str, Any]:
        """Mine repeated FAILURE tool-call sequences; route each through the C4 pipeline.

        Workstream C6 — closed-loop agent mining. Mines RunTrace/
        OutcomeEvaluation/ToolCall provenance (:mod:`.trace_pattern_miner`) for
        repeated FAILURE tool-call sequences and runs each mined pattern
        through the SAME CandidateInsight→Claim→Validation→Action-gate
        pipeline ``_run_insight_validation`` (workstream C4) uses — no
        reimplemented governance logic, no second confidence-floor policy:

        1. **CandidateInsight** (:func:`~.candidate_insight.
           candidates_from_sequential_patterns`) extracts each mined
           pattern's real ``support`` signal as its confidence (never
           fabricated); a below-floor pattern is counted but NEVER
           materialized as a ``ClaimNode``.
        2. **EvidenceBundle** (C1) packages the raw pattern as the claim's
           evidence trail.
        3. **ClaimNode** — persisted as a KG ``Claim`` with
           ``status="proposal"``, ``is_verified=False`` — ALWAYS,
           unconditionally (the propose-only floor every other stage in this
           controller already guarantees).
        4. **Validation** — REUSES :class:`~.promotion_governance.
           PromotionGovernanceValidator` AS-IS.
        5. **Action gate** — SAFETY-CRITICAL: every above-floor pattern is run
           through ``action_policy.decide()`` under the reserved
           ``kind="route_policy_update"`` (shipped default
           ``approval_required``) BEFORE anything else happens to it.
        6. **Governed routing/prompt/tool change + OutcomeRouter.record()** —
           ONLY when the decision above ``allowed`` (auto/auto_notify —
           requires an operator to have relaxed the shipped
           ``route_policy_update`` tier) AND governance is valid does this
           apply the change AND record the pattern's outcome via the EXISTING
           :class:`~agent_utilities.orchestration.outcome_router.
           OutcomeRouter`: a repeated failure sequence is a ``reward=0.0``
           data point for the ``(task_class, choice)`` its leading tool call
           represents, steering future ``OutcomeRouter.select()`` calls away
           from it.

        SAFETY INVARIANT — ``OutcomeRouter.record()`` MUST NEVER execute
        before the ``action_policy.decide()`` call above, on ANY path,
        for ANY candidate. This method has exactly one call to each, in that
        textual order, inside the same per-candidate loop iteration, with the
        ``record()`` call gated on the ``decision`` variable ``decide()``
        assigns — there is no branch that reaches ``router.record()`` without
        first having executed ``action_policy.decide()`` for that same
        candidate. See ``tests/unit/knowledge_graph/test_trace_pattern_miner.py::
        test_gate_runs_before_any_outcome_record`` for the enforced ordering
        (a mock ``action_policy``/``router`` pair that fails the test if
        ``record()`` is ever observed before ``decide()``) and
        ``test_route_policy_update_default_never_auto`` for the shipped
        policy's fail-closed default.

        Every sub-step is independently best-effort (mirroring the
        ``_run_insight_validation``/``_mine_*`` sub-step tolerance) so one bad
        candidate never blocks the rest. Best-effort + gated
        (``KG_LOOP_TRACE_MINING``, default ON — this stage is itself
        propose-only regardless of the flag).
        """
        from agent_utilities.observability.trace_ontology import (
            load_trace_cursor,
            save_trace_cursor,
        )
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )
        from agent_utilities.orchestration.outcome_router import (
            OutcomeRouter,
            outcome_reward,
        )

        from .candidate_insight import candidates_from_sequential_patterns
        from .claim_flywheel import ClaimFlywheel
        from .promotion_governance import PromotionGovernanceValidator
        from .trace_pattern_miner import mine_trace_patterns

        errors: list[str] = []
        cursor_consumer = "trace-pattern-miner"
        prior_cursor = load_trace_cursor(self.engine, cursor_consumer)
        mine_result = mine_trace_patterns(
            self.engine, after_sequence=prior_cursor.event_sequence
        )
        errors.extend(mine_result.get("errors") or [])
        candidates = candidates_from_sequential_patterns(mine_result.get("patterns"))
        below_floor = [c for c in candidates if not c.clears_floor]
        eligible = [c for c in candidates if c.clears_floor]

        validator = PromotionGovernanceValidator(self.engine)
        action_policy = get_action_policy(self.engine)
        router = OutcomeRouter(namespace="trace_pattern_miner")
        # X3 — the epistemic mining flywheel's lifecycle overlay (CONCEPT:AU-KG.
        # evolution.mining-flywheel); see ``_run_insight_validation`` for why one
        # instance per cycle.
        flywheel = ClaimFlywheel(self.engine)

        persisted = 0
        routed = 0
        examples: list[dict[str, Any]] = []

        for cand in eligible:
            claim = cand.to_claim_node()
            bundle = cand.to_evidence_bundle()

            # -- X3: a retracted claim is never re-proposed. --
            if flywheel.is_retracted(claim.id):
                if len(examples) < 5:
                    examples.append(
                        {
                            "claim_id": claim.id,
                            "finding_type": cand.finding_type,
                            "confidence": round(cand.confidence, 4),
                            "skipped": "retracted",
                        }
                    )
                continue

            spec = {
                "id": claim.id,
                "name": claim.name,
                "goal": claim.claim_text,
                "description": claim.claim_text,
                "quality_score": claim.confidence,
                "type": "Claim",
            }

            # -- persist the proposal (ALWAYS; propose-only is the floor every
            # other stage in this controller already guarantees). --
            try:
                self.engine.add_node(
                    claim.id,
                    "Claim",
                    properties={
                        **claim.model_dump(mode="json", exclude={"type"}),
                        "status": "proposal",
                        "evidence_bundle_json": bundle.model_dump_json(),
                    },
                )
                persisted += 1
            except Exception as e:  # noqa: BLE001 — persistence is best-effort
                errors.append(f"trace_mining:persist {claim.id}: {e}")
                continue

            # -- X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): the SAME
            # shared writeback seam ``_run_insight_validation`` uses — see
            # ``_register_derived_claim`` docstring. --
            self._register_derived_claim(claim, errors, "trace_mining")

            try:
                flywheel.propose(claim.id, reason=f"mined {cand.finding_type} finding")
            except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                errors.append(f"trace_mining:flywheel_propose {claim.id}: {e}")

            # -- validation: REUSE promotion_governance as-is, never reimplemented. --
            try:
                verdict = validator.validate(spec)
            except Exception as e:  # noqa: BLE001 — a validator error holds, never crashes
                errors.append(f"trace_mining:validate {claim.id}: {e}")
                continue

            try:
                flywheel.validate(
                    claim.id, verdict.valid, reason="; ".join(verdict.failures)
                )
            except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                errors.append(f"trace_mining:flywheel_validate {claim.id}: {e}")

            task_class, choice = self._trace_pattern_route(cand)

            # ── SAFETY: action_policy.decide() MUST run — and complete —
            # before ANY OutcomeRouter.record() call for this candidate. See
            # the docstring above; do not reorder. ──
            try:
                decision = action_policy.decide(
                    ActionRequest(
                        kind="route_policy_update",
                        target=claim.id,
                        params={
                            "finding_type": cand.finding_type,
                            "confidence": cand.confidence,
                            "governance_valid": verdict.valid,
                            "task_class": task_class,
                            "choice": choice,
                        },
                        source="loop_engine",
                        reason=(
                            "apply a routing/prompt/tool change from a mined "
                            "repeated-failure tool-call pattern"
                        ),
                    )
                )
            except Exception as e:  # noqa: BLE001 — fail closed, never crash
                errors.append(f"trace_mining:action_policy {claim.id}: {e}")
                continue

            if decision.decision == "deny":
                try:
                    flywheel.reject(
                        claim.id,
                        reason=f"action_policy denied: {decision.reason}",
                        action_decision=decision.decision,
                    )
                except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
                    errors.append(f"trace_mining:flywheel_reject {claim.id}: {e}")

            record: dict[str, Any] = {
                "claim_id": claim.id,
                "finding_type": cand.finding_type,
                "confidence": round(cand.confidence, 4),
                "governance_valid": verdict.valid,
                "action_decision": decision.decision,
                "routed": False,
            }

            # -- ONLY reachable after action_policy.decide() (above) returned.
            # Gated on verdict.valid + decision.allowed (auto/auto_notify),
            # mirroring the X3 autonomy-tier gate ``_run_insight_validation``
            # uses. The shipped route_policy_update tier is
            # approval_required, so this branch never fires out of the box. --
            if verdict.valid and decision.allowed and task_class and choice:
                try:
                    router.record(
                        task_class,
                        choice,
                        outcome_reward(success=False, latency_s=0.0),
                    )
                    record["routed"] = True
                    routed += 1
                except Exception as e:  # noqa: BLE001 — learning must never break the cycle
                    errors.append(f"trace_mining:route {claim.id}: {e}")

                if record["routed"]:
                    self._accept_routed_claim(
                        cand,
                        claim,
                        router,
                        task_class,
                        choice,
                        decision.decision,
                        flywheel,
                        errors,
                    )

            if len(examples) < 5:
                examples.append(record)

        completed_cursor = prior_cursor
        if not errors:
            completed_cursor = save_trace_cursor(
                self.engine,
                cursor_consumer,
                int(
                    mine_result.get("next_event_sequence")
                    or prior_cursor.event_sequence
                ),
            )

        return {
            "candidates": len(candidates),
            "below_floor": len(below_floor),
            "eligible": len(eligible),
            "persisted_claims": persisted,
            "routed": routed,
            "failure_traces": mine_result.get("failure_traces", 0),
            "sequences_mined": mine_result.get("sequences_mined", 0),
            "after_event_sequence": prior_cursor.event_sequence,
            "next_event_sequence": completed_cursor.event_sequence,
            "cursor_advanced": completed_cursor > prior_cursor,
            "examples": examples,
            "errors": errors,
        }

    @staticmethod
    def _trace_pattern_route(cand: Any) -> tuple[str, str]:
        """Derive an ``OutcomeRouter`` ``(task_class, choice)`` pair from a mined pattern.

        ``task_class`` is a fixed namespace for all trace-mined patterns (the
        router's own ``namespace`` already scopes these apart from every other
        ``OutcomeRouter`` consumer); ``choice`` is the pattern's LEADING tool
        call — the first decision point the failure sequence hinges on. Empty
        (never guessed) when the pattern carries no source ids.
        """
        if not cand.source_ids:
            return "", ""
        return "failure_tool_sequence", str(cand.source_ids[0])

    def _accept_routed_claim(
        self,
        cand: Any,
        claim: Any,
        router: Any,
        task_class: str,
        choice: str,
        action_decision: str,
        flywheel: Any,
        errors: list[str],
    ) -> None:
        """X3 flywheel LOOP 2 — process/routing quality (``SequentialPattern``) →
        accept → outcome → durable bandit feedback.

        Called ONLY after ``router.record()`` already applied the routing
        change (this never makes its own routing decision). Records the
        flywheel's VALIDATED→ACCEPTED transition, then closes the loop with
        TWO deliberately distinct rewards
        (:meth:`~.claim_flywheel.ClaimFlywheel.record_outcome`): the claim's
        OWN outcome (``reward=cand.confidence`` — the pattern's mined
        ``support``, i.e. how well-evidenced accepting this claim was; a
        confidently-mined claim is not itself "bad" just because what it
        teaches the router is negative, so this does NOT auto-deprecate a
        well-supported claim) versus the bandit's ``durable_reward=0.0`` — the
        SAME negative observation ``router.record()`` already fed the
        in-process ``OutcomeRouter``, ALSO persisted DURABLY onto that
        router's bandit key (CONCEPT:AU-P1-3) so the learned routing
        preference survives a process restart instead of resetting to the
        neutral 0.5 prior on the next cycle. Best-effort throughout — never
        unwinds the routing change that already happened.
        """
        try:
            flywheel.accept(
                claim.id,
                reason="routing change applied",
                action_decision=action_decision,
            )
        except Exception as e:  # noqa: BLE001 — the audit overlay is best-effort
            errors.append(f"trace_mining:flywheel_accept {claim.id}: {e}")

        try:
            flywheel.record_outcome(
                claim.id,
                reward=cand.confidence,
                durable_reward=0.0,
                note="repeated failure pattern routed away from",
                durable_key=router.key(task_class, choice),
            )
        except Exception as e:  # noqa: BLE001 — outcome feedback is best-effort
            errors.append(f"trace_mining:outcome {claim.id}: {e}")

    # -- belief revision / confidence propagation (CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision) -- #
    def _run_belief_revision(self) -> dict[str, Any]:
        """Confidence-propagation + light-TMS pass over the KG's ``Belief`` nodes.

        Workstream C2: loads every ``:Belief`` node, runs
        :class:`~..adaptation.belief_revision.BeliefRevisionPass` (which
        internally re-runs :class:`~..adaptation.contradiction_detector.
        ContradictionDetector` over the belief statements to discover fresh
        friction, unions it with each belief's already-recorded
        ``contradicted_by_node_ids``/``supported_by_node_ids``, and recomputes
        an explainable confidence for every belief), then persists each
        recomputed outcome as a ``:BeliefRevisionProposal`` node — a NEW
        advisory node, never a mutation of the live ``Belief`` node's
        ``confidence``/``contradicted_by_node_ids`` fields. This mirrors the
        SAME propose-only doctrine the ``TeamSpec``/``SearchTask`` stages
        already use in this controller (persist a ``status: "proposal"``
        node; never touch the canonical record) — the Critic flags, it does
        not arbitrate. A human/agent reviewer (or a future promotion path)
        decides whether/how to fold a proposal back into the live belief.

        Each of the three sub-steps below (query, recompute, persist) is
        independently best-effort — mirroring the ``_mine_*`` sub-step
        tolerance pattern — so a failure in one never blocks the others or
        raises out of the stage (the outer ``_stage`` wrapper in
        ``run_one_cycle`` would catch it regardless, but the finer-grained
        tolerance here keeps partial results instead of an all-or-nothing
        failure). Best-effort + gated (``KG_LOOP_BELIEF_REVISION``, default
        ON — degrades to a ``skipped`` result with no beliefs to revise, so
        it's safe to leave on everywhere).
        """
        from ..adaptation.belief_revision import BeliefRevisionPass
        from ..adaptation.contradiction_detector import ContradictionDetector

        errors: list[str] = []

        try:
            rows = (
                self.engine.query_cypher(
                    "MATCH (b:Belief) RETURN b.id AS id, b.statement AS statement, "
                    "b.confidence AS confidence, "
                    "b.evidence_node_ids AS evidence_node_ids, "
                    "b.supported_by_node_ids AS supported_by_node_ids, "
                    "b.contradicted_by_node_ids AS contradicted_by_node_ids, "
                    "b.last_reviewed AS last_reviewed LIMIT $limit",
                    {"limit": 200},
                )
                or []
            )
        except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
            errors.append(f"belief_revision:query: {e}")
            return {"skipped": True, "reason": "query failed", "errors": errors}

        beliefs = self._parse_belief_rows(rows, errors)
        if len(beliefs) < 2:
            # Nothing to compare against — not an error, just nothing to do yet.
            return {
                "skipped": True,
                "reason": "fewer than 2 Belief nodes",
                "beliefs_scanned": len(beliefs),
                "errors": errors,
            }

        try:
            revisions = BeliefRevisionPass(
                contradiction_detector=ContradictionDetector(), use_engine=True
            ).scan(beliefs)
        except Exception as e:  # noqa: BLE001 — recompute failure degrades, never raises
            errors.append(f"belief_revision:scan: {e}")
            return {
                "skipped": True,
                "reason": "revision scan failed",
                "beliefs_scanned": len(beliefs),
                "errors": errors,
            }

        persisted = 0
        examples: list[dict[str, Any]] = []
        for revision in revisions:
            payload = revision.to_dict()
            if len(examples) < 5:
                examples.append(payload)
            if not self.propose_only:
                continue
            try:
                self.engine.add_node(
                    f"BeliefRevisionProposal:{revision.belief_id}:{revision.last_reviewed}",
                    {
                        "type": "BeliefRevisionProposal",
                        "status": "proposal",
                        **payload,
                    },
                )
                persisted += 1
            except Exception as e:  # noqa: BLE001 — persistence is best-effort
                errors.append(f"belief_revision:persist {revision.belief_id}: {e}")

        return {
            "skipped": False,
            "beliefs_scanned": len(beliefs),
            "revisions": len(revisions),
            "persisted_nodes": persisted,
            "examples": examples,
            "errors": errors,
        }

    @staticmethod
    def _parse_belief_rows(rows: list[Any], errors: list[str]) -> list[Any]:
        """Best-effort ``Belief`` row → ``BeliefNode`` parsing, one row at a time.

        A single malformed row (bad confidence, missing id) is recorded and
        skipped rather than aborting the whole pass.
        """
        from agent_utilities.models.knowledge_graph import (
            BeliefNode,
            RegistryNodeType,
        )

        beliefs = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            try:
                raw_confidence = row.get("confidence")
                confidence = 0.5 if raw_confidence is None else float(raw_confidence)
                confidence = max(0.0, min(1.0, confidence))
                beliefs.append(
                    BeliefNode(
                        id=row["id"],
                        type=RegistryNodeType.BELIEF,
                        name=str(row["id"]),
                        statement=row.get("statement") or "",
                        confidence=confidence,
                        evidence_node_ids=list(row.get("evidence_node_ids") or []),
                        supported_by_node_ids=list(
                            row.get("supported_by_node_ids") or []
                        ),
                        contradicted_by_node_ids=list(
                            row.get("contradicted_by_node_ids") or []
                        ),
                        last_reviewed=row.get("last_reviewed") or "",
                    )
                )
            except Exception as e:  # noqa: BLE001 — one bad row never blocks the rest
                errors.append(f"belief_revision:parse {row.get('id')}: {e}")
        return beliefs

    def _distill_skills(self) -> dict[str, Any]:
        """Distil connector processes into propose-only skill candidates.

        The connector→skill synthesis stage (CONCEPT:AU-KG.ontology.connector-agnostic-proposal/2.83): the
        :class:`ConnectorSkillDistiller` queries the KG over the ontology classes
        (BusinessProcess flowsTo-chains, BusinessTask, Capability) of EVERY
        connected system, classifies atomic-skill vs skill-workflow candidates,
        dedups against the existing skill registry, and writes SkillProposal /
        SkillWorkflowProposal nodes (with AUTOMATES + DERIVED_FROM provenance
        edges) — propose-only. Drafting SKILL.md artifacts is deferred to
        review/approval to keep the cycle cheap.

        Reuses the per-cycle embedder (built once for the acquire_resolve stage)
        for semantic dedup rather than constructing a fresh one — and only when
        embeddings are actually reachable, so an embedding outage degrades the
        dedup to the deterministic name pass instead of stalling the cycle.
        """
        from ..distillation.skill_synthesizer import ConnectorSkillDistiller

        embed_fn = None
        try:
            from ..enrichment.semantic import make_embed_fn
            from .search import _ACQUIRE_TIMEOUT_S, bounded_embed

            probe = make_embed_fn()
            if bounded_embed(probe, "ping", _ACQUIRE_TIMEOUT_S) is not None:
                embed_fn = probe
        except Exception as e:  # noqa: BLE001 — embedder optional, name-pass still runs
            logger.debug("[KG-2.90] embedder probe failed: %s", e)

        distiller = ConnectorSkillDistiller(self.engine, embed_fn=embed_fn)
        return distiller.run().to_dict()

    # -- SkillOpt-native ReflACT skill evolution (CONCEPT:AU-AHE.optimization.skillopt-native-reflact) ---- #
    def _discover_skill_evolution_targets(self) -> list[dict[str, Any]]:
        """Default discovery hook — degrades to ``[]`` with nothing registered.

        The foundation ships with NO auto-discovery query: no ``:SkillEvalSuite`` KG
        type exists yet to enumerate (a real one would query the KG for skills
        carrying a registered held-out eval suite, mirroring how
        ``_run_insight_validation`` is driven by ``mine_discovery``'s output rather
        than a fresh KG scan of its own). Constructor-injectable via
        ``skill_eval_targets_provider`` (mirrors ``develop_runner``/``skill_runner``)
        so real callers and tests can supply concrete targets without that KG type
        existing — each target is a dict with keys ``skill_id``, ``content``,
        ``signal`` (a :class:`~.skill_evolution.SkillSignalProvider`), and optional
        ``executor``/``edit_fn`` overrides.
        """
        return []

    def _run_skill_evolution(self) -> dict[str, Any]:
        """Run one ReflACT cycle per registered skill-evolution target (CONCEPT:AU-AHE.optimization.skillopt-native-reflact).

        Sibling to ``_distill_skills`` (which proposes brand-new skills from
        connector-mapped processes): this stage evolves the markdown of EXISTING
        skills. Delegates the actual Rollout->Reflect->Aggregate/Select/Update->
        Evaluate->gate pipeline to :func:`~.skill_evolution.run_reflact_cycle` for
        each discovered target; a single target's failure never blocks the others
        (mirrors the ``_mine_*`` sub-step tolerance).
        """
        provider = self._skill_eval_targets_provider or self._discover_skill_evolution_targets
        try:
            targets = provider() or []
        except Exception as e:  # noqa: BLE001
            return {"targets": 0, "results": [], "promoted": 0, "errors": [f"discover: {e}"]}

        if not targets:
            return {
                "skipped": True,
                "reason": "no_registered_skill_eval_targets",
                "targets": 0,
            }

        from .skill_evolution import run_reflact_cycle

        results: list[dict[str, Any]] = []
        errors: list[str] = []
        promoted = 0
        for target in targets:
            skill_id = target.get("skill_id", "?")
            try:
                rep = run_reflact_cycle(
                    self.engine,
                    skill_id,
                    target["content"],
                    signal=target["signal"],
                    executor=target.get("executor"),
                    edit_fn=target.get("edit_fn"),
                )
                results.append(rep)
                if rep.get("promoted"):
                    promoted += 1
                errors.extend(rep.get("errors", []))
            except Exception as e:  # noqa: BLE001 — one target never blocks the others
                errors.append(f"{skill_id}: {e}")

        return {
            "targets": len(targets),
            "results": results,
            "promoted": promoted,
            "errors": errors,
        }

    # -- develop / skill Loop execution (CONCEPT:AU-KG.research.these-properties-carry L3) ---------------- #
    def _run_execute_loops(self, loops: list[dict[str, Any]]) -> dict[str, Any]:
        """Advance every non-research Loop one step through the same hot path.

        A ``develop`` Loop runs its ``validation_cmd`` once (act→validate); a
        ``skill`` Loop runs its ``skill_ref`` skill/skill-workflow. Each transitions
        the Loop's lifecycle (``completed`` on success / terminal, else it stays
        active for the next cycle). Best-effort: a failing Loop is recorded, never
        aborts the cycle. This is what makes goals + skill runs first-class Loops
        advanced by the one controller, not separate engines.
        """
        from .loops import claim_loop, mark_loop_status

        out: dict[str, Any] = {
            "develop": 0,
            "skill": 0,
            "completed": 0,
            "skipped": 0,
            "results": [],
        }
        for loop in loops:
            kind = loop.get("kind", "research")
            if kind not in ("develop", "skill"):
                continue
            # Atomically claim the Loop (status → running via the engine CAS)
            # before advancing it. A lost race means a concurrent cycle / peer
            # host / graph_loops run already owns it — skip rather than
            # double-drive. (CONCEPT:AU-KG.compute.user-override-prompt-library)
            if not claim_loop(self.engine, loop["id"]):
                out["skipped"] += 1
                out["results"].append(
                    {"id": loop["id"], "kind": kind, "status": "skipped(claimed)"}
                )
                continue
            res = self._iterate(loop)
            out[kind] += 1
            status = res.get("status", "pending")
            mark_loop_status(
                self.engine,
                loop["id"],
                status,
                output=str(res.get("output", ""))[:2000],
            )
            if status == "completed":
                out["completed"] += 1
            out["results"].append({"id": loop["id"], "kind": kind, "status": status})
        return out

    def _iterate(self, loop: dict[str, Any]) -> dict[str, Any]:
        """Advance ANY Loop one step, dispatched by kind (CONCEPT:AU-KG.research.these-properties-carry).

        The single kind-agnostic step the controller runs everywhere — the
        per-cycle execute stage, the durable :meth:`run_loop`, and the
        goal adapter all funnel through here, so research/develop/skill share one
        execution path. Returns ``{"status", "output", "done"?}``.
        """
        kind = loop.get("kind", "research")
        if kind == "develop":
            return self._advance_develop(loop)
        if kind == "skill":
            return self._advance_skill(loop)
        if kind == "external_event":
            return self._advance_external_event(loop)
        return self._advance_research(loop)

    def _advance_research(self, loop: dict[str, Any]) -> dict[str, Any]:
        """Run one research iteration: acquire related sources and ADDRESS the topic.

        The same acquire→resolve the cycle's research stage does, exposed as a
        single durable-able step so a research Loop can be driven to completion by
        :meth:`run_loop` exactly like develop/skill — durability is cross-cutting.
        """
        from ..adaptation.topic_resolver import mark_addressed
        from ..enrichment.semantic import make_embed_fn
        from .search import acquire_for_topic_perspectival

        try:
            embed_fn = make_embed_fn()
            srcs = acquire_for_topic_perspectival(self.engine, loop, embed_fn=embed_fn)
        except Exception as e:  # noqa: BLE001 — best-effort
            return {"status": "pending", "output": f"acquire failed: {e}"}
        if srcs:
            mark_addressed(self.engine, loop["id"], srcs, source="loop_engine")
            return {
                "status": "completed",
                "output": f"addressed by {len(srcs)} sources",
                "done": True,
            }
        return {"status": "pending", "output": "no sources found"}

    def _advance_develop(self, loop: dict[str, Any]) -> dict[str, Any]:
        """Run one develop iteration.

        A spec-bound develop Loop (carrying ``spec_id``, created by the OS-5.73
        spec-review approval) feeds the approved spec into the EXISTING promotion
        pipeline via ``develop_spec`` → ``governed_publish`` (CONCEPT:AU-KG.research.close-distill-develop-seam) — the
        ``merge_promotion`` human gate + capability ratchet stay on that path. A plain
        develop Loop runs its ``validation_cmd`` and completes on exit 0 (unchanged).
        """
        spec_id = (loop.get("spec_id") or "").strip()
        if spec_id:
            from .spec_proposals import develop_spec

            res = develop_spec(self.engine, spec_id)
            status = str(res.get("status", ""))
            # 'published'/'approval_queued' = the governed pipeline ran + queued a
            # reviewable branch → the develop step did its job (complete). Hard
            # failures stop the loop rather than retrying a broken synthesis forever.
            done = status in ("published", "approval_queued", "approved")
            import json as _json

            return {
                "status": "completed" if done else "failed",
                "output": _json.dumps(res, default=str)[:2000],
                "done": done,
            }
        cmd = (loop.get("validation_cmd") or "").strip()
        if not cmd:
            # no command to validate → nothing to advance; leave it active
            return {"status": loop.get("status", "pending"), "output": ""}
        if self._develop_runner is None:
            from agent_utilities.core.config import config

            if not config.kg_loop_allow_host_validation:
                return {
                    "status": "pending",
                    "output": (
                        "host validation is disabled; configure a governed "
                        "develop runner or explicitly enable the dangerous host runner"
                    ),
                }
        runner = self._develop_runner or _default_develop_runner
        ok, output = runner(cmd, self.codebase_root)
        from agent_utilities.http.redaction import redact_text
        from agent_utilities.security.persistence_privacy import (
            PersistencePrivacyGuard,
        )

        safe_output, _ = PersistencePrivacyGuard().sanitize_text(
            redact_text(str(output)[:4096])
        )
        return {
            "status": "completed" if ok else "pending",
            "output": safe_output,
        }

    def _advance_skill(self, loop: dict[str, Any]) -> dict[str, Any]:
        """Run a skill / skill-workflow Loop to its completion state."""
        ref = (loop.get("skill_ref") or "").strip()
        if not ref:
            return {"status": "failed", "output": "skill Loop has no skill_ref"}
        if self._skill_runner is not None:
            ok, output = self._skill_runner(ref, loop.get("objective", ""))
        else:
            ok, output = _default_skill_runner(
                ref, loop.get("objective", ""), self.engine
            )
        return {"status": "completed" if ok else "failed", "output": output}

    def _advance_external_event(self, loop: dict[str, Any]) -> dict[str, Any]:
        """Advance an ``external_event`` Loop: poll its real-world signal once.

        Generalizes ``deploy_watch``'s poll-until-window-or-signal into an
        objective-agnostic exit: the loop *completes* when a registered signal
        probe fires (PR merged, ticket closed, deploy healthy). ``run_loop``'s
        own ``event_probe`` guard is the enforced surface — this per-step advance
        makes the same kind drivable through the shared execute path. Non-firing
        is a benign ``pending`` (keep polling); a missing probe is a hard
        ``failed`` (nothing to wait on). (CONCEPT:AU-AHE.harness.loop-exit-conditions)
        """
        probe = self._external_event_probe(loop)
        if probe is None:
            return {
                "status": "failed",
                "output": f"external_event Loop {loop.get('id')!r} has no resolvable event probe",
            }
        if self._event_fired(probe):
            return {
                "status": "external_event_satisfied",
                "output": "external event signal fired",
                "done": True,
            }
        return {"status": "pending", "output": "awaiting external event"}

    # -- harness-enforced loop-exit helpers (CONCEPT:AU-AHE.harness.loop-exit-conditions) --- #
    def _external_event_probe(
        self, loop: dict[str, Any]
    ) -> Callable[[], bool] | None:
        """Resolve an ``external_event`` Loop's signal probe (exit 8).

        Looks up the loop's ``event_ref`` (falling back to its id) in the
        injected probe registry. Returns ``None`` when no probe is registered —
        the loop then cannot complete on a signal (surfaced as a failed advance),
        never silently waits forever.
        """
        ref = str(loop.get("event_ref") or loop.get("id") or "").strip()
        probe = self._event_probes.get(ref)
        if probe is None and ref:
            # Also accept a probe keyed by the bare objective/name for convenience.
            probe = self._event_probes.get(str(loop.get("name") or "").strip())
        return probe

    @staticmethod
    def _event_fired(event_probe: Callable[[], bool]) -> bool:
        """Poll an external-event probe once, best-effort (never raises)."""
        try:
            return bool(event_probe())
        except Exception as e:  # noqa: BLE001 — a flaky probe is 'not fired', not fatal
            logger.debug("run_loop event_probe failed: %s", e)
            return False

    @staticmethod
    def _budget_exceeded(resource_optimizer: Any) -> bool:
        """True when the injected budget authority reports exhaustion (exit 3).

        Duck-typed over :class:`~agent_utilities.core.resource_optimizer.
        ResourceOptimizer` (``is_budget_exceeded()``); best-effort so a probe
        error never crashes the loop (treated as 'not exceeded').
        """
        checker = getattr(resource_optimizer, "is_budget_exceeded", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception as e:  # noqa: BLE001
            logger.debug("run_loop budget check failed: %s", e)
            return False

    @staticmethod
    def _budget_detail(resource_optimizer: Any) -> str:
        """A compact budget summary for the ``budget_exceeded`` exit_reason."""
        summary = getattr(resource_optimizer, "summary", None)
        if callable(summary):
            try:
                return str(summary())[:500]
            except Exception:  # noqa: BLE001
                return "budget exhausted"
        return "budget exhausted"

    def _progress_window_node_id(self, loop_id: str) -> str:
        return f"{loop_id}:progress_window"

    def _load_progress_window(self, loop_id: str) -> list[str]:
        """Read the persisted rolling progress-signature window (exit 5).

        Best-effort: a fresh loop (or an engine without Cypher) starts empty.
        """
        q = getattr(self.engine, "query_cypher", None)
        if not callable(q):
            return []
        try:
            rows = q(
                "MATCH (n) WHERE n.id = $id RETURN n.hashes AS hashes LIMIT 1",
                {"id": self._progress_window_node_id(loop_id)},
            )
        except Exception:  # noqa: BLE001
            return []
        if not rows:
            return []
        row = rows[0]
        raw = row.get("hashes") if isinstance(row, dict) else None
        if isinstance(raw, str):
            return [h for h in raw.split(",") if h]
        if isinstance(raw, list):
            return [str(h) for h in raw]
        return []

    def _persist_progress_window(
        self, loop_id: str, hashes: list[str], window: int
    ) -> None:
        """Persist the last-N progress signatures on the Loop's window node (exit 5).

        Mirrors the fan-out ``_STALL_THRESHOLD`` pattern's durable counter: only
        the trailing ``window`` (default 3) signatures are kept, so a stalled loop
        is detectable across a resume, not just within one process. Best-effort.
        """
        keep = max(1, window)
        tail = hashes[-keep:]
        add_node = getattr(self.engine, "add_node", None)
        if not callable(add_node):
            return
        try:
            add_node(
                self._progress_window_node_id(loop_id),
                "LoopProgressWindow",
                properties={
                    "loop_id": loop_id,
                    "hashes": ",".join(tail),
                    "window": keep,
                },
            )
        except Exception as e:  # noqa: BLE001 — progress persistence is best-effort
            logger.debug("run_loop progress-window persist failed: %s", e)

    # -- engine-native resumable run-to-completion (CONCEPT:AU-KG.research.these-properties-carry) --- #
    async def run_loop(
        self,
        loop: dict[str, Any],
        *,
        max_iterations: int | None = None,
        on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
        desired_state: Callable[[], str | None] | None = None,
        sleep_s: float = 0.0,
        goal_evaluator: Any = None,
        resource_optimizer: Any = None,
        deadline: float | None = None,
        max_duration_s: float | None = None,
        no_progress_window: int | None = None,
        max_consecutive_failures: int | None = None,
        event_probe: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Drive one Loop to completion under its native WorkItem lease.

        The generalized fold of the goal runner: resumable, corrigible iteration
        once for research/develop/skill alike, with all EIGHT agent-loop exit
        conditions enforced by the harness (not merely requested in a prompt).
        Each exit is a guarded transition to a DISTINCT terminal
        :class:`~..research.loops.LoopStatus` so the reason a loop stopped is
        diagnosable (CONCEPT:AU-AHE.harness.loop-exit-conditions):

        1. **GOAL MET** -> ``completed`` — a *measured* pass from ``goal_evaluator``
           (deterministic validation for develop loops; a rubric/LLM judge for
           research/skill loops), never the callee merely self-declaring done.
        2. **TURN CAP** -> ``max_iterations_exceeded`` — ``it >= max_iterations``.
        3. **BUDGET CAP** -> ``budget_exceeded`` — ``resource_optimizer.
           is_budget_exceeded()`` (a HARD stop, not just a tier downgrade).
        4. **WALL CLOCK** -> ``wall_clock_exceeded`` — an overall ``deadline`` /
           ``max_duration_s`` checked every iteration in the while-condition,
           independent of the per-substep timeouts.
        5. **NO PROGRESS** -> ``stalled`` — the last N ``(status, output,
           checkpoint)`` signatures identical.
        6. **HUMAN INTERRUPT** -> ``cancelled``/``paused`` — ``desired_state`` (the
           fleet kill switch, evaluated OUTSIDE the step, before any risky work).
        7. **ERROR THRESHOLD** -> ``error_threshold_exceeded`` — N consecutive
           non-terminal failures (a :class:`ConsecutiveFailureGuard` lifting the
           engine breaker's threshold+reset semantics), reset on any progress.
        8. **EXTERNAL EVENT** -> ``external_event_satisfied`` — an ``event_probe``
           (or an ``external_event`` Loop kind) firing on a real-world signal.

        Durability is unchanged: **resume** from the WorkItem's fenced
        ``checkpoint_id``; **one authority** (lease, checkpoint, terminal outcome
        all on the engine-native WorkItem, no sidecar). Every new knob is
        config-overridable with a safe default.

        Returns ``{"id", "status", "iterations", "exit_reason"?, "interrupted"?}``.
        """
        import asyncio
        import time as _time

        from agent_utilities.core.config import config as _cfg
        from agent_utilities.orchestration import work_item as _wi
        from agent_utilities.orchestration.loop_guards import (
            ConsecutiveFailureGuard,
            GoalEvaluation,
            build_goal_evaluator,
            deadline_passed,
            progress_signature,
            resolve_deadline,
            window_is_stalled,
        )

        from .loops import (
            LoopStatus,
            claim_loop,
            is_terminal,
            mark_loop_status,
            send_loop_statechart_event,
            statechart_active_state,
            to_status,
        )

        loop_id = loop["id"]
        kind = str(loop.get("kind") or "research").strip().lower()
        max_it = int(max_iterations or loop.get("max_iterations") or 20)
        status = to_status(loop.get("status") or "running", default=LoopStatus.RUNNING)

        # -- exit 4 WALL CLOCK: resolve one overall monotonic deadline up front. --
        start_monotonic = _time.monotonic()
        if max_duration_s is None and deadline is None and _cfg.kg_loop_max_duration_s:
            max_duration_s = float(_cfg.kg_loop_max_duration_s)
        deadline = resolve_deadline(deadline, max_duration_s, start_monotonic)

        # -- exit 5 NO PROGRESS: config-defaulted window + rolling signatures. --
        stall_window = int(
            no_progress_window
            if no_progress_window is not None
            else _cfg.kg_loop_no_progress_window
        )
        progress_hashes: list[str] = list(self._load_progress_window(loop_id))

        # -- exit 7 ERROR THRESHOLD: per-loop guard (engine-breaker semantics). --
        fail_guard = ConsecutiveFailureGuard(
            threshold=int(
                max_consecutive_failures
                if max_consecutive_failures is not None
                else _cfg.kg_loop_max_consecutive_failures
            )
        )

        # -- exit 1 GOAL MET: injected evaluator, else the default per-kind one
        # (deterministic for develop; rubric/LLM judge for research/skill,
        # degrading to callee-trust offline). --
        evaluator = goal_evaluator
        if evaluator is None:
            evaluator = build_goal_evaluator(
                loop,
                threshold=float(_cfg.kg_loop_goal_eval_threshold),
                enable_llm_judge=bool(_cfg.kg_loop_goal_eval_enabled),
            )

        # -- exit 8 EXTERNAL EVENT: an ``external_event`` loop resolves its probe
        # from the registered probes when the caller didn't pass one directly. --
        if event_probe is None and kind == "external_event":
            event_probe = self._external_event_probe(loop)

        # The native claim transaction owns expired-lease recovery. A negative
        # result is authoritative; no sidecar checkpoint can grant re-entry.
        won = claim_loop(self.engine, loop_id)
        if not won:
            it = self._resume_iteration(self.engine, loop_id)
            logger.info(
                "run_loop: Loop %s already claimed by another driver — yielding.",
                loop_id,
            )
            return {
                "id": loop_id,
                "status": str(loop.get("status") or "running"),
                "iterations": it,
                "skipped": True,
            }
        it = self._resume_iteration(self.engine, loop_id)
        item_id = _wi.loop_work_item_id(loop_id)
        claim = _wi.current_work_item_claim(self.engine, item_id)
        if claim is None:
            raise _wi.WorkItemBackendUnavailable(
                f"Loop {loop_id!r} lost its native claim before execution"
            )

        def _finish(
            final: LoopStatus,
            *,
            reason: str = "",
            interrupted: bool = False,
        ) -> dict[str, Any]:
            """Guarded transition to a terminal state: commit + build the result.

            The ``completed`` happy path stays the bare ``{"id", "status",
            "iterations"}`` shape internal callers assert on; every other exit
            additionally carries an ``exit_reason`` so the abnormal/exhaustion
            terminals are diagnosable (never a generic 'failed').
            """
            mark_loop_status(
                self.engine, loop_id, final.value, iteration=it, output=reason[:2000]
            )
            out: dict[str, Any] = {
                "id": loop_id,
                "status": final.value,
                "iterations": it,
            }
            if final is not LoopStatus.COMPLETED and reason:
                out["exit_reason"] = reason
            if interrupted:
                out["interrupted"] = True
            return out

        # exit 4 is enforced in the while-condition (alongside the turn cap);
        # the precise terminal status is decided right after the loop exits.
        while (
            it < max_it
            and not deadline_passed(deadline)
            and not is_terminal(status)
        ):
            # -- exits 3/6/8 (BUDGET CAP / HUMAN INTERRUPT / EXTERNAL EVENT):
            # compute each signal exactly as before (corrigible kill switch
            # evaluated OUTSIDE/BEFORE the step so a risky iteration never
            # starts once a pause/kill is desired — SAFE-1.5), then hand the
            # DECISION to the Loop's eg-statechart ``pretick`` transition
            # (guard declaration order mirrors today's precedence: pause,
            # then kill/cancel/stop, then budget, then external event). --
            desired = desired_state() if desired_state is not None else None
            human_signal: str | None = None
            corrig_summary = ""
            if desired:
                from agent_utilities.core.corrigibility import corrigibility_decision

                _corrig_status, corrig_summary = corrigibility_decision(desired)
                human_signal = (
                    desired if desired in ("pause", "kill", "cancel", "stop") else "pause"
                )
            budget_exceeded_flag = resource_optimizer is not None and self._budget_exceeded(
                resource_optimizer
            )
            external_event_fired_flag = event_probe is not None and self._event_fired(
                event_probe
            )
            pre_result = send_loop_statechart_event(
                self.engine,
                loop_id,
                "pretick",
                payload={
                    "human_signal": human_signal,
                    "budget_exceeded": bool(budget_exceeded_flag),
                    "external_event_fired": bool(external_event_fired_flag),
                },
            )
            if pre_result is None:
                raise _wi.WorkItemBackendUnavailable(
                    f"Loop {loop_id!r} has no eg-statechart instance for its WorkItem"
                )
            pre_active = statechart_active_state(pre_result)
            if pre_active in (LoopStatus.PAUSED.value, LoopStatus.CANCELLED.value):
                return _finish(
                    to_status(pre_active, default=LoopStatus.FAILED),
                    reason=corrig_summary or f"human interrupt: {desired}",
                    interrupted=True,
                )
            if pre_active == LoopStatus.BUDGET_EXCEEDED.value:
                return _finish(
                    LoopStatus.BUDGET_EXCEEDED,
                    reason=f"resource budget exceeded: {self._budget_detail(resource_optimizer)}",
                )
            if pre_active == LoopStatus.EXTERNAL_EVENT_SATISFIED.value:
                return _finish(
                    LoopStatus.EXTERNAL_EVENT_SATISFIED,
                    reason="external event signal fired",
                )

            it += 1

            async def _step() -> dict[str, Any]:
                # _iterate may block (subprocess validation / workflow run); offload
                # to a thread so the loop never stalls the event loop.
                return await asyncio.to_thread(self._iterate, loop)

            outcome = await _step()
            outcome = outcome if isinstance(outcome, dict) else {"status": "pending"}
            step_status = to_status(
                outcome.get("status", "pending"), default=LoopStatus.FAILED
            )

            # Fence the iteration durably BEFORE committing any lifecycle so a
            # crash resumes after the last committed step (one WorkItem authority).
            if not _wi.checkpoint_work_item(
                self.engine,
                item_id,
                claim,
                f"checkpoint:iteration:{it}",
            ):
                raise _wi.WorkItemBackendUnavailable(
                    f"Loop {loop_id!r} lost its native lease while checkpointing"
                )

            # -- exit 1 GOAL MET: measure the step. The evaluator is the authority
            # on completion — a callee that self-declares ``completed`` is trusted
            # ONLY when a live measurement confirms it (or when no measurement is
            # available, i.e. the offline/legacy fallback). --
            verdict: GoalEvaluation | None = None
            if evaluator is not None and (
                kind == "develop"
                or step_status is LoopStatus.COMPLETED
                or outcome.get("done")
            ):
                try:
                    verdict = evaluator(loop, outcome)
                except Exception as e:  # noqa: BLE001 — a judge error never crashes the loop
                    logger.debug("run_loop goal_evaluator failed: %s", e)
                    verdict = None

            decided = step_status
            measured_pass = False
            if verdict is not None and verdict.measured:
                if verdict.passed:
                    decided = LoopStatus.COMPLETED
                    measured_pass = True
                elif step_status is LoopStatus.COMPLETED:
                    # Self-declared done but the measurement REJECTS it -> do not
                    # trust it; keep working (demote to a non-terminal heartbeat).
                    decided = LoopStatus.RUNNING

            # -- exit 7 classification. A NON-TERMINAL (retryable) failure is a step
            # that errored yet should be retried (the outcome carries ``error`` /
            # ``retryable``) — distinct from a clean terminal ``failed`` give-up,
            # which still stops the loop at once (legacy-trust, below). Only
            # retryable failures accumulate toward the error threshold; a retryable
            # failure keeps the lease alive as a RUNNING heartbeat so it can be
            # retried until the guard trips or progress resets it. --
            retryable_failure = decided is LoopStatus.FAILED and bool(
                outcome.get("retryable") or outcome.get("error")
            )
            heartbeat_status = LoopStatus.RUNNING if retryable_failure else decided

            # Commit only a NON-terminal heartbeat here; terminal states are
            # committed once, at their guarded ``_finish`` transition below.
            if not is_terminal(heartbeat_status):
                mark_loop_status(
                    self.engine,
                    loop_id,
                    heartbeat_status.value,
                    iteration=it,
                    output=str(outcome.get("output", ""))[:2000],
                )

            if on_iteration is not None:
                try:
                    on_iteration(it, outcome)
                except Exception as e:  # noqa: BLE001 — observability never blocks
                    logger.debug("run_loop on_iteration callback failed: %s", e)

            # -- exit 5 NO PROGRESS data: hash the substantive result and roll the
            # window (persisted on the Loop node, mirroring the fanout
            # _STALL_THRESHOLD pattern). --
            sig = progress_signature(
                decided.value,
                str(outcome.get("output", "")),
                str(outcome.get("checkpoint", "")),
            )
            progressed = (not progress_hashes) or sig != progress_hashes[-1]
            progress_hashes.append(sig)
            self._persist_progress_window(loop_id, progress_hashes, stall_window)

            # -- exit 7 ERROR THRESHOLD signal: N consecutive retryable failures;
            # any non-failure progress resets the run (engine-breaker semantics).
            # ``error_threshold_tripped`` is precomputed here as a plain boolean
            # because ``eg-statechart``'s numeric ``Guard::Ge`` reads persistent
            # machine CONTEXT, never the event payload, and this chart's context
            # is always empty — the CALLER does the comparison and sends the
            # boolean (mirrors the Rust-side W2.5 correction). --
            error_threshold_tripped = False
            if retryable_failure:
                error_threshold_tripped = fail_guard.record_failure()
            elif progressed:
                fail_guard.record_success()

            # -- exit 5 NO PROGRESS: the last N signatures identical -> stalled. --
            stalled_flag = window_is_stalled(progress_hashes, stall_window)

            # -- legacy trust: a terminal status the evaluator did NOT override
            # (offline / no measurement) — the callee's self-declared verdict. --
            callee_terminal = decided.value if is_terminal(decided) else None

            # -- exit 2 TURN CAP / exit 4 WALL CLOCK signals, precomputed as
            # plain booleans for the same reason as error_threshold_tripped. --
            turn_cap_reached = it >= max_it
            deadline_flag = deadline_passed(deadline)

            # Whichever of running/pending/validating this iteration continues
            # as (the ordinary non-terminal heartbeat) — None once ``decided``
            # is itself terminal (the legacy-trust/turn-cap/wall-clock guards
            # above take it from here instead).
            heartbeat_target = (
                heartbeat_status.value if not is_terminal(heartbeat_status) else None
            )

            post_result = send_loop_statechart_event(
                self.engine,
                loop_id,
                "posttick",
                payload={
                    "measured_pass": measured_pass,
                    "retryable_failure": retryable_failure,
                    "error_threshold_tripped": error_threshold_tripped,
                    "stalled": stalled_flag,
                    "callee_terminal": callee_terminal,
                    "turn_cap_reached": turn_cap_reached,
                    "deadline_passed": deadline_flag,
                    "heartbeat_target": heartbeat_target,
                },
            )
            if post_result is None:
                raise _wi.WorkItemBackendUnavailable(
                    f"Loop {loop_id!r} has no eg-statechart instance for its WorkItem"
                )
            active = statechart_active_state(post_result)
            status = to_status(active, default=LoopStatus.FAILED)

            if is_terminal(status):
                if status is LoopStatus.COMPLETED:
                    if measured_pass:
                        return _finish(
                            LoopStatus.COMPLETED,
                            reason=(
                                f"goal met (measured score={verdict.score:.2f}): "
                                f"{verdict.detail}"
                            ),
                        )
                    return _finish(LoopStatus.COMPLETED)
                if (
                    status is LoopStatus.ERROR_THRESHOLD_EXCEEDED
                    and error_threshold_tripped
                ):
                    return _finish(
                        status,
                        reason=(
                            f"{fail_guard.count} consecutive non-terminal failures "
                            f"(threshold {fail_guard.threshold})"
                        ),
                    )
                if status is LoopStatus.STALLED and stalled_flag:
                    return _finish(
                        status,
                        reason=(
                            f"no progress across the last {stall_window} iterations "
                            "(identical status/output)"
                        ),
                    )
                if status is LoopStatus.MAX_ITERATIONS_EXCEEDED and turn_cap_reached:
                    return _finish(
                        status,
                        reason=f"turn cap reached: max_iterations={max_it} without convergence",
                    )
                if status is LoopStatus.WALL_CLOCK_EXCEEDED and deadline_flag:
                    return _finish(
                        status,
                        reason=(
                            "overall wall-clock deadline exceeded after "
                            f"{_time.monotonic() - start_monotonic:.1f}s"
                        ),
                    )
                return _finish(status, reason=f"callee terminal status: {status.value}")

            # A retryable failure keeps looping (already counted); it must NOT
            # fall into the ``done``-flag break on the same iteration.
            if retryable_failure:
                if sleep_s:
                    await asyncio.sleep(sleep_s)
                continue
            if outcome.get("done"):
                break
            if sleep_s:
                await asyncio.sleep(sleep_s)

        # -- The while-condition fell through: decide the PRECISE terminal cause
        # so the exit is diagnosable rather than a generic 'failed'. --
        if is_terminal(status):
            return {"id": loop_id, "status": status.value, "iterations": it}
        # exit 4 WALL CLOCK (checked in the while-condition above).
        if deadline_passed(deadline):
            return _finish(
                LoopStatus.WALL_CLOCK_EXCEEDED,
                reason=(
                    f"overall wall-clock deadline exceeded after "
                    f"{_time.monotonic() - start_monotonic:.1f}s"
                ),
            )
        # exit 2 TURN CAP.
        if it >= max_it:
            return _finish(
                LoopStatus.MAX_ITERATIONS_EXCEEDED,
                reason=f"turn cap reached: max_iterations={max_it} without convergence",
            )
        # Any other non-terminal fall-through (e.g. a ``done`` flag on a
        # non-terminal status) settles as a plain failure.
        return _finish(
            LoopStatus.FAILED, reason="loop ended without reaching a terminal state"
        )

    @staticmethod
    def _resume_iteration(engine: Any, loop_id: str) -> int:
        """Read the last fenced iteration from the Loop WorkItem."""

        from agent_utilities.orchestration import work_item as _wi

        item = _wi.get_work_item(engine, _wi.loop_work_item_id(loop_id))
        checkpoint = str((item or {}).get("checkpoint_id") or "")
        match = re.fullmatch(r"checkpoint:iteration:([1-9][0-9]*)", checkpoint)
        return int(match.group(1)) if match else 0

    def _run_standardize(self) -> dict[str, Any]:
        """Run the enterprise standardization + consolidation pass (CONCEPT:AU-KG.ontology.populated-at-import-real-3).

        Propose-only: materializes enterprise-standard interfaces, scores per-asset
        conformance drift, and emits ranked consolidation recommendations. No source
        asset is mutated and nothing auto-merges.
        """
        from ..standardization import run_standardization_pass

        return run_standardization_pass(self.engine)

    def _run_archivebox_intake(self) -> dict[str, Any]:
        """Pull new preserved ArchiveBox snapshots into the KG (delta, idempotent).

        Delegates to the unified ``sync_source`` entrypoint (``_sync_archivebox``):
        enumerate snapshots past the watermark, ingest each archived URL through the
        DOCUMENT path (ArchiveBox-preferred fetch + research-paper extraction).
        (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
        """
        from ..core.source_sync import sync_source

        return sync_source(self.engine, "archivebox", mode="delta")

    def _run_freshrss_intake(self) -> dict[str, Any]:
        """Pull curated FreshRSS items through the world-model gate (delta).

        Delegates to the unified ``sync_source`` entrypoint (``_sync_freshrss``):
        enumerate items past the GReader ``ot`` watermark and route each through the
        :class:`WorldModelPipelineRunner` relevance gate — only KG-relevant/novel (or
        agent-force-flagged) items are fully ingested; Research/arXiv items route to
        the research path. (CONCEPT:AU-KG.ingest.news-finance-tech-sibling / KG-2.117)
        """
        from ..core.source_sync import sync_source

        return sync_source(self.engine, "freshrss", mode="delta")

    def _run_breadth(self) -> dict[str, Any]:
        """Ingest the OSS/repos/docs corpus (idempotent).

        Roots come, in order of precedence, from the explicit
        ``KG_BREADTH_LIBRARY_ROOTS`` / ``KG_BREADTH_REPO_ROOTS`` (comma-separated)
        overrides, else are auto-discovered from the XDG ``workspace.yml`` — the
        single declaration of ALL ecosystem projects we want assimilated. So the
        loop self-configures: ``assimilate`` always has the codebase capability
        map to compare research against, with no env config required. Content-
        addressed ingest makes re-runs cheap. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
        """
        from dataclasses import asdict

        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.workspace_config import workspace_project_roots

        from ..assimilation import run_breadth_ingest

        # Read a fresh AgentConfig() (not the import-time singleton) so runtime
        # root changes are honored.
        _cfg = AgentConfig()
        libs_raw = _cfg.kg_breadth_library_roots
        repos_raw = _cfg.kg_breadth_repo_roots
        libs = [p.strip() for p in (libs_raw or "").split(",") if p.strip()]
        repos = [p.strip() for p in (repos_raw or "").split(",") if p.strip()]
        # No explicit roots ⇒ self-configure from the workspace.yml ecosystem.
        if not libs and not repos:
            repos = workspace_project_roots()
        if not libs and not repos:
            return {"skipped": True, "reason": "no roots configured or discoverable"}
        return asdict(
            run_breadth_ingest(self.engine, library_roots=libs, repo_roots=repos)
        )

    def _finalize_metrics(self, report: dict[str, Any], start: float) -> None:
        """Attach cycle metrics, log a health summary, persist an EvolutionCycle node."""
        import time
        import uuid

        m = report["metrics"]
        m["duration_ms"] = round((time.monotonic() - start) * 1000, 1)
        m["error_count"] = len(report["errors"])
        assim = (
            report.get("assimilate")
            if isinstance(report.get("assimilate"), dict)
            else {}
        )
        m["open_gaps"] = (assim or {}).get("open_gaps", 0)
        logger.info(
            "golden-loop cycle: duration=%sms errors=%d intake=%d open_gaps=%s stages=%s",
            m["duration_ms"],
            m["error_count"],
            report["topics_intake"],
            m["open_gaps"],
            m["stage_ms"],
        )
        if report["errors"]:
            logger.warning("golden-loop cycle errors: %s", report["errors"])
        # Monitoring: persist a queryable EvolutionCycle node (best-effort).
        # One node type (``EvolutionCycle``) and id convention (``evo_cycle_<ts>``)
        # shared with the daemon tick (``engine_tasks._tick_evolution``) so a
        # ``MATCH (e:EvolutionCycle)`` sees both on-demand and scheduled cycles;
        # ``triggered_by`` discriminates the source. ``errors``/``stage_ms`` are
        # JSON-encoded: the durable (Postgres) backend cannot adapt a raw
        # dict/list into a column value.
        import json
        import time as _time

        now_iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        # Share the id with the live beacon (CONCEPT:AU-KG.research.evolutionstate-live-surface-per) so the finalized
        # EvolutionCycle and the mid-flight beacon cross-reference one cycle.
        cycle_id = self._cycle_id or (
            f"evo_cycle_{_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
        )
        # Conform to the EvolutionCycle table schema (schema_definition.py): only
        # known columns are first-class; cycle-specific metrics go in ``metadata``
        # (a JSON STRING column) so the durable (Postgres) backend accepts them.
        try:
            self.engine.add_node(
                cycle_id,
                "EvolutionCycle",
                properties={
                    "triggered_by": "loop_engine",
                    "topics_scanned": report["topics_intake"],
                    "created_at": now_iso,
                    "timestamp": now_iso,
                    "metadata": json.dumps(
                        {
                            "duration_ms": m["duration_ms"],
                            "error_count": m["error_count"],
                            "errors": report["errors"][:10],
                            "topics_intake": report["topics_intake"],
                            "open_gaps": m["open_gaps"],
                            "stage_ms": m["stage_ms"],
                        }
                    ),
                },
            )
        except Exception as e:  # noqa: BLE001 - monitoring persist is best-effort
            logger.debug("EvolutionCycle persist failed: %s", e)

        # CONCEPT:AU-AHE.sdd.recursive-improvement-instrumentation-aggregating / SAFE-1.3 — recursive-improvement velocity. Read the
        # loop's own audit streams (EvolutionCycle + ProposalPublication +
        # CapabilityRatchetResult) back into one velocity reading and persist it, so
        # the loop self-instruments: is it still improving, how fast, and is it
        # emitting code or only prose? A stalling verdict is the research-gets-harder
        # signal. Best-effort — never aborts the cycle.
        try:
            from .improvement_ledger import ImprovementLedger

            velocity = ImprovementLedger(self.engine).record()
            report["velocity"] = velocity.to_dict()
            if velocity.verdict == "stalling":
                logger.warning(
                    "[AHE-3.26] self-improvement stalling: %s",
                    "; ".join(velocity.signals),
                )
        except Exception as e:  # noqa: BLE001 — instrumentation never blocks the loop
            logger.debug("[AHE-3.26] velocity ledger failed: %s", e)

        # CONCEPT:AU-KG.research.saturation-gauge-aggregates-four — saturation gauge. Aggregate open_gaps trend + the just-
        # recorded velocity verdict + ingestion coverage into ONE 0..1 reading and
        # stamp it on the report + the live beacon; when saturated (and stalling),
        # surface a request-more recommendation (NEVER auto-fetch). Best-effort.
        try:
            from .evolution_state import (
                _open_gaps_trend,
                emit_saturation_signal,
                saturation_gauge,
            )

            gaps = _open_gaps_trend(self.engine)
            verdict = str((report.get("velocity") or {}).get("verdict", "idle"))
            gauge = saturation_gauge(
                coverage_pct=None,  # cheap path: skip the coverage probe in the hot loop
                velocity_verdict=verdict,
                gaps_recent=float(gaps.get("recent", 0) or 0),
                gaps_prior=float(gaps.get("prior", 0) or 0),
            )
            report["saturation"] = gauge
            if gauge.get("request_more"):
                emit_saturation_signal(self.engine, gauge)
                logger.info(
                    "[KG-2.291] evolution saturated (gauge=%.2f) — %s",
                    gauge["gauge"],
                    gauge["recommendation"],
                )
        except Exception as e:  # noqa: BLE001 — gauge is observability only
            logger.debug("[KG-2.291] saturation gauge failed: %s", e)
            gauge = None

        # Close out the live beacon (CONCEPT:AU-KG.research.evolutionstate-live-surface-per).
        if self._beacon is not None:
            try:
                self._beacon.finish(
                    open_gaps=m.get("open_gaps", 0),
                    errors=m.get("error_count", 0),
                    saturation=(gauge or {}).get("gauge") if gauge else None,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("beacon.finish failed: %s", e)

    def _distill_specs(self, topics: list[dict[str, Any]]) -> list[str]:
        """Distil ``SpecDraft`` markdown into ``.specify/specs/kg-distilled/``."""
        from ..enrichment.cards import make_lite_llm_fn
        from ..enrichment.distill import what_specs_could_we_build, write_spec_drafts
        from ..enrichment.extractors.document import Concept

        # Bounded inputs: the intake topics as concepts; edges/code maps left
        # empty so distillation stays cheap (candidates come from concept value).
        concepts = [
            Concept(id=t["id"], name=t["name"], kind="topic", summary="", source_ids=[])
            for t in topics
        ]
        specs = what_specs_could_we_build(
            self.codebase_root, concepts, [], {}, make_lite_llm_fn(), limit=3
        )
        if not specs:
            return []
        # propose_only: write DRAFTS under .specify/ only.
        paths = write_spec_drafts(specs, self.codebase_root)

        # CONCEPT:AU-KG.research.close-distill-develop-seam — close the distill→develop seam. Persist each draft as a
        # first-class, queryable :SpecProposal (status pending_review) linked to its
        # source concepts, so the distilled spec is no longer a dead-end .md file but
        # a develop-able + reviewable work item. The spec is fed into the existing
        # promotion pipeline only AFTER the OS-5.73 spec-review checkpoint approves it.
        from agent_utilities.core.config import config as _cfg

        from .spec_proposals import auto_advance_specs, persist_spec_proposal

        spec_ids: list[str] = []
        padded_paths = paths + [""] * (len(specs) - len(paths))
        for spec, path in zip(specs, padded_paths, strict=False):
            sid = persist_spec_proposal(self.engine, spec, spec_path=path)
            if sid:
                spec_ids.append(sid)
        self._beacon and self._beacon.enter(
            "distill",
            detail=f"distilled {len(spec_ids)} spec(s): "
            + ", ".join(s.title for s in specs[:3]),
        )
        # Default = review-first (propose-and-hold). Only when KG_LOOP_AUTO_DEVELOP is
        # explicitly on does the 24/7 loop auto-advance specs through the spec_promotion
        # gate (which itself defaults to approval_required, so it only develops where an
        # operator relaxed the tier). Acquisition is never auto-run.
        if getattr(_cfg, "kg_loop_auto_develop", False) and spec_ids:
            try:
                auto_advance_specs(self.engine)
            except Exception as e:  # noqa: BLE001 — never blocks the cycle
                logger.debug("[OS-5.73] auto_advance_specs failed: %s", e)
        return paths

    def _synthesize_team(self, topics: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Synthesize a team proposal addressing the open topics; persist nodes."""
        cap = self._capability_search()
        if cap is None:
            return None
        from ..enrichment.cards import make_lite_llm_fn
        from ..enrichment.synthesize import persist_synthesis, synthesize_team

        names = ", ".join(t["name"] for t in topics[:5]) or "open KG topics"
        goal = f"Propose how to address these open knowledge-graph topics: {names}"
        team, members = synthesize_team(goal, cap, make_lite_llm_fn(), max_members=4)
        nodes = edges = 0
        if self.propose_only:
            # Persist the PROPOSAL (TeamSpec/AgentSpec nodes) — not executed.
            try:
                nodes, edges = persist_synthesis(self.engine.backend, team, *members)
            except Exception as e:  # noqa: BLE001
                logger.debug("persist_synthesis failed: %s", e)

        # GOVERNED auto-merge (CONCEPT:AU-AHE.assimilation.research-auto-merge): consider promoting the team
        # proposal to active. Disabled by default → stays proposal-only; only a
        # high-quality, governance-valid proposal auto-merges when enabled.
        merge: dict[str, Any] | None = None
        try:
            ev = self._merger.consider(team)
            merge = {
                "proposal_id": ev.proposal_id,
                "quality_score": round(ev.quality_score, 4),
                "merged": ev.merged,
                "reason": ev.reason,
                "audit_ref": ev.audit_ref,
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("auto-merge consideration failed: %s", e)

        return {
            "goal": goal,
            "lead": getattr(team, "lead", None) or getattr(team, "name", None),
            "members": [getattr(m, "name", "?") for m in members],
            "persisted_nodes": nodes,
            "persisted_edges": edges,
            "auto_merge": merge,
        }

    def _synthesize_search_tasks(self, limit: int = 5) -> dict[str, Any]:
        """Build shortcut-resistant deep-search tasks from the evidence graph.

        Selects candidate answer entities, runs the FORT-distilled synthesizer
        (CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72), keeps only tasks whose shortcut report is
        clear, drafts a JSONL corpus under ``.specify/specs/search-tasks/`` and
        (propose-only) persists each as a ``SearchTask`` node. Returns a summary.
        """
        import json
        from pathlib import Path

        from ..search_synthesis import synthesize

        reader = _EngineReader(self.engine)
        rows = reader.query("MATCH (n) RETURN n LIMIT $k", {"k": limit * 4})
        candidates = [
            (r.get("n") or {}).get("id") for r in rows if (r.get("n") or {}).get("id")
        ]

        tasks: list[dict[str, Any]] = []
        for answer_id in candidates:
            if len(tasks) >= limit:
                break
            if not answer_id:
                continue
            try:
                task = synthesize(reader, str(answer_id), hops=2)
            except Exception as e:  # noqa: BLE001
                logger.debug("search-task synthesis failed for %s: %s", answer_id, e)
                continue
            if task.risk_report.clear and task.difficulty >= 1:
                tasks.append(task.to_dict())

        corpus_path = ""
        if tasks:
            out_dir = Path(self.codebase_root) / ".specify" / "specs" / "search-tasks"
            out_dir.mkdir(parents=True, exist_ok=True)
            corpus_file = out_dir / "tasks.jsonl"
            corpus_file.write_text(
                "\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8"
            )
            corpus_path = str(corpus_file)

        persisted = 0
        if self.propose_only:
            for t in tasks:
                try:
                    self.engine.add_node(
                        f"SearchTask:{t['answer_id']}",
                        {
                            "type": "SearchTask",
                            "question": t["question"],
                            "answer_id": t["answer_id"],
                            "difficulty": t["difficulty"],
                            "status": "proposal",
                        },
                    )
                    persisted += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("SearchTask persist failed: %s", e)

        return {
            "candidates": len(candidates),
            "tasks": len(tasks),
            "persisted_nodes": persisted,
            "corpus_path": corpus_path,
        }


def _default_develop_runner(cmd: str, cwd: str) -> tuple[bool, str]:
    """Run a develop Loop's validation command once; success = exit code 0.

    Synchronous (the controller advances one iteration per cycle), timeout-bounded,
    best-effort — mirrors the durable goal loop's validation step (``sessions``) but
    as a single step in the unified hot path. (CONCEPT:AU-KG.research.these-properties-carry)

    This dangerous compatibility runner is disabled by default. When explicitly
    enabled, it accepts one bounded argv command, resolves an operator-allowlisted
    executable from ``PATH``, does not invoke a shell, and passes only a minimal
    non-secret environment. Prefer an injected governed sandbox runner.
    """
    import os
    import shlex
    import shutil
    import signal
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    from agent_utilities.core.config import config

    if not config.kg_loop_allow_host_validation:
        return False, "host validation is disabled"
    if not cmd or len(cmd.encode("utf-8")) > 16 * 1024:
        return False, "validation command is empty or exceeds the configured limit"
    try:
        argv = shlex.split(cmd, posix=os.name != "nt")
    except ValueError:
        return False, "validation command has invalid quoting"
    if (
        not argv
        or len(argv) > 64
        or any(len(arg) > 4096 or "\x00" in arg for arg in argv)
    ):
        return False, "validation command arguments exceed the configured limits"

    executable_name = Path(argv[0]).name.lower()
    if executable_name.endswith(".exe"):
        executable_name = executable_name[:-4]
    allowed = {
        value.strip().lower()
        for value in config.kg_loop_host_validation_executables.split(",")
        if value.strip()
    }
    if (
        argv[0] != Path(argv[0]).name
        or executable_name not in allowed
        or executable_name in {"sh", "bash", "zsh", "cmd", "powershell", "pwsh"}
    ):
        return False, "validation executable is not operator-allowlisted"
    executable = shutil.which(argv[0])
    if executable is None:
        return False, "validation executable is unavailable"
    root = Path(cwd).expanduser().resolve(strict=False)
    if not root.is_dir():
        return False, "validation working directory is unavailable"

    env_names = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
    }
    child_env = {name: os.environ[name] for name in env_names if name in os.environ}
    child_env.update({"CI": "1", "NO_COLOR": "1", "PYTHONNOUSERSITE": "1"})

    output_limit = 2 * 1024 * 1024
    proc: subprocess.Popen[bytes] | None = None
    stop_reason = ""
    try:
        with tempfile.TemporaryFile(mode="w+b") as output:
            popen_kwargs: dict[str, Any] = {
                "cwd": root,
                "env": child_env,
                "stdin": subprocess.DEVNULL,
                "stdout": output,
                "stderr": subprocess.STDOUT,
                "shell": False,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen([executable, *argv[1:]], **popen_kwargs)
            deadline = time.monotonic() + 600
            while proc.poll() is None:
                if os.fstat(output.fileno()).st_size > output_limit:
                    stop_reason = "validation output limit exceeded"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "validation command timed out"
                    break
                time.sleep(0.05)
            if stop_reason:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
            proc.wait(timeout=5)
            size = os.fstat(output.fileno()).st_size
            output.seek(max(0, size - 2000))
            tail = output.read(2000).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — never abort the cycle
        if proc is not None and proc.poll() is None:
            proc.kill()
        return False, f"validation command failed to run ({type(exc).__name__})"
    if stop_reason:
        return False, f"{stop_reason}\n{tail}"
    out = f"exit={proc.returncode}\n{tail}"
    return proc.returncode == 0, out


def _default_skill_runner(
    skill_ref: str, objective: str, engine: Any = None
) -> tuple[bool, str]:
    """Execute a skill / skill-workflow Loop via the orchestration engine.

    Compiles (if needed) and runs the workflow named/identified by ``skill_ref``;
    best-effort so a missing orchestrator degrades to a failed step, never a crash.
    (CONCEPT:AU-KG.research.these-properties-carry)
    """
    try:
        from ...orchestration.manager import Orchestrator

        mgr = Orchestrator(engine)
        wid = skill_ref
        if not skill_ref.startswith("workflow:"):
            wid = _run_coro(mgr.compile_workflow(skill_ref, objective or skill_ref))
        result = _run_coro(mgr.execute_workflow(wid, task=objective))
        return True, str(result)[:2000]
    except Exception as e:  # noqa: BLE001
        return False, f"skill execution failed: {e}"


class _EngineReader:
    """Adapt an :class:`IntelligenceGraphEngine` to the search-synthesis read API."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def query(self, cypher: str, params: Any = None) -> list[dict[str, Any]]:
        backend = getattr(self._engine, "backend", None)
        if backend is not None and hasattr(backend, "execute"):
            return backend.execute(cypher, params or {}) or []
        if hasattr(self._engine, "query"):
            return self._engine.query(cypher, params or {}) or []
        return []


def run_assimilation_pass(
    engine: Any = None,
    *,
    synthesize: bool = False,
    top_n: int = 5,
    force: bool = False,
    synth_fn: Any = None,
    restrict_to: set[str] | None = None,
    matrix_node_id: str = "feature_matrix:latest",
) -> dict[str, Any]:
    """Run only the graph-compute assimilation middle (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

    dedup → auto-satisfy → synergy → rank (idempotent via the watermark); with
    ``synthesize=True`` also generate grounded SDD plan proposals for the top-N
    open gaps. The MCP ``graph_evolution(action="assimilate")`` action and the
    evolution skill call this; the daemon runs it as part of ``run_one_cycle``.

    ``synth_fn`` overrides plan synthesis (e.g. the deterministic offline
    ``assimilation.plan_synthesis._default_synth``) so a caller/test can run fully
    offline without the planner LLM; ``None`` keeps the default (timeout-bounded
    LLM, falling back to the offline synthesizer).
    """
    if engine is None:
        from ..core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_or_create()
    # OS-5.71 — the assimilation pass runs OFF the task queue, so profile it under a
    # contextvar span (capturing enrich-embeds + matcher LLM-judges automatically)
    # and persist it as a :ProfileSpan so profile_report covers it like any lane.
    from ..core.ingest_profile import profile_ingest, record_offqueue_span

    with profile_ingest("assimilate") as _prof:
        rep = LoopController(engine)._run_assimilate(
            force=force, restrict_to=restrict_to, matrix_node_id=matrix_node_id
        )
    record_offqueue_span(engine, "assimilate", _prof)
    rep["profile"] = _prof.to_dict()
    # Synthesis is idempotent — plans upsert by ``plan_id`` — so run it whenever a
    # caller asks for it, even if the rank pass was skipped as "unchanged".
    # Previously synthesis was gated behind the rank watermark, so a prior bare
    # ``assimilate()`` bumped the watermark and silently suppressed a follow-up
    # ``synthesize`` (the only reason ``force`` was ever needed). The watermark's
    # job is to avoid redundant *re-ranking*, not to block an explicit synthesis
    # request. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    if synthesize:
        from ..assimilation import synthesize_plans

        plans = synthesize_plans(engine, top_n=top_n, synth_fn=synth_fn)
        rep["proposed_plans"] = [
            {"plan_id": p.plan_id, "feature_id": p.feature_id, "title": p.title}
            for p in plans
        ]
        if rep.get("skipped"):
            # We still did real work (synthesis); reflect that to the caller.
            rep["skipped"] = False
            rep.setdefault("reason", "synthesis-only (rank unchanged)")
    return rep
