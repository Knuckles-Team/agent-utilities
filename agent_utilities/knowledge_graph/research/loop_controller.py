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
Exposed on-demand (the ``graph_loops`` / ``graph_orchestrate`` MCP tools and the REST
twin) and via a throttled daemon tick.
"""

from __future__ import annotations

import hashlib
import logging
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
    ) -> None:
        self.engine = engine
        self.codebase_root = codebase_root or setting("WORKSPACE_PATH") or "."
        # Execution backends for the non-research Loop kinds (CONCEPT:AU-KG.research.these-properties-carry L3),
        # injectable so the develop/skill stages are unit-testable without a real
        # subprocess / workflow engine. Defaults are wired lazily on first use.
        self._develop_runner = develop_runner
        self._skill_runner = skill_runner
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
    ) -> dict[str, Any]:
        """Execute one cycle. Returns a structured, JSON-able report.

        Stages (each best-effort + timed; one failing stage never aborts the cycle):
        ``breadth`` (env ``KG_LOOP_BREADTH`` — ingest the OSS/repos/docs corpus,
        idempotent/content-addressed) → ``assimilate`` (dedup→gap→synergy→rank,
        idempotent via the state watermark) → ``reason`` → ``mine_discovery`` (env
        ``KG_LOOP_MINE_DISCOVERY``, default ON — the discovery-flywheel mining pass,
        CONCEPT:AU-KG.evolution.mining-flywheel) → ``belief_revision`` (env
        ``KG_LOOP_BELIEF_REVISION``, default ON — confidence propagation + light
        TMS over ``Belief`` nodes (CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision))
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
            f"evo_cycle_{time.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
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
            rows = q(f"MATCH (n) WHERE n.type IN [{type_list}] RETURN count(n) AS c")
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
        hash of ((id, status, content_hash)) over the input node types.
        """
        c = self._cheap_input_count()
        if c is not None:
            return f"count:{c}"
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return ""
        try:
            node_iter = graph.nodes(data=True)
        except TypeError:
            return ""
        items = sorted(
            (nid, str(d.get("status", "")), str(d.get("content_hash", "")))
            for nid, d in node_iter
            if isinstance(d, dict)
            and str(d.get("type", "")).lower() in _WATERMARK_TYPES
        )
        return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]

    def _load_watermark(self) -> str | None:
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
            if not claim_loop(
                self.engine, loop["id"], current_status=str(loop.get("status") or "")
            ):
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
        runner = self._develop_runner or _default_develop_runner
        if not cmd:
            # no command to validate → nothing to advance; leave it active
            return {"status": loop.get("status", "pending"), "output": ""}
        ok, output = runner(cmd, self.codebase_root)
        return {"status": "completed" if ok else "pending", "output": output}

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

    # -- durable, resumable run-to-completion (CONCEPT:AU-KG.research.these-properties-carry + OS-5.16) --- #
    async def run_loop(
        self,
        loop: dict[str, Any],
        *,
        max_iterations: int | None = None,
        on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
        desired_state: Callable[[], str | None] | None = None,
        sleep_s: float = 0.0,
        durable: Any = None,
    ) -> dict[str, Any]:
        """Drive ONE Loop to completion, durably and resumably — for any kind.

        This is the generalized fold of the old goal-runner: it owns the durable,
        checkpointed, corrigible iteration machinery once, for research/develop/skill
        alike (durability is cross-cutting, not goal-specific):

        - **Resume** from the last durable checkpoint (``DurableExecutionManager``,
          backend-selected SQLite/Postgres via ``state_store``) so a crash/redelivery
          continues near the in-flight iteration instead of replaying from zero.
        - **Durable iteration**: each step runs under an idempotency key
          (``<loop>:iter:<n>``) — at-least-once retries, exactly-once effect.
        - **Corrigible interruption** (SAFE-1.5): ``desired_state`` (e.g. a fleet
          pause/kill signal) is honored each iteration — checkpoint and yield without
          resistance.
        - Advances the Loop node lifecycle (:func:`mark_loop_status`) each step and
          invokes ``on_iteration`` so a caller can record observability (e.g. the
          goal console).

        Returns ``{"id", "status", "iterations", "interrupted"?}``.
        """
        import asyncio

        from agent_utilities.orchestration.durable_execution import (
            DurableExecutionManager,
        )

        from .loops import TERMINAL_STATUS, claim_loop, mark_loop_status

        loop_id = loop["id"]
        max_it = int(max_iterations or loop.get("max_iterations") or 20)
        if durable is None:
            durable = DurableExecutionManager(session_id=loop_id)
        it = self._resume_iteration(durable)
        status = str(loop.get("status") or "running")
        # Atomically claim the Loop as in-flight (status → running via the engine
        # CAS) so a concurrent daemon cycle / peer host / graph_loops run can't
        # double-drive it — only the winner advances. A resume (it > 0) is a
        # legitimate re-entry of a Loop this caller already owns (durable
        # rehydration left it 'running'/'orphaned'), so a lost CAS there is not
        # fatal; on a fresh start (it == 0) a lost claim means someone else owns
        # it and we yield. (CONCEPT:AU-KG.compute.user-override-prompt-library, was a blind flip — KG-2.78)
        won = claim_loop(self.engine, loop_id, current_status=status)
        if not won and it == 0:
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
        # On resume (it > 0) we already own the Loop; the claim CAS won't flip a
        # node already 'running', so re-stamp the resumed iteration onto it.
        if it > 0:
            mark_loop_status(self.engine, loop_id, "running", iteration=it)

        while it < max_it and status not in TERMINAL_STATUS:
            if desired_state is not None:
                desired = desired_state()
                if desired:
                    from agent_utilities.core.corrigibility import (
                        corrigibility_decision,
                    )

                    final, summary = corrigibility_decision(desired)
                    fstatus = final.value if final is not None else "paused"
                    mark_loop_status(
                        self.engine, loop_id, fstatus, iteration=it, output=summary
                    )
                    return {
                        "id": loop_id,
                        "status": fstatus,
                        "iterations": it,
                        "interrupted": True,
                    }

            it += 1

            async def _step() -> dict[str, Any]:
                # _iterate may block (subprocess validation / workflow run); offload
                # to a thread so the durable loop never stalls the event loop.
                return await asyncio.to_thread(self._iterate, loop)

            outcome = await durable.arun_durable_action(
                node_id=f"{loop_id}:iter:{it}",
                action=_step,
                idempotency_key=f"{loop_id}:{it}",
                state={"iteration": it, "kind": loop.get("kind", "research")},
            )
            outcome = outcome if isinstance(outcome, dict) else {"status": "pending"}
            status = str(outcome.get("status", "pending"))
            mark_loop_status(
                self.engine,
                loop_id,
                status,
                iteration=it,
                output=str(outcome.get("output", ""))[:2000],
            )
            if on_iteration is not None:
                try:
                    on_iteration(it, outcome)
                except Exception as e:  # noqa: BLE001 — observability never blocks
                    logger.debug("run_loop on_iteration callback failed: %s", e)
            if status in TERMINAL_STATUS or outcome.get("done"):
                break
            if sleep_s:
                await asyncio.sleep(sleep_s)

        if status not in TERMINAL_STATUS and status != "completed":
            # ran out of iterations without converging
            status = "failed"
            mark_loop_status(self.engine, loop_id, status, iteration=it)
        return {"id": loop_id, "status": status, "iterations": it}

    @staticmethod
    def _resume_iteration(durable: Any) -> int:
        """Read the durable checkpoint → number of already-applied iterations."""
        import json

        try:
            pending = durable.resume_session()
        except Exception as e:  # noqa: BLE001 — recovery is best-effort
            logger.debug("durable resume failed: %s", e)
            return 0
        if not pending:
            return 0
        state = pending.get("state")
        try:
            state = json.loads(state) if isinstance(state, str) else (state or {})
        except (TypeError, ValueError):
            state = {}
        prior = state.get("iteration")
        # the pending iteration was in flight (never completed) → re-run it
        return (prior - 1) if isinstance(prior, int) and prior > 0 else 0

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
            f"evo_cycle_{_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
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

    Security: ``cmd`` is the operator-authored validation command from a develop
    Loop definition (e.g. ``pytest -q && ruff check``), a trusted internal source —
    the same trust boundary as the engine's other intentional dynamic-execution
    sites. ``shell=True`` is deliberate so those commands can use shell operators
    (``&&``, pipes, env expansion); it is never fed external/untrusted input.
    """
    import subprocess

    try:
        proc = subprocess.run(
            cmd,
            shell=True,  # nosec B602 — trusted operator-authored validation command (see docstring)
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as e:  # noqa: BLE001 — never abort the cycle
        return False, f"validation command failed to run: {e}"
    out = f"exit={proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-500:]}"
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
    open gaps. The MCP ``graph_orchestrate(action="assimilate")`` action and the
    evolution skill call this; the daemon runs it as part of ``run_one_cycle``.

    ``synth_fn`` overrides plan synthesis (e.g. the deterministic offline
    ``assimilation.plan_synthesis._default_synth``) so a caller/test can run fully
    offline without the planner LLM; ``None`` keeps the default (timeout-bounded
    LLM, falling back to the offline synthesizer).
    """
    if engine is None:
        from ..core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
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
