"""The self-evolution "golden loop" controller (propose-only v1).

CONCEPT:KG-2.7 / KG-2.10 — research assimilation + orchestration synthesis.

Composes existing primitives into one cycle that makes the KG self-improving
WITHOUT auto-merging anything (propose-only):

    intake  → unresolved ``Concept`` topics (no ``ADDRESSED_BY``)
    acquire → semantically related sources for each topic (research/search)
    resolve → ``ADDRESSES`` edges source→topic so the loop converges
    distill → ``SpecDraft`` markdown into ``.specify/specs/kg-distilled/`` (gated)
    synth   → a ``TeamSpec``/``AgentSpec`` proposal persisted to the KG

Every artifact is a DRAFT/proposal: spec markdown under ``.specify/`` and KG
proposal nodes. No code execution, no PR merge, no edits outside ``.specify``.
Exposed on-demand (skill-workflow / MCP) and via a throttled daemon tick.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from agent_utilities.core.config import setting

from ..adaptation.topic_resolver import mark_addressed, unresolved_topics
from .search import acquire_for_topic

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
    # Enterprise standardization inputs (CONCEPT:KG-2.49): new harvested assets or
    # edited standards re-trigger the standardize stage.
    "enterprise_resource",
    "enterprise_standard",
}
_WATERMARK_NODE = "assimilation:watermark"


class GoldenLoopController:
    """Run one propose-only self-evolution cycle over the KG."""

    def __init__(
        self,
        engine: Any,
        *,
        codebase_root: str | None = None,
        propose_only: bool = True,
        auto_merge: bool | None = None,
        regression_check: Any = None,
    ) -> None:
        self.engine = engine
        self.codebase_root = codebase_root or setting("WORKSPACE_PATH") or "."
        # propose_only is always True in v1 — kept explicit so a future
        # human-approved apply path is a deliberate flip, never accidental.
        self.propose_only = propose_only
        # Governed auto-merge (CONCEPT:AHE-3.14) — OFF by default. Enabled
        # explicitly (auto_merge=True) or via KG_GOLDEN_AUTO_MERGE=1; only then
        # do high-quality, governance-valid proposals promote proposal→active.
        # ``regression_check`` gates failure-remediation merges (CONCEPT:AHE-3.18)
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
    ) -> dict[str, Any]:
        """Execute one cycle. Returns a structured, JSON-able report.

        Stages (each best-effort + timed; one failing stage never aborts the cycle):
        ``breadth`` (env ``KG_GOLDEN_BREADTH`` — ingest the OSS/repos/docs corpus,
        idempotent/content-addressed) → ``assimilate`` (dedup→gap→synergy→rank,
        idempotent via the state watermark) → intake/acquire/resolve →
        ``distill`` (env ``KG_GOLDEN_DISTILL``) → ``synthesize``. The report carries
        a ``metrics`` block (per-stage timings + error count) and is persisted as an
        ``EvolutionCycle`` node for monitoring.
        """
        import time

        from agent_utilities.core.config import config

        if distill is None:
            distill = config.kg_golden_distill
        if breadth is None:
            breadth = config.kg_golden_breadth
        if standardize is None:
            standardize = config.kg_golden_standardize

        report: dict[str, Any] = {
            "propose_only": self.propose_only,
            "topics_intake": 0,
            "topics_resolved": 0,
            "sources_linked": 0,
            "breadth": None,
            "assimilate": None,
            "standardize": None,
            "spec_drafts": [],
            "team": None,
            "errors": [],
            "metrics": {"stage_ms": {}},
        }
        cycle_start = time.monotonic()

        def _stage(name: str, fn):
            """Run a stage best-effort, capture timing + any error."""
            t0 = time.monotonic()
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

        # -1. BREADTH — ingest the OSS/repos/docs corpus (idempotent; opt-in).
        if breadth:
            report["breadth"] = _stage("breadth", self._run_breadth)

        # 0. ASSIMILATE — graph-compute middle (dedup/gap/synergy/rank), idempotent.
        if assimilate:
            report["assimilate"] = _stage(
                "assimilate", lambda: self._run_assimilate(force=force_assimilate)
            )

        # 0b. STANDARDIZE — enterprise standardization + consolidation (CONCEPT:KG-2.49),
        # propose-only. Gated (KG_GOLDEN_STANDARDIZE) since it requires a harvested
        # enterprise estate; idempotent (CONFORMS_TO/ABSORBED_INTO cleared on re-write).
        if standardize:
            report["standardize"] = _stage("standardize", self._run_standardize)

        # 1. INTAKE — open topics the loop should address. Caller-supplied topics
        # (e.g. the failure-ingest tick's just-materialized failure_gap concepts)
        # bypass the generic unresolved_topics scan so a brand-new gap is addressed
        # deterministically instead of competing for a slot in an arbitrarily-
        # ordered, limited scan over hundreds of existing concepts. (CONCEPT:AHE-3.18)
        if topics is not None:
            topics = topics[:max_topics] if max_topics else list(topics)
            report["metrics"]["stage_ms"]["intake"] = 0.0
        else:
            topics = (
                _stage("intake", lambda: unresolved_topics(self.engine, max_topics))
                or []
            )
        report["topics_intake"] = len(topics)

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
                    srcs = acquire_for_topic(self.engine, t, embed_fn=embed_fn)
                    if srcs:
                        n = mark_addressed(
                            self.engine, t["id"], srcs, source="golden_loop"
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

        self._finalize_metrics(report, cycle_start)
        return report

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

    def _run_assimilate(self, *, force: bool = False) -> dict[str, Any]:
        """Run dedup → auto-satisfy → synergy → rank over the feature graph.

        Idempotent: if the input watermark is unchanged since the last cycle (and
        not ``force``), skip the work. The ranked gaps are exclusion-filtered to
        ``open_features`` (satisfied/superseded/implemented features are never
        re-proposed). CONCEPT:KG-2.7.
        """
        pre = self._state_watermark()
        if not force and pre and pre == self._load_watermark():
            return {"skipped": True, "reason": "unchanged", "watermark": pre}

        from ..assimilation import (
            auto_satisfy,
            dedup_features,
            rank_features,
            synergy_bundles,
        )

        dedup = dedup_features(self.engine)
        gap = auto_satisfy(self.engine)
        syn = synergy_bundles(self.engine)
        ranked = rank_features(self.engine)

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
            "duplicates_superseded": dedup.duplicates_superseded,
            "auto_satisfied": gap.satisfied,
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
            "watermark": watermark,
        }

    def _run_standardize(self) -> dict[str, Any]:
        """Run the enterprise standardization + consolidation pass (CONCEPT:KG-2.49).

        Propose-only: materializes enterprise-standard interfaces, scores per-asset
        conformance drift, and emits ranked consolidation recommendations. No source
        asset is mutated and nothing auto-merges.
        """
        from ..standardization import run_standardization_pass

        return run_standardization_pass(self.engine)

    def _run_breadth(self) -> dict[str, Any]:
        """Ingest the OSS/repos/docs corpus from env-configured roots (idempotent).

        ``KG_BREADTH_LIBRARY_ROOTS`` / ``KG_BREADTH_REPO_ROOTS`` are comma-separated
        paths. No roots ⇒ no-op. Content-addressed ingest means re-runs are cheap.
        """
        from dataclasses import asdict

        from agent_utilities.core.config import AgentConfig

        from ..assimilation import run_breadth_ingest

        # Roots come from the AgentConfig fields (populated from env/.env) — so a
        # deployment configures breadth auto-ingest once and ``golden_loop`` (one
        # call / the daemon) ingests it. Read a fresh AgentConfig() (not the
        # import-time singleton) so runtime root changes are honored. (CONCEPT:KG-2.7)
        _cfg = AgentConfig()
        libs_raw = _cfg.kg_breadth_library_roots
        repos_raw = _cfg.kg_breadth_repo_roots
        libs = [p.strip() for p in (libs_raw or "").split(",") if p.strip()]
        repos = [p.strip() for p in (repos_raw or "").split(",") if p.strip()]
        if not libs and not repos:
            return {"skipped": True, "reason": "no roots configured"}
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
        # ``errors``/``stage_ms`` are JSON-encoded: the durable (Postgres) backend
        # cannot adapt a raw dict/list into a column value.
        import json

        try:
            self.engine.add_node(
                f"evolution_cycle:{uuid.uuid4().hex[:10]}",
                "orchestration_cycle",
                properties={
                    "duration_ms": m["duration_ms"],
                    "error_count": m["error_count"],
                    "errors": json.dumps(report["errors"][:10]),
                    "topics_intake": report["topics_intake"],
                    "open_gaps": m["open_gaps"],
                    "stage_ms": json.dumps(m["stage_ms"]),
                },
            )
        except Exception as e:  # noqa: BLE001 - monitoring persist is best-effort
            logger.debug("EvolutionCycle persist failed: %s", e)

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
        return write_spec_drafts(specs, self.codebase_root)

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

        # GOVERNED auto-merge (CONCEPT:AHE-3.14): consider promoting the team
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


def run_golden_loop_cycle(engine: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience entry: run one cycle against the active (or given) engine."""
    if engine is None:
        from ..core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
    return GoldenLoopController(engine).run_one_cycle(**kwargs)


def run_assimilation_pass(
    engine: Any = None,
    *,
    synthesize: bool = False,
    top_n: int = 5,
    force: bool = False,
    synth_fn: Any = None,
) -> dict[str, Any]:
    """Run only the graph-compute assimilation middle (CONCEPT:KG-2.7).

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
    rep = GoldenLoopController(engine)._run_assimilate(force=force)
    # Synthesis is idempotent — plans upsert by ``plan_id`` — so run it whenever a
    # caller asks for it, even if the rank pass was skipped as "unchanged".
    # Previously synthesis was gated behind the rank watermark, so a prior bare
    # ``assimilate()`` bumped the watermark and silently suppressed a follow-up
    # ``synthesize`` (the only reason ``force`` was ever needed). The watermark's
    # job is to avoid redundant *re-ranking*, not to block an explicit synthesis
    # request. (CONCEPT:KG-2.7)
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
