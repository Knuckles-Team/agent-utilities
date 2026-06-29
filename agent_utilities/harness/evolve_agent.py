from __future__ import annotations

"""AHE Evolve Agent.

CONCEPT:AHE-3.0 — Agentic Harness Engineering (Evolution Loop)
CONCEPT:ORCH-1.8 — Workflow Distillation Integration

The Evolve Agent reads distilled evidence, identifies component-level
failure attribution, proposes targeted edits, and records falsifiable
predictions in a ChangeManifest.

This agent operates in two modes:
    - **Lightweight (in-graph)**: Runs as a specialist node in the HSM
      during normal operation, proposing incremental improvements.
    - **Full (background)**: Spawned as an async background task from
      the agent server for comprehensive evolution rounds.

Both modes spawn from the single agent server — no external deployment
dependencies.

Architecture:
    Reads: EvidenceCorpus (distilled traces, NOT raw logs)
    Writes: ComponentEdits + ChangeManifest
    Uses: KG for epistemic state, files for normative state, git for causal boundary
"""


import logging
import subprocess
import time
from typing import Any

from .component_registry import HarnessComponentRegistry
from .evidence_corpus import EvidenceCorpus, FailureCluster
from .manifest import ChangeManifest, ComponentEdit, ComponentType

logger = logging.getLogger(__name__)


class EvolveAgent:
    """AHE Evolve Agent — proposes and applies harness improvements.

    The Evolve Agent closes the AHE feedback loop by:
        1. Reading the layered EvidenceCorpus (progressive disclosure)
        2. Identifying component-level failure attribution
        3. Proposing targeted edits to harness components
        4. Recording falsifiable predictions for each edit
        5. Applying edits to workspace files
        6. Creating git commits with structured messages
        7. Returning a new ChangeManifest

    Args:
        workspace_path: Root of the workspace for file operations.
        registry: The harness component registry.
        knowledge_engine: Optional KG engine for epistemic queries.
    """

    def __init__(
        self,
        workspace_path: str,
        registry: HarnessComponentRegistry | None = None,
        knowledge_engine: Any = None,
        dspy_optimizer_type: str = "BootstrapFewShot",
        feedback_service: Any = None,
    ) -> None:
        self.workspace_path = workspace_path
        self.registry = registry or HarnessComponentRegistry(workspace_path)
        self.knowledge_engine = knowledge_engine
        self.dspy_optimizer_type = dspy_optimizer_type
        self._feedback_service = feedback_service

    async def evolve(
        self,
        evidence: EvidenceCorpus,
        current_manifest: ChangeManifest | None = None,
    ) -> ChangeManifest:
        """Run one evolution round.

        When the evidence corpus is too large for a single LLM context
        window, automatically delegates to RLM for deep analysis
        (CONCEPT:ORCH-1.1 × CONCEPT:ORCH-1.1).

        Args:
            evidence: The distilled evidence corpus for this round.
            current_manifest: The previous round's manifest for context.

        Returns:
            A new ChangeManifest with proposed edits and predictions.
        """
        logger.info(
            f"EvolveAgent: Starting evolution round. "
            f"Evidence: {evidence.total_tasks} tasks, "
            f"pass rate: {evidence.pass_rate:.1%}"
        )

        manifest = ChangeManifest(
            baseline_score=evidence.benchmark_score,
            parent_round_id=current_manifest.round_id if current_manifest else None,
        )

        # 1. Read Layer 1 (overview) — always
        logger.info("EvolveAgent: Analyzing overview evidence...")
        self._analyze_overview(evidence)

        # 2. Deep analysis via RLM if evidence is large
        rlm_edits = await self._deep_analyze_evidence(evidence)
        for edit in rlm_edits:
            manifest.add_edit(edit)

        # 3. Drill into failure clusters — selective (skip if RLM already handled)
        if not rlm_edits:
            top_clusters = evidence.get_top_failure_clusters(n=5)
            logger.info(
                f"EvolveAgent: Analyzing {len(top_clusters)} failure clusters..."
            )
            for cluster in top_clusters:
                # CONCEPT:AHE-3.1 - Attempt DSPy mathematical optimization first
                edits = await self._dspy_optimize_cluster(cluster, evidence)

                # Fallback to LLM heuristic edits if DSPy isn't applicable
                if not edits:
                    edits = await self._propose_edits_for_cluster(cluster, evidence)

                for edit in edits:
                    manifest.add_edit(edit)

        # 4. Record predictions
        total_predicted_fixes = len(manifest.get_all_predicted_fixes())
        if total_predicted_fixes > 0:
            predicted_improvement = min(0.1 * total_predicted_fixes, 0.5)  # Cap at +50%
            manifest.predicted_score = evidence.benchmark_score + predicted_improvement

        logger.info(
            f"EvolveAgent: Proposed {len(manifest.edits)} edits, "
            f"predicted fixes: {total_predicted_fixes}, "
            f"predicted score: {manifest.predicted_score}"
        )

        return manifest

    async def _deep_analyze_evidence(
        self, evidence: EvidenceCorpus
    ) -> list[ComponentEdit]:
        """RLM-powered deep analysis for large evidence corpora.

        CONCEPT:ORCH-1.1 × CONCEPT:ORCH-1.1 — RLM for AHE Evolution Loop

        When the serialized evidence exceeds the context threshold,
        delegates to an RLM sub-agent that programmatically loops
        over all entries, cross-references with KG data, and produces
        a prioritized list of edit proposals.

        Args:
            evidence: The full evidence corpus for this round.

        Returns:
            List of proposed ComponentEdits (empty if RLM not triggered).
        """
        from ..rlm.config import RLMConfig

        rlm_config = RLMConfig()
        evidence_json = evidence.model_dump_json()

        if not rlm_config.should_trigger(output_size=len(evidence_json)):
            return []

        logger.info(
            f"EvolveAgent: Evidence corpus ({len(evidence_json):,} chars) "
            f"exceeds RLM threshold. Using RLM for deep analysis."
        )

        from ..rlm.repl import RLMEnvironment

        env = RLMEnvironment(
            context=evidence_json,
            config=rlm_config,
            graph_deps=None,  # KG access via helpers if available
        )

        try:
            rlm_result = await env.run_full_rlm(
                "Analyze the EvidenceCorpus in `context` (JSON). "
                "Identify the top failure patterns and their component attributions. "
                "For each, propose a ComponentEdit with: "
                "'component_type' (one of: system_prompt, tool_description, "
                "tool_implementation, middleware, skill, orchestrator_skill, "
                "worker_skill, sub_agent, long_term_memory), "
                "'file_path' (estimated target file), "
                "'edit_summary' (what to change), "
                "'predicted_fixes' (list of task_ids expected to fix), "
                "'predicted_regressions' (list of task_ids that might regress). "
                "Output a JSON array of edit objects via FINAL_VAR('edits', json_string)."
            )

            import json

            edit_data = json.loads(rlm_result)
            edits = []
            for ed in edit_data:
                try:
                    comp_type = ComponentType(
                        ed.get("component_type", "tool_implementation")
                    )
                except (ValueError, KeyError):
                    comp_type = ComponentType.TOOL_IMPLEMENTATION
                edits.append(
                    ComponentEdit(
                        component_type=comp_type,
                        file_path=ed.get("file_path", "unknown"),
                        edit_summary=ed.get("edit_summary", ""),
                        predicted_fixes=ed.get("predicted_fixes", []),
                        predicted_regressions=ed.get("predicted_regressions", []),
                        evidence_references=[f"rlm_deep_analysis:{evidence.round_id}"],
                    )
                )
            logger.info(
                f"EvolveAgent: RLM deep analysis produced {len(edits)} edit proposals."
            )
            return edits
        except Exception as e:
            logger.warning(f"EvolveAgent: RLM deep analysis failed: {e}")
            return []

    def _analyze_overview(self, evidence: EvidenceCorpus) -> None:
        """Analyze the overview layer for high-level patterns."""
        if evidence.pass_rate > 0.9:
            logger.info("EvolveAgent: High pass rate — focusing on edge-case failures.")
        elif evidence.pass_rate < 0.5:
            logger.warning(
                "EvolveAgent: Low pass rate — major systemic issues detected."
            )

    async def _propose_edits_for_cluster(
        self,
        cluster: FailureCluster,
        evidence: EvidenceCorpus,
    ) -> list[ComponentEdit]:
        """Propose targeted edits to address a failure cluster.

        Examines the cluster's component attribution and proposes
        specific file-level edits with predictions.

        Args:
            cluster: The failure cluster to address.
            evidence: The full evidence corpus for context.

        Returns:
            List of proposed ComponentEdits.
        """
        edits: list[ComponentEdit] = []

        if not cluster.component_attribution:
            logger.info(
                f"EvolveAgent: Cluster '{cluster.label}' has no component "
                f"attribution — skipping."
            )
            return edits

        # Find registered files for this component type
        registered = self.registry.get_components_by_type(cluster.component_attribution)

        if not registered:
            logger.info(
                f"EvolveAgent: No registered files for component type "
                f"'{cluster.component_attribution.value}' — skipping."
            )
            return edits

        # Propose an edit for the most relevant file
        target_file = registered[0].file_path
        edit = ComponentEdit(
            component_type=cluster.component_attribution,
            file_path=target_file,
            edit_summary=(
                f"Address '{cluster.label}' failure cluster "
                f"({cluster.frequency} tasks affected). "
                f"Root cause: {cluster.root_cause_summary[:200]}"
            ),
            predicted_fixes=cluster.task_ids[:10],  # Cap predictions
            predicted_regressions=[],  # Conservative — no predicted regressions
            evidence_references=[f"cluster:{cluster.cluster_id}"],
        )
        edits.append(edit)

        return edits

    def _build_trainset(
        self, evidence: EvidenceCorpus, cluster: FailureCluster
    ) -> list[Any]:
        """Passing traces in this cluster as DSPy demonstrations (CONCEPT:AHE-3.40)."""
        try:
            import dspy
        except ImportError:
            return []
        trainset: list[Any] = []
        for t in getattr(evidence, "traces", []) or []:
            if t.task_id in cluster.task_ids and getattr(t, "passed", False):
                trainset.append(
                    dspy.Example(
                        context=getattr(t, "context", ""),
                        task=getattr(t, "query", ""),
                        response=getattr(t, "output", ""),
                    ).with_inputs("context", "task")
                )
        return trainset

    def _kg_bridge(self) -> Any | None:
        """Lazily build the DSPyKGBridge from the knowledge engine (closes the prior
        Wire-First gap: the bridge existed but nothing called it)."""
        if self.knowledge_engine is None:
            return None
        try:
            from agent_utilities.knowledge_graph.dspy_kg_bridge import DSPyKGBridge

            return DSPyKGBridge(self.knowledge_engine, self.workspace_path)
        except Exception:  # noqa: BLE001
            return None

    async def _dspy_optimize_cluster(
        self,
        cluster: FailureCluster,
        evidence: EvidenceCorpus,
    ) -> list[ComponentEdit]:
        """DSPy-optimize a failing component via the optimizable-target registry.

        CONCEPT:AHE-3.1 / AHE-3.40 — generalized from the original system-prompt-only,
        exact-match path. Dispatches on ``cluster.component_attribution`` to a registered
        :class:`OptimizableTarget` (system prompt, MCP tool description, agent skill),
        builds a trainset of passing traces, compiles + demo-refines under the **real**
        graded metric (AHE-3.39/3.43), writes a system-prompt blueprint's compiled state
        back to disk, and persists the optimization to the KG for *every* target. Returns
        no edits for unregistered types, so the LLM-heuristic fallback still handles them.
        """
        import json
        import os

        from .dspy_optimization import get_target, run_dspy_optimization

        edits: list[ComponentEdit] = []
        attribution = cluster.component_attribution
        if attribution is None:
            return edits
        target = get_target(attribution)
        if target is None:
            return edits

        registered = self.registry.get_components_by_type(attribution)
        if not registered:
            return edits

        reg = registered[0]
        target_file = reg.file_path
        full_path = os.path.join(self.workspace_path, target_file)

        # Load the artifact: a JSON blueprint for system prompts, otherwise the
        # registration's text (description) — the target handler knows which key to read.
        artifact: dict[str, Any] = {}
        is_json = target_file.endswith(".json") and os.path.exists(full_path)
        if is_json:
            with open(full_path, encoding="utf-8") as f:
                artifact = json.load(f)
        else:
            text = reg.description or ""
            if not text and os.path.exists(full_path):
                try:
                    text = open(full_path, encoding="utf-8").read()[:4000]
                except OSError:
                    text = ""
            stem = os.path.splitext(os.path.basename(target_file))[0]
            artifact = {
                "name": stem,
                "description": text,
                "sop": text,
                "docstring": text,
            }
        artifact["__file_path__"] = target_file

        trainset = self._build_trainset(evidence, cluster)
        if not trainset:
            logger.info(
                "EvolveAgent: no passing traces for cluster %s to bootstrap DSPy.",
                cluster.label,
            )
            return edits

        result = run_dspy_optimization(
            target,
            artifact,
            trainset,
            optimizer_name=getattr(self, "dspy_optimizer_type", "BootstrapFewShot"),
        )
        if result is None:
            return edits

        # System-prompt blueprints carry the compiled state on disk (existing behavior).
        if target.component_type == "system_prompt" and is_json:
            artifact.pop("__file_path__", None)
            artifact["dspy_compiled_state"] = result.compiled_state
            artifact["few_shot_examples"] = result.demos
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=4)

        # Persist the optimization for EVERY target (was dead code before).
        bridge = self._kg_bridge()
        if bridge is not None:
            try:
                await bridge.ingest_evolved_component(
                    kg_label=target.kg_label,
                    component_type=target.component_type,
                    identifier=target.task_name(artifact),
                    file_path=target_file,
                    compiled_state=result.compiled_state,
                    version=str(artifact.get("version", "unknown")),
                    optimizer=result.optimizer,
                    demos=result.demos,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("EvolveAgent: KG persist failed: %s", e)

        edits.append(
            ComponentEdit(
                component_type=attribution,
                file_path=target_file,
                edit_summary=(
                    f"DSPy {result.optimizer} optimization "
                    f"({result.trainset_size} traces, {len(result.demos)} demos kept)."
                ),
                predicted_fixes=cluster.task_ids[:10],
                predicted_regressions=[],
                evidence_references=[f"dspy_optimization:{cluster.cluster_id}"],
            )
        )
        return edits

    async def apply_edits(
        self,
        manifest: ChangeManifest,
        dry_run: bool = True,
        auto_apply: bool | None = None,
    ) -> ChangeManifest:
        """Apply manifest edits to the workspace (CONCEPT:AHE-3.71).

        For a **system-prompt** edit carrying a DSPy-hardened candidate body
        (``edit.metadata['candidate_blueprint']``, produced by
        :meth:`harden_agent_prompt`) this is no longer a placeholder: the candidate is
        written to its ``StructuredPrompt`` file via ``.save()`` and committed — but ONLY
        when (a) it beat baseline (``edit.metadata['promote']``) and (b) the
        ``KG_AGENT_AUTO_APPLY`` gate is on. Otherwise the cycle is **propose-only**: a
        queryable :class:`ProposedPromptChange` audit record is written (before/after metric
        + decision + the rejected/held candidate) and the live prompt is left untouched. A
        prompt rewrite is high-impact, so it is never silent.

        Args:
            manifest: The manifest with edits to apply.
            dry_run: If True, never write source (forces propose-only).
            auto_apply: Override the ``KG_AGENT_AUTO_APPLY`` gate (read from config when
                ``None``). Pass explicitly to make tests deterministic.

        Returns:
            The manifest with git commit SHAs / audit records populated.
        """
        if auto_apply is None:
            auto_apply = self._auto_apply_enabled()

        for edit in manifest.edits:
            if edit.component_type == ComponentType.SYSTEM_PROMPT and edit.metadata.get(
                "candidate_blueprint"
            ):
                self._apply_prompt_edit(
                    edit, manifest, auto_apply=auto_apply, dry_run=dry_run
                )
                continue

            if dry_run:
                logger.info(
                    f"EvolveAgent [DRY RUN]: Would edit {edit.file_path} — "
                    f"{edit.edit_summary}"
                )
                continue

            commit_sha = self._git_commit_edit(edit)
            edit.git_commit_sha = commit_sha
            self.registry.record_edit(edit.file_path, edit.id)

        return manifest

    @staticmethod
    def _auto_apply_enabled() -> bool:
        """Read the canonical ``KG_AGENT_AUTO_APPLY`` gate (default OFF / shadow)."""
        try:
            from agent_utilities.core.config import config as _cfg

            return bool(_cfg.kg_agent_auto_apply)
        except Exception:  # noqa: BLE001 - absent config ⇒ safest default (shadow)
            return False

    def _apply_prompt_edit(
        self,
        edit: ComponentEdit,
        manifest: ChangeManifest,
        *,
        auto_apply: bool,
        dry_run: bool,
    ) -> None:
        """Gated write + audit for a hardened system-prompt candidate (CONCEPT:AHE-3.71)."""
        import os

        from agent_utilities.prompting.structured import StructuredPrompt

        meta = edit.metadata
        promote = bool(meta.get("promote", False))
        before = float(meta.get("baseline_score", 0.0))
        after = float(meta.get("candidate_score", 0.0))

        if promote and auto_apply and not dry_run:
            try:
                candidate = StructuredPrompt.model_validate(meta["candidate_blueprint"])
                full_path = os.path.join(self.workspace_path, edit.file_path)
                candidate.save(full_path)
                sha = self._git_commit_edit(edit)
                edit.git_commit_sha = sha
                self.registry.record_edit(edit.file_path, edit.id)
                status, applied = "applied", True
                logger.info(
                    "EvolveAgent: APPLIED hardened prompt %s (%.3f → %.3f) commit=%s",
                    edit.file_path,
                    before,
                    after,
                    sha,
                )
            except Exception as e:  # noqa: BLE001 - a write failure must not crash the loop
                logger.error("EvolveAgent: prompt apply failed: %s", e)
                status, applied = "error", False
        elif promote:
            status, applied = "proposed", False
            logger.info(
                "EvolveAgent: PROPOSED hardened prompt %s (%.3f → %.3f) — held for review "
                "(auto-apply gated off).",
                edit.file_path,
                before,
                after,
            )
        else:
            status, applied = "rejected", False
            logger.info(
                "EvolveAgent: REJECTED candidate for %s (%.3f → %.3f did not beat baseline).",
                edit.file_path,
                before,
                after,
            )

        edit.metadata["apply_status"] = status
        self._record_proposed_change(edit, manifest, status, before, after, applied)

    def _record_proposed_change(
        self,
        edit: ComponentEdit,
        manifest: ChangeManifest,
        status: str,
        before: float,
        after: float,
        applied: bool,
    ) -> str:
        """Persist a queryable + approvable ``ProposedPromptChange`` audit record.

        CONCEPT:AHE-3.71 — the transparency surface. Every hardening decision (applied /
        proposed / rejected) lands as a git-diffable JSON under
        ``.specify/proposals/`` AND, best-effort, a ``ProposedPromptChange`` KG node — so a
        human/Claude can review the before/after metric and the held candidate and approve
        it (:meth:`approve_proposed_change`) rather than have it land silently.
        """
        import json
        import os

        meta = edit.metadata
        proposal_id = f"prompt_change:{edit.id.split(':')[-1]}"
        record = {
            "id": proposal_id,
            "type": "ProposedPromptChange",
            "agent_id": meta.get("agent_id", ""),
            "file_path": edit.file_path,
            "round_id": manifest.round_id,
            "edit_id": edit.id,
            "status": status,
            "applied": applied,
            "baseline_score": round(before, 4),
            "candidate_score": round(after, 4),
            "delta": round(after - before, 4),
            "optimizer": meta.get("optimizer", ""),
            "trainset_size": meta.get("trainset_size", 0),
            "candidate_version_hash": meta.get("candidate_version_hash", ""),
            "candidate_blueprint": meta.get("candidate_blueprint", {}),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        proposals_dir = os.path.join(self.workspace_path, ".specify", "proposals")
        os.makedirs(proposals_dir, exist_ok=True)
        proposal_path = os.path.join(proposals_dir, f"{proposal_id}.json")
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if self.knowledge_engine is not None and hasattr(
            self.knowledge_engine, "add_node"
        ):
            props = {k: v for k, v in record.items() if k != "candidate_blueprint"}
            props["candidate_blueprint_json"] = json.dumps(
                record["candidate_blueprint"]
            )[:8000]
            try:  # two add_node shapes in the fleet — try the kw form, then properties=.
                self.knowledge_engine.add_node(proposal_id, **props)
            except Exception:  # noqa: BLE001
                try:
                    self.knowledge_engine.add_node(
                        proposal_id, "ProposedPromptChange", properties=props
                    )
                except Exception as e:  # noqa: BLE001 - persistence best-effort
                    logger.debug("ProposedPromptChange KG persist failed: %s", e)

        meta["proposal_id"] = proposal_id
        meta["proposal_path"] = proposal_path
        return proposal_path

    def approve_proposed_change(self, proposal_id: str) -> dict[str, Any]:
        """Human/Claude approval path for a shadow proposal (CONCEPT:AHE-3.71).

        Applies a previously **proposed** (or rejected, if force-approved) candidate to
        source — the steerable counterpart to the auto-apply gate, so a winning prompt can
        go live by review instead of by flipping the global flag. Returns a status dict.
        """
        import json
        import os

        from agent_utilities.prompting.structured import StructuredPrompt

        proposal_path = os.path.join(
            self.workspace_path, ".specify", "proposals", f"{proposal_id}.json"
        )
        if not os.path.exists(proposal_path):
            return {"approved": False, "error": f"no proposal {proposal_id}"}
        with open(proposal_path, encoding="utf-8") as f:
            record = json.load(f)
        blueprint = record.get("candidate_blueprint") or {}
        if not blueprint:
            return {"approved": False, "error": "proposal carries no candidate"}
        candidate = StructuredPrompt.model_validate(blueprint)
        full_path = os.path.join(self.workspace_path, record["file_path"])
        candidate.save(full_path)
        record["status"] = "applied"
        record["applied"] = True
        record["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("EvolveAgent: APPROVED + applied proposal %s", proposal_id)
        return {"approved": True, "file_path": record["file_path"], "id": proposal_id}

    async def harden_agent_prompt(
        self,
        agent_id: str,
        prompt_path: str,
        *,
        feedback_service: Any = None,
        min_delta: float = 0.0,
        auto_apply: bool | None = None,
    ) -> Any:
        """Run ONE metric → optimize → evaluate → (gated) apply cycle for an agent's prompt.

        CONCEPT:AHE-3.73 — the closed agent-hardening cycle (uses the AHE-3.71 gated apply
        and AHE-3.72 per-agent attribution), end-to-end for one agent:

        1. **Attribute** — pool the agent's ``action_outcome`` cases into a per-agent
           trainset + eval slice (:meth:`FeedbackService.build_agent_trainset`).
        2. **Optimize** — run :func:`run_dspy_optimization` on the ``system_prompt`` target
           with that trainset; degrade to the labeled successes as demos when no LM is
           reachable to compile.
        3. **Build** — fold the optimized demos into a candidate ``StructuredPrompt``.
        4. **Evaluate** — score baseline vs candidate against the agent's eval slice.
        5. **Decide + apply** — :func:`should_promote`, then :meth:`apply_edits` writes the
           winner ONLY under ``KG_AGENT_AUTO_APPLY``; otherwise it is held as a queryable
           proposal. Always leaves an audit trail (ProposedPromptChange + the manifest).

        Returns a :class:`PromptHardeningOutcome`.
        """
        import os

        from agent_utilities.prompting.structured import StructuredPrompt

        from .dspy_optimization import (
            PromptHardeningOutcome,
            build_hardened_prompt,
            get_target,
            run_dspy_optimization,
            score_prompt_against_corpus,
            should_promote,
        )

        fb = feedback_service or self._feedback_service
        if fb is None:
            return PromptHardeningOutcome(
                agent_id=agent_id,
                prompt_path=prompt_path,
                status="no_data",
                detail="no FeedbackService available to pool per-agent outcomes",
            )

        cases = fb.agent_eval_cases(agent_id)
        trainset = fb.build_agent_trainset(agent_id)
        if not cases or not trainset:
            return PromptHardeningOutcome(
                agent_id=agent_id,
                prompt_path=prompt_path,
                trainset_size=len(trainset),
                status="no_data",
                detail=f"per-agent corpus empty (cases={len(cases)} train={len(trainset)})",
            )

        full_path = os.path.join(self.workspace_path, prompt_path)
        baseline = StructuredPrompt.load(full_path)

        # Optimize — best-effort DSPy compile, then fall back to the labeled successes as
        # demos so the cycle still hardens the prompt offline (no LM required).
        target = get_target("system_prompt")
        result = None
        if target is not None:
            artifact = baseline.model_dump(exclude_none=True)
            artifact["__file_path__"] = prompt_path
            result = run_dspy_optimization(
                target,
                artifact,
                list(trainset),
                optimizer_name=self.dspy_optimizer_type,
            )
        # Use the DSPy-bootstrapped demos when the compile produced any; otherwise (no LM
        # reachable to roll out a bootstrap) fall back to the agent's labeled successes as
        # demos, so the cycle still hardens the prompt offline.
        if result is not None and result.demos:
            demos = result.demos
            optimizer_name = result.optimizer
            optimized_instruction = result.optimized_instruction
        else:
            demos = list(trainset)
            optimizer_name = "labeled-successes"
            optimized_instruction = ""

        candidate = build_hardened_prompt(
            baseline, demos, optimized_instruction=optimized_instruction
        )

        baseline_score = score_prompt_against_corpus(baseline.render(), cases)
        candidate_score = score_prompt_against_corpus(candidate.render(), cases)
        promote = should_promote(baseline_score, candidate_score, min_delta=min_delta)

        edit = ComponentEdit(
            component_type=ComponentType.SYSTEM_PROMPT,
            file_path=prompt_path,
            edit_summary=(
                f"Harden {agent_id} system prompt via {optimizer_name} "
                f"({len(trainset)} outcomes): {baseline_score:.3f} → {candidate_score:.3f}"
            ),
            evidence_references=[f"agent_outcomes:{agent_id}"],
            metadata={
                "agent_id": agent_id,
                "promote": promote,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "optimizer": optimizer_name,
                "trainset_size": len(trainset),
                "candidate_version_hash": candidate.version_hash(),
                "candidate_blueprint": candidate.model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            },
        )
        manifest = ChangeManifest(baseline_score=baseline_score)
        manifest.add_edit(edit)

        await self.apply_edits(manifest, dry_run=False, auto_apply=auto_apply)

        status = edit.metadata.get("apply_status", "rejected")
        return PromptHardeningOutcome(
            agent_id=agent_id,
            prompt_path=prompt_path,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            promote=promote,
            applied=status == "applied",
            status=status,
            trainset_size=len(trainset),
            optimizer=optimizer_name,
            candidate_version_hash=candidate.version_hash(),
            detail=edit.metadata.get("proposal_path", ""),
        )

    def _git_commit_edit(self, edit: ComponentEdit) -> str | None:
        """Create a git commit for a component edit.

        Uses a structured commit message format for AHE traceability:
        ``ahe(<component_type>): <edit_summary>``
        """
        try:
            # Stage the file
            subprocess.run(  # nosec B603 B607
                ["git", "add", edit.file_path],
                cwd=self.workspace_path,
                capture_output=True,
                timeout=10,
            )

            # Commit with structured message
            commit_msg = (
                f"ahe({edit.component_type.value}): {edit.edit_summary[:72]}\n\n"
                f"Predicted fixes: {', '.join(edit.predicted_fixes[:5])}\n"
                f"Predicted regressions: {', '.join(edit.predicted_regressions[:5])}\n"
                f"Evidence: {', '.join(edit.evidence_references[:3])}"
            )
            result = subprocess.run(  # nosec B603 B607
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Get the commit SHA
                sha_result = subprocess.run(  # nosec B603 B607
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return sha_result.stdout.strip()
            else:
                logger.warning(f"Git commit failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Git commit exception: {e}")
            return None

    async def persist_manifest(self, manifest: ChangeManifest) -> str:
        """Persist a manifest to both .specify/ and the Knowledge Graph.

        Dual storage: files for git-diffable normative state,
        KG for epistemic queries.

        Returns:
            The file path where the manifest was saved.
        """
        # 1. Save to .specify/manifests/
        file_path = manifest.to_sdd_path(self.workspace_path)
        with open(file_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        logger.info(f"EvolveAgent: Manifest saved to {file_path}")

        # 2. Save to Knowledge Graph
        if self.knowledge_engine:
            try:
                self.knowledge_engine.add_memory(
                    content=manifest.model_dump_json(),
                    name=f"ChangeManifest {manifest.round_id}",
                    category="ahe_manifest",
                    tags=["ahe", "manifest", manifest.round_id],
                )
                logger.info(
                    f"EvolveAgent: Manifest saved to Knowledge Graph "
                    f"(round: {manifest.round_id})"
                )
            except Exception as e:
                logger.warning(f"EvolveAgent: KG persistence failed: {e}")

        return file_path
