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
    ) -> None:
        self.workspace_path = workspace_path
        self.registry = registry or HarnessComponentRegistry(workspace_path)
        self.knowledge_engine = knowledge_engine
        self.dspy_optimizer_type = dspy_optimizer_type

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
    ) -> ChangeManifest:
        """Apply manifest edits to the workspace.

        In dry_run mode, only logs what would be changed without
        modifying files.

        Args:
            manifest: The manifest with edits to apply.
            dry_run: If True, don't modify files.

        Returns:
            The manifest with git commit SHAs populated.
        """
        for edit in manifest.edits:
            if dry_run:
                logger.info(
                    f"EvolveAgent [DRY RUN]: Would edit {edit.file_path} — "
                    f"{edit.edit_summary}"
                )
                continue

            # In real evolution, this is where the LLM generates the actual
            # code changes. For now, we commit with a structured message.
            commit_sha = self._git_commit_edit(edit)
            edit.git_commit_sha = commit_sha

            # Record in registry
            self.registry.record_edit(edit.file_path, edit.id)

        return manifest

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
