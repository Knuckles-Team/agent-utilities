"""AHE Evolve Agent.

CONCEPT:AU-AHE.harness.harness-evolution — Agentic Harness Engineering (Evolution Loop)
CONCEPT:AU-ORCH.optimization.workflow-distillation — Workflow Distillation Integration

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

from __future__ import annotations

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
        feedback_service: Any = None,
    ) -> None:
        self.workspace_path = workspace_path
        self.registry = registry or HarnessComponentRegistry(workspace_path)
        self.knowledge_engine = knowledge_engine
        self._feedback_service = feedback_service
        self._proposal_targets: dict[str, str] = {}

    def _resolve_component_path(self, relative_path: str) -> str:
        """Resolve one registered relative component without workspace escape."""
        import os

        path = str(relative_path or "")
        if (
            not path
            or "\x00" in path
            or os.path.isabs(path)
            or (os.name != "nt" and "\\" in path)
        ):
            raise ValueError("component path is invalid")
        root = os.path.realpath(self.workspace_path)
        candidate = os.path.realpath(os.path.join(root, path))
        try:
            contained = os.path.commonpath((root, candidate)) == root
        except ValueError:
            contained = False
        if not contained:
            raise ValueError("component path escapes workspace")
        return candidate

    async def evolve(
        self,
        evidence: EvidenceCorpus,
        current_manifest: ChangeManifest | None = None,
    ) -> ChangeManifest:
        """Run one evolution round.

        When the evidence corpus is too large for a single LLM context
        window, automatically delegates to RLM for deep analysis
        (CONCEPT:AU-ORCH.scheduling.rlm-evolution-loop × CONCEPT:AU-ORCH.scheduling.rlm-evolution-loop).

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
                edits = await self._optimize_cluster(cluster, evidence)

                # Unregistered components remain on the general edit-proposal path.
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

        CONCEPT:AU-ORCH.scheduling.rlm-evolution-loop × CONCEPT:AU-ORCH.scheduling.rlm-evolution-loop — RLM for AHE Evolution Loop

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
                "'predicted_fixes' (list of task_ids expected to fix), "
                "'predicted_regressions' (list of task_ids that might regress). "
                "Do not emit paths, source content, prompts, examples, identities, or "
                "credentials. "
                "Output a JSON array of edit objects via FINAL_VAR('edits', json_string)."
            )

            import json

            from .optimization_backend import opaque_program_reference

            if not isinstance(rlm_result, str) or len(rlm_result) > 256 * 1024:
                raise ValueError("RLM edit response is invalid")
            edit_data = json.loads(rlm_result)
            if not isinstance(edit_data, list):
                raise ValueError("RLM edit response is invalid")
            allowed_tasks = {
                str(getattr(trace, "task_id", "") or "")
                for trace in (getattr(evidence, "traces", []) or [])
                if getattr(trace, "task_id", "")
            }

            def governed_task_refs(row: dict[str, Any], field: str) -> list[str]:
                values = row.get(field)
                if not isinstance(values, list):
                    return []
                return [
                    opaque_program_reference("task", task_id)
                    for task_id in dict.fromkeys(str(value) for value in values)
                    if task_id in allowed_tasks
                ][:10]

            edits: list[ComponentEdit] = []
            for ed in edit_data[:16]:
                if not isinstance(ed, dict):
                    continue
                try:
                    comp_type = ComponentType(
                        ed.get("component_type", "tool_implementation")
                    )
                except (TypeError, ValueError):
                    continue
                registered = self.registry.get_components_by_type(comp_type)
                if not registered:
                    continue

                target_path = registered[0].file_path
                edits.append(
                    ComponentEdit(
                        component_type=comp_type,
                        file_path=target_path,
                        edit_summary=(
                            f"Governed deep analysis proposed a {comp_type.value} change."
                        ),
                        predicted_fixes=governed_task_refs(ed, "predicted_fixes"),
                        predicted_regressions=governed_task_refs(
                            ed,
                            "predicted_regressions"
                        ),
                        evidence_references=[
                            opaque_program_reference("evidence", evidence.round_id)
                        ],
                        metadata={
                            "component_ref": opaque_program_reference(
                                "component", target_path
                            )
                        },
                    )
                )
            logger.info("EvolveAgent: deep analysis produced %d proposals", len(edits))
            return edits
        except Exception as exc:
            logger.warning(
                "EvolveAgent: RLM deep analysis failed (%s)", type(exc).__name__
            )
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

        from .optimization_backend import opaque_program_reference

        if not cluster.component_attribution:
            logger.info("EvolveAgent: unattributed cluster skipped")
            return edits

        # Find registered files for this component type
        registered = self.registry.get_components_by_type(cluster.component_attribution)

        if not registered:
            logger.info(
                "EvolveAgent: no registered component for type %s",
                cluster.component_attribution.value,
            )
            return edits

        # Propose an edit for the most relevant file
        target_file = registered[0].file_path
        edit = ComponentEdit(
            component_type=cluster.component_attribution,
            file_path=target_file,
            edit_summary=(
                f"Address a governed failure cluster ({cluster.frequency} tasks) "
                f"for {cluster.component_attribution.value}."
            ),
            predicted_fixes=[
                opaque_program_reference("task", task_id)
                for task_id in cluster.task_ids[:10]
            ],
            predicted_regressions=[],  # Conservative — no predicted regressions
            evidence_references=[
                opaque_program_reference("evidence", cluster.cluster_id)
            ],
            metadata={
                "component_ref": opaque_program_reference("component", target_file)
            },
        )
        edits.append(edit)

        return edits

    def _build_trainset(
        self, evidence: EvidenceCorpus, cluster: FailureCluster
    ) -> list[Any]:
        """Return passing traces in this cluster as provider-neutral examples."""
        trainset: list[Any] = []
        for t in getattr(evidence, "traces", []) or []:
            if t.task_id in cluster.task_ids and getattr(t, "passed", False):
                trainset.append(
                    {
                        "context": getattr(t, "context", ""),
                        "task": getattr(t, "query", ""),
                        "response": getattr(t, "output", ""),
                    }
                )
        return trainset

    async def _optimize_cluster(
        self,
        cluster: FailureCluster,
        evidence: EvidenceCorpus,
    ) -> list[ComponentEdit]:
        """Optimize a failing component through the native target registry.

        Dispatches on ``cluster.component_attribution`` to a registered
        :class:`OptimizableTarget` (system prompt, MCP tool description, agent skill),
        builds a trainset of passing traces, submits the engine-owned job, and returns a
        reference-only proposal. It never writes the selected system-prompt candidate;
        the reviewed apply path owns that mutation. The native jobs plane persists the
        governed optimization trajectory.
        """
        import json
        import os

        from .optimization_backend import opaque_program_reference
        from .program_optimization import get_target, run_program_optimization

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
        try:
            full_path = self._resolve_component_path(target_file)
        except ValueError:
            logger.warning("EvolveAgent: registered component path is invalid")
            return edits

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
                "EvolveAgent: no passing traces for governed evidence %s",
                opaque_program_reference("evidence", cluster.cluster_id),
            )
            return edits

        result = run_program_optimization(
            target,
            artifact,
            trainset,
            engine=self.knowledge_engine,
        )
        if result is None:
            return edits

        candidate_version_hash = ""
        if target.component_type == "system_prompt" and is_json:
            candidate_version_hash = self._compiled_prompt_candidate(
                full_path, result.compiled_state
            ).version_hash()

        edits.append(
            ComponentEdit(
                component_type=attribution,
                file_path=target_file,
                edit_summary=(
                    f"Native {result.optimizer} optimization "
                    f"({result.trainset_size} traces, "
                    f"{len(result.demonstration_refs)} governed references kept)."
                ),
                predicted_fixes=[
                    opaque_program_reference("task", task_id)
                    for task_id in cluster.task_ids[:10]
                ],
                predicted_regressions=[],
                evidence_references=[
                    opaque_program_reference("evidence", cluster.cluster_id)
                ],
                metadata={
                    "agent_ref": opaque_program_reference(
                        "agent", target.task_name(artifact)
                    ),
                    "component_ref": opaque_program_reference(
                        "component", target_file
                    ),
                    "promote": True,
                    "auto_apply_eligible": False,
                    "baseline_score": 0.0,
                    "candidate_score": result.confidence,
                    "optimizer": result.optimizer,
                    "trainset_size": result.trainset_size,
                    "candidate_version_hash": candidate_version_hash,
                    "program_compiled_state": result.compiled_state,
                },
            )
        )
        return edits

    async def apply_edits(
        self,
        manifest: ChangeManifest,
        dry_run: bool = True,
        auto_apply: bool | None = None,
    ) -> ChangeManifest:
        """Apply manifest edits to the workspace (CONCEPT:AU-AHE.harness.hardening-transparency-surface).

        For a **system-prompt** edit carrying reference-only native compiled state,
        the state is attached to its ``StructuredPrompt`` blueprint and committed — but ONLY
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
                "program_compiled_state"
            ):
                self._apply_prompt_edit(
                    edit, manifest, auto_apply=auto_apply, dry_run=dry_run
                )
                continue

            if dry_run:
                from .optimization_backend import opaque_program_reference

                logger.info(
                    "EvolveAgent [DRY RUN]: would edit component %s",
                    opaque_program_reference("component", edit.file_path),
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

    @staticmethod
    def _compiled_prompt_candidate(full_path: str, state: dict[str, Any]) -> Any:
        """Attach validated reference-only program state to the current prompt."""
        from agent_utilities.prompting.structured import (
            ProgramCompiledState,
            StructuredPrompt,
        )

        from .program_optimization import _bump_patch

        candidate = StructuredPrompt.load(full_path).model_copy(deep=True)
        candidate.program_compiled_state = ProgramCompiledState.model_validate(state)
        candidate.prompt_version = _bump_patch(candidate.prompt_version)
        return candidate

    def _apply_prompt_edit(
        self,
        edit: ComponentEdit,
        manifest: ChangeManifest,
        *,
        auto_apply: bool,
        dry_run: bool,
    ) -> None:
        """Gated write + audit for a hardened system-prompt candidate (CONCEPT:AU-AHE.harness.hardening-transparency-surface)."""
        import math

        from .optimization_backend import (
            is_opaque_program_reference,
            opaque_program_reference,
        )

        meta = edit.metadata
        promote = meta.get("promote") is True
        auto_apply_eligible = meta.get("auto_apply_eligible", True) is True
        try:
            before = float(meta.get("baseline_score", 0.0))
            after = float(meta.get("candidate_score", 0.0))
        except (TypeError, ValueError):
            meta["apply_status"] = "error"
            logger.error("EvolveAgent: prompt scores are invalid")
            return
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or not 0.0 <= before <= 1.0
            or not 0.0 <= after <= 1.0
        ):
            meta["apply_status"] = "error"
            logger.error("EvolveAgent: prompt scores are invalid")
            return
        component_ref = str(meta.get("component_ref") or "")
        if not is_opaque_program_reference(component_ref, namespace="component"):
            component_ref = opaque_program_reference("component", edit.file_path)

        if promote and auto_apply_eligible and auto_apply and not dry_run:
            try:
                full_path = self._resolve_component_path(edit.file_path)
                candidate = self._compiled_prompt_candidate(
                    full_path, meta["program_compiled_state"]
                )
                candidate.save(full_path)
                sha = self._git_commit_edit(edit)
                edit.git_commit_sha = sha
                self.registry.record_edit(edit.file_path, edit.id)
                status, applied = "applied", True
                logger.info(
                    "EvolveAgent: APPLIED hardened prompt %s (%.3f → %.3f) commit=%s",
                    component_ref,
                    before,
                    after,
                    sha,
                )
            except Exception as exc:  # noqa: BLE001 - a write failure must not crash the loop
                logger.error("EvolveAgent: prompt apply failed (%s)", type(exc).__name__)
                status, applied = "error", False
        elif promote:
            status, applied = "proposed", False
            logger.info(
                "EvolveAgent: PROPOSED hardened prompt %s (%.3f → %.3f) — held for review "
                "(automatic apply not eligible or gated off).",
                component_ref,
                before,
                after,
            )
        else:
            status, applied = "rejected", False
            logger.info(
                "EvolveAgent: REJECTED candidate for %s (%.3f → %.3f did not beat baseline).",
                component_ref,
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

        CONCEPT:AU-AHE.harness.hardening-transparency-surface — the transparency surface. Every hardening decision (applied /
        proposed / rejected) lands as a reference-only JSON record under
        ``.specify/proposals/`` AND, best-effort, a ``ProposedPromptChange`` KG node — so a
        a reviewer can inspect the before/after metric and compiled references and approve
        them (:meth:`approve_proposed_change`) rather than have them land silently.
        """
        import json
        import math
        import os
        import re

        from agent_utilities.prompting.structured import ProgramCompiledState

        from .optimization_backend import (
            is_opaque_program_reference,
            opaque_program_reference,
        )

        meta = edit.metadata
        compiled_state = ProgramCompiledState.model_validate(
            meta.get("program_compiled_state") or {}
        ).model_dump()
        evidence_refs = list(dict.fromkeys(edit.evidence_references))
        if any(not is_opaque_program_reference(reference) for reference in evidence_refs):
            raise ValueError("proposal evidence references are invalid")
        if status not in {"applied", "proposed", "rejected", "error"}:
            raise ValueError("proposal status is invalid")
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or not 0.0 <= before <= 1.0
            or not 0.0 <= after <= 1.0
        ):
            raise ValueError("proposal scores must be bounded")
        proposal_id = opaque_program_reference("proposal", edit.id)
        agent_ref = str(meta.get("agent_ref") or "")
        if not is_opaque_program_reference(agent_ref, namespace="agent"):
            agent_ref = opaque_program_reference(
                "agent", agent_ref or compiled_state["program_ref"]
            )
        component_ref = str(meta.get("component_ref") or edit.file_path)
        if not is_opaque_program_reference(component_ref, namespace="component"):
            component_ref = opaque_program_reference("component", component_ref)
        version_hash = str(meta.get("candidate_version_hash") or "")
        if re.fullmatch(r"[0-9a-f]{16}", version_hash) is None:
            version_hash = compiled_state["id"].rsplit(":", 1)[-1][:16]
        try:
            trainset_size = min(max(int(meta.get("trainset_size", 0)), 0), 1_000_000)
        except (TypeError, ValueError):
            trainset_size = 0
        record = {
            "id": proposal_id,
            "node_type": "ProposedPromptChange",
            "agent_ref": agent_ref,
            "component_ref": component_ref,
            "round_ref": opaque_program_reference("round", manifest.round_id),
            "edit_ref": opaque_program_reference("edit", edit.id),
            "status": status,
            "applied": bool(applied),
            "baseline_score": round(before, 4),
            "candidate_score": round(after, 4),
            "delta": round(after - before, 4),
            "optimizer": compiled_state["optimizer"],
            "trainset_size": trainset_size,
            "candidate_version_hash": version_hash,
            "auto_apply_eligible": bool(meta.get("auto_apply_eligible", True)),
            "program_compiled_state": compiled_state,
            "evidence_refs": evidence_refs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        integrity_material = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        record["integrity_ref"] = opaque_program_reference(
            "proposal_integrity", integrity_material
        )

        proposals_dir = self._resolve_component_path(".specify/proposals")
        os.makedirs(proposals_dir, exist_ok=True)
        proposal_token = proposal_id.rsplit(":", 1)[-1]
        proposal_path = os.path.join(
            proposals_dir, f"prompt-proposal-{proposal_token}.json"
        )
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        self._proposal_targets[proposal_id] = edit.file_path

        if self.knowledge_engine is not None and hasattr(
            self.knowledge_engine, "add_node"
        ):
            props = {k: v for k, v in record.items() if k != "program_compiled_state"}
            props["program_compiled_state_json"] = json.dumps(
                record["program_compiled_state"], sort_keys=True
            )
            try:
                self.knowledge_engine.add_node(
                    proposal_id, "ProposedPromptChange", properties=props
                )
            except Exception as exc:  # noqa: BLE001 - persistence best-effort
                logger.debug(
                    "ProposedPromptChange KG persist failed (%s)",
                    type(exc).__name__,
                )

        meta["proposal_ref"] = proposal_id
        return proposal_id

    def approve_proposed_change(self, proposal_id: str) -> dict[str, Any]:
        """Reviewed approval path for a shadow proposal (CONCEPT:AU-AHE.harness.hardening-transparency-surface).

        Applies a previously **proposed** (or rejected, if force-approved) candidate to
        source — the steerable counterpart to the auto-apply gate, so a winning prompt can
        go live by review instead of by flipping the global flag. Returns a status dict.
        """
        import hmac
        import json
        import os

        from .optimization_backend import (
            is_opaque_program_reference,
            opaque_program_reference,
        )

        if not is_opaque_program_reference(proposal_id, namespace="proposal"):
            return {"approved": False, "error": "proposal_ref_invalid"}
        proposal_token = proposal_id.rsplit(":", 1)[-1]
        proposal_path = self._resolve_component_path(
            f".specify/proposals/prompt-proposal-{proposal_token}.json"
        )
        if not os.path.exists(proposal_path):
            return {"approved": False, "error": "proposal_not_found"}
        with open(proposal_path, encoding="utf-8") as f:
            record = json.load(f)
        if not isinstance(record, dict):
            return {"approved": False, "error": "proposal_record_invalid"}
        integrity_ref = str(record.pop("integrity_ref", "") or "")
        expected_integrity_ref = opaque_program_reference(
            "proposal_integrity",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        if not is_opaque_program_reference(
            integrity_ref, namespace="proposal_integrity"
        ) or not hmac.compare_digest(integrity_ref, expected_integrity_ref):
            return {"approved": False, "error": "proposal_integrity_mismatch"}
        record["integrity_ref"] = integrity_ref
        if record.get("id") != proposal_id:
            return {"approved": False, "error": "proposal_ref_mismatch"}
        if record.get("status") not in {"proposed", "rejected"} or record.get(
            "applied"
        ) is not False:
            return {"approved": False, "error": "proposal_not_approvable"}
        compiled_state = record.get("program_compiled_state") or {}
        if not compiled_state:
            return {"approved": False, "error": "compiled_state_missing"}
        target_file = self._proposal_targets.get(proposal_id)
        if target_file is None:
            component_ref = str(record.get("component_ref") or "")
            target_file = next(
                (
                    path
                    for path in self.registry.get_all_components()
                    if opaque_program_reference("component", path) == component_ref
                ),
                None,
            )
        if target_file is None:
            return {"approved": False, "error": "proposal_target_unavailable"}
        if opaque_program_reference("component", target_file) != record.get(
            "component_ref"
        ):
            return {"approved": False, "error": "proposal_target_mismatch"}
        try:
            full_path = self._resolve_component_path(target_file)
        except ValueError:
            return {"approved": False, "error": "proposal_target_invalid"}
        candidate = self._compiled_prompt_candidate(full_path, compiled_state)
        candidate.save(full_path)
        record["status"] = "applied"
        record["applied"] = True
        record["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.pop("integrity_ref", None)
        record["integrity_ref"] = opaque_program_reference(
            "proposal_integrity",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("EvolveAgent: APPROVED + applied proposal %s", proposal_id)
        return {
            "approved": True,
            "component_ref": record["component_ref"],
            "id": proposal_id,
        }

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

        CONCEPT:AU-AHE.harness.callers-feed-back-per — the closed agent-hardening cycle (uses the AHE-3.71 gated apply
        and AHE-3.72 per-agent attribution), end-to-end for one agent:

        1. **Attribute** — pool the agent's ``action_outcome`` cases into a per-agent
           trainset + eval slice (:meth:`FeedbackService.build_agent_trainset`).
        2. **Optimize** — run :func:`run_program_optimization` on the ``system_prompt`` target
           with that trainset; native-engine unavailability fails closed.
        3. **Build** — persist only compiled references; resolve selected examples
           ephemerally from the governed corpus while constructing model context.
        4. **Evaluate** — score baseline vs candidate against the agent's eval slice.
        5. **Decide + apply** — :func:`should_promote`, then :meth:`apply_edits` writes the
           winner ONLY under ``KG_AGENT_AUTO_APPLY``; otherwise it is held as a queryable
           proposal. Always leaves an audit trail (ProposedPromptChange + the manifest).

        Returns a :class:`PromptHardeningOutcome`.
        """
        from agent_utilities.prompting.structured import StructuredPrompt

        from .optimization_backend import opaque_program_reference
        from .program_optimization import (
            PromptHardeningOutcome,
            get_target,
            render_program_prompt_for_execution,
            run_program_optimization,
            score_prompt_against_corpus,
            should_promote,
        )

        agent_ref = opaque_program_reference("agent", agent_id)
        component_ref = opaque_program_reference("component", prompt_path)
        fb = feedback_service or self._feedback_service
        if fb is None:
            return PromptHardeningOutcome(
                agent_ref=agent_ref,
                component_ref=component_ref,
                status="no_data",
                detail="governed_feedback_store_unavailable",
            )

        cases = fb.agent_eval_cases(agent_id)
        trainset = fb.build_agent_trainset(agent_id)
        if not cases or not trainset:
            return PromptHardeningOutcome(
                agent_ref=agent_ref,
                component_ref=component_ref,
                trainset_size=len(trainset),
                status="no_data",
                detail="governed_agent_corpus_empty",
            )

        try:
            full_path = self._resolve_component_path(prompt_path)
        except ValueError:
            return PromptHardeningOutcome(
                agent_ref=agent_ref,
                component_ref=component_ref,
                trainset_size=len(trainset),
                status="error",
                detail="prompt_component_invalid",
            )
        baseline = StructuredPrompt.load(full_path)

        # Optimize natively. There is no raw-example compatibility path: source,
        # proposals, reports, and graph artifacts retain references only.
        target = get_target("system_prompt")
        result = None
        if target is not None:
            artifact = baseline.model_dump(exclude_none=True)
            artifact["__file_path__"] = prompt_path
            result = run_program_optimization(
                target,
                artifact,
                list(trainset),
                engine=self.knowledge_engine,
            )
        if result is None:
            return PromptHardeningOutcome(
                agent_ref=agent_ref,
                component_ref=component_ref,
                trainset_size=len(trainset),
                status="error",
                detail="native_program_optimization_unavailable",
            )
        resolver = getattr(fb, "resolve_program_demonstrations", None)
        demos = (
            list(resolver(result.demonstration_refs)) if callable(resolver) else []
        )
        if len(demos) != len(result.demonstration_refs):
            return PromptHardeningOutcome(
                agent_ref=agent_ref,
                component_ref=component_ref,
                trainset_size=len(trainset),
                optimizer=result.optimizer,
                status="error",
                detail="governed_demonstration_resolution_failed",
            )

        candidate_execution = render_program_prompt_for_execution(baseline, demos)
        candidate = self._compiled_prompt_candidate(full_path, result.compiled_state)

        baseline_score = score_prompt_against_corpus(baseline.render(), cases)
        candidate_score = score_prompt_against_corpus(candidate_execution, cases)
        promote = should_promote(baseline_score, candidate_score, min_delta=min_delta)

        edit = ComponentEdit(
            component_type=ComponentType.SYSTEM_PROMPT,
            file_path=prompt_path,
            edit_summary=(
                f"Harden governed system prompt via {result.optimizer} "
                f"({len(trainset)} outcomes): {baseline_score:.3f} → {candidate_score:.3f}"
            ),
            evidence_references=list(result.compiled_state["evidence_refs"]),
            metadata={
                "agent_ref": agent_ref,
                "component_ref": component_ref,
                "promote": promote,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "optimizer": result.optimizer,
                "trainset_size": len(trainset),
                "candidate_version_hash": candidate.version_hash(),
                "program_compiled_state": result.compiled_state,
            },
        )
        manifest = ChangeManifest(baseline_score=baseline_score)
        manifest.add_edit(edit)

        await self.apply_edits(manifest, dry_run=False, auto_apply=auto_apply)

        status = edit.metadata.get("apply_status", "rejected")
        return PromptHardeningOutcome(
            agent_ref=agent_ref,
            component_ref=component_ref,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            promote=promote,
            applied=status == "applied",
            status=status,
            trainset_size=len(trainset),
            optimizer=result.optimizer,
            candidate_version_hash=candidate.version_hash(),
            detail=edit.metadata.get("proposal_ref", ""),
        )

    def _git_commit_edit(self, edit: ComponentEdit) -> str | None:
        """Create a git commit for a component edit.

        Uses a content-safe structured commit message with opaque component,
        summary, task, and evidence references.
        """
        import re

        try:
            from .optimization_backend import (
                is_opaque_program_reference,
                opaque_program_reference,
            )

            self._resolve_component_path(edit.file_path)
            # Stage the file
            staged = subprocess.run(  # nosec B603 B607
                ["git", "add", "--", edit.file_path],
                cwd=self.workspace_path,
                capture_output=True,
                timeout=10,
            )
            if staged.returncode != 0:
                logger.warning("Git stage failed (status=%s)", staged.returncode)
                return None

            def reference(namespace: str, value: Any) -> str:
                if is_opaque_program_reference(value):
                    return str(value)
                return opaque_program_reference(namespace, value)

            # The commit message is durable telemetry: retain references, not raw
            # trace labels, summaries, paths, or source-system identifiers.
            commit_msg = (
                f"ahe({edit.component_type.value}): governed component update\n\n"
                f"Component-ref: {reference('component', edit.file_path)}\n"
                f"Summary-ref: {reference('summary', edit.edit_summary)}\n"
                "Predicted-fix-refs: "
                + ", ".join(reference("task", value) for value in edit.predicted_fixes[:5])
                + "\nPredicted-regression-refs: "
                + ", ".join(
                    reference("task", value) for value in edit.predicted_regressions[:5]
                )
                + "\nEvidence-refs: "
                + ", ".join(
                    reference("evidence", value)
                    for value in edit.evidence_references[:3]
                )
            )
            result = subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "commit",
                    "--only",
                    "-m",
                    commit_msg,
                    "--",
                    edit.file_path,
                ],
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
                commit_sha = sha_result.stdout.strip()
                if sha_result.returncode == 0 and re.fullmatch(
                    r"[0-9a-f]{40,64}", commit_sha
                ):
                    return commit_sha
                logger.warning("Git commit reference lookup failed")
                return None
            else:
                logger.warning("Git commit failed (status=%s)", result.returncode)
                return None
        except Exception as exc:
            logger.error("Git commit failed (%s)", type(exc).__name__)
            return None

    @staticmethod
    def _manifest_persistence_payload(manifest: ChangeManifest) -> dict[str, Any]:
        """Project an operational manifest into its reference-only durable form."""
        import json
        import math
        import re

        from agent_utilities.prompting.structured import ProgramCompiledState

        from .optimization_backend import (
            is_opaque_program_reference,
            opaque_program_reference,
        )

        def reference(namespace: str, value: Any) -> str:
            rendered = str(value or "")
            if is_opaque_program_reference(rendered):
                return rendered
            return opaque_program_reference(namespace, rendered or namespace)

        def reference_list(namespace: str, values: Any) -> list[str]:
            if not isinstance(values, list | tuple):
                return []
            return list(
                dict.fromkeys(reference(namespace, value) for value in values)
            )[:1_000]

        def finite(value: Any) -> float | None:
            if value is None or isinstance(value, bool):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if math.isfinite(number) and abs(number) <= 1_000_000_000:
                return number
            return None

        def timestamp(value: Any) -> str:
            rendered = str(value or "")
            return (
                rendered
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rendered)
                else ""
            )

        persisted_edits: list[dict[str, Any]] = []
        for edit in manifest.edits[:1_000]:
            component_ref = edit.metadata.get("component_ref")
            if not is_opaque_program_reference(component_ref):
                component_ref = opaque_program_reference("component", edit.file_path)
            else:
                component_ref = str(component_ref)
            row: dict[str, Any] = {
                "edit_ref": reference("edit", edit.id),
                "component_type": edit.component_type.value,
                "component_ref": component_ref,
                "summary_ref": reference("summary", edit.edit_summary),
                "predicted_fix_refs": reference_list("task", edit.predicted_fixes),
                "predicted_regression_refs": reference_list(
                    "task", edit.predicted_regressions
                ),
                "evidence_refs": reference_list(
                    "evidence", edit.evidence_references
                ),
                "timestamp": timestamp(edit.timestamp),
                "smoke_passed": edit.smoke_passed,
            }
            if edit.diff_content:
                row["diff_ref"] = reference("diff", edit.diff_content)
            if edit.git_commit_sha:
                row["commit_ref"] = reference("commit", edit.git_commit_sha)
            if edit.attribution_signature:
                row["attribution_ref"] = reference(
                    "attribution",
                    json.dumps(
                        edit.attribution_signature,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                )
            if edit.capability_evidence:
                row["capability_evidence_refs"] = [
                    reference(
                        "capability_evidence",
                        json.dumps(
                            item,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                    )
                    for item in edit.capability_evidence[:1_000]
                ]

            metadata: dict[str, Any] = {}
            for key in ("agent_ref", "component_ref", "proposal_ref"):
                value = edit.metadata.get(key)
                if is_opaque_program_reference(value):
                    metadata[key] = str(value)
            for key in ("promote", "auto_apply_eligible"):
                value = edit.metadata.get(key)
                if isinstance(value, bool):
                    metadata[key] = value
            for key in ("baseline_score", "candidate_score"):
                value = finite(edit.metadata.get(key))
                if value is not None:
                    metadata[key] = value
            trainset_size = edit.metadata.get("trainset_size")
            if isinstance(trainset_size, int) and not isinstance(trainset_size, bool):
                metadata["trainset_size"] = min(max(trainset_size, 0), 1_000_000)
            candidate_hash = str(edit.metadata.get("candidate_version_hash") or "")
            if re.fullmatch(r"[0-9a-f]{16}", candidate_hash):
                metadata["candidate_version_hash"] = candidate_hash
            apply_status = edit.metadata.get("apply_status")
            if apply_status in {"applied", "proposed", "rejected", "error"}:
                metadata["apply_status"] = apply_status
            compiled = edit.metadata.get("program_compiled_state")
            if compiled is not None:
                metadata["program_compiled_state"] = (
                    ProgramCompiledState.model_validate(compiled).model_dump()
                )
            if metadata:
                row["compiled_metadata"] = metadata
            persisted_edits.append(row)

        payload: dict[str, Any] = {
            "schema_version": 1,
            "type": "ChangeManifest",
            "round_ref": reference("round", manifest.round_id),
            "parent_round_ref": (
                reference("round", manifest.parent_round_id)
                if manifest.parent_round_id
                else None
            ),
            "edits": persisted_edits,
            "baseline_score": finite(manifest.baseline_score),
            "predicted_score": finite(manifest.predicted_score),
            "actual_score": finite(manifest.actual_score),
            "verification_status": (
                manifest.verification_status
                if manifest.verification_status
                in {"pending", "confirmed", "reverted", "error"}
                else "error"
            ),
            "timestamp": timestamp(manifest.timestamp),
        }
        if manifest.verification_result is not None:
            result = manifest.verification_result
            payload["verification"] = {
                "fix_precision": finite(result.fix_precision),
                "fix_recall": finite(result.fix_recall),
                "regression_precision": finite(result.regression_precision),
                "overall_delta": finite(result.overall_delta),
                "random_baseline_precision": finite(
                    result.random_baseline_precision
                ),
                "attribution_lift": finite(result.attribution_lift),
                "attribution_reliable": bool(result.attribution_reliable),
                "recommendation": (
                    result.recommendation
                    if result.recommendation
                    in {"confirm", "partial_revert", "full_revert"}
                    else ""
                ),
                "unexpected_regression_refs": reference_list(
                    "task", result.unexpected_regressions
                ),
                "confirmed_fix_refs": reference_list(
                    "task", result.confirmed_fixes
                ),
                "confirmed_regression_refs": reference_list(
                    "task", result.confirmed_regressions
                ),
                "false_positive_fix_refs": reference_list(
                    "task", result.false_positive_fixes
                ),
                "unattributed_edit_refs": reference_list(
                    "edit", result.unattributed_edits
                ),
            }
        return payload

    async def persist_manifest(self, manifest: ChangeManifest) -> str:
        """Persist a manifest to both .specify/ and the Knowledge Graph.

        Dual storage: files for git-diffable normative state,
        KG for epistemic queries.

        Returns:
            The file path where the manifest was saved.
        """
        import json
        import os

        payload = self._manifest_persistence_payload(manifest)
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        round_ref = str(payload["round_ref"])

        # 1. Save the reference-only record to .specify/manifests/.
        proposed_path = manifest.to_sdd_path(self.workspace_path)
        relative_path = os.path.relpath(proposed_path, self.workspace_path)
        file_path = self._resolve_component_path(relative_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.write("\n")
        logger.info("EvolveAgent: manifest saved to the governed workspace store")

        # 2. Save to Knowledge Graph
        if self.knowledge_engine:
            try:
                self.knowledge_engine.add_memory(
                    content=serialized,
                    name=f"ChangeManifest {round_ref}",
                    category="ahe_manifest",
                    tags=["ahe", "manifest", round_ref],
                )
                logger.info("EvolveAgent: manifest saved to the Knowledge Graph")
            except Exception as exc:
                logger.warning(
                    "EvolveAgent: KG persistence failed (%s)", type(exc).__name__
                )

        return file_path
