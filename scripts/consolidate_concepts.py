#!/usr/bin/env python3
"""Concept Tag Consolidation Script.

Batch-renames CONCEPT: tags across all .py and .md files to the
canonical 37-concept set defined in docs/concept_map.md.

Run: python3 scripts/consolidate_concepts.py --dry-run
     python3 scripts/consolidate_concepts.py --apply
"""

# ruff: noqa: F601 — completed one-off migration; the duplicate source keys are a
# frozen (last-wins) artifact of the historical consolidation run, not reconstructable.

import argparse
import re
from pathlib import Path

# ── Rename map: old tag → new canonical tag ──────────────────────────
# Only tags that CHANGE are listed here.
# Tags that stay the same (e.g., ORCH-1.0, KG-2.0) are NOT listed.
RENAME_MAP = {
    # ── ORCH synthesis ──
    "AU-ORCH.planning.legal-automation-roadmap": "AU-ORCH.execution.execution-budget-caps",  # state management → Execution Safety & State
    "AU-ORCH.planning.legal-automation-roadmap": "AU-ORCH.planning.legal-automation-roadmap",  # agent orchestrator → Agent Orchestrator (renumber)
    "ORCH-1.14": "AU-ORCH.adapter.hot-cache-invalidation",  # fallback chains → Specialist Routing
    "ORCH-1.15": "AU-ORCH.planning.recursion-nesting-depth",  # pipelined routing → HTN Planning
    "ORCH-1.16": "AU-ORCH.planning.recursion-nesting-depth",  # query-time routing → HTN Planning
    "ORCH-1.17": "AU-ORCH.adapter.hot-cache-invalidation",  # autorouting → Specialist Routing
    "ORCH-1.18": "AU-ORCH.adapter.kg-graph-materialization",  # dynamic wiring → Capability Wiring
    "ORCH-1.19": "AU-ORCH.adapter.kg-graph-materialization",  # subgraph synthesis → Capability Wiring
    "AU-ORCH.execution.service-registry-initialization": "AU-ORCH.adapter.kg-graph-materialization",  # capability auto-activation → Capability Wiring
    # ORCH-1.21 → ORCH-1.4 (already implied by wiring engine)
    # ── KG synthesis (most aggressive) ──
    "KG-001": "AU-KG.query.object-graph-mapper",
    "KG-002": "AU-KG.query.object-graph-mapper",
    "KG-003": "AU-KG.query.object-graph-mapper",
    "KG-004": "AU-KG.query.object-graph-mapper",
    "KG-3.00": "AU-KG.query.object-graph-mapper",
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.memory.tiered-memory-caching",  # project-aware context → Tiered Memory
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.ingest.engineering-rules",  # semantic subsumption → Ontology & Epistemics
    "AU-KG.domains.legal-automation": "AU-KG.ingest.engineering-rules",  # agent reasoning → Ontology & Epistemics
    "AU-KG.ontology.pack-owl-closure": "AU-KG.memory.auto-similarity-memory-graph",  # auto-similarity → Graph Integrity
    "AU-KG.research.research-state-domain-pack": "AU-KG.memory.auto-similarity-memory-graph",  # hybrid retriever → Graph Integrity
    "AU-KG.ontology.conformance-check": "AU-KG.memory.auto-similarity-memory-graph",  # consistency → Graph Integrity
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.compute.cross-pillar-synergy",  # cross-pillar synergy → Inductive Knowledge
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.compute.spectral-cluster-navigator",  # analogy engine → Topological Analysis
    "AU-KG.retrieval.relational-intent-retrieval": "AU-KG.compute.spectral-cluster-navigator",  # spectral clusters → Topological Analysis
    "AU-KG.ontology.schema-pack-lifecycle-audit": "AU-KG.compute.spectral-cluster-navigator",  # blast radius → Topological Analysis
    # Finance domain collapse
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.research.research-pipeline-runner",  # trading strategies → Finance
    "AU-KG.ontology.derived-property-registry": "AU-KG.research.research-pipeline-runner",  # portfolio optimization → Finance
    "AU-KG.ontology.default-runtime-bound-import": "AU-KG.research.research-pipeline-runner",  # market microstructure → Finance
    "AU-KG.ontology.batch-actions-executor": "AU-KG.research.research-pipeline-runner",  # embedding alignment → Finance  (WAIT: this is also KG-2.7)
    "AU-KG.ontology.edit-ledger-writeback": "AU-KG.research.research-pipeline-runner",  # embedding index → Finance
    "AU-KG.ontology.batch-incremental-sync-live": "AU-KG.research.research-pipeline-runner",  # anti-collapse → Finance (WAIT: also KG-2.7)
    "AU-KG.ontology.link-type-pivot": "AU-KG.research.research-pipeline-runner",  # similarity search → Finance
    "AU-KG.ontology.redact-object-materialize-restricted": "AU-KG.research.research-pipeline-runner",  # optimal execution → Finance
    "AU-KG.ontology.ontology-property-types": "AU-KG.research.research-pipeline-runner",  # order routing → Finance
    "AU-KG.ingest.chunk-overlap-stage": "AU-KG.research.research-pipeline-runner",  # slippage model → Finance
    "AU-KG.ontology.populated-at-import-real-3": "AU-KG.research.research-pipeline-runner",  # execution analytics → Finance
    # Research intelligence collapse
    "AU-KG.query.vendor-agnostic-traversal": "AU-KG.query.vendor-agnostic-traversal",  # source resolver → Research Intelligence
    "AU-KG.research.zero-llm-pack-link": "AU-KG.query.vendor-agnostic-traversal",  # research sub-agent → Research Intelligence
    "AU-KG.ontology.value-type-shacl-load": "AU-KG.query.vendor-agnostic-traversal",  # research orchestration → Research Intelligence
    # Memory stability
    "EG-KG.compute.compiled-semantic-reasoner": "AU-KG.query.vendor-agnostic-traversal",  # versioned mutations → Memory Stability
    # Multi-domain architecture
    "EG-KG.txn.per-graph-write-isolation": "AU-KG.query.vendor-agnostic-traversal",  # multi-domain → Multi-Domain Architecture
    "AU-KG.ontology.authoritative-tbox": "AU-KG.query.vendor-agnostic-traversal",  # team sharing → Multi-Domain Architecture
    "AU-KG.ontology.descriptive-process-world-gains": "AU-KG.query.vendor-agnostic-traversal",  # state checkpointing → Multi-Domain Architecture
    # Enterprise domain collapse
    "AU-KG.retrieval.evidence-graph-workspace": "AU-KG.query.vendor-agnostic-traversal",  # enterprise core → Enterprise
    "AU-KG.ingest.world-model-gate": "AU-KG.query.vendor-agnostic-traversal",  # HR → Enterprise
    # Vectorized retrieval
    "AU-KG.enrichment.contextual-retrieval-enrichment": "AU-KG.query.vendor-agnostic-traversal",  # context-window filtering → Vectorized Retrieval
    # Finance sub-concepts (KG-2.7-2.76)
    **{f"KG-2.{i}": "AU-KG.research.research-pipeline-runner" for i in range(60, 77)},
    # ── AHE synthesis ──
    "AHE-3.10": "AU-AHE.evaluation.adaptive-reasoning-effort",  # decomposed rewards → Continuous Evaluation
    "AU-AHE.evaluation.longmemeval-validation-harness": "AU-AHE.evaluation.adaptive-reasoning-effort",  # multi-strategy eval → Continuous Evaluation
    "AU-AHE.harness.pre-emit-quality-gate": "AU-AHE.harness.evolutionary-aggregation",  # config versioning → Evolution Engine
    "AU-AHE.assimilation.research-auto-merge": "AU-AHE.harness.evolutionary-aggregation",  # engineering patterns → Evolution Engine
    "AU-AHE.harness.self-improvement-overview": "AU-AHE.evaluation.interpretability-tests",  # coalition composition → Team & Synergy
    "AU-AHE.harness.width-diverse-best-k": "AU-AHE.evaluation.interpretability-tests",  # synergy scoring → Team & Synergy
    "AU-AHE.harness.self-evolution-narrative": "AU-AHE.evaluation.backtest-harness",  # self-model → Distributed Evolution
    "AU-AHE.harness.evolution-checkpoint": "AU-AHE.evaluation.backtest-harness",  # stability → Distributed Evolution
    "AU-AHE.harness.concept-2": "AU-AHE.harness.self-evolution-narrative",  # heavy thinking → Heavy Thinking (renumber)
    "AU-AHE.harness.preference-corpus-reliability": "AU-AHE.harness.self-evolution-narrative",  # background intelligence → Heavy Thinking
    "AU-AHE.harness.self-improvement-overview": "AU-AHE.harness.evolution-checkpoint",  # backtest eval → Backtest & Curriculum (renumber)
    "AU-AHE.optimization.physical-distillation-engine": "AU-AHE.harness.evolution-checkpoint",  # horizon-aware → Backtest & Curriculum
    "AU-AHE.harness.capability-ratchet": "AU-AHE.harness.concept-2",  # OWL specs → KG-Native Task Detection
    "AU-AHE.evaluation.capability-benchmark-regression-ratchet": "AU-AHE.harness.concept-2",  # task detection → KG-Native Task Detection
    "AU-AHE.evaluation.failure-analysis-loop": "AU-AHE.harness.concept-2",  # topological reasoning → KG-Native Task Detection
    "AU-AHE.harness.failure-evolution": "AU-AHE.harness.concept-2",  # auto-healing → KG-Native Task Detection
    "AU-AHE.optimization.gitops-commit-automation": "AU-ORCH.execution.execution-budget-caps",  # structured retry → Execution Safety
    # ── ECO synthesis ──
    "AU-ECO.mcp.fastmcp-middleware": "AU-ECO.messaging.native-backend-abstraction",  # universal skills → Tool Interface
    "AU-ECO.toolkit.journey-map-milestones": "AU-ECO.messaging.native-backend-abstraction",  # skill loading → Tool Interface
    "AU-ECO.mcp.toolkit-live-discovery": "AU-ECO.messaging.native-backend-abstraction",  # bridges → Tool Interface
    "AU-ECO.bus.pluggable-queue-backend": "AU-ECO.messaging.native-backend-abstraction",  # bridge → Tool Interface
    "AU-ECO.toolkit.journey-map-narrative": "AU-ECO.mcp.fastmcp-middleware",  # A2A network → A2A (renumber)
    "AU-ECO.ui.company-infrastructure-orchestration": "AU-ECO.toolkit.journey-map-narrative",  # community telemetry → Telemetry (renumber)
    "AU-OS.deployment.infra-orchestration": "AU-ECO.toolkit.journey-map-narrative",  # ecosystem topology → Telemetry
    "AU-ECO.toolkit.journey-map-adoption": "AU-ECO.ui.company-infrastructure-orchestration",  # market data → Market Data (renumber)
    "AU-OS.governance.lint-enforcement-hook": "AU-ECO.toolkit.journey-map-adoption",  # durable exec → KG MCP (renumber)
    "AU-ECO.toolkit.self-documenting-plugin-bundle": "AU-ECO.toolkit.journey-map-adoption",  # jupyter sandbox → KG MCP
    "AU-OS.deployment.infra-orchestration": "AU-ECO.toolkit.journey-map-adoption",  # KG MCP → KG MCP
    # ── OS synthesis ──
    "AU-OS.governance.reactive-multi-axis-budget": "AU-OS.config.secrets-authentication",  # session concurrency → Security
    "AU-OS.governance.wasm-micro-agent-sandbox": "AU-OS.config.secrets-authentication",  # prompt injection → Security
    "AU-OS.safety.ontological-guardrail": "AU-OS.config.secrets-authentication",  # prompt governance → Security
    "AU-OS.host.homeostatic-recovery-daemon": "AU-OS.governance.reactive-multi-axis-budget",  # tool repetition → Guardrails (renumber)
    "AU-OS.deployment.platform-journey": "AU-OS.governance.reactive-multi-axis-budget",  # guardrail engine → Guardrails
    "AU-OS.observability.run-wide-correlation-id": "AU-OS.governance.reactive-multi-axis-budget",  # vulnerability scanner → Guardrails
    "AU-OS.host.homeostatic-recovery-daemon": "AU-OS.governance.wasm-micro-agent-sandbox",  # token tracking → Telemetry (renumber)
    "AU-OS.observability.deterministic-replay": "AU-OS.governance.wasm-micro-agent-sandbox",  # audit logger → Telemetry
    "AU-OS.scaling.epistemic-dynamic-priority-quota": "AU-OS.governance.wasm-micro-agent-sandbox",  # observability → Telemetry
    "AU-OS.state.fleet-supervisory-plane-at": "AU-OS.safety.doom-loop-detection",  # lifecycle → Kernel
    "OS-5.19": "AU-OS.safety.doom-loop-detection",  # session persistence → Kernel
    "OS-5.20": "AU-OS.safety.doom-loop-detection",  # paths → Kernel
}


def fix_trailing_dots(text: str) -> str:
    """Remove trailing dots from CONCEPT tags (e.g., CONCEPT:AU-KG.query.object-graph-mapper → CONCEPT:AU-KG.query.object-graph-mapper)."""
    # Match CONCEPT:XX-N.N. where the final dot is NOT followed by a digit
    return re.sub(r"(CONCEPT:[A-Z]+-\d+\.\d+)\.(?!\d)", r"\1", text)


def apply_renames(text: str) -> str:
    """Apply the rename map to all CONCEPT tags in text."""
    # Sort by longest key first to avoid partial matches
    sorted_keys = sorted(RENAME_MAP.keys(), key=len, reverse=True)

    for old_tag in sorted_keys:
        new_tag = RENAME_MAP[old_tag]
        # Replace both CONCEPT:TAG and bare TAG references in context
        text = text.replace(f"CONCEPT:{old_tag}", f"CONCEPT:{new_tag}")

    return text


def process_file(path: Path, dry_run: bool = True) -> tuple[bool, int]:
    """Process a single file. Returns (changed, num_replacements)."""
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False, 0

    # Step 1: Fix trailing dots
    result = fix_trailing_dots(original)
    # Step 2: Apply renames
    result = apply_renames(result)

    if result == original:
        return False, 0

    # Count changes
    changes = sum(1 for a, b in zip(original, result, strict=False) if a != b)

    if not dry_run:
        path.write_text(result, encoding="utf-8")

    return True, changes


def main():
    parser = argparse.ArgumentParser(description="Consolidate CONCEPT tags")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show changes without applying (default)",
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    args = parser.parse_args()

    dry_run = not args.apply
    root = Path(__file__).parent.parent

    # Find all .py and .md files (excluding __pycache__, .git, node_modules)
    files = []
    for ext in ("*.py", "*.md"):
        for f in root.rglob(ext):
            if any(
                skip in str(f)
                for skip in ["__pycache__", ".git/", "node_modules", ".venv"]
            ):
                continue
            files.append(f)

    print(f"Scanning {len(files)} files...")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING CHANGES'}")
    print()

    changed_files = []
    total_changes = 0

    for f in sorted(files):
        changed, count = process_file(f, dry_run=dry_run)
        if changed:
            rel = f.relative_to(root)
            changed_files.append((rel, count))
            total_changes += count

    if changed_files:
        print(f"{'Would change' if dry_run else 'Changed'} {len(changed_files)} files:")
        for rel, count in changed_files:
            print(f"  {rel} ({count} chars changed)")
    else:
        print("No changes needed.")

    print(f"\nTotal: {len(changed_files)} files, {total_changes} character changes")

    if dry_run and changed_files:
        print("\nRun with --apply to apply changes.")


if __name__ == "__main__":
    main()
