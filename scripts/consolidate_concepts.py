#!/usr/bin/env python3
"""Concept Tag Consolidation Script.

Batch-renames CONCEPT: tags across all .py and .md files to the
canonical 37-concept set defined in docs/concept_map.md.

Run: python3 scripts/consolidate_concepts.py --dry-run
     python3 scripts/consolidate_concepts.py --apply
"""

import argparse
import re
from pathlib import Path

# ── Rename map: old tag → new canonical tag ──────────────────────────
# Only tags that CHANGE are listed here.
# Tags that stay the same (e.g., ORCH-1.0, KG-2.0) are NOT listed.
RENAME_MAP = {
    # ── ORCH consolidation ──
    "ORCH-1.5": "ORCH-1.3",  # state management → Execution Safety & State
    "ORCH-1.6": "ORCH-1.5",  # agent orchestrator → Agent Orchestrator (renumber)
    "ORCH-1.14": "ORCH-1.2",  # fallback chains → Specialist Routing
    "ORCH-1.15": "ORCH-1.1",  # pipelined routing → HTN Planning
    "ORCH-1.16": "ORCH-1.1",  # query-time routing → HTN Planning
    "ORCH-1.17": "ORCH-1.2",  # autorouting → Specialist Routing
    "ORCH-1.18": "ORCH-1.4",  # dynamic wiring → Capability Wiring
    "ORCH-1.19": "ORCH-1.4",  # subgraph synthesis → Capability Wiring
    "ORCH-1.20": "ORCH-1.4",  # capability auto-activation → Capability Wiring
    # ORCH-1.21 → ORCH-1.4 (already implied by wiring engine)
    # ── KG consolidation (most aggressive) ──
    "KG-001": "KG-2.0",
    "KG-002": "KG-2.0",
    "KG-003": "KG-2.0",
    "KG-004": "KG-2.0",
    "KG-3.00": "KG-2.0",
    "KG-2.10": "KG-2.1",  # context compaction → Tiered Memory
    "KG-2.13": "KG-2.1",  # chat recall → Tiered Memory
    "KG-2.14": "KG-2.1",  # project-aware context → Tiered Memory
    "KG-2.16": "KG-2.2",  # semantic subsumption → Ontology & Epistemics
    "KG-2.23": "KG-2.2",  # agent reasoning → Ontology & Epistemics
    "KG-2.36": "KG-2.3",  # auto-similarity → Graph Integrity
    "KG-2.37": "KG-2.3",  # hybrid retriever → Graph Integrity
    "KG-2.38": "KG-2.3",  # consistency → Graph Integrity
    "KG-2.19": "KG-2.4",  # cross-pillar synergy → Inductive Knowledge
    "KG-2.15": "KG-2.5",  # analogy engine → Topological Analysis
    "KG-2.34": "KG-2.5",  # spectral clusters → Topological Analysis
    "KG-2.35": "KG-2.5",  # blast radius → Topological Analysis
    # Finance domain collapse
    "KG-2.7": "KG-2.6",  # risk scoring → Finance
    "KG-2.8": "KG-2.6",  # financial models → Finance
    "KG-2.9": "KG-2.6",  # trading strategies → Finance
    "KG-2.40": "KG-2.6",  # portfolio optimization → Finance
    "KG-2.41": "KG-2.6",  # market microstructure → Finance
    "KG-2.42": "KG-2.6",  # embedding alignment → Finance  (WAIT: this is also KG-2.8)
    "KG-2.43": "KG-2.6",  # embedding index → Finance
    "KG-2.44": "KG-2.6",  # anti-collapse → Finance (WAIT: also KG-2.8)
    "KG-2.45": "KG-2.6",  # similarity search → Finance
    "KG-2.46": "KG-2.6",  # optimal execution → Finance
    "KG-2.47": "KG-2.6",  # order routing → Finance
    "KG-2.48": "KG-2.6",  # slippage model → Finance
    "KG-2.49": "KG-2.6",  # execution analytics → Finance
    # Research intelligence collapse
    "KG-2.11": "KG-2.7",  # research pipeline → Research Intelligence (renumber)
    "KG-2.12": "KG-2.7",  # source resolver → Research Intelligence
    "KG-2.33": "KG-2.7",  # research sub-agent → Research Intelligence
    "KG-2.39": "KG-2.7",  # research orchestration → Research Intelligence
    # Memory stability
    "KG-2.17": "KG-2.8",  # versioned mutations → Memory Stability
    # Multi-domain architecture
    "KG-2.51": "KG-2.9",  # multi-domain → Multi-Domain Architecture
    "KG-2.52": "KG-2.9",  # team sharing → Multi-Domain Architecture
    "KG-2.53": "KG-2.9",  # state checkpointing → Multi-Domain Architecture
    # Enterprise domain collapse
    "KG-2.80": "KG-2.10",  # enterprise core → Enterprise
    "KG-2.85": "KG-2.10",  # governance → Enterprise
    "KG-2.90": "KG-2.10",  # infrastructure → Enterprise
    "KG-2.95": "KG-2.10",  # HR → Enterprise
    # Vectorized retrieval
    "KG-2.50": "KG-2.11",  # context-window filtering → Vectorized Retrieval
    # Finance sub-concepts (KG-2.60-2.76)
    **{f"KG-2.{i}": "KG-2.6" for i in range(60, 77)},
    # ── AHE consolidation ──
    "AHE-3.10": "AHE-3.1",  # decomposed rewards → Continuous Evaluation
    "AHE-3.12": "AHE-3.1",  # multi-strategy eval → Continuous Evaluation
    "AHE-3.13": "AHE-3.2",  # config versioning → Evolution Engine
    "AHE-3.14": "AHE-3.2",  # engineering patterns → Evolution Engine
    "AHE-3.15": "AHE-3.3",  # coalition composition → Team & Synergy
    "AHE-3.16": "AHE-3.3",  # synergy scoring → Team & Synergy
    "AHE-3.5": "AHE-3.4",  # self-model → Distributed Evolution
    "AHE-3.6": "AHE-3.4",  # stability → Distributed Evolution
    "AHE-3.7": "AHE-3.5",  # heavy thinking → Heavy Thinking (renumber)
    "AHE-3.17": "AHE-3.5",  # background intelligence → Heavy Thinking
    "AHE-3.8": "AHE-3.6",  # backtest eval → Backtest & Curriculum (renumber)
    "AHE-3.9": "AHE-3.6",  # horizon-aware → Backtest & Curriculum
    "AHE-3.23": "AHE-3.7",  # OWL specs → KG-Native Task Detection
    "AHE-3.24": "AHE-3.7",  # task detection → KG-Native Task Detection
    "AHE-3.25": "AHE-3.7",  # topological reasoning → KG-Native Task Detection
    "AHE-3.18": "AHE-3.7",  # auto-healing → KG-Native Task Detection
    "AHE-3.11": "ORCH-1.3",  # structured retry → Execution Safety
    # ── ECO consolidation ──
    "ECO-4.1": "ECO-4.0",  # universal skills → Tool Interface
    "ECO-4.5": "ECO-4.0",  # skill loading → Tool Interface
    "ECO-4.6": "ECO-4.0",  # bridges → Tool Interface
    "ECO-4.9": "ECO-4.0",  # tool assignment → Tool Interface
    "ECO-4.10": "ECO-4.0",  # bridge → Tool Interface
    "ECO-4.2": "ECO-4.1",  # A2A network → A2A (renumber)
    "ECO-4.3": "ECO-4.2",  # community telemetry → Telemetry (renumber)
    "ECO-4.7": "ECO-4.2",  # ecosystem topology → Telemetry
    "ECO-4.4": "ECO-4.3",  # market data → Market Data (renumber)
    "ECO-4.11": "ECO-4.4",  # durable exec → KG MCP (renumber)
    "ECO-4.12": "ECO-4.4",  # jupyter sandbox → KG MCP
    "ECO-4.13": "ECO-4.4",  # KG MCP → KG MCP
    # ── OS consolidation ──
    "OS-5.3": "OS-5.1",  # session concurrency → Security
    "OS-5.4": "OS-5.1",  # prompt injection → Security
    "OS-5.10": "OS-5.1",  # prompt governance → Security
    "OS-5.5": "OS-5.3",  # tool repetition → Guardrails (renumber)
    "OS-5.8": "OS-5.3",  # guardrail engine → Guardrails
    "OS-5.11": "OS-5.3",  # vulnerability scanner → Guardrails
    "OS-5.6": "OS-5.4",  # token tracking → Telemetry (renumber)
    "OS-5.7": "OS-5.4",  # audit logger → Telemetry
    "OS-5.9": "OS-5.4",  # observability → Telemetry
    "OS-5.18": "OS-5.0",  # lifecycle → Kernel
    "OS-5.19": "OS-5.0",  # session persistence → Kernel
    "OS-5.20": "OS-5.0",  # paths → Kernel
}


def fix_trailing_dots(text: str) -> str:
    """Remove trailing dots from CONCEPT tags (e.g., CONCEPT:KG-2.0 → CONCEPT:KG-2.0)."""
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
