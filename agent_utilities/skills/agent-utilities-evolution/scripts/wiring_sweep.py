#!/usr/bin/env python3
"""AST-based Wiring Sweep for agent-utilities.

Part of the agent-utilities-evolution skill. Provides a comprehensive
codebase analysis for concept traceability, dead code detection,
import graph analysis, and wiring gap identification.

CONCEPT:AU-AHE.evaluation.continuous-evaluation — Continuous Evaluation Engine (code health diagnostic)

Usage:
    python wiring_sweep.py /path/to/agent-utilities [--json] [--markdown]
    python wiring_sweep.py /path/to/agent-utilities --output report.md
    python wiring_sweep.py /path/to/agent-utilities --json --output report.json

Reports:
    - Concept coverage: Code → Tests → Docs 1:1:1 traceability
    - Import graph: Module dependency analysis with orphan detection
    - Dead code: Unreferenced functions/classes (via call graph)
    - Wiring gaps: Cross-module integration issues
    - Summary statistics with health score
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Core AST Analysis Engine
# ---------------------------------------------------------------------------


class WiringSweep:
    """AST-based codebase wiring analysis engine.

    Builds import and call graphs from Python source files,
    then cross-references against concept registries and docs.
    """

    CONCEPT_RE = re.compile(r"CONCEPT:([A-Z]+-\d+\.\d+)")
    TOOL_FLAG_RE = re.compile(r"^[A-Z]+TOOL$")

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()
        self.src_dir = self.root / "agent_utilities"
        self.test_dir = self.root / "tests"
        self.docs_dir = self.root / "docs"

        # Data structures
        self.modules: dict[str, dict[str, Any]] = {}
        self.import_graph: dict[str, set[str]] = defaultdict(set)
        self.reverse_import_graph: dict[str, set[str]] = defaultdict(set)
        self.call_graph: dict[str, set[str]] = defaultdict(set)
        self.concept_to_files: dict[str, list[str]] = defaultdict(list)
        self.concept_to_tests: dict[str, list[str]] = defaultdict(list)
        self.concept_to_docs: dict[str, list[str]] = defaultdict(list)
        self.all_definitions: dict[str, list[str]] = defaultdict(list)
        self.all_references: set[str] = set()

        # Valid concepts from concept_map.md
        self.valid_concepts: set[str] = set()
        self.mermaid_total_nodes = 0
        self.mermaid_mapped_nodes = 0
        self.mermaid_missing_nodes: list[dict[str, str]] = []
        self.mermaid_invalid_nodes: list[dict[str, str]] = []

        # Load concepts
        self._load_canonical_concepts()

        # Results
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute the full analysis pipeline."""
        start = time.monotonic()

        self._scan_source_files()
        self._scan_test_files()
        self._scan_docs()
        self._build_import_graph()
        self._analyze_orphans()
        self._analyze_concept_gaps()
        self._analyze_dead_definitions()
        self._compute_health_score()

        self.results["duration_seconds"] = round(time.monotonic() - start, 2)
        return self.results

    # -- Phase 1: Source File Scanning ---

    def _scan_source_files(self) -> None:
        """Parse all .py files in agent_utilities/."""
        if not self.src_dir.exists():
            self.results["error"] = f"Source directory not found: {self.src_dir}"
            return

        total_files = 0
        total_functions = 0
        total_classes = 0
        total_lines = 0
        syntax_errors: list[str] = []

        for py_file in sorted(self.src_dir.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue

            total_files += 1
            rel = str(py_file.relative_to(self.root))

            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                syntax_errors.append(rel)
                continue

            lines = text.count("\n") + 1
            total_lines += lines

            # Parse AST
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError:
                syntax_errors.append(rel)
                continue

            # Extract definitions and references
            functions: list[str] = []
            classes: list[str] = []
            imports: list[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    functions.append(node.name)
                    total_functions += 1
                    self.all_definitions[node.name].append(rel)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    total_classes += 1
                    self.all_definitions[node.name].append(rel)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        for alias in node.names or []:
                            self.all_references.add(alias.name)
                elif isinstance(node, ast.Name):
                    self.all_references.add(node.id)
                elif isinstance(node, ast.Attribute):
                    self.all_references.add(node.attr)

            # Extract concept tags
            concepts = self.CONCEPT_RE.findall(text)

            self.modules[rel] = {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "concepts": list(set(concepts)),
                "lines": lines,
                "is_init": py_file.name == "__init__.py",
            }

            for c in set(concepts):
                self.concept_to_files[c].append(rel)

        self.results["source"] = {
            "total_files": total_files,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_lines": total_lines,
            "syntax_errors": syntax_errors,
        }

    # -- Phase 2: Test File Scanning ---

    def _scan_test_files(self) -> None:
        """Scan test files for concept tags."""
        if not self.test_dir.exists():
            return

        test_count = 0
        for py_file in sorted(self.test_dir.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            test_count += 1
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            concepts = self.CONCEPT_RE.findall(text)
            rel = str(py_file.relative_to(self.root))
            for c in set(concepts):
                self.concept_to_tests[c].append(rel)

            # Also scan for references to track test coverage
            for node in ast.walk(ast.parse(text, filename=rel)):
                if isinstance(node, ast.ImportFrom | ast.Import):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        for alias in node.names or []:
                            self.all_references.add(alias.name)

        self.results["tests"] = {
            "total_test_files": test_count,
            "concepts_with_test_tags": len(self.concept_to_tests),
        }

    # -- Phase 3: Documentation Scanning ---

    def _scan_docs(self) -> None:
        """Scan docs for concept references."""
        if not self.docs_dir.exists():
            return

        for md_file in sorted(self.docs_dir.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Match both "CONCEPT:X-Y.Z" and bare "X-Y.Z" in docs
            concepts = re.findall(r"(?:CONCEPT:)?([A-Z]+-\d+\.\d+)", text)
            rel = str(md_file.relative_to(self.root))
            for c in set(concepts):
                self.concept_to_docs[c].append(rel)

            # Parse Mermaid nodes for Concept IDs and validate
            nodes = self._parse_mermaid_nodes(text)
            for concept_id, line in nodes:
                self.mermaid_total_nodes += 1
                if concept_id is None:
                    self.mermaid_missing_nodes.append({"file": rel, "line": line})
                elif self.valid_concepts and concept_id not in self.valid_concepts:
                    self.mermaid_invalid_nodes.append(
                        {"file": rel, "concept": concept_id, "line": line}
                    )
                else:
                    self.mermaid_mapped_nodes += 1

    def _load_canonical_concepts(self) -> None:
        """Extract all valid CONCEPT IDs from concept_map.md."""
        concept_map = self.docs_dir / "concept_map.md"
        if not concept_map.exists():
            return
        try:
            content = concept_map.read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(r"`([A-Z]+-\d+\.\d+)`", content)
            for m in matches:
                self.valid_concepts.add(m)
        except Exception:
            pass

    def _parse_mermaid_nodes(self, content: str) -> list[tuple[str | None, str]]:
        """Extract nodes with Concept IDs from Mermaid blocks, handles multi-node lines robustly."""
        nodes = []
        in_mermaid = False

        for line in content.split("\n"):
            if line.strip().startswith("```mermaid"):
                in_mermaid = True
                continue
            if in_mermaid and line.strip() == "```":
                in_mermaid = False
                continue

            if in_mermaid:
                line_str = line.strip()
                # Skip empty lines, comments, and structure definitions like 'subgraph' or 'direction'
                if (
                    not line_str
                    or line_str.startswith("%%")
                    or line_str.startswith("subgraph")
                    or line_str.startswith("direction")
                    or line_str.startswith("style")
                    or line_str.startswith("end")
                    or line_str.startswith("graph")
                    or line_str.startswith("flowchart")
                    or line_str.startswith("C4Context")
                    or line_str.startswith("C4Container")
                    or line_str.startswith("C4Component")
                    or line_str.startswith("title")
                ):
                    continue

                # Exclude C4 diagram nodes which do not map to canonical concepts
                if any(
                    line_str.startswith(x)
                    for x in [
                        "Person",
                        "System",
                        "System_Ext",
                        "Container",
                        "Component",
                        "Rel",
                    ]
                ):
                    continue

                # Split line by arrow connections or transitions to process multi-node lines
                parts = re.split(
                    r"\s*(?:--+|-\.-+|==+)(?:>|<|>)?(?:\|[^|]+\|)?\s*", line_str
                )
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Match node definitions like A[Label], A("Label"), etc.
                    match = re.match(
                        r'^([A-Za-z0-9_]+)\s*([\[\(\{>]+.*[\]\)\}"]+)$', part
                    )
                    if match:
                        brackets_label = match.group(2)
                        # Extract text inside the outermost brackets/quotes
                        label_match = re.search(
                            r'[\[\(\{">]+(.*)[\]\)\}"]+', brackets_label
                        )
                        label = label_match.group(1) if label_match else brackets_label
                        label = label.replace('"', "").strip()
                        # Clean up any trailing brackets/quotes
                        label = re.sub(r'^[\[\(\{">]+|[\]\)\}"]+$', "", label).strip()

                        # Exclude non-architectural process nodes
                        if any(
                            x in label.lower()
                            for x in [
                                "phase ",
                                "stage ",
                                "step ",
                                "background research",
                                "synthesis",
                                "feature recommendations",
                                "wiring audit",
                            ]
                        ):
                            continue
                        # Exclude basic shapes and boundaries that represent generic groupings
                        if any(
                            y in label.lower()
                            for y in [
                                "<b>",
                                "<br",
                                "pydantic",
                                "scripts/",
                                "git:",
                                "fastapi",
                                "vite",
                                "react",
                                "textual",
                                "rich",
                                "httpx",
                                "neo4j",
                                "networkx",
                                "database",
                                "sqlite",
                                "postgresql",
                            ]
                        ):
                            continue
                        if any(
                            y == label.lower()
                            for y in [
                                "nx",
                                "val",
                                "exp",
                                "evo",
                                "db",
                                "ui",
                                "api",
                                "cli",
                                "auth",
                                "mcp",
                                "htn",
                                "c4",
                            ]
                        ):
                            continue

                        concept_match = re.search(r"([A-Z]+-\d+\.\d+)", part)
                        if concept_match:
                            nodes.append((concept_match.group(1), part))
                        else:
                            nodes.append((None, part))

        return nodes

    # -- Phase 4: Import Graph ---

    def _build_import_graph(self) -> None:
        """Build module-to-module import edges."""
        for rel_path, data in self.modules.items():
            for imp in data["imports"]:
                # Normalize to project-relative
                if imp.startswith("agent_utilities"):
                    target = imp.replace(".", "/") + ".py"
                    # Check if it's a package
                    pkg_init = imp.replace(".", "/") + "/__init__.py"
                    if target in self.modules:
                        self.import_graph[rel_path].add(target)
                        self.reverse_import_graph[target].add(rel_path)
                    elif pkg_init in self.modules:
                        self.import_graph[rel_path].add(pkg_init)
                        self.reverse_import_graph[pkg_init].add(rel_path)

    # -- Phase 5: Orphan Analysis ---

    def _analyze_orphans(self) -> None:
        """Find modules never imported by any non-test, non-init module."""
        orphans: list[dict[str, Any]] = []

        for rel_path, data in self.modules.items():
            if data["is_init"]:
                continue
            if rel_path.startswith("tests/"):
                continue

            importers = self.reverse_import_graph.get(rel_path, set())
            # Filter out self-imports and test imports
            real_importers = {
                imp
                for imp in importers
                if imp != rel_path and not imp.startswith("tests/")
            }

            if not real_importers:
                # Check if it's re-exported from an __init__.py
                pkg_dir = str(Path(rel_path).parent)
                init_path = pkg_dir + "/__init__.py"
                module_name = Path(rel_path).stem

                init_reexports = False
                if init_path in self.modules:
                    init_data = self.modules[init_path]
                    # Check if the init imports from this module
                    for imp in init_data["imports"]:
                        if module_name in imp:
                            init_reexports = True
                            break

                if not init_reexports:
                    orphans.append(
                        {
                            "module": rel_path,
                            "functions": len(data["functions"]),
                            "classes": len(data["classes"]),
                            "lines": data["lines"],
                            "concepts": data["concepts"],
                        }
                    )

        self.results["orphans"] = {
            "count": len(orphans),
            "modules": sorted(orphans, key=lambda x: -x["lines"]),
        }

    # -- Phase 6: Concept Gap Analysis ---

    def _analyze_concept_gaps(self) -> None:
        """Find concepts missing test or doc coverage."""
        all_concepts = sorted(set(self.concept_to_files.keys()))
        gaps: list[dict[str, Any]] = []

        for concept in all_concepts:
            in_code = self.concept_to_files.get(concept, [])
            in_tests = self.concept_to_tests.get(concept, [])
            in_docs = self.concept_to_docs.get(concept, [])

            if not in_tests or not in_docs:
                gaps.append(
                    {
                        "concept": concept,
                        "in_code": len(in_code),
                        "in_tests": len(in_tests),
                        "in_docs": len(in_docs),
                        "missing_tests": len(in_tests) == 0,
                        "missing_docs": len(in_docs) == 0,
                        "code_files": in_code[:3],  # Top 3 for brevity
                    }
                )

        # Also check for concepts in docs but not in code
        docs_only = sorted(
            set(self.concept_to_docs.keys()) - set(self.concept_to_files.keys())
        )

        self.results["concept_gaps"] = {
            "total_concepts_in_code": len(all_concepts),
            "total_concepts_in_tests": len(self.concept_to_tests),
            "total_concepts_in_docs": len(self.concept_to_docs),
            "gaps": gaps,
            "docs_only_concepts": docs_only[:20],
        }

    # -- Phase 7: Dead Definition Detection ---

    def _analyze_dead_definitions(self) -> None:
        """Find functions/classes defined but never referenced."""
        # Skip common patterns that are always "used"
        SKIP_NAMES = {
            "__init__",
            "__str__",
            "__repr__",
            "__eq__",
            "__hash__",
            "__len__",
            "__getitem__",
            "__setitem__",
            "__contains__",
            "__iter__",
            "__next__",
            "__enter__",
            "__exit__",
            "__call__",
            "__post_init__",
            "__aenter__",
            "__aexit__",
            "run",
            "main",
            "setup",
            "teardown",
            "setUp",
            "tearDown",
            "model_post_init",
            "model_validator",
            "field_validator",
        }
        SKIP_PREFIXES = ("test_", "Test", "_", "conftest")

        potentially_dead: list[dict[str, str]] = []

        for name, files in self.all_definitions.items():
            if name in SKIP_NAMES:
                continue
            if any(name.startswith(p) for p in SKIP_PREFIXES):
                continue
            if name in self.all_references:
                continue

            # Only report if defined in exactly 1 file (avoid shared names)
            if len(files) == 1:
                potentially_dead.append({"name": name, "file": files[0]})

        self.results["potentially_dead"] = {
            "count": len(potentially_dead),
            "definitions": sorted(potentially_dead, key=lambda x: x["file"])[
                :50
            ],  # Cap at 50
        }

    # -- Phase 8: Health Score ---

    def _compute_health_score(self) -> None:
        """Compute an overall health score (0-100)."""
        total_concepts = self.results.get("concept_gaps", {}).get(
            "total_concepts_in_code", 1
        )
        gaps = self.results.get("concept_gaps", {}).get("gaps", [])
        orphans = self.results.get("orphans", {}).get("count", 0)
        dead = self.results.get("potentially_dead", {}).get("count", 0)
        syntax_errors = len(self.results.get("source", {}).get("syntax_errors", []))

        # Build results for mermaid diagrams
        self.results["mermaid_diagrams"] = {
            "total_nodes": self.mermaid_total_nodes,
            "mapped_nodes": self.mermaid_mapped_nodes,
            "coverage_pct": round(
                (self.mermaid_mapped_nodes / max(1, self.mermaid_total_nodes)) * 100, 1
            ),
            "invalid_usages": self.mermaid_invalid_nodes,
            "missing_ids": self.mermaid_missing_nodes,
        }

        # Scoring:
        # - Concept coverage: 30 points (decoupled for 10 points mermaid)
        # - Mermaid Diagram mapping: 10 points
        # - No orphans: 25 points
        # - No dead code: 25 points
        # - No syntax errors: 10 points

        concept_score = max(0, 30 * (1 - len(gaps) / max(total_concepts, 1)))

        # Mermaid score calculation with invalid concept penalty
        if self.mermaid_total_nodes > 0:
            mermaid_coverage = self.mermaid_mapped_nodes / self.mermaid_total_nodes
            mermaid_score = round(10 * mermaid_coverage, 1)
        else:
            mermaid_score = 10.0

        invalid_mermaid_penalty = min(5.0, len(self.mermaid_invalid_nodes) * 2.0)
        mermaid_score = max(0.0, mermaid_score - invalid_mermaid_penalty)

        orphan_score = max(0, 25 - orphans * 2)
        dead_score = max(0, 25 - dead)
        syntax_score = 10 if syntax_errors == 0 else 0

        total = round(
            concept_score + mermaid_score + orphan_score + dead_score + syntax_score
        )
        self.results["health_score"] = {
            "total": min(100, total),
            "concept_coverage": round(concept_score, 1),
            "mermaid_coverage": round(mermaid_score, 1),
            "orphan_penalty": round(25 - orphan_score, 1),
            "dead_code_penalty": round(25 - dead_score, 1),
            "syntax_errors": syntax_errors,
        }

    # -- Output Formatters ---

    def to_json(self) -> str:
        """Serialize results to JSON."""
        return json.dumps(self.results, indent=2, default=str)

    def to_markdown(self) -> str:
        """Render results as a markdown report."""
        lines: list[str] = []
        lines.append("# Wiring Sweep Report\n")
        lines.append(
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        )
        lines.append(f"**Project**: `{self.root.name}`\n")

        # Health Score
        hs = self.results.get("health_score", {})
        score = hs.get("total", 0)
        emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        lines.append(f"\n## {emoji} Health Score: {score}/100\n")
        lines.append("| Component | Score |")
        lines.append("|-----------|-------|")
        lines.append(f"| Concept Coverage | {hs.get('concept_coverage', 0)}/30 |")
        lines.append(
            f"| Mermaid Diagram Coverage | {hs.get('mermaid_coverage', 0)}/10 |"
        )
        lines.append(f"| Orphan Penalty | -{hs.get('orphan_penalty', 0)} of 25 |")
        lines.append(f"| Dead Code Penalty | -{hs.get('dead_code_penalty', 0)} of 25 |")
        lines.append(
            f"| Syntax Errors | {hs.get('syntax_errors', 0)} (10pt bonus if 0) |"
        )

        # Source stats
        src = self.results.get("source", {})
        lines.append("\n## Source Statistics\n")
        lines.append(f"- **Files**: {src.get('total_files', 0)}")
        lines.append(f"- **Functions**: {src.get('total_functions', 0)}")
        lines.append(f"- **Classes**: {src.get('total_classes', 0)}")
        lines.append(f"- **Lines**: {src.get('total_lines', 0):,}")
        lines.append(f"- **Syntax Errors**: {len(src.get('syntax_errors', []))}")

        # Concept Gaps
        cg = self.results.get("concept_gaps", {})
        lines.append("\n## Concept Traceability\n")
        lines.append(f"- **In Code**: {cg.get('total_concepts_in_code', 0)} concepts")
        lines.append(f"- **In Tests**: {cg.get('total_concepts_in_tests', 0)} concepts")
        lines.append(f"- **In Docs**: {cg.get('total_concepts_in_docs', 0)} concepts")

        # Mermaid Diagrams
        md = self.results.get("mermaid_diagrams", {})
        lines.append("\n## Mermaid Diagram Concept Mapping\n")
        lines.append(f"- **Total Diagram Nodes**: {md.get('total_nodes', 0)}")
        lines.append(f"- **Mapped Concept Nodes**: {md.get('mapped_nodes', 0)}")
        lines.append(f"- **Diagram Coverage**: {md.get('coverage_pct', 0)}%")

        invalid_usages = md.get("invalid_usages", [])
        if invalid_usages:
            lines.append("\n### ❌ Invalid Concept Usages in Diagrams\n")
            lines.append("| File | Invalid Concept | Line |")
            lines.append("|------|-----------------|------|")
            for iu in invalid_usages:
                lines.append(f"| `{iu['file']}` | `{iu['concept']}` | `{iu['line']}` |")

        missing_ids = md.get("missing_ids", [])
        if missing_ids:
            lines.append("\n### ⚠️ Nodes Missing Concept IDs\n")
            lines.append("| File | Line |")
            lines.append("|------|------|")
            for mi in missing_ids[:20]:
                lines.append(f"| `{mi['file']}` | `{mi['line']}` |")
            if len(missing_ids) > 20:
                lines.append(f"| ... | and {len(missing_ids) - 20} more |")

        gaps = cg.get("gaps", [])
        if gaps:
            lines.append("\n### Gaps\n")
            lines.append("| Concept | Code | Tests | Docs | Missing |")
            lines.append("|---------|:----:|:-----:|:----:|---------|")
            for g in gaps:
                missing = []
                if g["missing_tests"]:
                    missing.append("tests")
                if g["missing_docs"]:
                    missing.append("docs")
                lines.append(
                    f"| `{g['concept']}` | {g['in_code']} | "
                    f"{g['in_tests']} | {g['in_docs']} | "
                    f"{', '.join(missing)} |"
                )

        # Orphans
        orph = self.results.get("orphans", {})
        lines.append(f"\n## Orphan Modules: {orph.get('count', 0)}\n")
        if orph.get("modules"):
            lines.append("| Module | Lines | Functions | Classes | Concepts |")
            lines.append("|--------|------:|----------:|--------:|----------|")
            for m in orph["modules"][:20]:
                concepts = ", ".join(m.get("concepts", [])) or "—"
                lines.append(
                    f"| `{m['module']}` | {m['lines']} | "
                    f"{m['functions']} | {m['classes']} | {concepts} |"
                )

        # Dead definitions
        dead = self.results.get("potentially_dead", {})
        lines.append(f"\n## Potentially Dead Definitions: {dead.get('count', 0)}\n")
        if dead.get("definitions"):
            lines.append("> [!NOTE]")
            lines.append("> These may be false positives for MRO mixins, decorators,")
            lines.append("> dynamically dispatched methods, or `__all__` exports.\n")
            lines.append("| Name | File |")
            lines.append("|------|------|")
            for d in dead["definitions"][:30]:
                lines.append(f"| `{d['name']}` | `{d['file']}` |")

        lines.append(
            f"\n---\n*Sweep completed in {self.results.get('duration_seconds', 0)}s*\n"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AST-based wiring sweep for agent-utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        help="Path to the agent-utilities project root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of markdown",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Output markdown report (default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write report to file instead of stdout",
    )
    args = parser.parse_args()

    sweep = WiringSweep(args.project_root)
    results = sweep.run()

    if args.json:
        output = sweep.to_json()
    else:
        output = sweep.to_markdown()

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Exit with non-zero if health score < 60
    score = results.get("health_score", {}).get("total", 0)
    return 0 if score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
