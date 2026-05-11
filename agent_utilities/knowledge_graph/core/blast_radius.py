#!/usr/bin/python
from __future__ import annotations

"""Symbol Blast Radius Analyzer.

CONCEPT:KG-2.5 — Symbol Blast Radius Analyzer

Traces how widely a Python symbol (function, class, variable) is used
across a codebase. Adapted from contextplus's blast-radius.ts with
KG integration for structural impact scoring.

Provides regex-based symbol usage tracking, definition-line exclusion,
and low-usage warnings for potential dead code detection.
"""


import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any

from agent_utilities.models.knowledge_graph import BlastRadiusNode

logger = logging.getLogger(__name__)

# Python definition patterns — used to exclude definition lines from usage counts
_DEFINITION_PATTERNS = [
    re.compile(r"^\s*(?:def|class|async\s+def)\s+{symbol}\b"),
    re.compile(r"^\s*(?:{symbol})\s*[=:]"),
    re.compile(r"^\s*(?:import|from)\s+.*\b{symbol}\b"),
]

# File patterns to exclude from analysis
_EXCLUDE_PATTERNS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".egg-info",
}


class BlastRadiusAnalyzer:
    """Analyzes how widely a symbol is used across a Python codebase.

    CONCEPT:KG-2.5 — Symbol Blast Radius Analyzer

    Adapted from contextplus's ``getBlastRadius()`` function for Python
    codebases with KG integration and structural impact scoring.

    Example::

        analyzer = BlastRadiusAnalyzer("/path/to/project")
        result = analyzer.analyze("my_function", definition_file="src/utils.py")
        print(f"Impact: {result.impact_score:.2f}, used in {result.file_count} files")
    """

    def __init__(
        self,
        root_dir: str | Path,
        file_extensions: set[str] | None = None,
    ):
        """Initialize the analyzer.

        Args:
            root_dir: Root directory of the codebase to analyze.
            file_extensions: Set of file extensions to scan (default: {'.py'}).
        """
        self._root = Path(root_dir)
        self._extensions = file_extensions or {".py"}

    def _iter_files(self) -> list[Path]:
        """Iterate over all eligible source files."""
        files: list[Path] = []
        for ext in self._extensions:
            for path in self._root.rglob(f"*{ext}"):
                # Skip excluded directories
                parts = path.parts
                if any(p in _EXCLUDE_PATTERNS for p in parts):
                    continue
                files.append(path)
        return files

    @staticmethod
    def _is_definition_line(line: str, symbol: str) -> bool:
        """Check if a line is a definition of the symbol."""
        for pattern_template in _DEFINITION_PATTERNS:
            pattern = re.compile(
                pattern_template.pattern.format(symbol=re.escape(symbol))
            )
            if pattern.search(line):
                return True
        return False

    def analyze(
        self,
        symbol_name: str,
        definition_file: str | None = None,
        symbol_type: str = "function",
    ) -> BlastRadiusNode:
        """Analyze the blast radius of a symbol.

        Args:
            symbol_name: The symbol to search for.
            definition_file: Optional file where the symbol is defined
                (to exclude definition lines).
            symbol_type: Type of symbol (function, class, variable, constant).

        Returns:
            BlastRadiusNode with usage analysis results.
        """
        symbol_pattern = re.compile(rf"\b{re.escape(symbol_name)}\b")
        usages: list[dict[str, Any]] = []
        definition_line = 0
        files_with_usage: set[str] = set()

        for file_path in self._iter_files():
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            rel_path = str(file_path.relative_to(self._root))
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if not symbol_pattern.search(line):
                    continue

                line_num = i + 1
                is_def = False

                # Check if this is the definition
                if definition_file and rel_path == definition_file:
                    if self._is_definition_line(line, symbol_name):
                        is_def = True
                        definition_line = line_num

                if not is_def:
                    usages.append(
                        {
                            "file": rel_path,
                            "line": line_num,
                            "context": line.strip()[:120],
                        }
                    )
                    files_with_usage.add(rel_path)

        usage_count = len(usages)
        file_count = len(files_with_usage)
        # Impact score: log-scaled by usage count and file diversity
        impact_score = (
            min(
                1.0,
                (math.log1p(usage_count) * math.log1p(file_count)) / 10.0,
            )
            if usage_count > 0
            else 0.0
        )

        return BlastRadiusNode(
            id=f"br_{uuid.uuid4().hex[:8]}",
            name=f"Blast radius: {symbol_name}",
            description=(
                f"Symbol '{symbol_name}' has {usage_count} usages "
                f"across {file_count} files"
            ),
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            definition_file=definition_file or "",
            definition_line=definition_line,
            usage_count=usage_count,
            file_count=file_count,
            impact_score=impact_score,
            is_low_usage=usage_count <= 1,
            usages=usages,
        )
