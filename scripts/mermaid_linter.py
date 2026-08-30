#!/usr/bin/env python3
"""Validate Mermaid blocks embedded in Markdown files."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections.abc import Sequence
from typing import TypedDict


class Finding(TypedDict):
    line: int
    message: str


BlockLine = tuple[int, str]

UNQUOTED_SHAPE_CONTENT_RE = re.compile(
    r"\b[a-zA-Z0-9_-]+\s*(?:\[(?!\")|\((?!\")|\{(?!\")|\(\[(?!\")|\(\((?!\")|\{\{(?!\")|\[\((?!\")|\[\[(?!\")|>\s*(?!\"))(?P<label>.*?)(?:\]|\)|\}|\)\]|\)\)|\}\}|\)\]|\]\))$"
)
ILLEGAL_UNQUOTED_CHARS = ["(", ")", "[", "]", "{", "}", "<", ">", "&", ";", ","]

# Longer shapes precede shorter variants. Edge labels are masked so an
# asymmetric shape cannot start inside ``|label|`` and consume the next node.
FLOW_SHAPE_PATTERN = re.compile(
    r"\b[a-zA-Z0-9_-]+\s*(?:"
    r'\(\[(".*?")\]\)|\[\[(".*?")\]\]|\[\((".*?")\)\]|'
    r'\(\((".*?")\)\)|\{\{(".*?")\}\}|>\s*(".*?")\]|'
    r'\[(".*?")\]|\((".*?")\)|\{(".*?")\}|'
    r"\(\[(.*?)\]\)|\[\[(.*?)\]\]|\[\((.*?)\)\]|"
    r"\(\((.*?)\)\)|\{\{(.*?)\}\}|>\s*(.*?)\]|"
    r"\[(.*?)\]|\((.*?)\)|\{(.*?)\})"
)
EDGE_LABEL_PATTERN = re.compile(r"\|[^|\n]*\|")
SEQUENCE_ARROW_PATTERN = re.compile(r"(-->>|->>|-->|->|--x|-x|--\)|-\))")
SEQUENCE_OPENERS = frozenset(("alt", "loop", "rect", "opt", "par", "critical", "break"))
SEQUENCE_DIRECTIVES = frozenset(
    "participant actor note autonumber activate deactivate alt else end loop rect opt par critical break".split()
)
SEQUENCE_ARROWS = frozenset(("->", "-->", "->>", "-->>", "-x", "--x", "-)", "--)"))
DEFAULT_EXCLUDES = frozenset(
    ".venv .git .mypy_cache .pytest_cache build dist node_modules __pycache__".split()
)


def _diagram_header(block_lines: Sequence[BlockLine]) -> tuple[int, str] | None:
    for index, (_, line) in enumerate(block_lines):
        clean = line.strip()
        if clean and not clean.startswith("%%"):
            return index, clean.split()[0].lower()
    return None


def _validate_flow_shapes(line: int, clean: str) -> list[Finding]:
    findings: list[Finding] = []
    scan_line = EDGE_LABEL_PATTERN.sub(" ", clean)
    for match in FLOW_SHAPE_PATTERN.findall(scan_line):
        label = next((item for item in match if item), "").strip()
        violations = [char for char in ILLEGAL_UNQUOTED_CHARS if char in label]
        if label and not (label.startswith('"') and label.endswith('"')) and violations:
            findings.append(
                {
                    "line": line,
                    "message": f"Unquoted special character(s) {violations} in node label '{label}'. Enclose the label in double quotes, e.g. A(\"Label\")",
                }
            )
    return findings


def _validate_flow_arrow(line: int, clean: str) -> list[Finding]:
    if "-- |" not in clean and "--  |" not in clean:
        return []
    return [
        {
            "line": line,
            "message": "Invalid arrow label syntax. Use '-->|label|' or '-- label -->' instead of '-- |label| -->'.",
        }
    ]


def _validate_sequence_structure(
    line: int, first_word: str, stack: list[BlockLine]
) -> list[Finding]:
    findings: list[Finding] = []
    if first_word in SEQUENCE_OPENERS:
        stack.append((line, first_word))
    elif first_word == "else" and (
        not stack or stack[-1][1] not in ("alt", "critical")
    ):
        findings.append(
            {
                "line": line,
                "message": "Found 'else' statement without a matching active 'alt' or 'critical' block.",
            }
        )
    elif first_word == "end" and not stack:
        findings.append(
            {
                "line": line,
                "message": "Found 'end' statement without a matching opening block (alt, loop, rect, opt, par, etc.).",
            }
        )
    elif first_word == "end":
        stack.pop()
    return findings


def _validate_sequence_arrow(line: int, clean: str, first_word: str) -> list[Finding]:
    if first_word in SEQUENCE_DIRECTIVES or ":" not in clean:
        return []
    sender_receiver = clean.split(":", 1)[0].strip()
    arrow_match = SEQUENCE_ARROW_PATTERN.search(sender_receiver)
    if not arrow_match:
        return []
    arrow = arrow_match.group(1)
    if arrow in SEQUENCE_ARROWS:
        return []
    return [
        {
            "line": line,
            "message": f"Potential invalid arrow syntax '{arrow}' in sequence diagram. Use standard arrows: ->, -->, ->>, -->>, etc.",
        }
    ]


def validate_mermaid_block(
    filepath: str, start_line: int, block_lines: Sequence[BlockLine]
) -> list[Finding]:
    """Validate one Mermaid block and return its line-numbered findings."""
    if not block_lines:
        return []
    header = _diagram_header(block_lines)
    if header is None:
        return [
            {
                "line": start_line,
                "message": "Empty or undeclared Mermaid diagram block.",
            }
        ]

    header_index, diagram_type = header
    sequence_stack: list[BlockLine] = []
    findings: list[Finding] = []
    for line, raw_line in block_lines[header_index + 1 :]:
        clean = raw_line.strip()
        if not clean or clean.startswith("%%"):
            continue
        findings.extend(
            _validate_content_line(
                diagram_type, line, clean, sequence_stack=sequence_stack
            )
        )
    findings.extend(
        {
            "line": line,
            "message": f"Unclosed sequence diagram block: '{block_type}' has no matching 'end'.",
        }
        for line, block_type in sequence_stack
    )
    return findings


def _validate_content_line(
    diagram_type: str, line: int, clean: str, *, sequence_stack: list[BlockLine]
) -> list[Finding]:
    findings: list[Finding] = []
    if clean.replace('\\"', "").count('"') % 2:
        findings.append(
            {
                "line": line,
                "message": "Mismatched double quotes on line (odd number of quotes).",
            }
        )
    if diagram_type in ("graph", "flowchart"):
        findings.extend(_validate_flow_shapes(line, clean))
        findings.extend(_validate_flow_arrow(line, clean))
    elif diagram_type == "sequencediagram":
        words = clean.split()
        first_word = words[0].lower() if words else ""
        findings.extend(_validate_sequence_structure(line, first_word, sequence_stack))
        findings.extend(_validate_sequence_arrow(line, clean, first_word))
    return findings


def check_file_for_mermaid(filepath: str) -> list[Finding]:
    """Scan a Markdown file for Mermaid code blocks."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as stream:
            lines = stream.readlines()
    except Exception as exc:
        return [{"line": 0, "message": f"Error reading file: {exc}"}]

    findings: list[Finding] = []
    in_block = False
    block_start_line = 0
    block_lines: list[BlockLine] = []
    for line_number, raw_line in enumerate(lines, 1):
        clean = raw_line.strip()
        if clean.startswith("```mermaid"):
            in_block = True
            block_start_line = line_number
            block_lines = []
        elif in_block and clean.startswith("```"):
            in_block = False
            findings.extend(
                validate_mermaid_block(filepath, block_start_line, block_lines)
            )
        elif in_block:
            block_lines.append((line_number, raw_line))
    if in_block:
        findings.append(
            {
                "line": block_start_line,
                "message": "Unclosed Mermaid code block (missing closing ```).",
            }
        )
    return findings


def _files_to_scan(files: Sequence[str], extra_excludes: Sequence[str]) -> list[str]:
    if files:
        return [path for path in files if path.endswith(".md") and os.path.exists(path)]
    excludes = DEFAULT_EXCLUDES.union(extra_excludes)
    paths: list[str] = []
    for root, dirs, names in os.walk(os.environ.get("AGENT_PACKAGES_ROOT", ".")):
        visible_excludes = excludes.union(fnmatch.filter(dirs, ".*"))
        dirs[:] = [name for name in dirs if name not in visible_excludes]
        paths.extend(os.path.join(root, name) for name in names if name.endswith(".md"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively scan and validate Mermaid diagrams across the workspace."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to scan. If none, scans the project recursively.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Directories/files to exclude from recursive scan.",
    )
    args = parser.parse_args()
    paths = _files_to_scan(args.files, args.exclude)
    total = 0
    report: dict[str, list[Finding]] = {}
    for path in paths:
        normalized = os.path.normpath(path)
        findings = check_file_for_mermaid(normalized)
        if findings:
            report[normalized] = findings
            total += len(findings)
    if total:
        print("\n" + "=" * 80)
        print(
            f"MERMAID DIAGRAM SYNTAX VERIFICATION FAILED: Found {total} syntax errors!"
        )
        print("=" * 80)
        for filepath, findings in report.items():
            print(f"\nFile: {filepath}")
            for finding in findings:
                print(f"  [Line {finding['line']}] - {finding['message']}")
        print("\n" + "=" * 80)
        print("Please correct all syntax errors in the Mermaid diagrams above.")
        print("=" * 80 + "\n")
        sys.exit(1)
    print(
        "MERMAID DIAGRAM SYNTAX VERIFICATION PASSED: All diagrams are syntactically valid!"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
