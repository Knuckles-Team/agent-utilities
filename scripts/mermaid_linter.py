#!/usr/bin/env python3
import argparse
import os
import sys
import re
from pathlib import Path

# Common shape patterns in flowchart/graph diagrams:
# e.g., id[label], id(label), id{label}, id((label)), id([label]), id>label]
# This pattern matches: id followed by shape brackets with unquoted contents containing special characters
UNQUOTED_SHAPE_CONTENT_RE = re.compile(
    r'\b[a-zA-Z0-9_-]+\s*(?:\[(?!\")|\((?!\")|\{(?!\")|\(\[(?!\")|\(\((?!\")|\{\{(?!\")|\[\((?!\")|\[\[(?!\")|>\s*(?!\"))(?P<label>.*?)(?:\]|\)|\}|\)\]|\)\)|\}\}|\)\]|\]\))$'
)

# Common special characters that are illegal in unquoted node labels:
ILLEGAL_UNQUOTED_CHARS = ['(', ')', '[', ']', '{', '}', '<', '>', '&', ';', ',']

def validate_mermaid_block(filepath, start_line, block_lines):
    """
    Validates a single Mermaid code block and returns a list of findings (line, message).
    """
    findings = []
    if not block_lines:
        return findings

    # Find the diagram type, skipping initialization blocks (e.g. %%{init: ...}%%) and comments
    diagram_type = None
    diag_line_num = start_line
    header_idx = -1

    for idx, (ln, l) in enumerate(block_lines):
        l_clean = l.strip()
        if not l_clean:
            continue
        if l_clean.startswith("%%"):
            continue
        words = l_clean.split()
        if words:
            diagram_type = words[0].lower()
            diag_line_num = ln
            header_idx = idx
            break

    if not diagram_type:
        findings.append({
            "line": start_line,
            "message": "Empty or undeclared Mermaid diagram block."
        })
        return findings

    # Validate lines starting after the header
    active_lines = block_lines[header_idx + 1:] if header_idx != -1 else block_lines

    # Stack for sequence diagrams block nesting
    seq_stack = []
    
    # Simple brackets balance check for non-mindmap diagrams
    is_mindmap = diagram_type == "mindmap"

    for ln, l in active_lines:
        l_clean = l.strip()
        if not l_clean or l_clean.startswith("%%"):
            continue

        # 1. Check double quote parity (ignoring escaped quotes)
        unescaped = l_clean.replace('\\"', '')
        if unescaped.count('"') % 2 != 0:
            findings.append({
                "line": ln,
                "message": "Mismatched double quotes on line (odd number of quotes)."
            })

        # 2. Check for flowchart/graph node shape errors
        if diagram_type in ("graph", "flowchart"):
            # Check for unquoted special characters in node definitions.
            # We use a regex that matches individual flowchart node definitions on a line.
            shape_pattern = re.compile(
                r'\b[a-zA-Z0-9_-]+\s*(?:'
                # 1. Double-quoted labels first (longer shapes first to prevent partial matching)
                r'\(\[(".*?")\]\)|'     # Stadium ([ "..." ])
                r'\[\[(".*?")\]\]|'     # Subroutine [[ "..." ]]
                r'\[\((".*?")\)\]|'     # Cylindrical [( "..." )]
                r'\(\((".*?")\)\)|'     # Circle (( "..." ))
                r'\{\{(".*?")\}\}|'     # Hexagon {{ "..." }}
                r'>\s*(".*?")\]|'       # Asymmetric > "..." ]
                r'\[(".*?")\]|'         # Rectangle [ "..." ]
                r'\((".*?")\)|'         # Round ( "..." )
                r'\{(".*?")\}|'         # Decision { "..." }
                # 2. Unquoted labels fallback (longer shapes first)
                r'\(\[(.*?)\]\)|'       # Stadium ([ ... ])
                r'\[\[(.*?)\]\]|'       # Subroutine [[ ... ]]
                r'\[\((.*?)\)\]|'       # Cylindrical [( ... )]
                r'\(\((.*?)\)\)|'       # Circle (( ... ))
                r'\{\{(.*?)\}\}|'       # Hexagon {{ ... }}
                r'>\s*(.*?)\]|'         # Asymmetric > ... ]
                r'\[(.*?)\]|'           # Rectangle [ ... ]
                r'\((.*?)\)|'           # Round ( ... )
                r'\{(.*?)\}'            # Decision { ... }
                r')'
            )
            matches = shape_pattern.findall(l_clean)
            for m in matches:
                label = next((item for item in m if item), "").strip()
                if label:
                    if label.startswith('"') and label.endswith('"'):
                        continue
                    
                    violations = [c for c in ILLEGAL_UNQUOTED_CHARS if c in label]
                    if violations:
                        findings.append({
                            "line": ln,
                            "message": f"Unquoted special character(s) {violations} in node label '{label}'. Enclose the label in double quotes, e.g. A(\"Label\")"
                        })

            # 3. Check for invalid arrow connections or labels in flowcharts
            # Common error: A -- |label| --> B (which is invalid, correct is A -->|label| B or A -- label --> B)
            if "-- |" in l_clean or "--  |" in l_clean:
                findings.append({
                    "line": ln,
                    "message": "Invalid arrow label syntax. Use '-->|label|' or '-- label -->' instead of '-- |label| -->'."
                })

        # 4. Check sequence diagram block nesting
        elif diagram_type == "sequencediagram":
            words = l_clean.split()
            first_word = words[0].lower() if words else ""
            
            if first_word in ("alt", "loop", "rect", "opt", "par", "critical", "break"):
                seq_stack.append((ln, first_word))
            elif first_word == "else":
                if not seq_stack or seq_stack[-1][1] not in ("alt", "critical"):
                    findings.append({
                        "line": ln,
                        "message": "Found 'else' statement without a matching active 'alt' or 'critical' block."
                    })
            elif first_word == "end":
                if not seq_stack:
                    findings.append({
                        "line": ln,
                        "message": "Found 'end' statement without a matching opening block (alt, loop, rect, opt, par, etc.)."
                    })
                else:
                    seq_stack.pop()

            # Check sequence diagram arrow styles
            # Valid sequence diagram arrows are: ->, -->, ->>, -->>, -x, --x, -), --)
            # A common mistake is spaces inside the arrow like "- ->" or using invalid ones like "->>" or "-->>"
            # sequenceDiagram does not support three-headed or custom flowchart arrows.
            # We can check for arrow-like patterns on lines that are messages (not participant/note/etc)
            if not first_word in ("participant", "actor", "note", "autonumber", "activate", "deactivate", "alt", "else", "end", "loop", "rect", "opt", "par", "critical", "break"):
                # If there's an arrow-like sequence but it's invalid, warn
                # Sequence lines are usually: A->B: Message or A -> B : Message
                # Let's match potential communication lines
                if ":" in l_clean:
                    parts = l_clean.split(":", 1)
                    sender_receiver = parts[0].strip()
                    # Look for arrow in sender_receiver
                    # Common arrows: ->, -->, ->>, -->>, -x, --x, -), --)
                    # If there's a sequence of hyphens/arrows that doesn't match these:
                    arrow_match = re.search(r'(-->>|->>|-->|->|--x|-x|--\)|-\))', sender_receiver)
                    if arrow_match:
                        arrow = arrow_match.group(1)
                        if arrow not in ("->", "-->", "->>", "-->>", "-x", "--x", "-)", "--)"):
                            findings.append({
                                "line": ln,
                                "message": f"Potential invalid arrow syntax '{arrow}' in sequence diagram. Use standard arrows: ->, -->, ->>, -->>, etc."
                            })

    # Unclosed blocks at the end of a sequence diagram
    for ln, block_type in seq_stack:
        findings.append({
            "line": ln,
            "message": f"Unclosed sequence diagram block: '{block_type}' has no matching 'end'."
        })

    return findings

def check_file_for_mermaid(filepath):
    """
    Scans a markdown file for Mermaid code blocks and validates them.
    """
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return [{
            "line": 0,
            "message": f"Error reading file: {e}"
        }]

    in_block = False
    block_start_line = 0
    block_lines = []

    for i, line in enumerate(lines, 1):
        clean_line = line.strip()
        if clean_line.startswith("```mermaid"):
            in_block = True
            block_start_line = i
            block_lines = []
            continue
        elif in_block and clean_line.startswith("```"):
            in_block = False
            findings.extend(validate_mermaid_block(filepath, block_start_line, block_lines))
            continue

        if in_block:
            block_lines.append((i, line))

    if in_block:
        findings.append({
            "line": block_start_line,
            "message": "Unclosed Mermaid code block (missing closing ```)."
        })

    return findings

def main():
    parser = argparse.ArgumentParser(description="Recursively scan and validate Mermaid diagrams across the workspace.")
    parser.add_argument("files", nargs="*", help="Specific files to scan. If none, scans the project recursively.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Directories/files to exclude from recursive scan.")
    args = parser.parse_args()

    files_to_scan = []
    default_excludes = {".venv", ".git", ".mypy_cache", ".pytest_cache", "build", "dist", "node_modules", "__pycache__"}
    excludes = default_excludes.union(set(args.exclude))

    if args.files:
        for f in args.files:
            if f.endswith(".md") and os.path.exists(f):
                files_to_scan.append(f)
    else:
        # Scan recursively under current working directory or /home/apps/workspace/agent-packages/
        scan_dir = "."
        if os.path.exists("/home/apps/workspace/agent-packages"):
            scan_dir = "/home/apps/workspace/agent-packages"
            
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in excludes and not d.startswith(".")]
            for file in files:
                if file.endswith(".md"):
                    files_to_scan.append(os.path.join(root, file))

    total_violations = 0
    report = {}

    for filepath in files_to_scan:
        filepath = os.path.normpath(filepath)
        findings = check_file_for_mermaid(filepath)
        if findings:
            report[filepath] = findings
            total_violations += len(findings)

    if total_violations > 0:
        print("\n" + "=" * 80)
        print(f"MERMAID DIAGRAM SYNTAX VERIFICATION FAILED: Found {total_violations} syntax errors!")
        print("=" * 80)
        for fp, findings in report.items():
            print(f"\nFile: {fp}")
            for f in findings:
                print(f"  [Line {f['line']}] - {f['message']}")
        print("\n" + "=" * 80)
        print("Please correct all syntax errors in the Mermaid diagrams above.")
        print("=" * 80 + "\n")
        sys.exit(1)
    else:
        print("MERMAID DIAGRAM SYNTAX VERIFICATION PASSED: All diagrams are syntactically valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()
