#!/usr/bin/env python3
import os
import re
import sys

# Rendered at the SAME indentation as whatever hook already sits under the
# target `hooks:` key (see `_hook_indent` / `_render_hook_block` below). This
# constant is the fallback used ONLY when `hooks:` has no existing `- id:`
# entry to measure against.
DEFAULT_HOOK_INDENT = "      "  # 6 spaces -- matches this fleet's established depth

HOOK_LINES = [
    "- id: check-mermaid",
    "name: Check Mermaid syntax",
    'entry: bash -c \'python3 "${AGENT_UTILITIES_REPO:?set AGENT_UTILITIES_REPO}/scripts/mermaid_linter.py" "$@"\' --',
    "language: system",
    "files: \\.md$",
    "pass_filenames: true",
]

NEW_REPO_BLOCK = """- repo: local
  hooks:
    - id: check-mermaid
      name: Check Mermaid syntax
      entry: bash -c 'python3 "${AGENT_UTILITIES_REPO:?set AGENT_UTILITIES_REPO}/scripts/mermaid_linter.py" "$@"' --
      language: system
      files: \\.md$
      pass_filenames: true
"""

# Matches the first `- id: <name>` hook entry (any leading whitespace) so the
# indentation of a NEWLY inserted hook can be matched to hooks already present
# under the same `hooks:` key -- this is the fix for INFRA-5 (RMDD bug ledger):
# the previous version hardcoded a 6/8-space block and inserted it immediately
# after `hooks:\n` with no regard for the indentation already used by sibling
# hooks (frequently 2/4 spaces, the `agent-package-builder` scaffold's depth),
# producing a `.pre-commit-config.yaml` that fails to parse as YAML at all.
_EXISTING_HOOK_RE = re.compile(r"^([ \t]*)-\s*id:\s*\S", re.MULTILINE)


def _hook_indent(content: str, hooks_idx: int) -> str:
    """Indentation of the first existing `- id:` hook after `hooks_idx`.

    Falls back to `DEFAULT_HOOK_INDENT` when the `hooks:` list is empty (no
    sibling hook to measure against) -- never a bare guess independent of the
    file being edited.
    """
    match = _EXISTING_HOOK_RE.search(content, hooks_idx)
    if match:
        return match.group(1)
    return DEFAULT_HOOK_INDENT


def _render_hook_block(indent: str) -> str:
    inner = indent + "  "
    lines = [f"{indent}{HOOK_LINES[0]}"]
    lines.extend(f"{inner}{line}" for line in HOOK_LINES[1:])
    return "\n".join(lines) + "\n"


def inject_hook_to_file(filepath):
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if "check-mermaid" in content:
        print(f"Already injected: {filepath}")
        return False

    # Try to find a repo: local block
    repo_local_idx = content.find("repo: local")
    if repo_local_idx != -1:
        # Find 'hooks:' line below 'repo: local'
        hooks_idx = content.find("hooks:", repo_local_idx)
        if hooks_idx != -1:
            # We want to insert after the 'hooks:\n' line
            newline_idx = content.find("\n", hooks_idx)
            if newline_idx != -1:
                # Match the indentation of a hook already under this `hooks:`
                # key (INFRA-5 fix) instead of a hardcoded depth.
                indent = _hook_indent(content, newline_idx + 1)
                hook_block = _render_hook_block(indent)
                # Insert the hook block right after 'hooks:\n'
                new_content = (
                    content[: newline_idx + 1] + hook_block + content[newline_idx + 1 :]
                )
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Successfully injected (existing repo: local): {filepath}")
                return True

    # If no 'repo: local' was found or no 'hooks:' under it, append a new 'repo: local' at the end
    # Check if we need to prepend a newline
    prefix = "\n" if not content.endswith("\n") else ""
    new_content = content + prefix + NEW_REPO_BLOCK
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully appended new repo: local block: {filepath}")
    return True


def main():
    scan_dir = os.environ.get("AGENT_PACKAGES_ROOT", "")
    if not scan_dir or not os.path.exists(scan_dir):
        print("AGENT_PACKAGES_ROOT must name an existing directory.")
        sys.exit(1)

    count = 0
    for root, dirs, files in os.walk(scan_dir):
        # Skip hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file == ".pre-commit-config.yaml":
                filepath = os.path.join(root, file)
                if inject_hook_to_file(filepath):
                    count += 1

    print(f"\nCompleted hook injection for {count} pre-commit configuration files.")


if __name__ == "__main__":
    main()
