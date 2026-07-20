#!/usr/bin/env python3
import os
import sys

HOOK_BLOCK = """      - id: check-mermaid
        name: Check Mermaid syntax
        entry: bash -c 'python3 "${AGENT_UTILITIES_REPO:?set AGENT_UTILITIES_REPO}/scripts/mermaid_linter.py" "$@"' --
        language: system
        files: \\.md$
        pass_filenames: true
"""

NEW_REPO_BLOCK = """- repo: local
  hooks:
    - id: check-mermaid
      name: Check Mermaid syntax
      entry: bash -c 'python3 "${AGENT_UTILITIES_REPO:?set AGENT_UTILITIES_REPO}/scripts/mermaid_linter.py" "$@"' --
      language: system
      files: \\.md$
      pass_filenames: true
"""


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
                # Insert the HOOK_BLOCK right after 'hooks:\n'
                new_content = (
                    content[: newline_idx + 1] + HOOK_BLOCK + content[newline_idx + 1 :]
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
