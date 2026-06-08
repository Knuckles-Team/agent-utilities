#!/usr/bin/env python3
import os
import re
import sys

# Standardized block with 2 spaces of indentation for list items
STANDARD_BLOCK = """  - id: check-mermaid
    name: Check Mermaid syntax
    entry: python3 /home/apps/workspace/agent-packages/agent-utilities/scripts/mermaid_linter.py
    language: system
    files: \\.md$
    pass_filenames: true"""


def fix_hook_in_file(filepath):
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if "check-mermaid" not in content:
        return False

    # Regex to match check-mermaid hook block with any indentation
    pattern = re.compile(
        r"^\s*-\s*id:\s*check-mermaid\n"
        r"^\s*name:\s*Check Mermaid syntax\n"
        r"^\s*entry:\s*python3 /home/apps/workspace/agent-packages/agent-utilities/scripts/mermaid_linter\.py\n"
        r"^\s*language:\s*system\n"
        r"^\s*files:\s*\\\.md\$\n"
        r"^\s*pass_filenames:\s*true\n?",
        re.MULTILINE,
    )

    new_content, count = pattern.subn(STANDARD_BLOCK + "\n", content)
    if count > 0:
        # Also let's clean up any double 'hooks:' indentation inconsistencies or trailing whitespace
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully fixed indentation in: {filepath}")
        return True

    return False


def main():
    scan_dir = "/home/apps/workspace/agent-packages"
    if not os.path.exists(scan_dir):
        print(f"Directory {scan_dir} does not exist.")
        sys.exit(1)

    count = 0
    for root, dirs, files in os.walk(scan_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file == ".pre-commit-config.yaml":
                filepath = os.path.join(root, file)
                if fix_hook_in_file(filepath):
                    count += 1

    print(f"\nSuccessfully standardized {count} pre-commit configuration files.")


if __name__ == "__main__":
    main()
