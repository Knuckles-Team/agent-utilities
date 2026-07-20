#!/usr/bin/env python3
import os
import re
import sys

# Standardized block with 2 spaces of indentation for list items
STANDARD_BLOCK = """  - id: check-mermaid
    name: Check Mermaid syntax
    entry: bash -c 'python3 "${AGENT_UTILITIES_REPO:?set AGENT_UTILITIES_REPO}/scripts/mermaid_linter.py" "$@"' --
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
        r"^\s*entry:.*mermaid_linter\.py.*\n"
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
    scan_dir = os.environ.get("AGENT_PACKAGES_ROOT", "")
    if not scan_dir or not os.path.exists(scan_dir):
        print("AGENT_PACKAGES_ROOT must name an existing directory.")
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
