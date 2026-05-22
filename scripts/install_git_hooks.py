#!/usr/bin/env python3
import os
import stat
from pathlib import Path

HOOK_CONTENT = """#!/bin/sh
# Post-commit hook to trigger KG ingestion of diffs
# CONCEPT:KG-3.0 - Continuous Ingestion

# Generate a unique temp file path in /tmp
PATCH_FILE="/tmp/kg_diff_$(date +%s)_$$.patch"

# Get the diff of the latest commit
git diff HEAD~1 HEAD > "$PATCH_FILE"

# Feed it to the submit_diff script
SCRIPT_PATH="$(git rev-parse --show-toplevel)/agent-packages/agent-utilities/scripts/submit_diff.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    # Fallback if we are in the agent-utilities repo itself
    SCRIPT_PATH="$(git rev-parse --show-toplevel)/scripts/submit_diff.py"
fi

uv run python "$SCRIPT_PATH" "$PATCH_FILE" || true

# Clean up the temporary patch file
rm -f "$PATCH_FILE"
"""



def install_hooks(workspace_path: Path):
    git_dir = workspace_path / ".git"
    if not git_dir.exists():
        print(f"Skipping {workspace_path}: Not a git repository.")
        return

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    post_commit_path = hooks_dir / "post-commit"
    post_commit_path.write_text(HOOK_CONTENT)

    # Make executable
    st = os.stat(post_commit_path)
    os.chmod(post_commit_path, st.st_mode | stat.S_IEXEC)
    print(f"Installed post-commit hook in {post_commit_path}")


def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    print(f"Scanning workspace root: {workspace_root}")
    for item in workspace_root.iterdir():
        if item.is_dir() and (item / ".git").exists():
            install_hooks(item)

    # Also check the agent-utilities repo itself
    install_hooks(workspace_root / "agent-packages" / "agent-utilities")


if __name__ == "__main__":
    main()
