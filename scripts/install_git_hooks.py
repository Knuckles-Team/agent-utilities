#!/usr/bin/env python3
import os
from pathlib import Path


def uninstall_hooks(workspace_path: Path):
    git_dir = workspace_path / ".git"
    if not git_dir.exists():
        return

    post_commit_path = git_dir / "hooks" / "post-commit"
    if post_commit_path.exists():
        try:
            post_commit_path.unlink()
            print(f"Removed post-commit hook from {post_commit_path}")
        except Exception as e:
            print(f"Error removing hook from {post_commit_path}: {e}")


def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    print(f"Scanning workspace root for hooks cleanup: {workspace_root}")
    for item in workspace_root.iterdir():
        if item.is_dir() and (item / ".git").exists():
            uninstall_hooks(item)

    # Also check the agent-utilities repo itself
    uninstall_hooks(workspace_root / "agent-packages" / "agent-utilities")


if __name__ == "__main__":
    main()

