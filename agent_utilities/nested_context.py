from pathlib import Path


def get_nested_context(
    target_dir: str | Path, workspace_root: str | Path | None = None
) -> str:
    """
    Recursively aggregates context and instructions (AGENTS.md, CONTEXT.md, INSTRUCTIONS.md) from the target
    directory up to the workspace root, enabling subfolder-level overriding.

    CONCEPT:CTX-1.0: Nested Subfolder Instructions
    """
    target_dir = Path(target_dir).resolve()

    if not target_dir.is_dir():
        target_dir = target_dir.parent

    # If no root is provided, stop at the user's home directory to prevent unbounded climbing
    if workspace_root is None:
        workspace_root = Path.home()
    else:
        workspace_root = Path(workspace_root).resolve()

    context_files = [
        ".specify/AGENTS.md",
        "AGENTS.md",
        "CONTEXT.md",
        "INSTRUCTIONS.md",
        "SKILL.md",
    ]
    aggregated_context = []

    current_dir = target_dir
    # Climb up and collect context files
    # We collect from root downwards to let inner folders override/append properly.
    collected_paths = []
    while current_dir != workspace_root.parent and str(current_dir) != "/":
        for cf in context_files:
            cf_path = current_dir / cf
            if cf_path.exists() and cf_path.is_file():
                collected_paths.append(cf_path)

        if current_dir == workspace_root:
            break

        current_dir = current_dir.parent

    # Reverse so root applies first, then subfolders append
    collected_paths.reverse()

    for cf_path in collected_paths:
        try:
            with open(cf_path, encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    aggregated_context.append(f"--- Context from {cf_path} ---")
                    aggregated_context.append(content)
        except Exception:  # nosec B110
            pass

    return "\n\n".join(aggregated_context)
