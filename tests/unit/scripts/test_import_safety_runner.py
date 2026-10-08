"""Import safety must execute against the bootstrapped repository environment."""

import shlex
from pathlib import Path

import yaml


def test_import_hook_uses_repository_interpreter_without_workspace_rebinding():
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / ".config/pre-commit.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "check-import-safety"
    )
    argv = shlex.split(hook["entry"])
    assert argv[:5] == [
        "scripts/hook_python.sh",
        "scripts/check_import_safety.py",
        "--package",
        "agent_utilities",
        "--simulate-windows",
    ]
    excludes = [
        "knowledge_graph.core.file_lock",
        "__main__",
        "agent.factory",
        "server",
        "mcp.toolset_factory",
        "mcp.tools",
        "mcp.verbose_tools",
        "patterns",
        "knowledge_graph.adaptation.trace_distiller",
        "cli",
        "core.unified_install",
        "governance.concept_allocator",
        "governance.lanes",
        "governance.merge_queue",
        "knowledge_graph.core.engine_lock",
        "knowledge_graph.core.host_lock",
        "mcp.eunomia_principal",
        "mcp.middlewares",
        "mcp.multiplexer",
        "mcp.shared_multiplexer",
        "mcp.kg_coordinator",
    ]
    assert argv[5:] == [
        part for name in excludes for part in ("--exclude", "agent_utilities." + name)
    ]
    assert hook["language"] == "system"
    assert hook["pass_filenames"] is False
