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
    ]
    assert argv[5:] == [
        part for name in excludes for part in ("--exclude", "agent_utilities." + name)
    ]
    assert hook["language"] == "system"
    assert hook["pass_filenames"] is False


def test_simulation_exclusions_apply_to_import_origin_not_cached_module(tmp_path):
    """An excluded platform adapter cannot exempt imports in its consumers."""
    import os
    import subprocess
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip(
            "POSIX-host simulation regression; native Windows uses real imports"
        )
    root = Path(__file__).resolve().parents[3]
    package = tmp_path / "sample_imports"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "adapter.py").write_text(
        "import sys\nif sys.platform != 'win32':\n    import fcntl\n"
    )
    consumer = package / "consumer.py"
    consumer.write_text("from . import adapter\n")
    command = [
        sys.executable,
        str(root / "scripts/check_import_safety.py"),
        "--package",
        "sample_imports",
        "--simulate-windows",
        "--exclude",
        "sample_imports.adapter",
    ]
    env = dict(os.environ, PYTHONPATH=str(tmp_path))

    def run():
        return subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
        )

    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    for statement in ("import fcntl", "from fcntl import flock"):
        consumer.write_text("from . import adapter\n" + statement + "\n")
        result = run()
        assert result.returncode == 1
        assert "sample_imports.consumer: ModuleNotFoundError" in result.stdout
    consumer.write_text("from . import adapter\n")
    (package / "adapter_sibling.py").write_text("import fcntl\n")
    result = run()
    assert result.returncode == 1
    assert "sample_imports.adapter_sibling: ModuleNotFoundError" in result.stdout
    (package / "adapter_sibling.py").unlink()
    (package / "helper.py").write_text("import fcntl\n")
    (package / "adapter.py").write_text("from . import helper\n")
    result = run()
    assert result.returncode == 1
    assert "sample_imports.consumer: ModuleNotFoundError" in result.stdout


def test_simulation_uses_only_immediate_valid_origin(monkeypatch):
    import builtins
    import importlib.util
    import sys

    import pytest

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "import_safety_origin_test", root / "scripts/check_import_safety.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    native_import = builtins.__import__
    platform = sys.platform
    # Register restoration before installing the process-local hook.
    monkeypatch.setattr(builtins, "__import__", native_import)
    checker.install_windows_simulation(
        blocked=("au_missing_platform_module",), excluded=("sample.adapter",)
    )
    for origin in (
        None,
        {},
        {"__name__": None},
        {"__name__": 12},
        {"__name__": "sample.adapter_sibling"},
        {"__name__": "sample.nonexcluded_helper"},
    ):
        with pytest.raises(ModuleNotFoundError, match="simulated"):
            builtins.__import__("au_missing_platform_module", origin)
    for origin in ("sample.adapter", "sample.adapter.child"):
        with pytest.raises(ModuleNotFoundError) as failure:
            builtins.__import__("au_missing_platform_module", {"__name__": origin})
        assert "simulated" not in str(failure.value)
    assert sys.platform == platform
