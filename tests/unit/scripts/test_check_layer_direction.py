"""Focused tests for AU's layer census and product dependency gate."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from scripts import layer_direction


def _census() -> ModuleType:
    return layer_direction


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    package = tmp_path / "fixture_pkg"
    for relative, source in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package


def test_longest_prefix_classifies_deliberate_config_contract() -> None:
    census = _census()

    assert census.classify("core.config") == ("core.config", "contracts")
    assert census.classify("core.config.models") == ("core.config", "contracts")
    assert ("core.config", "composition") in census.LAYER_RULES
    assert ("core.config", "contracts") in census.CLASSIFICATION_EXCEPTIONS
    assert census.classify("server.app") == ("server.app", "composition")
    assert census.classify("server.worker") == ("server", "adapters")
    assert census.classify("protocols.source_connectors.sql") == (
        "protocols.source_connectors",
        "ports",
    )


def test_parser_treats_config_import_as_inward_contract_dependency(
    tmp_path: Path,
) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "core/config.py": "def setting(name: str) -> str: return name\n",
            "models/user.py": (
                "from fixture_pkg.core.config import setting\n"
                "VALUE = setting('EXAMPLE')\n"
            ),
        },
    )

    violations, boundaries, _, _ = census.scan(package)

    assert violations == []
    assert boundaries == []


def test_parser_resolves_specific_composition_submodule(
    tmp_path: Path,
) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "server/app.py": "def build() -> None: pass\n",
            "security/policy.py": "from fixture_pkg.server import app\n",
        },
    )

    violations, boundaries, _, _ = census.scan(package)

    assert len(violations) == 1
    assert boundaries == []
    assert violations[0].kind == "outward-import"
    assert violations[0].pair == "security -> server.app"
    assert violations[0].tgt_layer == "composition"


def test_valid_scan_is_report_only_even_with_violations(tmp_path: Path, capsys) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "server/app.py": "def build() -> None: pass\n",
            "models/user.py": "from fixture_pkg.server import app\n",
        },
    )

    assert census.main(["--list", str(package)]) == 0
    output = capsys.readouterr().out
    assert "layer-direction violations: 1 (report-only)" in output
    assert "classification exception: core.config -> contracts" in output
    assert "fixture_pkg/models/user.py:1: outward-import" in output
    assert "single-contract boundary violations: 0 (blocking)" in output
    assert "accepted AU product dependency direction is intact" in output


def test_type_checking_else_import_remains_eager() -> None:
    census = _census()
    tree = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import fixture_pkg.models\n"
        "else:\n"
        "    import fixture_pkg.server\n"
    )
    walker = census.ImportWalker()
    walker.visit(tree)

    fixture_imports = [
        eager
        for node, eager in walker.found
        if isinstance(node, ast.Import) and node.names[0].name.startswith("fixture_pkg")
    ]
    assert fixture_imports == [False, True]


def test_parse_failure_marks_census_incomplete(tmp_path: Path, capsys) -> None:
    census = _census()
    package = _package(tmp_path, {"models/broken.py": "def broken(:\n"})

    assert census.main([str(package)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "CANNOT RUN" in captured.err
    assert "models/broken.py" in captured.err


def test_relative_adapter_crosstalk_and_unassigned_edges(tmp_path: Path) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "__init__.py": "from . import workflows\n",
            "gateway/api.py": "from ..mcp import tools\n",
            "mcp/tools.py": "TOOLS = ()\n",
            "models/user.py": "NAME = 'user'\n",
            "workflows/task.py": "import fixture_pkg.models.user\n",
        },
    )

    violations, boundaries, unassigned_files, unassigned_edges = census.scan(package)

    assert [(item.kind, item.pair) for item in violations] == [
        ("adapter-crosstalk", "gateway -> mcp")
    ]
    assert boundaries == []
    assert unassigned_files["<package-root>"] == 1
    assert unassigned_files["workflows"] == 1
    assert unassigned_edges["<package-root> -> workflows"] == 1
    assert unassigned_edges["workflows -> models"] == 1


def test_duplicate_aliases_count_one_target_statement(tmp_path: Path) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "server/app.py": "def build() -> None: pass\n",
            "models/user.py": "from fixture_pkg.server import app, app as other\n",
        },
    )

    violations, boundaries, _, _ = census.scan(package)

    assert len(violations) == 1
    assert boundaries == []
    assert violations[0].pair == "models -> server.app"


def test_missing_root_is_operational_error(tmp_path: Path, capsys) -> None:
    census = _census()

    assert census.main([str(tmp_path / "missing")]) == 2
    assert "CANNOT RUN" in capsys.readouterr().err


def test_single_contract_boundary_blocks_graphos_and_dynamic_probe(
    tmp_path: Path, capsys
) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "orchestration/task.py": (
                "import graph_os\n"
                "from importlib import import_module as load_module\n"
                "VALUE = load_module('graph_os.runtime')\n"
            )
        },
    )

    assert census.main([str(package)]) == 1
    output = capsys.readouterr().out
    assert "single-contract boundary violations: 2 (blocking)" in output
    assert "graphos-composition-import: graph_os" in output
    assert "graphos-composition-import: graph_os.runtime" in output


def test_single_contract_boundary_allows_sdk_root_and_public_eg_modules(
    tmp_path: Path,
) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "orchestration/task.py": (
                "import agent_connector_sdk\n"
                "from epistemic_graph.client import EpistemicGraphClient\n"
                "from epistemic_graph.generated.storage import SourceIngestRequest\n"
                "from epistemic_graph import parser\n"
            )
        },
    )

    _, boundaries, _, _ = census.scan(package)

    assert boundaries == []


def test_single_contract_boundary_blocks_sdk_and_eg_implementation_modules(
    tmp_path: Path,
) -> None:
    census = _census()
    package = _package(
        tmp_path,
        {
            "orchestration/task.py": (
                "from agent_connector_sdk import runner\n"
                "from epistemic_graph.server import serve\n"
                "import epistemic_graph._wire\n"
            )
        },
    )

    _, boundaries, _, _ = census.scan(package)

    assert [(item.reason, item.target) for item in boundaries] == [
        ("connector-sdk-implementation-import", "agent_connector_sdk.runner"),
        ("epistemic-graph-internal-import", "epistemic_graph.server"),
        ("epistemic-graph-private-import", "epistemic_graph._wire"),
    ]
