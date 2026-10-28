"""Keep the retired certification authority and its imports out of Meristem."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.spec("AU-BOUNDARY-R002.3.5")
def test_deployment_certification_oidc_deleted():
    assert not (ROOT / "agent_utilities/deployment/certification_oidc.py").exists()
    release_contract = ROOT / "tests/gates/test_exact_local_release_contract.py"
    if release_contract.exists():
        assert "certification_oidc.py" not in release_contract.read_text(
            encoding="utf-8"
        )
    for path in (ROOT / "agent_utilities").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                names.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
            ):
                names = [node.args[0].value]
            else:
                continue
            assert not any("certification_oidc" in name.split(".") for name in names), (
                path
            )
