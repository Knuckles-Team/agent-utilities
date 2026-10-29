"""Check the historical evidence guard against a hermetic Git history."""

import subprocess
from pathlib import Path

import pytest

from scripts.check_boundary_history import (
    BACKENDS,
    BASE,
    OIDC,
    check_deployment,
    check_toolkit,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test Author")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    for name in (BACKENDS, OIDC, BASE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial\n")
    _git(tmp_path, "add", BACKENDS, OIDC, BASE)
    _git(tmp_path, "commit", "-qm", "original")
    original = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/gate.py").write_text("pass\n")
    _git(tmp_path, "add", "scripts/gate.py")
    _git(tmp_path, "commit", "-qm", "gate")
    gate = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "rm", "-q", BACKENDS)
    _git(tmp_path, "commit", "-qm", "remove backends")
    backends_cut = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "rm", "-q", OIDC, BASE)
    _git(tmp_path, "commit", "-qm", "remove oidc and base")
    cut = _git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, original, gate, backends_cut, cut


@pytest.mark.spec("AU-BOUNDARY-R002.1")
def test_deployment_history_requires_original_and_later_deletions(
    history: tuple[Path, str, str, str, str],
) -> None:
    root, original, _, backends_cut, cut = history
    check_deployment(root, original, backends_cut, cut, "HEAD")
    with pytest.raises(ValueError, match="original deployment evidence"):
        check_deployment(root, backends_cut, backends_cut, cut, "HEAD")
    with pytest.raises(ValueError, match="deployment deletion evidence"):
        check_deployment(root, original, backends_cut, backends_cut, "HEAD")


@pytest.mark.spec("AU-BOUNDARY-R010.1")
def test_toolkit_history_requires_gate_only_change_and_later_deletion(
    history: tuple[Path, str, str, str, str],
) -> None:
    root, original, gate, _, cut = history
    check_toolkit(root, gate, cut, "HEAD")
    with pytest.raises(ValueError, match="gate-only"):
        check_toolkit(root, original, cut, "HEAD")
    with pytest.raises(ValueError, match="toolkit history evidence"):
        check_toolkit(root, gate, gate, "HEAD")


@pytest.mark.spec("AU-BOUNDARY-R005.1")
def test_a_e_census_attribution_is_historical_and_names_exclusions(
    fleet_history: tuple[Path, str, str, str],
) -> None:
    from scripts.check_boundary_history import check_fleet_census

    root, census, merge, wrong = fleet_history
    check_fleet_census(root, census, merge, wrong, "HEAD")
    with pytest.raises(ValueError, match="status attribution"):
        check_fleet_census(root, census, merge, census, "HEAD")


@pytest.fixture
def fleet_history(tmp_path: Path) -> tuple[Path, str, str, str]:
    import json

    from scripts.check_boundary_history import FLEET_ARTIFACT, FLEET_STATUS

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test Author")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "initial")
    harness = tmp_path / "agent_utilities/harness/generative_model_policy.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("pass\n")
    _git(tmp_path, "add", str(harness.relative_to(tmp_path)))
    _git(tmp_path, "commit", "-qm", "unrelated harness change")
    wrong = _git(tmp_path, "rev-parse", "HEAD")
    artifact = tmp_path / FLEET_ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "R005 (connector packages A–E)\nR007 (connector packages K–O)\nR008 (connector packages P–S)\nsanctioned SDK-gap\n"
    )
    rows = [
        {"id": target, "delivery_state": "LANDED", "evidence": [{"commit": wrong}]}
        for target in ("AU-BOUNDARY-R005", "AU-BOUNDARY-R007", "AU-BOUNDARY-R008")
    ]
    (tmp_path / FLEET_STATUS).write_text(json.dumps({"requirements": rows}))
    _git(tmp_path, "add", FLEET_ARTIFACT, FLEET_STATUS)
    _git(tmp_path, "commit", "-qm", "record census")
    census = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "merge census")
    merge = _git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, census, merge, wrong


@pytest.mark.spec("AU-BOUNDARY-R007.1")
def test_k_o_census_attribution_is_historical(
    fleet_history: tuple[Path, str, str, str],
) -> None:
    from scripts.check_boundary_history import check_fleet_census

    root, census, merge, wrong = fleet_history
    check_fleet_census(root, census, merge, wrong, "HEAD", target="AU-BOUNDARY-R007")
    with pytest.raises(ValueError, match="status attribution"):
        check_fleet_census(
            root, census, merge, census, "HEAD", target="AU-BOUNDARY-R007"
        )


@pytest.mark.spec("AU-BOUNDARY-R008.1")
def test_p_s_census_attribution_is_historical(
    fleet_history: tuple[Path, str, str, str],
) -> None:
    from scripts.check_boundary_history import check_fleet_census

    root, census, merge, wrong = fleet_history
    check_fleet_census(root, census, merge, wrong, "HEAD", target="AU-BOUNDARY-R008")
    with pytest.raises(ValueError, match="status attribution"):
        check_fleet_census(
            root, census, merge, census, "HEAD", target="AU-BOUNDARY-R008"
        )
