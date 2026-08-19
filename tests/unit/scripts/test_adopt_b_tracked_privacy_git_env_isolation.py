"""NE-044 sub-gate 2 acceptance (AU-ADOPT-B), later fixed under NE-059: does
``scripts/check_tracked_privacy.py`` run "with no ambient Git-state
dependency" the same way ``aad9ab52`` (BUG-180) fixed
``scripts/check_current_only_contract.py``'s ``_tracked_or_walked``?

Originally: NO. ``check_tracked_privacy.py``'s ``_git_file_names`` (the
function backing BOTH ``_tracked_artifacts`` and ``_runtime_source_artifacts``
-- i.e. the gate's entire tracked-file inventory) called
``subprocess.run(["git", "ls-files", "--cached", "--others",
"--exclude-standard"], cwd=root, ...)`` with NO explicit ``env=`` and NO call
to ``scripts/_git_subprocess_env.py::strip_inherited_git_repository_env()``
anywhere in the module -- unlike its sibling ``check_current_only_contract.py``,
which ``aad9ab52`` fixed by calling that exact primitive at import time
(scripts/check_current_only_contract.py:36,45). Two more call sites in
``derive_local_identifiers`` (``git rev-parse --git-common-dir`` and ``git
config --get user.name|email``) shared the same unguarded shape.

Real-world consequence, empirically confirmed live against the actual shared
``agent-utilities`` repo prior to the fix: pointing ``GIT_DIR``/
``GIT_WORK_TREE``/``GIT_INDEX_FILE`` at a DIFFERENT live worktree of the same
shared repo while
scanning from this one changed the raw ``git ls-files --cached --others
--exclude-standard`` inventory (4995 -> 5000 files) with zero source
changes -- the exact "gate returns a different verdict for the same tree
depending on invocation method, no code change in between" symptom
``check_tracked_privacy.py``'s own ``main()`` already documented as an
unexplained mystery under the name "D-ORC-53". This was its root cause.

NE-059 closed the gap: ``check_tracked_privacy.py`` now imports
``scripts/_git_subprocess_env.py`` and calls
``strip_inherited_git_repository_env()`` at module import time (the same
process-wide chokepoint ``aad9ab52`` established for
``check_current_only_contract.py``), *and* every one of its three ``git``
``subprocess.run`` call sites now also passes an explicit
``env=sanitized_git_env()``, so a call cannot regress silently by omitting
the strip precondition. This file's tests are now inverted from their
original "prove the defect" shape to "prove the fix holds": the structural
test asserts the primitive IS wired, and the known-bad-input test asserts a
poisoned ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` pointed at a decoy repo whose
``.git/info/exclude`` would otherwise hide the real tracked file no longer
has any effect -- the gate finds the same file whether the environment is
poisoned or clean. (Reverting the fix flips both back to red -- proven
manually before landing NE-059, not re-verified automatically here.)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRIVACY_SPEC = importlib.util.spec_from_file_location(
    "check_tracked_privacy_adopt_b", _REPO_ROOT / "scripts" / "check_tracked_privacy.py"
)
privacy = importlib.util.module_from_spec(_PRIVACY_SPEC)
assert _PRIVACY_SPEC is not None and _PRIVACY_SPEC.loader is not None
sys.modules[_PRIVACY_SPEC.name] = privacy
_PRIVACY_SPEC.loader.exec_module(privacy)

_CURRENT_ONLY_SPEC = importlib.util.spec_from_file_location(
    "check_current_only_contract_adopt_b",
    _REPO_ROOT / "scripts" / "check_current_only_contract.py",
)
current_only = importlib.util.module_from_spec(_CURRENT_ONLY_SPEC)
assert _CURRENT_ONLY_SPEC is not None and _CURRENT_ONLY_SPEC.loader is not None
sys.modules[_CURRENT_ONLY_SPEC.name] = current_only
_CURRENT_ONLY_SPEC.loader.exec_module(current_only)

ENV_MODULE = _REPO_ROOT / "scripts" / "_git_subprocess_env.py"
ENV_SPEC = importlib.util.spec_from_file_location(
    "_git_subprocess_env_adopt_b", ENV_MODULE
)
git_env = importlib.util.module_from_spec(ENV_SPEC)
assert ENV_SPEC is not None and ENV_SPEC.loader is not None
sys.modules[ENV_SPEC.name] = git_env
ENV_SPEC.loader.exec_module(git_env)


def test_privacy_gate_now_calls_the_shared_env_stripping_primitive() -> None:
    """Structural proof the asymmetry is closed (NE-059):
    ``check_current_only_contract.py`` (fixed by aad9ab52) and
    ``check_tracked_privacy.py`` (fixed by NE-059) both import and call
    ``strip_inherited_git_repository_env`` at module scope."""
    current_only_source = (
        _REPO_ROOT / "scripts" / "check_current_only_contract.py"
    ).read_text(encoding="utf-8")
    privacy_source = (_REPO_ROOT / "scripts" / "check_tracked_privacy.py").read_text(
        encoding="utf-8"
    )

    assert "strip_inherited_git_repository_env" in current_only_source
    assert "strip_inherited_git_repository_env" in privacy_source
    assert "_git_subprocess_env" in privacy_source
    # NE-059 went further than the module-level strip alone: every git
    # subprocess.run call site also carries its own explicit sanitized env=,
    # so no future call can regress silently by omitting the precondition.
    assert "sanitized_git_env" in privacy_source


def _make_synth_repo(tmp_path: Path, *, filename: str, content: str) -> Path:
    repo = tmp_path / "synth-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bug180@test.local"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "bug180"], cwd=repo, check=True)
    (repo / filename).write_text(content)
    subprocess.run(["git", "add", filename], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    return repo


def _make_decoy_repo_excluding(tmp_path: Path, *, pattern: str) -> Path:
    """A second, real git repo whose GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE, if inherited,
    would redirect a ``git ... `` call made with ``cwd`` pointed elsewhere --
    exactly the shape a real ``git commit`` produces for its hooks. Its own
    ``.git/info/exclude`` names ``pattern`` -- the dangerous half:
    ``--exclude-standard`` reads exclude rules from whatever ``GIT_DIR`` is
    active, not from the working tree being scanned, so a real tracked file
    can be silently excluded by a DIFFERENT repository's rules."""
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bug180@test.local"], cwd=decoy, check=True
    )
    subprocess.run(["git", "config", "user.name", "bug180"], cwd=decoy, check=True)
    (decoy / "unrelated.py").write_text("# nothing\n")
    subprocess.run(["git", "add", "unrelated.py"], cwd=decoy, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=decoy, check=True)
    (decoy / ".git" / "info" / "exclude").write_text(f"{pattern}\n")
    return decoy


def test_plain_invocation_finds_the_real_tracked_file() -> None:
    """Positive control: with no ambient poisoning, the privacy gate's own
    ``_git_file_names`` correctly finds the real tracked file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        synth = _make_synth_repo(
            tmp_path, filename="tracked_leak.py", content="# host.internal.example\n"
        )
        names = privacy._git_file_names(
            synth, ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
        )
    assert names == ["tracked_leak.py"]


def test_known_bad_input_poisoned_git_dir_no_longer_reaches_the_privacy_call_unguarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The known-bad input, now closed (NE-059): poisoned
    ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` naming a DIFFERENT repository whose
    ``.git/info/exclude`` happens to name the real file under test.
    ``check_tracked_privacy.py``'s ``_git_file_names`` now passes its own
    explicit ``env=sanitized_git_env()`` on every call, so the poisoned
    ambient env never reaches the subprocess at all -- neither
    ``tests/conftest.py``'s ``LeakedGitPointerEnvError`` backstop nor any
    other guard needs to intervene, and outside pytest (a real ``git commit``
    hook, no conftest loaded) the call is just as immune. The gate must find
    the same real tracked file whether the environment is poisoned or clean.
    """
    synth = _make_synth_repo(
        tmp_path, filename="tracked_leak.py", content="# host.internal.example\n"
    )
    decoy = _make_decoy_repo_excluding(tmp_path, pattern="tracked_leak.py")

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))

    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    # No pytest.raises: the fixed call must complete normally, not merely be
    # intercepted by the session-wide backstop.
    names = privacy._git_file_names(synth, command)
    assert names == ["tracked_leak.py"]


def test_the_same_known_bad_input_is_harmless_once_stripped_like_the_fixed_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive contrast: applying the SAME primitive
    ``check_current_only_contract.py`` already uses, before the call, makes
    the identical poisoned environment harmless -- proving the fix that
    exists for the sibling gate would also fix this one, and that
    ``check_tracked_privacy.py`` is simply missing the (already-available,
    already-proven) call, not missing a new mechanism."""
    synth = _make_synth_repo(
        tmp_path, filename="tracked_leak.py", content="# host.internal.example\n"
    )
    decoy = _make_decoy_repo_excluding(tmp_path, pattern="tracked_leak.py")

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))

    git_env.strip_inherited_git_repository_env()

    names = privacy._git_file_names(
        synth, ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    )
    assert names == ["tracked_leak.py"]
