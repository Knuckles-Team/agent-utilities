"""Exercise engine identity checks and the literal release workflow steps."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/release/verify_test_engine.py"
SPEC = importlib.util.spec_from_file_location("verify_test_engine", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
GATES = WORKFLOW["jobs"]["gates"]


@pytest.fixture
def wheel(tmp_path):
    path = tmp_path / "epistemic_graph-1.0-py3-none-any.whl"
    path.write_bytes(b"synthetic test wheel")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [None, "id", "name", "run", "source", "expired"])
def test_official_artifact_identity(tmp_path, mutation):
    data = {
        "id": 12,
        "name": "wheel",
        "expired": False,
        "workflow_run": {"id": 34, "head_sha": "a" * 40},
    }
    if mutation == "run":
        data["workflow_run"]["id"] = 35
    elif mutation == "source":
        data["workflow_run"]["head_sha"] = "b" * 40
    elif mutation:
        data[mutation] = True if mutation == "expired" else "wrong"
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(data))
    if mutation:
        with pytest.raises(ValueError):
            verifier.verify_artifact(path, 12, "wheel", 34, "a" * 40)
    else:
        verifier.verify_artifact(path, 12, "wheel", 34, "a" * 40)


@pytest.mark.parametrize("mutation", [None, "bytes", "missing", "name", "symlink"])
def test_wheel_identity(wheel, mutation):
    path, digest = wheel
    filename = path.name
    if mutation == "bytes":
        path.write_bytes(b"tampered")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "name":
        filename = "other.whl"
    elif mutation == "symlink":
        target = path.with_suffix(".original")
        path.rename(target)
        path.symlink_to(target)
    if mutation:
        with pytest.raises(ValueError):
            verifier.verify_wheel(path, filename, digest)
    else:
        verifier.verify_wheel(path, filename, digest)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_receipt",
        "url",
        "digest",
        "shadow",
        "url_fragment",
        "legacy_hash",
        "unhashed",
        "fragment_conflict",
        "receipt_conflict",
    ],
)
def test_importable_engine_still_requires_installed_identity(
    wheel, monkeypatch, mutation
):
    path, digest = wheel
    direct = {"url": path.as_uri(), "archive_info": {"hashes": {"sha256": digest}}}
    if mutation == "url":
        direct["url"] = "https://example.test/other.whl"
    elif mutation == "digest":
        direct["archive_info"]["hashes"]["sha256"] = "0" * 64
    elif mutation in {"url_fragment", "fragment_conflict", "receipt_conflict"}:
        direct["url"] += "#sha256=" + (
            "0" * 64 if mutation == "fragment_conflict" else digest
        )
        direct["archive_info"] = (
            {"hashes": {"sha256": "0" * 64}} if mutation == "receipt_conflict" else {}
        )
    elif mutation == "legacy_hash":
        direct["archive_info"] = {"hash": "sha256=" + digest}
    elif mutation == "unhashed":
        direct["archive_info"] = {}
    owned = path.parent / "numeric.py"
    distribution = SimpleNamespace(
        read_text=lambda _: (
            None if mutation == "missing_receipt" else json.dumps(direct)
        ),
        files=[owned],
        locate_file=lambda f: f,
    )
    monkeypatch.setattr(
        verifier.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(
        verifier.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            __file__=str(path if mutation == "shadow" else owned)
        ),
    )
    if mutation not in {None, "url_fragment", "legacy_hash"}:
        with pytest.raises(ValueError):
            verifier.verify_installed(path, digest)
    else:
        verifier.verify_installed(path, digest)


def _step(fragment):
    return next(s["run"] for s in GATES["steps"] if fragment in s.get("name", ""))


def _executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


def _run_literal_step(fragment, tmp_path, binaries):
    return subprocess.run(
        ["bash", "-c", _step(fragment)],
        cwd=tmp_path,
        env={
            **os.environ,
            **GATES["env"],
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_literal_rest_fallback_follows_redirect_even_with_importable_engine(tmp_path):
    scripts = tmp_path / "scripts/release"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # The previous import-only shortcut would return success without a download.
    _executable(
        binaries / "python",
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec {sys.executable} "$@"\n',
    )
    identity = {
        "id": int(GATES["env"]["EG_WHEEL_ARTIFACT_ID"]),
        "name": GATES["env"]["EG_WHEEL_ARTIFACT_NAME"],
        "expired": False,
        "workflow_run": {
            "id": int(GATES["env"]["EG_WHEEL_RUN_ID"]),
            "head_sha": GATES["env"]["EG_WHEEL_SOURCE"],
        },
    }
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    _executable(
        binaries / "gh",
        '#!/bin/sh\nif [ "$1" = api ]; then cat identity.json; else exit 1; fi\n',
    )
    _executable(
        binaries / "curl",
        f"""#!{sys.executable}
import os, sys, zipfile
from pathlib import Path
args = sys.argv[1:]
Path('curl-args.json').write_text(__import__('json').dumps(args))
if '--location' not in args:
    print('302', end='')
else:
    with zipfile.ZipFile(args[args.index('-o') + 1], 'w') as archive:
        archive.writestr(os.environ['EG_WHEEL_FILENAME'], b'synthetic')
    print('200', end='')
""",
    )
    env = {
        **os.environ,
        **GATES["env"],
        "GH_TOKEN": "synthetic",
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        ["bash", "-c", _step("Download frozen")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        tmp_path / "eg-wheel" / env["EG_WHEEL_FILENAME"]
    ).read_bytes() == b"synthetic"
    args = json.loads((tmp_path / "curl-args.json").read_text())
    assert args[args.index("--proto-redir") + 1] == "=https"
    assert "--location-trusted" not in args


def test_literal_checksum_rejects_wrong_bytes_despite_importable_engine(tmp_path):
    scripts = tmp_path / "scripts/release"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "python",
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec {sys.executable} "$@"\n',
    )
    (tmp_path / "eg-wheel").mkdir()
    (tmp_path / "eg-wheel" / GATES["env"]["EG_WHEEL_FILENAME"]).write_bytes(b"wrong")
    result = _run_literal_step("Verify downloaded", tmp_path, binaries)
    assert result.returncode != 0
    assert "sha256 mismatch" in result.stderr


def test_literal_install_binds_digest_even_with_importable_engine(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "python",
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec {sys.executable} "$@"\n',
    )
    _executable(
        binaries / "uv",
        f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath("install-args.json").write_text(json.dumps(sys.argv[1:]))\n',
    )
    result = _run_literal_step("Install the frozen", tmp_path, binaries)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads((tmp_path / "install-args.json").read_text())
    wheel = tmp_path / "eg-wheel" / GATES["env"]["EG_WHEEL_FILENAME"]
    assert (
        args[-1]
        == f"epistemic-graph[full] @ {wheel.as_uri()}#sha256={GATES['env']['EG_WHEEL_SHA256']}"
    )
    assert args[args.index("--reinstall-package") + 1] == "epistemic-graph"
    assert args[args.index("--python") + 1] == str(tmp_path / ".venv/bin/python")


def test_release_dependency_and_unconditional_install():
    assert "engine-release-order" in WORKFLOW["jobs"]["build"]["needs"]
    assert "check_eg_pypi_resolvable.py" in str(
        WORKFLOW["jobs"]["engine-release-order"]
    )
    source = _step("Install the frozen")
    assert "--reinstall-package epistemic-graph" in source
    assert "import epistemic_graph" not in source
    assert (
        ".venv/bin/python -I scripts/release/verify_test_engine.py installed"
        in _step("Assert the real")
    )
