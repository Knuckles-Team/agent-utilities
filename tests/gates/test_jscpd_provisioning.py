"""The AU consumer preserves clone gates while using the reviewed provider."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts._gate_skip import find_local_tool

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "c0a089c83eea9d0d08f48e6c00681eb989268d9a"
ARCHIVE_SHA256 = "5521ddf30a7d8b09fd0a48fca1aa6251179567d1a56e76115d679516090f6d7a"


def _scanner_prefix(tmp_path):
    prefix = tmp_path / "home/.local"
    (prefix / "bin").mkdir(parents=True)
    dupehound = prefix / "bin/dupehound"
    dupehound.write_text("#!/bin/sh\necho 'dupehound 0.1.2'\n")
    dupehound.chmod(0o755)
    (prefix / "providers").mkdir()
    return prefix


def test_consumer_pins_merged_provider_and_reverifies_cached_source():
    installer = (ROOT / "scripts/install_scanners.sh").read_text()
    assert f'PIPELINES_REV="{PROVIDER}"' in installer
    assert f'PIPELINES_ARCHIVE_SHA256="{ARCHIVE_SHA256}"' in installer
    assert installer.index('"$provider_archive" | sha256sum --check') < installer.index(
        "tar -xzf"
    )
    assert 'install_jscpd.py" --root "$SCANNER_ROOT/jscpd"' in installer
    assert "JSCPD_BIN=$jscpd_bin_dir/jscpd" in installer
    assert "jscpd.provenance.json" in installer
    assert "npm install" not in installer


def test_workflow_keeps_exact_clone_range_and_gate_commands():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    jobs = workflow["jobs"]
    steps = {s.get("name"): s for s in jobs["clone-scanners"]["steps"]}
    install = steps["Install exact clone scanner toolchain"]["run"]
    assert "rustup default 1.95.0" in install
    assert "rustup toolchain install 1.97.0 --profile minimal" in install
    assert "if" not in steps["Install exact clone scanner toolchain"]
    assert jobs["publish-docker"]["uses"].endswith("@" + PROVIDER)
    for name, command in (
        (
            "Dupehound changed-function gate",
            'python3 scripts/check_dupehound.py --base-ref "$CX_DUP_BASE_REF"',
        ),
        (
            "jscpd changed-block gate",
            'python3 scripts/check_duplication.py enforce --base-ref "$CX_DUP_BASE_REF"',
        ),
    ):
        gate = steps[name]
        assert gate["env"] == {
            "CX_DUP_BASE_REF": "${{ steps.clone-range.outputs.base_sha }}",
            "CX_DUP_HEAD_REF": "${{ steps.clone-range.outputs.head_sha }}",
        }
        assert gate["run"].splitlines() == [
            "set -euo pipefail",
            'test "$(git rev-parse HEAD)" = "$CX_DUP_HEAD_REF"',
            command,
        ]
    assert (
        steps["jscpd full-tree census (advisory evidence)"]["run"]
        == "python3 scripts/check_duplication.py census"
    )


def test_corrupt_cached_provider_fails_before_download_or_execution(tmp_path):
    prefix = _scanner_prefix(tmp_path)
    (prefix / f"providers/{PROVIDER}.tar.gz").write_bytes(b"corrupt cached source")
    deny = tmp_path / "deny"
    deny.mkdir()
    marker = tmp_path / "unexpected-tool"
    for name in ("curl", "npm", "cargo", "rustup"):
        tool = deny / name
        tool.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 97\n')
        tool.chmod(0o755)
    env = {
        **os.environ,
        "SCANNER_ROOT": str(prefix),
        "PATH": f"{deny}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        [shutil.which("bash"), str(ROOT / "scripts/install_scanners.sh")],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "FAILED" in result.stderr + result.stdout
    assert not marker.exists()
    assert not list(prefix.glob("pipelines.*"))


@pytest.mark.parametrize("existing", ["absent", "file", "symlink"])
def test_local_install_exposes_verified_scanner(tmp_path, monkeypatch, existing):
    prefix = _scanner_prefix(tmp_path)
    verified = prefix / "jscpd/verified/bin/jscpd"
    verified.parent.mkdir(parents=True)
    verified.write_text("#!/bin/sh\necho 'cpd 5.0.16'\n")
    verified.chmod(0o755)
    verified.with_suffix(".provenance.json").write_text('{"fixture": true}\n')
    marker = tmp_path / "legacy-executed"
    if existing != "absent":
        legacy = prefix / "bin/jscpd" if existing == "file" else tmp_path / "legacy"
        legacy.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 97\n')
        legacy.chmod(0o755)
        if existing == "symlink":
            (prefix / "bin/jscpd").symlink_to(legacy)

    # Substitute only the external provider boundary, retaining archive checking
    # and the actual consumer installer. No scanner download or build is needed.
    provider = b'import sys\nprint(sys.argv[-1] + "/verified/bin")\n'
    archive = prefix / f"providers/{PROVIDER}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("provider/scripts/install_jscpd.py")
        member.size = len(provider)
        bundle.addfile(member, io.BytesIO(provider))
    fixture_repo = tmp_path / "repo"
    installer = fixture_repo / "scripts/install_scanners.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(
        (ROOT / "scripts/install_scanners.sh")
        .read_text()
        .replace(ARCHIVE_SHA256, hashlib.sha256(archive.read_bytes()).hexdigest())
    )
    shutil.copyfile(ROOT / "pyproject.toml", fixture_repo / "pyproject.toml")
    monkeypatch.setenv("HOME", str(prefix.parent))
    monkeypatch.setenv("SCANNER_ROOT", str(prefix))
    for name in ("GITHUB_ENV", "GITHUB_PATH", "JSCPD_BIN"):
        monkeypatch.delenv(name, raising=False)
    result = subprocess.run(
        [shutil.which("bash"), str(installer)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    selected = find_local_tool("jscpd")
    assert selected == str(prefix / "bin/jscpd")
    assert Path(selected).resolve() == verified
    assert (
        subprocess.check_output([selected, "--version"], text=True).strip()
        == "cpd 5.0.16"
    )
    assert not marker.exists()
