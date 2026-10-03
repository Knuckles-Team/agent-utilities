"""The AU consumer preserves clone gates while using the reviewed provider."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "c0a089c83eea9d0d08f48e6c00681eb989268d9a"
ARCHIVE_SHA256 = "5521ddf30a7d8b09fd0a48fca1aa6251179567d1a56e76115d679516090f6d7a"


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
    prefix = tmp_path / "scanners"
    (prefix / "bin").mkdir(parents=True)
    dupehound = prefix / "bin/dupehound"
    dupehound.write_text("#!/bin/sh\necho 'dupehound 0.1.2'\n")
    dupehound.chmod(0o755)
    (prefix / "providers").mkdir()
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
