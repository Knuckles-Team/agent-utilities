"""Opt-in, scale=1 production soak/chaos certification.

There is no skip or mock branch.  Selecting this test is an explicit request to
run the signed exact-release campaign against provisioned production-equivalent
infrastructure; an incomplete environment fails immediately.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_utilities.core.config import AgentConfig


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.certification
def test_exact_release_scale_one_soak_and_fault_campaign() -> None:
    root = Path(__file__).resolve().parents[3]
    config = AgentConfig()
    assert config.certification_mode == "production"
    assert config.cert_release_manifest is not None
    assert config.cert_artifacts_dir is not None
    release = Path(config.cert_release_manifest)
    artifacts = Path(config.cert_artifacts_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.certification.campaign",
            "--release",
            str(release),
            "--artifacts-dir",
            str(artifacts),
        ],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=266_400,
    )
    assert result.returncode == 0, {
        "stdout_digest": __import__("hashlib").sha256(result.stdout).hexdigest(),
        "stderr_digest": __import__("hashlib").sha256(result.stderr).hexdigest(),
    }
    evidence = json.loads((artifacts / "operational-evidence.json").read_text())
    assert evidence["result"] == "pass"
    assert evidence["campaign"]["scale"] == 1.0
    assert 86_400 <= evidence["campaign"]["durationSeconds"] <= 259_200
    assert evidence["signature"]["subjectDigest"].startswith("sha256:")
