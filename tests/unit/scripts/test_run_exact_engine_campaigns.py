"""Contracts for the serial exact-engine campaign producer."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import source_freeze_gate
from scripts.certification import run_exact_engine_campaigns as campaigns

ROOT = Path(__file__).resolve().parents[3]


def _aggregate(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_freeze_evidence(path: Path, producer_root: Path) -> Path:
    manifest_payload = (
        ROOT / "deploy/release/source-freeze-gates.json"
    ).read_bytes()
    manifest = json.loads(manifest_payload)
    repository_ids = [item["id"] for item in manifest["repositories"]]
    repository_digests = {
        identifier: format(index + 11, "064x")
        for index, identifier in enumerate(repository_ids)
    }
    repository_digests["epistemic-graph"] = source_freeze_gate.source_tree_digest(
        producer_root
    )
    token = re.compile(r"^\{repo:([a-z][a-z0-9-]{2,63})\}(.*)$")
    command_rows = []
    for command in manifest["commands"]:
        identifiers = {command["repository"]}
        identifiers.update(
            match.group(1)
            for item in command["argv"]
            if (match := token.fullmatch(item)) is not None
        )
        digest = _aggregate(
            {
                identifier: repository_digests[identifier]
                for identifier in repository_ids
                if identifier in identifiers
            }
        )
        command_rows.append(
            {
                "id": command["id"],
                "status": "passed",
                "exit_code": 0,
                "termination": "exited",
                "source_digest_before": digest,
                "source_digest_after": digest,
            }
        )
    source_digest = _aggregate(repository_digests)
    evidence = {
        "schema": "source-freeze-evidence/1",
        "status": "passed",
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "source_digest_before": source_digest,
        "source_digest_after": source_digest,
        "tools": [
            {"id": "git", "sha256": "3" * 64},
            {"id": "rg", "sha256": "4" * 64},
        ],
        "repositories": [
            {
                "id": identifier,
                "sha256_before": repository_digests[identifier],
                "sha256_after": repository_digests[identifier],
            }
            for identifier in repository_ids
        ],
        "commands": command_rows,
        "gates": [
            {
                "id": item["id"],
                "required_evidence": item["evidence_classes"],
                "source_status": (
                    "passed"
                    if "local-source" in item["evidence_classes"]
                    else "not-applicable"
                ),
                "remaining_evidence": [
                    value
                    for value in item["evidence_classes"]
                    if value != "local-source"
                ],
            }
            for item in manifest["gates"]
        ],
    }
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    producer_root = tmp_path / "epistemic-graph"
    scripts = producer_root / "scripts"
    scripts.mkdir(parents=True)
    for name in campaigns.PRODUCER_SCRIPTS.values():
        (scripts / name).write_text("# exact producer fixture\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    work_root = private / "work"
    output_parent = private / "evidence"
    work_root.mkdir(mode=0o700)
    output_parent.mkdir(mode=0o700)
    engine = private / "engine"
    engine.write_bytes(b"exact-engine-fixture")
    engine.chmod(0o500)
    campaign_python = private / "python"
    campaign_python.write_bytes(b"exact-python-fixture")
    campaign_python.chmod(0o500)
    authority = private / "authority.json"
    authority.write_text("{}\n", encoding="utf-8")
    authority.chmod(0o600)
    return {
        "producer_root": producer_root,
        "private": private,
        "work_root": work_root,
        "output_parent": output_parent,
        "output": output_parent / "campaigns",
        "engine": engine,
        "engine_digest": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "python": campaign_python,
        "python_digest": hashlib.sha256(campaign_python.read_bytes()).hexdigest(),
        "authority": authority,
        "source": private / "source-freeze.json",
    }


def test_direct_script_help_bootstraps_from_unrelated_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/certification/run_exact_engine_campaigns.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert b"--source-freeze-evidence" in result.stdout


def test_campaign_inventory_is_exact_and_performance_precedes_multimodal() -> None:
    assert campaigns.CAMPAIGN_ORDER == (
        "performance",
        "fault-restart",
        "protocol-authorization",
        "multimodal",
        "knowledge-batch",
        "reasoning-repair",
    )
    assert set(campaigns.PRODUCER_SCRIPTS) == set(campaigns.CAMPAIGN_ORDER)
    assert set(campaigns.OUTPUT_FILES) == set(campaigns.CAMPAIGN_ORDER)
    assert len(set(campaigns.OUTPUT_FILES.values())) == 6


def test_multimodal_argv_binds_exact_performance_bytes(tmp_path: Path) -> None:
    performance = tmp_path / "performance.json"
    argv = campaigns._campaign_argv(
        "multimodal",
        python=tmp_path / "python",
        script=tmp_path / "producer.py",
        engine=tmp_path / "engine",
        engine_sha256="a" * 64,
        output=tmp_path / "multimodal.json",
        authority_config=tmp_path / "authority.json",
        work_root=tmp_path,
        performance_evidence=performance,
        performance_digest="b" * 64,
        markdown_output=tmp_path / "performance.md",
    )

    assert argv[1:4] == ["-E", "-s", "-B"]
    assert argv[argv.index("--performance-evidence") + 1] == str(performance)
    assert argv[argv.index("--performance-evidence-sha256") + 1] == "b" * 64
    assert "--output" in argv


def test_source_freeze_binds_the_exact_producer_tree(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    evidence = _source_freeze_evidence(values["source"], values["producer_root"])
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()

    observed = campaigns._source_freeze_binding(
        evidence,
        evidence_digest,
        values["producer_root"],
    )
    assert observed == source_freeze_gate.source_tree_digest(values["producer_root"])

    next(iter((values["producer_root"] / "scripts").iterdir())).write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(
        campaigns.CampaignOrchestrationError,
        match="producer_source_digest_mismatch",
    ):
        campaigns._source_freeze_binding(
            evidence,
            evidence_digest,
            values["producer_root"],
        )


def _run_with_fake_producers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str | None = None,
) -> tuple[dict[str, Any], list[tuple[str, list[str]]]]:
    values = _fixture(tmp_path)
    producer_digest = source_freeze_gate.source_tree_digest(values["producer_root"])
    monkeypatch.setattr(
        campaigns,
        "_source_freeze_binding",
        lambda *_args: producer_digest,
    )
    monkeypatch.setattr(
        campaigns.source_freeze_gate,
        "source_tree_digest",
        lambda _root: producer_digest,
    )
    calls: list[tuple[str, list[str]]] = []

    def runner(
        argv: list[str],
        *,
        campaign: str,
        cwd: Path,
        environment: dict[str, str],
    ) -> None:
        del cwd, environment
        calls.append((campaign, argv))
        if campaign == fail_at:
            raise campaigns.CampaignOrchestrationError(f"{campaign}_failed")
        option = "--json-output" if campaign == "performance" else "--output"
        output = Path(argv[argv.index(option) + 1])
        output.write_text(json.dumps({"campaign": campaign}) + "\n", encoding="utf-8")
        output.chmod(0o600)
        if campaign == "performance":
            markdown = Path(argv[argv.index("--markdown-output") + 1])
            markdown.write_text("# private temporary report\n", encoding="utf-8")

    monkeypatch.setattr(campaigns, "_run_bounded", runner)

    def validate(
        campaign: str,
        path: Path,
        *,
        engine_sha256: str,
        performance_digest: str | None,
    ) -> str:
        del engine_sha256
        if campaign == "multimodal":
            assert performance_digest is not None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(campaigns, "_validate_campaign_evidence", validate)
    values["result"] = campaigns.run_campaigns(
        release_id="release-fixture",
        engine_binary=values["engine"],
        engine_sha256=values["engine_digest"],
        campaign_python=values["python"],
        campaign_python_sha256=values["python_digest"],
        epistemic_graph_root=values["producer_root"],
        source_freeze_evidence=values["source"],
        source_freeze_sha256="c" * 64,
        authority_config=values["authority"],
        work_root=values["work_root"],
        output_dir=values["output"],
    )
    return values, calls


def test_runner_publishes_only_six_validated_json_files_in_exact_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, calls = _run_with_fake_producers(tmp_path, monkeypatch)

    assert [name for name, _argv in calls] == list(campaigns.CAMPAIGN_ORDER)
    assert set(path.name for path in values["result"].values()) == set(
        campaigns.OUTPUT_FILES.values()
    )
    assert {path.name for path in values["output"].iterdir()} == set(
        campaigns.OUTPUT_FILES.values()
    )
    assert all(path.stat().st_mode & 0o077 == 0 for path in values["output"].iterdir())
    multimodal = dict(calls)["multimodal"]
    performance = values["output"] / campaigns.OUTPUT_FILES["performance"]
    expected = hashlib.sha256(performance.read_bytes()).hexdigest()
    assert multimodal[multimodal.index("--performance-evidence-sha256") + 1] == expected
    assert not (values["output"] / "performance.md").exists()


def test_failure_removes_staging_and_never_publishes_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        campaigns.CampaignOrchestrationError,
        match="fault-restart_failed",
    ):
        _run_with_fake_producers(tmp_path, monkeypatch, fail_at="fault-restart")

    output_parent = tmp_path / "private/evidence"
    assert not (output_parent / "campaigns").exists()
    assert list(output_parent.iterdir()) == []


@pytest.mark.parametrize("campaign", campaigns.CAMPAIGN_ORDER)
def test_nonconforming_producer_output_is_never_accepted(
    tmp_path: Path,
    campaign: str,
) -> None:
    output = tmp_path / f"{campaign}.json"
    output.write_text("{}\n", encoding="utf-8")
    output.chmod(0o600)

    with pytest.raises(
        campaigns.CampaignOrchestrationError,
        match=rf"{re.escape(campaign)}_evidence_invalid",
    ):
        campaigns._validate_campaign_evidence(
            campaign,
            output,
            engine_sha256="a" * 64,
            performance_digest="b" * 64,
        )


def test_subprocess_boundary_is_argv_only_and_bounded() -> None:
    source = Path(campaigns.__file__).read_text(encoding="utf-8")

    assert "shell=False" in source
    assert "shell=True" not in source
    assert "os.environ.copy" not in source
    assert "CAMPAIGN_TIMEOUT_SECONDS" in source
    assert "_MAX_CHILD_OUTPUT_BYTES" in source
