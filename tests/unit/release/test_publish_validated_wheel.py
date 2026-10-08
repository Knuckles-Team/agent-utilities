"""Caller wiring: publication policy remains in the pipelines contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import publish_validated_wheel as caller

FILE = "agent_utilities-2.5.0-py3-none-any.whl"


@pytest.mark.parametrize("pending", [[], [FILE]])
def test_uploads_only_current_missing_files_and_always_postverifies(
    monkeypatch, pending
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        if command[0] == "uv":
            return SimpleNamespace(stdout="")
        assert command[:3] == [sys.executable, "-I", caller.CONTRACT]
        # Preflight can differ: only the immediately revalidated missing list
        # authorizes upload (another publisher may have completed meanwhile).
        return SimpleNamespace(
            stdout=json.dumps([FILE] if command[3] == "preflight" else pending)
        )

    monkeypatch.setattr(caller.subprocess, "run", run)
    caller.publish("2.5.0", Path("publication.json"))
    checks = [command for command in calls if command[0] != "uv"]
    assert [command[3] for command in checks] == ["preflight", "missing", "postverify"]
    assert all(command[4:] == checks[0][4:] for command in checks)
    assert checks[0][4:] == [
        "--directory",
        "dist",
        "--package",
        "agent-utilities",
        "--version",
        "2.5.0",
        "--manifest",
        "publication.json",
    ]
    uploads = [command for command in calls if command[0] == "uv"]
    assert uploads == (
        [
            [
                "uv",
                "publish",
                "--publish-url",
                "https://upload.pypi.org/legacy/",
                f"dist/{FILE}",
            ]
        ]
        if pending
        else []
    )
    if pending:
        assert calls.index(uploads[0]) == 2


@pytest.mark.parametrize("failure", ["preflight", "missing", "uv", "postverify"])
def test_failed_phase_stops_publication(monkeypatch, failure):
    calls = []

    def run(command, **kwargs):
        phase = "uv" if command[0] == "uv" else command[3]
        calls.append(phase)
        if phase == failure:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps([FILE]))

    monkeypatch.setattr(caller.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        caller.publish("2.5.0", Path("publication.json"))
    sequence = ["preflight", "missing", "uv", "postverify"]
    assert calls == sequence[: sequence.index(failure) + 1]
