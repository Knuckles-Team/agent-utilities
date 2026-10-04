"""Execute candidate boundaries and model the literal Actions job conditions."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.release import verify_test_engine

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
JOBS = WORKFLOW["jobs"]
GATES = JOBS["gates"]
RUNTIME = JOBS["numeric-runtime-gate"]
STATES = ("success", "failure", "cancelled", "skipped", None)
CANDIDATE_EVENTS = ("pr", "fork_pr", "main")
EVENTS = (*CANDIDATE_EVENTS, "tag", "bad_tag", "branch", "release", "manual")


def _event(kind):
    event = {
        "github.event_name": "push",
        "github.ref_type": "branch",
        "github.ref": "refs/heads/main",
        "github.ref_name": "main",
    }
    if kind in {"pr", "fork_pr"}:
        event.update(
            {
                "github.event_name": "pull_request",
                "github.ref": "refs/pull/7/merge",
                "github.ref_name": "7/merge",
            }
        )
    elif kind in {"tag", "bad_tag"}:
        name = "v1.2.3" if kind == "tag" else "v-not-semver"
        event.update(
            {
                "github.ref_type": "tag",
                "github.ref": "refs/tags/" + name,
                "github.ref_name": name,
            }
        )
    elif kind == "branch":
        event["github.ref"] = "refs/heads/topic"
    elif kind in {"release", "manual"}:
        event["github.event_name"] = (
            "release" if kind == "release" else "workflow_dispatch"
        )
    return event


def _scheduled(job, event, results, *, cancelled=False, ancestors=()):
    """Restricted expression evaluation including Actions' implicit success().

    Actions applies success() unless an explicit status function is present;
    failure/skips propagate through the needs chain. This is intentionally not
    a general expression engine: new syntax must extend this test's model.
    """
    expression = (
        job.get("if", "true").strip().removeprefix("${{").removesuffix("}}").strip()
    )
    dependencies = job.get("needs", [])
    dependencies = [dependencies] if isinstance(dependencies, str) else dependencies
    status = [results.get(name) for name in dependencies] + list(ancestors)
    successful = not cancelled and all(value == "success" for value in status)
    explicit = re.search(r"\b(cancelled|success|failure|always)\s*\(", expression)
    if not explicit and not successful:
        return False
    context = {
        **event,
        **{f"needs.{name}.result": value for name, value in results.items()},
    }
    functions = {
        "cancelled": lambda: cancelled,
        "success": lambda: successful,
        "failure": lambda: "failure" in status,
        "always": lambda: True,
        "startsWith": lambda value, prefix: value.startswith(prefix),
    }
    return _actions_expression(expression, context, functions)


def _actions_expression(expression, context, functions):
    def replace(match):
        token = match.group()
        if token.startswith("'") or token in functions:
            return token
        if token in {"true", "false"}:
            return str(token == "true")
        if token.startswith(("github.", "needs.")):
            return repr(context.get(token))
        raise AssertionError(f"unmodeled Actions token: {token}")

    translated = re.sub(r"'[^']*'|[A-Za-z_][\w.-]*", replace, expression)
    translated = translated.replace("&&", " and ").replace("||", " or ")
    translated = re.sub(r"!(?!=)", " not ", translated).strip()
    tree = ast.parse("(" + translated + ")", mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    assert all(isinstance(node, allowed) for node in ast.walk(tree))
    assert all(
        node.id in functions for node in ast.walk(tree) if isinstance(node, ast.Name)
    )
    return bool(_expression_value(tree.body, functions))


def _expression_value(node, functions):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call):
        return functions[node.func.id](
            *(_expression_value(arg, functions) for arg in node.args)
        )
    if isinstance(node, ast.UnaryOp):
        return not _expression_value(node.operand, functions)
    if isinstance(node, ast.BoolOp):
        return _boolean_value(node, functions)
    if isinstance(node, ast.Compare):
        return _comparison_value(node, functions)
    raise AssertionError(f"unsupported expression node: {node!r}")


def _boolean_value(node, functions):
    values = (_expression_value(value, functions) for value in node.values)
    return all(values) if isinstance(node.op, ast.And) else any(values)


def _comparison_value(node, functions):
    assert len(node.ops) == len(node.comparators) == 1
    equal = _expression_value(node.left, functions) == _expression_value(
        node.comparators[0], functions
    )
    return equal if isinstance(node.ops[0], ast.Eq) else not equal


@pytest.mark.parametrize("kind", EVENTS)
def test_actual_build_condition_for_every_dependency_result(kind):
    for gates, scanners, public, cancelled in product(
        STATES, STATES, STATES, (False, True)
    ):
        results = {
            "gates": gates,
            "clone-scanners": scanners,
            "engine-release-order": public,
        }
        eligible = (kind in CANDIDATE_EVENTS and public == "skipped") or (
            kind in {"tag", "bad_tag"} and public == "success"
        )
        expected = not cancelled and gates == scanners == "success" and eligible
        assert (
            _scheduled(JOBS["build"], _event(kind), results, cancelled=cancelled)
            == expected
        )


@pytest.mark.parametrize("kind", EVENTS)
def test_runtime_condition_requires_real_upstream_success(kind):
    for build, gates, cancelled in product(STATES, STATES, (False, True)):
        results = {"build": build, "gates": gates}
        ancestors = ("skipped",) if kind in CANDIDATE_EVENTS else ("success",)
        expected = (
            not cancelled
            and build == gates == "success"
            and kind in (*CANDIDATE_EVENTS, "tag", "bad_tag")
        )
        assert (
            _scheduled(
                RUNTIME, _event(kind), results, cancelled=cancelled, ancestors=ancestors
            )
            == expected
        )


@pytest.mark.parametrize(
    "job_name,results",
    [
        (
            "build",
            {
                "gates": "success",
                "clone-scanners": "success",
                "engine-release-order": "skipped",
            },
        ),
        ("numeric-runtime-gate", {"gates": "success", "build": "success"}),
    ],
)
def test_removing_status_guard_restores_the_skipped_ancestor_bug(job_name, results):
    mutated = copy.deepcopy(JOBS[job_name])
    assert _scheduled(mutated, _event("pr"), results, ancestors=("skipped",))
    mutated["if"] = mutated["if"].replace("!cancelled() &&", "")
    assert not _scheduled(mutated, _event("pr"), results, ancestors=("skipped",))


def _step(job, fragment):
    return next(step for step in job["steps"] if fragment in step.get("name", ""))


def test_events_permissions_source_and_required_test_universe():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))
    assert set(triggers) == {"push", "pull_request"}
    assert "types" not in triggers["pull_request"]
    assert triggers["push"] == {"branches": ["main"], "tags": ["v*"]}
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert GATES["permissions"] == {"contents": "read", "actions": "read"}
    assert "permissions" not in RUNTIME
    for name in ("gates", "build", "numeric-runtime-gate"):
        checkout = JOBS[name]["steps"][0]
        assert checkout["with"]["ref"] == "${{ github.sha }}"
        assert checkout["with"]["persist-credentials"] is False
    assert RUNTIME["strategy"] == {
        "fail-fast": False,
        "matrix": {"os": ["ubuntu-latest", "windows-latest"]},
    }
    assert (
        _step(GATES, "Test suite")["run"].strip()
        == "python3 -m pytest -q -n auto --dist loadfile"
    )
    assert not any(job.get("continue-on-error") for job in JOBS.values())
    assert not any(
        step.get("continue-on-error")
        for job in JOBS.values()
        for step in job.get("steps", [])
    )
    names = [step.get("name", "") for step in GATES["steps"]]
    assert names.index("Verify official candidate producer provenance") < names.index(
        "Install the frozen epistemic-graph[full] wheel"
    )
    assert names.index("Test suite") < names.index(
        "Acquire verified Windows candidate for installed-wheel smoke"
    )
    assert names.index(
        "Acquire verified Windows candidate for installed-wheel smoke"
    ) < names.index("Preserve qualified Windows candidate")


def test_public_gate_and_all_publication_jobs_remain_ineligible_on_candidates():
    for kind in CANDIDATE_EVENTS:
        assert not _scheduled(JOBS["engine-release-order"], _event(kind), {})
        results = {"numeric-runtime-gate": "success"}
        for name in ("publish-pypi", "docker-publish-approval", "publish-docker"):
            assert not _scheduled(JOBS[name], _event(kind), results)
            results[name] = "skipped"
    for kind in ("tag", "bad_tag"):
        assert _scheduled(JOBS["engine-release-order"], _event(kind), {})


def test_normalized_unchanged_jobs_and_triggers():
    # Immutable AU21 32cefe26 source, normalized after resolving YAML anchors.
    expected = {
        "clone-scanners": "66b91b69984b11c6259dd648118f42ebf99a46d99bc78acb0864a3ee2cf0b75a",
        "publish-pypi": "4f30f46dac299bd842ae14db8987a4e5f299c78e6de4685842e87de374d10051",
        "docker-publish-approval": "209d1f37b609b1d20946e46bd5fd36f14a092b4fbe2a86c66bca1c2630f8a0f1",
        "publish-docker": "feb3cc77abb9860c4052ac04a8d20076fd6ea47dd6b9319fc2ef32802e4f675b",
        "triggers": "16f93a659bb35b714d8cba3cbde998e7e5eac47e136b7e630756653317e01142",
    }
    current = {name: JOBS[name] for name in expected if name != "triggers"}
    current["triggers"] = WORKFLOW.get("on", WORKFLOW.get(True))
    for name, digest in expected.items():
        actual = hashlib.sha256(
            yaml.safe_dump(current[name], sort_keys=True).encode()
        ).hexdigest()
        assert actual == digest, name


@pytest.fixture
def metadata():
    repository = {"id": 1248541673, "full_name": "Knuckles-Team/epistemic-graph"}
    source = "7f5179650d87e6f763b129cfeea6dfb4f8b0273b"
    run_id = 36889097785
    data = {
        "actions/runs/36889097785/attempts/1": {
            "id": run_id,
            "run_attempt": 1,
            "head_sha": source,
            "workflow_id": 333719087,
            "path": ".github/workflows/release.yml",
            "event": "push",
            "head_branch": "v2.27.0",
            "status": "completed",
            "conclusion": "cancelled",
            "repository": repository.copy(),
            "head_repository": repository.copy(),
        }
    }
    platforms = [
        (
            110489463714,
            "wheel linux-x86_64 / wheel linux-x86_64 (trusted tag)",
            11196548715,
            "wheel-linux-x86_64",
        ),
        (110489401420, "wheel windows-x86_64", 11193049469, "wheel-windows-x86_64"),
    ]
    for job, job_name, artifact, artifact_name in platforms:
        data[f"actions/jobs/{job}"] = {
            "id": job,
            "name": job_name,
            "run_id": run_id,
            "run_attempt": 1,
            "head_sha": source,
            "status": "completed",
            "conclusion": "success",
            "run_url": f"https://api.github.com/repos/Knuckles-Team/epistemic-graph/actions/runs/{run_id}",
            "started_at": "2026-10-01T17:00:00Z",
            "completed_at": "2026-10-01T23:00:00Z",
            "steps": [{"name": "Upload wheel artifact", "conclusion": "success"}],
        }
        data[f"actions/artifacts/{artifact}"] = {
            "id": artifact,
            "name": artifact_name,
            "expired": False,
            "created_at": "2026-10-01T22:00:00Z",
            "workflow_run": {
                "id": run_id,
                "head_sha": source,
                "repository_id": repository["id"],
                "head_repository_id": repository["id"],
            },
        }
    return data


def _inline(step, monkeypatch, tmp_path, environment):
    assert step["shell"] == "python"
    monkeypatch.chdir(tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    exec(compile(step["run"], step["name"], "exec"), {"__name__": "__main__"})


def _producer(metadata, monkeypatch, tmp_path, event="pull_request"):
    def api(command, **kwargs):
        assert command[:2] == ["gh", "api"]
        prefix = "repos/Knuckles-Team/epistemic-graph/"
        assert command[2].startswith(prefix)
        return json.dumps(metadata[command[2].removeprefix(prefix)])

    monkeypatch.setattr(subprocess, "check_output", api)
    _inline(
        _step(GATES, "Verify official candidate producer"),
        monkeypatch,
        tmp_path,
        {**GATES["env"], "EVENT_NAME": event, "EVENT_REF": "refs/pull/7/merge"},
    )
    return json.loads((tmp_path / "engine-contract.json").read_text())


def test_exact_cancelled_run_requires_both_successful_candidate_producers(
    metadata, monkeypatch, tmp_path
):
    contract = _producer(metadata, monkeypatch, tmp_path)
    assert contract["ubuntu-latest"]["artifact_id"] == 11196548715
    assert contract["windows-latest"]["artifact_id"] == 11193049469
    assert contract["windows-latest"]["run_attempt"] == 1


PROVENANCE_FIELDS = (
    [
        ("actions/runs/36889097785/attempts/1", key)
        for key in (
            "id",
            "run_attempt",
            "head_sha",
            "workflow_id",
            "path",
            "event",
            "status",
            "conclusion",
            "head_branch",
            "repository.id",
            "head_repository.id",
            "repository.full_name",
            "head_repository.full_name",
        )
    ]
    + [
        (f"actions/jobs/{job}", key)
        for job in (110489463714, 110489401420)
        for key in (
            "id",
            "name",
            "run_id",
            "run_attempt",
            "head_sha",
            "status",
            "conclusion",
            "run_url",
            "steps",
        )
    ]
    + [
        (f"actions/artifacts/{artifact}", key)
        for artifact in (11196548715, 11193049469)
        for key in (
            "id",
            "name",
            "expired",
            "created_at",
            "workflow_run.id",
            "workflow_run.head_sha",
            "workflow_run.repository_id",
            "workflow_run.head_repository_id",
        )
    ]
)


@pytest.mark.parametrize("endpoint,key", PROVENANCE_FIELDS)
@pytest.mark.parametrize("missing", [False, True])
def test_producer_metadata_missing_or_wrong_fails_closed(
    metadata, endpoint, key, missing, monkeypatch, tmp_path
):
    target = metadata[endpoint]
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    if missing:
        del target[parts[-1]]
    else:
        target[parts[-1]] = [] if key == "steps" else "wrong"
    with pytest.raises((SystemExit, KeyError, TypeError, ValueError)):
        _producer(metadata, monkeypatch, tmp_path)
    assert not (tmp_path / "engine-contract.json").exists()


def test_producer_os_swap_with_identical_wheel_bytes_is_rejected(
    metadata, monkeypatch, tmp_path
):
    metadata["actions/artifacts/11193049469"]["name"] = "wheel-linux-x86_64"
    with pytest.raises(SystemExit, match="artifact mismatch: name"):
        _producer(metadata, monkeypatch, tmp_path)


def test_fork_artifact_denial_never_uses_another_credential(monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gh", stderr="403")

    monkeypatch.setattr(subprocess, "check_output", denied)
    with pytest.raises(subprocess.CalledProcessError):
        _inline(
            _step(GATES, "Verify official candidate producer"),
            monkeypatch,
            tmp_path,
            {
                **GATES["env"],
                "EVENT_NAME": "pull_request",
                "EVENT_REF": "refs/pull/7/merge",
            },
        )
    assert not (tmp_path / "engine-contract.json").exists()


@pytest.fixture
def runtime_case(tmp_path, monkeypatch):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/agent_utilities-1.2.3-py3-none-any.whl").write_bytes(
        b"AU fixture"
    )
    engine = tmp_path / "engine-candidate"
    engine.mkdir()
    filename = "epistemic_graph-2.27.0-py3-none-any.whl"
    (engine / filename).write_bytes(b"engine fixture")
    pin = {
        "filename": filename,
        "sha256": hashlib.sha256(b"engine fixture").hexdigest(),
        "platform": "linux-x86_64",
        "artifact_id": 12,
        "run_attempt": 1,
    }
    receipt = {
        "engine": pin,
        "runner": "ubuntu-latest",
        "repository": "example/consumer",
        "run_id": "123",
        "run_attempt": "1",
        "source": "a" * 40,
    }
    (engine / "candidate-receipt.json").write_text(json.dumps(receipt))
    env = {
        "EVENT_NAME": "pull_request",
        "EVENT_REF": "refs/pull/7/merge",
        "REF_TYPE": "branch",
        "REF_NAME": "7/merge",
        "CONSUMER_SOURCE": "a" * 40,
        "MATRIX_OS": "ubuntu-latest",
        "ENGINE_CONTRACT": json.dumps({"ubuntu-latest": pin}),
        "GITHUB_REPOSITORY": "example/consumer",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    calls = []
    state = {"failure": None}

    def checked(command, **kwargs):
        calls.append((command, kwargs))
        stage = "venv" if "venv" in command else "install"
        if "wheel" in command or "installed" in command:
            stage = command[3]
        if state["failure"] == stage:
            raise subprocess.CalledProcessError(7, command)
        if stage == "wheel":
            verify_test_engine.verify_wheel(Path(command[4]), command[6], command[8])

    monkeypatch.setattr(subprocess, "check_call", checked)
    monkeypatch.setattr(shutil, "which", lambda name: "/pinned/uv")
    return SimpleNamespace(
        env=env,
        calls=calls,
        state=state,
        receipt=receipt,
        engine=engine,
        step=_step(RUNTIME, "Install exact Agent Utilities"),
    )


@pytest.mark.parametrize("platform", ["posix", "nt"])
@pytest.mark.parametrize("failure", [None, "wheel", "venv", "install", "installed"])
def test_runtime_checked_subprocesses_stop_at_first_native_error(
    runtime_case, platform, failure, monkeypatch, tmp_path
):
    case = runtime_case
    case.state["failure"] = failure
    facade = SimpleNamespace(name=platform, environ=os.environ)
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "os", facade)
        if failure:
            with pytest.raises(subprocess.CalledProcessError):
                _inline(case.step, isolated, tmp_path, case.env)
        else:
            _inline(case.step, isolated, tmp_path, case.env)
    assert (tmp_path / "outputs").exists() == (failure is None)
    if failure is None:
        command = next(
            command for command, _ in case.calls if command[0] == "/pinned/uv"
        )
        assert "--no-deps" not in command
        assert command[-1].startswith("epistemic-graph[full] @ file:")
        assert command[-1].endswith("#sha256=" + case.receipt["engine"]["sha256"])
        suffix = "Scripts/python.exe" if platform == "nt" else "bin/python"
        assert command[command.index("--python") + 1].endswith(suffix)


@pytest.mark.parametrize(
    "key", ["runner", "repository", "run_id", "run_attempt", "source", "engine"]
)
def test_consumer_receipt_rejects_os_source_and_rerun_substitution(
    runtime_case, key, monkeypatch, tmp_path
):
    case = runtime_case
    case.receipt[key] = "wrong"
    (case.engine / "candidate-receipt.json").write_text(json.dumps(case.receipt))
    with pytest.raises(SystemExit, match="receipt"):
        _inline(case.step, monkeypatch, tmp_path, case.env)
    assert case.calls == []


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "missing"])
def test_candidate_consumer_uses_real_wheel_verifier(
    runtime_case, mutation, monkeypatch, tmp_path
):
    wheel = runtime_case.engine / runtime_case.receipt["engine"]["filename"]
    if mutation == "bytes":
        wheel.write_bytes(b"modified")
    else:
        target = wheel.with_suffix(".original")
        wheel.rename(target)
        if mutation == "symlink":
            wheel.symlink_to(target)
    with pytest.raises(ValueError):
        _inline(runtime_case.step, monkeypatch, tmp_path, runtime_case.env)
    assert len(runtime_case.calls) == 1


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_public_install_cannot_use_candidate_cache_or_resolver_overrides(
    runtime_case, platform, monkeypatch, tmp_path
):
    case = runtime_case
    case.env.update(
        EVENT_NAME="push",
        EVENT_REF="refs/tags/v1.2.3",
        REF_TYPE="tag",
        REF_NAME="v1.2.3",
        ENGINE_CONTRACT="invalid candidate data must not be read",
    )
    for key in (
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_OFFLINE",
        "UV_OVERRIDE",
        "UV_CONFIG_FILE",
        "UV_CACHE_DIR",
        "PIP_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_CONFIG_FILE",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ):
        case.env[key] = "candidate-poison"
    with monkeypatch.context() as isolated:
        isolated.setitem(
            sys.modules, "os", SimpleNamespace(name=platform, environ=os.environ)
        )
        _inline(case.step, isolated, tmp_path, case.env)
    assert len(case.calls) == 2
    venv, install = case.calls
    assert venv[0][1:4] == ["-I", "-m", "venv"]
    assert "--system-site-packages" not in venv[0]
    command, settings = install
    assert command[1:5] == ["--no-config", "--no-cache", "pip", "install"]
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert len(command) == 10
    assert command[-1].endswith("agent_utilities-1.2.3-py3-none-any.whl")
    assert Path(settings["cwd"]).parent == tmp_path
    assert "candidate-poison" not in settings["env"].values()
    assert not any("engine-candidate" in argument for argument in command)


def test_unavailable_public_resolution_is_not_rescued_by_candidate(
    runtime_case, monkeypatch, tmp_path
):
    case = runtime_case
    case.env.update(
        EVENT_NAME="push",
        EVENT_REF="refs/tags/v1.2.3",
        REF_TYPE="tag",
        REF_NAME="v1.2.3",
    )
    case.state["failure"] = "install"
    with pytest.raises(subprocess.CalledProcessError):
        _inline(case.step, monkeypatch, tmp_path, case.env)
    assert not (tmp_path / "outputs").exists()
    assert len(case.calls) == 2


@pytest.mark.parametrize(
    "fragment", ["Smoke installed numeric", "Smoke installed Langfuse"]
)
def test_smokes_use_the_same_isolated_interpreter_and_propagate_errors(
    fragment, monkeypatch, tmp_path
):
    step = _step(RUNTIME, fragment)
    assert (
        step["env"]["RUNTIME_PYTHON"] == "${{ steps.runtime-install.outputs.python }}"
    )
    assert step["working-directory"] == "${{ runner.temp }}"
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(subprocess, "check_call", failed)
    with pytest.raises(subprocess.CalledProcessError):
        _inline(
            step,
            monkeypatch,
            tmp_path,
            {"RUNTIME_PYTHON": "/fresh/python", "GITHUB_WORKSPACE": str(ROOT)},
        )
    assert calls[0][:2] == ["/fresh/python", "-I"]


def test_candidate_transport_is_bound_to_the_current_run_attempt():
    download = _step(RUNTIME, "Download this run")
    assert (
        download["with"]["name"]
        == "eg-candidate-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.os }}"
    )
    assert set(download["with"]) == {"name", "path"}
    for kind in EVENTS:
        assert _scheduled({"if": download["if"]}, _event(kind), {}) == (
            kind in CANDIDATE_EVENTS
        )
    for fragment in ("Preserve qualified Linux", "Preserve qualified Windows"):
        upload = _step(GATES, fragment)
        assert upload["with"]["if-no-files-found"] == "error"
        assert (
            "${{ github.run_id }}-${{ github.run_attempt }}" in upload["with"]["name"]
        )


@pytest.mark.parametrize("failure", ["metadata", "download", "checksum"])
def test_windows_acquisition_cannot_hide_native_failure(
    metadata, failure, monkeypatch, tmp_path
):
    contract = _producer(metadata, monkeypatch, tmp_path)
    pin = contract["windows-latest"]

    def api(command, **kwargs):
        if failure == "metadata":
            raise subprocess.CalledProcessError(1, command)
        return json.dumps(metadata["actions/artifacts/11193049469"]).encode()

    def checked(command, **kwargs):
        if command[0] == "gh":
            if failure == "download":
                raise subprocess.CalledProcessError(1, command)
            with zipfile.ZipFile(kwargs["stdout"], "w") as archive:
                archive.writestr(pin["filename"], b"incorrect wheel bytes")
        elif command[3] == "artifact":
            verify_test_engine.verify_artifact(
                Path(command[4]),
                pin["artifact_id"],
                pin["artifact_name"],
                pin["run_id"],
                pin["source"],
            )
        else:
            verify_test_engine.verify_wheel(
                Path(command[4]), pin["filename"], pin["sha256"]
            )

    monkeypatch.setattr(subprocess, "check_output", api)
    monkeypatch.setattr(subprocess, "check_call", checked)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        _inline(_step(GATES, "Acquire verified Windows"), monkeypatch, tmp_path, {})


@pytest.mark.parametrize("mutation", ["old_barrier", "skipped_tag", "missing_windows"])
def test_scheduling_regressions_are_caught_by_the_qualification_oracles(
    mutation, monkeypatch
):
    if mutation == "missing_windows":
        monkeypatch.setitem(RUNTIME["strategy"]["matrix"], "os", ["ubuntu-latest"])
        with pytest.raises(AssertionError):
            test_events_permissions_source_and_required_test_universe()
        return
    build = JOBS["build"]
    if mutation == "old_barrier":
        monkeypatch.delitem(build, "if")
        kind = "pr"
    else:
        monkeypatch.setitem(
            build,
            "if",
            build["if"].replace(
                "needs.engine-release-order.result == 'success'",
                "needs.engine-release-order.result == 'skipped'",
            ),
        )
        kind = "tag"
    with pytest.raises(AssertionError):
        test_actual_build_condition_for_every_dependency_result(kind)


def test_checksum_removal_is_caught_by_the_real_verifier_oracle(
    runtime_case, monkeypatch, tmp_path
):
    class RemoveWheelCheck(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and any(
                isinstance(child, ast.Constant) and child.value == "wheel"
                for child in ast.walk(node.value)
            ):
                return None
            return node

    modified = RemoveWheelCheck().visit(ast.parse(runtime_case.step["run"]))
    monkeypatch.setitem(runtime_case.step, "run", ast.unparse(modified))
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_candidate_consumer_uses_real_wheel_verifier(
            runtime_case, "bytes", monkeypatch, tmp_path
        )


def test_ignored_installer_exit_is_caught_by_the_native_failure_oracle(
    runtime_case, monkeypatch, tmp_path
):
    checked = subprocess.check_call

    def unchecked(command, **kwargs):
        try:
            checked(command, **kwargs)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
        return 0

    monkeypatch.setattr(subprocess, "call", unchecked)
    monkeypatch.setitem(
        runtime_case.step,
        "run",
        runtime_case.step["run"].replace("subprocess.check_call(", "subprocess.call("),
    )
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_runtime_checked_subprocesses_stop_at_first_native_error(
            runtime_case, "nt", "install", monkeypatch, tmp_path
        )
