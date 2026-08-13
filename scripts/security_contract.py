#!/usr/bin/env python3
"""Execute bounded repository security hooks and validate their evidence.

This is the **single canonical copy** for the whole fleet (CONCEPT:AU-OS.governance.single-canonical-source-security-contract).
It previously existed as an identical or near-identical file copy-pasted into ~74
`agent-packages/*` repos, which let one defect (an unguarded `import resource`,
Unix-only) replicate 74-fold and drift into several inconsistent variants. Every
consumer repository now reaches this module through
`scripts/run_agent_utilities_gate.py --script scripts/security_contract.py -- ...`
(the same locate-au-and-run pattern already used for `check_no_legacy_markers.py`,
`mermaid_linter.py`, and `check_stubs.py`) instead of keeping a local copy.

The reusable security workflow calls this module directly from the checked-out
repository.  A repository contract names argv arrays rather than shell strings,
so untrusted configuration cannot introduce an extra shell-evaluation layer.
All paths are relative regular files below the repository root, outputs are
bounded, hook environments exclude credential-like variables, and missing or
malformed evidence fails closed.

Usage: python scripts/security_contract.py --contract <path> {validate,run-hook,check-licenses} ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

if os.name == "nt":
    import ctypes
    from ctypes import wintypes


MAX_CONTRACT_BYTES = 128 * 1024
MAX_EVIDENCE_BYTES = 128 * 1024
MAX_SBOM_BYTES = 64 * 1024 * 1024
MAX_COMPONENTS = 100_000
MAX_LICENSE_DECLARATIONS = 64
MAX_LICENSE_EXPRESSION_BYTES = 4_096
MAX_LICENSE_TOKENS = 256
MAX_LICENSE_NESTING = 32
MAX_ARGV_ITEMS = 64
MAX_ARGUMENT_BYTES = 4_096
MAX_HOOK_OUTPUT_BYTES = 4 * 1024 * 1024
_HOOK_READ_CHUNK_BYTES = 64 * 1024
HOOK_KINDS = ("fuzz", "authenticated_negative")
_SENSITIVE_ENV = re.compile(
    r"(?:^|_)(?:AUTHORIZATION|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY)(?:_|$)",
    re.IGNORECASE,
)
_SPDX_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,127}$")
_SPDX_OPERATORS = frozenset({"AND", "OR", "WITH"})

# Windows Job Object constants/structures (CONCEPT:AU-OS.host.windows-job-object-process-tree-bound).
# A Job Object with KILL_ON_JOB_CLOSE is the Windows analogue of a POSIX process
# group: assigning the hook process to it, then terminating/closing the job,
# kills the hook and every descendant it spawned atomically -- the same
# guarantee `os.killpg(..., SIGKILL)` gives on POSIX.
if os.name == "nt":
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001  # SET_QUOTA | TERMINATE

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = tuple(
            (name, ctypes.c_uint64)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        )

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )


class SecurityContractError(RuntimeError):
    """A stable, path- and secret-free contract failure."""


class _SpdxSyntaxError(ValueError):
    """An internal, detail-free SPDX parsing failure."""


class _SpdxParser:
    """Parse the bounded current SPDX expression grammar without dependencies."""

    def __init__(self, tokens: tuple[str, ...]) -> None:
        self._tokens = tokens
        self._position = 0
        self._nesting = 0

    def parse(self) -> tuple[Any, ...]:
        node = self._parse_or()
        if self._position != len(self._tokens):
            raise _SpdxSyntaxError
        return node

    def _peek(self) -> str | None:
        if self._position >= len(self._tokens):
            return None
        return self._tokens[self._position]

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise _SpdxSyntaxError
        self._position += 1
        return token

    def _parse_or(self) -> tuple[Any, ...]:
        nodes = [self._parse_and()]
        while self._peek() == "OR":
            self._take()
            nodes.append(self._parse_and())
        return nodes[0] if len(nodes) == 1 else ("or", tuple(nodes))

    def _parse_and(self) -> tuple[Any, ...]:
        nodes = [self._parse_with()]
        while self._peek() == "AND":
            self._take()
            nodes.append(self._parse_with())
        return nodes[0] if len(nodes) == 1 else ("and", tuple(nodes))

    def _parse_with(self) -> tuple[Any, ...]:
        node = self._parse_primary()
        if self._peek() != "WITH":
            return node
        self._take()
        exception = self._take()
        if node[0] != "id" or not _is_spdx_identifier(exception):
            raise _SpdxSyntaxError
        return ("with", node[1], exception)

    def _parse_primary(self) -> tuple[Any, ...]:
        token = self._take()
        if token == "(":
            self._nesting += 1
            if self._nesting > MAX_LICENSE_NESTING:
                raise _SpdxSyntaxError
            node = self._parse_or()
            if self._take() != ")":
                raise _SpdxSyntaxError
            self._nesting -= 1
            return ("group", node)
        if not _is_spdx_identifier(token):
            raise _SpdxSyntaxError
        return ("id", token)


def _is_spdx_identifier(value: str) -> bool:
    return bool(_SPDX_IDENTIFIER.fullmatch(value)) and value not in _SPDX_OPERATORS


def _bounded_utf8(value: str, maximum_bytes: int) -> bool:
    try:
        return len(value.encode("utf-8")) <= maximum_bytes
    except UnicodeEncodeError:
        return False


def _tokenize_spdx(expression: str) -> tuple[str, ...]:
    if (
        not isinstance(expression, str)
        or not expression
        or not _bounded_utf8(expression, MAX_LICENSE_EXPRESSION_BYTES)
        or "\x00" in expression
    ):
        raise _SpdxSyntaxError
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        character = expression[position]
        if character.isspace():
            position += 1
            continue
        if character in "()":
            tokens.append(character)
            position += 1
        else:
            end = position
            while end < len(expression):
                candidate = expression[end]
                if candidate.isspace() or candidate in "()":
                    break
                end += 1
            token = expression[position:end]
            if token not in _SPDX_OPERATORS and not _is_spdx_identifier(token):
                raise _SpdxSyntaxError
            tokens.append(token)
            position = end
        if len(tokens) > MAX_LICENSE_TOKENS:
            raise _SpdxSyntaxError
    if not tokens:
        raise _SpdxSyntaxError
    return tuple(tokens)


def _parse_spdx(expression: str) -> tuple[Any, ...]:
    return _SpdxParser(_tokenize_spdx(expression)).parse()


def _spdx_identifiers(node: tuple[Any, ...]) -> set[str]:
    kind = node[0]
    if kind == "id":
        return {node[1]}
    if kind == "with":
        return {node[1], node[2]}
    if kind == "group":
        return _spdx_identifiers(node[1])
    identifiers: set[str] = set()
    for child in node[1]:
        identifiers.update(_spdx_identifiers(child))
    return identifiers


def _spdx_node_is_allowed(
    node: tuple[Any, ...],
    *,
    allowed: set[str],
    allowed_exceptions: set[str],
) -> bool:
    kind = node[0]
    if kind == "id":
        return node[1] in allowed
    if kind == "with":
        return node[1] in allowed and node[2] in allowed_exceptions
    if kind == "group":
        return _spdx_node_is_allowed(
            node[1], allowed=allowed, allowed_exceptions=allowed_exceptions
        )
    children = node[1]
    if kind == "and":
        return all(
            _spdx_node_is_allowed(
                child, allowed=allowed, allowed_exceptions=allowed_exceptions
            )
            for child in children
        )
    if kind == "or":
        return any(
            _spdx_node_is_allowed(
                child, allowed=allowed, allowed_exceptions=allowed_exceptions
            )
            for child in children
        )
    raise _SpdxSyntaxError


def _spdx_expression_is_allowed(
    expression: str,
    *,
    allowed: set[str],
    allowed_exceptions: set[str],
    denied: set[str],
) -> bool:
    """Evaluate one expression while making every explicit denial terminal."""

    try:
        node = _parse_spdx(expression)
    except _SpdxSyntaxError:
        return False
    if _spdx_identifiers(node) & denied:
        return False
    return _spdx_node_is_allowed(
        node, allowed=allowed, allowed_exceptions=allowed_exceptions
    )


def _relative_file(root: Path, value: str, *, maximum_bytes: int) -> Path:
    if not isinstance(value, str) or not value or len(value.encode()) > 4_096:
        raise SecurityContractError("security contract path is invalid")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SecurityContractError("security contract path is invalid")
    joined = root.joinpath(candidate)
    try:
        if joined.is_symlink() or not joined.is_file():
            raise SecurityContractError("security contract file is unavailable")
        resolved = joined.resolve(strict=True)
        resolved.relative_to(root)
        size = resolved.stat().st_size
    except (OSError, ValueError) as exc:
        raise SecurityContractError("security contract file is unavailable") from exc
    if size > maximum_bytes:
        raise SecurityContractError("security contract file exceeds its boundary")
    return resolved


def _read_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        if len(payload) > maximum_bytes:
            raise SecurityContractError("security evidence exceeds its boundary")
        value = json.loads(payload)
    except SecurityContractError:
        raise
    except Exception as exc:
        raise SecurityContractError("security contract JSON is invalid") from exc
    if not isinstance(value, dict):
        raise SecurityContractError("security contract JSON must be an object")
    return value


def load_contract(root: Path, reference: str) -> dict[str, Any]:
    """Load and fully validate one versioned assurance contract."""

    contract = _read_json(
        _relative_file(root, reference, maximum_bytes=MAX_CONTRACT_BYTES),
        maximum_bytes=MAX_CONTRACT_BYTES,
    )
    if set(contract) != {"version", "hooks", "license_policy"}:
        raise SecurityContractError("security contract fields are invalid")
    if contract.get("version") != 2 or not isinstance(contract.get("hooks"), dict):
        raise SecurityContractError("security contract version is unsupported")
    hooks = contract["hooks"]
    if set(hooks) != set(HOOK_KINDS):
        raise SecurityContractError("security contract must declare every hook")
    for kind in HOOK_KINDS:
        hook = hooks[kind]
        if not isinstance(hook, dict) or set(hook) != {
            "argv",
            "timeout_seconds",
            "evidence",
            "min_cases",
        }:
            raise SecurityContractError("security hook declaration is invalid")
        argv = hook["argv"]
        if (
            not isinstance(argv, list)
            or not 1 <= len(argv) <= MAX_ARGV_ITEMS
            or any(
                not isinstance(argument, str)
                or not argument
                or len(argument.encode()) > MAX_ARGUMENT_BYTES
                or "\x00" in argument
                for argument in argv
            )
        ):
            raise SecurityContractError("security hook argv is invalid")
        timeout = hook["timeout_seconds"]
        if not isinstance(timeout, int) or not 1 <= timeout <= 900:
            raise SecurityContractError("security hook timeout is invalid")
        min_cases = hook["min_cases"]
        if not isinstance(min_cases, int) or not 1 <= min_cases <= 100_000_000:
            raise SecurityContractError("security hook case threshold is invalid")
        evidence = hook["evidence"]
        if (
            not isinstance(evidence, str)
            or Path(evidence).is_absolute()
            or ".." in Path(evidence).parts
        ):
            raise SecurityContractError("security hook evidence path is invalid")
    policy = contract["license_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "allowed",
        "allowed_exceptions",
        "denied",
    }:
        raise SecurityContractError("license policy declaration is invalid")
    for key in ("allowed", "allowed_exceptions", "denied"):
        values = policy[key]
        if (
            not isinstance(values, list)
            or len(values) > 1_024
            or any(
                not isinstance(value, str) or not _is_spdx_identifier(value)
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise SecurityContractError("license policy identifiers are invalid")
    if not policy["allowed"]:
        raise SecurityContractError("license policy must declare an allow-list")
    if set(policy["denied"]) & (
        set(policy["allowed"]) | set(policy["allowed_exceptions"])
    ):
        raise SecurityContractError("license policy allow and deny lists overlap")
    return contract


def _hook_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        if _SENSITIVE_ENV.search(key):
            continue
        if key in {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "CI",
            "GITHUB_ACTIONS",
            "RUNNER_OS",
            "RUNNER_ARCH",
            "VIRTUAL_ENV",
            "PYTHONHOME",
            # Windows subprocess creation and DLL/temp-file resolution needs these
            # to start at all -- without them the hook process fails to launch,
            # which is a platform-restricted feature, not a bounded environment.
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "COMSPEC",
            "TEMP",
            "TMP",
            "PATHEXT",
        }:
            environment[key] = value
    environment["SECURITY_HOOK"] = "1"
    return environment


def _hook_session_kwargs() -> dict[str, Any]:
    """``Popen`` kwargs that put the hook in its own killable process tree.

    POSIX gets a new session (process group); Windows gets a new process group
    that a bound Job Object (``_bind_hook_job``) can terminate atomically.
    """

    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def _bind_hook_job(pid: int) -> int | None:
    """Bind *pid* to a new kill-on-close Windows Job Object, or ``None`` on failure.

    This is the Windows analogue of a POSIX process group: every process the
    hook spawns joins the same job, so one call terminates the whole tree.
    """

    job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not ctypes.windll.kernel32.SetInformationJobObject(
        job,
        _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        ctypes.windll.kernel32.CloseHandle(job)
        return None
    handle = ctypes.windll.kernel32.OpenProcess(
        _PROCESS_SET_QUOTA_AND_TERMINATE, False, pid
    )
    if not handle:
        ctypes.windll.kernel32.CloseHandle(job)
        return None
    try:
        if not ctypes.windll.kernel32.AssignProcessToJobObject(job, handle):
            ctypes.windll.kernel32.CloseHandle(job)
            return None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return job


def _close_hook_job(job: int) -> None:
    ctypes.windll.kernel32.TerminateJobObject(job, 1)
    ctypes.windll.kernel32.CloseHandle(job)


def _terminate_hook_tree(process: subprocess.Popen[bytes], job: int | None) -> None:
    """Kill the hook and every process it spawned, on whichever platform this is.

    Same intent as the POSIX-only ``os.killpg(pid, SIGKILL)`` this replaces;
    Windows reaches it through the Job Object bound in ``_bind_hook_job``.
    """

    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return
    if job is not None:
        ctypes.windll.kernel32.TerminateJobObject(job, 1)
    else:
        process.kill()


def _pump_bounded_output(
    process: subprocess.Popen[bytes], log: Any, *, maximum_bytes: int
) -> bool:
    """Copy the hook's combined stdout/stderr into *log*, capped at *maximum_bytes*.

    Returns ``False`` once the boundary would be exceeded (the caller then
    terminates the hook tree). Replaces a POSIX ``RLIMIT_FSIZE`` rlimit set via
    ``preexec_fn`` -- which is unavailable on Windows and only bounds a single
    file descriptor's writes anyway -- with one mechanism that behaves
    identically on every platform, so this half of the boundary needs no
    platform branch at all.
    """

    assert process.stdout is not None
    written = 0
    while True:
        chunk = process.stdout.read(_HOOK_READ_CHUNK_BYTES)
        if not chunk:
            return True
        remaining = maximum_bytes - written
        if remaining <= 0:
            return False
        if len(chunk) > remaining:
            log.write(chunk[:remaining])
            return False
        log.write(chunk)
        written += len(chunk)


def _validate_hook_evidence(
    kind: str, hook: dict[str, Any], evidence: dict[str, Any]
) -> None:
    required = {"version", "kind", "passed", "cases", "failures"}
    if not required.issubset(evidence) or evidence.get("version") != 1:
        raise SecurityContractError("security hook evidence schema is invalid")
    if evidence.get("kind") != kind or evidence.get("passed") is not True:
        raise SecurityContractError("security hook did not pass")
    cases = evidence.get("cases")
    failures = evidence.get("failures")
    if (
        not isinstance(cases, int)
        or cases < hook["min_cases"]
        or not isinstance(failures, int)
        or failures != 0
    ):
        raise SecurityContractError("security hook evidence threshold was not met")
    if kind == "fuzz" and evidence.get("crashes") != 0:
        raise SecurityContractError("fuzz hook reported a crash")
    if (
        kind == "authenticated_negative"
        and evidence.get("unauthorized_acceptances") != 0
    ):
        raise SecurityContractError("authenticated-negative hook reported a bypass")


def run_hook(root: Path, contract: dict[str, Any], kind: str, result_root: str) -> None:
    """Run one declared hook without a shell and require bounded passing evidence."""

    if kind not in HOOK_KINDS:
        raise SecurityContractError("security hook kind is invalid")
    results = root.joinpath(result_root)
    if Path(result_root).is_absolute() or ".." in Path(result_root).parts:
        raise SecurityContractError("security result root is invalid")
    results.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        resolved_results = results.resolve(strict=True)
        resolved_results.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SecurityContractError("security result root is invalid") from exc
    if results.is_symlink() or resolved_results != results or not results.is_dir():
        raise SecurityContractError("security result root is invalid")
    hook = contract["hooks"][kind]
    evidence = root.joinpath(hook["evidence"])
    try:
        evidence.relative_to(resolved_results)
    except ValueError as exc:
        raise SecurityContractError(
            "security hook evidence must stay in the result root"
        ) from exc
    evidence.unlink(missing_ok=True)
    log_path = results.joinpath(f"{kind}.log")
    timed_out = False
    within_bound = True
    return_code: int | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                hook["argv"],
                cwd=root,
                env=_hook_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                **_hook_session_kwargs(),
            )
            job = _bind_hook_job(process.pid) if os.name == "nt" else None
            try:

                def _on_timeout() -> None:
                    nonlocal timed_out
                    timed_out = True
                    _terminate_hook_tree(process, job)

                watchdog = threading.Timer(hook["timeout_seconds"], _on_timeout)
                watchdog.start()
                try:
                    within_bound = _pump_bounded_output(
                        process, log, maximum_bytes=MAX_HOOK_OUTPUT_BYTES
                    )
                    if not within_bound:
                        _terminate_hook_tree(process, job)
                    return_code = process.wait()
                finally:
                    watchdog.cancel()
            finally:
                if job is not None:
                    _close_hook_job(job)
    except SecurityContractError:
        raise
    except Exception as exc:
        raise SecurityContractError("security hook execution failed") from exc
    if timed_out:
        raise SecurityContractError("security hook exceeded its time boundary")
    if not within_bound:
        raise SecurityContractError("security hook exceeded its output boundary")
    if return_code != 0:
        raise SecurityContractError("security hook returned a failure")
    evidence_file = _relative_file(
        root,
        evidence.relative_to(root).as_posix(),
        maximum_bytes=MAX_EVIDENCE_BYTES,
    )
    try:
        evidence_file.relative_to(resolved_results)
    except ValueError as exc:
        raise SecurityContractError(
            "security hook evidence must stay in the result root"
        ) from exc
    evidence_value = _read_json(evidence_file, maximum_bytes=MAX_EVIDENCE_BYTES)
    _validate_hook_evidence(kind, hook, evidence_value)


def _component_licenses(component: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
    """Return bounded declarations and whether any declaration was malformed."""

    licenses = component.get("licenses")
    if not isinstance(licenses, list) or not licenses:
        return (), False
    if len(licenses) > MAX_LICENSE_DECLARATIONS:
        return (), True
    values: list[str] = []
    malformed = False
    for declaration in licenses:
        if not isinstance(declaration, dict):
            malformed = True
            continue
        if "expression" in declaration:
            value = declaration["expression"]
        else:
            license_value = declaration.get("license")
            if not isinstance(license_value, dict):
                malformed = True
                continue
            identifier = license_value.get("id")
            name = license_value.get("name")
            value = identifier if isinstance(identifier, str) and identifier else name
        if (
            not isinstance(value, str)
            or not value
            or not _bounded_utf8(value, MAX_LICENSE_EXPRESSION_BYTES)
        ):
            malformed = True
            continue
        values.append(value)
    return tuple(values), malformed


def check_licenses(
    root: Path,
    contract: dict[str, Any],
    sbom_reference: str,
    output_reference: str,
) -> None:
    """Apply bounded SPDX semantics and a fail-closed policy to CycloneDX."""

    sbom = _read_json(
        _relative_file(root, sbom_reference, maximum_bytes=MAX_SBOM_BYTES),
        maximum_bytes=MAX_SBOM_BYTES,
    )
    if sbom.get("bomFormat") != "CycloneDX":
        raise SecurityContractError("software bill of materials is not CycloneDX")
    components = sbom.get("components", [])
    if not isinstance(components, list) or len(components) > MAX_COMPONENTS:
        raise SecurityContractError("software bill of materials is invalid")
    policy = contract["license_policy"]
    allowed = set(policy["allowed"])
    allowed_exceptions = set(policy["allowed_exceptions"])
    denied = set(policy["denied"])
    unknown = 0
    violations = 0
    for component in components:
        if not isinstance(component, dict):
            raise SecurityContractError(
                "software bill of materials component is invalid"
            )
        licenses, malformed = _component_licenses(component)
        if not licenses:
            if malformed:
                violations += 1
            else:
                unknown += 1
            continue
        if malformed or any(
            not _spdx_expression_is_allowed(
                value,
                allowed=allowed,
                allowed_exceptions=allowed_exceptions,
                denied=denied,
            )
            for value in licenses
        ):
            violations += 1
    passed = violations == 0 and unknown == 0
    if Path(output_reference).is_absolute() or ".." in Path(output_reference).parts:
        raise SecurityContractError("license evidence path is invalid")
    output = root.joinpath(output_reference)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise SecurityContractError("license evidence path is invalid") from exc
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        output.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SecurityContractError("license evidence path is invalid") from exc
    if output.parent.is_symlink():
        raise SecurityContractError("license evidence path is invalid")
    output.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "license_policy",
                "passed": passed,
                "components": len(components),
                "unknown": unknown,
                "violations": violations,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not passed:
        raise SecurityContractError(
            "software bill of materials violates license policy"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded repository security contract")
    parser.add_argument("--contract", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("validate")
    hook_parser = subparsers.add_parser("run-hook")
    hook_parser.add_argument("--kind", required=True, choices=HOOK_KINDS)
    hook_parser.add_argument("--result-root", default="security-results")
    license_parser = subparsers.add_parser("check-licenses")
    license_parser.add_argument("--sbom", required=True)
    license_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    try:
        contract = load_contract(root, args.contract)
        if args.action == "run-hook":
            run_hook(root, contract, args.kind, args.result_root)
        elif args.action == "check-licenses":
            check_licenses(root, contract, args.sbom, args.output)
    except SecurityContractError as exc:
        print(f"security contract failed: {exc}", file=sys.stderr)
        return 1
    print(f"security contract {args.action}: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
