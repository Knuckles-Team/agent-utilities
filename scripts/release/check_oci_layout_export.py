#!/usr/bin/env python3
"""Fail closed when the canonical OCI-layout export contract drifts."""

from __future__ import annotations

import ast
import hashlib
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts" / "release" / "export_oci_layout.py"
DOCUMENTATION = ROOT / "docs" / "release" / "oci-layout-export.md"
TESTS = ROOT / "tests" / "unit" / "release" / "test_export_oci_layout.py"

_REQUIRED_FUNCTIONS = {
    "_open_private_parent",
    "_resolve_container_cli",
    "_run_container_export",
    "_assert_metadata_private",
    "validate_oci_layout",
    "export_oci_layout",
    "main",
}
_REQUIRED_SOURCE = (
    '"podman"',
    '"oci-archive"',
    '"/proc/self/fd/{cli_descriptor}"',
    "os.O_EXCL",
    'getattr(os, "O_NOFOLLOW", 0)',
    "os.link(",
    "follow_symlinks=False",
    "subprocess.DEVNULL",
    "pass_fds=(cli_descriptor,)",
    "start_new_session=True",
    "preexec_fn=_child_limits",
    '"archive_metadata_privacy_violation"',
    '"archive_blob_digest_mismatch"',
    '"output_exists"',
    '"oci-layout-export-status/1"',
)
_REQUIRED_DOCUMENTATION = (
    "```mermaid",
    "generate_component_evidence.py",
    "digest-addressed",
    "private directory",
    "does not invoke a shell",
)
_REQUIRED_TESTS = (
    "test_export_is_no_replace_private_and_shell_free",
    "test_mutable_image_reference_is_rejected_before_invocation",
    "test_output_symlink_and_symlink_parent_are_rejected",
    "test_invalid_oci_archives_are_rejected_without_publication",
    "test_metadata_privacy_violation_is_rejected",
    "test_failure_status_is_bounded_and_path_free",
)


def _read(path: Path) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_size > 2 * 1024 * 1024
    ):
        raise ValueError("contract input is not a bounded regular file")
    return path.read_text(encoding="utf-8")


def _validate_ast(source: str) -> None:
    tree = ast.parse(source)
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    if not _REQUIRED_FUNCTIONS.issubset(functions):
        raise ValueError("exporter functions drifted")
    popen_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"
    ]
    if len(popen_calls) != 1:
        raise ValueError("export must have one container process boundary")
    keywords = {item.arg: item.value for item in popen_calls[0].keywords if item.arg}
    shell = keywords.get("shell")
    if not isinstance(shell, ast.Constant) or shell.value is not False:
        raise ValueError("container process boundary must disable shell execution")
    forbidden = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr)
        in {("os", "system"), ("subprocess", "run"), ("subprocess", "call")}
    }
    if forbidden:
        raise ValueError("exporter contains a second process path")


def main() -> int:
    try:
        exporter = _read(EXPORTER)
        documentation = _read(DOCUMENTATION)
        tests = _read(TESTS)
        _validate_ast(exporter)
        if any(token not in exporter for token in _REQUIRED_SOURCE):
            raise ValueError("exporter security contract drifted")
        if any(token not in documentation for token in _REQUIRED_DOCUMENTATION):
            raise ValueError("exporter documentation contract drifted")
        if any(token not in tests for token in _REQUIRED_TESTS):
            raise ValueError("exporter negative test matrix drifted")
        digest = hashlib.sha256(
            "\0".join((exporter, documentation, tests)).encode("utf-8")
        ).hexdigest()
    except Exception:
        print(
            json.dumps(
                {"error": "OciLayoutExportContractInvalid", "ok": False},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"digest": digest, "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
