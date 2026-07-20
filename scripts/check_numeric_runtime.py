#!/usr/bin/env python3
"""Fail closed unless the installed native numeric runtime is usable."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import stat
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def main() -> int:
    try:
        declared = [
            Requirement(value)
            for value in importlib.metadata.requires("agent-utilities") or ()
            if canonicalize_name(Requirement(value).name) == "epistemic-graph"
        ]
        engine_version = Version(importlib.metadata.version("epistemic-graph"))
        engine_metadata = importlib.metadata.metadata("epistemic-graph")
        provided_extras = {
            canonicalize_name(value)
            for value in engine_metadata.get_all("Provides-Extra") or ()
        }
        server_name = "epistemic-graph-server.exe" if os.name == "nt" else "epistemic-graph-server"
        server = Path(sys.executable).with_name(server_name)
        server_metadata = server.lstat()
        engine_numeric = importlib.import_module("epistemic_graph.numeric")
        utilities_numeric = importlib.import_module("agent_utilities.numeric")
        kernel_ready = getattr(engine_numeric, "__kernel__", None) == "eg-numeric"
        computation_ready = utilities_numeric.xp.sum([1.0, 2.0, 3.0]) == 6.0
        release_binding_ready = (
            len(declared) == 1
            and declared[0].extras == {"full"}
            and declared[0].url is None
            and declared[0].marker is None
            and engine_version in declared[0].specifier
            and {"full", "numeric"} <= provided_extras
            and stat.S_ISREG(server_metadata.st_mode)
            and not server.is_symlink()
            and os.access(server, os.X_OK)
        )
    except Exception:
        print("numeric_runtime_gate=failed reason=import_or_compute")
        return 1
    if not (kernel_ready and computation_ready and release_binding_ready):
        print("numeric_runtime_gate=failed reason=contract")
        return 1
    print("numeric_runtime_gate=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
