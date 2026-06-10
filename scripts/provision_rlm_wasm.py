#!/usr/bin/env python3
"""Provision the CPython-WASI payload for the RLM ``wasm`` sandbox tier (CONCEPT:ORCH-1.38).

Downloads a self-contained ``python.wasm`` (CPython compiled to WASI, stdlib embedded) into the
platform cache where :func:`agent_utilities.rlm.sandboxes.wasm_backend._resolve_payload` looks
for it, verifying the SHA-256. The payload is ~25MB and intentionally kept out of the repo.

Usage::

    python scripts/provision_rlm_wasm.py            # download + verify into the cache
    python scripts/provision_rlm_wasm.py --print     # just print the target cache path

After provisioning, ``WasmSandbox().is_available()`` is True and the router will use the wasm
tier for self-contained compute. Alternatively, point ``$RLM_WASM_PYTHON`` at any python.wasm.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import platformdirs

# VMware wasm-language-runtimes CPython-3.12 WASI build (stdlib embedded; single file).
_PAYLOAD_URL = (
    "https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/"
    "python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm"
)
_PAYLOAD_SHA256 = "e5dc5a398b07b54ea8fdb503bf68fb583d533f10ec3f930963e02b9505f7a763"
_PAYLOAD_NAME = "python-3.12.0.wasm"


def _cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("agent-utilities")) / "rlm-wasm"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="print the target path and exit",
    )
    args = parser.parse_args()

    target = _cache_dir() / _PAYLOAD_NAME
    if args.print_only:
        print(target)
        return 0

    if target.is_file() and _sha256(target) == _PAYLOAD_SHA256:
        print(f"Already provisioned and verified: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {_PAYLOAD_URL}\n  -> {target}")
    urllib.request.urlretrieve(_PAYLOAD_URL, target)  # noqa: S310 - pinned GitHub release URL

    actual = _sha256(target)
    if actual != _PAYLOAD_SHA256:
        target.unlink(missing_ok=True)
        print(
            f"ERROR: checksum mismatch\n  expected {_PAYLOAD_SHA256}\n  actual   {actual}",
            file=sys.stderr,
        )
        return 1
    print(f"Provisioned and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
