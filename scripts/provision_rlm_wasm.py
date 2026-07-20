#!/usr/bin/env python3
"""Provision the CPython-WASI payload for the RLM ``wasm`` sandbox tier (CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox).

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
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import platformdirs

# VMware wasm-language-runtimes CPython-3.12 WASI build (stdlib embedded; single file).
_PAYLOAD_URL = (
    "https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/"
    "python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm"
)
_PAYLOAD_SHA256 = "e5dc5a398b07b54ea8fdb503bf68fb583d533f10ec3f930963e02b9505f7a763"
_PAYLOAD_NAME = "python-3.12.0.wasm"
_MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)


def _validate_download_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower().rstrip(".") not in _ALLOWED_DOWNLOAD_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or len(url) > 8_192
    ):
        raise RuntimeError("payload download target was rejected")


class _ValidatedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self._redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        self._redirects += 1
        if self._redirects > 3:
            raise RuntimeError("payload download exceeded its redirect boundary")
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _tls_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    capath = os.environ.get("SSL_CERT_DIR")
    return ssl.create_default_context(cafile=cafile or None, capath=capath or None)


def _download_verified(target: Path) -> None:
    _validate_download_url(_PAYLOAD_URL)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_tls_context()),
        _ValidatedRedirects(),
    )
    temporary = target.with_name(target.name + ".part")
    temporary.unlink(missing_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    digest = hashlib.sha256()
    written = 0
    try:
        with opener.open(_PAYLOAD_URL, timeout=60) as response, os.fdopen(
            descriptor, "wb"
        ) as output:
            descriptor = -1
            _validate_download_url(response.geturl())
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > _MAX_PAYLOAD_BYTES:
                raise RuntimeError("payload exceeds its safe size boundary")
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_PAYLOAD_BYTES:
                    raise RuntimeError("payload exceeds its safe size boundary")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not written or digest.hexdigest() != _PAYLOAD_SHA256:
            raise RuntimeError("payload integrity verification failed")
        os.replace(temporary, target)
    except (OSError, ssl.SSLError, urllib.error.URLError, ValueError, RuntimeError):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("payload download or verification failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


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
    try:
        _download_verified(target)
    except RuntimeError:
        print("ERROR: payload download or verification failed", file=sys.stderr)
        return 1
    print(f"Provisioned and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
