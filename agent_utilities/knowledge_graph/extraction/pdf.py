"""One bounded, license-approved PDF text extraction path."""

from __future__ import annotations

import math
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

MAX_PDF_BYTES = 256 * 1024 * 1024
MAX_PDF_PAGES = 5_000
MAX_PDF_CHARS = 8_000_000
MAX_PDF_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_PDF_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
MAX_PDF_CPU_SECONDS = 15
MAX_PDF_WALL_SECONDS = 20.0


def _apply_worker_limits(cpu_seconds: int, address_space_bytes: int) -> None:
    """Apply best-effort Unix process limits without weakening wall-time control."""

    try:
        import resource
    except ImportError:  # Windows has no resource module.
        return
    for resource_id, requested in (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, address_space_bytes),
    ):
        try:
            _soft, hard = resource.getrlimit(resource_id)
            limit = (
                requested if hard == resource.RLIM_INFINITY else min(requested, hard)
            )
            resource.setrlimit(resource_id, (limit, limit))
        except (OSError, ValueError):
            continue


def _pdf_worker(
    connection: Any,
    source: str,
    max_bytes: int,
    max_pages: int,
    max_chars: int,
    max_output_bytes: int,
    cpu_seconds: int,
    address_space_bytes: int,
) -> None:
    """Extract inside the killable child and emit one bounded byte message."""

    try:
        _apply_worker_limits(cpu_seconds, address_space_bytes)
        path = Path(source)
        if path.is_symlink():
            return
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
                return
            from pypdf import PdfReader

            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted or len(reader.pages) > max_pages:
                return
            parts: list[str] = []
            remaining = max_chars
            for page in reader.pages:
                if remaining <= 0:
                    break
                text = page.extract_text() or ""
                if len(text) > remaining:
                    text = text[:remaining]
                parts.append(text)
                remaining -= len(text)
            payload = "\n".join(parts)[:max_chars].encode("utf-8")
            if len(payload) > max_output_bytes:
                return
            connection.send_bytes(payload)
    except Exception:  # noqa: BLE001 - detail-free child failure boundary
        try:
            connection.send_bytes(b"")
        except Exception:  # noqa: BLE001 - parent may already have timed out
            pass
    finally:
        connection.close()


def _stop_worker(process: multiprocessing.Process) -> None:
    process.join(timeout=0.2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)


def _valid_limits(
    *,
    max_bytes: int,
    max_pages: int,
    max_chars: int,
    max_output_bytes: int,
    cpu_seconds: int,
    address_space_bytes: int,
    timeout_seconds: float,
) -> bool:
    return (
        isinstance(max_bytes, int)
        and 1 <= max_bytes <= MAX_PDF_BYTES
        and isinstance(max_pages, int)
        and 1 <= max_pages <= MAX_PDF_PAGES
        and isinstance(max_chars, int)
        and 1 <= max_chars <= MAX_PDF_CHARS
        and isinstance(max_output_bytes, int)
        and 1 <= max_output_bytes <= MAX_PDF_OUTPUT_BYTES
        and isinstance(cpu_seconds, int)
        and 1 <= cpu_seconds <= MAX_PDF_CPU_SECONDS
        and isinstance(address_space_bytes, int)
        and 64 * 1024 * 1024 <= address_space_bytes <= MAX_PDF_ADDRESS_SPACE_BYTES
        and isinstance(timeout_seconds, (int, float))
        and math.isfinite(timeout_seconds)
        and 0 < timeout_seconds <= MAX_PDF_WALL_SECONDS
    )


def read_pdf_text(
    source: str | Path,
    *,
    max_bytes: int = MAX_PDF_BYTES,
    max_pages: int = MAX_PDF_PAGES,
    max_chars: int = MAX_PDF_CHARS,
    max_output_bytes: int = MAX_PDF_OUTPUT_BYTES,
    cpu_seconds: int = MAX_PDF_CPU_SECONDS,
    address_space_bytes: int = MAX_PDF_ADDRESS_SPACE_BYTES,
    timeout_seconds: float = MAX_PDF_WALL_SECONDS,
) -> str:
    """Extract text in a resource-limited worker, failing closed on timeout.

    The caller process never parses document structures. Symlinks, non-files,
    oversized inputs, excessive page counts, encrypted or malformed documents,
    worker crashes, oversized IPC output, and deadline expiration all return an
    empty string. Callers may lower any limit but cannot raise the hard ceilings.
    """

    if not _valid_limits(
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_chars=max_chars,
        max_output_bytes=max_output_bytes,
        cpu_seconds=cpu_seconds,
        address_space_bytes=address_space_bytes,
        timeout_seconds=timeout_seconds,
    ):
        return ""
    path = Path(source)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
            return ""
    except OSError:
        return ""

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdf_worker,
        args=(
            child,
            str(path),
            max_bytes,
            max_pages,
            max_chars,
            max_output_bytes,
            cpu_seconds,
            address_space_bytes,
        ),
        daemon=True,
    )
    try:
        process.start()
        child.close()
        if not parent.poll(timeout_seconds):
            return ""
        try:
            payload = parent.recv_bytes(maxlength=max_output_bytes)
        except (EOFError, OSError):
            return ""
        return payload.decode("utf-8", errors="ignore")[:max_chars]
    except (AssertionError, OSError, RuntimeError):
        return ""
    finally:
        child.close()
        parent.close()
        if process.pid is not None:
            _stop_worker(process)
