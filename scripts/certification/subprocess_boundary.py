"""Bounded, content-redacting subprocess transport for certification adapters."""

from __future__ import annotations

import math
import subprocess
import threading
from dataclasses import dataclass
from typing import BinaryIO

MAX_ADAPTER_INPUT_BYTES = 1_048_576
MAX_ADAPTER_OUTPUT_BYTES = 1_048_576


class AdapterBoundaryError(RuntimeError):
    """An adapter violated a fixed execution boundary; content is never included."""


@dataclass(frozen=True, slots=True)
class AdapterResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def run_bounded(
    command: list[str],
    *,
    payload: bytes | None = None,
    timeout: float,
    maximum_output_bytes: int = MAX_ADAPTER_OUTPUT_BYTES,
) -> AdapterResult:
    """Run one already-validated argv with a combined captured-output ceiling."""

    try:
        timeout_seconds = float(timeout)
    except (TypeError, ValueError) as exc:
        raise AdapterBoundaryError("adapter_boundary_invalid") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or isinstance(timeout, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 10_000
        or type(maximum_output_bytes) is not int
        or not 1 <= maximum_output_bytes <= MAX_ADAPTER_OUTPUT_BYTES
        or (
            payload is not None
            and (
                not isinstance(payload, bytes)
                or len(payload) > MAX_ADAPTER_INPUT_BYTES
            )
        )
    ):
        raise AdapterBoundaryError("adapter_boundary_invalid")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except OSError as exc:
        raise AdapterBoundaryError("adapter_start_failed") from exc

    stdout = bytearray()
    stderr = bytearray()
    output_size = 0
    output_lock = threading.Lock()
    overflow = threading.Event()
    stream_error = threading.Event()

    def drain(stream: BinaryIO, destination: bytearray) -> None:
        nonlocal output_size
        try:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    return
                with output_lock:
                    remaining = maximum_output_bytes - output_size
                    if remaining > 0:
                        retained = chunk[:remaining]
                        destination.extend(retained)
                        output_size += len(retained)
                    if len(chunk) > remaining:
                        overflow.set()
                if overflow.is_set():
                    process.kill()
                    return
        except OSError:
            if process.poll() is None:
                stream_error.set()
                process.kill()
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    readers = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    writer: threading.Thread | None = None
    if payload is not None:

        def write_payload() -> None:
            assert process.stdin is not None
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        writer = threading.Thread(target=write_payload, daemon=True)
        writer.start()

    timed_out = False
    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                raise AdapterBoundaryError("adapter_termination_failed") from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if writer is not None:
            writer.join(timeout=5)
        for reader in readers:
            reader.join(timeout=5)
    if timed_out:
        raise AdapterBoundaryError("adapter_timeout")
    if overflow.is_set():
        raise AdapterBoundaryError("adapter_output_limit")
    if stream_error.is_set() or any(reader.is_alive() for reader in readers):
        raise AdapterBoundaryError("adapter_stream_failed")
    return AdapterResult(
        returncode=returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
    )
