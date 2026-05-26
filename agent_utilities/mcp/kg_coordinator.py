import logging
import os
import socket
import subprocess
import sys
import time

import httpx
import psutil

logger = logging.getLogger("agent_utilities.mcp.kg_coordinator")

DEFAULT_KG_PORT = int(os.getenv("KG_SERVER_PORT", "8100"))
DEFAULT_KG_HOST = os.getenv("KG_SERVER_HOST", "127.0.0.1")


class KGCoordinator:
    """Coordinating helper for the centralized Knowledge Graph server process.

    CONCEPT:KG-1.0 - Centralized KG Coordination Protocol.
    """

    @staticmethod
    def is_port_open(host: str = DEFAULT_KG_HOST, port: int = DEFAULT_KG_PORT) -> bool:
        """Check if the TCP port is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except Exception:
                return False

    @classmethod
    def is_server_healthy(
        cls, host: str = DEFAULT_KG_HOST, port: int = DEFAULT_KG_PORT
    ) -> bool:
        """Verify the health of the running KG server by querying its HTTP interface."""
        if not cls.is_port_open(host, port):
            return False

        try:
            # Try to fetch the tools list or root to see if the HTTP server is responsive and healthy
            url = f"http://{host}:{port}/tools"
            httpx.get(url, timeout=3.0)
            # Any HTTP status code (including 404 Not Found) means the web server is alive and responding!
            return True
        except Exception as e:
            logger.debug(f"KG server health check failed: {e}")
            return False

    @classmethod
    def cleanup_rogue_instances(cls, port: int = DEFAULT_KG_PORT) -> None:
        """Find and terminate any process listening on the designated KG port or running the kg_server."""
        terminated = False
        # 1. Terminate process bound to the target port
        try:
            for conn in psutil.net_connections():
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and len(conn.laddr) > 1
                    and conn.laddr[1] == port
                ):
                    pid = conn.pid
                    if pid and pid != os.getpid():
                        try:
                            proc = psutil.Process(pid)
                            logger.warning(
                                f"Found rogue process {proc.name()} (PID: {pid}) on port {port}. Terminating..."
                            )
                            try:
                                proc.terminate()
                                proc.wait(timeout=1.0)
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                            terminated = True
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"psutil.net_connections lookup failed: {e}")

        # 2. Terminate any lingering kg_server.py processes
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if (
                    "agent_utilities.mcp.kg_server" in cmd_str
                    and proc.pid != os.getpid()
                ):
                    logger.warning(
                        f"Found lingering KG server process (PID: {proc.pid}). Terminating gracefully..."
                    )
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    terminated = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if terminated:
            time.sleep(0.5)  # Allow OS to free up port binding

    @classmethod
    def spawn_server(
        cls, host: str = DEFAULT_KG_HOST, port: int = DEFAULT_KG_PORT
    ) -> bool:
        """Spawn the centralized HTTP/SSE KG server in the background as a detached daemon process.

        Uses a file-based PID lock (G4) to prevent thundering herd when multiple
        agents attempt to spawn the server simultaneously. Only one process wins
        the lock election; all others wait for health checks.
        """
        import fcntl
        from pathlib import Path

        import platformdirs

        # G4: File-based PID lock for spawn election
        lock_dir = Path(platformdirs.user_runtime_dir("agent-utilities"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "kg_spawn.lock"

        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")  # noqa: SIM115
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            # Another agent is already spawning — just wait for health
            logger.info(
                "Another agent is spawning the KG server. Waiting for health..."
            )
            if lock_fd:
                lock_fd.close()
            for _ in range(25):
                time.sleep(0.2)
                if cls.is_server_healthy(host, port):
                    logger.info("KG server became healthy (spawned by another agent).")
                    return True
            logger.warning("KG server did not become healthy after waiting.")
            return False

        try:
            # We won the election — check one more time in case server came up
            if cls.is_server_healthy(host, port):
                logger.info("KG server already healthy (race resolved).")
                return True

            cls.cleanup_rogue_instances(port)

            logger.info(f"Starting centralized KG server daemon on {host}:{port}...")

            # Build command using uv run
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "agent_utilities.mcp.kg_server",
                "--transport",
                "sse",
                "--host",
                host,
                "--port",
                str(port),
            ]

            # Detach background process
            try:
                if sys.platform == "win32":
                    # Windows detached process creation flags
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    DETACHED_PROCESS = 0x00000008
                    subprocess.Popen(
                        cmd,
                        creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                    )
                else:
                    # Unix detached daemon using preexec_fn=os.setsid
                    subprocess.Popen(
                        cmd,
                        preexec_fn=os.setsid,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                    )
            except Exception as e:
                logger.error(f"Failed to spawn background centralized KG server: {e}")
                return False

            # Poll port until active and healthy (up to 5 seconds)
            for i in range(25):
                time.sleep(0.2)
                if cls.is_server_healthy(host, port):
                    logger.info(
                        "Centralized KG server daemon successfully initialized and healthy."
                    )
                    return True

            logger.error(
                "KG server spawned but failed to become healthy within timeout."
            )
            return False
        finally:
            # Release the spawn election lock
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    @classmethod
    def get_kg_client(
        cls, host: str = DEFAULT_KG_HOST, port: int = DEFAULT_KG_PORT
    ) -> None:
        """Check server state, auto-heal if unhealthy, and coordinate connection."""
        if not cls.is_server_healthy(host, port):
            cls.spawn_server(host, port)
