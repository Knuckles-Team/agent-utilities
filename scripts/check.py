import asyncio
import os
import sys
from pathlib import Path


async def main():
    repository_manager = Path(
        os.environ.get("REPOSITORY_MANAGER_ROOT", Path(__file__).resolve().parents[2] / "repository-manager")
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys, os; print(sys.executable); print(os.environ.get('VIRTUAL_ENV'))",
        cwd=repository_manager,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    print("STDOUT:", stdout.decode())
    print("STDERR:", stderr.decode())


asyncio.run(main())
