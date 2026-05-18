import asyncio
import subprocess
import sys


async def main():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys, os; print(sys.executable); print(os.environ.get('VIRTUAL_ENV'))",
        cwd="/home/apps/workspace/agent-packages/repository-manager",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    print("STDOUT:", stdout.decode())
    print("STDERR:", stderr.decode())


asyncio.run(main())
