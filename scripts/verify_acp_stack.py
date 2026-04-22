import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acp_validator")


def load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f".env not found at {env_path}")


async def validate_acp_stack():
    load_env()
    base_url = os.getenv("AGENT_URL", "http://localhost:8000")
    acp_url = f"{base_url}/acp"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Test Session Creation
        logger.info("Testing session creation...")
        try:
            resp = await client.post(f"{acp_url}/sessions")
            resp.raise_for_status()
            session_id = resp.json().get("session_id")
            logger.info(f"✅ Session created: {session_id}")
        except Exception as e:
            logger.error(f"❌ Session creation failed: {e}")
            return

        # 2. Test Mode Switching & Planning (Slash Command /plan)
        logger.info("Testing planning mode (/plan)...")
        try:
            # In ACP, modes are typically switched via RPC or as part of the message context
            # We'll send a request specifically targeting the 'plan' mode if supported by the client logic,
            # but for the server check, we verify the stream emits plan events.
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "sessionId": session_id,
                    "content": "/plan List all files in current directory and analyze them",
                    "modeId": "plan",
                },
                "id": 1,
            }
            # Note: The exact RPC method might vary by pydantic-acp version,
            # but we check the response/stream.
            resp = await client.post(f"{acp_url}/rpc/{session_id}", json=payload)
            logger.info(f"RPC Response: {resp.status_code}")

            # Check the stream for plan-updated events
            async with client.stream("GET", f"{acp_url}/stream/{session_id}") as stream:
                count = 0
                async for line in stream.aiter_lines():
                    if "data: " in line:
                        event = json.loads(line[6:])
                        if event.get("type") == "plan-updated":
                            logger.info(
                                f"✅ Received Plan Update: {event.get('plan').get('title')}"
                            )
                            count += 1
                        if event.get("type") == "text" and "DONE" in event.get(
                            "content", ""
                        ):
                            break
                    if count > 5:
                        break  # Sufficient evidence

                if count == 0:
                    logger.warning(
                        "⚠️ No native plan-updated events detected. Check if mode 'plan' is active."
                    )
                else:
                    logger.info(f"✅ Native ACP planning verified ({count} updates).")

        except Exception as e:
            logger.error(f"❌ Planning test failed: {e}")

        # 3. Test Approval Flow (Dangerous Tool)
        logger.info("Testing approval workflow...")
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "sessionId": session_id,
                    "content": "Delete all files in the workspace",  # Should trigger guard
                },
                "id": 2,
            }
            # We don't actually want to delete files, we just want to see if 'approval-required' is emitted.
            async with client.stream("GET", f"{acp_url}/stream/{session_id}") as stream:
                found_approval = False
                async for line in stream.aiter_lines():
                    if "data: " in line:
                        event = json.loads(line[6:])
                        if event.get("type") == "approval-required":
                            logger.info("✅ Approval request detected as expected.")
                            found_approval = True
                            break
                if not found_approval:
                    logger.warning(
                        "⚠️ No approval-required event detected for dangerous tool."
                    )
        except Exception as e:
            logger.error(f"❌ Approval test failed: {e}")

        # 4. Test Workspace Mirroring (MEMORY.md)
        logger.info("Testing workspace mirroring...")
        try:
            # We check if the file exists and has content after a plan was triggered
            # This assumes the server is running on this local machine
            mem_path = (
                Path(__file__).resolve().parent.parent
                / "agent_utilities"
                / "agent_data"
                / "MEMORY.md"
            )
            if mem_path.exists():
                content = mem_path.read_text()
                if "Agent Plan (Auto-generated from ACP State)" in content:
                    logger.info(
                        "✅ Workspace mirroring verified: MEMORY.md contains ACP plan."
                    )
                else:
                    logger.warning(
                        "⚠️ MEMORY.md exists but doesn't contain ACP plan header."
                    )
            else:
                logger.error(f"❌ MEMORY.md not found at {mem_path}")
        except Exception as e:
            logger.error(f"❌ Mirroring test failed: {e}")


if __name__ == "__main__":
    asyncio.run(validate_acp_stack())
