import asyncio

import httpx


async def verify_acp():
    base_url = "http://localhost:8001"

    print(f"Verifying ACP endpoint at {base_url}...")

    async with httpx.AsyncClient() as client:
        # 1. Create session
        try:
            print("1. Creating session...")
            res = await client.post(f"{base_url}/acp/sessions")
            if res.status_code != 200:
                print(f"FAILED: /sessions returned {res.status_code}")
                return

            data = res.json()
            session_id = data.get("session_id")
            print(f"SUCCESS: Session created: {session_id}")

            # 2. Test RPC
            print("2. Testing RPC (prompt)...")
            rpc_payload = {
                "jsonrpc": "2.0",
                "method": "prompt",
                "params": {"text": "hello"},
                "id": 1,
            }
            res = await client.post(
                f"{base_url}/acp/rpc/{session_id}", json=rpc_payload
            )
            if res.status_code != 200:
                print(f"FAILED: /rpc returned {res.status_code}")
                return

            print(f"SUCCESS: RPC accepted: {res.json()}")

            # 3. Test Stream connectivity
            print("3. Testing Stream connectivity...")
            async with client.stream(
                "GET", f"{base_url}/acp/stream/{session_id}"
            ) as stream:
                if stream.status_code != 200:
                    print(f"FAILED: /stream returned {stream.status_code}")
                    return

                # Check first line
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        print(f"SUCCESS: Stream received event: {line}")
                        break
                    elif line.startswith("event: open"):
                        print("SUCCESS: Stream opened")
                        continue

            print("\nVerification Complete: ACP Adapter is functional!")

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    # Note: This requires the server to be running on 8001 with ENABLE_ACP=true
    # For now, this is a smoke test script for the user.
    asyncio.run(verify_acp())
