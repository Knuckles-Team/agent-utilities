import asyncio
import json

import httpx

from agent_utilities.security.secrets_client import SecretsConfig, create_secrets_client


async def main():
    config = SecretsConfig(
        backend="sqlite", sqlite_path="~/.agent-utilities/secrets.db"
    )
    client = create_secrets_client(config)
    try:
        val = client.get("xai/oauth_tokens")
        if val:
            token = json.loads(val).get("access_token")
        else:
            token = None
    except Exception as e:
        print("Could not get secret:", e)
        return

    if not token:
        print("No token")
        return

    payload = {
        "model": "grok-4.20-reasoning",
        "input": [
            {
                "role": "user",
                "content": "Detailed content of the X post status ID 2057129225593741768 canonical URL https://x.com/i/status/2057129225593741768",
            }
        ],
        "tools": [{"type": "x_search"}],
        "store": False,
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    print("Sending request...")
    try:
        with httpx.Client(timeout=60.0) as http_client:
            resp = http_client.post(
                "https://api.x.ai/v1/responses", headers=headers, json=payload
            )
            print("Status:", resp.status_code)
            print("Response:", resp.json())
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
