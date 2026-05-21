import asyncio

import httpx


async def main():
    url = "http://10.0.0.18:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen/qwen3.5-9b",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 10,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            print("Sending request to LM Studio...")
            res = await client.post(url, json=payload, headers=headers)
            print(f"Status code: {res.status_code}")
            print(f"Response: {res.json()}")
        except Exception as e:
            print(f"Request failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
