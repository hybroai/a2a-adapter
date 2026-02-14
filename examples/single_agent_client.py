"""
Example: A2A Client — Test Any Running Agent

This is a client example (not a server). Use it to test any A2A agent
started by the other examples.

Prerequisites:
- An A2A agent server running (e.g., python examples/n8n_agent.py)

Usage:
    python examples/single_agent_client.py
"""

import asyncio
import json

import httpx


async def send_message(url: str, text: str):
    """Send a message/send request to an A2A agent."""
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        return resp.json()


async def send_stream(url: str, text: str):
    """Send a message/stream request (SSE) to an A2A agent."""
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        for line in resp.text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                event = json.loads(line[5:].strip())
                yield event


async def main():
    agent_url = "http://localhost:9000"

    # 1. Fetch agent card
    print("Fetching agent card...")
    async with httpx.AsyncClient() as client:
        card = (await client.get(f"{agent_url}/.well-known/agent.json")).json()
    print(f"Agent: {card['name']} — {card.get('description', '')}")
    print()

    # 2. Send a message
    print("Sending message: 'What is 25 * 37 + 18?'")
    result = await send_message(agent_url, "What is 25 * 37 + 18?")
    print(f"Response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
