"""
Live E2E test: start real Claude Code + Codex A2A servers, send messages.

This is NOT a CI test — it requires:
- `claude` CLI installed and authenticated
- `codex` CLI installed and authenticated
- Network access

Usage:
    cd a2a-adapters
    uv run python tests/e2e_live_test.py
"""

import asyncio
import json
import os
import socket
import sys
import uuid

import httpx
import uvicorn

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from a2a_adapter import ClaudeCodeAdapter, CodexAdapter, to_a2a


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def step(msg: str) -> None:
    print(f"\n  >> {msg}")


def ok(msg: str) -> None:
    print(f"     [OK] {msg}")


def fail(msg: str) -> None:
    print(f"     [FAIL] {msg}")


# ──── A2A JSON-RPC helpers ────


def make_send_request(message: str) -> dict:
    """Build a JSON-RPC message/send request."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
                "messageId": str(uuid.uuid4()),
            },
            "configuration": {
                "acceptedOutputModes": ["text"],
            },
        },
    }


def make_stream_request(message: str) -> dict:
    """Build a JSON-RPC message/stream request."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
                "messageId": str(uuid.uuid4()),
            },
            "configuration": {
                "acceptedOutputModes": ["text"],
            },
        },
    }


# ──── Server launcher ────


async def start_server(app, port: int) -> asyncio.Task:
    """Start uvicorn in background, return the task."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for server to be ready
    for _ in range(50):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=2)
                if r.status_code == 200:
                    return task, server
        except (httpx.ConnectError, httpx.ReadError):
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError(f"Server on port {port} did not start in time")


# ──── Tests ────


async def test_agent_card(base_url: str, name: str) -> None:
    step(f"GET {name} agent card")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{base_url}/.well-known/agent-card.json", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    card = r.json()
    assert "name" in card, "Agent card missing 'name'"
    assert "capabilities" in card, "Agent card missing 'capabilities'"
    ok(f"name={card['name']}, streaming={card.get('capabilities', {}).get('streaming')}")


async def test_message_send(base_url: str, name: str, message: str, timeout: float = 120) -> str:
    step(f"message/send to {name}: '{message}'")
    payload = make_send_request(message)
    async with httpx.AsyncClient() as c:
        r = await c.post(
            base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    body = r.json()

    # JSON-RPC response
    if "error" in body:
        fail(f"JSON-RPC error: {json.dumps(body['error'], indent=2)}")
        raise AssertionError(f"JSON-RPC error from {name}")

    result = body.get("result", {})
    status = result.get("status", {}).get("state", "unknown")
    ok(f"task state: {status}")

    status_msg = result.get("status", {}).get("message", {})

    # Extract text from artifacts or result
    # A2A uses "kind": "text" (not "type"), and parts have "kind" field
    text = ""
    artifacts = result.get("artifacts", [])
    for art in artifacts:
        for part in art.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
                text += part.get("text", "")

    # Also try status.message.parts
    if not text and status_msg:
        for part in status_msg.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
                text += part.get("text", "")

    if text:
        preview = text[:200].replace("\n", " ")
        ok(f"response ({len(text)} chars): {preview}...")
    else:
        fail("No text in response")
        print(f"     [DEBUG] full body: {json.dumps(body, indent=2)[:1000]}")

    return text


async def test_message_stream(base_url: str, name: str, message: str, timeout: float = 120) -> str:
    step(f"message/stream to {name}: '{message}'")
    payload = make_stream_request(message)
    chunks = []
    async with httpx.AsyncClient() as c:
        async with c.stream(
            "POST",
            base_url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            timeout=timeout,
        ) as response:
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        try:
                            event = json.loads(data)
                            chunks.append(event)
                        except json.JSONDecodeError:
                            pass

    # Extract final state
    final_state = "unknown"
    text_parts = []
    for chunk in chunks:
        result = chunk.get("result", {})
        kind = result.get("kind", "")
        state = result.get("status", {}).get("state")
        if state:
            final_state = state

        # artifact-update events (streaming): singular "artifact"
        if kind == "artifact-update":
            artifact = result.get("artifact", {})
            for part in artifact.get("parts", []):
                if part.get("kind") == "text" or part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

        # status-update with message.parts (final event)
        if kind == "status-update":
            status_msg = result.get("status", {}).get("message", {})
            if status_msg and not text_parts:
                for part in status_msg.get("parts", []):
                    if part.get("kind") == "text" or part.get("type") == "text":
                        text_parts.append(part.get("text", ""))

    ok(f"received {len(chunks)} SSE events, final state: {final_state}")

    text = "".join(text_parts)
    if text:
        preview = text[:200].replace("\n", " ")
        ok(f"response ({len(text)} chars): {preview}...")
    else:
        fail("No text in stream events")

    return text


# ──── Main ────


async def main():
    working_dir = os.getcwd()
    claude_port = find_free_port()
    codex_port = find_free_port()

    header("E2E Live Test: Claude Code + Codex A2A Adapters")
    print(f"  Working dir: {working_dir}")
    print(f"  Claude Code port: {claude_port}")
    print(f"  Codex port:       {codex_port}")

    # ── Create adapters ──
    header("1. Creating adapters")

    claude_adapter = ClaudeCodeAdapter(
        working_dir=working_dir,
        timeout=120,
        name="Claude Code E2E",
        description="E2E test Claude Code agent",
    )
    ok("ClaudeCodeAdapter created")

    codex_adapter = CodexAdapter(
        working_dir=working_dir,
        timeout=120,
        name="Codex E2E",
        description="E2E test Codex agent",
    )
    ok("CodexAdapter created")

    # ── Build A2A apps ──
    claude_url = f"http://127.0.0.1:{claude_port}"
    codex_url = f"http://127.0.0.1:{codex_port}"

    claude_app = to_a2a(claude_adapter, url=claude_url)
    codex_app = to_a2a(codex_adapter, url=codex_url)
    ok("A2A apps built")

    # ── Start servers ──
    header("2. Starting A2A servers")
    servers = []
    try:
        claude_task, claude_server = await start_server(claude_app, claude_port)
        servers.append(claude_server)
        ok(f"Claude Code server started on {claude_url}")

        codex_task, codex_server = await start_server(codex_app, codex_port)
        servers.append(codex_server)
        ok(f"Codex server started on {codex_url}")

        # ── Test agent cards ──
        header("3. Agent Card Discovery")
        await test_agent_card(claude_url, "Claude Code")
        await test_agent_card(codex_url, "Codex")

        # ── Test message/send ──
        header("4. message/send (invoke)")

        simple_prompt = "What is 2+2? Answer with just the number."

        claude_send_text = await test_message_send(
            claude_url, "Claude Code", simple_prompt, timeout=120
        )

        codex_send_text = await test_message_send(codex_url, "Codex", simple_prompt, timeout=120)

        # ── Test message/stream (Claude only, Codex doesn't support streaming) ──
        header("5. message/stream (stream)")

        stream_prompt = "What is 3+3? Answer with just the number."

        claude_stream_text = await test_message_stream(
            claude_url, "Claude Code", stream_prompt, timeout=120
        )

        # ── Verify Codex doesn't support streaming ──
        step("Verify Codex agent card reports streaming=false")
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{codex_url}/.well-known/agent-card.json", timeout=10)
            card = r.json()
            streaming = card.get("capabilities", {}).get("streaming", None)
            if streaming is False:
                ok("Codex correctly reports streaming=false")
            else:
                fail(f"Codex streaming={streaming}, expected false")

        # ── Summary ──
        header("6. Results Summary")
        results = {
            "Claude Code message/send": bool(claude_send_text),
            "Codex message/send": bool(codex_send_text),
            "Claude Code message/stream": bool(claude_stream_text),
        }
        all_pass = True
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status} - {test_name}")
            if not passed:
                all_pass = False

        if all_pass:
            print("\n  All tests passed!")
        else:
            print("\n  Some tests failed!")
            sys.exit(1)

    finally:
        header("Cleanup")
        for s in servers:
            s.should_exit = True
        await claude_adapter.close()
        await codex_adapter.close()
        ok("Servers and adapters shut down")


if __name__ == "__main__":
    asyncio.run(main())
