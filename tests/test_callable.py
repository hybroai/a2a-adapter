"""Tests for CallableAdapter (v0.2)."""

import asyncio
import threading
import time

import pytest

from a2a_adapter.integrations.callable import CallableAdapter


# ── Invoke ───────────────────────────────────────────────────


class TestCallableAdapterInvoke:
    async def test_string_response(self):
        async def echo(inputs):
            return f"echo: {inputs['message']}"

        adapter = CallableAdapter(func=echo)
        result = await adapter.invoke("hello")
        assert result == "echo: hello"

    async def test_dict_response_with_known_key(self):
        async def agent(inputs):
            return {"response": "answer"}

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert result == "answer"

    async def test_dict_response_fallback_to_json(self):
        async def agent(inputs):
            return {"custom_key": 42}

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert "custom_key" in result
        assert "42" in result

    async def test_non_string_cast(self):
        async def agent(inputs):
            return 42

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert result == "42"

    async def test_context_id_passed_through(self):
        captured = {}

        async def agent(inputs):
            captured.update(inputs)
            return "ok"

        adapter = CallableAdapter(func=agent)
        await adapter.invoke("hi", context_id="ctx-123")
        assert captured["context_id"] == "ctx-123"


# ── Streaming ────────────────────────────────────────────────


class TestCallableAdapterStream:
    async def test_streaming_yields_chunks(self):
        async def gen(inputs):
            for word in inputs["message"].split():
                yield word

        adapter = CallableAdapter(func=gen, streaming=True)
        chunks = []
        async for chunk in adapter.stream("hello world"):
            chunks.append(chunk)
        assert chunks == ["hello", "world"]

    async def test_non_string_chunks_cast(self):
        async def gen(inputs):
            yield 1
            yield 2

        adapter = CallableAdapter(func=gen, streaming=True)
        chunks = []
        async for chunk in adapter.stream("go"):
            chunks.append(chunk)
        assert chunks == ["1", "2"]


# ── supports_streaming ───────────────────────────────────────


class TestCallableAdapterStreaming:
    def test_not_streaming_by_default(self):
        async def f(inputs):
            return "ok"

        adapter = CallableAdapter(func=f)
        assert adapter.supports_streaming() is False

    def test_streaming_when_enabled(self):
        async def f(inputs):
            yield "ok"

        adapter = CallableAdapter(func=f, streaming=True)
        assert adapter.supports_streaming() is True


# ── Metadata ─────────────────────────────────────────────────


class TestCallableAdapterMetadata:
    def test_default_metadata(self):
        async def my_func(inputs):
            return "ok"

        adapter = CallableAdapter(func=my_func)
        meta = adapter.get_metadata()
        assert meta.name == "my_func"
        assert meta.streaming is False

    def test_custom_metadata(self):
        async def f(inputs):
            return "ok"

        adapter = CallableAdapter(
            func=f,
            name="TestAgent",
            description="A test agent",
            streaming=True,
            skills=[{"id": "s1", "name": "skill1", "description": "d"}],
            provider={"organization": "Acme"},
            documentation_url="https://example.com/docs",
            icon_url="https://example.com/icon.png",
        )
        meta = adapter.get_metadata()
        assert meta.name == "TestAgent"
        assert meta.description == "A test agent"
        assert meta.streaming is True
        assert len(meta.skills) == 1
        assert meta.provider == {"organization": "Acme"}
        assert meta.documentation_url == "https://example.com/docs"
        assert meta.icon_url == "https://example.com/icon.png"


# ── Sync callables and timeout ───────────────────────────────


class TestCallableAdapterSyncAndTimeout:
    async def test_sync_callable_runs_in_a_thread(self):
        main_thread = threading.get_ident()
        seen = {}

        def agent(inputs):
            seen["thread"] = threading.get_ident()
            return f"sync: {inputs['message']}"

        adapter = CallableAdapter(func=agent)
        assert await adapter.invoke("hi") == "sync: hi"
        assert seen["thread"] != main_thread

    async def test_sync_callable_does_not_block_the_loop(self):
        def slow(inputs):
            time.sleep(0.2)
            return "done"

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            assert await CallableAdapter(func=slow).invoke("x") == "done"
        finally:
            task.cancel()
        assert ticks >= 5

    async def test_sync_callable_returning_awaitable_is_awaited(self):
        async def agent(inputs):
            return "from coroutine"

        adapter = CallableAdapter(func=lambda inputs: agent(inputs))
        assert await adapter.invoke("x") == "from coroutine"

    async def test_async_callable_object_runs_on_the_loop(self):
        class Agent:
            async def __call__(self, inputs):
                asyncio.get_running_loop()  # raises off the loop thread
                return "callable object"

        assert await CallableAdapter(func=Agent()).invoke("x") == "callable object"

    async def test_timeout_defaults_to_none(self):
        async def f(inputs):
            return "x"

        assert CallableAdapter(func=f).timeout is None

    async def test_async_timeout(self):
        async def slow(inputs):
            await asyncio.sleep(1)

        adapter = CallableAdapter(func=slow, timeout=0.05)
        with pytest.raises(TimeoutError, match="callable timed out after 0.05 seconds"):
            await adapter.invoke("x")

    async def test_sync_timeout(self):
        def slow(inputs):
            time.sleep(0.3)

        adapter = CallableAdapter(func=slow, timeout=0.05)
        with pytest.raises(TimeoutError, match="callable timed out after 0.05 seconds"):
            await adapter.invoke("x")

    async def test_stream_timeout(self):
        async def gen(inputs):
            for i in range(10):
                await asyncio.sleep(0.03)
                yield str(i)

        adapter = CallableAdapter(func=gen, streaming=True, timeout=0.1)
        with pytest.raises(TimeoutError, match="callable timed out after 0.1 seconds"):
            _ = [c async for c in adapter.stream("x")]


async def test_dict_response_with_non_json_values():
    import json
    from datetime import datetime

    when = datetime(2026, 1, 2, 3, 4, 5)

    async def agent(inputs):
        return {"when": when}

    result = await CallableAdapter(func=agent).invoke("q")
    assert json.loads(result) == {"when": str(when)}


async def test_sync_in_thread_false_calls_on_the_loop():
    """A sync wrapper that needs the running loop can opt out of the thread."""

    async def agent(inputs):
        return "from task"

    adapter = CallableAdapter(
        func=lambda inputs: asyncio.ensure_future(agent(inputs)),
        sync_in_thread=False,
    )
    assert await adapter.invoke("x") == "from task"
