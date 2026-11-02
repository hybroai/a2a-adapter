"""Helpers shared by the Python framework adapters (LangChain, LangGraph,
CrewAI, callable)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, TypeVar

T = TypeVar("T")


async def run_with_timeout(
    awaitable: Awaitable[T], timeout: float | None, what: str,
) -> T:
    """Await ``awaitable``, raising ``TimeoutError("<what> timed out after
    N seconds")`` if it takes longer than ``timeout`` (``None``: no limit).

    A ``TimeoutError`` raised by the awaitable itself propagates unchanged.
    """
    scope = asyncio.timeout(timeout)
    try:
        async with scope:
            return await awaitable
    except TimeoutError as e:
        if scope.expired():
            raise TimeoutError(f"{what} timed out after {timeout} seconds") from e
        raise


async def iter_with_timeout(
    iterable: AsyncIterable[T], timeout: float | None, what: str,
) -> AsyncIterator[T]:
    """Yield from ``iterable`` with ``timeout`` seconds for the whole stream.

    Raises the same ``TimeoutError`` as :func:`run_with_timeout` once the
    deadline passes, and closes the underlying iterator however iteration
    ends. Time the consumer spends between items counts toward the deadline.
    """
    iterator = aiter(iterable)
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    try:
        while True:
            scope = asyncio.timeout_at(deadline)
            try:
                async with scope:
                    item = await anext(iterator)
            except StopAsyncIteration:
                return
            except TimeoutError as e:
                if scope.expired():
                    raise TimeoutError(f"{what} timed out after {timeout} seconds") from e
                raise
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def content_to_text(content: Any, sep: str = " ") -> str:
    """Extract text from a LangChain message ``content``.

    ``content`` is either a string or a list of content blocks: strings,
    dicts such as ``{"type": "text", "text": ...}`` (newer chat models), or
    objects with a ``text`` attribute. Non-text blocks (tool calls, images,
    reasoning) are skipped. Text blocks are joined with ``sep``; use ``""``
    for streaming chunks, which are deltas.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type", "text") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return sep.join(parts)
    return str(content)
