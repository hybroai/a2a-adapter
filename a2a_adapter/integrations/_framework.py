"""Helpers shared by the Python framework adapters (LangChain, LangGraph,
CrewAI, callable) and the n8n adapter's ``input_mapper``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Dict, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def run_with_timeout(
    awaitable: Awaitable[T],
    timeout: float | None,
    what: str,
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
    iterable: AsyncIterable[T],
    timeout: float | None,
    what: str,
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
                break
            except TimeoutError as e:
                if scope.expired():
                    raise TimeoutError(f"{what} timed out after {timeout} seconds") from e
                raise
            yield item
    except BaseException:
        # Keep the original error (or GeneratorExit) over a cleanup failure.
        try:
            await _aclose(iterator)
        except Exception:
            logger.warning("Error closing %s stream", what, exc_info=True)
        raise
    await _aclose(iterator)


async def _aclose(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        await aclose()


def apply_input_mapper(
    mapper: Callable[[str, str | None], Any],
    user_input: str,
    context_id: str | None,
) -> Dict[str, Any]:
    """Call a user ``input_mapper``; its errors fail the request.

    Raises ``RuntimeError("input_mapper failed: <type>: <message>")`` chained
    to the original exception, or ``TypeError`` if it returns a non-mapping.
    """
    try:
        mapped = mapper(user_input, context_id)
    except Exception as e:
        raise RuntimeError(f"input_mapper failed: {type(e).__name__}: {e}") from e
    if not isinstance(mapped, Mapping):
        raise TypeError(f"input_mapper must return a dict, got {type(mapped).__name__}")
    return dict(mapped)


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
