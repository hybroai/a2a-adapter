"""Helpers shared by the Python framework adapters (LangChain, LangGraph,
CrewAI, callable)."""

from __future__ import annotations

from typing import Any


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
