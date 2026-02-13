"""Tests for the v0.2 OpenClawAdapter.

Covers: invoke (subprocess execution), cancel (process kill), session ID
sanitization, command building, output extraction, MIME detection, metadata,
and integration with to_a2a / build_agent_card / load_adapter.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a_adapter.integrations.openclaw import (
    OpenClawAdapter,
    VALID_THINKING_LEVELS,
    _INVALID_SESSION_CHARS_RE,
    _SESSION_ID_MAX_LEN,
)
from a2a_adapter.base_adapter import AdapterMetadata
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Test: Constructor Validation ────


def test_valid_thinking_levels():
    for level in VALID_THINKING_LEVELS:
        adapter = OpenClawAdapter(thinking=level)
        assert adapter.thinking == level


def test_invalid_thinking_level():
    with pytest.raises(ValueError, match="Invalid thinking level"):
        OpenClawAdapter(thinking="invalid")


def test_auto_session_id():
    adapter = OpenClawAdapter()
    assert adapter.session_id.startswith("a2a-")


def test_custom_session_id():
    adapter = OpenClawAdapter(session_id="my-session")
    assert adapter.session_id == "my-session"


# ──── Test: Session ID Sanitization ────


def test_context_id_to_session_id_none():
    adapter = OpenClawAdapter(session_id="default")
    assert adapter._context_id_to_session_id(None) == "default"


def test_context_id_to_session_id_empty():
    adapter = OpenClawAdapter(session_id="default")
    assert adapter._context_id_to_session_id("") == "default"


def test_context_id_to_session_id_valid():
    adapter = OpenClawAdapter()
    result = adapter._context_id_to_session_id("my-context-123")
    assert result == "a2a-my-context-123"


def test_context_id_to_session_id_sanitizes():
    adapter = OpenClawAdapter()
    result = adapter._context_id_to_session_id("My Context ID!")
    assert result.startswith("a2a-")
    assert "!" not in result
    # Should be lowercase
    assert result == result.lower()


def test_context_id_to_session_id_truncates():
    adapter = OpenClawAdapter()
    long_id = "a" * 200
    result = adapter._context_id_to_session_id(long_id)
    assert len(result) <= _SESSION_ID_MAX_LEN


def test_context_id_to_session_id_all_invalid_chars():
    adapter = OpenClawAdapter(session_id="fallback")
    result = adapter._context_id_to_session_id("!!!")
    assert result == "fallback"


# ──── Test: Command Building ────


def test_build_command_basic():
    adapter = OpenClawAdapter(
        openclaw_path="/usr/bin/openclaw",
        thinking="medium",
    )
    cmd = adapter._build_command("hello world", "session-1")
    assert cmd[0] == "/usr/bin/openclaw"
    assert "agent" in cmd
    assert "--local" in cmd
    assert "--json" in cmd
    assert "--message" in cmd
    idx = cmd.index("--message")
    assert cmd[idx + 1] == "hello world"
    assert "--session-id" in cmd
    idx = cmd.index("--session-id")
    assert cmd[idx + 1] == "session-1"
    assert "--thinking" in cmd
    idx = cmd.index("--thinking")
    assert cmd[idx + 1] == "medium"


def test_build_command_with_agent_id():
    adapter = OpenClawAdapter(agent_id="research-agent")
    cmd = adapter._build_command("test", "session-1")
    assert "--agent" in cmd
    idx = cmd.index("--agent")
    assert cmd[idx + 1] == "research-agent"


def test_build_command_no_agent_id():
    adapter = OpenClawAdapter(agent_id=None)
    cmd = adapter._build_command("test", "session-1")
    assert "--agent" not in cmd


# ──── Test: Output Extraction ────


def test_extract_response_text_basic():
    output = {
        "payloads": [
            {"text": "Hello from OpenClaw"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert result == "Hello from OpenClaw"


def test_extract_response_text_multiple_payloads():
    output = {
        "payloads": [
            {"text": "Part 1"},
            {"text": "Part 2"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert "Part 1" in result
    assert "Part 2" in result


def test_extract_response_text_empty():
    output = {"payloads": []}
    result = OpenClawAdapter._extract_response_text(output)
    assert result == ""


def test_extract_response_text_no_payloads():
    output = {}
    result = OpenClawAdapter._extract_response_text(output)
    assert result == ""


def test_extract_response_text_skips_empty():
    output = {
        "payloads": [
            {"text": ""},
            {"text": "real content"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert result == "real content"


# ──── Test: MIME Type Detection ────


def test_mime_type_png():
    assert OpenClawAdapter._detect_mime_type("image.png") == "image/png"


def test_mime_type_jpeg():
    assert OpenClawAdapter._detect_mime_type("photo.jpg") == "image/jpeg"
    assert OpenClawAdapter._detect_mime_type("photo.jpeg") == "image/jpeg"


def test_mime_type_pdf():
    assert OpenClawAdapter._detect_mime_type("doc.pdf") == "application/pdf"


def test_mime_type_unknown():
    assert OpenClawAdapter._detect_mime_type("file.xyz") == "application/octet-stream"


def test_mime_type_case_insensitive():
    assert OpenClawAdapter._detect_mime_type("IMAGE.PNG") == "image/png"


# ──── Test: Invoke with mocked subprocess ────


@pytest.mark.asyncio
async def test_invoke_success():
    """Test invoke with a mocked subprocess that returns valid JSON."""
    adapter = OpenClawAdapter(session_id="test")

    mock_output = json.dumps({
        "payloads": [{"text": "mocked response"}],
        "meta": {},
    })

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(mock_output.encode(), b"")
    )
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await adapter.invoke("test message")

    assert result == "mocked response"


@pytest.mark.asyncio
async def test_invoke_nonzero_exit():
    """Test invoke raises on non-zero exit code."""
    adapter = OpenClawAdapter(session_id="test")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"error details")
    )
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="exit code 1"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_empty_output():
    """Test invoke raises on empty stdout."""
    adapter = OpenClawAdapter(session_id="test")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"")
    )
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="empty output"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_invalid_json():
    """Test invoke raises on invalid JSON output."""
    adapter = OpenClawAdapter(session_id="test")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(b"not json", b"")
    )
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_binary_not_found():
    """Test invoke raises FileNotFoundError when binary missing."""
    adapter = OpenClawAdapter(openclaw_path="/nonexistent/openclaw")

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("not found"),
    ):
        with pytest.raises(FileNotFoundError, match="not found"):
            await adapter.invoke("test")


# ──── Test: Cancel ────


@pytest.mark.asyncio
async def test_cancel_kills_process():
    """Cancel should kill the tracked subprocess."""
    adapter = OpenClawAdapter()

    mock_proc = MagicMock()
    mock_proc.returncode = None  # Still running
    mock_proc.kill = MagicMock()

    adapter._current_process = mock_proc
    await adapter.cancel()
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_no_process():
    """Cancel is a no-op when no process is running."""
    adapter = OpenClawAdapter()
    await adapter.cancel()  # Should not raise


@pytest.mark.asyncio
async def test_cancel_already_finished():
    """Cancel is a no-op when process already finished."""
    adapter = OpenClawAdapter()

    mock_proc = MagicMock()
    mock_proc.returncode = 0  # Already finished
    mock_proc.kill = MagicMock()

    adapter._current_process = mock_proc
    await adapter.cancel()
    mock_proc.kill.assert_not_called()


# ──── Test: Close ────


@pytest.mark.asyncio
async def test_close_kills_process():
    adapter = OpenClawAdapter()
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    adapter._current_process = mock_proc
    await adapter.close()
    mock_proc.kill.assert_called_once()


# ──── Test: Streaming NOT supported ────


def test_no_streaming():
    adapter = OpenClawAdapter()
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = OpenClawAdapter()
    meta = adapter.get_metadata()
    assert meta.name == "OpenClawAdapter"
    assert meta.streaming is False


def test_metadata_custom():
    adapter = OpenClawAdapter(
        name="My OpenClaw Agent",
        description="Does everything",
    )
    meta = adapter.get_metadata()
    assert meta.name == "My OpenClaw Agent"
    assert meta.description == "Does everything"


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = OpenClawAdapter(
        name="OpenClaw Agent",
        description="A test agent",
    )
    card = build_agent_card(adapter, url="http://localhost:9005")
    assert card.name == "OpenClaw Agent"
    assert card.capabilities.streaming is False


def test_to_a2a_builds_app():
    adapter = OpenClawAdapter()
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_openclaw():
    adapter = load_adapter({
        "adapter": "openclaw",
        "session_id": "test-session",
        "thinking": "high",
        "timeout": 120,
    })
    assert isinstance(adapter, OpenClawAdapter)
    assert adapter.session_id == "test-session"
    assert adapter.thinking == "high"
    assert adapter.timeout == 120


# ──── Test: Flat import ────


def test_flat_import():
    from a2a_adapter import OpenClawAdapter as OC
    assert OC is OpenClawAdapter


# ──── Test: Async context manager ────


@pytest.mark.asyncio
async def test_context_manager():
    async with OpenClawAdapter() as adapter:
        assert isinstance(adapter, OpenClawAdapter)
