"""
Unit tests for OpenClawAgentAdapter.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a_adapter.integrations.openclaw import OpenClawAgentAdapter, VALID_THINKING_LEVELS
from a2a.types import Message, MessageSendParams, Task, TaskState, TextPart, Role, Part
from a2a.server.tasks import InMemoryTaskStore


def make_message_send_params(text: str, context_id: str | None = None) -> MessageSendParams:
    """Helper to create MessageSendParams with correct A2A types."""
    return MessageSendParams(
        message=Message(
            message_id="test-msg-id",
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            context_id=context_id,
        )
    )


def make_openclaw_output(text: str, media_urls: list[str] | None = None) -> dict:
    """Helper to create OpenClaw JSON output format."""
    payload = {"text": text, "mediaUrl": None, "mediaUrls": media_urls or []}
    return {
        "payloads": [payload],
        "meta": {
            "durationMs": 1234,
            "agentMeta": {
                "sessionId": "test-session",
                "provider": "anthropic",
                "model": "claude-opus-4-5",
                "usage": {"input": 100, "output": 50},
            },
        },
    }


class TestOpenClawAdapterInit:
    """Tests for OpenClawAgentAdapter initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        adapter = OpenClawAgentAdapter()
        
        assert adapter.session_id.startswith("a2a-")
        assert adapter.agent_id is None
        assert adapter.thinking == "low"
        assert adapter.timeout == 600
        assert adapter.openclaw_path == "openclaw"
        assert adapter.working_directory is None
        assert adapter.env_vars == {}
        assert adapter.async_mode is True
        assert adapter.task_store is not None

    def test_custom_initialization(self):
        """Test custom initialization values."""
        adapter = OpenClawAgentAdapter(
            session_id="custom-session",
            agent_id="main",
            thinking="high",
            timeout=600,
            openclaw_path="/usr/local/bin/openclaw",
            working_directory="/tmp",
            env_vars={"CUSTOM_VAR": "value"},
            async_mode=False,
        )
        
        assert adapter.session_id == "custom-session"
        assert adapter.agent_id == "main"
        assert adapter.thinking == "high"
        assert adapter.timeout == 600
        assert adapter.openclaw_path == "/usr/local/bin/openclaw"
        assert adapter.working_directory == "/tmp"
        assert adapter.env_vars == {"CUSTOM_VAR": "value"}
        assert adapter.async_mode is False

    def test_invalid_thinking_level_raises(self):
        """Test that invalid thinking level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid thinking level"):
            OpenClawAgentAdapter(thinking="invalid")

    def test_valid_thinking_levels(self):
        """Test all valid thinking levels are accepted."""
        for level in VALID_THINKING_LEVELS:
            adapter = OpenClawAgentAdapter(thinking=level)
            assert adapter.thinking == level

    def test_auto_generates_session_id(self):
        """Test that session_id is auto-generated if not provided."""
        adapter1 = OpenClawAgentAdapter()
        adapter2 = OpenClawAgentAdapter()
        
        assert adapter1.session_id != adapter2.session_id
        assert adapter1.session_id.startswith("a2a-")
        assert adapter2.session_id.startswith("a2a-")


class TestOpenClawAdapterToFramework:
    """Tests for to_framework input mapping."""

    @pytest.mark.asyncio
    async def test_extracts_message_text(self):
        """Test that to_framework extracts message text correctly."""
        adapter = OpenClawAgentAdapter(session_id="test-session")
        
        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)
        
        assert payload["message"] == "hello world"
        assert payload["session_id"] == "test-session"
        assert payload["thinking"] == "low"

    @pytest.mark.asyncio
    async def test_includes_agent_id_when_set(self):
        """Test that agent_id is included when set."""
        adapter = OpenClawAgentAdapter(session_id="test-session", agent_id="main")
        
        params = make_message_send_params("test")
        payload = await adapter.to_framework(params)
        
        assert payload["agent_id"] == "main"

    @pytest.mark.asyncio
    async def test_handles_multiple_text_parts(self):
        """Test that multiple text parts are joined correctly."""
        adapter = OpenClawAgentAdapter()
        
        params = MessageSendParams(
            message=Message(
                message_id="test-msg-id",
                role=Role.user,
                parts=[
                    Part(root=TextPart(text="part one")),
                    Part(root=TextPart(text="part two")),
                ],
                context_id=None,
            )
        )
        payload = await adapter.to_framework(params)
        
        assert "part one" in payload["message"]
        assert "part two" in payload["message"]


class TestOpenClawAdapterContextIdMapping:
    """Tests for A2A context_id to OpenClaw session_id mapping."""

    def test_context_id_to_session_id_with_uuid(self):
        """Test that UUID context_id is properly converted."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Standard UUID format
        context_id = "550e8400-e29b-41d4-a716-446655440000"
        session_id = adapter._context_id_to_session_id(context_id)
        
        assert session_id == "a2a-550e8400-e29b-41d4-a716-446655440000"

    def test_context_id_to_session_id_with_special_chars(self):
        """Test that special characters are sanitized."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Context ID with colons, slashes, and spaces
        context_id = "user:123/session:456 test"
        session_id = adapter._context_id_to_session_id(context_id)
        
        # Should be sanitized: lowercase, special chars replaced with hyphen
        assert session_id.startswith("a2a-")
        assert ":" not in session_id
        assert "/" not in session_id
        assert " " not in session_id
        # Should contain the sanitized version
        assert "user-123-session-456-test" in session_id

    def test_context_id_to_session_id_with_none(self):
        """Test that None context_id falls back to default session."""
        adapter = OpenClawAgentAdapter(session_id="my-default-session")
        
        session_id = adapter._context_id_to_session_id(None)
        
        assert session_id == "my-default-session"

    def test_context_id_to_session_id_with_empty_string(self):
        """Test that empty context_id falls back to default session."""
        adapter = OpenClawAgentAdapter(session_id="my-default-session")
        
        session_id = adapter._context_id_to_session_id("")
        
        assert session_id == "my-default-session"

    def test_context_id_to_session_id_truncates_long_ids(self):
        """Test that long context_ids are truncated to 64 chars."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Very long context ID (100 chars)
        context_id = "a" * 100
        session_id = adapter._context_id_to_session_id(context_id)
        
        # Should be truncated: 'a2a-' (4 chars) + 60 chars = 64 total
        assert len(session_id) == 64
        assert session_id.startswith("a2a-")

    def test_context_id_to_session_id_removes_leading_trailing_hyphens(self):
        """Test that leading/trailing hyphens are removed after sanitization."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Context ID that would result in leading/trailing hyphens
        context_id = "---test---"
        session_id = adapter._context_id_to_session_id(context_id)
        
        assert session_id == "a2a-test"

    def test_context_id_to_session_id_handles_all_invalid_chars(self):
        """Test that context_id with only invalid chars falls back to default."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Context ID with only special characters
        context_id = "::://"
        session_id = adapter._context_id_to_session_id(context_id)
        
        # After sanitization, nothing remains, so fall back to default
        assert session_id == "default-session"

    def test_context_id_to_session_id_preserves_underscores(self):
        """Test that underscores are preserved (valid in OpenClaw session IDs)."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        context_id = "user_session_123"
        session_id = adapter._context_id_to_session_id(context_id)
        
        assert session_id == "a2a-user_session_123"

    @pytest.mark.asyncio
    async def test_to_framework_uses_context_id_for_session(self):
        """Test that to_framework maps context_id to session_id."""
        adapter = OpenClawAgentAdapter(session_id="default-session")
        
        # Create params with a context_id
        params = make_message_send_params("hello", context_id="my-context-123")
        payload = await adapter.to_framework(params)
        
        # Session ID should be derived from context_id, not the default
        assert payload["session_id"] == "a2a-my-context-123"

    @pytest.mark.asyncio
    async def test_to_framework_falls_back_to_default_session(self):
        """Test that to_framework uses default session when no context_id."""
        adapter = OpenClawAgentAdapter(session_id="my-default-session")
        
        # Create params without context_id
        params = make_message_send_params("hello", context_id=None)
        payload = await adapter.to_framework(params)
        
        # Should use the adapter's default session_id
        assert payload["session_id"] == "my-default-session"


class TestOpenClawAdapterBuildCommand:
    """Tests for command building."""

    def test_builds_basic_command(self):
        """Test basic command building."""
        adapter = OpenClawAgentAdapter(session_id="test-session")
        
        framework_input = {
            "message": "hello",
            "session_id": "test-session",
            "agent_id": None,
            "thinking": "low",
        }
        cmd = adapter._build_command(framework_input)
        
        assert cmd[0] == "openclaw"
        assert "agent" in cmd
        assert "--local" in cmd
        assert "--message" in cmd
        assert "hello" in cmd
        assert "--json" in cmd
        assert "--session-id" in cmd
        assert "test-session" in cmd
        assert "--thinking" in cmd
        assert "low" in cmd

    def test_includes_agent_flag_when_set(self):
        """Test that --agent flag is included when agent_id is set."""
        adapter = OpenClawAgentAdapter(session_id="test-session", agent_id="main")
        
        framework_input = {
            "message": "hello",
            "session_id": "test-session",
            "agent_id": "main",
            "thinking": "low",
        }
        cmd = adapter._build_command(framework_input)
        
        assert "--agent" in cmd
        assert "main" in cmd

    def test_custom_openclaw_path(self):
        """Test custom openclaw binary path."""
        adapter = OpenClawAgentAdapter(
            session_id="test-session",
            openclaw_path="/custom/path/openclaw",
        )
        
        framework_input = {
            "message": "hello",
            "session_id": "test-session",
            "agent_id": None,
            "thinking": "low",
        }
        cmd = adapter._build_command(framework_input)
        
        assert cmd[0] == "/custom/path/openclaw"


class TestOpenClawAdapterFromFramework:
    """Tests for from_framework output mapping."""

    @pytest.mark.asyncio
    async def test_extracts_text_response(self):
        """Test extraction of text response."""
        adapter = OpenClawAgentAdapter()
        
        params = make_message_send_params("hello", context_id="ctx-123")
        framework_output = make_openclaw_output("Response text")
        
        result = await adapter.from_framework(framework_output, params)
        
        assert isinstance(result, Message)
        assert result.role == Role.agent
        assert result.context_id == "ctx-123"
        assert result.parts[0].root.text == "Response text"

    @pytest.mark.asyncio
    async def test_handles_media_urls(self):
        """Test handling of media URLs in response."""
        adapter = OpenClawAgentAdapter()

        params = make_message_send_params("hello")
        framework_output = make_openclaw_output(
            "Response with image",
            media_urls=["https://example.com/image.png"],
        )

        result = await adapter.from_framework(framework_output, params)

        assert len(result.parts) == 2
        assert result.parts[0].root.text == "Response with image"
        # Second part should be a FilePart
        assert hasattr(result.parts[1].root, "file")
        assert result.parts[1].root.file.uri == "https://example.com/image.png"
        assert result.parts[1].root.file.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_handles_empty_payloads(self):
        """Test handling of empty payloads."""
        adapter = OpenClawAgentAdapter()
        
        params = make_message_send_params("hello")
        framework_output = {"payloads": [], "meta": {}}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert isinstance(result, Message)
        assert len(result.parts) == 1
        assert result.parts[0].root.text == ""


class TestOpenClawAdapterMimeTypeDetection:
    """Tests for MIME type detection."""

    def test_detects_image_types(self):
        """Test detection of image MIME types."""
        assert OpenClawAgentAdapter._detect_mime_type("file.png") == "image/png"
        assert OpenClawAgentAdapter._detect_mime_type("file.jpg") == "image/jpeg"
        assert OpenClawAgentAdapter._detect_mime_type("file.jpeg") == "image/jpeg"
        assert OpenClawAgentAdapter._detect_mime_type("file.gif") == "image/gif"
        assert OpenClawAgentAdapter._detect_mime_type("file.webp") == "image/webp"
        assert OpenClawAgentAdapter._detect_mime_type("file.svg") == "image/svg+xml"

    def test_detects_video_types(self):
        """Test detection of video MIME types."""
        assert OpenClawAgentAdapter._detect_mime_type("file.mp4") == "video/mp4"
        assert OpenClawAgentAdapter._detect_mime_type("file.webm") == "video/webm"

    def test_detects_audio_types(self):
        """Test detection of audio MIME types."""
        assert OpenClawAgentAdapter._detect_mime_type("file.mp3") == "audio/mpeg"
        assert OpenClawAgentAdapter._detect_mime_type("file.wav") == "audio/wav"

    def test_detects_document_types(self):
        """Test detection of document MIME types."""
        assert OpenClawAgentAdapter._detect_mime_type("file.pdf") == "application/pdf"

    def test_fallback_for_unknown_types(self):
        """Test fallback for unknown file types."""
        assert OpenClawAgentAdapter._detect_mime_type("file.xyz") == "application/octet-stream"
        assert OpenClawAgentAdapter._detect_mime_type("file") == "application/octet-stream"


class TestOpenClawAdapterAsyncMode:
    """Tests for async task mode."""

    def test_supports_async_tasks_true_by_default(self):
        """Test that async mode is enabled by default."""
        adapter = OpenClawAgentAdapter()
        assert adapter.supports_async_tasks() is True

    def test_supports_async_tasks_false_when_disabled(self):
        """Test that async mode can be disabled."""
        adapter = OpenClawAgentAdapter(async_mode=False)
        assert adapter.supports_async_tasks() is False

    def test_async_mode_creates_task_store(self):
        """Test that async mode creates an InMemoryTaskStore by default."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        assert adapter.task_store is not None
        assert isinstance(adapter.task_store, InMemoryTaskStore)

    def test_async_mode_accepts_custom_task_store(self):
        """Test that a custom TaskStore can be provided."""
        custom_store = InMemoryTaskStore()
        adapter = OpenClawAgentAdapter(async_mode=True, task_store=custom_store)
        assert adapter.task_store is custom_store

    @pytest.mark.asyncio
    async def test_get_task_raises_when_not_async_mode(self):
        """Test that get_task raises RuntimeError when not in async mode."""
        adapter = OpenClawAgentAdapter(async_mode=False)
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.get_task("some-task-id")

    @pytest.mark.asyncio
    async def test_cancel_task_raises_when_not_async_mode(self):
        """Test that cancel_task raises RuntimeError when not in async mode."""
        adapter = OpenClawAgentAdapter(async_mode=False)
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.cancel_task("some-task-id")

    @pytest.mark.asyncio
    async def test_delete_task_raises_when_not_async_mode(self):
        """Test that delete_task raises RuntimeError when not in async mode."""
        adapter = OpenClawAgentAdapter(async_mode=False)
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.delete_task("some-task-id")

    @pytest.mark.asyncio
    async def test_get_task_returns_none_for_unknown_id(self):
        """Test that get_task returns None for unknown task ID."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        result = await adapter.get_task("nonexistent-task-id")
        assert result is None
        
        await adapter.close()


class TestOpenClawAdapterAsyncExecution:
    """Tests for async task execution."""

    @pytest.mark.asyncio
    async def test_handle_async_returns_task_immediately(self):
        """Test that handle() returns a Task immediately in async mode."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock subprocess to simulate slow execution
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            
            async def slow_communicate():
                await asyncio.sleep(0.5)
                proc.returncode = 0
                return (
                    json.dumps(make_openclaw_output("response")).encode(),
                    b"",
                )
            
            proc.communicate = slow_communicate
            proc.kill = MagicMock()
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message", context_id="async-ctx")
            
            # handle() should return immediately with a Task
            result = await adapter.handle(params)
            
            # Verify we got a Task back
            assert isinstance(result, Task)
            assert result.status.state == TaskState.working
            assert result.context_id == "async-ctx"
            
            # Clean up
            await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_completes_in_background(self):
        """Test that the task completes in the background and can be polled."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (
                    json.dumps(make_openclaw_output("command result")).encode(),
                    b"",
                )
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for background task to complete
            await asyncio.sleep(0.1)
            
            # Poll for completed task
            completed_task = await adapter.get_task(task_id)
            
            assert completed_task is not None
            assert completed_task.status.state == TaskState.completed
            # Response is in artifacts (A2A spec: task outputs go in artifacts)
            assert completed_task.artifacts is not None
            assert len(completed_task.artifacts) == 1
            assert "command result" in completed_task.artifacts[0].parts[0].root.text
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_handles_failure(self):
        """Test that failed commands result in failed task state."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock subprocess that fails
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 1
            
            async def communicate():
                return (b"", b"Command failed: error message")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for background task to fail
            await asyncio.sleep(0.1)
            
            # Poll for failed task
            failed_task = await adapter.get_task(task_id)
            
            assert failed_task is not None
            assert failed_task.status.state == TaskState.failed
            assert failed_task.status.message is not None
            assert "failed" in failed_task.status.message.parts[0].root.text.lower()
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_cancel_task_marks_task_as_canceled(self):
        """Test that cancel_task() marks the task as canceled."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock a slow subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            
            async def slow_communicate():
                await asyncio.sleep(10)  # Very slow
                proc.returncode = 0
                return (json.dumps(make_openclaw_output("result")).encode(), b"")
            
            proc.communicate = slow_communicate
            proc.kill = MagicMock()
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Cancel the task
            canceled_task = await adapter.cancel_task(task_id)
            
            assert canceled_task is not None
            assert canceled_task.status.state == TaskState.canceled
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_includes_history(self):
        """Test that completed tasks include conversation history."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("user question", context_id="history-ctx")
            
            task = await adapter.handle(params)
            
            # Initial task should have the user message in history
            assert task.history is not None
            assert len(task.history) == 1
            assert task.history[0].role == Role.user
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            completed_task = await adapter.get_task(task.id)
            
            # Completed task should have both user and agent messages
            assert completed_task.history is not None
            assert len(completed_task.history) == 2
            assert completed_task.history[0].role == Role.user
            assert completed_task.history[1].role == Role.agent
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test that tasks timeout after timeout seconds."""
        adapter = OpenClawAgentAdapter(
            async_mode=True,
            timeout=1,  # 1 second timeout for testing
        )
        
        # Mock a very slow subprocess that exceeds timeout
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            
            async def very_slow_communicate():
                await asyncio.sleep(10)  # Much longer than 1 second timeout
                proc.returncode = 0
                return (json.dumps(make_openclaw_output("result")).encode(), b"")
            
            proc.communicate = very_slow_communicate
            proc.kill = MagicMock()
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for timeout to occur
            await asyncio.sleep(1.5)
            
            # Task should be failed due to timeout
            timed_out_task = await adapter.get_task(task_id)
            
            assert timed_out_task is not None
            assert timed_out_task.status.state == TaskState.failed
            assert "timed out" in timed_out_task.status.message.parts[0].root.text.lower()
            
            await adapter.close()


class TestOpenClawAdapterDeleteTask:
    """Tests for delete_task functionality."""

    @pytest.mark.asyncio
    async def test_delete_task_removes_completed_task(self):
        """Test that delete_task() removes a completed task."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Verify task is completed
            completed_task = await adapter.get_task(task_id)
            assert completed_task.status.state == TaskState.completed
            
            # Delete the task
            result = await adapter.delete_task(task_id)
            assert result is True
            
            # Task should no longer exist
            deleted_task = await adapter.get_task(task_id)
            assert deleted_task is None
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_task_raises_for_running_task(self):
        """Test that delete_task() raises for non-terminal state tasks."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock a slow subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            
            async def slow_communicate():
                await asyncio.sleep(10)
                proc.returncode = 0
                return (json.dumps(make_openclaw_output("result")).encode(), b"")
            
            proc.communicate = slow_communicate
            proc.kill = MagicMock()
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            
            # Try to delete a running task
            with pytest.raises(ValueError, match="Cannot delete task"):
                await adapter.delete_task(task.id)
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_task_returns_false_for_unknown_id(self):
        """Test that delete_task() returns False for unknown task ID."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        result = await adapter.delete_task("nonexistent-task-id")
        assert result is False
        
        await adapter.close()


class TestOpenClawAdapterTTLCleanup:
    """Tests for TTL-based task cleanup."""

    def test_ttl_defaults(self):
        """Test default TTL configuration."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        assert adapter._task_ttl == 3600  # 1 hour default
        assert adapter._cleanup_interval == 300  # 5 minutes default
        assert adapter._cleanup_task is None  # Not started until first handle()

    def test_ttl_can_be_disabled(self):
        """Test that TTL can be disabled by setting to None."""
        adapter = OpenClawAgentAdapter(async_mode=True, task_ttl_seconds=None)
        
        assert adapter._task_ttl is None

    def test_ttl_custom_values(self):
        """Test custom TTL configuration."""
        adapter = OpenClawAgentAdapter(
            async_mode=True,
            task_ttl_seconds=7200,
            cleanup_interval_seconds=600,
        )
        
        assert adapter._task_ttl == 7200
        assert adapter._cleanup_interval == 600

    @pytest.mark.asyncio
    async def test_cleanup_task_starts_on_first_handle(self):
        """Test that cleanup task starts lazily on first handle()."""
        adapter = OpenClawAgentAdapter(async_mode=True, task_ttl_seconds=3600)
        
        # Cleanup task should not be started yet
        assert adapter._cleanup_task is None
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            await adapter.handle(params)
            
            # Cleanup task should now be started
            assert adapter._cleanup_task is not None
            assert not adapter._cleanup_task.done()
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_completion_time_recorded(self):
        """Test that task completion time is recorded for TTL tracking."""
        adapter = OpenClawAgentAdapter(async_mode=True, task_ttl_seconds=3600)
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Completion time should be recorded
            assert task.id in adapter._task_completion_times
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_tasks(self):
        """Test that cleanup removes tasks older than TTL."""
        # Use very short TTL for testing
        adapter = OpenClawAgentAdapter(
            async_mode=True,
            task_ttl_seconds=1,  # 1 second TTL
            cleanup_interval_seconds=60,  # Don't auto-run during test
        )
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Task should exist
            assert await adapter.get_task(task_id) is not None
            
            # Wait for TTL to expire
            await asyncio.sleep(1.1)
            
            # Manually trigger cleanup
            await adapter._cleanup_expired_tasks()
            
            # Task should be deleted
            assert await adapter.get_task(task_id) is None
            assert task_id not in adapter._task_completion_times
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_cleanup_does_not_remove_unexpired_tasks(self):
        """Test that cleanup does not remove tasks within TTL."""
        adapter = OpenClawAgentAdapter(
            async_mode=True,
            task_ttl_seconds=3600,  # 1 hour TTL
            cleanup_interval_seconds=60,
        )
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Manually trigger cleanup
            await adapter._cleanup_expired_tasks()
            
            # Task should still exist (TTL not expired)
            assert await adapter.get_task(task_id) is not None
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_no_cleanup_when_ttl_disabled(self):
        """Test that cleanup does nothing when TTL is disabled."""
        adapter = OpenClawAgentAdapter(
            async_mode=True,
            task_ttl_seconds=None,  # Disabled
        )
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Completion time should NOT be recorded when TTL is disabled
            assert task_id not in adapter._task_completion_times
            
            # Cleanup should be a no-op
            await adapter._cleanup_expired_tasks()
            
            # Task should still exist
            assert await adapter.get_task(task_id) is not None
        
        await adapter.close()


class TestOpenClawAdapterSyncMode:
    """Tests for synchronous mode."""

    @pytest.mark.asyncio
    async def test_sync_mode_blocks_until_complete(self):
        """Test that sync mode blocks until command completes."""
        adapter = OpenClawAgentAdapter(async_mode=False)
        
        # Mock subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = 0
            
            async def communicate():
                return (json.dumps(make_openclaw_output("sync response")).encode(), b"")
            
            proc.communicate = communicate
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test message", context_id="sync-ctx")
            
            result = await adapter.handle(params)
            
            # Should return Message, not Task
            assert isinstance(result, Message)
            assert result.role == Role.agent
            assert result.context_id == "sync-ctx"
            assert "sync response" in result.parts[0].root.text


class TestOpenClawAdapterLifecycle:
    """Tests for adapter lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_cancels_pending_background_tasks(self):
        """Test that close() cancels all pending background tasks."""
        adapter = OpenClawAgentAdapter(async_mode=True)
        
        # Mock a slow subprocess
        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            
            async def slow_communicate():
                await asyncio.sleep(10)
                proc.returncode = 0
                return (json.dumps(make_openclaw_output("result")).encode(), b"")
            
            proc.communicate = slow_communicate
            proc.kill = MagicMock()
            return proc
        
        with patch("asyncio.create_subprocess_exec", mock_create_subprocess):
            params = make_message_send_params("test")
            
            # Start a task
            task = await adapter.handle(params)
            
            # Verify background task is running
            assert task.id in adapter._background_tasks
            
            # Close the adapter
            await adapter.close()
            
            # Verify background tasks are cleared
            assert len(adapter._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_context_manager_support(self):
        """Test async context manager support."""
        async with OpenClawAgentAdapter(async_mode=True) as adapter:
            assert adapter is not None
            assert adapter.async_mode is True

    def test_supports_streaming_returns_false(self):
        """Test that OpenClaw adapter does not support streaming."""
        adapter = OpenClawAgentAdapter()
        assert adapter.supports_streaming() is False


class TestOpenClawAdapterLoader:
    """Tests for loading OpenClaw adapter via loader."""

    @pytest.mark.asyncio
    async def test_load_openclaw_adapter(self):
        """Test loading OpenClaw adapter via load_a2a_agent."""
        from a2a_adapter import load_a2a_agent
        
        adapter = await load_a2a_agent({
            "adapter": "openclaw",
            "session_id": "loader-test-session",
            "agent_id": "main",
            "thinking": "medium",
            "timeout": 600,
        })
        
        assert isinstance(adapter, OpenClawAgentAdapter)
        assert adapter.session_id == "loader-test-session"
        assert adapter.agent_id == "main"
        assert adapter.thinking == "medium"
        assert adapter.timeout == 600
        assert adapter.async_mode is True  # Default

    @pytest.mark.asyncio
    async def test_load_openclaw_adapter_sync_mode(self):
        """Test loading OpenClaw adapter in sync mode."""
        from a2a_adapter import load_a2a_agent
        
        adapter = await load_a2a_agent({
            "adapter": "openclaw",
            "async_mode": False,
        })
        
        assert isinstance(adapter, OpenClawAgentAdapter)
        assert adapter.async_mode is False

    @pytest.mark.asyncio
    async def test_load_openclaw_adapter_defaults(self):
        """Test loading OpenClaw adapter with defaults."""
        from a2a_adapter import load_a2a_agent
        
        adapter = await load_a2a_agent({
            "adapter": "openclaw",
        })
        
        assert isinstance(adapter, OpenClawAgentAdapter)
        assert adapter.session_id.startswith("a2a-")
        assert adapter.agent_id is None
        assert adapter.thinking == "low"
        assert adapter.timeout == 600
        assert adapter.openclaw_path == "openclaw"
        
        await adapter.close()
