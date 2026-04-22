"""
Unit tests for N8nAgentAdapter.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from a2a_adapter.integrations.n8n import N8nAgentAdapter
from a2a.types import Message, SendMessageRequest, Task, TaskState, Role, Part
from a2a.server.tasks import InMemoryTaskStore


def make_message_send_params(text: str, context_id: str | None = None) -> SendMessageRequest:
    """Helper to create SendMessageRequest with correct A2A types."""
    return SendMessageRequest(
        message=Message(
            message_id="test-msg-id",
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
            context_id=context_id or "",
        )
    )


class TestN8nAdapterContextId:
    """Tests for context_id handling in N8nAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_context_id(self):
        """Test that to_framework extracts context_id from params.message."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id="ctx-123")
        payload = await adapter.to_framework(params)
        
        assert payload["metadata"]["context_id"] == "ctx-123"

    @pytest.mark.asyncio
    async def test_to_framework_handles_missing_context_id(self):
        """Test that to_framework handles missing context_id gracefully."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")

        params = make_message_send_params("hello", context_id=None)
        payload = await adapter.to_framework(params)

        # V1.0 uses empty string ("") instead of None for unset optional string fields
        assert payload["metadata"]["context_id"] == ""

    @pytest.mark.asyncio
    async def test_to_framework_with_template_adds_context_id_at_root(self):
        """Test that context_id is added at root level when using payload_template."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            payload_template={"custom_field": "value"}
        )
        
        params = make_message_send_params("hello", context_id="ctx-456")
        payload = await adapter.to_framework(params)
        
        # With template, context_id should be at root, not in metadata
        assert payload["context_id"] == "ctx-456"
        assert "metadata" not in payload
        assert payload["custom_field"] == "value"

    @pytest.mark.asyncio
    async def test_to_framework_with_template_respects_existing_context_id(self):
        """Test that template's context_id is not overwritten."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            payload_template={"context_id": "template-ctx"}
        )
        
        params = make_message_send_params("hello", context_id="request-ctx")
        payload = await adapter.to_framework(params)
        
        # Template's context_id should be preserved
        assert payload["context_id"] == "template-ctx"

    @pytest.mark.asyncio
    async def test_from_framework_preserves_context_id(self):
        """Test that from_framework preserves context_id on the response Message."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id="ctx-789")
        framework_output = {"output": "response text"}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert isinstance(result, Message)
        assert result.context_id == "ctx-789"
        assert result.role == Role.ROLE_AGENT

    @pytest.mark.asyncio
    async def test_from_framework_handles_missing_context_id(self):
        """Test that from_framework handles missing context_id gracefully."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")

        params = make_message_send_params("hello", context_id=None)
        framework_output = {"output": "response text"}

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        # V1.0 uses empty string ("") instead of None for unset optional string fields
        assert result.context_id == ""

    @pytest.mark.asyncio
    async def test_handle_preserves_context_id_end_to_end(self):
        """Test that handle() preserves context_id through the full request/response cycle."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "n8n response"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message", context_id="e2e-ctx-id")
            result = await adapter.handle(params)
            
            # Verify context_id is preserved in response
            assert result.context_id == "e2e-ctx-id"
            
            # Verify context_id was sent in the payload
            call_args = mock_client.post.call_args
            sent_payload = call_args.kwargs["json"]
            assert sent_payload["metadata"]["context_id"] == "e2e-ctx-id"


class TestN8nAdapterBasic:
    """Basic functionality tests for N8nAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_message_text(self):
        """Test that to_framework extracts message text correctly."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)
        
        assert payload["message"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_custom_message_field(self):
        """Test that custom message_field is used."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            message_field="query"
        )
        
        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)
        
        assert payload["query"] == "test query"
        assert "message" not in payload

    @pytest.mark.asyncio
    async def test_from_framework_extracts_output_field(self):
        """Test that from_framework extracts the output field."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello")
        framework_output = {"output": "the response"}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert result.parts[0].text == "the response"

    @pytest.mark.asyncio
    async def test_from_framework_handles_array_response(self):
        """Test that from_framework handles array responses."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello")
        framework_output = [{"output": "first"}, {"output": "second"}]
        
        result = await adapter.from_framework(framework_output, params)
        
        assert "first" in result.parts[0].text
        assert "second" in result.parts[0].text

    def test_supports_streaming_returns_false(self):
        """Test that N8N adapter does not support streaming."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        assert adapter.supports_streaming() is False


class TestN8nAdapterAsyncMode:
    """Tests for async task mode in N8nAgentAdapter."""

    def test_supports_async_tasks_false_by_default(self):
        """Test that async mode is disabled by default."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        assert adapter.supports_async_tasks() is False

    def test_supports_async_tasks_true_when_enabled(self):
        """Test that async mode can be enabled."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        assert adapter.supports_async_tasks() is True

    def test_async_mode_creates_task_store(self):
        """Test that async mode creates an InMemoryTaskStore by default."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        assert adapter.task_store is not None
        assert isinstance(adapter.task_store, InMemoryTaskStore)

    def test_async_mode_accepts_custom_task_store(self):
        """Test that a custom TaskStore can be provided."""
        custom_store = InMemoryTaskStore()
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
            task_store=custom_store,
        )
        assert adapter.task_store is custom_store

    @pytest.mark.asyncio
    async def test_get_task_raises_when_not_async_mode(self):
        """Test that get_task raises RuntimeError when not in async mode."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.get_task("some-task-id")

    @pytest.mark.asyncio
    async def test_cancel_task_raises_when_not_async_mode(self):
        """Test that cancel_task raises RuntimeError when not in async mode."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.cancel_task("some-task-id")

    @pytest.mark.asyncio
    async def test_handle_async_returns_task_immediately(self):
        """Test that handle() returns a Task immediately in async mode."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Mock the HTTP client to simulate slow response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "delayed response"}
        mock_response.raise_for_status = MagicMock()
        
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow n8n workflow
            return mock_response
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_post
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message", context_id="async-ctx")
            
            # handle() should return immediately with a Task
            result = await adapter.handle(params)
            
            # Verify we got a Task back
            assert isinstance(result, Task)
            assert result.status.state == TaskState.TASK_STATE_WORKING
            assert result.context_id == "async-ctx"
            
            # Clean up
            await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_completes_in_background(self):
        """Test that the task completes in the background and can be polled."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "workflow result"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for background task to complete
            await asyncio.sleep(0.1)
            
            # Poll for completed task
            completed_task = await adapter.get_task(task_id)
            
            assert completed_task is not None
            assert completed_task.status.state == TaskState.TASK_STATE_COMPLETED
            assert completed_task.status.message is not None
            assert "workflow result" in completed_task.status.message.parts[0].text
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_handles_failure(self):
        """Test that failed workflows result in failed task state."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = RuntimeError("Connection failed")
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for background task to fail
            await asyncio.sleep(0.1)
            
            # Poll for failed task
            failed_task = await adapter.get_task(task_id)
            
            assert failed_task is not None
            assert failed_task.status.state == TaskState.TASK_STATE_FAILED
            assert failed_task.status.message is not None
            assert "failed" in failed_task.status.message.parts[0].text.lower()
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_cancel_task_marks_task_as_canceled(self):
        """Test that cancel_task() marks the task as canceled."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Mock a slow HTTP client
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow
            return MagicMock(status_code=200, json=lambda: {"output": "result"})
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_post
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Cancel the task
            canceled_task = await adapter.cancel_task(task_id)
            
            assert canceled_task is not None
            assert canceled_task.status.state == TaskState.TASK_STATE_CANCELED
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_get_task_returns_none_for_unknown_id(self):
        """Test that get_task returns None for unknown task ID."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        result = await adapter.get_task("nonexistent-task-id")
        assert result is None
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_includes_history(self):
        """Test that completed tasks include conversation history."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "response"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("user question", context_id="history-ctx")
            
            task = await adapter.handle(params)

            # Initial task should have the user message in history
            assert task.history is not None
            assert len(task.history) == 1
            assert task.history[0].role == Role.ROLE_USER

            # Wait for completion
            await asyncio.sleep(0.1)

            completed_task = await adapter.get_task(task.id)

            # Completed task should have both user and agent messages
            assert completed_task.history is not None
            assert len(completed_task.history) == 2
            assert completed_task.history[0].role == Role.ROLE_USER
            assert completed_task.history[1].role == Role.ROLE_AGENT
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_close_cancels_pending_background_tasks(self):
        """Test that close() cancels all pending background tasks."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Mock a slow HTTP client
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(status_code=200, json=lambda: {"output": "result"})
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_post
            mock_get_client.return_value = mock_client
            
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
    async def test_task_timeout(self):
        """Test that tasks timeout after async_timeout seconds."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
            async_timeout=1,  # 1 second timeout for testing
        )
        
        # Mock a very slow HTTP client that exceeds timeout
        async def very_slow_post(*args, **kwargs):
            await asyncio.sleep(10)  # Much longer than 1 second timeout
            return MagicMock(status_code=200, json=lambda: {"output": "result"})
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = very_slow_post
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for timeout to occur
            await asyncio.sleep(1.5)
            
            # Task should be failed due to timeout
            timed_out_task = await adapter.get_task(task_id)
            
            assert timed_out_task is not None
            assert timed_out_task.status.state == TaskState.TASK_STATE_FAILED
            assert "timed out" in timed_out_task.status.message.parts[0].text.lower()
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_cancel_task_prevents_race_condition(self):
        """Test that cancel_task() properly prevents race conditions."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Track when workflow execution attempts to save
        save_calls = []
        original_save = adapter.task_store.save

        async def tracking_save(task, context):
            save_calls.append(task.status.state)
            await original_save(task, context)

        adapter.task_store.save = tracking_save
        
        # Mock a slow HTTP client
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.5)  # Slow enough to cancel
            return MagicMock(status_code=200, json=lambda: {"output": "result"})
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_post
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message")
            
            # Get initial task
            task = await adapter.handle(params)
            task_id = task.id
            
            # Cancel immediately
            await asyncio.sleep(0.1)  # Let it start
            await adapter.cancel_task(task_id)
            
            # Final state should be canceled
            final_task = await adapter.get_task(task_id)
            assert final_task.status.state == TaskState.TASK_STATE_CANCELED
            
            # The last save should be "canceled", not "completed" or "failed"
            assert save_calls[-1] == TaskState.TASK_STATE_CANCELED
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_task_removes_completed_task(self):
        """Test that delete_task() removes a completed task."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "response"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            task_id = task.id
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            # Verify task is completed
            completed_task = await adapter.get_task(task_id)
            assert completed_task.status.state == TaskState.TASK_STATE_COMPLETED
            
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
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        # Mock a slow HTTP client
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(status_code=200, json=lambda: {"output": "result"})
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_post
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test")
            task = await adapter.handle(params)
            
            # Try to delete a running task
            with pytest.raises(ValueError, match="Cannot delete task"):
                await adapter.delete_task(task.id)
            
            await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_task_returns_false_for_unknown_id(self):
        """Test that delete_task() returns False for unknown task ID."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            async_mode=True,
        )
        
        result = await adapter.delete_task("nonexistent-task-id")
        assert result is False
        
        await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_task_raises_when_not_async_mode(self):
        """Test that delete_task raises RuntimeError when not in async mode."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.delete_task("some-task-id")
