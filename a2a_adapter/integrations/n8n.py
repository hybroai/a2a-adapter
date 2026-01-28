"""
n8n adapter for A2A Protocol.

This adapter enables n8n workflows to be exposed as A2A-compliant agents
by forwarding A2A messages to n8n webhooks.

Supports two modes:
- Synchronous (default): Blocks until n8n workflow completes, returns Message
- Async Task Mode: Returns Task immediately, processes in background, supports polling

Supports flexible input handling:
- input_mapper: Custom function for full control over input transformation
- parse_json_input: Automatic JSON parsing for structured inputs
- message_field: Simple text mapping to a single key (default fallback)
"""

import json
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

import httpx
from httpx import HTTPStatusError, ConnectError, ReadTimeout

from a2a.types import (
    Message,
    MessageSendParams,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    Role,
    Part,
)
from ..adapter import BaseAgentAdapter

# Lazy import for TaskStore to avoid hard dependency
try:
    from a2a.server.tasks import TaskStore, InMemoryTaskStore
    _HAS_TASK_STORE = True
except ImportError:
    _HAS_TASK_STORE = False
    TaskStore = None  # type: ignore
    InMemoryTaskStore = None  # type: ignore

logger = logging.getLogger(__name__)


class N8nAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating n8n workflows as A2A agents.

    This adapter forwards A2A message requests to an n8n webhook URL and
    translates the response back to A2A format.
    
    Supports two execution modes:
    
    1. **Synchronous Mode** (default): 
       - Blocks until the n8n workflow completes
       - Returns a Message with the workflow result
       - Best for quick workflows (< 30 seconds)
    
    2. **Async Task Mode** (async_mode=True):
       - Returns a Task with state="working" immediately
       - Processes the workflow in the background
       - Clients can poll get_task() for status updates
       - Best for long-running workflows
       - Tasks time out after async_timeout seconds (default: 300)
    
    **Memory Considerations (Async Mode)**:
    
    When using InMemoryTaskStore (the default), completed tasks remain in memory
    indefinitely. For production use, either:
    
    1. Call delete_task() after retrieving completed tasks to free memory
    2. Use DatabaseTaskStore for persistent storage with external cleanup
    3. Implement a periodic cleanup routine for old completed tasks
    
    Example cleanup pattern::
    
        task = await adapter.get_task(task_id)
        if task and task.status.state in ("completed", "failed", "canceled"):
            # Process the result...
            await adapter.delete_task(task_id)  # Free memory
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 30,
        headers: Dict[str, str] | None = None,
        max_retries: int = 2,
        backoff: float = 0.25,
        payload_template: Dict[str, Any] | None = None,
        message_field: str = "message",
        async_mode: bool = False,
        task_store: "TaskStore | None" = None,
        async_timeout: int = 300,
        # Flexible input handling parameters
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the n8n adapter.

        Args:
            webhook_url: The n8n webhook URL to send requests to.
            timeout: HTTP request timeout in seconds (default: 30).
            headers: Optional additional HTTP headers to include in requests.
            max_retries: Number of retry attempts for transient failures (default: 2).
            backoff: Base backoff seconds; multiplied by 2**attempt between retries.
            payload_template: Optional base payload dict to merge with message.
                              Use this to add static fields your n8n workflow expects.
            message_field: Field name for the user message (default: "message").
                           Used as fallback when JSON parsing fails or is disabled.
            async_mode: If True, return Task immediately and process in background.
                        If False (default), block until workflow completes.
            task_store: Optional TaskStore for persisting task state. If not provided
                        and async_mode is True, uses InMemoryTaskStore.
            async_timeout: Timeout for async task execution in seconds (default: 300).
            parse_json_input: If True (default), attempt to parse input as JSON and use
                              the parsed dict directly as webhook payload.
            input_mapper: Optional custom function to transform raw input to webhook payload.
                          Signature: (raw_input: str, context_id: str | None) -> dict.
                          When provided, this takes highest priority over other methods.
            default_inputs: Optional dict of default values to merge with parsed inputs.
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.headers = dict(headers) if headers else {}
        self.max_retries = max(0, int(max_retries))
        self.backoff = float(backoff)
        self.payload_template = dict(payload_template) if payload_template else {}
        self.message_field = message_field
        self._client: httpx.AsyncClient | None = None

        # Flexible input handling configuration
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}
        
        # Async task mode configuration
        self.async_mode = async_mode
        self.async_timeout = async_timeout
        self._background_tasks: Dict[str, "asyncio.Task[None]"] = {}
        self._cancelled_tasks: set[str] = set()  # Track cancelled task IDs
        
        # Initialize task store for async mode
        if async_mode:
            if not _HAS_TASK_STORE:
                raise ImportError(
                    "Async task mode requires the A2A SDK with task support. "
                    "Install with: pip install a2a-sdk"
                )
            self.task_store: "TaskStore" = task_store or InMemoryTaskStore()
        else:
            self.task_store = task_store  # type: ignore

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            # Use async_timeout for async mode since workflows may take longer
            timeout = self.async_timeout if self.async_mode else self.timeout
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """
        Handle a non-streaming A2A message request.
        
        In sync mode (default): Blocks until workflow completes, returns Message.
        In async mode: Returns Task immediately, processes in background.
        """
        if self.async_mode:
            return await self._handle_async(params)
        else:
            return await self._handle_sync(params)

    async def _handle_sync(self, params: MessageSendParams) -> Message:
        """Handle request synchronously - blocks until workflow completes."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        result = await self.from_framework(framework_output, params)
        # In sync mode, always return Message
        if isinstance(result, Task):
            # Extract message from completed task if needed
            if result.status and result.status.message:
                return result.status.message
            # Fallback: create a message from task
            return Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=result.context_id,
                parts=[Part(root=TextPart(text="Task completed"))],
            )
        return result

    async def _handle_async(self, params: MessageSendParams) -> Task:
        """
        Handle request asynchronously - returns Task immediately, processes in background.
        
        1. Creates a Task with state="working"
        2. Saves the task to the TaskStore
        3. Starts a background coroutine to execute the workflow
        4. Returns the Task immediately
        """
        # Generate IDs
        task_id = str(uuid.uuid4())
        context_id = self.extract_context_id(params) or str(uuid.uuid4())
        
        # Extract the initial message for history
        initial_message = None
        if hasattr(params, "message") and params.message:
            initial_message = params.message
        
        # Create initial task with "working" state
        now = datetime.now(timezone.utc).isoformat()
        task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.working,
                timestamp=now,
            ),
            history=[initial_message] if initial_message else None,
        )
        
        # Save initial task state
        await self.task_store.save(task)
        logger.debug("Created async task %s with state=working", task_id)
        
        # Start background processing with timeout
        bg_task = asyncio.create_task(
            self._execute_workflow_with_timeout(task_id, context_id, params)
        )
        self._background_tasks[task_id] = bg_task
        
        # Clean up background task reference when done and handle exceptions
        def _on_task_done(t: "asyncio.Task[None]") -> None:
            self._background_tasks.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)
            # Check for unhandled exceptions (shouldn't happen, but log if they do)
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    logger.error(
                        "Unhandled exception in background task %s: %s",
                        task_id,
                        exc,
                    )
        
        bg_task.add_done_callback(_on_task_done)
        
        return task

    async def _execute_workflow_with_timeout(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """
        Execute the workflow with a timeout wrapper.
        
        This ensures that long-running workflows don't hang indefinitely.
        """
        try:
            await asyncio.wait_for(
                self._execute_workflow_background(task_id, context_id, params),
                timeout=self.async_timeout,
            )
        except asyncio.TimeoutError:
            # Check if task was cancelled (don't overwrite canceled state)
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return
            
            logger.error("Task %s timed out after %s seconds", task_id, self.async_timeout)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=f"Workflow timed out after {self.async_timeout} seconds"))],
            )
            
            timeout_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=error_message,
                    timestamp=now,
                ),
            )
            await self.task_store.save(timeout_task)

    async def _execute_workflow_background(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """
        Execute the n8n workflow in the background and update task state.
        
        This runs as a background coroutine after the initial Task is returned.
        """
        try:
            logger.debug("Starting background execution for task %s", task_id)
            
            # Execute the workflow (this may take a while)
            framework_input = await self.to_framework(params)
            framework_output = await self.call_framework(framework_input, params)
            
            # Check if task was cancelled during execution
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled during execution, not updating state", task_id)
                return
            
            # Convert to message
            response_text = self._extract_response_text(framework_output)
            response_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=response_text))],
            )
            
            # Build history
            history = []
            if hasattr(params, "message") and params.message:
                history.append(params.message)
            history.append(response_message)
            
            # Update task to completed state
            now = datetime.now(timezone.utc).isoformat()
            completed_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=response_message,
                    timestamp=now,
                ),
                history=history,
            )
            
            await self.task_store.save(completed_task)
            logger.debug("Task %s completed successfully", task_id)
            
        except asyncio.CancelledError:
            # Task was cancelled - don't update state, cancel_task() handles it
            logger.debug("Task %s was cancelled", task_id)
            raise  # Re-raise to properly cancel the task
            
        except Exception as e:
            # Check if task was cancelled (don't overwrite canceled state)
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return
            
            # Update task to failed state
            logger.error("Task %s failed: %s", task_id, e)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=f"Workflow failed: {str(e)}"))],
            )
            
            failed_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=error_message,
                    timestamp=now,
                ),
            )
            
            await self.task_store.save(failed_task)

    def _extract_response_text(self, framework_output: Dict[str, Any] | list) -> str:
        """Extract response text from n8n webhook output."""
        if isinstance(framework_output, list):
            if len(framework_output) == 0:
                return ""
            elif len(framework_output) == 1:
                return self._extract_text_from_item(framework_output[0])
            else:
                texts = []
                for item in framework_output:
                    if isinstance(item, dict):
                        text = self._extract_text_from_item(item)
                        if text:
                            texts.append(text)
                return "\n".join(texts) if texts else json.dumps(framework_output, indent=2)
        elif isinstance(framework_output, dict):
            return self._extract_text_from_item(framework_output)
        else:
            return str(framework_output)

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Build the n8n webhook payload from A2A params.

        Processing priority:
        1. input_mapper (custom function) - highest priority
        2. parse_json_input (auto JSON parsing)
        3. message_field (fallback for plain text)

        Args:
            params: A2A message parameters.

        Returns:
            dict with the user message and any configured template fields.
        """
        # Use base class utility for raw input extraction
        raw_input = self.extract_raw_input(params)
        context_id = self.extract_context_id(params)

        # Priority 1: Custom input_mapper function (highest priority)
        if self.input_mapper is not None:
            try:
                mapped_inputs = self.input_mapper(raw_input, context_id)
                logger.debug("Used input_mapper to transform input")
                # Merge: payload_template < default_inputs < mapped_inputs
                return {
                    **self.payload_template,
                    **self.default_inputs,
                    **mapped_inputs,
                }
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input:
            parsed = self.try_parse_json(raw_input)
            if parsed is not None:
                logger.debug("Parsed JSON input")
                # Merge: payload_template < default_inputs < parsed
                return {
                    **self.payload_template,
                    **self.default_inputs,
                    **parsed,
                }

        # Priority 3: Fallback to simple text mapping with message_field
        logger.debug("Using message_field '%s' fallback for plain text input", self.message_field)
        return self._build_default_payload(raw_input, context_id)

    def _build_default_payload(self, user_message: str, context_id: str | None) -> Dict[str, Any]:
        """Build default payload with message_field."""
        # Build payload with custom template support
        payload: Dict[str, Any] = {
            **self.payload_template,  # Start with template (e.g., {"name": "A2A Agent"})
            **self.default_inputs,    # Add default inputs
            self.message_field: user_message,  # Add message with custom field name
        }

        # Add metadata only if not using custom template
        if not self.payload_template:
            payload["metadata"] = {
                "context_id": context_id,
            }
        else:
            # With custom template, add context_id at root if not already present
            if "context_id" not in payload:
                payload["context_id"] = context_id

        return payload

    # ---------- Framework call ----------

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any] | list:
        """
        Execute the n8n workflow by POSTing to the webhook URL with retries/backoff.

        Error policy:
          - 4xx: no retry, raise ValueError with a concise message (likely bad request/user/config).
          - 5xx / network timeouts / connect errors: retry with exponential backoff, then raise RuntimeError.
        """
        client = await self._get_client()
        req_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": req_id,
            **self.headers,
        }

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = await client.post(
                    self.webhook_url,
                    json=framework_input,
                    headers=headers,
                )
                dur_ms = int((time.monotonic() - start) * 1000)

                # Explicitly surface 4xx without retry.
                if 400 <= resp.status_code < 500:
                    text = (await resp.aread()).decode(errors="ignore")
                    raise ValueError(
                        f"n8n webhook returned {resp.status_code} "
                        f"(req_id={req_id}, {dur_ms}ms): {text[:512]}"
                    )

                # For 5xx, httpx will raise in raise_for_status().
                resp.raise_for_status()
                return resp.json()

            except HTTPStatusError as e:
                # Only 5xx should reach here (4xx is handled above).
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff * (2**attempt))
                    continue
                raise RuntimeError(
                    f"n8n upstream 5xx after retries (req_id={req_id}): {e}"
                ) from e

            except (ConnectError, ReadTimeout) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff * (2**attempt))
                    continue
                raise RuntimeError(
                    f"n8n upstream unavailable/timeout after retries (req_id={req_id}): {e}"
                ) from e

        # Should never reach here, but keeps type-checkers happy.
        raise RuntimeError("Unexpected error in call_framework retry loop.")

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Dict[str, Any] | list, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert n8n webhook response to A2A Message.

        Handles both n8n response formats:
        - Single object: {"output": "..."} (first entry only)
        - Array of objects: [{"output": "..."}, ...] (all entries)

        Args:
            framework_output: JSON response from n8n (dict or list).
            params: Original A2A parameters.

        Returns:
            A2A Message with the n8n response text.
        """
        # Handle array format (all entries from last node)
        if isinstance(framework_output, list):
            if len(framework_output) == 0:
                response_text = ""
            elif len(framework_output) == 1:
                # Single item in array - extract it
                response_text = self._extract_text_from_item(framework_output[0])
            else:
                # Multiple items - combine all outputs
                texts = []
                for item in framework_output:
                    if isinstance(item, dict):
                        text = self._extract_text_from_item(item)
                        if text:
                            texts.append(text)
                response_text = "\n".join(texts) if texts else json.dumps(framework_output, indent=2)
        elif isinstance(framework_output, dict):
            # Handle single object format (first entry only)
            response_text = self._extract_text_from_item(framework_output)
        else:
            # Fallback for unexpected types
            response_text = str(framework_output)

        # Use base class utility for context_id extraction
        context_id = self.extract_context_id(params)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(root=TextPart(text=response_text))],
        )

    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from a single n8n output item.

        Checks common field names in order of priority.

        Args:
            item: A dictionary from n8n workflow output.

        Returns:
            Extracted text string.
        """
        if not isinstance(item, dict):
            return str(item)
        if "output" in item:
            return str(item["output"])
        elif "result" in item:
            return str(item["result"])
        elif "message" in item:
            return str(item["message"])
        elif "text" in item:
            return str(item["text"])
        elif "response" in item:
            return str(item["response"])
        elif "content" in item:
            return str(item["content"])
        else:
            # Fallback: serialize entire item as JSON
            return json.dumps(item, indent=2)

    # ---------- Async Task Support ----------

    def supports_async_tasks(self) -> bool:
        """Check if this adapter supports async task execution."""
        return self.async_mode

    async def get_task(self, task_id: str) -> Task | None:
        """
        Get the current status of a task by ID.
        
        This method is used for polling task status in async task execution mode.
        
        Args:
            task_id: The ID of the task to retrieve
            
        Returns:
            The Task object with current status, or None if not found
            
        Raises:
            RuntimeError: If async mode is not enabled
        """
        if not self.async_mode:
            raise RuntimeError(
                "get_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )
        
        task = await self.task_store.get(task_id)
        if task:
            logger.debug("Retrieved task %s with state=%s", task_id, task.status.state)
        else:
            logger.debug("Task %s not found", task_id)
        return task

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the task store.
        
        This can be used to clean up completed/failed tasks to prevent memory leaks
        when using InMemoryTaskStore. Only tasks in terminal states (completed, 
        failed, canceled) should be deleted.
        
        Args:
            task_id: The ID of the task to delete
            
        Returns:
            True if the task was deleted, False if not found or still running
            
        Raises:
            RuntimeError: If async mode is not enabled
            ValueError: If the task is still running (not in a terminal state)
        """
        if not self.async_mode:
            raise RuntimeError(
                "delete_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )
        
        task = await self.task_store.get(task_id)
        if not task:
            return False
        
        # Only allow deletion of tasks in terminal states
        terminal_states = {TaskState.completed, TaskState.failed, TaskState.canceled}
        if task.status.state not in terminal_states:
            raise ValueError(
                f"Cannot delete task {task_id} with state={task.status.state}. "
                f"Only tasks in terminal states ({', '.join(s.value for s in terminal_states)}) can be deleted."
            )
        
        await self.task_store.delete(task_id)
        logger.debug("Deleted task %s", task_id)
        return True

    async def cancel_task(self, task_id: str) -> Task | None:
        """
        Attempt to cancel a running task.
        
        Note: This only cancels the background asyncio task. If the HTTP request
        to n8n is already in flight, it cannot be cancelled on the n8n side.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            The updated Task object with state="canceled", or None if not found
        """
        if not self.async_mode:
            raise RuntimeError(
                "cancel_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )
        
        # Mark task as cancelled to prevent race conditions
        self._cancelled_tasks.add(task_id)
        
        # Cancel the background task if still running and wait for it
        bg_task = self._background_tasks.get(task_id)
        if bg_task and not bg_task.done():
            bg_task.cancel()
            logger.debug("Cancelling background task for %s", task_id)
            # Wait for the task to actually finish
            try:
                await bg_task
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
            except Exception:
                pass  # Task may have failed, we're cancelling anyway
        
        # Update task state to canceled
        task = await self.task_store.get(task_id)
        if task:
            now = datetime.now(timezone.utc).isoformat()
            canceled_task = Task(
                id=task_id,
                context_id=task.context_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    timestamp=now,
                ),
                history=task.history,
            )
            await self.task_store.save(canceled_task)
            logger.debug("Task %s marked as canceled", task_id)
            return canceled_task
        
        return None

    # ---------- Lifecycle ----------

    async def close(self) -> None:
        """Close the HTTP client and cancel pending background tasks."""
        # Mark all tasks as cancelled to prevent state updates
        for task_id in self._background_tasks:
            self._cancelled_tasks.add(task_id)
        
        # Cancel all pending background tasks
        tasks_to_cancel = []
        for task_id, bg_task in list(self._background_tasks.items()):
            if not bg_task.done():
                bg_task.cancel()
                tasks_to_cancel.append(bg_task)
                logger.debug("Cancelling background task %s during close", task_id)
        
        # Wait for all cancelled tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        self._background_tasks.clear()
        self._cancelled_tasks.clear()
        
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def supports_streaming(self) -> bool:
        """This adapter does not support streaming responses."""
        return False


