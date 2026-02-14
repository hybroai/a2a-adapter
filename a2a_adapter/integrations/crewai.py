"""
CrewAI adapter for A2A Protocol.

This adapter enables CrewAI crews to be exposed as A2A-compliant agents
by translating A2A messages to crew inputs and crew outputs back to A2A.

Contains:
    - CrewAIAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - CrewAIAgentAdapter (v0.1): Legacy interface, deprecated

Input handling modes (in priority order):
1. input_mapper: Custom function for full control over input transformation
2. parse_json_input: Automatic JSON parsing for structured inputs
3. inputs_key: Simple text mapping to a single key (default fallback)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from ..adapter import BaseAgentAdapter
from ..base_adapter import AdapterMetadata, BaseA2AAdapter

# Lazy import for TaskStore to avoid hard dependency
try:
    from a2a.server.tasks import InMemoryTaskStore, TaskStore

    _HAS_TASK_STORE = True
except ImportError:
    _HAS_TASK_STORE = False
    TaskStore = None  # type: ignore
    InMemoryTaskStore = None  # type: ignore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# v0.2 Adapter (new, recommended)
# ═══════════════════════════════════════════════════════════════


class CrewAIAdapter(BaseA2AAdapter):
    """Adapter for CrewAI crews.

    Executes a CrewAI Crew via ``kickoff_async()`` (with sync fallback)
    and returns the crew result as text. CrewAI does not support streaming,
    so only ``invoke()`` is implemented.

    Example::

        from crewai import Crew, Agent, Task as CrewTask
        from a2a_adapter import CrewAIAdapter, serve_agent

        researcher = Agent(role="Researcher", ...)
        task = CrewTask(description="Research {inputs}", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task])

        adapter = CrewAIAdapter(crew=crew)
        serve_agent(adapter, port=9004)
    """

    def __init__(
        self,
        crew: Any,
        inputs_key: str = "inputs",
        timeout: int = 300,
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        """Initialize the CrewAI adapter.

        Args:
            crew: A CrewAI Crew instance to execute.
            inputs_key: Key for passing text inputs to the crew (default: "inputs").
            timeout: Execution timeout in seconds (default: 300).
            parse_json_input: Auto-parse JSON input (default: True).
            input_mapper: Custom function for input transformation.
            default_inputs: Default values to merge with parsed inputs.
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
        """
        self.crew = crew
        self.inputs_key = inputs_key
        self.timeout = timeout
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Execute the crew and return the result as text."""
        inputs = self._build_inputs(user_input, context_id)
        try:
            result = await asyncio.wait_for(
                self.crew.kickoff_async(inputs=inputs),
                timeout=self.timeout,
            )
        except AttributeError:
            # Fallback for older CrewAI versions without async support
            logger.warning("CrewAI kickoff_async not available, using sync fallback")
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.crew.kickoff(inputs=inputs)),
                timeout=self.timeout,
            )
        return self._extract_output(result)

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or "CrewAIAdapter",
            description=self._description,
            streaming=False,
        )

    # ──── Internal: Input Building ────

    def _build_inputs(self, user_input: str, context_id: str | None) -> Dict[str, Any]:
        """Build crew inputs with 3-priority pipeline."""
        # Priority 1: Custom input_mapper
        if self.input_mapper is not None:
            try:
                mapped = self.input_mapper(user_input, context_id)
                return {**self.default_inputs, **mapped}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input and user_input.strip().startswith("{"):
            try:
                parsed = json.loads(user_input)
                if isinstance(parsed, dict):
                    return {**self.default_inputs, **parsed}
            except json.JSONDecodeError:
                pass

        # Priority 3: Fallback to inputs_key
        return {**self.default_inputs, self.inputs_key: user_input}

    # ──── Internal: Output Extraction ────

    @staticmethod
    def _extract_output(output: Any) -> str:
        """Extract text from CrewAI output (CrewOutput, dict, str)."""
        # CrewOutput with .raw attribute
        if hasattr(output, "raw"):
            return str(output.raw)

        # CrewOutput with .result attribute
        if hasattr(output, "result"):
            return str(output.result)

        # Dictionary output
        if isinstance(output, dict):
            for key in ("output", "result", "response", "answer", "text"):
                if key in output:
                    return str(output[key])
            return json.dumps(output, indent=2)

        return str(output)


# ═══════════════════════════════════════════════════════════════
# v0.1 Adapter (legacy, deprecated — will be removed in v0.3)
# ═══════════════════════════════════════════════════════════════


class CrewAIAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating CrewAI crews as A2A agents.

    This adapter handles the translation between A2A protocol messages
    and CrewAI's crew execution model.

    Supports two execution modes:

    1. **Synchronous Mode** (default):
       - Blocks until the crew completes execution
       - Returns a Message with the crew result
       - Best for quick crews (< 30 seconds)

    2. **Async Task Mode** (async_mode=True):
       - Returns a Task with state="working" immediately
       - Processes the crew execution in the background
       - Clients can poll get_task() for status updates
       - Best for long-running crews

    Input Handling (in priority order):

    1. **input_mapper** (highest priority):
       Custom function for full control over input transformation.
       Use when you need complex parsing or validation logic.

    2. **parse_json_input** (default: True):
       Automatically parse JSON input and pass directly to crew.
       Perfect for structured inputs matching tasks.yaml variables.

    3. **inputs_key** (fallback):
       Map plain text to a single key when JSON parsing fails.

    Example:
        >>> from crewai import Crew, Agent, Task as CrewTask
        >>>
        >>> researcher = Agent(role="Researcher", ...)
        >>> task = CrewTask(description="Research topic", agent=researcher)
        >>> crew = Crew(agents=[researcher], tasks=[task])
        >>>
        >>> # Basic usage with auto JSON parsing
        >>> adapter = CrewAIAgentAdapter(crew=crew)
        >>>
        >>> # With custom input mapper
        >>> def my_mapper(raw_input: str, context_id: str | None) -> dict:
        ...     data = json.loads(raw_input)
        ...     return {"customer_domain": data.get("domain", "default.com")}
        >>> adapter = CrewAIAgentAdapter(crew=crew, input_mapper=my_mapper)
        >>>
        >>> # With default values
        >>> adapter = CrewAIAgentAdapter(
        ...     crew=crew,
        ...     default_inputs={"customer_domain": "example.com"}
        ... )
    """

    def __init__(
        self,
        crew: Any,  # Type: crewai.Crew (avoiding hard dependency)
        inputs_key: str = "inputs",
        async_mode: bool = False,
        task_store: "TaskStore | None" = None,
        async_timeout: int = 600,  # 10 minutes default for crews
        timeout: int = 300,  # 5 minutes default for sync mode
        # New flexible input handling parameters
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the CrewAI adapter.

        Args:
            crew: A CrewAI Crew instance to execute
            inputs_key: The key name for passing text inputs to the crew (default: "inputs").
                        Used as fallback when JSON parsing fails or is disabled.
            async_mode: If True, return Task immediately and process in background.
                        If False (default), block until crew completes.
            task_store: Optional TaskStore for persisting task state. If not provided
                        and async_mode is True, uses InMemoryTaskStore.
            async_timeout: Timeout for async task execution in seconds (default: 600).
            timeout: Timeout for sync mode execution in seconds (default: 300).
                     CrewAI crews can be long-running, so this is set higher than other adapters.
            parse_json_input: If True (default), attempt to parse input as JSON and use
                              the parsed dict directly as crew inputs. This allows sending
                              structured data matching your tasks.yaml variable names
                              (e.g., {"customer_domain": "...", "project_description": "..."}).
            input_mapper: Optional custom function to transform raw input to crew inputs.
                          Signature: (raw_input: str, context_id: str | None) -> dict.
                          When provided, this takes highest priority over other methods.
            default_inputs: Optional dict of default values to merge with parsed inputs.
                            Useful for providing fallback values when some fields are missing.
        """
        self.crew = crew
        self.inputs_key = inputs_key
        self.timeout = timeout

        # Flexible input handling configuration
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}

        # Async task mode configuration
        self.async_mode = async_mode
        self.async_timeout = async_timeout
        self._background_tasks: Dict[str, "asyncio.Task[None]"] = {}
        self._cancelled_tasks: set[str] = set()

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

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """
        Handle a non-streaming A2A message request.

        In sync mode (default): Blocks until crew completes, returns Message.
        In async mode: Returns Task immediately, processes in background.
        """
        if self.async_mode:
            return await self._handle_async(params)
        else:
            return await self._handle_sync(params)

    async def _handle_sync(self, params: MessageSendParams) -> Message:
        """Handle request synchronously - blocks until crew completes."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        result = await self.from_framework(framework_output, params)

        # In sync mode, always return Message
        if isinstance(result, Task):
            if result.status and result.status.message:
                return result.status.message
            return Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=result.context_id,
                parts=[Part(root=TextPart(text="Crew completed"))],
            )
        return result

    async def _handle_async(self, params: MessageSendParams) -> Task:
        """
        Handle request asynchronously - returns Task immediately, processes in background.
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
            self._execute_crew_with_timeout(task_id, context_id, params)
        )
        self._background_tasks[task_id] = bg_task

        # Clean up background task reference when done
        def _on_task_done(t: "asyncio.Task[None]") -> None:
            self._background_tasks.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)
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

    async def _execute_crew_with_timeout(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """Execute the crew with a timeout wrapper."""
        try:
            await asyncio.wait_for(
                self._execute_crew_background(task_id, context_id, params),
                timeout=self.async_timeout,
            )
        except asyncio.TimeoutError:
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return

            logger.error("Task %s timed out after %s seconds", task_id, self.async_timeout)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=f"Crew timed out after {self.async_timeout} seconds"))],
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

    async def _execute_crew_background(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """Execute the CrewAI crew in the background and update task state."""
        try:
            logger.debug("Starting background execution for task %s", task_id)

            # Execute the crew
            framework_input = await self.to_framework(params)
            framework_output = await self.call_framework(framework_input, params)

            # Check if task was cancelled during execution
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled during execution", task_id)
                return

            # Convert to message
            response_text = self._extract_output_text(framework_output)
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
            logger.debug("Task %s was cancelled", task_id)
            raise

        except Exception as e:
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return

            logger.error("Task %s failed: %s", task_id, e)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=f"Crew failed: {str(e)}"))],
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

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to CrewAI crew inputs.

        Processing priority:
        1. input_mapper (custom function) - highest priority
        2. parse_json_input (auto JSON parsing)
        3. inputs_key (fallback for plain text)

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with crew input data matching tasks.yaml variables
        """
        # Extract raw message content
        raw_input = self.extract_raw_input(params)
        context_id = self.extract_context_id(params)

        # Priority 1: Custom input_mapper function (highest priority)
        if self.input_mapper is not None:
            try:
                mapped_inputs = self.input_mapper(raw_input, context_id)
                logger.debug("Used input_mapper to transform input: %s", mapped_inputs)
                # Merge with default_inputs (user-provided values take precedence)
                return {**self.default_inputs, **mapped_inputs, "context_id": context_id}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back to parse_json_input", e)

        # Priority 2: Auto JSON parsing (using base class utility)
        if self.parse_json_input:
            parsed = self.try_parse_json(raw_input)
            if parsed is not None:
                logger.debug("Parsed JSON input: %s", parsed)
                # Merge: default_inputs < parsed < context_id
                return {**self.default_inputs, **parsed, "context_id": context_id}

        # Priority 3: Fallback to simple text mapping with inputs_key
        logger.debug("Using inputs_key fallback for plain text input")
        return {
            **self.default_inputs,
            self.inputs_key: raw_input,
            "message": raw_input,
            "context_id": context_id,
        }


    # ---------- Framework call ----------

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Any:
        """
        Execute the CrewAI crew with the provided inputs.

        Args:
            framework_input: Input dictionary for the crew
            params: Original A2A parameters (for context)

        Returns:
            CrewAI crew execution output

        Raises:
            Exception: If crew execution fails
            asyncio.TimeoutError: If execution exceeds timeout (sync mode)
        """
        logger.debug("Executing CrewAI crew with inputs: %s", framework_input)

        try:
            # CrewAI supports async execution via kickoff_async
            try:
                result = await asyncio.wait_for(
                    self.crew.kickoff_async(inputs=framework_input),
                    timeout=self.timeout,
                )
                logger.debug("CrewAI crew returned: %s", type(result).__name__)
                return result
            except AttributeError:
                # Fallback for older CrewAI versions without async support
                # Note: This will block the event loop
                logger.warning("CrewAI kickoff_async not available, using sync fallback")
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self.crew.kickoff(inputs=framework_input)
                    ),
                    timeout=self.timeout,
                )
                return result
        except asyncio.TimeoutError as e:
            logger.error("CrewAI crew timed out after %s seconds", self.timeout)
            raise RuntimeError(f"Crew timed out after {self.timeout} seconds") from e

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert CrewAI crew output to A2A Message.

        Args:
            framework_output: Output from crew execution
            params: Original A2A parameters

        Returns:
            A2A Message with the crew's response
        """
        response_text = self._extract_output_text(framework_output)
        context_id = self.extract_context_id(params)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(root=TextPart(text=response_text))],
        )

    def _extract_output_text(self, framework_output: Any) -> str:
        """
        Extract text content from CrewAI output.

        Args:
            framework_output: Output from the crew

        Returns:
            Extracted text string
        """
        # CrewOutput object with raw attribute
        if hasattr(framework_output, "raw"):
            return str(framework_output.raw)

        # CrewOutput object with result attribute
        if hasattr(framework_output, "result"):
            return str(framework_output.result)

        # Dictionary output
        if isinstance(framework_output, dict):
            for key in ["output", "result", "response", "answer", "text"]:
                if key in framework_output:
                    return str(framework_output[key])
            # Fallback: serialize as JSON
            return json.dumps(framework_output, indent=2)

        # String or other type - convert to string
        return str(framework_output)

    # ---------- Async Task Support ----------

    def supports_async_tasks(self) -> bool:
        """Check if this adapter supports async task execution."""
        return self.async_mode

    async def get_task(self, task_id: str) -> Task | None:
        """
        Get the current status of a task by ID.

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

    async def cancel_task(self, task_id: str) -> Task | None:
        """
        Attempt to cancel a running task.

        Note: CrewAI crews cannot be gracefully interrupted once started.
        This will mark the task as cancelled but the crew may continue running.

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

        # Cancel the background task if still running
        bg_task = self._background_tasks.get(task_id)
        if bg_task and not bg_task.done():
            bg_task.cancel()
            logger.debug("Cancelling background task for %s", task_id)
            try:
                await bg_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

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
        """Cancel pending background tasks."""
        for task_id in self._background_tasks:
            self._cancelled_tasks.add(task_id)

        tasks_to_cancel = []
        for task_id, bg_task in list(self._background_tasks.items()):
            if not bg_task.done():
                bg_task.cancel()
                tasks_to_cancel.append(bg_task)
                logger.debug("Cancelling background task %s during close", task_id)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._background_tasks.clear()
        self._cancelled_tasks.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def supports_streaming(self) -> bool:
        """CrewAI does not support streaming responses."""
        return False
