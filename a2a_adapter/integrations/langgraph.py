"""
LangGraph adapter for A2A Protocol.

This adapter enables LangGraph compiled workflows to be exposed as A2A-compliant
agents with support for both streaming and non-streaming modes.

Contains:
    - LangGraphAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - LangGraphAgentAdapter (v0.1): Legacy interface, deprecated

Supports flexible input handling:
- input_mapper: Custom function for full control over input transformation
- parse_json_input: Automatic JSON parsing for structured inputs
- input_key: Simple text mapping to a single key (default fallback)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict

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


class LangGraphAdapter(BaseA2AAdapter):
    """Adapter for LangGraph compiled state graphs.

    Supports both streaming (via ``astream()``) and non-streaming
    (via ``ainvoke()``) execution. Streaming yields state deltas
    as they're produced by the graph.

    Example::

        from langgraph.graph import StateGraph
        from a2a_adapter import LangGraphAdapter, serve_agent

        graph = builder.compile()

        adapter = LangGraphAdapter(graph=graph)
        serve_agent(adapter, port=9003)
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        output_key: str | None = None,
        state_key: str | None = None,
        timeout: int = 60,
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        """Initialize the LangGraph adapter.

        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: Key in the state dict for input (default: "messages").
            output_key: Optional key to extract from final state.
            state_key: Optional key for extracting streaming state text.
            timeout: Execution timeout in seconds (default: 60).
            parse_json_input: Auto-parse JSON input (default: True).
            input_mapper: Custom function for input transformation.
            default_inputs: Default values to merge with parsed inputs.
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.state_key = state_key or output_key
        self.timeout = timeout
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Invoke the graph and return a text response."""
        inputs = self._build_inputs(user_input, context_id)
        result = await asyncio.wait_for(
            self.graph.ainvoke(inputs), timeout=self.timeout,
        )
        return self._extract_output(result)

    async def stream(self, user_input: str, context_id: str | None = None, **kwargs) -> AsyncIterator[str]:
        """Stream state deltas from the graph via astream()."""
        inputs = self._build_inputs(user_input, context_id)
        last_text = ""
        async for state in self.graph.astream(inputs):
            text = self._extract_streaming_text(state)
            if text and text != last_text:
                delta = text[len(last_text):] if text.startswith(last_text) else text
                last_text = text
                if delta:
                    yield delta

    def supports_streaming(self) -> bool:
        """Auto-detect streaming support from the graph."""
        return hasattr(self.graph, "astream")

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or "LangGraphAdapter",
            description=self._description,
            streaming=self.supports_streaming(),
        )

    # ──── Internal: Input Building ────

    def _build_inputs(self, user_input: str, context_id: str | None) -> Dict[str, Any]:
        """Build graph inputs with 3-priority pipeline."""
        # Priority 1: Custom input_mapper
        if self.input_mapper is not None:
            try:
                return {**self.default_inputs, **self.input_mapper(user_input, context_id)}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input and user_input.strip().startswith("{"):
            try:
                parsed = json.loads(user_input)
                if isinstance(parsed, dict):
                    clean = {k: v for k, v in parsed.items() if k != "context_id"}
                    return {**self.default_inputs, **clean}
            except json.JSONDecodeError:
                pass

        # Priority 3: Fallback to input_key
        return self._build_default_input(user_input)

    def _build_default_input(self, user_message: str) -> Dict[str, Any]:
        """Build default input based on input_key."""
        base = dict(self.default_inputs)
        if self.input_key == "messages":
            try:
                from langchain_core.messages import HumanMessage
                base["messages"] = [HumanMessage(content=user_message)]
            except ImportError:
                base["messages"] = [{"role": "user", "content": user_message}]
        else:
            base[self.input_key] = user_message
        return base

    # ──── Internal: Output Extraction ────

    def _extract_output(self, state: Any) -> str:
        """Extract text from final graph state."""
        if not isinstance(state, dict):
            return str(state)

        # Use output_key if specified
        if self.output_key and self.output_key in state:
            return self._extract_value_text(state[self.output_key])

        # Try "messages" key (common in chat workflows)
        if "messages" in state:
            messages = state["messages"]
            if messages:
                return self._extract_message_content(messages[-1])

        # Try common output keys
        for key in ("output", "response", "result", "answer", "text", "content"):
            if key in state:
                return self._extract_value_text(state[key])

        # Fallback: serialize state (excluding internal keys)
        clean = {k: v for k, v in state.items() if not k.startswith("_")}
        return json.dumps(clean, indent=2, default=str)

    def _extract_streaming_text(self, state: Any) -> str:
        """Extract text from intermediate streaming state."""
        if not isinstance(state, dict):
            return str(state)

        if self.state_key and self.state_key in state:
            return self._extract_value_text(state[self.state_key])

        if "messages" in state:
            messages = state["messages"]
            if messages:
                return self._extract_message_content(messages[-1])

        for key in ("output", "response", "text", "content"):
            if key in state:
                return self._extract_value_text(state[key])

        return ""

    @staticmethod
    def _extract_value_text(value: Any) -> str:
        """Extract text from a value (str, dict, list)."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("text", "content", "output"):
                if key in value:
                    return str(value[key])
            return json.dumps(value, indent=2, default=str)
        if isinstance(value, list):
            return "\n".join(
                LangGraphAdapter._extract_value_text(item) for item in value
            )
        return str(value)

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        """Extract content from a message object (LangChain or dict)."""
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                return " ".join(parts)
            return str(content)

        if isinstance(message, dict):
            for key in ("content", "text"):
                if key in message:
                    return str(message[key])

        return str(message)


# ═══════════════════════════════════════════════════════════════
# v0.1 Adapter (legacy, deprecated — will be removed in v0.3)
# ═══════════════════════════════════════════════════════════════


class LangGraphAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating LangGraph compiled workflows as A2A agents.

    This adapter works with LangGraph's CompiledGraph objects (the result of
    calling .compile() on a StateGraph) and supports both streaming and
    non-streaming execution modes.

    Supports three execution patterns:

    1. **Synchronous Mode** (default):
       - Blocks until the workflow completes
       - Returns a Message with the final result
       - Best for quick workflows (< 30 seconds)

    2. **Streaming Mode**:
       - Streams intermediate results as they're produced
       - Uses LangGraph's astream() method
       - Best for real-time feedback during execution

    3. **Async Task Mode** (async_mode=True):
       - Returns a Task with state="working" immediately
       - Processes the workflow in the background
       - Clients can poll get_task() for status updates
       - Best for long-running workflows

    Input Handling (in priority order):

    1. **input_mapper** (highest priority):
       Custom function for full control over input transformation.
       Signature: (raw_input: str, context_id: str | None) -> dict

    2. **parse_json_input** (default: True):
       Automatically parse JSON input and pass directly to graph.

    3. **input_key** (fallback):
       Map plain text to a single key when JSON parsing fails.

    Example:
        >>> from langgraph.graph import StateGraph
        >>> from typing import TypedDict
        >>>
        >>> class State(TypedDict):
        ...     messages: list
        ...     output: str
        >>>
        >>> def process(state: State) -> State:
        ...     return {"output": f"Processed: {state['messages'][-1]}"}
        >>>
        >>> builder = StateGraph(State)
        >>> builder.add_node("process", process)
        >>> builder.set_entry_point("process")
        >>> builder.set_finish_point("process")
        >>> graph = builder.compile()
        >>>
        >>> # Basic usage
        >>> adapter = LangGraphAgentAdapter(graph=graph)
        >>>
        >>> # With custom input mapper
        >>> def my_mapper(raw_input: str, context_id: str | None) -> dict:
        ...     return {"messages": [{"role": "user", "content": raw_input}]}
        >>> adapter = LangGraphAgentAdapter(graph=graph, input_mapper=my_mapper)
    """

    def __init__(
        self,
        graph: Any,  # Type: CompiledGraph (avoiding hard dependency)
        input_key: str = "messages",
        output_key: str | None = None,
        state_key: str | None = None,
        async_mode: bool = False,
        task_store: "TaskStore | None" = None,
        async_timeout: int = 300,
        timeout: int = 60,  # Sync mode timeout
        # Flexible input handling parameters
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the LangGraph adapter.

        Args:
            graph: A LangGraph CompiledGraph instance (result of StateGraph.compile())
            input_key: The key in the state dict for input messages (default: "messages").
                       Set to "input" for simple string input workflows.
                       Used as fallback when JSON parsing fails or is disabled.
            output_key: Optional key to extract from final state. If None, the adapter
                        will try common keys like "output", "response", "messages".
            state_key: Optional key to use when extracting state for streaming events.
                       If None, uses output_key or auto-detection.
            async_mode: If True, return Task immediately and process in background.
                        If False (default), block until workflow completes.
            task_store: Optional TaskStore for persisting task state. If not provided
                        and async_mode is True, uses InMemoryTaskStore.
            async_timeout: Timeout for async task execution in seconds (default: 300).
            timeout: Timeout for sync mode execution in seconds (default: 60).
            parse_json_input: If True (default), attempt to parse input as JSON and use
                              the parsed dict directly as graph state.
            input_mapper: Optional custom function to transform raw input to graph state.
                          Signature: (raw_input: str, context_id: str | None) -> dict.
                          When provided, this takes highest priority over other methods.
            default_inputs: Optional dict of default values to merge with parsed inputs.
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.state_key = state_key or output_key
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
            if result.status and result.status.message:
                return result.status.message
            return Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=result.context_id,
                parts=[Part(root=TextPart(text="Workflow completed"))],
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
            self._execute_workflow_with_timeout(task_id, context_id, params)
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

    async def _execute_workflow_with_timeout(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """Execute the workflow with a timeout wrapper."""
        try:
            await asyncio.wait_for(
                self._execute_workflow_background(task_id, context_id, params),
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
        """Execute the LangGraph workflow in the background and update task state."""
        try:
            logger.debug("Starting background execution for task %s", task_id)

            # Execute the workflow
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

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to LangGraph state input.

        Processing priority:
        1. input_mapper (custom function) - highest priority
        2. parse_json_input (auto JSON parsing)
        3. input_key (fallback for plain text)

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with graph state input
        """
        # Use base class utility for raw input extraction
        raw_input = self.extract_raw_input(params)
        context_id = self.extract_context_id(params)

        # Priority 1: Custom input_mapper function (highest priority)
        if self.input_mapper is not None:
            try:
                mapped_inputs = self.input_mapper(raw_input, context_id)
                logger.debug("Used input_mapper to transform input")
                return {**self.default_inputs, **mapped_inputs}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input:
            parsed = self.try_parse_json(raw_input)
            if parsed is not None:
                logger.debug("Parsed JSON input")
                # Remove context_id from parsed input as LangGraph doesn't need it
                parsed_clean = {k: v for k, v in parsed.items() if k != "context_id"}
                return {**self.default_inputs, **parsed_clean}

        # Priority 3: Fallback to text mapping with input_key
        logger.debug("Using input_key '%s' fallback for plain text input", self.input_key)
        return self._build_default_input(raw_input)

    def _build_default_input(self, user_message: str) -> Dict[str, Any]:
        """Build default input based on input_key."""
        base_input = dict(self.default_inputs)

        if self.input_key == "messages":
            # LangGraph message format (for chat-like workflows)
            # Try to use LangChain message format if available
            try:
                from langchain_core.messages import HumanMessage
                base_input["messages"] = [HumanMessage(content=user_message)]
            except ImportError:
                # Fallback to dict format
                base_input["messages"] = [{"role": "user", "content": user_message}]
        else:
            # Simple input key (e.g., "input", "query", etc.)
            base_input[self.input_key] = user_message

        return base_input

    # ---------- Framework call ----------

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow with the provided input.

        Args:
            framework_input: Input state dictionary for the graph
            params: Original A2A parameters (for context)

        Returns:
            Final state from the graph execution

        Raises:
            Exception: If graph execution fails
            asyncio.TimeoutError: If execution exceeds timeout (sync mode)
        """
        logger.debug("Invoking LangGraph with input: %s", framework_input)

        try:
            result = await asyncio.wait_for(
                self.graph.ainvoke(framework_input),
                timeout=self.timeout,
            )
            logger.debug("LangGraph returned state with keys: %s", list(result.keys()) if isinstance(result, dict) else type(result).__name__)
            return result
        except asyncio.TimeoutError as e:
            logger.error("LangGraph workflow timed out after %s seconds", self.timeout)
            raise RuntimeError(f"Workflow timed out after {self.timeout} seconds") from e

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Dict[str, Any], params: MessageSendParams
    ) -> Message | Task:
        """
        Convert LangGraph final state to A2A Message.

        Args:
            framework_output: Final state from graph execution
            params: Original A2A parameters

        Returns:
            A2A Message with the workflow's response
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
        Extract text content from LangGraph state.

        Handles various output patterns:
        - output_key specified: extract that key
        - "messages" key: extract last message content
        - Common keys: "output", "response", "result", "answer"

        Args:
            framework_output: Final state from the graph

        Returns:
            Extracted text string
        """
        if not isinstance(framework_output, dict):
            return str(framework_output)

        # Use output_key if specified
        if self.output_key and self.output_key in framework_output:
            return self._extract_value_text(framework_output[self.output_key])

        # Try "messages" key (common in chat workflows)
        if "messages" in framework_output:
            messages = framework_output["messages"]
            if messages and len(messages) > 0:
                last_message = messages[-1]
                return self._extract_message_content(last_message)

        # Try common output keys
        for key in ["output", "response", "result", "answer", "text", "content"]:
            if key in framework_output:
                return self._extract_value_text(framework_output[key])

        # Fallback: serialize entire state (excluding internal keys)
        clean_state = {k: v for k, v in framework_output.items() if not k.startswith("_")}
        return json.dumps(clean_state, indent=2, default=str)

    def _extract_value_text(self, value: Any) -> str:
        """Extract text from a value (handles strings, dicts, lists)."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            # Try common text keys in dict
            for key in ["text", "content", "output"]:
                if key in value:
                    return str(value[key])
            return json.dumps(value, indent=2, default=str)
        if isinstance(value, list):
            # Join list items
            return "\n".join(self._extract_value_text(item) for item in value)
        return str(value)

    def _extract_message_content(self, message: Any) -> str:
        """Extract content from a message object (LangChain or dict)."""
        # LangChain message with content attribute
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                return " ".join(text_parts)
            return str(content)

        # Dict message
        if isinstance(message, dict):
            if "content" in message:
                return str(message["content"])
            if "text" in message:
                return str(message["text"])

        return str(message)

    # ---------- Streaming support ----------

    async def handle_stream(
        self, params: MessageSendParams
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle a streaming A2A message request.

        Uses LangGraph's astream() or astream_events() to yield intermediate
        results as the workflow executes.

        Args:
            params: A2A message parameters

        Yields:
            Server-Sent Events compatible dictionaries with streaming chunks
        """
        framework_input = await self.to_framework(params)
        context_id = self.extract_context_id(params)
        message_id = str(uuid.uuid4())

        logger.debug("Starting LangGraph stream with input: %s", framework_input)

        accumulated_text = ""
        last_state = None

        # Stream from LangGraph
        async for state in self.graph.astream(framework_input):
            last_state = state

            # Extract text from current state
            text = self._extract_streaming_text(state)

            if text and text != accumulated_text:
                # Calculate the new chunk (delta)
                new_content = text[len(accumulated_text):] if text.startswith(accumulated_text) else text
                accumulated_text = text

                if new_content:
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "content": new_content,
                        }),
                    }

        # Use final state if we have it, otherwise use accumulated
        final_text = self._extract_output_text(last_state) if last_state else accumulated_text

        # Send final message with complete response
        final_message = Message(
            role=Role.agent,
            message_id=message_id,
            context_id=context_id,
            parts=[Part(root=TextPart(text=final_text))],
        )

        # Send completion event
        yield {
            "event": "done",
            "data": json.dumps({
                "status": "completed",
                "message": final_message.model_dump() if hasattr(final_message, "model_dump") else str(final_message),
            }),
        }

        logger.debug("LangGraph stream completed")

    def _extract_streaming_text(self, state: Any) -> str:
        """
        Extract text from intermediate streaming state.

        Args:
            state: Intermediate state from astream()

        Returns:
            Current text content
        """
        if not isinstance(state, dict):
            return str(state)

        # Use state_key if specified
        if self.state_key and self.state_key in state:
            return self._extract_value_text(state[self.state_key])

        # Try messages (for chat workflows)
        if "messages" in state:
            messages = state["messages"]
            if messages:
                # Get content from last message
                last = messages[-1]
                return self._extract_message_content(last)

        # Try common keys
        for key in ["output", "response", "text", "content"]:
            if key in state:
                return self._extract_value_text(state[key])

        return ""

    def supports_streaming(self) -> bool:
        """
        Check if the graph supports streaming.

        Returns:
            True if the graph has an astream method
        """
        return hasattr(self.graph, "astream")

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
