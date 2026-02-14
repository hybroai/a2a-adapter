"""
Single-agent A2A server helpers.

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`a2a_adapter.server` instead:

    - ``build_agent_app()`` → ``to_a2a()``
    - ``serve_agent(card, adapter)`` → ``serve_agent(adapter)``
    - ``AdapterRequestHandler`` → removed (SDK's DefaultRequestHandler)

    This module will be removed in v0.3.0.
"""

import warnings
from typing import AsyncGenerator

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import UnsupportedOperationError
from a2a.utils.errors import ServerError
from a2a.server.context import ServerCallContext
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    CancelTaskResponse,
    DeleteTaskPushNotificationConfigParams,
    DeleteTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigResponse,
    GetTaskRequest,
    GetTaskResponse,
    ListTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigResponse,
    Message,
    MessageSendParams,
    PushNotificationConfig,
    SetTaskPushNotificationConfigRequest,
    SetTaskPushNotificationConfigResponse,
    Task,
    TaskResubscriptionRequest,
    TaskStatusUpdateEvent,
)

from .adapter import BaseAgentAdapter


class AdapterRequestHandler(RequestHandler):
    """
    Wrapper that adapts BaseAgentAdapter to A2A's RequestHandler interface.
    
    This class bridges the gap between our adapter abstraction and the
    official A2A SDK's RequestHandler protocol.
    
    Supports:
    - Basic message send (sync and async)
    - Task get/cancel for async adapters
    - Push notification configuration for adapters that support it
    """

    def __init__(self, adapter: BaseAgentAdapter):
        """
        Initialize the request handler with an adapter.
        
        Args:
            adapter: The BaseAgentAdapter instance to wrap
        """
        self.adapter = adapter

    async def on_message_send(
        self, 
        params: MessageSendParams,
        context: ServerCallContext
    ) -> Message | Task:
        """
        Handle a non-streaming message send request.
        
        Args:
            params: A2A message parameters
            context: Server call context
            
        Returns:
            A2A Message or Task response
        """
        return await self.adapter.handle(params)

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext
    ) -> AsyncGenerator[Message, None]:
        """
        Handle a streaming message send request.
        
        Args:
            params: A2A message parameters
            context: Server call context
            
        Yields:
            A2A Message responses
        """
        async for event in self.adapter.handle_stream(params):
            yield event

    # Task-related methods
    
    async def on_get_task(
        self,
        params: GetTaskRequest,
        context: ServerCallContext
    ) -> GetTaskResponse:
        """Get task status."""
        if not self.adapter.supports_async_tasks():
            raise ServerError(error=UnsupportedOperationError())
        
        task = await self.adapter.get_task(params.id)
        if task is None:
            raise ServerError(error=UnsupportedOperationError(message=f"Task {params.id} not found"))
        
        return GetTaskResponse(result=task)

    async def on_cancel_task(
        self,
        params: CancelTaskRequest,
        context: ServerCallContext
    ) -> CancelTaskResponse:
        """Cancel task."""
        if not self.adapter.supports_async_tasks():
            raise ServerError(error=UnsupportedOperationError())
        
        task = await self.adapter.cancel_task(params.id)
        if task is None:
            raise ServerError(error=UnsupportedOperationError(message=f"Task {params.id} not found"))
        
        return CancelTaskResponse(result=task)

    async def on_resubscribe_to_task(
        self,
        params: TaskResubscriptionRequest,
        context: ServerCallContext
    ) -> AsyncGenerator[TaskStatusUpdateEvent, None]:
        """Resubscribe to task - not supported."""
        raise ServerError(error=UnsupportedOperationError())
        yield  # Make this an async generator

    # Push notification methods
    
    async def on_set_task_push_notification_config(
        self,
        params: SetTaskPushNotificationConfigRequest,
        context: ServerCallContext
    ) -> SetTaskPushNotificationConfigResponse:
        """Set push notification config."""
        if not hasattr(self.adapter, 'supports_push_notifications') or not self.adapter.supports_push_notifications():
            raise ServerError(error=UnsupportedOperationError())
        
        success = await self.adapter.set_push_notification_config(
            params.taskId,
            params.pushNotificationConfig
        )
        if not success:
            raise ServerError(error=UnsupportedOperationError(message=f"Task {params.taskId} not found"))
        
        return SetTaskPushNotificationConfigResponse(result=params.pushNotificationConfig)

    async def on_get_task_push_notification_config(
        self,
        params: GetTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> GetTaskPushNotificationConfigResponse:
        """Get push notification config."""
        if not hasattr(self.adapter, 'supports_push_notifications') or not self.adapter.supports_push_notifications():
            raise ServerError(error=UnsupportedOperationError())
        
        config = await self.adapter.get_push_notification_config(params.taskId)
        if config is None:
            raise ServerError(error=UnsupportedOperationError(message=f"No push config for task {params.taskId}"))
        
        return GetTaskPushNotificationConfigResponse(result=config)

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> ListTaskPushNotificationConfigResponse:
        """List push notification configs - not supported (would need to track all configs)."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> DeleteTaskPushNotificationConfigResponse:
        """Delete push notification config."""
        if not hasattr(self.adapter, 'supports_push_notifications') or not self.adapter.supports_push_notifications():
            raise ServerError(error=UnsupportedOperationError())
        
        success = await self.adapter.delete_push_notification_config(params.taskId)
        if not success:
            raise ServerError(error=UnsupportedOperationError(message=f"No push config for task {params.taskId}"))
        
        return DeleteTaskPushNotificationConfigResponse(result={})


def build_agent_app(
    agent_card: AgentCard,
    adapter: BaseAgentAdapter,
):
    """Build an ASGI application for a single A2A agent.

    .. deprecated:: 0.2.0
        Use :func:`a2a_adapter.server.to_a2a` instead::

            from a2a_adapter import to_a2a
            app = to_a2a(adapter)

    Args:
        agent_card: A2A AgentCard describing the agent's capabilities
        adapter: BaseAgentAdapter implementation for the agent framework

    Returns:
        ASGI application ready to be served
    """
    warnings.warn(
        "build_agent_app() is deprecated, use to_a2a(adapter) instead. "
        "See migration guide: https://github.com/hybro-ai/a2a-adapter/blob/main/docs/migration-v0.2.md",
        DeprecationWarning,
        stacklevel=2,
    )
    handler = AdapterRequestHandler(adapter)

    app_builder = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    # Build and return the actual ASGI application
    return app_builder.build()


def serve_agent(
    agent_card: AgentCard,
    adapter: BaseAgentAdapter,
    host: str = "0.0.0.0",
    port: int = 9000,
    log_level: str = "info",
    **uvicorn_kwargs,
) -> None:
    """
    Start serving a single A2A agent.

    This is a convenience function that builds the agent application and
    starts a uvicorn server to serve it.

    Args:
        agent_card: A2A AgentCard describing the agent's capabilities
        adapter: BaseAgentAdapter implementation for the agent framework
        host: Host address to bind to (default: "0.0.0.0")
        port: Port to listen on (default: 9000)
        log_level: Logging level (default: "info")
        **uvicorn_kwargs: Additional arguments to pass to uvicorn.run()

    Example:
        >>> from a2a.types import AgentCard
        >>> from a2a_adapter import load_a2a_agent, serve_agent
        >>>
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "n8n",
        ...     "webhook_url": "https://n8n.example.com/webhook"
        ... })
        >>> card = AgentCard(name="My Agent", description="...")
        >>> serve_agent(card, adapter, port=9000)
    """
    app = build_agent_app(agent_card, adapter)

    # Use uvicorn.run directly (not inside asyncio.run context)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        **uvicorn_kwargs,
    )

