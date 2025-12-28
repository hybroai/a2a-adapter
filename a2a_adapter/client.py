"""
Single-agent A2A server helpers.

This module provides utilities for creating and serving A2A-compliant
agent servers using the official A2A.
"""

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

    # Task-related methods (not supported by default)
    
    async def on_get_task(
        self,
        params: GetTaskRequest,
        context: ServerCallContext
    ) -> GetTaskResponse:
        """Get task status - not supported."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_cancel_task(
        self,
        params: CancelTaskRequest,
        context: ServerCallContext
    ) -> CancelTaskResponse:
        """Cancel task - not supported."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_resubscribe_to_task(
        self,
        params: TaskResubscriptionRequest,
        context: ServerCallContext
    ) -> AsyncGenerator[TaskStatusUpdateEvent, None]:
        """Resubscribe to task - not supported."""
        raise ServerError(error=UnsupportedOperationError())
        yield  # Make this an async generator

    # Push notification methods (not supported by default)
    
    async def on_set_task_push_notification_config(
        self,
        params: SetTaskPushNotificationConfigRequest,
        context: ServerCallContext
    ) -> SetTaskPushNotificationConfigResponse:
        """Set push notification config - not supported."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_get_task_push_notification_config(
        self,
        params: GetTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> GetTaskPushNotificationConfigResponse:
        """Get push notification config - not supported."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> ListTaskPushNotificationConfigResponse:
        """List push notification configs - not supported."""
        raise ServerError(error=UnsupportedOperationError())

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext
    ) -> DeleteTaskPushNotificationConfigResponse:
        """Delete push notification config - not supported."""
        raise ServerError(error=UnsupportedOperationError())


def build_agent_app(
    agent_card: AgentCard,
    adapter: BaseAgentAdapter,
):
    """
    Build an ASGI application for a single A2A agent.
    
    This function creates a complete A2A-compliant server application using
    the official A2A SDK, configured with the provided agent card and adapter.
    
    Args:
        agent_card: A2A AgentCard describing the agent's capabilities
        adapter: BaseAgentAdapter implementation for the agent framework
        
    Returns:
        ASGI application ready to be served
        
    Example:
        >>> from a2a.types import AgentCard
        >>> from a2a_adapter.integrations.n8n import N8nAgentAdapter
        >>> 
        >>> card = AgentCard(
        ...     name="Math Agent",
        ...     description="Performs mathematical operations"
        ... )
        >>> adapter = N8nAgentAdapter(webhook_url="https://n8n.example.com/webhook")
        >>> app = build_agent_app(card, adapter)
    """
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

