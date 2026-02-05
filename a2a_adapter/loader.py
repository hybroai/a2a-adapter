"""
Adapter factory for loading framework-specific adapters.

This module provides the load_a2a_agent function which acts as a factory
for creating appropriate adapter instances based on configuration.
"""

from typing import Any, Dict

from .adapter import BaseAgentAdapter


async def load_a2a_agent(config: Dict[str, Any]) -> BaseAgentAdapter:
    """
    Factory function to load an agent adapter based on configuration.

    This function inspects the 'adapter' key in the config dictionary and
    instantiates the appropriate adapter class with the provided configuration.

    Args:
        config: Configuration dictionary with at least an 'adapter' key.
                Additional keys depend on the adapter type:

                - n8n: requires 'webhook_url', optional 'timeout', 'headers',
                       'payload_template', 'message_field', 'async_mode'
                - crewai: requires 'crew' (CrewAI Crew instance),
                          optional 'inputs_key', 'async_mode'
                - langchain: requires 'runnable', optional 'input_key', 'output_key'
                - langgraph: requires 'graph' (CompiledGraph instance),
                             optional 'input_key', 'output_key', 'async_mode'
                - callable: requires 'callable' (async function),
                            optional 'supports_streaming'
                - openclaw: optional 'session_id', 'agent_id', 'thinking',
                            'timeout', 'openclaw_path', 'working_directory',
                            'env_vars', 'async_mode', 'task_store',
                            'task_ttl_seconds', 'cleanup_interval_seconds'

    Returns:
        Configured BaseAgentAdapter instance

    Raises:
        ValueError: If adapter type is unknown or required config is missing
        ImportError: If required framework package is not installed

    Examples:
        >>> # Load n8n adapter (basic)
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "n8n",
        ...     "webhook_url": "https://n8n.example.com/webhook/agent",
        ...     "timeout": 30
        ... })

        >>> # Load n8n adapter with custom payload mapping
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "n8n",
        ...     "webhook_url": "http://localhost:5678/webhook/my-workflow",
        ...     "payload_template": {"name": "A2A Agent"},  # Static fields
        ...     "message_field": "event"  # Use "event" instead of "message"
        ... })

        >>> # Load CrewAI adapter
        >>> from crewai import Crew, Agent, Task
        >>> crew = Crew(agents=[...], tasks=[...])
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "crewai",
        ...     "crew": crew
        ... })

        >>> # Load LangChain adapter
        >>> from langchain_core.runnables import RunnablePassthrough
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "langchain",
        ...     "runnable": chain,
        ...     "input_key": "input"
        ... })

        >>> # Load LangGraph adapter
        >>> from langgraph.graph import StateGraph
        >>> graph = builder.compile()
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "langgraph",
        ...     "graph": graph,
        ...     "input_key": "messages"
        ... })

        >>> # Load callable adapter
        >>> async def my_agent(inputs: dict) -> str:
        ...     return f"Processed: {inputs['message']}"
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "callable",
        ...     "callable": my_agent
        ... })

        >>> # Load OpenClaw adapter
        >>> adapter = await load_a2a_agent({
        ...     "adapter": "openclaw",
        ...     "session_id": "my-session",
        ...     "agent_id": "main",
        ...     "thinking": "low",
        ...     "timeout": 300,
        ... })
    """
    adapter_type = config.get("adapter")

    if not adapter_type:
        raise ValueError("Config must include 'adapter' key specifying adapter type")

    if adapter_type == "n8n":
        from .integrations.n8n import N8nAgentAdapter

        webhook_url = config.get("webhook_url")
        if not webhook_url:
            raise ValueError("n8n adapter requires 'webhook_url' in config")

        return N8nAgentAdapter(
            webhook_url=webhook_url,
            timeout=config.get("timeout", 30),
            headers=config.get("headers"),
            payload_template=config.get("payload_template"),
            message_field=config.get("message_field", "message"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 300),
        )

    elif adapter_type == "crewai":
        from .integrations.crewai import CrewAIAgentAdapter

        crew = config.get("crew")
        if crew is None:
            raise ValueError("crewai adapter requires 'crew' instance in config")

        return CrewAIAgentAdapter(
            crew=crew,
            inputs_key=config.get("inputs_key", "inputs"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 600),
        )

    elif adapter_type == "langchain":
        from .integrations.langchain import LangChainAgentAdapter

        runnable = config.get("runnable")
        if runnable is None:
            raise ValueError("langchain adapter requires 'runnable' in config")

        return LangChainAgentAdapter(
            runnable=runnable,
            input_key=config.get("input_key", "input"),
            output_key=config.get("output_key"),
        )

    elif adapter_type == "langgraph":
        from .integrations.langgraph import LangGraphAgentAdapter

        graph = config.get("graph")
        if graph is None:
            raise ValueError("langgraph adapter requires 'graph' (CompiledGraph) in config")

        return LangGraphAgentAdapter(
            graph=graph,
            input_key=config.get("input_key", "messages"),
            output_key=config.get("output_key"),
            state_key=config.get("state_key"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 300),
        )

    elif adapter_type == "callable":
        from .integrations.callable import CallableAgentAdapter

        func = config.get("callable")
        if func is None:
            raise ValueError("callable adapter requires 'callable' function in config")

        return CallableAgentAdapter(
            func=func,
            supports_streaming=config.get("supports_streaming", False),
        )

    elif adapter_type == "openclaw":
        from .integrations.openclaw import OpenClawAgentAdapter

        return OpenClawAgentAdapter(
            session_id=config.get("session_id"),
            agent_id=config.get("agent_id"),
            thinking=config.get("thinking", "low"),
            timeout=config.get("timeout", 600),
            openclaw_path=config.get("openclaw_path", "openclaw"),
            working_directory=config.get("working_directory"),
            env_vars=config.get("env_vars"),
            async_mode=config.get("async_mode", True),
            task_store=config.get("task_store"),
            task_ttl_seconds=config.get("task_ttl_seconds", 3600),
            cleanup_interval_seconds=config.get("cleanup_interval_seconds", 300),
        )

    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Supported types: n8n, crewai, langchain, langgraph, callable, openclaw"
        )
