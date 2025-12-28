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
                       'payload_template', 'message_field'
                - crewai: requires 'crew' (CrewAI Crew instance)
                - langchain: requires 'runnable', optional 'input_key', 'output_key'
                - callable: requires 'callable' (async function)
    
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
        )
    
    elif adapter_type == "crewai":
        from .integrations.crewai import CrewAIAgentAdapter
        
        crew = config.get("crew")
        if crew is None:
            raise ValueError("crewai adapter requires 'crew' instance in config")
        
        return CrewAIAgentAdapter(
            crew=crew,
            inputs_key=config.get("inputs_key", "inputs"),
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
    
    elif adapter_type == "callable":
        from .integrations.callable import CallableAgentAdapter
        
        func = config.get("callable")
        if func is None:
            raise ValueError("callable adapter requires 'callable' function in config")
        
        return CallableAgentAdapter(
            func=func,
            supports_streaming=config.get("supports_streaming", False),
        )
    
    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Supported types: n8n, crewai, langchain, callable"
        )

