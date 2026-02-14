"""
Adapter factory and registry for loading adapters from configuration.

This module provides two mechanisms for creating adapters:

1. **load_adapter(config)** — Factory function for config-driven deployments.
   Looks up adapters by name from the built-in map and the user registry.

2. **register_adapter(name)** — Decorator for third-party adapters to register
   themselves, enabling discovery via load_adapter().

Design rationale:
    For config-driven deployments (YAML/JSON configs, orchestration systems),
    the loader creates adapters from dictionaries. The registry pattern allows
    third-party adapters to register themselves without modifying core code.

Backwards compatibility:
    load_a2a_agent() is preserved as a deprecated async wrapper around
    load_adapter() for v0.1 users.
"""

import importlib
import logging
import warnings
from typing import Any, Dict, Type

from .base_adapter import BaseA2AAdapter

logger = logging.getLogger(__name__)

# ──── Registry ────

_REGISTRY: Dict[str, Type[BaseA2AAdapter]] = {}

_BUILTIN_MAP: Dict[str, tuple[str, str]] = {
    "n8n": ("a2a_adapter.integrations.n8n", "N8nAdapter"),
    "crewai": ("a2a_adapter.integrations.crewai", "CrewAIAdapter"),
    "langchain": ("a2a_adapter.integrations.langchain", "LangChainAdapter"),
    "langgraph": ("a2a_adapter.integrations.langgraph", "LangGraphAdapter"),
    "callable": ("a2a_adapter.integrations.callable", "CallableAdapter"),
    "openclaw": ("a2a_adapter.integrations.openclaw", "OpenClawAdapter"),
}


def register_adapter(name: str):
    """Decorator for third-party adapters to register themselves.

    Registered adapters become available via load_adapter() by name.

    Args:
        name: Unique adapter name for config-driven lookup.

    Returns:
        A class decorator that registers the adapter.

    Example:
        >>> from a2a_adapter import register_adapter, BaseA2AAdapter
        >>>
        >>> @register_adapter("my_framework")
        ... class MyFrameworkAdapter(BaseA2AAdapter):
        ...     async def invoke(self, user_input, context_id=None):
        ...         return "Hello from my framework!"
        >>>
        >>> # Now loadable via config:
        >>> adapter = load_adapter({"adapter": "my_framework"})
    """

    def decorator(cls: Type[BaseA2AAdapter]):
        if name in _REGISTRY:
            logger.warning(
                "Overwriting registered adapter %r (was %s, now %s)",
                name,
                _REGISTRY[name].__name__,
                cls.__name__,
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def load_adapter(config: dict) -> BaseA2AAdapter:
    """Factory: create an adapter from a configuration dictionary.

    Looks up the adapter class by the "adapter" key in config, first
    checking the user registry, then the built-in map. Remaining config
    keys are passed as constructor kwargs.

    Args:
        config: Configuration dictionary. Must include an "adapter" key.
            All other keys are passed to the adapter constructor.

    Returns:
        Configured BaseA2AAdapter instance.

    Raises:
        ValueError: If "adapter" key is missing or adapter type is unknown.
        ImportError: If required framework package is not installed.

    Example:
        >>> adapter = load_adapter({
        ...     "adapter": "n8n",
        ...     "webhook_url": "http://localhost:5678/webhook/my-agent",
        ...     "timeout": 60,
        ... })
    """
    config = dict(config)  # Shallow copy to avoid mutating caller's dict
    adapter_type = config.pop("adapter", None)

    if not adapter_type:
        raise ValueError(
            "Config must include 'adapter' key specifying the adapter type"
        )

    # Priority 1: User-registered adapters
    cls = _REGISTRY.get(adapter_type)

    # Priority 2: Built-in adapters (lazy import)
    if cls is None and adapter_type in _BUILTIN_MAP:
        module_path, class_name = _BUILTIN_MAP[adapter_type]
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Adapter {adapter_type!r} requires additional dependencies. "
                f"Install with: pip install a2a-adapter[{adapter_type}]"
            ) from e

    if cls is None:
        registered = sorted(set(_REGISTRY) | set(_BUILTIN_MAP))
        raise ValueError(
            f"Unknown adapter type: {adapter_type!r}. "
            f"Available: {', '.join(registered)}"
        )

    return cls(**config)


# ──── v0.1 Backwards Compatibility ────

# Map v0.1 adapter class names to v0.2 adapter class names
# These are used by load_a2a_agent() for transparent migration
_V1_BUILTIN_MAP: Dict[str, tuple[str, str]] = {
    "n8n": ("a2a_adapter.integrations.n8n", "N8nAgentAdapter"),
    "crewai": ("a2a_adapter.integrations.crewai", "CrewAIAgentAdapter"),
    "langchain": ("a2a_adapter.integrations.langchain", "LangChainAgentAdapter"),
    "langgraph": ("a2a_adapter.integrations.langgraph", "LangGraphAgentAdapter"),
    "callable": ("a2a_adapter.integrations.callable", "CallableAgentAdapter"),
    "openclaw": ("a2a_adapter.integrations.openclaw", "OpenClawAgentAdapter"),
}


async def load_a2a_agent(config: Dict[str, Any]) -> Any:
    """Factory function to load an agent adapter based on configuration.

    .. deprecated:: 0.2.0
        Use :func:`load_adapter` instead. This async wrapper exists only
        for backwards compatibility with v0.1 code.

    Args:
        config: Configuration dictionary with at least an 'adapter' key.

    Returns:
        Configured adapter instance (v0.1 BaseAgentAdapter).
    """
    warnings.warn(
        "load_a2a_agent() is deprecated, use load_adapter() instead. "
        "See migration guide: https://github.com/hybro-ai/a2a-adapter/blob/main/docs/migration-v0.2.md",
        DeprecationWarning,
        stacklevel=2,
    )

    config = dict(config)
    adapter_type = config.get("adapter")

    if not adapter_type:
        raise ValueError("Config must include 'adapter' key specifying adapter type")

    if adapter_type not in _V1_BUILTIN_MAP:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Supported types: {', '.join(sorted(_V1_BUILTIN_MAP))}"
        )

    module_path, class_name = _V1_BUILTIN_MAP[adapter_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Remove 'adapter' key before passing to constructor
    config.pop("adapter")

    # v0.1 adapters have different constructor signatures and
    # validation, so we need to handle them individually
    if adapter_type == "n8n":
        if not config.get("webhook_url"):
            raise ValueError("n8n adapter requires 'webhook_url' in config")
        return cls(
            webhook_url=config.get("webhook_url"),
            timeout=config.get("timeout", 30),
            headers=config.get("headers"),
            payload_template=config.get("payload_template"),
            message_field=config.get("message_field", "message"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 300),
            parse_json_input=config.get("parse_json_input", True),
            input_mapper=config.get("input_mapper"),
            default_inputs=config.get("default_inputs"),
        )
    elif adapter_type == "crewai":
        if config.get("crew") is None:
            raise ValueError("crewai adapter requires 'crew' instance in config")
        return cls(
            crew=config.get("crew"),
            inputs_key=config.get("inputs_key", "inputs"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 600),
            timeout=config.get("timeout", 300),
            parse_json_input=config.get("parse_json_input", True),
            input_mapper=config.get("input_mapper"),
            default_inputs=config.get("default_inputs"),
        )
    elif adapter_type == "langchain":
        if config.get("runnable") is None:
            raise ValueError("langchain adapter requires 'runnable' in config")
        return cls(
            runnable=config.get("runnable"),
            input_key=config.get("input_key", "input"),
            output_key=config.get("output_key"),
            timeout=config.get("timeout", 60),
            parse_json_input=config.get("parse_json_input", True),
            input_mapper=config.get("input_mapper"),
            default_inputs=config.get("default_inputs"),
        )
    elif adapter_type == "langgraph":
        if config.get("graph") is None:
            raise ValueError("langgraph adapter requires 'graph' (CompiledGraph) in config")
        return cls(
            graph=config.get("graph"),
            input_key=config.get("input_key", "messages"),
            output_key=config.get("output_key"),
            state_key=config.get("state_key"),
            async_mode=config.get("async_mode", False),
            task_store=config.get("task_store"),
            async_timeout=config.get("async_timeout", 300),
            timeout=config.get("timeout", 60),
            parse_json_input=config.get("parse_json_input", True),
            input_mapper=config.get("input_mapper"),
            default_inputs=config.get("default_inputs"),
        )
    elif adapter_type == "callable":
        if config.get("callable") is None:
            raise ValueError("callable adapter requires 'callable' function in config")
        return cls(
            func=config.get("callable"),
            supports_streaming=config.get("supports_streaming", False),
        )
    elif adapter_type == "openclaw":
        return cls(
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
        raise ValueError(f"Unknown adapter type: {adapter_type}")
