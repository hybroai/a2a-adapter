# A2A Adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Open Source A2A Protocol Adapter SDK for Different Agent Frameworks**

A Python SDK that enables seamless integration of various agent frameworks (n8n, CrewAI, LangChain, etc.) with the [A2A (Agent-to-Agent) Protocol](https://github.com/a2a-protocol/a2a-protocol). Build interoperable AI agent systems that can communicate across different platforms and frameworks.

## Features

✨ **Framework Agnostic**: Integrate n8n workflows, CrewAI crews, LangChain chains, or custom agents
🔌 **Simple API**: 3-line setup to expose any agent as A2A-compliant
🌊 **Streaming Support**: Built-in streaming for LangChain and custom adapters
🎯 **Type Safe**: Leverages official A2A SDK types
🔧 **Extensible**: Easy to add custom adapters for new frameworks
📦 **Minimal Dependencies**: Optional dependencies per framework

## Architecture

```
┌─────────────────┐
│   A2A Caller    │  (Other A2A Agents)
└────────┬────────┘
         │ A2A Protocol (HTTP + JSON-RPC 2.0)
         ▼
┌─────────────────┐
│  A2A Adapter    │  (This SDK)
│   - N8n         │
│   - CrewAI      │
│   - LangChain   │
│   - Custom      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Your Agent     │  (n8n workflow / CrewAI crew / Chain)
└─────────────────┘
```

**Single-Agent Design**: Each server hosts exactly one agent. Multi-agent orchestration is handled externally via A2A protocol or orchestration frameworks like LangGraph.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Installation

### Basic Installation

```bash
pip install a2a-adapter
```

### With Framework Support

```bash
# For n8n (HTTP webhooks)
pip install a2a-adapter

# For CrewAI
pip install a2a-adapter[crewai]

# For LangChain
pip install a2a-adapter[langchain]

# For LangGraph
pip install a2a-adapter[langgraph]

# Install all frameworks
pip install a2a-adapter[all]

# For development
pip install a2a-adapter[dev]
```

## Quick Start

### 🚀 Easy Start with Examples

For the fastest way to get started, use the included examples:

```bash
# Clone and setup
git clone <repository>
cd a2a-adapter
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Start an agent
./run_agent.sh n8n        # N8n workflow agent
./run_agent.sh crewai     # CrewAI agent
./run_agent.sh langchain  # LangChain agent

# Stop with Ctrl+C
```

**Environment Variables:**

```bash
export N8N_WEBHOOK_URL="https://your-n8n.com/webhook/your-workflow"
```

### 📝 Manual Setup

### 1. N8n Workflow Agent

Expose an n8n workflow as an A2A agent:

```python
import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def main():
    # Load adapter
    adapter = await load_a2a_agent({
        "adapter": "n8n",
        "webhook_url": "https://n8n.example.com/webhook/math",
        "timeout": 30
    })

    # Define agent card
    card = AgentCard(
        name="Math Agent",
        description="Performs mathematical calculations via n8n"
    )

    # Start server
    serve_agent(agent_card=card, adapter=adapter, port=9000)

asyncio.run(main())
```

### 2. CrewAI Agent

Expose a CrewAI crew as an A2A agent:

```python
import asyncio
from crewai import Crew, Agent, Task
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

# Create your crew
crew = Crew(
    agents=[...],
    tasks=[...],
    verbose=True
)

async def main():
    adapter = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew
    })

    card = AgentCard(
        name="Research Crew",
        description="Multi-agent research team"
    )

    serve_agent(agent_card=card, adapter=adapter, port=8001)

asyncio.run(main())
```

### 3. LangChain Agent (with Streaming)

Expose a LangChain chain with streaming support:

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

# Create chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm

async def main():
    adapter = await load_a2a_agent({
        "adapter": "langchain",
        "runnable": chain,
        "input_key": "input"
    })

    card = AgentCard(
        name="Chat Agent",
        description="Streaming chat agent powered by GPT-4"
    )

    serve_agent(agent_card=card, adapter=adapter, port=8002)

asyncio.run(main())
```

### 4. Custom Adapter

Create a custom agent with any async function:

```python
import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def my_agent_function(inputs: dict) -> str:
    """Your custom agent logic."""
    message = inputs["message"]
    return f"Echo: {message}"

async def main():
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": my_agent_function
    })

    card = AgentCard(
        name="Echo Agent",
        description="Simple echo agent"
    )

    serve_agent(agent_card=card, adapter=adapter, port=8003)

asyncio.run(main())
```

## Advanced Usage

### Custom Adapter Class

For full control, subclass `BaseAgentAdapter`:

```python
from a2a_adapter import BaseAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart

class SentimentAnalyzer(BaseAgentAdapter):
    async def to_framework(self, params: MessageSendParams):
        # Extract user message
        text = params.messages[-1].content[0].text
        return {"text": text}

    async def call_framework(self, framework_input, params):
        # Your analysis logic
        sentiment = analyze_sentiment(framework_input["text"])
        return {"sentiment": sentiment}

    async def from_framework(self, framework_output, params):
        # Convert to A2A Message
        return Message(
            role="assistant",
            content=[TextPart(
                type="text",
                text=f"Sentiment: {framework_output['sentiment']}"
            )]
        )

# Use your custom adapter
adapter = SentimentAnalyzer()
serve_agent(agent_card=card, adapter=adapter, port=8004)
```

### Streaming Custom Adapter

Implement `handle_stream()` for streaming responses:

```python
class StreamingAdapter(BaseAgentAdapter):
    async def handle_stream(self, params: MessageSendParams):
        """Yield SSE-compatible events."""
        for chunk in generate_response_chunks():
            yield {
                "event": "message",
                "data": json.dumps({"type": "content", "content": chunk})
            }

        yield {
            "event": "done",
            "data": json.dumps({"status": "completed"})
        }

    def supports_streaming(self):
        return True
```

### Using with LangGraph

Integrate A2A agents into LangGraph workflows:

```python
from langgraph.graph import StateGraph
from a2a.client import A2AClient

# Create A2A client
math_agent = A2AClient(base_url="http://localhost:9000")

# Use in LangGraph node
async def call_math_agent(state):
    response = await math_agent.send_message(
        MessageSendParams(messages=[...])
    )
    return {"result": response}

# Add to graph
graph = StateGraph(...)
graph.add_node("math", call_math_agent)
```

See [examples/06_langgraph_single_agent.py](examples/06_langgraph_single_agent.py) for complete example.

## Configuration

### N8n Adapter

```python
{
    "adapter": "n8n",
    "webhook_url": "https://n8n.example.com/webhook/agent",  # Required
    "timeout": 30,  # Optional, default: 30
    "headers": {    # Optional
        "Authorization": "Bearer token"
    }
}
```

### CrewAI Adapter

```python
{
    "adapter": "crewai",
    "crew": crew_instance,  # Required: CrewAI Crew object
    "inputs_key": "inputs"  # Optional, default: "inputs"
}
```

### LangChain Adapter

```python
{
    "adapter": "langchain",
    "runnable": chain,       # Required: Any Runnable
    "input_key": "input",    # Optional, default: "input"
    "output_key": None       # Optional, extracts specific key from output
}
```

### Callable Adapter

```python
{
    "adapter": "callable",
    "callable": async_function,      # Required: async function
    "supports_streaming": False      # Optional, default: False
}
```

## Examples

The `examples/` directory contains complete working examples:

- **01_single_n8n_agent.py** - N8n workflow agent
- **02_single_crewai_agent.py** - CrewAI multi-agent crew
- **03_single_langchain_agent.py** - LangChain streaming agent
- **04_single_agent_client.py** - A2A client for testing
- **05_custom_adapter.py** - Custom adapter implementations
- **06_langgraph_single_agent.py** - LangGraph + A2A integration

Run any example:

```bash
# Start an agent server
python examples/01_single_n8n_agent.py

# In another terminal, test with client
python examples/04_single_agent_client.py
```

## Testing

```bash
# Install dev dependencies
pip install a2a-adapter[dev]

# Run unit tests
pytest tests/unit/

# Run integration tests (requires framework dependencies)
pytest tests/integration/

# Run all tests
pytest
```

## API Reference

### Core Functions

#### `load_a2a_agent(config: Dict[str, Any]) -> BaseAgentAdapter`

Factory function to create an adapter from configuration.

**Args:**

- `config`: Dictionary with `"adapter"` key and framework-specific options

**Returns:**

- Configured `BaseAgentAdapter` instance

**Raises:**

- `ValueError`: If adapter type is unknown or required config is missing
- `ImportError`: If required framework package is not installed

#### `build_agent_app(agent_card: AgentCard, adapter: BaseAgentAdapter) -> ASGIApp`

Build an ASGI application for serving an A2A agent.

**Args:**

- `agent_card`: A2A AgentCard describing the agent
- `adapter`: Adapter instance

**Returns:**

- ASGI application ready to be served

#### `serve_agent(agent_card, adapter, host="0.0.0.0", port=9000, **kwargs)`

Start serving an A2A agent (convenience function).

**Args:**

- `agent_card`: A2A AgentCard
- `adapter`: Adapter instance
- `host`: Host address (default: "0.0.0.0")
- `port`: Port number (default: 9000)
- `**kwargs`: Additional arguments passed to `uvicorn.run()`

### BaseAgentAdapter

Abstract base class for all adapters.

#### Methods

##### `async def handle(params: MessageSendParams) -> Message | Task`

Handle a non-streaming A2A message request.

##### `async def handle_stream(params: MessageSendParams) -> AsyncIterator[Dict]`

Handle a streaming A2A message request. Override in subclasses that support streaming.

##### `@abstractmethod async def to_framework(params: MessageSendParams) -> Any`

Convert A2A message parameters to framework-specific input.

##### `@abstractmethod async def call_framework(framework_input: Any, params: MessageSendParams) -> Any`

Execute the underlying agent framework.

##### `@abstractmethod async def from_framework(framework_output: Any, params: MessageSendParams) -> Message | Task`

Convert framework output to A2A Message or Task.

##### `def supports_streaming() -> bool`

Check if this adapter supports streaming responses.

## Framework Support

| Framework           | Adapter                 | Streaming   | Status     |
| ------------------- | ----------------------- | ----------- | ---------- |
| **n8n**             | `N8nAgentAdapter`       | ❌          | ✅ Stable  |
| **CrewAI**          | `CrewAIAgentAdapter`    | ❌          | ✅ Stable  |
| **LangChain**       | `LangChainAgentAdapter` | ✅          | ✅ Stable  |
| **Custom Function** | `CallableAgentAdapter`  | ✅ Optional | ✅ Stable  |
| **AutoGen**         | -                       | -           | 🔜 Planned |
| **Semantic Kernel** | -                       | -           | 🔜 Planned |

## Contributing

We welcome contributions! To add support for a new framework:

1. Create `a2a_adapter/integrations/{framework}.py`
2. Implement a class extending `BaseAgentAdapter`
3. Add to `loader.py` factory function
4. Update `integrations/__init__.py`
5. Add optional dependency to `pyproject.toml`
6. Create an example in `examples/`
7. Add tests in `tests/`
8. Update this README

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed guidance.

## Roadmap

- [x] Core adapter abstraction
- [x] N8n adapter
- [x] CrewAI adapter
- [x] LangChain adapter with streaming
- [x] Callable adapter
- [x] Comprehensive examples
- [ ] Task support (async execution pattern)
- [ ] Artifact support (file uploads/downloads)
- [ ] AutoGen adapter
- [ ] Semantic Kernel adapter
- [ ] Haystack adapter
- [ ] Middleware system (logging, metrics, rate limiting)
- [ ] Configuration validation with Pydantic
- [ ] Docker images for quick deployment

## FAQ

### Q: Can I run multiple agents in one process?

**A:** This SDK is designed for single-agent-per-process. For multi-agent systems, run multiple A2A servers and orchestrate them externally using the A2A protocol or tools like LangGraph.

### Q: Does this support the latest A2A protocol version?

**A:** Yes, we use the official A2A SDK which stays up-to-date with protocol changes.

### Q: Can I use this with my custom agent framework?

**A:** Absolutely! Use the `CallableAgentAdapter` for simple cases or subclass `BaseAgentAdapter` for full control.

### Q: What about authentication and rate limiting?

**A:** These concerns are handled at the infrastructure level (reverse proxy, API gateway) or by the official A2A SDK. Adapters focus solely on framework integration.

### Q: How do I debug adapter issues?

**A:** Set `log_level="debug"` in `serve_agent()` and check logs. Each adapter logs framework calls and responses.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

Built with ❤️ by [HYBRO AI](https://hybro.ai)

Powered by the [A2A Protocol](https://github.com/a2a-protocol/a2a-protocol)

## Support

- 📚 [Documentation](https://github.com/hybro-ai/a2a-adapter)
- 🐛 [Issue Tracker](https://github.com/hybro-ai/a2a-adapter/issues)
- 💬 [Discussions](https://github.com/hybro-ai/a2a-adapter/discussions)

---

**Star ⭐ this repo if you find it useful!**
