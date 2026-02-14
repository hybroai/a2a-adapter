# Quick Start Guide

Get your first A2A agent running in under 5 minutes.

This guide will help you expose your existing agent (n8n workflow, CrewAI crew, LangChain chain, LangGraph workflow, or any function) as an A2A-compatible agent.

## Prerequisites

- Python 3.11+
- An agent to expose (n8n workflow, CrewAI crew, LangChain chain, or custom function)

## Step 1: Install

```bash
pip install a2a-adapter
```

For specific frameworks:

```bash
pip install a2a-adapter[crewai]        # CrewAI
pip install a2a-adapter[langchain]     # LangChain
pip install a2a-adapter[langgraph]     # LangGraph
pip install a2a-adapter[all]           # Everything
```

## Step 2: Create Your Agent

Choose your framework — every example follows the same 3-line pattern: **import**, **adapter**, **serve**.

### Option A: n8n Workflow

```python
# my_agent.py
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(
    webhook_url="https://your-n8n.com/webhook/workflow-id",
    name="My N8n Agent",
    description="My n8n workflow as an A2A agent",
)
serve_agent(adapter, port=9000)
```

### Option B: LangChain Chain (with streaming)

```python
# my_agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from a2a_adapter import LangChainAdapter, serve_agent

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ])
    | ChatOpenAI(model="gpt-4o-mini", streaming=True)
)

adapter = LangChainAdapter(runnable=chain, input_key="input", name="Chat Agent")
serve_agent(adapter, port=9000)  # Streaming auto-detected!
```

### Option C: LangGraph Workflow (with streaming)

```python
# my_agent.py
from a2a_adapter import LangGraphAdapter, serve_agent

graph = build_my_graph().compile()  # Your LangGraph CompiledGraph

adapter = LangGraphAdapter(
    graph=graph,
    name="Research Agent",
    description="LangGraph research workflow as an A2A agent",
)
serve_agent(adapter, port=9000)
```

### Option D: CrewAI Crew

```python
# my_agent.py
from crewai import Agent, Crew, Process
from a2a_adapter import CrewAIAdapter, serve_agent

crew = Crew(agents=[...], tasks=[...], process=Process.sequential)

adapter = CrewAIAdapter(crew=crew, name="Research Crew", timeout=600)
serve_agent(adapter, port=9000)
```

### Option E: OpenClaw Agent

```python
# my_agent.py
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(thinking="low", name="OpenClaw Agent")
serve_agent(adapter, port=9000)
```

### Option F: Custom Function

```python
# my_agent.py
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs['message']}"

adapter = CallableAdapter(func=my_agent, name="Echo Agent")
serve_agent(adapter, port=9000)
```

### Option G: Custom Adapter Class

For full control, subclass `BaseA2AAdapter`:

```python
# my_agent.py
from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

class MyAdapter(BaseA2AAdapter):
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        return f"You said: {user_input}"

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(name="My Agent", description="Custom A2A agent")

serve_agent(MyAdapter(), port=9000)
```

## Step 3: Run Your Agent

```bash
python my_agent.py
```

Your agent is now running at `http://localhost:9000`.

The A2A SDK automatically:
- Generates an **AgentCard** from your adapter metadata
- Serves it at `/.well-known/agent.json`
- Handles **task management**, **JSON-RPC 2.0**, and **SSE streaming**

## Step 4: Test Your Agent

### Using curl

```bash
# Fetch the auto-generated agent card
curl http://localhost:9000/.well-known/agent.json

# Send a message via JSON-RPC 2.0
curl -X POST http://localhost:9000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "messageId": "msg-1",
        "parts": [{"kind": "text", "text": "Hello!"}]
      }
    }
  }'
```

### Using the example client

```bash
python examples/single_agent_client.py
```

### Using httpx (Python)

```python
import asyncio, httpx

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        # Fetch agent card
        card = (await client.get("http://localhost:9000/.well-known/agent.json")).json()
        print(f"Agent: {card['name']}")

        # Send a message
        resp = await client.post("http://localhost:9000", json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "msg-1",
                    "parts": [{"kind": "text", "text": "Hello!"}],
                }
            },
        })
        print(resp.json())

asyncio.run(main())
```

## What's Next?

### Supported Frameworks

| Framework | Adapter | Streaming |
|---|---|---|
| n8n | `N8nAdapter` | - |
| LangChain | `LangChainAdapter` | Auto-detected |
| LangGraph | `LangGraphAdapter` | Auto-detected |
| CrewAI | `CrewAIAdapter` | - |
| OpenClaw | `OpenClawAdapter` | - |
| Any function | `CallableAdapter` | Optional |
| Custom class | `BaseA2AAdapter` | Optional |

### Advanced Usage

#### ASGI Deployment (Gunicorn / Hypercorn)

```python
from a2a_adapter import N8nAdapter, to_a2a

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
app = to_a2a(adapter)  # Returns Starlette ASGI app

# Deploy: gunicorn app:app -k uvicorn.workers.UvicornWorker
```

#### Register a Third-party Adapter

```python
from a2a_adapter import register_adapter, BaseA2AAdapter

@register_adapter("my_framework")
class MyFrameworkAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "Hello from my framework!"
```

### Next Steps

1. **Explore examples** — See [examples/](examples/) for complete working code
2. **Read the API** — See [README.md](README.md) for full API reference
3. **Understand the design** — See [ARCHITECTURE.md](ARCHITECTURE.md) for the layered architecture
4. **Build multi-agent systems** — Connect multiple A2A agents together
5. **Create custom adapters** — Integrate your own frameworks

## Troubleshooting

**Import errors:**

```bash
pip install a2a-adapter[langchain]  # Or [crewai], [langgraph], [all]
```

**Port already in use:**

```bash
lsof -i :9000           # Find the process
kill <PID>              # Kill it
# Or use a different port:
serve_agent(adapter, port=8001)
```

**Missing API keys (LangChain / CrewAI):**

```bash
export OPENAI_API_KEY="sk-..."
```

**Need more help?** See [GETTING_STARTED_DEBUG.md](GETTING_STARTED_DEBUG.md) for detailed debugging.

## Additional Resources

- [Full Documentation](README.md) — Complete API reference
- [Architecture Guide](ARCHITECTURE.md) — Design and implementation details
- [Examples](examples/) — Complete working examples
- [Debug Guide](GETTING_STARTED_DEBUG.md) — Troubleshooting and debugging
- [Contributing](CONTRIBUTING.md) — How to contribute
