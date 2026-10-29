# Quick Start Guide

Get your first A2A agent running in under 5 minutes.

`a2a-adapter` can be used in two ways. Pick the one that matches what you have:

| You have... | Use | Go to |
|---|---|---|
| Your own agent code, or a LangChain / LangGraph / CrewAI / n8n agent | the **Python SDK** | [Path 1](#path-1-make-your-own-agent-a2a-compatible-sdk) |
| A local agent you already run in a terminal: Claude Code, Codex, Pi, or OpenClaw | the **CLI** | [Path 2](#path-2-start-a-local-agent-with-one-command-cli) |

Both need Python 3.11+ and produce the same kind of A2A server.

## Path 1: Make Your Own Agent A2A-Compatible (SDK)

Use this path when you're writing agent code, or wrapping a framework such as
LangChain, LangGraph, CrewAI, or n8n.

### Step 1: Install into your project

Install into your project's virtual environment, adding the extra for your
framework:

```bash
pip install a2a-adapter
pip install "a2a-adapter[crewai]"     # CrewAI
pip install "a2a-adapter[langchain]"  # LangChain
pip install "a2a-adapter[langgraph]"  # LangGraph
pip install "a2a-adapter[all]"        # Everything
```

### Step 2: Wrap your agent

Choose your framework — every example follows the same 3-line pattern: **import**, **adapter**, **serve**.

#### Option A: n8n Workflow

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

#### Option B: LangChain Chain (with streaming)

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

#### Option C: LangGraph Workflow (with streaming)

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

#### Option D: CrewAI Crew

```python
# my_agent.py
from crewai import Agent, Crew, Process
from a2a_adapter import CrewAIAdapter, serve_agent

crew = Crew(agents=[...], tasks=[...], process=Process.sequential)

adapter = CrewAIAdapter(crew=crew, name="Research Crew", timeout=600)
serve_agent(adapter, port=9000)
```

#### Option E: OpenClaw Agent

```python
# my_agent.py
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(thinking="low", name="OpenClaw Agent")
serve_agent(adapter, port=9000)
```

#### Option F: Ollama (Local LLM)

```python
# my_agent.py
from a2a_adapter import OllamaAdapter, serve_agent
from a2a_adapter.integrations.ollama import OllamaClient

client = OllamaClient(model="llama3.2")
adapter = OllamaAdapter(client=client, name="Local LLM Agent")
serve_agent(adapter, port=9000)
```

#### Option G: Custom Function

```python
# my_agent.py
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs['message']}"

adapter = CallableAdapter(func=my_agent, name="Echo Agent")
serve_agent(adapter, port=9000)
```

#### Option H: Custom Adapter Class

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

### Step 3: Run it

```bash
python my_agent.py
```

Your agent is now running at `http://localhost:9000`.

The A2A SDK automatically:
- Generates an **AgentCard** from your adapter metadata
- Serves it at `/.well-known/agent-card.json` (with the legacy
  `/.well-known/agent.json` alias)
- Handles **task management**, **JSON-RPC 2.0**, and **SSE streaming**

## Path 2: Start a Local Agent with One Command (CLI)

Use this path to put a classic local agent (Claude Code, Codex, Pi, or OpenClaw)
on A2A without writing code.

### Step 1: Install the CLI

Install `a2a-adapter` as an isolated tool so the command lands on your `PATH`:

```bash
uv tool install a2a-adapter    # or: pipx install a2a-adapter
```

Don't have uv? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
uv fetches a suitable Python 3.11+ automatically. Upgrade later with
`uv tool upgrade a2a-adapter`.

### Step 2: Make sure the agent works on its own

a2a-adapter wraps the agent you already have; it doesn't install it. For
example, install Claude Code and run `claude` once to sign in, or install Pi and
configure its model provider. If the agent isn't on `PATH`, pass
`--executable /path/to/agent` (for Pi, you can also set `A2A_PI_COMMAND`).

### Step 3: Start it

Run the command from the project directory the agent should work in:

```bash
a2a-adapter claude --port 9010
a2a-adapter pi --port 9012
a2a-adapter codex --port 9011
a2a-adapter openclaw --port 9008
```

Use `--cwd /path/to/project` to select another directory. A2A server options
go before `--`; native agent options go after it:

```bash
a2a-adapter claude --port 9010 -- --model <model>
a2a-adapter pi --port 9012 -- --model <model> --thinking high
a2a-adapter codex --port 9011 -- --model <model> --sandbox workspace-write
a2a-adapter openclaw --port 9008 --agent-id main --thinking high
```

You don't need the agent's interactive UI to configure it. The adapter runs the
agent non-interactively, as you, in the `--cwd` folder, so it uses the agent's
normal config files (for example `~/.claude/settings.json` and `CLAUDE.md`, or
`~/.codex/config.toml` and `AGENTS.md`). Edit those for lasting settings, and
use flags after `--` for settings that apply only to this server. Changes apply
to the next request, except for Pi, which needs a server restart.

The server binds to `127.0.0.1:9000` unless you pass `--host`/`--port`. Its
Agent Card is named after the agent and the folder it works in, such as
`Claude Code (my-app)`, so several local agents stay distinguishable in an A2A
client. Override it with `--name` and `--description`. The card lists one
skill built from that name and description; describe what the agent offers more
precisely with one or more `--skill` options:

```bash
a2a-adapter claude --cwd ./reviewer --name "Security reviewer" \
  --skill '{"id": "security-review", "name": "Security review", "description": "Reviews a diff for security issues", "tags": ["code-review"]}'
```

Check that it's up by opening the Agent Card, for example
`http://127.0.0.1:9010/.well-known/agent-card.json`.

Run `a2a-adapter -help` to list supported agents, or
`a2a-adapter <agent> -help` for agent-specific options. Hermes is available as
a Python SDK adapter but is not supported by the CLI.

## Test Either Kind of Agent

The examples below use port `9000`; replace it with the port you started the
CLI on (for example `9010`).

### Using curl

```bash
# Fetch the auto-generated agent card
curl http://localhost:9000/.well-known/agent-card.json

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
        card = (await client.get("http://localhost:9000/.well-known/agent-card.json")).json()
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

### Supported Python SDK Adapters

| Framework | Adapter | Streaming |
|---|---|---|
| n8n | `N8nAdapter` | - |
| LangChain | `LangChainAdapter` | Auto-detected |
| LangGraph | `LangGraphAdapter` | Auto-detected |
| CrewAI | `CrewAIAdapter` | - |
| Claude Code | `ClaudeCodeAdapter` | Yes |
| Codex | `CodexAdapter` | - |
| OpenClaw | `OpenClawAdapter` | - |
| Ollama | `OllamaAdapter` | Yes |
| Pi | `PiAdapter` | Yes |
| Hermes | `HermesAdapter` | Yes |
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

**`a2a-adapter: command not found` after installing:** the package went into
an environment whose scripts aren't on your `PATH`. Install it as a tool with
`uv tool install a2a-adapter` (run `uv tool update-shell` if `~/.local/bin`
isn't on `PATH`).

**`No matching distribution found for a2a-adapter`:** your `pip` runs Python
older than 3.11 (macOS's built-in `pip` is Python 3.9). Use `uv tool install`,
which fetches a suitable Python for you.

**`error: externally-managed-environment`:** Homebrew or your Linux
distribution blocks system-wide `pip install`. Use `uv tool install` for the
CLI, or a virtual environment for the SDK.

**Import errors (SDK):**

```bash
pip install "a2a-adapter[langchain]"  # Or [crewai], [langgraph], [all]
```

**Port already in use:**

```bash
lsof -i :9000           # Find the process
kill <PID>              # Kill it
# Or use a different port:
serve_agent(adapter, port=8001)   # SDK
a2a-adapter claude --port 8001   # CLI
```

**Missing API keys (LangChain / CrewAI):**

```bash
export OPENAI_API_KEY="sk-..."
```

For development setup, test commands, and debugging guidance, see
[CONTRIBUTING.md](CONTRIBUTING.md#development-setup).

## Additional Resources

- [Full Documentation](README.md) — Complete API reference
- [Architecture Guide](ARCHITECTURE.md) — Design and implementation details
- [Examples](examples/) — Complete working examples
- [Contributing](CONTRIBUTING.md) — Development setup and contribution guide
