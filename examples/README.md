# A2A Adapter Examples

These are **Python SDK examples**. Each demonstrates the 3-line pattern:
`import` -> `adapter` -> `serve_agent`.

For the no-code CLI use case, run one of the supported local agent commands
directly from the target project directory:

```bash
a2a-adapter pi --port 9012
a2a-adapter codex --port 9011
a2a-adapter antigravity --port 9014
a2a-adapter claude --port 9010
a2a-adapter openclaw --port 9008
```

Use `a2a-adapter -help` for CLI usage. Hermes remains SDK-only in v1.

## Prerequisites

```bash
pip install -r examples/requirements.txt  # SDK, all framework extras, example-only deps
# Or install only what you need:
pip install a2a-adapter                # Core (n8n, callable)
pip install a2a-adapter[crewai]        # + CrewAI
pip install a2a-adapter[langchain]     # + LangChain
pip install a2a-adapter[langgraph]     # + LangGraph
```

```bash
# For LangChain/LangGraph/CrewAI/Codex examples
export OPENAI_API_KEY="your-key"

# For Claude Code examples
# npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY="your-key"

# For Codex examples
# npm install -g @openai/codex

# For Antigravity CLI examples: install agy and run it once to sign in

# For Pi examples
# Install Pi, or export A2A_PI_COMMAND="npx tsx /path/to/pi/packages/coding-agent/src/cli.ts"

# For Hermes examples — clone https://github.com/NousResearch/hermes-agent
# export PYTHONPATH=/path/to/hermes-agent:$PYTHONPATH
# Run `hermes setup` once for ~/.hermes/config.yaml and API keys.
```

## Examples

| File | Framework | Streaming | Port | Description |
|------|-----------|-----------|------|-------------|
| `n8n_agent.py` | n8n | - | 9000 | n8n webhook -> A2A server |
| `crewai_agent.py` | CrewAI | - | 8001 | CrewAI crew -> A2A server |
| `langchain_agent.py` | LangChain | Yes | 8002 | LangChain chain -> A2A server (streaming auto-detected) |
| `langgraph_server.py` | LangGraph | Yes | 9002 | LangGraph workflow -> A2A server |
| `openclaw_agent.py` | OpenClaw | - | 9008 | OpenClaw agent -> A2A server |
| `ollama_agent.py` | Ollama | Yes | 10010 | Local Ollama LLM -> A2A server (streaming) |
| `hermes_agent.py` | Hermes | Yes | 9010 | Hermes AI agent -> A2A server (streaming, multi-turn) |
| `claude_code_agent.py` | Claude Code | Yes | 9010 | Claude Code CLI -> A2A server (streaming, multi-turn) |
| `codex_agent.py` | Codex | - | 9011 | Codex CLI -> A2A server (multi-turn) |
| `antigravity_agent.py` | Antigravity CLI | Yes | 9014 | Antigravity CLI (agy) -> A2A server (streaming, multi-turn) |
| `pi_agent.py` | Pi | Yes | 9012 | One persistent Pi RPC session -> A2A server |
| `custom_adapter.py` | Custom | - | 8003 | Custom BaseA2AAdapter (sentiment analyzer) |
| `single_agent_client.py` | httpx | - | - | **Client**: test any A2A agent |
| `quickstart.py` | Mixed | - | 9000 | Quick start: callable, n8n, custom |

## Python SDK Quick Start

### Simplest possible agent (3 lines)

```python
from a2a_adapter import CallableAdapter, serve_agent

adapter = CallableAdapter(func=lambda inputs: f"Echo: {inputs['message']}", name="Echo")
serve_agent(adapter, port=9000)
```

### Run any example

```bash
python examples/n8n_agent.py            # n8n
python examples/langchain_agent.py      # LangChain (streaming)
python examples/langgraph_server.py     # LangGraph (streaming)
python examples/crewai_agent.py         # CrewAI
python examples/openclaw_agent.py       # OpenClaw
python examples/ollama_agent.py        # Ollama (local LLM, streaming)
python examples/hermes_agent.py        # Hermes (tool use, multi-turn, streaming)
python examples/claude_code_agent.py   # Claude Code (streaming, multi-turn)
python examples/codex_agent.py         # Codex (multi-turn)
python examples/antigravity_agent.py   # Antigravity CLI (streaming, multi-turn)
python examples/pi_agent.py            # Pi (persistent session, streaming)
python examples/custom_adapter.py       # Custom adapter (sentiment analyzer)
```

### Test any running agent

```bash
python examples/single_agent_client.py
```

## Testing with curl

```bash
# Fetch agent card
curl http://localhost:9000/.well-known/agent-card.json

# Send a message
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
        "parts": [{"kind": "text", "text": "What is 2+2?"}]
      }
    }
  }'
```

## Common Issues

| Issue | Fix |
|---|---|
| Port in use | `lsof -i :9000` then `kill <PID>`, or change port |
| Import error | `pip install a2a-adapter[crewai]` (or `[langchain]`, `[all]`) |
| No OPENAI_API_KEY | `export OPENAI_API_KEY="sk-..."` |
