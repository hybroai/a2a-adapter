# A2A Adapter Examples

Each example demonstrates the **3-line pattern**: `import` → `adapter` → `serve_agent`.

## Prerequisites

```bash
pip install a2a-adapter[all]   # All frameworks
# Or install individually:
pip install a2a-adapter                # Core (n8n, callable)
pip install a2a-adapter[crewai]        # + CrewAI
pip install a2a-adapter[langchain]     # + LangChain
pip install a2a-adapter[langgraph]     # + LangGraph
```

```bash
# For LangChain/LangGraph examples
export OPENAI_API_KEY="your-key"
```

## Examples

| # | File | Framework | Streaming | Port | Description |
|---|------|-----------|-----------|------|-------------|
| 01 | `01_single_n8n_agent.py` | n8n | - | 9000 | n8n webhook → A2A server |
| 02 | `02_single_crewai_agent.py` | CrewAI | - | 8001 | CrewAI crew → A2A server |
| 03 | `03_single_langchain_agent.py` | LangChain | Yes | 8002 | LangChain chain → A2A server (streaming auto-detected) |
| 04 | `04_single_agent_client.py` | httpx | - | - | **Client**: test any A2A agent |
| 05 | `05_custom_adapter.py` | Custom | - | 8003 | Custom adapter (subclass or callable) |
| 07 | `07_langgraph_server.py` | LangGraph | Yes | 9002 | LangGraph workflow → A2A server |
| 08 | `08_openclaw_agent.py` | OpenClaw | - | 9008 | OpenClaw agent → A2A server |
| 09 | `09_v02_quickstart.py` | Mixed | - | 9000 | Quick start: callable, n8n, custom |

## Quick Start

### Simplest possible agent (3 lines)

```python
from a2a_adapter import CallableAdapter, serve_agent

adapter = CallableAdapter(func=lambda inputs: f"Echo: {inputs['message']}", name="Echo")
serve_agent(adapter, port=9000)
```

### n8n

```bash
python examples/01_single_n8n_agent.py
```

### LangChain (with streaming)

```bash
export OPENAI_API_KEY="your-key"
python examples/03_single_langchain_agent.py
```

### Test any running agent

```bash
python examples/04_single_agent_client.py
```

## Testing with curl

```bash
# Fetch agent card
curl http://localhost:9000/.well-known/agent.json

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
