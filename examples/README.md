# A2A Adapter Examples

This directory contains complete, runnable examples demonstrating how to use the A2A Adapter SDK with different agent frameworks.

## Prerequisites

Install the required dependencies:

```bash
# Install base package
pip install a2a-adapter

# Install framework-specific dependencies as needed
pip install a2a-adapter[crewai]    # For CrewAI examples
pip install a2a-adapter[langchain]  # For LangChain examples
pip install a2a-adapter[langgraph]  # For LangGraph examples

# Or install all at once
pip install -r requirements.txt
```

Set up environment variables:

```bash
# For LangChain and LangGraph examples
export OPENAI_API_KEY="your-openai-api-key"

# For n8n example
export N8N_WEBHOOK_URL="https://your-n8n-instance.com/webhook/agent"
```

## Examples Overview

| File | Description | Port | Framework | Streaming |
|------|-------------|------|-----------|-----------|
| `01_single_n8n_agent.py` | n8n workflow agent | 9000 | n8n | ❌ |
| `02_single_crewai_agent.py` | CrewAI multi-agent crew | 8001 | CrewAI | ❌ |
| `03_single_langchain_agent.py` | LangChain chat agent | 8002 | LangChain | ✅ |
| `04_single_agent_client.py` | A2A client for testing | - | A2A SDK | - |
| `05_custom_adapter.py` | Custom adapter examples | 8003 | Custom | ❌ |
| `06_langgraph_single_agent.py` | Call A2A agents from LangGraph | - | LangGraph | - |
| `07_langgraph_server.py` | LangGraph workflow as A2A server | 9002 | LangGraph | ✅ |

## Running Examples

### Example 1: N8n Workflow Agent

Expose an n8n workflow as an A2A agent:

```bash
# Set your n8n webhook URL
export N8N_WEBHOOK_URL="https://n8n.example.com/webhook/math"

# Start the agent server
python examples/01_single_n8n_agent.py
```

The agent will be available at `http://localhost:9000`.

**n8n Workflow Setup:**

Your n8n workflow should:
1. Start with a Webhook trigger
2. Accept POST requests with JSON body: `{"message": "...", "metadata": {...}}`
3. Process the message
4. Return JSON response with `output`, `result`, or `message` field

### Example 2: CrewAI Agent

Run a multi-agent research crew:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key"

# Start the agent server
python examples/02_single_crewai_agent.py
```

The research crew will be available at `http://localhost:8001`.

### Example 3: LangChain Agent (Streaming)

Run a streaming chat agent powered by GPT-4:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key"

# Start the agent server
python examples/03_single_langchain_agent.py
```

The chat agent will be available at `http://localhost:8002` with streaming support.

### Example 4: Testing with A2A Client

Test any running agent server:

```bash
# Make sure an agent is running (e.g., example 01 on port 9000)
python examples/01_single_n8n_agent.py &

# In another terminal, run the client
python examples/04_single_agent_client.py
```

Edit the client script to:
- Change the target URL
- Modify the test message
- Enable streaming mode

### Example 5: Custom Adapter

Create custom adapter using two different approaches:

```bash
python examples/05_custom_adapter.py
```

Choose between:
1. **Subclassing BaseAgentAdapter** - Full control
2. **Using CallableAgentAdapter** - Simpler approach

### Example 6: Call A2A Agents from LangGraph

Use A2A agents within a LangGraph workflow:

```bash
# Start an agent server first
python examples/01_single_n8n_agent.py &

# Wait a moment, then run the LangGraph workflow
python examples/06_langgraph_single_agent.py
```

This demonstrates how to:
- Call A2A agents from LangGraph nodes
- Conditionally route to different agents
- Compose multiple agents in a workflow

### Example 7: LangGraph Workflow as A2A Server

Expose a LangGraph workflow as an A2A-compliant server:

```bash
# Set OpenAI API key (optional, for LLM version)
export OPENAI_API_KEY="your-key"

# Start the LangGraph A2A server
python examples/07_langgraph_server.py

# Or with real LLM integration
USE_LLM=true python examples/07_langgraph_server.py
```

The research agent will be available at `http://localhost:9002` with streaming support.

This demonstrates how to:
- Wrap LangGraph workflows as A2A servers
- Support streaming responses
- Handle both simple and chat-style inputs
- Extract output from workflow state

## Testing Your Agents

### Using curl

```bash
# Non-streaming request
curl -X POST http://localhost:9000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "messages.send",
    "params": {
      "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "What is 2+2?"}]
      }]
    },
    "id": 1
  }'

# Streaming request (using SSE)
curl -N http://localhost:8002/messages/stream?...
```

### Using Python A2A Client

```python
from a2a.client import A2AClient
from a2a.types import Message, MessageSendParams, TextPart

async def test_agent():
    client = A2AClient(base_url="http://localhost:9000")
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextPart(type="text", text="Hello!")]
            )
        ]
    )
    
    response = await client.send_message(params)
    print(response)
    
    await client.close()

import asyncio
asyncio.run(test_agent())
```

## Common Issues

### Port Already in Use

If you see "Address already in use" error:

```bash
# Find process using the port
lsof -i :9000

# Kill the process
kill -9 <PID>

# Or change the port in the example
serve_agent(..., port=8010)
```

### Import Errors

If you get import errors:

```bash
# Install missing framework
pip install a2a-adapter[crewai]  # or [langchain], [langgraph]

# Or install all
pip install a2a-adapter[all]
```

### OpenAI API Key

For LangChain/LangGraph examples:

```bash
# Export key
export OPENAI_API_KEY="sk-..."

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

## Next Steps

1. **Modify Examples**: Change prompts, models, or logic
2. **Create Custom Adapter**: Integrate your own agent frameworks
3. **Build Multi-Agent Systems**: Connect multiple A2A agents
4. **Deploy to Production**: Use Docker, Kubernetes, or serverless

## Resources

- [Main README](../README.md)
- [Architecture Documentation](../ARCHITECTURE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [A2A Protocol Spec](https://github.com/a2a-protocol/a2a-protocol)

## Support

Questions? Open an issue or discussion on GitHub!

