# Quick Start Guide

Get up and running with A2A Adapters in 5 minutes!

## Installation

```bash
pip install a2a-adapter
```

## Your First A2A Agent

Let's create a simple echo agent using the callable adapter:

```python
# my_first_agent.py
import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

# Define your agent logic
async def echo_agent(inputs: dict) -> str:
    message = inputs["message"]
    return f"Echo: {message.upper()}"

# Main function
async def main():
    # Load the adapter
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": echo_agent
    })
    
    # Define agent card
    card = AgentCard(
        name="Echo Agent",
        description="A simple echo agent that repeats your message in uppercase"
    )
    
    # Start the server
    print("🚀 Echo Agent is running on http://localhost:9000")
    serve_agent(agent_card=card, adapter=adapter, port=9000)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python my_first_agent.py
```

## Testing Your Agent

Create a test client:

```python
# test_client.py
import asyncio
from a2a.client import A2AClient
from a2a.types import Message, MessageSendParams, TextPart

async def main():
    # Connect to your agent
    client = A2AClient(base_url="http://localhost:9000")
    
    # Send a message
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextPart(type="text", text="hello world")]
            )
        ]
    )
    
    # Get response
    response = await client.send_message(params)
    print(f"Agent says: {response.content[0].text}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:

```bash
# In terminal 1
python my_first_agent.py

# In terminal 2
python test_client.py
# Output: Agent says: Echo: HELLO WORLD
```

## Next Steps

### Use with LangChain

```bash
pip install a2a-adapter[langchain] langchain-openai
export OPENAI_API_KEY="your-key"
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm

adapter = await load_a2a_agent({
    "adapter": "langchain",
    "runnable": chain,
    "input_key": "input"
})
```

### Use with CrewAI

```bash
pip install a2a-adapter[crewai]
export OPENAI_API_KEY="your-key"
```

```python
from crewai import Crew, Agent, Task

# Create your crew
crew = Crew(agents=[...], tasks=[...])

# Wrap as A2A agent
adapter = await load_a2a_agent({
    "adapter": "crewai",
    "crew": crew
})
```

### Use with n8n

```python
adapter = await load_a2a_agent({
    "adapter": "n8n",
    "webhook_url": "https://n8n.example.com/webhook/my-agent"
})
```

### Create Custom Adapter

```python
from a2a_adapter import BaseAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart

class MyCustomAdapter(BaseAgentAdapter):
    async def to_framework(self, params: MessageSendParams):
        # Extract input
        text = params.messages[-1].content[0].text
        return {"input": text}
    
    async def call_framework(self, framework_input, params):
        # Your custom logic here
        result = my_custom_processing(framework_input["input"])
        return {"output": result}
    
    async def from_framework(self, framework_output, params):
        # Convert to A2A Message
        return Message(
            role="assistant",
            content=[TextPart(
                type="text",
                text=framework_output["output"]
            )]
        )

# Use it
adapter = MyCustomAdapter()
serve_agent(agent_card=card, adapter=adapter, port=9000)
```

## Common Patterns

### Multi-Agent Communication

```python
# Agent A (port 9000) - Math Expert
# Agent B (port 8001) - Research Expert

# In your orchestrator:
from a2a.client import A2AClient

math_agent = A2AClient(base_url="http://localhost:9000")
research_agent = A2AClient(base_url="http://localhost:8001")

# Call agents as needed
math_result = await math_agent.send_message(...)
research_result = await research_agent.send_message(...)
```

### Streaming Responses

```python
# Only works with streaming-capable adapters (LangChain)
async for chunk in client.send_message_stream(params):
    print(chunk, end="", flush=True)
```

### Error Handling

```python
from httpx import HTTPError

try:
    response = await client.send_message(params)
except HTTPError as e:
    print(f"Agent call failed: {e}")
    # Handle error
```

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
serve_agent(..., port=8001)
```

### Import Errors

```bash
# Install the specific framework support
pip install a2a-adapter[langchain]
pip install a2a-adapter[crewai]
pip install a2a-adapter[all]  # All frameworks
```

### Connection Refused

Make sure your agent server is running:

```bash
# Check if server is running
curl http://localhost:9000/health
```

## Resources

- 📖 [Full Documentation](README.md)
- 🏗️ [Architecture Guide](ARCHITECTURE.md)
- 💻 [More Examples](examples/)
- 🤝 [Contributing](CONTRIBUTING.md)

## Help & Support

- GitHub Issues: Report bugs or request features
- GitHub Discussions: Ask questions and share ideas
- Examples Directory: See complete working examples

---

**Ready to build? Check out the [examples/](examples/) directory for more!** 🚀

