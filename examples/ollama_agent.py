"""Ollama A2A agent — expose a local Ollama model as an A2A server.

Requirements:
    - Ollama running locally: `ollama serve`
    - Model pulled: `ollama pull llama3.2:8b`

Usage:
    python ollama_agent.py
    # Agent card at http://localhost:10010/.well-known/agent-card.json
"""

from a2a_adapter import OllamaAdapter, serve_agent
from a2a_adapter.integrations.ollama import OllamaClient

client = OllamaClient(model="llama3.2:8b")
adapter = OllamaAdapter(
    client=client,
    name="My Ollama Agent",
    description="Local LLM via Ollama",
)
serve_agent(adapter, port=10010)
