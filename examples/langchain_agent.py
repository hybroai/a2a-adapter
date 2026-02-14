"""
Example: LangChain Chain as A2A Agent (with streaming)

Expose a LangChain chain as an A2A-compatible agent in 3 lines.
Streaming is auto-detected from the runnable.

Prerequisites:
- langchain-openai installed: pip install a2a-adapter[langchain]
- OPENAI_API_KEY set in environment

Usage:
    python examples/langchain_agent.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from a2a_adapter import LangChainAdapter, serve_agent

# --- Set up your LangChain chain (this is framework code, not adapter code) ---

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer questions clearly and concisely."),
        ("user", "{input}"),
    ])
    | ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.7)
)

# --- A2A: 3 lines ---

adapter = LangChainAdapter(
    runnable=chain,
    input_key="input",
    name="LangChain Chat Agent",
    description="AI chat assistant powered by GPT-4o-mini with streaming",
)

serve_agent(adapter, port=8002)  # Streaming auto-detected!
