"""Hermes Agent as A2A server — expose Hermes as an A2A-compatible agent.

Requirements:
    - Hermes configured: ~/.hermes/config.yaml and ~/.hermes/.env
      (run `hermes setup` if not yet configured)
    - Set HERMES_AGENT_PATH below to your hermes-agent directory,
      or export PYTHONPATH=/path/to/hermes-agent before running.

Usage:
    python examples/hermes_agent.py
    # Agent card at http://localhost:9010/.well-known/agent-card.json

Streaming is enabled by default — Hermes's stream_callback pipes tokens
through an asyncio.Queue to the A2A SSE stream.
"""

import os
import sys

HERMES_AGENT_PATH = os.path.expanduser("~/Projects/hermes-agent")

if HERMES_AGENT_PATH not in sys.path:
    sys.path.insert(0, HERMES_AGENT_PATH)

from a2a_adapter import HermesAdapter, serve_agent  # noqa: E402 (after the sys.path insert)

adapter = HermesAdapter(
    model="claude-opus-4.6",
    enabled_toolsets=["hermes-cli"],
    name="Hermes Agent",
    description="Multi-purpose AI assistant with tool use, persistent memory, and subagent delegation",
)

serve_agent(adapter, port=9010)
