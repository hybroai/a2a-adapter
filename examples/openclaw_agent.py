"""
Example: OpenClaw Agent as A2A Agent

Expose an OpenClaw personal AI agent as an A2A-compatible agent in 3 lines.

Prerequisites:
- OpenClaw CLI installed: npm install -g openclaw
- OpenClaw configured: openclaw config set anthropic.apiKey "your-key"

Usage:
    python examples/openclaw_agent.py
"""

from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(
    thinking="low",
    timeout=300,
    name="OpenClaw Agent",
    description="Personal AI super agent — coding, research, automation, and more",
)

serve_agent(adapter, port=9008)
