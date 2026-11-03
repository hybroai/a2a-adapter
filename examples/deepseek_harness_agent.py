"""DeepSeek Harness as A2A server — expose a dsh coding agent over A2A.

Requirements:
    - pip install "a2a-adapter[deepseek-harness]"
      (Linux x64/arm64, macOS 14+, or Windows x64; no Node.js needed)
    - export DEEPSEEK_API_KEY=...

Usage:
    python examples/deepseek_harness_agent.py
    python examples/deepseek_harness_agent.py /path/to/project
    # Agent card at http://localhost:9010/.well-known/agent-card.json

Each A2A contextId maps to its own durable dsh session, so a conversation
continues across turns and server restarts. Sessions are stored under
DSH_HOME when set, otherwise in a per-user state directory.

The A2A server has no authentication and the agent runs shell commands in
the working directory, so this binds to localhost only.
"""

import os
import sys

from a2a_adapter import DeepSeekHarnessAdapter, serve_agent

working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

adapter = DeepSeekHarnessAdapter(
    working_dir=working_dir,
    model=os.getenv("A2A_DSH_MODEL"),
    name="DeepSeek Harness",
    description="DeepSeek Harness coding agent",
)

serve_agent(adapter, host="127.0.0.1", port=9010)
