"""
Example: Google Antigravity CLI (agy) as A2A Agent

Expose the Antigravity CLI as an A2A-compatible agent. Each A2A message runs
`agy --output-format stream-json --print=...`, and each A2A conversation resumes
its own Antigravity conversation with `--conversation`.

Prerequisites:
- Antigravity CLI installed, with `agy` on PATH
- Signed in once with an interactive `agy` session (headless runs reuse the
  cached credentials), or the Gemini API key / ADC setup from the
  Antigravity docs

Security:
    agy's default headless mode (auto) runs shell commands and edits files in
    the working directory without asking; actions its permission policy does
    not allow are skipped. Restrict tools in
    ~/.gemini/antigravity-cli/settings.json, or pin a mode with
    cli_args=["--mode", "plan"]. skip_permissions (off by default) approves
    everything, including what the policy denies; trusted, sandboxed
    environments only:

        # Option 1: constructor argument
        adapter = AntigravityAdapter(working_dir=..., skip_permissions=True)

        # Option 2: environment variable
        A2A_ANTIGRAVITY_SKIP_PERMISSIONS=1 python examples/antigravity_agent.py

Usage:
    python examples/antigravity_agent.py
    python examples/antigravity_agent.py /path/to/project
    # Agent card at http://localhost:9014/.well-known/agent-card.json
"""

import os
import sys

from a2a_adapter import AntigravityAdapter, serve_agent

working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

adapter = AntigravityAdapter(
    working_dir=working_dir,
    timeout=600,
    name="Antigravity Agent",
    description="Google Antigravity coding agent: code generation, debugging, refactoring",
    # Native agy flags, for example a model from `agy models`:
    # cli_args=["--model", "<model>"],
    # Uncomment for trusted, sandboxed environments only:
    # skip_permissions=True,
)

print("Starting Antigravity A2A agent...")
print(f"  Working directory: {working_dir}")
print(f"  Skip permissions: {adapter.skip_permissions}")
print("  Agent card: http://localhost:9014/.well-known/agent-card.json")

serve_agent(adapter, port=9014)
