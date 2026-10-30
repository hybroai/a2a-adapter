# A2A Adapter

[![PyPI version](https://badge.fury.io/py/a2a-adapter.svg)](https://badge.fury.io/py/a2a-adapter)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Make any AI agent an A2A Protocol server: a few lines of Python for your own
agent, or one command for a local agent like Claude Code or Pi.**

`a2a-adapter` wraps agent CLIs, local agents, workflow engines, and Python agent
frameworks behind the [A2A (Agent-to-Agent) Protocol](https://github.com/a2aproject/A2A).
It handles AgentCard generation, JSON-RPC, task lifecycle, and streaming through
the official A2A SDK, so your adapter only needs to invoke the underlying agent.

Works with Claude Code, Codex, Pi, OpenClaw, n8n, LangGraph, LangChain,
CrewAI, Ollama, Hermes Agent, and custom Python functions.

## Two Ways to Use It

| | Python SDK | CLI |
|---|---|---|
| **Use it when** | You're writing an agent (or wrapping a framework) and want it to speak A2A | You want to put an existing local agent such as Claude Code or Pi on A2A, without writing code |
| **Install** | `pip install a2a-adapter` in your project's virtualenv | `uv tool install a2a-adapter` (or `pipx install a2a-adapter`) |
| **Run** | `serve_agent(adapter, port=9000)` in your code | `a2a-adapter claude --port 9010` |

Both need Python 3.11+ and serve the same A2A server, with an auto-generated
Agent Card at `/.well-known/agent-card.json`.

## 1. Use It as an SDK

Add A2A support to your own agent code. Install `a2a-adapter` into your
project's virtual environment, with extras for the framework you use:

```bash
pip install a2a-adapter              # core
pip install "a2a-adapter[langchain]" # or [crewai], [langgraph], [all]
```

Wrap your agent in an adapter and serve it:

```python
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs['message']}"

adapter = CallableAdapter(func=my_agent, name="Echo Agent")
serve_agent(adapter, port=9000)
```

Your agent is now an A2A server at `http://localhost:9000`, with its Agent
Card at `http://localhost:9000/.well-known/agent-card.json`. To mount it in an
existing ASGI app instead, use `to_a2a(adapter)`.

### Supported SDK Adapters

| Framework or agent | Adapter | Streaming | Example |
|---|---|---:|---|
| Claude Code | `ClaudeCodeAdapter` | Yes | [claude_code_agent.py](examples/claude_code_agent.py) |
| Codex | `CodexAdapter` | No | [codex_agent.py](examples/codex_agent.py) |
| Pi | `PiAdapter` | Yes | [pi_agent.py](examples/pi_agent.py) |
| OpenClaw | `OpenClawAdapter` | No | [openclaw_agent.py](examples/openclaw_agent.py) |
| n8n | `N8nAdapter` | No | [n8n_agent.py](examples/n8n_agent.py) |
| LangGraph | `LangGraphAdapter` | Yes | [langgraph_server.py](examples/langgraph_server.py) |
| LangChain | `LangChainAdapter` | Yes | [langchain_agent.py](examples/langchain_agent.py) |
| CrewAI | `CrewAIAdapter` | No | [crewai_agent.py](examples/crewai_agent.py) |
| Ollama | `OllamaAdapter` | Yes | [ollama_agent.py](examples/ollama_agent.py) |
| Hermes Agent | `HermesAdapter` | Yes | [hermes_agent.py](examples/hermes_agent.py) |
| Python callable | `CallableAdapter` | Optional | [quickstart.py](examples/quickstart.py) |
| Custom class | `BaseA2AAdapter` | Optional | [custom_adapter.py](examples/custom_adapter.py) |

## 2. Use It as a CLI

Start a classic local agent (Claude Code, Codex, Pi, or OpenClaw) as an A2A
server with one command.

**Install the CLI** as an isolated tool, so the `a2a-adapter` command lands on
your `PATH`:

```bash
uv tool install a2a-adapter    # or: pipx install a2a-adapter
```

Upgrade later with `uv tool upgrade a2a-adapter`. Avoid a system-wide
`pip install`: macOS's built-in `pip` runs Python 3.9, which is too old, and
Homebrew and many Linux distributions block it (PEP 668).

**Make sure the agent works on its own.** a2a-adapter wraps the agent you
already have; it doesn't install it. For example, install Claude Code and run
`claude` once to sign in. If the agent isn't on `PATH`, pass
`--executable /path/to/agent`.

**Start it from the project directory the agent should work in:**

```bash
a2a-adapter claude --port 9010
a2a-adapter pi --port 9012
a2a-adapter codex --port 9011
a2a-adapter openclaw --port 9008
```

Options before `--` configure the A2A server. Options after `--` are passed
to the underlying agent CLI:

```bash
a2a-adapter claude --port 9010 -- --model <model>
a2a-adapter pi --port 9012 -- --model <model> --thinking high
a2a-adapter codex --port 9011 -- --model <model> --sandbox workspace-write
a2a-adapter openclaw --port 9008 --agent-id main --thinking high
```

| Server option | Default | Purpose |
|---|---|---|
| `--cwd` | current directory | Directory the agent works in |
| `--host` / `--port` | `127.0.0.1` / `9000` | Bind address; use `--host 0.0.0.0` to accept remote connections |
| `--name` | `<Agent> (<folder>)`, e.g. `Claude Code (my-app)` | Agent Card display name |
| `--description` | `<Agent> agent working in <folder>` | Agent Card description |
| `--skill` | one skill from the card name and description | Agent Card skill as JSON (`id`, `name`, `description`, at least one entry in `tags`, optional `examples`); repeatable |
| `--timeout` | `600` | Per-request timeout in seconds |
| `--executable` | `claude`, `codex`, `pi`, `openclaw` | Path to the agent executable |

Pi source commands can be selected without a shell wrapper:

```bash
A2A_PI_COMMAND='npx tsx /path/to/pi/packages/coding-agent/src/cli.ts' \
  a2a-adapter pi --port 9012
```

Run `a2a-adapter -help` for the supported agents, or
`a2a-adapter <agent> -help` for agent-specific usage.
The adapter owns protocol and session flags such as Pi's `--mode rpc` and
Codex's `--json`; those flags cannot be overridden after `--`.

### Configuring the Agent

The adapter runs the agent non-interactively (for example `claude -p` or
`codex exec`), so there's no agent UI to change settings in. You configure it
the same way you would without the adapter, in two ways:

- **The agent's own config files.** The adapter runs the agent as you, in the
  `--cwd` folder, so it reads your usual setup: `~/.claude/settings.json`,
  `CLAUDE.md`, and MCP servers for Claude Code; `~/.codex/config.toml` and
  `AGENTS.md` for Codex; Pi's settings and installed extensions. Claude Code,
  Codex, and OpenClaw start a fresh process for each request, so changes apply
  to the next request. Pi keeps one process running, so restart the server
  after changing Pi's config.
- **Native CLI flags after `--`.** Use these for settings that should apply
  only to this server, such as a different model or sandbox mode.

## Documentation

- [Quick Start](QUICKSTART.md) - end-to-end setup and testing guide
- [Examples](examples/README.md) - runnable examples for each adapter
- [Architecture](ARCHITECTURE.md) - SDK delegation, request flow, and internals
- [Changelog](CHANGELOG.md) - release notes and migration history
- [Contributing](CONTRIBUTING.md) - development setup and contribution guide

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hybroai/a2a-adapter&type=Date&legend=top-left&v=2)](https://www.star-history.com/?repos=hybroai%2Fa2a-adapter&type=date&legend=top-left)

## License

Apache-2.0 - see [LICENSE](LICENSE).

Built by [HYBRO AI](https://hybro.ai). Powered by the
[A2A Protocol](https://github.com/a2aproject/A2A).
