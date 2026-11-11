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
pip install "a2a-adapter[langchain]" # or [crewai], [langgraph], [all] (all frameworks)
```

`[all]` leaves out `[deepseek-harness]`; install that extra on its own.

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

`serve_agent()` binds `127.0.0.1` by default. To accept remote connections,
bind all interfaces and require a token:

```python
import os

serve_agent(adapter, host="0.0.0.0", auth_token=os.environ["A2A_AUTH_TOKEN"])
```

Push notifications are off by default. `serve_agent(adapter,
push_notifications=True)` (or `to_a2a(..., push_notifications=True)`) turns
them on; push URLs on localhost, private networks and cloud-metadata addresses
are then rejected unless you also pass `allow_private_push_urls=True`. A
request carrying one fails: JSON-RPC `-32602` on the A2A 1.0 methods, HTTP 400
on REST, and `-32603` with the same "Rejected push notification URL" message on
the v0.3 method names (a2a-sdk 1.0 reports all v0.3 handler errors that way).
Push targets are checked again when the connection opens, and the connection
goes to the checked address; pushes do not use `HTTP(S)_PROXY` unless
`allow_private_push_urls=True`. Without `auth_token`, any
client can make the server send requests.

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
| DeepSeek Harness | `DeepSeekHarnessAdapter` | Yes | [deepseek_harness_agent.py](examples/deepseek_harness_agent.py) |
| DeepSeek Harness, headless CLI (stateless) | `DeepSeekHarnessHeadlessAdapter` | No | — |
| Python callable | `CallableAdapter` | Optional | [quickstart.py](examples/quickstart.py) |
| Custom class | `BaseA2AAdapter` | Optional | [custom_adapter.py](examples/custom_adapter.py) |

`CrewAIAdapter` runs each request on its own copy of the crew, and falls back
to one request at a time for crews that cannot be faithfully copied (a timed-out
kickoff may still overlap the next request there).

## 2. Use It as a CLI

Start a classic local agent (Claude Code, Codex, Pi, OpenClaw, or DeepSeek
Harness) as an A2A server with one command.

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
a2a-adapter dsh --port 9013
```

Options before `--` configure the A2A server. Options after `--` are passed
to the underlying agent CLI:

```bash
a2a-adapter claude --port 9010 -- --model <model>
a2a-adapter pi --port 9012 -- --model <model> --thinking high
a2a-adapter codex --port 9011 -- --model <model> --sandbox workspace-write
a2a-adapter openclaw --port 9008 --agent-id main --thinking high
```

The adapters send each prompt to Codex and OpenClaw on stdin, so long prompts
are not limited by the OS command-line length; OpenClaw itself rejects prompts over
4 MiB. This needs an OpenClaw release that supports
`openclaw agent --message-file`.

Each A2A context gets its own OpenClaw session: `<session_id>-<context_id>` when
the context id is valid as it is and the result fits in 64 characters, otherwise
a readable head plus a digest. A saved session whose context id was altered by earlier releases
(mixed case, punctuation, edge dashes) or did not fit (including UUIDs with a
`--session-id` over 27 characters) starts fresh on its next message. The old
history stays on disk, unused, because it may mix several conversations.

| Server option | Default | Purpose |
|---|---|---|
| `--cwd` | current directory | Directory the agent works in |
| `--host` / `--port` | `127.0.0.1` / `9000` | Bind address; use `--host 0.0.0.0` with `--auth-token` to accept remote connections |
| `--auth-token` | none, or `$A2A_AUTH_TOKEN` | Require `Authorization: Bearer <token>` on every request except the Agent Card; prefer the variable, since argv is visible in `ps` |
| `--push-notifications` | off | Accept push-notification configs and POST task updates to them |
| `--allow-private-push-urls` | off | With `--push-notifications`, also allow receivers on localhost and private networks |
| `--expose-error-details` | off | Send exception text to clients, for local debugging only |
| `--name` | `<Agent> (<folder>)`, e.g. `Claude Code (my-app)` | Agent Card display name |
| `--description` | `<Agent> agent working in <folder>` | Agent Card description |
| `--skill` | one skill from the card name and description | Agent Card skill as JSON (`id`, `name`, `description`, at least one entry in `tags`, optional `examples`); repeatable |
| `--timeout` | `600` | Total time per request, in seconds |
| `--executable` | `claude`, `codex`, `pi`, `openclaw`; for `dsh`, the runtime bundled with the SDK | Path to the agent executable |

On POSIX, cancel, timeout and shutdown stop the agent and everything it
started (shells, MCP servers, tool processes). Ctrl+C or closing the terminal stops running agents
immediately; SIGTERM lets in-flight requests finish (uvicorn's graceful
timeout), then stops them.
Agents run detached from the server's terminal, so a tool that prompts on
`/dev/tty` (an SSH passphrase, a GPG pinentry) fails instead of prompting.

**Errors.** When a task fails, the client gets a generic message with the task
id, and the exception goes to the server log: exception text can hold URLs,
paths and tokens. In your own adapter, raise `UserFacingError("...")` to show
the client that message. `--expose-error-details` (`expose_error_details=True`
on `to_a2a` / `serve_agent`) sends the raw text, for local debugging only. The
CLI's failed row always shows the error; the traceback needs `--debug`.

Pi source commands can be selected without a shell wrapper:

```bash
A2A_PI_COMMAND='npx tsx /path/to/pi/packages/coding-agent/src/cli.ts' \
  a2a-adapter pi --port 9012
```

DeepSeek Harness runs through its Python SDK rather than a CLI, so it needs
the extra (`uv tool install "a2a-adapter[deepseek-harness]"`; upgrading an
install made without it does not add it) and `DEEPSEEK_API_KEY` in the
environment. It takes no native options after `--`;
use `--model`, `--provider`, `--profile`, `--dsh-home`, `--reasoning-effort`,
`--max-tokens` and `--startup-timeout` instead. Each A2A conversation
(`contextId`) gets its own durable dsh session under `--dsh-home` (default:
`$DSH_HOME`, else `a2a-adapter/deepseek-harness/<hash of --cwd>/` in the
per-user state directory below), so it resumes after the server restarts.
Turns run one at a time across all conversations, and `--timeout` counts only
a turn's own run, not time spent waiting for earlier turns. On Windows, pass
`--executable` the runtime executable rather than the `dsh` console script.

```bash
a2a-adapter dsh --port 9013 --model <model>
```

Run `a2a-adapter -help` for the supported agents, or
`a2a-adapter <agent> -help` for agent-specific usage.
The adapter owns protocol and session flags such as Pi's `--mode rpc` and
Codex's `--json`; those flags cannot be overridden after `--`.

Claude Code, Codex and Pi keep their session state per user, outside the
project: under `$XDG_STATE_HOME/a2a-adapter/<agent>/<hash of --cwd>/`, or
`~/.local/state/a2a-adapter/...` when `XDG_STATE_HOME` is unset. Each working
directory gets its own store. With Claude Code and Codex, each A2A
conversation (`contextId`) resumes the same agent session after the server
restarts; requests without a `contextId` all share one saved session. Pi keeps
its single session (`--session-id`) for the working directory.

Versions up to 0.2.15 kept this state in `<cwd>/.a2a-adapter/`, which is no
longer read. To keep existing Codex or Pi sessions, move the contents of
`<cwd>/.a2a-adapter/<agent>/` into the new directory, which this prints
(run it in the agent's working directory; agent is `codex` or `pi`):

```bash
python -c "from a2a_adapter.base_adapter import default_state_dir; print(default_state_dir('codex', '.'))"
```

Otherwise delete the old folder. Old Claude Code entries were never reused and
can just be deleted.

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
