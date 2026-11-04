# a2a-adapter API Reference

**API line**: 0.2 | **A2A SDK**: `a2a-sdk` 1.x | **Python**: >=3.11 | **License**: Apache-2.0

For installation and quickstart, see SKILL.md.

## Public API

All imports come from `a2a_adapter`:

```python
from a2a_adapter import (
    # Core
    BaseA2AAdapter,           # Abstract base -- subclass for custom adapters
    AdapterMetadata,          # Dataclass for AgentCard auto-generation
    CancelledByAdapterError,  # Raise from invoke()/stream() when cancel() stopped the task
    # Server
    serve_agent,       # One-line server: adapter -> uvicorn
    to_a2a,            # Adapter -> ASGI app (for production deployment)
    build_agent_card,  # Adapter -> AgentCard
    # Loader
    load_adapter,      # Factory: config dict -> adapter instance
    register_adapter,  # Decorator: register third-party adapters
    # Built-in adapters (lazy-loaded)
    N8nAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    CrewAIAdapter,
    OpenClawAdapter,
    ClaudeCodeAdapter,
    CodexAdapter,
    HermesAdapter,
    PiAdapter,
    DeepSeekHarnessAdapter,
    DeepSeekHarnessHeadlessAdapter,
    CallableAdapter,
    OllamaAdapter,
    OllamaClient,
)
```

`BaseAgentAdapter`, `load_a2a_agent` and `build_agent_app` are deprecated v0.1 exports kept for backwards compatibility.

Every adapter below also accepts the AgentCard fields `name`, `description`, `skills`, `documentation_url` and `icon_url`, plus a provider dict (`provider=`, or `agent_provider=` on `HermesAdapter` and the DeepSeek Harness adapters, whose `provider` means the model provider).

## Built-in Adapters

### N8nAdapter -- n8n webhook

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(
    webhook_url="http://localhost:5678/webhook/agent",  # Required
    timeout=30,               # HTTP timeout (seconds)
    headers={"Authorization": "Bearer xxx"},  # Extra headers
    max_retries=2,            # Retry count
    backoff=0.25,             # Retry backoff in seconds
    retry_on_server_error=False,  # Also retry 5xx, read timeouts, dropped connections (idempotent webhooks only)
    payload_template=None,    # Base payload; its keys win over client-supplied JSON
    message_field="message",  # Payload field name (default: "message")
    parse_json_input=True,    # Auto-parse JSON strings
    input_mapper=None,        # Optional: (raw_input, context_id) -> dict
    default_inputs=None,      # Merge into every request
    multimodal_mode=False,    # Enable file/image support in requests and responses
    name="My Agent",
    description="Does X",
)
serve_agent(adapter, port=9000)
```

Multimodal mode adds `file_field`, `image_field`, `encode_files_base64`, `fetch_file_urls`, `max_file_bytes`, `supported_input_types` and `supported_output_types`. In multimodal mode the adapter returns multimodal responses when the workflow returns structured content.

### LangChainAdapter -- LangChain Runnable (auto-streaming)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import LangChainAdapter, serve_agent

chain = ChatPromptTemplate.from_template("Answer: {input}") | ChatOpenAI(model="gpt-4o-mini")
adapter = LangChainAdapter(
    runnable=chain,           # Required: any LangChain Runnable
    input_key="input",        # Key to wrap user text (default: "input")
    output_key=None,          # Extract specific output key
    timeout=60,               # Seconds
    streaming=None,           # None: auto-detect astream(); True/False to force
    name="LangChain Agent",
    description="Answers questions",
)
serve_agent(adapter, port=9001)
```

Also accepts `parse_json_input`, `input_mapper` and `default_inputs` (see Input Handling Pipeline).

### LangGraphAdapter -- LangGraph CompiledGraph (auto-streaming)

```python
from a2a_adapter import LangGraphAdapter, serve_agent

graph = builder.compile()  # Your LangGraph compiled graph
adapter = LangGraphAdapter(
    graph=graph,              # Required: CompiledGraph
    input_key="messages",     # Input key (default: "messages")
    output_key=None,          # Extract specific output key
    state_key=None,           # State key for streaming text (default: output_key or auto-detect)
    timeout=60,               # Seconds
    streaming=None,           # None: auto-detect; True/False to force
    config=None,              # Base RunnableConfig for every run
    thread_id_key="thread_id",  # configurable key that receives the A2A context_id
    name="LangGraph Agent",
    description="Workflow agent",
)
serve_agent(adapter, port=9002)
```

When `input_key="messages"`, user text is auto-wrapped in `HumanMessage` if `langchain_core` is available. Also accepts `parse_json_input`, `input_mapper` and `default_inputs`.

### CrewAIAdapter -- CrewAI Crew

```python
from a2a_adapter import CrewAIAdapter, serve_agent

adapter = CrewAIAdapter(
    crew=your_crew,           # Required: CrewAI Crew instance
    timeout=300,              # Execution timeout (seconds)
    inputs_key="inputs",      # Key for crew inputs (default: "inputs")
    name="CrewAI Agent",
    description="Research crew",
)
serve_agent(adapter, port=9003)  # No streaming support
```

Also accepts `parse_json_input`, `input_mapper` and `default_inputs`.

### OpenClawAdapter -- OpenClaw CLI

```python
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(
    thinking="low",           # off/minimal/low/medium/high/xhigh
    agent_id=None,            # OpenClaw agent ID, e.g. "main"
    session_id=None,          # Auto-generated if None
    timeout=600,              # Subprocess timeout (seconds)
    openclaw_path="openclaw", # Binary path
    cli_args=None,            # Extra native CLI arguments
    working_directory=None,   # CWD for subprocess
    env_vars=None,            # Extra environment variables
)
serve_agent(adapter, host="127.0.0.1", port=9004)  # Supports cancel() via process kill
```

The prompt goes to OpenClaw on stdin (`openclaw agent --message-file`), so it needs an OpenClaw release with that option.

### Local coding-agent adapters

Claude Code, Codex, Pi and DeepSeek Harness can be exposed from Python or directly through the `a2a-adapter` CLI (see CLI below). Hermes is an in-process Python SDK integration. These agents run tools on the host for whoever can reach the server, so bind to `127.0.0.1` unless clients are trusted.

```python
from a2a_adapter import (
    ClaudeCodeAdapter, CodexAdapter, PiAdapter, HermesAdapter,
    DeepSeekHarnessAdapter, DeepSeekHarnessHeadlessAdapter,
)

claude = ClaudeCodeAdapter(working_dir="/path/to/project")  # claude_path, cli_args, skip_permissions
codex = CodexAdapter(working_dir="/path/to/project")        # codex_path, cli_args, bypass_approvals,
                                                            # skip_git_check (needed outside a git repo), env_vars
pi = PiAdapter(working_dir="/path/to/project")              # pi_command, model, model_provider,
                                                            # session_id, session_dir, enforce_single_context
hermes = HermesAdapter(model=None, provider=None)           # enabled_toolsets, max_workers
dsh = DeepSeekHarnessAdapter(working_dir="/path/to/project")
dsh_headless = DeepSeekHarnessHeadlessAdapter(working_dir="/path/to/project")
```

Claude Code and Codex save one agent session per A2A `contextId` under `$XDG_STATE_HOME/a2a-adapter/<agent>/<hash of working_dir>/` (default `~/.local/state/...`), so conversations resume after a server restart.

Pi keeps a single session (`session_id`, default `a2a-pi-agent`, stored under `.../pi/<hash>/sessions`). By default the adapter binds to the first A2A `contextId` and fails tasks from any other context; pass `enforce_single_context=False` to let every context share that one Pi session.

**DeepSeekHarnessAdapter** wraps `deepseek-harness-sdk` (install `a2a-adapter[deepseek-harness]`, set `DEEPSEEK_API_KEY`):

```python
adapter = DeepSeekHarnessAdapter(
    working_dir="/path/to/project",  # Required
    model=None,               # SDK default: deepseek-v4-flash
    provider=None,            # SDK default: deepseek-official
    reasoning_effort=None,
    max_tokens=None,
    profile="sdk",            # workspace-write; "sdk-minimal" is full access (logs a warning)
    patches=(),               # dsh config patch files
    dsh_home=None,            # Default: $DSH_HOME, else per-user state dir
    dsh_bin=None,             # Default: runtime bundled with the SDK
    base_url=None,            # Model API base URL
    api_key=None,             # Prefer the DEEPSEEK_API_KEY environment variable
    timeout=600,              # Per-turn timeout (seconds)
    startup_timeout=30,       # Runtime launch timeout (seconds)
)
serve_agent(adapter, host="127.0.0.1", port=9013)
```

- One `dsh` runtime is launched on the first request and reused; each A2A `contextId` maps to a durable dsh session, so conversations resume after a restart.
- Turns run one at a time across all conversations. Cancel and timeout close the runtime (the SDK has no interrupt call); the next turn relaunches it and resumes the session.
- Streams each assistant message as the runtime finishes it (the runtime reports whole messages, not tokens).
- The `sdk` profile confines shell commands and file writes to `working_dir` and temp directories; reads and network access (including loopback and private hosts) are not confined, and `DSH_PERMISSION_MODE`, `patches` or another profile can change this.

**DeepSeekHarnessHeadlessAdapter** runs `dsh --profile headless` once per request with the task on stdin. It keeps no session state (`contextId` carries no history) and does not stream; prefer `DeepSeekHarnessAdapter`. Library only, no CLI subcommand. Options: `dsh_path` (default: the bundled runtime, else `dsh` on `PATH`), `cli_args`, `env_vars`, `dsh_home`, `timeout`.

See `examples/claude_code_agent.py`, `examples/codex_agent.py`, `examples/pi_agent.py`, `examples/hermes_agent.py` and `examples/deepseek_harness_agent.py` for complete setups.

### OllamaAdapter -- Local Ollama LLM (streaming)

```python
from a2a_adapter import OllamaAdapter, OllamaClient, serve_agent

client = OllamaClient(
    model="llama3.2",             # Ollama model name (default: "llama3.2")
    base_url="http://localhost:11434",  # Ollama server URL
    system_prompt=None,           # Optional system prompt
    temperature=None,             # Sampling temperature
    timeout=120,                  # HTTP timeout (seconds)
    keep_alive=None,              # Model keep-alive duration (e.g. "5m")
)
adapter = OllamaAdapter(
    client=client,                # Optional: OllamaClient instance
    name="Local LLM",
    description="Ollama-powered agent",
)
serve_agent(adapter, port=10010)  # Streaming always supported
```

Convenience shorthand (creates `OllamaClient` internally from the same keyword arguments):

```python
adapter = OllamaAdapter(model="llama3.2", name="Local LLM")
serve_agent(adapter, port=10010)
```

### CallableAdapter -- Any async/sync function

```python
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs.get('message', '')}"

adapter = CallableAdapter(
    func=my_agent,            # Required: callable(dict) -> str
    streaming=False,          # Set True if func is an async generator
    timeout=None,             # Optional seconds
    sync_in_thread=True,      # Run a sync func in a worker thread
    name="Echo Agent",
    description="Echoes input",
)
serve_agent(adapter, port=9005)
```

`func` receives `{"message": user_input, "context_id": context_id}`. For streaming, `func` should be an async generator yielding str chunks.

## CLI

```bash
a2a-adapter <claude|codex|pi|openclaw|dsh> [server options] [-- native agent options]
```

Run it from the project directory the agent should work in. Options before `--` configure the A2A server; options after `--` go to the agent CLI (`dsh` accepts none). The adapter owns protocol and session flags such as Pi's `--mode rpc` and Codex's `--json`.

| Server option | Default | Purpose |
|---|---|---|
| `--cwd` | current directory | Directory the agent works in |
| `--host` / `--port` | `127.0.0.1` / `9000` | Bind address; `--host 0.0.0.0` accepts remote connections |
| `--name` | `<Agent> (<folder>)` | Agent Card display name |
| `--description` | `<Agent> agent working in <folder>` | Agent Card description |
| `--skill` | one skill from the card name and description | Agent Card skill as JSON (`id`, `name`, `description`, at least one `tags` entry, optional `examples`); repeatable |
| `--timeout` | `600` | Per-request timeout in seconds |
| `--executable` | `claude`, `codex`, `pi`, `openclaw`; for `dsh`, the bundled runtime | Path to the agent executable |
| `--log-level` | `warning` | Level of Uvicorn's own log lines in the activity log |
| `-q` / `--quiet` | off | Show only the startup card and errors |
| `-v` / `--verbose` | off | Show full prompt and reply text under each activity row |
| `--debug` | off | Also show raw A2A request, response and stream payloads |

`-q`, `-v` and `--debug` are mutually exclusive. Once the server is listening, the CLI prints a startup card: the agent's name, its Agent Card, JSON-RPC and REST URLs, its capabilities and the backend command. After that it prints an activity log, with one row per A2A operation (`message/send`, `tasks/get`, ...) and per task state change, plus one-line previews of each prompt and reply. Output that isn't a terminal, or runs with `NO_COLOR` set, gets the same information as plain lines.

Agent-specific options:
- **`openclaw`:** `--agent-id`, `--session-id`, `--thinking` (default `low`).
- **`pi`:** `--session-id` (default `a2a-pi-agent`), `--session-dir`; `A2A_PI_COMMAND` selects a Pi source command.
- **`dsh`:** `--model`, `--provider`, `--profile`, `--dsh-home`, `--reasoning-effort`, `--max-tokens`, `--startup-timeout`.

`a2a-adapter <agent> -help` prints agent-specific usage.

## Server Functions

### `serve_agent()` -- development server

```python
serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,   # Override auto-generated card
    host: str = "0.0.0.0",                 # All interfaces; use "127.0.0.1" for local-only
    port: int = 9000,
    log_level: str = "info",
    *,
    display: TerminalDisplay | None = None,  # CLI startup card + activity log
    **kwargs,                              # Passed to uvicorn.run()
)
```

### `to_a2a()` -- production ASGI app

```python
app = to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,   # Default: InMemoryTaskStore
    **card_overrides,                      # name=, description=, url=, version=, streaming=,
                                           # provider=, documentation_url=, icon_url=
)
# Deploy with any ASGI server:
# gunicorn app:app -k uvicorn.workers.UvicornWorker
# hypercorn app:app
# daphne app:app
```

`streaming=False` does not disable auto-detection; override `supports_streaming()` to turn streaming off.

The app serves JSON-RPC at `/` (A2A 1.0 methods plus the v0.3 names for older clients), REST routes, and the AgentCard at `/.well-known/agent-card.json` (also the legacy `/.well-known/agent.json`). Auto-enabled capabilities on the generated AgentCard: `streaming` (auto-detected) and `push_notifications=True`.

### `build_agent_card()` -- generate AgentCard

```python
card = build_agent_card(
    adapter: BaseA2AAdapter,
    **overrides,     # name=, description=, url=, version=, streaming=,
                     # provider=, documentation_url=, icon_url=
)
```

Default url is `http://localhost:9000/`. Override with `url="http://prod:8080/"`.

## BaseA2AAdapter Method Reference

| Method | Required | Signature | Description |
|---|---|---|---|
| `invoke` | **Yes** | `async (str, str\|None, **kwargs) -> str \| list[Part]` | Execute agent, return text or multimodal |
| `stream` | No | `async (str, str\|None, **kwargs) -> AsyncIterator[str \| Part]` | Yield text/multimodal chunks |
| `cancel` | No | `async (str\|None, **kwargs) -> None` | Cancel current execution |
| `close` | No | `async () -> None` | Release resources |
| `get_metadata` | No | `() -> AdapterMetadata` | Metadata for AgentCard |
| `supports_streaming` | No | `() -> bool` | Auto-detects from stream() override |

All methods receiving `**kwargs` get `context=RequestContext` from the bridge layer. The bridge calls `cancel(context=ctx)` without `context_id`, so identify the task to stop with `kwargs["context"].task_id` (or `.context_id`).

**Behavioral notes:**
- `invoke()` is the ONLY required method; all others are optional
- `stream()` is auto-detected: if overridden, `supports_streaming()` returns `True`
- When the adapter supports streaming, the bridge uses `stream()` for every request, including non-streaming `SendMessage`; the reply then holds all streamed text
- Exceptions in `invoke()`/`stream()` are caught and converted to failed tasks automatically
- Raise `CancelledByAdapterError` from `invoke()`/`stream()` when `cancel()` stopped the task: the bridge treats it as already canceled, not failed, and sends nothing more
- `context_id` enables multi-turn conversations (same ID = same conversation)
- Adapter supports `async with` for resource management (`close()` called on exit)

## Custom Adapter Example

Full example showing all optional methods working together:

```python
from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

class MyAdapter(BaseA2AAdapter):
    # REQUIRED: the only method you must implement
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        # kwargs['context'] provides the full A2A RequestContext (see below)
        result = await call_my_framework(user_input)
        return str(result)

    # OPTIONAL: streaming support
    async def stream(self, user_input: str, context_id: str | None = None, **kwargs):
        async for chunk in my_framework_stream(user_input):
            yield str(chunk)

    # OPTIONAL: cancellation support (the bridge passes only context=ctx)
    async def cancel(self, context_id: str | None = None, **kwargs) -> None:
        context = kwargs.get("context")
        process = self._processes.get(context.task_id) if context else None
        if process:
            process.kill()

    # OPTIONAL: resource cleanup
    async def close(self) -> None:
        await self._client.aclose()

    # OPTIONAL: metadata for auto AgentCard generation
    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="My Agent",
            description="Does something useful",
            version="1.0.0",
            skills=[{"id": "main", "name": "Main Skill", "description": "..."}],
            streaming=True,  # Set True if stream() is implemented
        )

serve_agent(MyAdapter(), port=9000)
```

## AdapterMetadata

```python
AdapterMetadata(
    name="",                          # Agent name (defaults to class name in card)
    description="",                   # What the agent does
    version="1.0.0",                  # Semantic version
    skills=[],                        # List of skill dicts (see below)
    input_modes=["text"],             # Supported input MIME types
    output_modes=["text"],            # Supported output MIME types
    streaming=False,                  # Whether adapter supports streaming
    provider=None,                    # Dict with "organization" and "url" keys
    documentation_url=None,           # URL to agent documentation
    icon_url=None,                    # URL to agent icon
)
```

**Skill dict format:**
```python
{
    "id": "skill-0",            # Unique skill ID
    "name": "Main Skill",       # Human-readable name
    "description": "...",       # What this skill does
    "tags": ["tag1", "tag2"],   # Required by A2A: at least one tag
    "examples": ["..."],        # Optional example queries
    "input_modes": ["text"],    # Optional per-skill input modes
    "output_modes": ["text"],   # Optional per-skill output modes
}
```

## Multimodal Responses

Adapters can return multimodal content using the A2A SDK 1.x `Part` type (`text`, `url`, `raw`, `data`, plus `filename` and `media_type`):

```python
from a2a.types import Part

class MultimodalAdapter(BaseA2AAdapter):
    # Return list[Part] instead of str for multimodal output
    async def invoke(self, user_input, context_id=None, **kwargs) -> list[Part]:
        return [
            Part(text="Here's your report"),
            Part(
                url="http://example.com/report.pdf",
                filename="report.pdf",
                media_type="application/pdf",
            ),
        ]

    # stream() can also yield Part objects alongside str chunks
    async def stream(self, user_input, context_id=None, **kwargs):
        yield "Generating chart..."
        yield Part(url="http://example.com/chart.png", filename="chart.png", media_type="image/png")
```

**Return types:**
- `invoke()` -> `str` (text only) or `list[Part]` (multimodal)
- `stream()` yields `str` (text chunks) or `Part` (multimodal chunks)

## Config-Driven Loading

```python
from a2a_adapter import load_adapter, serve_agent

adapter = load_adapter({
    "adapter": "n8n",
    "webhook_url": "http://localhost:5678/webhook/agent",
    "timeout": 60,
})
serve_agent(adapter)
```

Valid adapter values: `"n8n"`, `"langchain"`, `"langgraph"`, `"crewai"`, `"openclaw"`, `"ollama"`, `"callable"`, `"claude-code"`, `"codex"`, `"hermes"`, `"pi"`, `"deepseek-harness"`, `"deepseek-harness-headless"`, or any registered name. The other keys are passed to the adapter's constructor.

## Third-Party Adapter Registration

```python
from a2a_adapter import register_adapter, BaseA2AAdapter, load_adapter

@register_adapter("my_framework")
class MyFrameworkAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "result"

# Now loadable via config:
adapter = load_adapter({"adapter": "my_framework"})
```

Registered adapters take priority over built-ins with the same name.

## Input Handling Pipeline

The n8n, LangChain, LangGraph, and CrewAI adapters support a configurable
input pipeline:

1. **`input_mapper`** (highest priority): custom function that transforms raw input
2. **JSON parse**: auto-parse input that looks like a JSON object
3. **Input key fallback**: wrap text under the adapter's configured input key

CLI-backed adapters, Hermes and DeepSeek Harness accept plain user text directly.

## Accessing RequestContext

The bridge layer passes `context=RequestContext` via `**kwargs`. Use it when you need more than just the text input:

```python
async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
    context = kwargs.get('context')  # A2A SDK RequestContext
    if context:
        # Access full message parts (text, files, data, etc.)
        for part in context.message.parts:
            ...
        # Access task ID
        task_id = context.task_id
    return "response"
```

## Resource Management

Adapters support `async with` for automatic cleanup:

```python
async with MyAdapter() as adapter:
    result = await adapter.invoke("hello")
# adapter.close() called automatically
```

## Adapter Capabilities Summary

| Adapter | Streaming | Cancel | Multimodal | Extra Deps |
|---|---|---|---|---|
| `N8nAdapter` | No | No | Yes | None |
| `LangChainAdapter` | Auto-detected | No | No | `langchain`, `langchain-core` |
| `LangGraphAdapter` | Auto-detected | No | No | `langgraph` |
| `CrewAIAdapter` | No | No | No | `crewai` |
| `OpenClawAdapter` | No | Yes (kill) | No | OpenClaw CLI |
| `ClaudeCodeAdapter` | Yes | Yes (kill) | No | Claude Code CLI |
| `CodexAdapter` | No | Yes (kill) | No | Codex CLI |
| `PiAdapter` | Yes | Yes | No | Pi CLI |
| `HermesAdapter` | Yes | Yes (interrupt) | No | Hermes on `PYTHONPATH` |
| `DeepSeekHarnessAdapter` | Yes (per message) | Yes (closes runtime) | No | `deepseek-harness` extra |
| `DeepSeekHarnessHeadlessAdapter` | No | Yes (kill) | No | `dsh` (from the `deepseek-harness` extra) |
| `OllamaAdapter` | Always | No | No | Ollama server |
| `CallableAdapter` | Optional | No | No | None |

## Architecture (Request Flow)

```
Client -> A2A Starlette app -> DefaultRequestHandler -> AdapterAgentExecutor -> YourAdapter.invoke()/stream()
```

The `AdapterAgentExecutor` bridge handles:
- Extracting user text from `RequestContext` via `context.get_user_input()`
- Routing to `stream()` when `adapter.supports_streaming()`, else `invoke()`
- Converting text/multimodal responses to A2A events via `TaskUpdater`
- Error handling (exception -> Task with state=failed, error message preserved)
- Cancellation (`adapter.cancel()` -> Task with state=canceled)

Users never interact with the bridge layer directly.

### Non-streaming adapter (`SendMessage`, v0.3 `message/send`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler
  -> executor.execute(ctx, queue)
    -> enqueue Task(submitted)
    -> TaskUpdater.start_work()
    -> adapter.invoke(user_input, context_id, context=ctx)
    -> TaskUpdater.add_artifact(result)
    -> TaskUpdater.complete()
  -> returns the Task
```

### Streaming adapter (`SendStreamingMessage` / v0.3 `message/stream`, and also `SendMessage`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler
  -> executor.execute(ctx, queue)
    -> enqueue Task(submitted)
    -> TaskUpdater.start_work()
    -> adapter.stream(user_input, context_id, context=ctx)
      -> each chunk -> TaskUpdater.add_artifact(chunk, artifact_id=<one id>, append=<not first chunk>)
         (one chunk is held back so the last can be marked last_chunk=True)
    -> TaskUpdater.complete()
  -> events SSE-streamed to the client (SendStreamingMessage) or aggregated into the Task (SendMessage)
```

### Cancellation (`CancelTask`, v0.3 `tasks/cancel`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler
  -> executor.cancel(ctx, queue)
    -> adapter.cancel(context=ctx)
    -> TaskUpdater.cancel()
```

## Decision Guide

| Scenario | Use |
|---|---|
| Wrap a function | `CallableAdapter(func=fn)` |
| n8n workflow | `N8nAdapter(webhook_url=...)` |
| LangChain chain | `LangChainAdapter(runnable=chain)` |
| LangGraph workflow | `LangGraphAdapter(graph=graph)` |
| CrewAI crew | `CrewAIAdapter(crew=crew)` |
| Claude Code CLI | `ClaudeCodeAdapter(working_dir=...)` |
| Codex CLI | `CodexAdapter(working_dir=...)` |
| Pi coding agent | `PiAdapter(working_dir=...)` |
| DeepSeek Harness (dsh) | `DeepSeekHarnessAdapter(working_dir=...)` |
| DeepSeek Harness, one stateless task per request | `DeepSeekHarnessHeadlessAdapter(working_dir=...)` |
| OpenClaw agent | `OpenClawAdapter(...)` |
| Hermes Agent | `HermesAdapter(...)` |
| Local Ollama model | `OllamaAdapter(model="llama3.2")` |
| Any other framework | Subclass `BaseA2AAdapter`, implement `invoke()` |
| Need streaming | Implement `stream()` or use an adapter whose Streaming column is not No above |
| Need multimodal output | Return `list[Part]` from `invoke()` |
| Production deploy | `to_a2a(adapter)` -> ASGI server |
| Config-driven | `load_adapter({"adapter": "n8n", ...})` |
