# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.17] - 2026-11-03

### Added

- `DeepSeekHarnessAdapter` (`"deepseek-harness"` in `load_adapter`) exposes a
  DeepSeek Harness (dsh) coding agent through `deepseek-harness-sdk`. Install
  with `pip install "a2a-adapter[deepseek-harness]"`. One runtime is launched
  on the first request and reused; each A2A `contextId` maps to a durable dsh
  session, so conversations survive server restarts. Turns are serialized,
  and cancelling the running task restarts the runtime because the SDK has no
  interrupt call. Streaming sends each assistant message as the runtime
  finishes it (the runtime reports whole messages, not tokens). The A2A
  executor streams `message/send` too, so its reply holds every assistant
  message of the turn, not only the last
- `a2a-adapter dsh` serves `DeepSeekHarnessAdapter` from the command line, with
  `--model`, `--provider`, `--profile`, `--dsh-home`, `--reasoning-effort`,
  `--max-tokens` and `--startup-timeout`; `--timeout` and `--executable` map to
  the turn timeout and the `dsh` binary. Arguments after `--` are rejected
- `DeepSeekHarnessHeadlessAdapter` (`"deepseek-harness-headless"` in
  `load_adapter`) runs `dsh --profile headless` once per request, with the task
  on stdin. It does not use dsh's `--session-id` resume, so every request is a
  fresh session and `contextId` carries no history; prefer
  `DeepSeekHarnessAdapter`. Library only, no CLI subcommand

### Fixed

- A streaming task cancelled mid-stream no longer sends its held-back chunk
  after the canceled state; the executor now stops on
  `CancelledByAdapterError` without flushing it

## [0.2.16] - 2026-11-03

### Changed

- Claude Code, Codex and Pi session state now defaults to a per-user location,
  `$XDG_STATE_HOME/a2a-adapter/<agent>/<hash of working dir>/` (or
  `~/.local/state/a2a-adapter/...`), instead of `<working_dir>/.a2a-adapter/`,
  which left an untracked folder in the agent's project. Explicit
  `session_store_path` / `session_dir` arguments still win. Existing state is
  not migrated: to keep it, move the contents of
  `<working_dir>/.a2a-adapter/<agent>/` into the directory
  `a2a_adapter.base_adapter.default_state_dir(<agent>, <working_dir>)` returns
  (see the README), or delete the old folder
- `N8nAdapter` no longer retries a webhook call that may have reached n8n: a
  5xx response, a read timeout or a reset or dropped connection now fails at
  once, because the workflow may already have run (a workflow that sends an email
  used to send it up to `max_retries + 1` times). Only failures before the
  request is sent (connect error, connect timeout, pool timeout) are retried.
  Set `retry_on_server_error=True` to retry the others too, if the webhook is
  idempotent. To make the stale keep-alive race (a request sent on a socket
  n8n just closed) unlikely, idle connections now expire after 2s, below
  n8n's 5s keep-alive timeout
- `N8nAdapter` in `multimodal_mode` no longer downloads client-supplied file
  URLs: the URL is passed to the webhook as `uri` without a `data` field.
  Set `fetch_file_urls=True` to download and embed the file as before; it now
  accepts only `http`/`https`, does not follow redirects, rejects files over
  `max_file_bytes` (default 10 MiB) or downloads slower than `timeout`
  seconds in total, and fails the request if a fetch fails instead of
  dropping the file. Private and internal addresses are not
  blocked, so enable it only for trusted clients
- `OpenClawAdapter` now runs on the shared CLI subprocess runtime, like
  Claude Code and Codex. It sends the prompt on stdin with
  `--message-file -` instead of `--message <text>`, so it needs an OpenClaw
  release with `--message-file`; OpenClaw caps the prompt at 4 MiB.
  Requests that map to the same OpenClaw session ID now run one at a time;
  that includes all requests without a `context_id`, which share the default
  session. Error messages now read `Command failed with exit code ...`,
  `Command timed out after ...` and `Command returned empty output` (no
  `OpenClaw` prefix), a missing `working_directory` raises
  `Working directory does not exist`, and a failed run reports the `error`
  from OpenClaw's JSON output before stderr. The private
  `_current_process` attribute is gone; `_build_command()` returns a
  `CommandResult`. The CLI also rejects `-m` (including glued forms
  such as `-mhello`) and `--message-file` in pass-through OpenClaw arguments
- `LangGraphAdapter` now passes the A2A `context_id` as
  `config["configurable"]["thread_id"]`, so a graph compiled with a
  checkpointer keeps memory per conversation; such graphs used to fail with
  `Checkpointer requires one or more of the following 'configurable' keys`.
  New `config` (base config for every call; a `thread_id` set there wins)
  and `thread_id_key` (rename the key, or `None` to turn the mapping off)
  options. The client chooses the `context_id`, so on a server shared by
  clients that must not read each other's threads, set `thread_id_key=None`
- `LangGraphAdapter.stream()` now runs the graph with
  `stream_mode="updates"` and emits each node's latest AI message (or its
  output key) once, separated by a blank line. Compiled graphs stream updates by
  default, which the adapter used to read as empty; with whole-state
  streams it echoed the user's message as the reply. Tool results and
  human messages are no longer emitted; fan-out (`Send`) writes grouped in a
  list are all emitted. Graphs passed to the adapter must accept
  `stream_mode`, and `config` unless `thread_id_key=None` (LangGraph
  compiled graphs accept both)
- `LangGraphAdapter`, `LangChainAdapter` and `CrewAIAdapter` fail the
  request when `input_mapper` raises (`input_mapper failed: <type>:
  <message>`) or returns something other than a dict; they used to log a
  warning and silently fall back to the default input
- `LangGraphAdapter` and `LangChainAdapter` streams are now bounded by
  `timeout` for the whole stream, like `invoke()`. Timeouts in these
  adapters, `CrewAIAdapter` and `CallableAdapter` raise
  `TimeoutError("<framework> timed out after N seconds")` instead of an
  empty `TimeoutError`. New `streaming` option (default `None`: detect
  `astream`) to turn streaming off explicitly
- `CallableAdapter` runs sync functions in a worker thread (they used to fail
  with `object str can't be used in 'await' expression`) and has a new
  `timeout` option (default `None`, no limit) covering calls and streams.
  A sync wrapper that needs the running event loop (for example one returning
  `asyncio.ensure_future(...)`) now fails in the thread; pass
  `sync_in_thread=False` to call it on the loop as before
- The default `HermesAdapter` agent card name is now `Hermes` instead of
  `HermesAdapter`, matching the other CLI adapters' product names

### Fixed

- A stream that fails mid-flight now closes its artifact with the last chunk it
  held back; it used to re-send the previous chunk (`A`, `B`, `B` instead of
  `A`, `B`, `C`), and a single chunk followed by an error was dropped
- Cancelling a non-streaming CLI run (`tasks/cancel` or server shutdown) now
  kills its subprocess instead of leaving it running
- The CLI no longer rejects Pi arguments such as `-print` or `-color` as the
  managed `-p` / `-c`; Pi matches flags exactly. Claude Code still rejects
  them, since its parser reads glued short flags (`-print` is `-p`), and Codex
  still rejects glued forms of its managed flags such as `-opath` or `-C/tmp`
- CLI adapter session stores are written atomically (temp file, then rename),
  so a crash mid-write no longer corrupts `sessions.json` and drops every
  saved session. A corrupt or unreadable store is now logged as a warning
  instead of being reset silently
- Claude Code conversations now resume after a server restart. Sessions were
  keyed by a random per-process prefix, so saved `--resume` ids were never
  matched again and the store filled with dead entries; they are now keyed by
  the A2A `context_id` (`_default` without one, shared by every such request
  and now kept across restarts), as Codex already did. The
  `ClaudeCodeAdapter.session_id` attribute is removed. Entries saved by older
  versions use the old keys and are never matched, so the old
  `claude-code/sessions.json` is not worth moving
- Hermes conversation history now survives a server restart: the SessionDB
  session id is `a2a-<context_id>` instead of a random per-process prefix
  plus the context id. A call without a `context_id` gets a fresh one-off
  session; it used to share one session with every other such call in the
  same process. The `HermesAdapter.session_id` attribute is removed. Since
  Hermes's SessionDB is per user, two Hermes servers that see the same
  `context_id` now share that conversation's history and its Hermes
  task-scoped resources
- `N8nAdapter` now retries `ConnectTimeout` and `PoolTimeout`, and wraps
  other transport errors (`ReadError`, `WriteError`, ...) in a `RuntimeError`
  with the request id; these used to escape as raw httpx errors
- `N8nAdapter` client JSON can no longer override `payload_template` keys or
  the context fields the adapter sets: precedence is now `default_inputs` <
  client JSON (or `input_mapper` output) < `payload_template` < context
  fields. `default_inputs` stay overridable. JSON messages now also carry the
  context id (`metadata.context_id`, or `context_id` with a template), merged
  into any client `metadata` object (a non-object client `metadata` is
  replaced, with a warning). A client `metadata.context_id` is overwritten
  with the real one unless the template sets `metadata`, and
  `payload_template` keys now win over `default_inputs` for plain-text
  messages too
- `N8nAdapter` multimodal responses skip file and image refs with no `url`,
  `uri` or base64 `data` (refs with `data` become inline file Parts) and
  treat a `null` or non-list `files`/`images` field as empty; these used to
  produce empty `{}` Parts or raise `TypeError`
- Non-streaming adapters no longer put empty `{}` Parts on the wire:
  `invoke()` results are filtered the same way as streamed chunks, a result
  with no content emits no artifact, and an empty text result completes with
  `[Empty response]` instead of an empty text Part
- Cancelling an `OpenClawAdapter` task now ends it as canceled; the killed
  child's exit code used to surface as a failed task
- Concurrent `OpenClawAdapter` tasks each track their own subprocess, so
  `tasks/cancel` kills the right one; a single shared slot used to be
  overwritten, so cancel could kill another task's process or none
- `OpenClawAdapter` prompts larger than the OS argument limit (128 KiB per
  argument on Linux) no longer fail with `Argument list too long`
- The legacy `OpenClawAgentAdapter` async mode now kills the `openclaw`
  process when a task times out, instead of leaving it
  running, and its push notifications serialize the task with
  `MessageToDict`; they used to fail on a missing `model_dump()`
- An `AttributeError` raised inside a CrewAI crew no longer runs the crew a
  second time through the sync `kickoff()` fallback (`CrewAIAdapter` and
  the legacy `CrewAIAgentAdapter`); the fallback is now chosen by checking
  for `kickoff_async` up front, and runs `kickoff()` in a thread
- LangChain and LangGraph message content given as a list of content blocks
  (`{"type": "text", "text": ...}`, as newer chat models return) is now read
  as text. LangGraph replies and LangChain stream chunks in that form used to
  come out empty or as the list's `repr`
- `CallableAdapter`, `LangChainAdapter` and `CrewAIAdapter` results with values
  JSON cannot encode (datetimes, pydantic models, messages) are serialized
  with `str()` instead of failing with `TypeError`

## [0.2.15] - 2026-11-01

### Added

- CLI `--skill` option (JSON, repeatable) to describe Agent Card skills; each needs a non-empty `tags`
  list, which A2A requires

### Changed

- CLI Agent Cards list one skill built from the card name and description by
  default, so they are A2A spec-compliant (a2a-sdk no longer warns that the
  required `skills` field is empty)

### Fixed

- Claude Code prompts are now sent on stdin instead of as a `-p` argument, so
  large prompts (over 128 KiB, e.g. big diffs) no longer fail with
  `OSError: [Errno 7] Argument list too long`. CLI adapters can opt in by
  setting the new `CommandResult.stdin` field

## [0.2.14] - 2026-10-16

### Changed

- CLI Agent Cards now default to `<Agent> (<folder>)`, such as
  `Claude Code (my-app)`, with a matching description; `--name` and
  `--description` still override them
- `ClaudeCodeAdapter`, `CodexAdapter`, `PiAdapter`, and `OpenClawAdapter`
  default their Agent Card names to the product name (`Claude Code`, `Codex`,
  `Pi`, `OpenClaw`) instead of the Python class name
- README and Quick Start reorganized around the two ways to use the package:
  the Python SDK and the CLI, with `uv tool install` / `pipx` for the CLI
- README and Quick Start explain configuring CLI agents through their own
  config files and native flags after `--`

### Fixed

- Codex and Claude Code failures now report the CLI's own error message (for
  example an unsupported model) instead of unrelated stderr warnings

## [0.2.13] - 2026-08-09

### Added

- Thin `a2a-adapter` command-line entry point for starting Pi, Codex, Claude,
  or OpenClaw as an A2A server from the current directory
- `-help`, `-h`, and `--help` usage output for the root command and every
  supported CLI agent
- Native agent argument passthrough after `--`, with adapter-owned protocol
  and session options protected from overrides
- `PiAdapter` for exposing one persistent Pi RPC process and durable session as
  a streaming A2A agent
- Configurable Pi source-entrypoint commands, graceful abort, context binding,
  and accepted-prompt crash protection

### Changed

- `PiAdapter`, `CodexAdapter`, `ClaudeCodeAdapter`, and `OpenClawAdapter` now
  accept argument sequences used by the thin CLI entry point
- Documentation now distinguishes the CLI use case from the Python SDK use case
- Hermes remains available through the Python SDK and is not part of the v1 CLI

## [0.2.12] - 2026-05-28

### Fixed

- Increase subprocess `StreamReader` limit to 10 MB in `base_adapter.py` to prevent `LimitOverrunError` on large output lines

### Changed

- Remove stale development plan and tracking documents (`PLAN-hermes-a2a-adapter.md`, `openclaw/TRACKING.md`)

## [0.2.11] - 2026-05-04

### Fixed

- Fix Part emptiness check for missing oneof content variant

### Added

- Regression tests for Part emptiness check edge cases
- Documentation for Hermes adapter and aligned HermesAdapter example

## [0.2.10] - 2026-04-22

### Changed

- Bump version to 0.2.10

## [0.2.9] - 2026-04-22

### Breaking Changes

- Upgraded `a2a-sdk` dependency to `>=1,<2` (A2A SDK V1.0)

### Changed

- Full A2A SDK V1.0 compatibility sweep across all adapters and core infrastructure
- Migrated all adapters (Callable, CrewAI, LangChain, LangGraph, N8n, OpenClaw) to V1.0 imports and types
- Migrated core infrastructure (`adapter.py`, `client.py`, `base_adapter.py`, `executor.py`) to V1.0
- N8n adapter multimodal rewrite for V1.0 proto Part fields
- Symbol-level `ImportError` guards in `client.py` for narrower dependency checks

### Fixed

- OpenClaw adapter now handles silent completions and explicit failures in responses
- All test suites migrated to A2A SDK V1.0 types (`Part`, `TextPart`, `AgentCard.url`, async `event_queue`)

### Removed

- Development docs cleaned up

## [0.2.8] - 2026-04-20

### Added

- `HermesAdapter` — adapter for Hermes agents with streaming support, run failure propagation, and same-session request serialization
- `ClaudeCodeAdapter` and `CodexAdapter` — adapters for Claude Code and Codex CLI agents with opt-in constructor flags
- 60 new unit tests for core adapter modules
- Hermes agent adapter test suite

### Fixed

- Fix malformed `TextPart` from empty streaming chunks
- Hermes adapter: close streaming artifacts on error, propagate run failures to A2A task state

## [0.2.7] - 2026-03-26

### Fixed

- Downgrade "Executing OpenClaw command" log from `INFO` to `DEBUG` to reduce noise when running agents via `hybro-hub agent start`
- Fix `OllamaClient._get_client()` mock stomping: guard `is_closed` check with `isinstance(httpx.AsyncClient)` so injected test mocks are not overwritten by a real client, fixing 5 unit tests that were hitting a live Ollama instance

## [0.2.6] - 2026-03-23

### Fixed

- Handle OpenClaw outputting JSON to stderr instead of stdout - now checks both streams
- Fix tests for cancel/close that were missing async mock for `proc.wait()`
- Improve error message when OpenClaw command returns empty output to include stderr for debugging

## [0.2.5] - 2026-03-23

### Fixed

- Remove the wrong package with name a2a. Only a2a-sdk is the right A2A sdk to keep.

## [0.2.4] - 2026-03-17

### Fixed

- Close adapter and HTTP clients on shutdown via Starlette lifespan — `adapter.close()` is now called when uvicorn shuts down
- Recreate closed HTTP clients transparently in OllamaClient, N8nAdapter, and OpenClawAdapter
- Prevent zombie processes by awaiting `proc.wait()` after `proc.kill()` in OpenClawAdapter

## [0.2.3] - 2026-03-13

### Changed

- Correct default llama model name, and error out when wrong model name is provided

## [0.2.2] - 2026-03-13

### Added

- GitHub Actions workflow for automated PyPI publishing via trusted publishers
- Multimodality support for N8nAdapter (file and image handling)

### Changed

- N8nAdapter now uses `isinstance` for FilePart check and filters file/image fields from text
- N8nAdapter reuses HTTP client for better performance
- N8nAdapter returns consistent `list[Part]` in multimodal mode
- MIME types are now configurable in N8nAdapter

### Fixed

- Improved Ollama error messages with model name and parsed error detail
- Use consistent `artifact_id` for streaming chunks in executor

### Refactored

- Extracted `_to_parts()` helper in executor for cleaner code

## [0.2.1] - 2026-03-08

### Added

- `OllamaClient` — standalone HTTP client for the Ollama API (`/api/chat`), with streaming support
- `OllamaAdapter` now accepts an `OllamaClient` instance, consistent with how `LangChainAdapter` accepts a runnable and `LangGraphAdapter` accepts a graph
- `OllamaClient` exported from `a2a_adapter` and `a2a_adapter.integrations`
- Convenience constructor preserved: `OllamaAdapter(model="...")` still works for simple cases

### Changed

- `OllamaAdapter.get_metadata()` no longer leaks the model name into AgentCard `name`/`description` defaults

### Refactored

- Separated Ollama HTTP client concerns from the A2A adapter layer for cleaner architecture

## [0.2.0] - 2026-02-09

### Breaking Changes

v0.1 API still works but is deprecated.

- New simplified adapter interface: `BaseA2AAdapter` with single `invoke()` method
- New server functions: `serve_agent(adapter)` and `to_a2a(adapter)` replace manual AgentCard + serve pattern
- New flat imports: `from a2a_adapter import N8nAdapter, serve_agent`

### Added

- `BaseA2AAdapter` — new abstract base with `invoke()` / `stream()` / `cancel()` / `close()`
- `AdapterMetadata` — dataclass for automatic AgentCard generation
- `AdapterAgentExecutor` — bridge layer connecting adapters to A2A SDK
- `server.py` — `to_a2a()`, `serve_agent()`, `build_agent_card()` utilities
- `load_adapter()` — new sync factory function with registry pattern
- `register_adapter()` — decorator for third-party adapter registration
- 6 new v0.2 adapter classes: `N8nAdapter`, `CallableAdapter`, `LangChainAdapter`, `LangGraphAdapter`, `CrewAIAdapter`, `OpenClawAdapter`
- Full streaming support for LangChain and LangGraph adapters
- Lazy imports for all adapter classes (optional deps only loaded on use)
- 117 new unit tests for v0.2 components

### Deprecated

- `BaseAgentAdapter` (v0.1) — use `BaseA2AAdapter`
- `load_a2a_agent()` — use `load_adapter()`
- `build_agent_app()` — use `to_a2a()`
- `client.py` — use `server.py`
- All `*AgentAdapter` classes — use new shorter names (e.g. `N8nAdapter`)

### Design

- SDK-First: delegates task management, SSE streaming, push notifications to A2A SDK
- ~85% code reduction in adapter implementations
- See [`docs/archive/DESIGN_V0.2.md`](docs/archive/DESIGN_V0.2.md) for the historical architecture rationale

## [0.1.6] - 2026-02-05

### Added

- Timeout & input mapper for adapters

### Fixed

- README fix

## [0.1.5] - 2026-02-04

### Added

- OpenClaw adapter
- Push notification support
- More unit tests

## [0.1.4] - 2026-01-22

### Added

- Init adapter for LangGraph
- Task support for CrewAI adapter
- More unit tests

## [0.1.3] - 2025-12-31

### Added

- CHANGELOG

### Fixed

- Broken PyPI links
