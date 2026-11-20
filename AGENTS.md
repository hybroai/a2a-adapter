# AGENTS.md

Instructions for coding agents working in this repository.

## Repository scope

- This repo is the `a2a-adapter` Python package: it serves agent frameworks and local coding-agent CLIs as A2A Protocol servers. Adapters turn text into text; the `a2a-sdk` handles the protocol.
- Main modules in `a2a_adapter/`: `base_adapter.py` (`BaseA2AAdapter`, the shared CLI subprocess runtime, session store), `executor.py` (`AdapterAgentExecutor`, the bridge to the SDK), `server.py` (`to_a2a`, `serve_agent`, `build_agent_card`, auth and push notifications), `loader.py` (`load_adapter`, `register_adapter`), `cli.py` (the `a2a-adapter` console script), `terminal.py` (terminal display), `exceptions.py`, and one module per adapter in `integrations/`.
- `adapter.py`, `client.py` and the `*AgentAdapter` classes are the deprecated v0.1 API. Keep them working; do not extend them.
- Read `ARCHITECTURE.md` for the layers, request flow, cancel semantics and process-group handling before changing `executor.py`, `server.py` or the subprocess runtime.

## Commands

Run these from the repo root; they match `.github/workflows/ci.yml`.

- Install: `uv sync --locked --extra dev --extra all` (the CI test job; the lint job uses `--extra dev` only. `--locked` fails if `uv.lock` is out of date with `pyproject.toml`; run `uv lock` after changing dependencies).
- Lint: `uv run ruff check .`
- Format check: `uv run black --check .` (line length 100).
- Tests: `uv run pytest` (runs everything under `tests/`). CI runs the tests on Python 3.11, 3.12 and 3.13 and lint on 3.12.
- Run a single file with `uv run pytest tests/unit/test_v02_n8n.py`.
- Before committing a change to the runtime (`server.py`, `executor.py`, `base_adapter.py`, `adapter.py`, `cli.py` or a CLI-backed integration), run the verify skill: `uv run python .claude/skills/verify/verify.py` (see `.claude/skills/verify/SKILL.md`). It drives the real server through the Claude Code and Codex adapters only. A failure means fixing the code; change a check only when the behaviour is meant to change, in the same commit.
- mypy is configured in `pyproject.toml` but CI does not run it; do not treat a mypy result as a CI gate.
- `deepseek-harness` is not in `[all]`, so CI runs its tests without the SDK installed. Keep those tests passing without it.

## Code rules

- A new adapter subclasses `BaseA2AAdapter` in `a2a_adapter/integrations/<name>.py` and implements `invoke()`; add `stream()`, `cancel()`, `close()` and `get_metadata()` as needed.
- Register a new adapter in all three places: `_BUILTIN_MAP` in `a2a_adapter/loader.py`, the lazy exports in `a2a_adapter/integrations/__init__.py`, and the lazy exports in `a2a_adapter/__init__.py`. Add an optional extra in `pyproject.toml` only if it needs a third-party package.
- CLI-backed adapters build a `CommandResult` and run it through `_run_subprocess` / `_run_subprocess_streaming`. Do not call `asyncio.create_subprocess_exec` directly in an adapter: the base runtime starts the CLI in its own session and kills its process group on cancel, timeout, close and shutdown.
- Send the user's prompt to a CLI on stdin (`CommandResult(stdin=...)`), not as an argument.
- Bound framework streams with `iter_with_timeout` from `integrations/_framework.py`; raise `AgentTimeoutError` for an adapter's own deadline.
- Exception text does not reach clients unless the exception is a `UserFacingError` or the server runs with `expose_error_details=True`. Never put URLs, paths, tokens or tool output in a `UserFacingError` message.
- When a change alters behaviour described in `ARCHITECTURE.md`, `README.md` or `.claude/skills/a2a-adapter/`, update those files in the same change.

## Test rules

- Unit tests (`tests/unit/`, plus `tests/test_*.py`) must not need a real agent CLI, model, API key or network. Mock HTTP with `httpx.MockTransport`, patch `asyncio.create_subprocess_exec`, or use a fake CLI started with `sys.executable` or `sh` (see `tests/process_helpers.py`). Shared stub adapters are in `tests/conftest.py`.
- Integration tests (`tests/integration/`) run the real ASGI app in-process through `httpx.ASGITransport` with stub or in-process adapters (for example a real LangGraph graph with an in-memory checkpointer). They cover the full A2A request flow; they do not start a network server or a real agent CLI.
- `tests/e2e_live_test.py` drives the real `claude` and `codex` CLIs. It is not collected by pytest and is not part of CI.
- `.claude/skills/verify-live/` runs the core A2A flows against one real installed agent. It is manual-only: run it only when a person asks for live testing.
- Never use real provider API keys, paid models, live agent CLIs or external services unless the user asks for live testing.
- A bug fix comes with a test that fails without the fix.

## Git and release

- Commit subjects follow Conventional Commits with a scope where one fits: `fix(executor): ...`, `feat(langgraph): ...`, `docs: ...`, `chore: ...`. release-please reads them to choose the version bump, so the type must be accurate.
- Releases are automated: `release-please.yml` opens the release PR, which bumps `pyproject.toml`, `a2a_adapter/__init__.py`, `.release-please-manifest.json` and `CHANGELOG.md`; merging it tags `v<version>` and dispatches `publish.yml`. Never edit the version or `CHANGELOG.md` by hand.
- Do not commit, push or open a PR unless the user asks.

## Code Review Rules

Each rule names the trigger in the diff, what to check, and the severity. These come from bugs fixed in this repo; commits are listed in parentheses.

1. User text reaches a CLI's argv (a prompt or message appended to a command list). Check it goes on stdin, or as a single `--flag=value` argument counted against the 128 KiB argv limit; a bare positional can be read as a flag and can exceed the limit. Must fix. (a5c4742, e1fa369, ba5e08d on `Jerome/antigravity-adapter`)
2. An adapter parses a CLI, HTTP or stream result. Check that an error record in a zero exit, a 2xx body or a stream line fails the task, that an empty result is not reported as success, and that a failed step (file fetch, input mapper) raises instead of being logged and skipped. Must fix. (c8588ae, d148750, c0c51b0, 9ad15f1, edff578, bfdf535)
3. A timeout on a read, `readline()` or stream loop. Check the deadline is computed once for the whole run and checked between chunks, including time the consumer or a suspended generator spends; a per-read timeout lets a chatty CLI run forever. Must fix. (73bc199, 0222f4f, a6eb02d, c26e96e)
4. A new subprocess spawn, a `proc.kill()`, a background thread, or a change to `cancel()` / `close()`. Check that cancel, timeout and close stop everything the run started: the CLI's whole process group (through the base runtime), worker threads (interrupt the agent) and remote runs. Must fix. (86e30f9, b015e74, ce776b7, 60913ae, be6042f on `Jerome/antigravity-adapter`)
5. A change in `executor.py` that enqueues a terminal state, notifies the observer, or reads a stream during cancel. Check each run sends exactly one terminal state, observer notifications follow a successful enqueue, and nothing is sent after cancel owns the terminal state. Must fix. (fd597e6, 0516a44, b811f94, e0728e9, 0878516, 512a408, e5a3057, 8b00725)
6. A retry loop or a resume-and-retry path. Check it retries only when the work cannot have started (connect failure, no output, no tool run); a 5xx, read timeout, dropped connection or partly streamed run must not re-run a workflow or turn by default. Must fix. (05142e7, b6a8f92, 3a892bd, ba5e08d on `Jerome/antigravity-adapter`)
7. A new dict, set or file keyed by `context_id`, `task_id` or session. Check entries are removed when the task or lock ends, or the collection is capped. Must fix. (c1aa685, 19761bf, 18bc6b4, 5ea4a3c, eb33770)
8. Code that derives a session name or key from `context_id`, or reuses one framework object across requests. Check two contexts can never map to one session (no lossy lowercasing or truncation without a digest) and concurrent requests do not share mutable state. Must fix. (7b35f17, 34d42ad)
9. Client-supplied data (message JSON, metadata, file URLs, push URLs) used in a payload or a server-side request. Check it cannot override operator config (`payload_template`, context fields) or make the server fetch an arbitrary or private host. Must fix. (b47d952, bfdf535, f47b9eb)
10. Output extraction changed in only one of `invoke()` and `stream()`, or a new framework output shape. Check both paths accept the same keys and message shapes, and that a final response is still returned when nothing was streamed. Should fix. (a60bd21, 8776135, eea59c1, 7432930)
11. A write to a session store or other state file. Check it writes a temp file in the same directory, fsyncs it and `os.replace`s it, and that a corrupt file is logged and ignored instead of crashing. Should fix. (4a5e078, 7e5aa90)
12. A behaviour change in `executor.py`, `server.py` or the subprocess runtime with no matching edit to `ARCHITECTURE.md`. Should fix. (the fixes 73bc199, 3a892bd, eb33770 and c1aa685 each updated it)

Do not flag:

- Anything Ruff or Black enforces; CI checks it.
- Missing mypy types; CI does not run mypy.
- The deprecated v0.1 API's style or design, unless the diff changes it.
- Issues in code the diff does not touch. If worth mentioning, label them pre-existing and keep them out of the findings.
