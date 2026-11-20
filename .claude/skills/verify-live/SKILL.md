---
name: verify-live
description: Manual only. Runs the core A2A flows (card, send, stream, cancel, timeout, errors, sessions) against one real installed agent CLI served by a2a-adapter, and reports where the real agent differs from what the adapter expects.
argument-hint: <adapter> [port]
disable-model-invocation: true
---

# verify-live

Runs this checkout's `a2a-adapter <adapter>` against the **real** agent CLI
installed on this machine and checks the core A2A flows end to end.

**This uses the owner's real agent account and can spend tokens.** It runs
only when a person invokes it (`/verify-live <adapter>`), never on a model's
own initiative, never in CI, and never as part of the default `verify` skill.
A full run sends about seven small turns to the model (two of them are cut
off on purpose). `error-flag` should be rejected before any model call;
`error-model` may make one API request that the provider rejects.

## Run

```bash
uv run python .claude/skills/verify-live/verify_live.py <adapter> [port]
```

- `<adapter>`: `claude`, `codex`, `pi`, `openclaw` or `dsh`.
- `[port]`: a fixed localhost port for every server the run starts (default: a free one).
- `--model M`: ask for a model (default: the CLI's own default, reported as such).
- `--only send,stream,...`: run a subset of `card, send, stream, cancel, timeout, error, session`.
- `-- <native args>`: passed to the agent after `--` on every server.
- `dsh` needs its SDK: `uv run --extra deepseek-harness python .claude/skills/verify-live/verify_live.py dsh`.

Not supported: `hermes` (in-process gateway) and `ollama` (HTTP client); the
`a2a-adapter` CLI has no subcommand for them.

What the driver does: checks the CLI is on `PATH` and records its
`--version`; makes a throwaway workspace (a fresh `git init`) under the system
temp dir; refuses a `[port]` that something already listens on; starts
`python -m a2a_adapter.cli <adapter>` on `127.0.0.1` with a unique card name
(startup and the `card` check confirm it is the server that answers) and with
`A2A_AUTH_TOKEN`, `A2A_CLAUDE_SKIP_PERMISSIONS` and `A2A_CODEX_BYPASS_APPROVALS`
removed from its environment; runs the checks; stops every server with
SIGTERM, then SIGINT (the server's Ctrl+C path, which kills its CLIs: SIGTERM
waits for in-flight requests with no limit), then SIGKILL, 10 s apart, also on
Ctrl+C; kills any process a check left
behind; and deletes the workspace and the adapter's session store, keeping
only the server logs (`--keep` keeps everything).

## Checks

| Check | Expected | How |
|---|---|---|
| `card` | Agent Card served; `capabilities.streaming` matches the adapter | `GET /.well-known/agent-card.json` |
| `send` | `TASK_STATE_COMPLETED` with a real reply (`[Empty response]` fails) | `SendMessage` "Reply with the single word ok." |
| `stream` | artifact chunks with 1..5 exactly once, in order; exactly one terminal status, last, `COMPLETED` | `SendStreamingMessage` "Count from 1 to 5…" |
| `session` | a second message in the same context recalls a code word | two `SendMessage`s, one context |
| `cancel` | `TASK_STATE_CANCELED`, and no process from the run alive afterwards | agent runs `sleep <unique>`; `CancelTask` once it is seen |
| `session-restart` | the context still recalls the code word after a server restart | stop, start again, ask |
| `timeout` | `TASK_STATE_FAILED` "timed out" near the limit; nothing left alive | server `--timeout 30`, agent runs `sleep <unique>` |
| `error-flag` | `TASK_STATE_FAILED` with a message within 45 s, not a timeout | native `--verify-live-no-such-flag`, server `--timeout 60` |
| `error-model` | `TASK_STATE_FAILED` with a message within 45 s, not a timeout | `--model verify-live-no-such-model`, server `--timeout 60` |
| `shutdown` | stopping each server stops every process it started | process tree before and after |

The leftover check finds the agent's `sleep` by its unique duration in every
process's command line, so it catches a tool process that moved into its own
session or was reparented to init, not only the CLI's process group. For
one-shot CLIs (claude, codex, openclaw) it also requires that no new child
of the server survives the cancel or timeout.

### Per adapter

| Adapter | Differences and skips (from the adapter code) |
|---|---|
| `claude` | streams; driver adds `--allowedTools=Bash(sleep:*)` so `claude -p` may run the `sleep` (nothing else is allowed) |
| `codex` | `streaming=false`: `stream` passes when the server rejects it with `-32004`; `cancel` uses `SendMessage` with `returnImmediately` |
| `pi` | streams; one Pi session per server (`PiAdapter` binds to its first context), so every check uses one context; driver sets `--session-id`/`--session-dir` inside the run; Pi's RPC process stays up between turns, so only the `sleep` marker is checked after cancel and timeout |
| `openclaw` | `streaming=false` as for codex; `error-model` skipped (no confirmed model flag); driver passes `--session-id`, because the default is random per server and a restart would start a new session |
| `dsh` | streams; native args are rejected, so `error-flag` is skipped and `--model` is an `a2a-adapter dsh` option; with `$DSH_HOME` set, the owner's dsh home is used and the run's sessions stay there, otherwise a throwaway per-workspace home is used and deleted; runtime stays up, so only the marker is checked |

## Reading the report

One line per check, then a findings table:

```
PASS  card             name='Claude Code (workspace)' streaming=True
FAIL  cancel           1 left alive (killed by the driver)
SKIP  error-model      no confirmed model flag for openclaw; not guessed

| flow | expected | observed | evidence |
| cancel | no process from the run alive after cancel | 1 left alive | pid 26412 pgid 26412: sleep 689.5550 |
```

- **PASS**: the real agent did what the adapter expects.
- **FAIL**: a difference. `evidence` holds the A2A reply or the server log row
  (redacted, truncated); a leftover lists its pid, pgid and command. Full
  server logs are at the `logs` path printed on top.
- **SKIP**: the adapter does not support the flow, or it was not selected.
- A `cancel` FAIL saying the command was never seen means the agent did not
  run `sleep` (refused, or lacks shell permission); the cancel itself was still
  checked. A `timeout` PASS noting the sleep had not started means the limit
  hit before the tool call; rerun with a larger `--short-timeout` to cover the
  process kill.
- Model wording (for example not answering exactly "ok") is noted, not failed.
- The error checks see the generic A2A message by design (exception text stays
  off the wire); the line shows what the CLI said, from the server log.
- Exit status: 0 no failures, 1 failures, 2 could not run (CLI missing, port
  busy, server did not start, `ps` unusable), 130 interrupted.

## Ratchet rule

Every difference a live run finds becomes a new mode or fixture in the default
`verify` skill's fake agent (`.claude/skills/verify/`) or a unit test, **written
before the fix** and seen failing, so the stubbed check catches it next time
without a real agent. Then fix the adapter. A live finding is not done until
that stubbed check exists. Record the CLI version from the report next to the
new fixture or test, as the adapters do for patterns found in a given release.
