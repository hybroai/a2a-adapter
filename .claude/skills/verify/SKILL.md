---
name: verify
description: End-to-end smoke check of the a2a-adapter runtime. Starts the real a2a-adapter server with a fake CLI agent and sends real A2A requests over localhost HTTP (send, stream, cancel, timeout, crash, auth, sessions, input-required). Run before committing any change under a2a_adapter/ that touches the server, executor, adapter base, CLI or a CLI-backed integration.
---

# verify

Unit tests mock the subprocess and the transport. This check runs the real
thing: `python -m a2a_adapter.cli claude|codex` on a free 127.0.0.1 port, with
`fake_agent.py` standing in for the agent CLI, driven by real A2A 1.0
JSON-RPC requests. No real CLIs, provider keys or network. POSIX only.

## When to run

Before committing any change under `a2a_adapter/` that touches `server.py`,
`executor.py`, `base_adapter.py`, `adapter.py`, `cli.py`, or a CLI-backed
integration. Run it in addition to `uv run pytest`, not instead of it.

Coverage: the server, executor and CLI, and the shared subprocess runtime in
`base_adapter.py` through the Claude Code (streaming) and Codex (invoke)
adapters. Other integrations (`openclaw.py`, `pi.py`, `dsh`, ...) do not run
here; their own parsing and process handling rely on unit tests.

## Run

From the repo root:

```bash
uv run python .claude/skills/verify/verify.py
```

About 20 to 25 seconds. `-k <name>` runs only matching scenarios
(`claude`, `codex`, `timeout`, `auth`, `input-required`).

## Read the output

One line per check: `PASS  <check> (<time>)` or `FAIL  <check> (<time>): <reason>`,
then `N passed, M failed in Ts`. Exit code 0 only if every check passed.

- `claude:` checks use the streaming subprocess path, `codex:` the invoke path.
- `scenario ran to the end` fails when the server did not start; the
  reason includes the tail of the server log.
- `no fake-agent process left alive` is the final sweep for leaked agent
  processes (a cancel, timeout or server shutdown that missed part of the tree).
- A server that does not stop on SIGINT within 5 s fails its scenario.
- `note: server took Ns to start` means the machine is loaded; it is not a failure.

## On a failure

The check describes behaviour the runtime promises (see `ARCHITECTURE.md`).
Fix the code under `a2a_adapter/` until it passes. Do not loosen a check,
raise a timeout, add a retry or skip a scenario to make it green. If the
behaviour itself is meant to change, update the check in the same commit
and say why in the commit message.

If a check fails only sometimes, treat it as a real bug (a race in the
runtime or in the harness) and find it; do not rerun until green.

## Files

- `verify.py`: the driver. One server per scenario, always torn down
  (SIGINT, then SIGKILL; also when the driver gets Ctrl+C, SIGTERM or SIGHUP),
  temp state dirs, fake-agent PIDs swept at the end. Servers import
  `a2a_adapter` from this checkout (`PYTHONPATH`), and HTTP ignores proxy env vars.
- `fake_agent.py`: speaks Claude Code stream-json or Codex `--json`
  depending on argv. The prompt picks the mode: `mode:reply|slow|hang|child|orphan|linger|drip|crash`
  and `tag:<name>` (the file it records its PIDs in).
