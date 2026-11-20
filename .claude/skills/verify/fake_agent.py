"""Fake coding-agent CLI for the verify skill. Standard library only.

Stands in for ``claude`` or ``codex``: the adapters launch it through
``a2a-adapter <agent> --executable <wrapper>``. The dialect follows argv:
``exec ...`` is Codex (``codex exec --json [resume ID] -``), anything else is
Claude Code (``claude -p --output-format stream-json [--resume ID]``). The
prompt arrives on stdin, as with the real CLIs.

The prompt picks the behaviour: ``mode:<name> tag:<tag> <text>``.

  reply   one answer naming the session and turn (the default)
  slow    five answers, 0.1 s apart (Claude streams each one)
  hang    print "started" and "working", then sleep forever
  child   like hang, plus a child process in the same group
  orphan  print both, start a child that keeps stdout open, then exit 0
  linger  print both, close stdout, keep running (stderr stays open)
  drip    print "tick N" every 0.2 s forever (never quiet long enough for a per-line timeout)
  crash   write a secret marker to stderr and exit 3

Every run records its PIDs (its own, plus its child's) in
``$FAKE_AGENT_STATE/pids/<tag>.json`` before it blocks, so the driver can
check nothing outlives a cancel or timeout. Untagged runs use a unique tag.
Sessions live in ``$FAKE_AGENT_STATE/sessions/<id>`` (the turn count),
which is what a resumed run continues.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

STATE = Path(os.environ["FAKE_AGENT_STATE"])
SECRET_STDERR = "FAKE-AGENT-SECRET-STDERR"


def parse_prompt(prompt: str) -> tuple[str, str, str]:
    mode, tag, words = "reply", f"untagged-{os.getpid()}", []
    for word in prompt.split():
        if word.startswith("mode:"):
            mode = word[5:]
        elif word.startswith("tag:"):
            tag = word[4:]
        else:
            words.append(word)
    return mode, tag, " ".join(words)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def record_pids(tag: str, child: subprocess.Popen | None = None) -> None:
    pids = [os.getpid()] + ([child.pid] if child else [])
    write_json(STATE / "pids" / f"{tag}.json", pids)


def start_child() -> subprocess.Popen:
    # Inherits stdout and stderr, so it holds the CLI's pipes like an MCP server would.
    # The state path in argv lets the driver tell this child from a reused PID.
    sleeper = "import time; time.sleep(3600)"
    return subprocess.Popen([sys.executable, "-c", sleeper, str(STATE)])


class Dialect:
    def __init__(self, argv: list[str]) -> None:
        self.codex = bool(argv) and argv[0] == "exec"
        flag = "resume" if self.codex else "--resume"
        self.resume = argv[argv.index(flag) + 1] if flag in argv else None

    def emit(self, record: dict) -> None:
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()

    def start(self, session: str) -> None:
        if self.codex:
            self.emit({"type": "thread.started", "thread_id": session})

    def say(self, text: str) -> None:
        if self.codex:
            self.emit({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
        else:
            content = [{"type": "text", "text": text}]
            self.emit({"type": "assistant", "message": {"content": content}})

    def finish(self, session: str) -> None:
        if self.codex:
            self.emit({"type": "turn.completed"})
        else:
            self.emit({"type": "result", "is_error": False, "session_id": session})


def main() -> int:
    dialect = Dialect(sys.argv[1:])
    mode, tag, text = parse_prompt(sys.stdin.read())

    if dialect.resume:
        session = dialect.resume
        if not (STATE / "sessions" / session).exists():
            # Each CLI's own wording, which its adapter recognises as a stale session.
            if dialect.codex:
                print(f"Error: thread not found: {session}", file=sys.stderr)
            else:
                print(f"No conversation found with session ID: {session}", file=sys.stderr)
            return 1
    else:
        session = uuid.uuid4().hex[:12]
    turn_file = STATE / "sessions" / session
    turn = int(turn_file.read_text()) + 1 if turn_file.exists() else 1
    write_json(turn_file, turn)

    if mode not in ("hang", "child", "orphan", "linger"):
        record_pids(tag)  # the blocking modes record once their child exists
    if mode == "crash":
        print(
            f"Traceback (most recent call last):\n  {SECRET_STDERR} /home/u/.secret",
            file=sys.stderr,
        )
        return 3

    dialect.start(session)
    if mode == "reply":
        auth_env = "present" if "A2A_AUTH_TOKEN" in os.environ else "absent"
        dialect.say(f"echo={text} session={session} turn={turn} auth_env={auth_env}")
    elif mode == "slow":
        for i in range(1, 6):
            dialect.say(f"chunk {i}")
            time.sleep(0.1)
    elif mode == "drip":
        for i in range(1, 100_000):
            dialect.say(f"tick {i}")
            time.sleep(0.2)
    elif mode in ("hang", "child", "orphan", "linger"):
        # Two answers: the executor holds the latest chunk back until the next
        # one arrives, so the client sees "started" while the agent hangs.
        dialect.say("started")
        dialect.say("working")
        child = start_child() if mode in ("child", "orphan") else None
        record_pids(tag, child)
        if mode == "orphan":
            return 0  # the child keeps stdout open after the CLI exits
        if mode == "linger":
            sys.stdout.flush()
            os.close(1)  # EOF for the adapter; the process keeps running
        time.sleep(3600)
    else:
        print(f"fake_agent: unknown mode {mode!r}", file=sys.stderr)
        return 2
    dialect.finish(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
