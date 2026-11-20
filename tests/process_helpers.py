"""Real-process helpers for process-tree tests (POSIX only).

The CLI used here is ``sh``: it starts a ``sleep 30`` grandchild that
inherits the pipes, writes "<sh pid> <grandchild pid>" to ``$PIDFILE`` and
waits. Killing only ``sh`` leaves the grandchild alive and holding the pipes.
"""

import asyncio
import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from a2a_adapter.base_adapter import BaseA2AAdapter, CommandResult, ParseResult, StreamEvent

WRITE_PIDS = 'echo "$$ $!" > "$PIDFILE.tmp" && mv "$PIDFILE.tmp" "$PIDFILE"'
GRANDCHILD_SCRIPT = f"sleep 30 & {WRITE_PIDS}; wait"
STREAMING_GRANDCHILD_SCRIPT = f"echo started; sleep 30 & {WRITE_PIDS}; wait"
# The CLI exits on its own and leaves the grandchild holding the pipes.
LEADER_EXITS_SCRIPT = f"sleep 30 & {WRITE_PIDS}; exit 0"
STREAMING_LEADER_EXITS_SCRIPT = f"echo started; sleep 30 & {WRITE_PIDS}; exit 0"


class TreeCliAdapter(BaseA2AAdapter):
    """CLI adapter whose command is a shell script with a grandchild."""

    def __init__(self, tmp_path: Path, *, script: str = GRANDCHILD_SCRIPT, timeout: float = 60):
        super().__init__(
            working_dir=str(tmp_path),
            timeout=timeout,
            env_vars={"PIDFILE": str(pidfile(tmp_path))},
        )
        self._script = script

    @staticmethod
    def _task_id(kwargs) -> str:
        context = kwargs.get("context")
        if context is not None and getattr(context, "task_id", None):
            return context.task_id
        return kwargs.get("task_id", "task-1")

    async def invoke(self, user_input, context_id=None, **kwargs):
        return await self._run_subprocess(
            context_id or "ctx", self._task_id(kwargs), user_input=user_input
        )

    async def stream(self, user_input, context_id=None, **kwargs):
        async for chunk in self._run_subprocess_streaming(
            context_id or "ctx", self._task_id(kwargs), user_input=user_input
        ):
            yield chunk

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        return CommandResult(args=["sh", "-c", self._script], used_resume=False)

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        return ParseResult(text=stdout_text.strip())

    def _handle_stream_line(self, line: str) -> StreamEvent | None:
        return StreamEvent(text=line)


def pidfile(tmp_path: Path) -> Path:
    return tmp_path / "pids"


async def wait_for_pids(tmp_path: Path, timeout: float = 5) -> tuple[int, int]:
    """Wait until the script wrote its PIDs; return (sh pid, grandchild pid)."""
    path = pidfile(tmp_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            sh_pid, grandchild_pid = path.read_text().split()
            return int(sh_pid), int(grandchild_pid)
        await asyncio.sleep(0.02)
    raise AssertionError("the CLI never wrote its PID file")


def _is_zombie(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return result.stdout.strip().startswith("Z")


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # reused by another user's process: ours is gone
        return False
    return not _is_zombie(pid)


async def wait_dead(pid: int, timeout: float = 5) -> bool:
    """Return True once ``pid`` is gone (or a zombie), False after ``timeout``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not is_alive(pid)


def kill_pids(tmp_path: Path) -> None:
    """Cleanup: SIGKILL the PIDs from the PID file. Safe: the test spawned them."""
    path = pidfile(tmp_path)
    if not path.exists():
        return
    for pid in (int(p) for p in path.read_text().split()):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


async def cancel_and_wait(task: asyncio.Task | None, timeout: float = 5) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await asyncio.wait_for(task, timeout)


# A CLI that starts its grandchild in a session of its own (as agy's terminal
# tool does), so killing the CLI's process group does not reach it. It writes
# "<cli pid> <grandchild pid>" to $PIDFILE; with "stream" it first prints a line.
# With "member", the grandchild is a detached helper whose own child sits in a
# group whose leader has exited: in no group the sweep kills by group, and
# $PIDFILE then names that child. With "loop", the CLI keeps starting
# detached children, each of which appends its own PID to $PIDFILE.all.
DETACHED_GRANDCHILD_CODE = """
import os, subprocess, sys, time
null = subprocess.DEVNULL
mode = sys.argv[1:]
if "stream" in mode:
    print("started", flush=True)
path = os.environ["PIDFILE"]

def sleeper(**kwargs):
    return subprocess.Popen(["sleep", "30"], stdin=null, stdout=null, stderr=null, **kwargs)

def publish(pid):
    with open(path + ".tmp", "w") as f:
        f.write(f"{os.getpid()} {pid}")
    os.replace(path + ".tmp", path)

if "member" in mode:
    helper = subprocess.Popen([sys.executable, "-c", MEMBER, path], start_new_session=True)
    while not os.path.exists(path + ".member"):
        time.sleep(0.01)
    publish(int(open(path + ".member").read()))
elif "loop" in mode:
    # Each tool writes its own PID, so one started just before the kill is seen.
    tool = ["sh", "-c", 'echo $$ >> "$0"; exec sleep 30', path + ".all"]
    def start_tool():
        return subprocess.Popen(tool, stdin=null, stdout=null, stderr=null,
                                start_new_session=True)
    publish(start_tool().pid)
    while True:
        start_tool()
        time.sleep(0.002)
else:
    publish(sleeper(start_new_session=True).pid)
time.sleep(30)
""".replace(
    "MEMBER",
    repr(
        "import os, subprocess, sys, time\n"
        "null = subprocess.DEVNULL\n"
        "cmd = ['sleep', '30']\n"
        "leader = subprocess.Popen(cmd, stdin=null, stdout=null, stderr=null, process_group=0)\n"
        "member = subprocess.Popen(cmd, stdin=null, stdout=null, stderr=null,\n"
        "                          process_group=leader.pid)\n"
        "leader.kill(); leader.wait()\n"
        "open(sys.argv[1] + '.member', 'w').write(str(member.pid))\n"
        "time.sleep(30)\n"
    ),
)


class DetachedCliAdapter(TreeCliAdapter):
    """CLI adapter whose grandchild runs in its own session (its own process group)."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        sweep: bool,
        streaming: bool = False,
        timeout: float = 60,
        mode: str | None = None,
    ):
        super().__init__(tmp_path, timeout=timeout)
        self._kill_detached_descendants = sweep
        self._streaming = streaming
        self._mode = mode

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        import sys

        args = [sys.executable, "-c", DETACHED_GRANDCHILD_CODE]
        if self._streaming:
            args.append("stream")
        if self._mode:
            args.append(self._mode)
        return CommandResult(args=args, used_resume=False)


def all_loop_pids(tmp_path: Path) -> list[int]:
    """PIDs the "loop" CLI appended to $PIDFILE.all."""
    path = Path(f"{pidfile(tmp_path)}.all")
    return [int(p) for p in path.read_text().split()] if path.exists() else []


def kill_all_pids(tmp_path: Path) -> None:
    """Cleanup for the "member" and "loop" modes too. Safe: the test spawned them."""
    kill_pids(tmp_path)
    for pid in all_loop_pids(tmp_path):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
