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
