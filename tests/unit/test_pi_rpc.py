"""Protocol tests for the persistent Pi JSONL RPC process."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import patch

import pytest

from a2a_adapter.integrations.pi import (
    PiCommandTimeoutError,
    PiProcessLostError,
    PiRpcProcess,
    PiTurnAbortedError,
)


class QueueReader:
    def __init__(self) -> None:
        self._lines: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._lines.get()

    async def read(self, _size: int = -1) -> bytes:
        return await self._lines.get()

    def feed_json(self, value: dict) -> None:
        self._lines.put_nowait((json.dumps(value) + "\n").encode())

    def feed_eof(self) -> None:
        self._lines.put_nowait(b"")


class CommandWriter:
    def __init__(self, on_command: Callable[[dict], None]) -> None:
        self._on_command = on_command

    def write(self, data: bytes) -> None:
        self._on_command(json.loads(data.decode()))

    async def drain(self) -> None:
        return None


class FakeProcess:
    def __init__(self, on_command: Callable[[dict], None]) -> None:
        self.stdout = QueueReader()
        self.stderr = QueueReader()
        self.stdin = CommandWriter(on_command)
        self.returncode: int | None = None
        self._exited = asyncio.Event()
        self.pid = 1234

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)

    def exit(self, code: int) -> None:
        if self.returncode is not None:
            return
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._exited.set()


def _rpc(tmp_path, *, timeout: float = 1) -> PiRpcProcess:
    return PiRpcProcess(
        command=["pi"],
        working_dir=str(tmp_path),
        session_id="session-1",
        session_dir=str(tmp_path / "sessions"),
        timeout=timeout,
        startup_timeout=timeout,
        cancel_grace_timeout=timeout,
    )


@pytest.mark.asyncio
async def test_prompt_streams_text_until_agent_settled(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
                }
            )
            process.stdout.feed_json({"type": "agent_end", "willRetry": True})
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": " world"},
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        chunks = [chunk async for chunk in rpc.prompt("hello")]

    assert chunks == ["hello", " world"]
    await rpc.close()


@pytest.mark.asyncio
async def test_process_loss_after_prompt_acceptance_is_not_retryable(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.exit(1)

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        with pytest.raises(PiProcessLostError) as exc_info:
            _ = [chunk async for chunk in rpc.prompt("do work")]

    assert exc_info.value.accepted is True


@pytest.mark.asyncio
async def test_abort_uses_rpc_and_waits_for_settled(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
        elif command["type"] == "abort":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "abort",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "error", "reason": "aborted"},
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        turn = asyncio.create_task(_collect(rpc.prompt("work")))
        await asyncio.sleep(0)
        await rpc.abort()
        with pytest.raises(PiTurnAbortedError, match="aborted"):
            await turn

    await rpc.close()


@pytest.mark.asyncio
async def test_blocking_extension_ui_is_cancelled_instead_of_hanging(tmp_path):
    process: FakeProcess
    extension_responses: list[dict] = []

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "extension_ui_request",
                    "id": "ui-1",
                    "method": "confirm",
                    "message": "Continue?",
                }
            )
        elif command["type"] == "extension_ui_response":
            extension_responses.append(command)
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "delta": "cancelled safely",
                    },
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        chunks = [chunk async for chunk in rpc.prompt("work")]

    assert chunks == ["cancelled safely"]
    assert extension_responses == [
        {"type": "extension_ui_response", "id": "ui-1", "cancelled": True}
    ]
    await rpc.close()


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_leave_pi_turn_running(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
        elif command["type"] == "abort":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "abort",
                    "success": True,
                }
            )

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        turn = asyncio.create_task(_collect(rpc.prompt("work")))
        await asyncio.sleep(0)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    assert process.returncode == -15
    assert rpc.is_running is False


@pytest.mark.asyncio
async def test_closed_stream_does_not_leave_pi_turn_running(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "partial"},
                }
            )
        elif command["type"] == "abort":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "abort",
                    "success": True,
                }
            )

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        stream = rpc.prompt("work")
        assert await anext(stream) == "partial"
        await stream.aclose()

    assert process.returncode == -15
    assert rpc.is_running is False


@pytest.mark.asyncio
async def test_prompt_acceptance_timeout_terminates_uncertain_process(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        # Intentionally never acknowledge the prompt. Pi may still have
        # accepted it, so keeping this shared process would be unsafe.

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path, timeout=0.01)

    with patch(
        "a2a_adapter.integrations.pi.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        with pytest.raises(PiCommandTimeoutError, match="'prompt' timed out"):
            await _collect(rpc.prompt("work"))

    assert process.returncode == -15
    assert rpc.is_running is False


async def _collect(iterator):
    return [item async for item in iterator]


# ── Process-tree kill (POSIX) ────────────────────────────────

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_posix_only = pytest.mark.skipif(_os.name != "posix", reason="process groups are POSIX-only")

# Starts a ``sleep 30`` grandchild that shares the pipes (other modes: see
# the branches below), writes "<leader pid>
# <grandchild pid>" to argv[1], optionally ignores SIGTERM, writes the ready
# file argv[2], then either sleeps or exits.
_PI_TREE_SCRIPT = r"""
import os, signal, subprocess, sys, time
null = subprocess.DEVNULL
if sys.argv[3] == "detached-exit":
    grandchild = subprocess.Popen(["sleep", "30"], stdin=null, stdout=null, stderr=null)
elif sys.argv[3] == "detached-release-exit":
    # Runs until the test creates argv[1] + ".release".
    wait = 'while [ ! -e "$0" ]; do sleep 0.05; done'
    grandchild = subprocess.Popen(
        ["sh", "-c", wait, sys.argv[1] + ".release"], stdin=null, stdout=null, stderr=null
    )
elif sys.argv[3] == "setsid-exit":
    # Leaves the group but keeps Pi's pipes, so wait() blocks while it runs.
    grandchild = subprocess.Popen(["sleep", "30"], start_new_session=True)
else:
    grandchild = subprocess.Popen(["sleep", "30"])
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(f"{os.getpid()} {grandchild.pid}")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
if sys.argv[3] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[3] == "term-graceful":
    def on_term(*_):
        time.sleep(0.3)  # a SIGKILL during this would leave no marker
        open(sys.argv[1] + ".termed", "w").close()
        os._exit(0)
    signal.signal(signal.SIGTERM, on_term)
open(sys.argv[2], "w").close()
if sys.argv[3] == "setsid-exit":
    time.sleep(0.5)  # let the test start its watchers before Pi exits
elif sys.argv[3] not in ("exit", "detached-exit", "detached-release-exit"):
    time.sleep(30)
"""


async def _spawn_pi_tree(tmp_path, mode: str):
    from asyncio.subprocess import PIPE, create_subprocess_exec

    from a2a_adapter.base_adapter import _spawn_cli

    pids, ready = tmp_path / "pids", tmp_path / "ready"
    proc = await _spawn_cli(
        create_subprocess_exec,
        _sys.executable,
        "-c",
        _PI_TREE_SCRIPT,
        str(pids),
        str(ready),
        mode,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not ready.exists():
        assert loop.time() < deadline, "child never became ready"
        await asyncio.sleep(0.02)
    leader, grandchild = (int(p) for p in pids.read_text().split())
    return proc, leader, grandchild


def _kill_test_pids(*pids: int) -> None:
    import signal

    for pid in pids:
        with __import__("contextlib").suppress(ProcessLookupError, PermissionError):
            _os.kill(pid, signal.SIGKILL)


async def _wait_leader_exit(proc, timeout: float = 5) -> None:
    # Not proc.wait(): it blocks while the grandchild holds the pipes.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while proc.returncode is None:
        assert loop.time() < deadline, "leader did not exit"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_spawn_starts_new_session(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)
    with patch("a2a_adapter.integrations.pi.create_subprocess_exec", return_value=process) as spawn:
        await rpc.start()
    try:
        if _os.name == "posix":
            assert spawn.call_args.kwargs.get("start_new_session") is True
    finally:
        await rpc.close()


@_posix_only
@pytest.mark.asyncio
async def test_terminate_signals_group_then_kills(tmp_path):
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "ignore-term")
    try:
        rpc = _rpc(tmp_path, timeout=0.2)
        rpc._process = proc
        await asyncio.wait_for(rpc._terminate_process(), 5)
        assert await wait_dead(grandchild), "grandchild survived _terminate_process()"
        assert await wait_dead(leader), "leader survived _terminate_process()"
    finally:
        _kill_test_pids(leader, grandchild)
        await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_terminate_kills_group_after_leader_died(tmp_path):
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "exit")
    try:
        await _wait_leader_exit(proc)
        rpc = _rpc(tmp_path, timeout=0.2)
        rpc._process = proc
        await asyncio.wait_for(rpc._terminate_process(), 5)
        assert await wait_dead(grandchild), "grandchild survived after the leader exited"
    finally:
        _kill_test_pids(leader, grandchild)
        await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_close_kills_group_after_leader_died(tmp_path):
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "exit")
    try:
        await _wait_leader_exit(proc)
        rpc = _rpc(tmp_path, timeout=0.2)
        rpc._process = proc
        await asyncio.wait_for(rpc.close(), 5)
        assert await wait_dead(grandchild), "grandchild survived rpc.close()"
    finally:
        _kill_test_pids(leader, grandchild)
        await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_adapter_kill_children_now_kills_pi_group(tmp_path):
    from a2a_adapter.integrations.pi import PiAdapter
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "sleep")
    try:
        rpc = _rpc(tmp_path, timeout=0.2)
        rpc._process = proc
        adapter = PiAdapter(working_dir=str(tmp_path))
        adapter._rpc = rpc
        adapter.kill_children_now()
        assert await wait_dead(grandchild), "grandchild survived kill_children_now()"
        assert await wait_dead(leader), "Pi survived kill_children_now()"
    finally:
        _kill_test_pids(leader, grandchild)
        await asyncio.wait_for(proc.wait(), 5)


async def _start_watchers(rpc, proc) -> None:
    """Start the RPC's exit and group watchers; return once Pi's exit is reported."""
    rpc._process = proc
    rpc._watch_task = asyncio.create_task(rpc._watch_process())
    rpc._group_task = asyncio.create_task(rpc._watch_group())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while rpc._failure is None:
        assert loop.time() < deadline, "the watcher never reported Pi's exit"
        await asyncio.sleep(0.02)


async def _stop_watchers(rpc) -> None:
    for task in (rpc._stdout_task, rpc._stderr_task, rpc._watch_task, rpc._group_task):
        if task is not None and not task.done():
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)


async def _wait_until(predicate, timeout: float = 5) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


@_posix_only
@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["close", "kill_now"])
async def test_group_still_killed_after_watcher_saw_pi_exit(tmp_path, how):
    """A grandchild off Pi's pipes outlives wait(); close/kill_now must still reach it."""
    from a2a_adapter.base_adapter import _owns_group
    from tests.process_helpers import is_alive, wait_dead

    proc, _leader, grandchild = await _spawn_pi_tree(tmp_path, "detached-exit")
    rpc = _rpc(tmp_path, timeout=0.2)
    dead = False
    try:
        await _start_watchers(rpc, proc)
        assert is_alive(grandchild), "test setup: the grandchild should outlive Pi"
        if how == "close":
            await asyncio.wait_for(rpc.close(), 5)
            assert rpc._watch_task.done() and rpc._group_task.done()
        else:
            rpc.kill_now()
        dead = await wait_dead(grandchild)
        assert dead, f"grandchild survived {how}() after Pi exited"
        assert not _owns_group(proc), "a killed group's ID must not stay ours"
    finally:
        if not dead:
            _kill_test_pids(grandchild)
        await _stop_watchers(rpc)


@_posix_only
@pytest.mark.asyncio
async def test_watcher_forgets_group_once_it_is_empty(tmp_path):
    from asyncio.subprocess import PIPE, create_subprocess_exec

    from a2a_adapter.base_adapter import _owns_group, _spawn_cli

    proc = await _spawn_cli(
        create_subprocess_exec, _sys.executable, "-c", "pass", stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    rpc = _rpc(tmp_path, timeout=0.2)
    try:
        await _start_watchers(rpc, proc)
        assert await _wait_until(lambda: not _owns_group(proc)), "empty group stayed ours"
    finally:
        await _stop_watchers(rpc)


@_posix_only
@pytest.mark.asyncio
async def test_watcher_forgets_group_when_last_member_exits_later(tmp_path):
    """The group stays ours while a detached member runs, and is forgotten after."""
    from a2a_adapter.base_adapter import _owns_group
    from tests.process_helpers import wait_dead

    proc, _leader, grandchild = await _spawn_pi_tree(tmp_path, "detached-release-exit")
    rpc = _rpc(tmp_path, timeout=0.2)
    dead = False
    try:
        await _start_watchers(rpc, proc)
        await asyncio.sleep(0.7)  # longer than one poll interval
        assert _owns_group(proc), "the detached member still runs: the group is ours"
        (tmp_path / "pids.release").touch()
        dead = await wait_dead(grandchild)
        assert dead, "test setup: the grandchild should exit once released"
        assert await _wait_until(lambda: not _owns_group(proc)), "empty group stayed ours"
    finally:
        if not dead:
            _kill_test_pids(grandchild)
        await _stop_watchers(rpc)


@_posix_only
@pytest.mark.asyncio
async def test_group_forgotten_while_a_setsid_child_holds_the_pipes(tmp_path):
    """Pi exits, its group is empty, but wait() blocks on a child in its own session."""
    from a2a_adapter.base_adapter import _owns_group

    proc, _leader, grandchild = await _spawn_pi_tree(tmp_path, "setsid-exit")
    rpc = _rpc(tmp_path, timeout=0.2)
    try:
        rpc._process = proc
        # As in start(), before Pi exits: wait() then also waits for the pipes.
        rpc._stdout_task = asyncio.create_task(rpc._read_stdout())
        rpc._stderr_task = asyncio.create_task(rpc._read_stderr())
        rpc._watch_task = asyncio.create_task(rpc._watch_process())
        rpc._group_task = asyncio.create_task(rpc._watch_group())
        await _wait_leader_exit(proc)
        assert await _wait_until(lambda: not _owns_group(proc)), "empty group stayed ours"
        assert not rpc._watch_task.done(), "test setup: wait() should still be blocked"
    finally:
        _kill_test_pids(grandchild)  # still running in its own session: ours to kill
        await _stop_watchers(rpc)


@_posix_only
@pytest.mark.asyncio
async def test_cancelled_close_still_finishes_with_sigkill(tmp_path):
    """Cancelling a close() caller during the SIGTERM grace does not stop the close."""
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "ignore-term")
    rpc = _rpc(tmp_path, timeout=0.2)
    rpc._process = proc
    rpc.cancel_grace_timeout = 0.5
    caller = asyncio.create_task(rpc.close())
    dead = False
    try:
        await asyncio.sleep(0.2)  # in the SIGTERM grace wait
        caller.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await caller
        dead = await wait_dead(leader)
        assert dead, "the close was abandoned after SIGTERM: Pi ignores it and survives"
        await asyncio.wait_for(rpc.close(), 5)  # a repeat close waits for the first
    finally:
        if not dead:
            _kill_test_pids(leader, grandchild)
        with __import__("contextlib").suppress(Exception):
            await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_close_task_cancelled_kills_group(tmp_path):
    """If the close itself is cancelled (e.g. loop shutdown), the group gets SIGKILL."""
    from tests.process_helpers import wait_dead

    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "ignore-term")
    rpc = _rpc(tmp_path, timeout=0.2)
    rpc._process = proc
    rpc.cancel_grace_timeout = 30
    caller = asyncio.create_task(rpc.close())
    dead = False
    try:
        await asyncio.sleep(0.2)
        rpc._close_task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await asyncio.wait_for(caller, 5)
        dead = await wait_dead(leader)
        assert dead, "a cancelled close left Pi running"
    finally:
        if not dead:
            _kill_test_pids(leader, grandchild)
        with __import__("contextlib").suppress(Exception):
            await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_repeat_close_waits_for_the_graceful_close(tmp_path):
    """A second close() during the first must not cut Pi's SIGTERM grace short."""
    proc, leader, grandchild = await _spawn_pi_tree(tmp_path, "term-graceful")
    rpc = _rpc(tmp_path, timeout=0.2)
    rpc._process = proc
    rpc.cancel_grace_timeout = 5
    try:
        first = asyncio.create_task(rpc.close())
        await asyncio.sleep(0.05)
        await asyncio.wait_for(rpc.close(), 10)
        await asyncio.wait_for(first, 10)
        assert (tmp_path / "pids.termed").exists(), "Pi was killed before its grace period"
    finally:
        _kill_test_pids(grandchild)
        with __import__("contextlib").suppress(Exception):
            await asyncio.wait_for(proc.wait(), 5)


@_posix_only
@pytest.mark.asyncio
async def test_close_forgets_group_after_killing_it(tmp_path):
    from a2a_adapter.base_adapter import _owns_group
    from tests.process_helpers import wait_dead

    proc, _leader, grandchild = await _spawn_pi_tree(tmp_path, "detached-exit")
    rpc = _rpc(tmp_path, timeout=0.2)
    dead = False
    try:
        await _start_watchers(rpc, proc)
        await asyncio.wait_for(rpc.close(), 5)
        dead = await wait_dead(grandchild)
        assert not _owns_group(proc)
    finally:
        if not dead:
            _kill_test_pids(grandchild)
        await _stop_watchers(rpc)
