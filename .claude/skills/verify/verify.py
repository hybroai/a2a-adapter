"""End-to-end smoke check for the a2a-adapter runtime.

Run from the repo root:  uv run python .claude/skills/verify/verify.py

Starts the real ``a2a-adapter`` server (``python -m a2a_adapter.cli``) on a
free localhost port per scenario, with fake_agent.py standing in for the
``claude`` and ``codex`` CLIs, and sends real A2A 1.0 JSON-RPC requests over
HTTP. Prints one PASS/FAIL line per check; exits 1 if any check failed.
No network beyond 127.0.0.1, no real CLIs, no provider keys.

``-k SUBSTRING`` runs only the scenarios whose name contains it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FAKE_AGENT = HERE / "fake_agent.py"
V1 = {"A2A-Version": "1.0"}
CARD_PATH = "/.well-known/agent-card.json"
STARTUP_TIMEOUT = 20.0
STOP_TIMEOUT = 5.0
DEAD_TIMEOUT = 5.0
SECRET_STDERR = "FAKE-AGENT-SECRET-STDERR"
GENERIC_ERROR = "Agent execution failed (task {}). Details are in the server log."
AGENT_TIMEOUT = 1.5  # --timeout of the timeout scenario

# ──── Results ────

results: list[tuple[str, bool, str]] = []


class CheckFailed(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


@contextlib.asynccontextmanager
async def check(name: str):
    """Run one check; record and print PASS or FAIL. Never raises."""
    start = time.monotonic()
    try:
        yield
    except Exception as error:  # a failed check must not stop the others
        if isinstance(error, CheckFailed):
            detail = str(error)
        elif isinstance(error, httpx.TimeoutException):
            detail = f"no response from the server ({type(error).__name__})"
        else:
            detail = f"{type(error).__name__}: {error}"
        results.append((name, False, detail))
        print(f"FAIL  {name} ({time.monotonic() - start:.2f}s): {detail}", flush=True)
    else:
        results.append((name, True, ""))
        print(f"PASS  {name} ({time.monotonic() - start:.2f}s)", flush=True)


# ──── Processes ────


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return not stat.stdout.strip().startswith("Z")


def command_line(pid: int) -> str | None:
    """``pid``'s command line, or None if ``ps`` could not tell."""
    result = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


class Workspace:
    """Temp dirs shared by one run: agent cwd, state home, fake-agent state."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cwd = root / "project"
        self.state_home = root / "xdg-state"
        self.fake_state = root / "fake-agent"
        for path in (self.cwd, self.state_home, self.fake_state):
            path.mkdir()
        # A wrapper, so the fake runs under this interpreter with no exec bit needed.
        self.executable = root / "fake-cli"
        self.executable.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_AGENT))} "$@"\n'
        )
        self.executable.chmod(0o755)

    async def pids(self, tag: str, timeout: float = 5.0) -> list[int]:
        """PIDs the fake agent recorded for ``tag``, once it has recorded them."""
        path = self.fake_state / "pids" / f"{tag}.json"
        deadline = time.monotonic() + timeout
        while not path.exists():
            expect(time.monotonic() < deadline, f"fake agent never recorded PIDs for {tag}")
            await asyncio.sleep(0.02)
        return json.loads(path.read_text())

    def all_pids(self) -> list[int]:
        pids_dir = self.fake_state / "pids"
        if not pids_dir.exists():
            return []
        return [pid for path in pids_dir.glob("*.json") for pid in json.loads(path.read_text())]

    def is_ours(self, pid: int, *, unknown: bool) -> bool:
        """Alive and still this run's fake agent or its child, not a reused PID.

        The fake's argv names fake_agent.py; its child's argv names this run's
        state dir. ``unknown`` is the answer when ``ps`` cannot tell.
        """
        if not is_alive(pid):
            return False
        command = command_line(pid)
        if command is None:
            return unknown
        return str(FAKE_AGENT) in command or str(self.fake_state) in command

    async def wait_dead(self, pids: list[int], timeout: float = DEAD_TIMEOUT) -> list[int]:
        """PIDs still alive after ``timeout``. Errs toward alive: a leak never passes."""
        deadline = time.monotonic() + timeout
        while True:
            alive = [pid for pid in pids if self.is_ours(pid, unknown=True)]
            if not alive or time.monotonic() >= deadline:
                return alive
            await asyncio.sleep(0.05)

    def kill_leftovers(self) -> None:
        """SIGKILL recorded fake-agent PIDs that are still alive and still ours.

        A PID that died earlier may have been reused, so a process is killed
        only if its command line names the fake agent or this run's state dir.
        """
        for pid in self.all_pids():
            if self.is_ours(pid, unknown=False):  # never kill what we cannot identify
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)


class Server:
    """One ``a2a-adapter`` server process on a free port."""

    def __init__(self, ws: Workspace, args: list[str], env: dict[str, str] | None = None):
        self.ws = ws
        self.args = args
        self.extra_env = env or {}
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.log_path = ws.root / f"server-{uuid.uuid4().hex[:6]}.log"
        # A unique card name, so readiness cannot mistake another server for this one.
        self.name = f"verify-{uuid.uuid4().hex[:8]}"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> "Server":
        env = {
            **os.environ,
            "XDG_STATE_HOME": str(self.ws.state_home),
            "FAKE_AGENT_STATE": str(self.ws.fake_state),
            # Import a2a_adapter from this checkout, never a stale or skipped
            # install (Python 3.13 ignores .pth files macOS marks hidden).
            "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
            **self.extra_env,
        }
        if "A2A_AUTH_TOKEN" not in self.extra_env:
            env.pop("A2A_AUTH_TOKEN", None)
        try:
            return await self._start(env)
        except BaseException:  # also Ctrl+C or SIGTERM mid-start: no half-started server
            await self.stop()
            raise

    async def _start(self, env: dict[str, str]) -> "Server":
        for _ in range(3):  # another process may grab the free port first
            self.port = free_port()
            with open(self.log_path, "ab") as log:
                self.proc = subprocess.Popen(
                    [sys.executable, *self.args, "--port", str(self.port), "--name", self.name],
                    cwd=REPO,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            started = time.monotonic()
            if await self._ready():
                took = time.monotonic() - started
                if took > 3:
                    print(f"note: server took {took:.1f}s to start", flush=True)
                return self
            await self.stop()
        raise RuntimeError(f"server did not start; log:\n{self.log_tail()}")

    async def _ready(self) -> bool:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        async with httpx.AsyncClient(timeout=1.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    return False
                with contextlib.suppress(httpx.HTTPError, ValueError):
                    resp = await client.get(self.url + CARD_PATH)
                    card = resp.json() if resp.status_code == 200 else None
                    if isinstance(card, dict) and card.get("name") == self.name:
                        return self.proc.poll() is None
                await asyncio.sleep(0.05)
        return False

    async def stop(self) -> bool:
        """SIGINT (stops running agents at once), then SIGKILL if it hangs.

        Returns False if SIGINT did not stop the server in time.
        """
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return True
        proc.send_signal(signal.SIGINT)
        try:
            deadline = time.monotonic() + STOP_TIMEOUT
            while proc.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        finally:  # also when a second interrupt cancels the wait
            stopped = proc.poll() is not None
            if not stopped:
                proc.kill()
                proc.wait()
        return stopped

    async def __aexit__(self, exc_type, *exc) -> None:
        stopped = await self.stop()
        if not stopped and exc_type is None:
            raise CheckFailed(f"server did not stop on SIGINT within {STOP_TIMEOUT}s")

    def log_tail(self, lines: int = 30) -> str:
        with contextlib.suppress(OSError):
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        return "(no log)"


def cli_args(agent: str, ws: Workspace, *extra: str) -> list[str]:
    return [
        "-m",
        "a2a_adapter.cli",
        agent,
        "--executable",
        str(ws.executable),
        "--cwd",
        str(ws.cwd),
        "--host",
        "127.0.0.1",
        "--quiet",
        *extra,
    ]


# ──── A2A client helpers ────


def rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}


def message(text: str, *, task_id: str | None = None, context_id: str | None = None) -> dict:
    msg = {"role": "ROLE_USER", "messageId": uuid.uuid4().hex, "parts": [{"text": text}]}
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    return {"message": msg}


def status_text(status: dict) -> str:
    return "".join(p.get("text", "") for p in status.get("message", {}).get("parts", []))


def artifact_text(task: dict) -> str:
    return "".join(p.get("text", "") for a in task.get("artifacts", []) for p in a["parts"])


async def call(client: httpx.AsyncClient, method: str, params: dict, **headers) -> dict:
    resp = await client.post("/", json=rpc(method, params), headers={**V1, **headers})
    expect(resp.status_code == 200, f"{method} returned HTTP {resp.status_code}: {resp.text}")
    body = resp.json()
    expect("error" not in body, f"{method} returned an error: {body.get('error')}")
    return body["result"]


async def send(client: httpx.AsyncClient, text: str, **ids) -> dict:
    return (await call(client, "SendMessage", message(text, **ids)))["task"]


async def stream(client: httpx.AsyncClient, text: str, events: list, **ids) -> None:
    """SendStreamingMessage; appends (arrival time, result) to ``events``."""
    body = rpc("SendStreamingMessage", message(text, **ids))
    async with client.stream("POST", "/", json=body, headers=V1) as resp:
        expect(resp.status_code == 200, f"stream returned HTTP {resp.status_code}")
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                event = json.loads(line[5:])
                expect("error" not in event, f"stream error event: {event.get('error')}")
                events.append((time.monotonic(), event["result"]))


def final_status(events: list) -> dict:
    expect(bool(events), "stream sent no events")
    last = events[-1][1]
    update = last.get("statusUpdate") or last.get("task")
    expect(update is not None, f"stream did not end with a status: {last}")
    return update["status"]


def client_for(server: Server, timeout: float = 15.0, **headers) -> httpx.AsyncClient:
    # trust_env=False: never route 127.0.0.1 through an HTTP(S)_PROXY from the environment.
    return httpx.AsyncClient(base_url=server.url, timeout=timeout, headers=headers, trust_env=False)


# ──── Scenarios ────


async def scenario_claude(ws: Workspace) -> None:
    """Claude Code adapter: the streaming subprocess path."""
    async with Server(ws, cli_args("claude", ws)) as server, client_for(server) as client:
        async with check("claude: agent card is served"):
            card = (await client.get(CARD_PATH)).json()
            expect(card.get("name") == server.name, f"unexpected card: {card}")
            expect("Claude Code" in card.get("description", ""), f"unexpected card: {card}")
            expect(card.get("capabilities", {}).get("streaming") is True, "card says no streaming")

        async with check("claude: message returns a completed task"):
            task = await send(client, "mode:reply hello")
            expect(task["status"]["state"] == "TASK_STATE_COMPLETED", f"state {task['status']}")
            text = artifact_text(task)
            expect("echo=hello" in text and "turn=1" in text, f"unexpected reply: {text!r}")

        async with check("claude: stream delivers chunks in order, then completes"):
            events: list = []
            await stream(client, "mode:slow", events)
            chunks = [
                (at, p["text"])
                for at, e in events
                if "artifactUpdate" in e
                for p in e["artifactUpdate"]["artifact"]["parts"]
            ]
            joined = "".join(text for _, text in chunks)
            want = "\n\n".join(f"chunk {i}" for i in range(1, 6))
            expect(joined == want, f"chunks {joined!r}, want {want!r}")
            expect(len(chunks) == 5, f"{len(chunks)} artifact events, want 5")
            spread = chunks[-1][0] - chunks[0][0]
            expect(spread > 0.2, f"chunks arrived together ({spread:.2f}s): not streamed")
            status = final_status(events)
            expect(status["state"] == "TASK_STATE_COMPLETED", f"final status {status}")

        async with check("claude: cancel mid-stream kills the agent and its child"):
            await cancel_mid_stream(ws, client, "claude-cancel")

        async with check("claude: crash gives a generic error, no stderr"):
            await crash_is_readable(client)

        ids: dict = {}
        async with check("claude: second message continues the session"):
            await continues_session(client, ids)
        async with check("claude: stale session is dropped and the run retried fresh"):
            await stale_session_retried(ws, client)

    async with check("claude: session store survives a server restart"):
        expect(bool(ids), "skipped: the session check before it failed")
        async with Server(ws, cli_args("claude", ws)) as server, client_for(server) as client:
            third = await send(client, "mode:reply three", context_id=ids["context"])
            text = artifact_text(third)
            want = f"session={ids['session']}"
            expect(want in text and "turn=3" in text, f"no resume: {text!r}")

    async with check("claude: stopping the server mid-run kills the agent and its child"):
        server = Server(ws, cli_args("claude", ws))
        async with server, client_for(server) as client:
            events: list = []
            tag = "claude-shutdown"
            streaming = asyncio.create_task(stream(client, f"mode:child tag:{tag}", events))
            try:
                pids = await ws.pids(tag)
                expect(await server.stop(), f"server did not stop on SIGINT in {STOP_TIMEOUT}s")
                alive = await ws.wait_dead(pids)
                expect(not alive, f"processes still alive after shutdown: {alive}")
            finally:
                streaming.cancel()
                with contextlib.suppress(BaseException):
                    await streaming


async def scenario_codex(ws: Workspace) -> None:
    """Codex adapter: the non-streaming (invoke) subprocess path."""
    async with Server(ws, cli_args("codex", ws)) as server, client_for(server) as client:
        async with check("codex: message returns a completed task"):
            task = await send(client, "mode:reply hello")
            expect(task["status"]["state"] == "TASK_STATE_COMPLETED", f"state {task['status']}")
            text = artifact_text(task)
            expect("echo=hello" in text and "turn=1" in text, f"unexpected reply: {text!r}")

        async with check("codex: cancel mid-run kills the agent and its child"):
            sent = asyncio.create_task(send(client, "mode:child tag:codex-cancel"))
            try:
                pids = await ws.pids("codex-cancel")
                task_id = await task_id_of_running(client, sent)
                canceled = await call(client, "CancelTask", {"id": task_id})
                expect(
                    canceled["status"]["state"] == "TASK_STATE_CANCELED",
                    f"cancel returned {canceled['status']}",
                )
                result = await asyncio.wait_for(sent, 10)
                expect(
                    result["status"]["state"] == "TASK_STATE_CANCELED",
                    f"SendMessage returned {result['status']}",
                )
                alive = await ws.wait_dead(pids)
                expect(not alive, f"processes still alive after cancel: {alive}")
            finally:
                sent.cancel()

        async with check("codex: crash gives a generic error, no stderr"):
            await crash_is_readable(client)

        async with check("codex: second message continues the session"):
            await continues_session(client, {})
        async with check("codex: stale session is dropped and the run retried fresh"):
            await stale_session_retried(ws, client)


async def scenario_timeout(ws: Workspace) -> None:
    """Both subprocess paths under a short --timeout."""
    for agent in ("claude", "codex"):
        args = cli_args(agent, ws, "--timeout", str(AGENT_TIMEOUT))
        async with Server(ws, args) as server, client_for(server) as client:
            async with check(f"{agent}: hanging agent is cut off at the total timeout"):
                await times_out(ws, client, "hang", f"{agent}-hang")
            async with check(f"{agent}: agent that keeps printing is cut off at the timeout"):
                await times_out(ws, client, "drip", f"{agent}-drip")
            async with check(f"{agent}: agent that closed stdout but runs on is cut off"):
                await times_out(ws, client, "linger", f"{agent}-linger")
            async with check(f"{agent}: child holding the pipes is killed at the timeout"):
                await times_out(ws, client, "orphan", f"{agent}-orphan")


async def scenario_auth(ws: Workspace) -> None:
    token = uuid.uuid4().hex
    env = {"A2A_AUTH_TOKEN": token}
    async with Server(ws, cli_args("claude", ws), env=env) as server, client_for(server) as client:
        async with check("auth: agent card stays public and declares bearer auth"):
            resp = await client.get(CARD_PATH)
            expect(resp.status_code == 200, f"card returned HTTP {resp.status_code}")
            schemes = resp.json().get("securitySchemes", {})
            http_schemes = [v.get("httpAuthSecurityScheme", {}) for v in schemes.values()]
            expect(
                any(h.get("scheme", "").lower() == "bearer" for h in http_schemes),
                f"card declares no bearer scheme: {schemes}",
            )

        async with check("auth: JSON-RPC and REST requests without the right token get 401"):
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                body = rpc("SendMessage", message("mode:reply hi"))
                resp = await client.post("/", json=body, headers={**V1, **headers})
                expect(resp.status_code == 401, f"{headers or 'no token'}: {resp.status_code}")
            # The REST routes too, not only JSON-RPC.
            resp = await client.get("/tasks/any", headers=V1)
            expect(
                resp.status_code == 401, f"REST GET /tasks/any without token: {resp.status_code}"
            )

        async with check("auth: request with the token completes; token not passed to agent"):
            async with client_for(server, Authorization=f"Bearer {token}") as authed:
                task = await send(authed, "mode:reply hi")
            expect(task["status"]["state"] == "TASK_STATE_COMPLETED", f"state {task['status']}")
            text = artifact_text(task)
            expect("auth_env=absent" in text, f"agent saw A2A_AUTH_TOKEN: {text!r}")


async def scenario_input_required(ws: Workspace) -> None:
    """No CLI integration pauses for input, so a minimal in-process adapter does."""
    args = [str(Path(__file__).resolve()), "--serve-input-required"]
    async with Server(ws, args) as server, client_for(server) as client:
        async with check("input-required: run pauses, then resumes on the reply"):
            task = await send(client, "write a draft")
            state = task["status"]["state"]
            expect(state == "TASK_STATE_INPUT_REQUIRED", f"first turn ended {task['status']}")
            expect(status_text(task["status"]) == "Approve?", f"prompt {task['status']}")
            done = await send(client, "yes", task_id=task["id"], context_id=task["contextId"])
            expect(done["id"] == task["id"], "reply started a new task")
            expect(done["status"]["state"] == "TASK_STATE_COMPLETED", f"reply {done['status']}")
            expect(artifact_text(done) == "approved=yes", f"reply text {artifact_text(done)!r}")


# ──── Shared check bodies ────


def session_of(text: str) -> str:
    for word in text.split():
        if word.startswith("session="):
            return word[len("session=") :]
    raise CheckFailed(f"no session in reply {text!r}")


async def continues_session(client: httpx.AsyncClient, ids: dict) -> None:
    """Two messages in one context share the agent session; fills ``ids``."""
    first = await send(client, "mode:reply one")
    session = session_of(artifact_text(first))
    second = await send(client, "mode:reply two", context_id=first["contextId"])
    text = artifact_text(second)
    expect(f"session={session}" in text and "turn=2" in text, f"no resume: {text!r}")
    ids.update(context=first["contextId"], session=session)


async def stale_session_retried(ws: Workspace, client: httpx.AsyncClient) -> None:
    """The agent lost the session: the adapter retries once without resume."""
    first = await send(client, "mode:reply one")
    session = session_of(artifact_text(first))
    (ws.fake_state / "sessions" / session).unlink()
    second = await send(client, "mode:reply two", context_id=first["contextId"])
    expect(second["status"]["state"] == "TASK_STATE_COMPLETED", f"state {second['status']}")
    text = artifact_text(second)
    expect("turn=1" in text and f"session={session}" not in text, f"not a fresh run: {text!r}")


async def task_id_of_running(client: httpx.AsyncClient, sent: asyncio.Task) -> str:
    """The id of the one working task: SendMessage blocks, so ask ListTasks."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        expect(not sent.done(), "SendMessage finished before it could be canceled")
        listed = await call(client, "ListTasks", {})
        working = [
            t for t in listed.get("tasks", []) if t["status"]["state"] == "TASK_STATE_WORKING"
        ]
        if len(working) == 1:
            return working[0]["id"]
        await asyncio.sleep(0.05)
    raise CheckFailed("no working task to cancel")


async def cancel_mid_stream(ws: Workspace, client: httpx.AsyncClient, tag: str) -> None:
    events: list = []
    streaming = asyncio.create_task(stream(client, f"mode:child tag:{tag}", events))
    try:
        pids = await ws.pids(tag)
        deadline = time.monotonic() + 5
        while not any("artifactUpdate" in e for _, e in events):
            expect(time.monotonic() < deadline, "no chunk arrived before the cancel")
            await asyncio.sleep(0.02)
        task_id = events[0][1]["task"]["id"]
        canceled = await call(client, "CancelTask", {"id": task_id})
        expect(canceled["status"]["state"] == "TASK_STATE_CANCELED", f"cancel {canceled['status']}")
        await asyncio.wait_for(streaming, 10)
        status = final_status(events)
        expect(status["state"] == "TASK_STATE_CANCELED", f"stream ended {status}")
        alive = await ws.wait_dead(pids)
        expect(not alive, f"processes still alive after cancel: {alive}")
    finally:
        streaming.cancel()


async def crash_is_readable(client: httpx.AsyncClient) -> None:
    resp = await client.post("/", json=rpc("SendMessage", message("mode:crash")), headers=V1)
    expect(SECRET_STDERR not in resp.text, "agent stderr reached the client")
    body = resp.json()
    expect("error" not in body, f"crash became a JSON-RPC error: {body.get('error')}")
    task = body["result"]["task"]
    expect(task["status"]["state"] == "TASK_STATE_FAILED", f"state {task['status']}")
    text = status_text(task["status"])
    expect(text == GENERIC_ERROR.format(task["id"]), f"error message {text!r}")


async def times_out(ws: Workspace, client: httpx.AsyncClient, mode: str, tag: str) -> None:
    start = time.monotonic()
    task = await send(client, f"mode:{mode} tag:{tag}")
    elapsed = time.monotonic() - start
    pids = await ws.pids(tag, timeout=0.5)
    expect(task["status"]["state"] == "TASK_STATE_FAILED", f"state {task['status']}")
    text = status_text(task["status"])
    want = f"Agent execution failed: Command timed out after {AGENT_TIMEOUT} seconds"
    expect(text == want, f"error message {text!r}, want {want!r}")
    expect(
        AGENT_TIMEOUT <= elapsed < AGENT_TIMEOUT + 3,
        f"took {elapsed:.2f}s with --timeout {AGENT_TIMEOUT}",
    )
    alive = await ws.wait_dead(pids)
    expect(not alive, f"processes still alive after timeout: {alive}")


# ──── Input-required server (runs in its own process) ────


def serve_input_required(port: int, name: str) -> None:
    from a2a.types import TaskState

    from a2a_adapter import BaseA2AAdapter, InputRequired, serve_agent
    from a2a_adapter.base_adapter import AdapterMetadata

    class ApprovalAdapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            task = kwargs["context"].current_task
            if task is not None and task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
                return f"approved={user_input}"
            return InputRequired("Approve?")

        def get_metadata(self):
            return AdapterMetadata(name=name, description="Pauses for approval")

    serve_agent(ApprovalAdapter(), host="127.0.0.1", port=port, log_level="warning")


# ──── Main ────

SCENARIOS = {
    "claude": scenario_claude,
    "codex": scenario_codex,
    "timeout": scenario_timeout,
    "auth": scenario_auth,
    "input-required": scenario_input_required,
}


async def run(selected: list[str]) -> int:
    start = time.monotonic()
    # SIGTERM or SIGHUP (e.g. a harness timing the run out) unwinds like
    # Ctrl+C, so every server is stopped and the temp dir removed.
    loop, main_task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, main_task.cancel)
    with tempfile.TemporaryDirectory(prefix="a2a-verify-smoke-") as root:
        ws = Workspace(Path(root))
        try:
            for name in selected:
                async with check(f"{name}: scenario ran to the end"):
                    await SCENARIOS[name](ws)
            async with check("no fake-agent process left alive"):
                alive = await ws.wait_dead(ws.all_pids(), timeout=1.0)
                expect(not alive, f"still alive: {alive}")
        except asyncio.CancelledError:
            print("\ninterrupted; servers stopped", flush=True)
            return 1
        finally:
            ws.kill_leftovers()
    failed = [name for name, ok, _ in results if not ok]
    total = time.monotonic() - start
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed in {total:.1f}s")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-k", default="", help="run only scenarios whose name contains this")
    parser.add_argument("--serve-input-required", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve_input_required:
        serve_input_required(args.port, args.name)
        return 0
    if os.name != "posix":
        print("verify needs POSIX (process groups); skipping", file=sys.stderr)
        return 1
    selected = [name for name in SCENARIOS if args.k in name]
    if not selected:
        parser.error(f"no scenario matches {args.k!r}; have {', '.join(SCENARIOS)}")
    return asyncio.run(run(selected))


if __name__ == "__main__":
    sys.exit(main())
