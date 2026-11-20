"""Run the core A2A flows against ONE real, installed agent CLI.

Manual only: this drives the owner's real agent with their real account and
can spend tokens. See SKILL.md next to this file.

    uv run python .claude/skills/verify-live/verify_live.py <adapter> [--port N]

It starts ``a2a-adapter <adapter>`` (this checkout's code) on 127.0.0.1 in a
throwaway workspace, runs each check with the smallest prompt that exercises
it, always stops every server it started, and prints one PASS / FAIL / SKIP
line per check plus a findings table (flow, expected, observed, evidence).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
V1_HEADERS = {"A2A-Version": "1.0"}
CARD_PATH = "/.well-known/agent-card.json"
CHECKS = ("card", "send", "stream", "cancel", "timeout", "error", "session")

# Server settings this driver must control: an inherited auth token would make
# every request 401, and the "dangerous" switches would widen what the real
# agent may do during the run.
_SCRUBBED_ENV = (
    "A2A_AUTH_TOKEN",
    "A2A_CLAUDE_SKIP_PERMISSIONS",
    "A2A_CODEX_BYPASS_APPROVALS",
)


# ──── Adapter profiles (each line confirmed against a2a_adapter/) ────


@dataclass(frozen=True)
class Profile:
    binary: str
    # get_metadata().streaming: False means the server rejects SendStreamingMessage.
    streaming: bool
    # Native flag after "--" that selects a model (None: unconfirmed or not native).
    model_flag: str | None = None
    # a2a-adapter option that selects a model (dsh takes no native args).
    model_option: str | None = None
    # Native args that let the agent run the `sleep` used by cancel and timeout.
    tool_args: tuple[str, ...] = ()
    # PiAdapter binds the server to the first context_id it sees.
    single_context: bool = False
    # Keeps one runtime process alive between turns (Pi RPC, dsh runtime).
    persistent: bool = False
    # cli.py rejects native args after "--" for dsh.
    native_args: bool = True
    # default_state_dir() agent name of the adapter's session store.
    state_agent: str | None = None


PROFILES: dict[str, Profile] = {
    # claude -p denies tools that need approval; allow only `sleep`.
    "claude": Profile(
        "claude",
        streaming=True,
        model_flag="--model",
        tool_args=("--allowedTools=Bash(sleep:*)",),
        state_agent="claude-code",
    ),
    "codex": Profile("codex", streaming=False, model_flag="--model", state_agent="codex"),
    "pi": Profile("pi", streaming=True, model_flag="--model", single_context=True, persistent=True),
    "openclaw": Profile("openclaw", streaming=False),
    "dsh": Profile(
        "dsh",
        streaming=True,
        model_option="--model",
        persistent=True,
        native_args=False,
        state_agent="deepseek-harness",
    ),
}

UNSUPPORTED = {
    "hermes": "HermesAdapter is an in-process gateway; a2a-adapter has no hermes subcommand",
    "ollama": "OllamaAdapter is an HTTP client; a2a-adapter has no ollama subcommand",
}


# ──── Results ────


@dataclass
class Result:
    check: str
    status: str  # PASS, FAIL or SKIP
    detail: str
    expected: str = ""
    observed: str = ""
    evidence: str = ""


def passed(check: str, detail: str) -> Result:
    return Result(check, "PASS", detail)


def skipped(check: str, reason: str) -> Result:
    return Result(check, "SKIP", reason)


def failed(check: str, expected: str, observed: str, evidence: str = "") -> Result:
    return Result(check, "FAIL", observed, expected, observed, evidence)


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"\b(?:sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_\-]{8,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?key|token|secret|password|authorization)\b"
        r"(\"?\s*[=:]\s*\"?|\s+)[^\s\",]+"
    ),
    re.compile(r"\b[A-Za-z0-9_\-+/]{40,}={0,2}"),
)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def redact(text: str, limit: int = 240) -> str:
    """Strip ANSI, mask anything token-shaped, collapse whitespace, truncate."""
    text = _ANSI.sub("", text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ──── Processes ────


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    pgid: int
    command: str

    def describe(self) -> str:
        return f"pid {self.pid} pgid {self.pgid}: {self.command[:80]}"


def ps_table() -> list[Proc]:
    done = subprocess.run(
        ["ps", "-Ao", "pid=,ppid=,pgid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    out = done.stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        with contextlib.suppress(ValueError):
            procs.append(
                Proc(
                    int(parts[0]), int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else ""
                )
            )
    # An empty or partial table would make every leftover check pass unseen.
    if done.returncode != 0 or all(p.pid != os.getpid() for p in procs):
        raise ProcessListError(f"cannot list processes with ps (exit {done.returncode})")
    return procs


class ProcessListError(RuntimeError):
    pass


def _ps_or_empty() -> list[Proc]:
    try:
        return ps_table()
    except ProcessListError:
        return []


def descendants(root: int, table: list[Proc] | None = None) -> list[Proc]:
    table = ps_table() if table is None else table
    children: dict[int, list[Proc]] = {}
    for proc in table:
        children.setdefault(proc.ppid, []).append(proc)
    found, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child.pid)
    return found


def marker_procs(marker: str) -> list[Proc]:
    """Every process whose command line holds ``marker``, wherever it lives.

    Group- and parent-independent on purpose: a tool process that moved into
    its own session, or was reparented to init, is still found.
    """
    me = os.getpid()
    # "sleep <marker>" as whole words, so no unrelated argv that merely
    # contains the digits is ever reported or killed.
    pattern = re.compile(rf"(?:^|[\s'\"/])sleep\s+{re.escape(marker)}(?=$|[\s'\";])")
    return [p for p in ps_table() if pattern.search(p.command) and p.pid != me]


def kill_pids(pids: list[int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def new_marker() -> str:
    """A unique ``sleep`` duration: it lives in the real sleep process's argv."""
    return f"{secrets.randbelow(300) + 600}.{secrets.randbelow(9000) + 1000}"


def sleep_prompt(marker: str) -> str:
    return (
        "Use your shell tool to run exactly this command and wait for it to finish: "
        f"sleep {marker}\nWhen it finishes, reply with the single word done."
    )


def port_is_free(port: int) -> bool:
    """False if anything accepts connections on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ──── Run state and servers ────


class Run:
    def __init__(self, args: argparse.Namespace, profile: Profile, executable: str) -> None:
        self.args = args
        self.adapter: str = args.adapter
        self.profile = profile
        self.executable = executable
        self.dir = Path(tempfile.mkdtemp(prefix=f"a2a-verify-live-{self.adapter}-")).resolve()
        self.workspace = self.dir / "workspace"
        self.logs = self.dir / "logs"
        self.workspace.mkdir()
        self.logs.mkdir()
        # Codex refuses to run outside a git repository.
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=False)
        self.tag = secrets.token_hex(4)
        self.shared_context = f"verify-live-{self.tag}"
        self.markers: list[str] = []
        self.servers: list[Server] = []
        self.teardown_leftovers: list[str] = []
        self.results: list[Result] = []
        self.could_not_run = False

    def context(self, name: str) -> str:
        if self.profile.single_context:
            return self.shared_context
        return f"verify-live-{name}-{self.tag}"

    def marker(self) -> str:
        marker = new_marker()
        self.markers.append(marker)
        return marker

    def server_args(self) -> list[str]:
        """Options that keep sessions inside this run and stable across restarts."""
        if self.adapter == "pi":
            return ["--session-id", self.shared_context, "--session-dir", str(self.dir / "pi")]
        if self.adapter == "openclaw":
            return ["--session-id", self.shared_context]
        # dsh keeps its default home: with $DSH_HOME set, the owner's home (config
        # and the run's sessions stay there); else a throwaway per-workspace dir.
        return []

    def model_args(self, model: str | None) -> tuple[list[str], list[str]]:
        """(a2a-adapter options, native args) that select ``model``."""
        if not model:
            return [], []
        if self.profile.model_option:
            return [self.profile.model_option, model], []
        if self.profile.model_flag:
            return [], [self.profile.model_flag, model]
        return [], []

    def server(
        self,
        label: str,
        *,
        timeout: float | None = None,
        model: str | None = None,
        native: tuple[str, ...] = (),
    ) -> Server:
        model = model if model is not None else self.args.model
        server_opts, model_native = self.model_args(model)
        native_args: list[str] = []
        if self.profile.native_args:
            native_args = [*self.profile.tool_args, *model_native, *native, *self.args.native]
        return Server(
            self,
            label,
            timeout=timeout or self.args.agent_timeout,
            options=[*self.server_args(), *server_opts],
            native=native_args,
        )

    def cleanup(self) -> None:
        for marker in self.markers:
            try:
                left = marker_procs(marker)
                kill_pids([p.pid for p in left])
                self.teardown_leftovers += [f"cleanup: {p.describe()}" for p in left]
            except ProcessListError as exc:
                self.teardown_leftovers.append(f"marker {marker}: unchecked ({exc})")
                self.could_not_run = True
        if self.args.keep:
            return
        if self.profile.state_agent:
            from a2a_adapter.base_adapter import default_state_dir

            shutil.rmtree(
                default_state_dir(self.profile.state_agent, str(self.workspace)),
                ignore_errors=True,
            )
        for child in self.dir.iterdir():
            if child != self.logs:
                shutil.rmtree(child, ignore_errors=True)


class ServerError(RuntimeError):
    pass


class Server:
    def __init__(
        self, run: Run, label: str, *, timeout: float, options: list[str], native: list[str]
    ) -> None:
        self.run = run
        self.label = label
        self.timeout = timeout
        self.port = run.args.port or free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = run.logs / f"{label}.log"
        # A unique card name proves the server that answers is the one started here.
        self.card_name = f"verify-live-{run.tag}-{label}"
        self.command = [
            sys.executable,
            "-m",
            "a2a_adapter.cli",
            run.adapter,
            "--cwd",
            str(run.workspace),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--timeout",
            str(timeout),
            "--executable",
            run.executable,
            "--name",
            self.card_name,
            *options,
        ]
        if native:
            self.command += ["--", *native]
        self.proc: subprocess.Popen | None = None
        self.client: httpx.AsyncClient | None = None
        self.children: set[tuple[int, str]] = set()

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    async def __aenter__(self) -> Server:
        if not port_is_free(self.port):
            # Never send live traffic to a server this run did not start.
            raise ServerError(f"port {self.port} is already in use; pick another or omit it")
        env = os.environ.copy()
        for name in _SCRUBBED_ENV:
            env.pop(name, None)
        env["PYTHONUNBUFFERED"] = "1"
        with self.log_path.open("w") as log:
            log.write(f"$ {' '.join(self.command)}\n")
        log = self.log_path.open("a")
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
        log.close()
        self.run.servers.append(self)
        self.client = httpx.AsyncClient(base_url=self.base_url, trust_env=False, timeout=10)
        deadline = time.monotonic() + self.run.args.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                await self.stop()
                raise ServerError(f"{self.label} server exited at startup: {self.log_tail()}")
            with contextlib.suppress(httpx.HTTPError, ValueError):
                response = await self.client.get(CARD_PATH)
                if response.status_code == 200 and response.json().get("name") == self.card_name:
                    if self.proc.poll() is None:
                        return self
            await asyncio.sleep(0.25)
        await self.stop()
        raise ServerError(f"{self.label} server not ready in time: {self.log_tail()}")

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def stop(self) -> None:
        """SIGTERM, SIGINT, then SIGKILL, 10 s apart; check nothing is left."""
        if self.client is not None:
            with contextlib.suppress(Exception):  # e.g. its event loop is gone
                await self.client.aclose()
            self.client = None
        proc = self.proc
        if proc is None or self not in self.run.servers:
            return
        # Stays in run.servers until it is confirmed dead, so an interrupted stop
        # is finished by main()'s teardown. Re-entrant: the snapshot accumulates.
        if proc.poll() is None:
            try:
                self.children |= {(p.pid, p.command) for p in descendants(proc.pid)}
            except ProcessListError as exc:
                self.run.teardown_leftovers.append(f"{self.label}: unchecked ({exc})")
                self.run.could_not_run = True
        # SIGTERM drains in-flight requests with no limit (uvicorn's default), so a
        # request still running after an interrupt gets SIGINT next: the server's
        # Ctrl+C path kills its CLIs. SIGKILL last, which leaves them alive.
        for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGINT, 10.0)):
            if proc.poll() is not None:
                break
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(sig)
            deadline = time.monotonic() + wait
            while proc.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.25)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        # Matched on pid AND command line, so a reused pid is never reported or killed.
        deadline = time.monotonic() + 5
        while True:
            try:
                table = ps_table()
            except ProcessListError as exc:
                self.run.teardown_leftovers.append(f"{self.label}: unchecked ({exc})")
                self.run.could_not_run = True
                alive = []
                break
            alive = [p for p in table if (p.pid, p.command) in self.children]
            if not alive or time.monotonic() > deadline:
                break
            await asyncio.sleep(0.25)
        if alive:
            self.run.teardown_leftovers += [f"{self.label}: {p.describe()}" for p in alive]
            kill_pids([p.pid for p in alive])
        self.run.servers.remove(self)

    def kill_now(self) -> None:
        """Last resort when stop() itself failed: SIGKILL the server's group."""
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        # Its children, matched on pid and command so a reused pid is spared.
        kill_pids([p.pid for p in _ps_or_empty() if (p.pid, p.command) in self.children])

    def log_tail(self, lines: int = 3) -> str:
        with contextlib.suppress(OSError):
            text = self.log_path.read_text(errors="replace").splitlines()
            return redact(" | ".join(text[-lines:]), 400)
        return "(no log)"

    def log_lines_with(self, needle: str, lines: int = 2) -> str:
        with contextlib.suppress(OSError):
            found = [
                line
                for line in self.log_path.read_text(errors="replace").splitlines()
                if needle in line
            ]
            return redact(" | ".join(found[:lines]), 300)
        return ""

    # ──── A2A requests (v1.0 JSON-RPC) ────

    async def rpc(self, method: str, params: dict, timeout: float) -> dict:
        assert self.client is not None
        body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
        response = await self.client.post("/", json=body, headers=V1_HEADERS, timeout=timeout)
        return response.json()

    async def send(
        self, text: str, context_id: str, *, timeout: float, return_immediately: bool = False
    ) -> dict:
        params: dict = {"message": user_message(text, context_id)}
        if return_immediately:
            params["configuration"] = {"returnImmediately": True}
        return await self.rpc("SendMessage", params, timeout)

    async def stream(
        self,
        text: str,
        context_id: str,
        *,
        timeout: float,
        on_event: Callable[[dict], Awaitable[None] | None] | None = None,
    ) -> tuple[list[dict], dict | None]:
        """(SSE results in arrival order, a JSON error if the server refused)."""
        assert self.client is not None
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "SendStreamingMessage",
            "params": {"message": user_message(text, context_id)},
        }
        events: list[dict] = []
        async with self.client.stream(
            "POST", "/", json=body, headers=V1_HEADERS, timeout=timeout
        ) as response:
            if "text/event-stream" not in response.headers.get("content-type", ""):
                payload = json.loads(await response.aread())
                return events, payload.get("error") or payload
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:])
                if "error" in payload:
                    return events, payload["error"]
                events.append(payload.get("result", {}))
                if on_event is not None:
                    maybe = on_event(events[-1])
                    if maybe is not None:
                        await maybe
        return events, None


def user_message(text: str, context_id: str) -> dict:
    return {
        "role": "ROLE_USER",
        "messageId": uuid.uuid4().hex,
        "contextId": context_id,
        "parts": [{"text": text}],
    }


def task_of(reply: dict) -> dict | None:
    result = reply.get("result")
    if isinstance(result, dict):
        return result.get("task", result if "status" in result else None)
    return None


def state_of(task: dict | None) -> str:
    return (task or {}).get("status", {}).get("state", "(none)")


def status_text(task: dict | None) -> str:
    message = (task or {}).get("status", {}).get("message") or {}
    return "".join(part.get("text", "") for part in message.get("parts", []))


def reply_text(task: dict | None) -> str:
    text = status_text(task)
    if text:
        return text
    return "".join(
        part.get("text", "")
        for artifact in (task or {}).get("artifacts", [])
        for part in artifact.get("parts", [])
    )


def error_of(reply: dict) -> str:
    error = reply.get("error")
    if not error:
        return ""
    return f"JSON-RPC error {error.get('code')}: {error.get('message')}"


def event_state(event: dict) -> str | None:
    if "statusUpdate" in event:
        return event["statusUpdate"].get("status", {}).get("state")
    if "task" in event:
        return event["task"].get("status", {}).get("state")
    return None


def event_task_id(event: dict) -> str | None:
    for key in ("task", "statusUpdate", "artifactUpdate"):
        if key in event:
            value = event[key]
            return value.get("id") or value.get("taskId")
    return None


TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED"}
# What the executor puts in a completed task when the adapter returned no text.
_EMPTY_REPLIES = {"", "[Empty response]", "[Streamed non-text content]"}


def is_empty(text: str) -> bool:
    return text.strip() in _EMPTY_REPLIES


# ──── Leftover check (shared shape for cancel and timeout) ────


async def settle_leftovers(
    run: Run, server: Server, marker: str, baseline: set[int], wait: float = 15.0
) -> list[Proc]:
    """Processes the run left behind: the marker `sleep` anywhere, plus (for
    one-shot CLIs) server descendants that were not there before the run.
    Waits up to ``wait`` seconds for them to go, then kills what is left."""
    deadline = time.monotonic() + wait
    while True:
        left = marker_procs(marker)
        if not run.profile.persistent:
            seen = {p.pid for p in left}
            left += [
                p for p in descendants(server.pid) if p.pid not in baseline and p.pid not in seen
            ]
        if not left or time.monotonic() > deadline:
            break
        await asyncio.sleep(0.5)
    kill_pids([p.pid for p in left])
    return left


async def wait_for_marker(marker: str, wait: float, done: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and not done():
        if marker_procs(marker):
            return True
        await asyncio.sleep(0.5)
    return bool(marker_procs(marker))


# ──── Checks ────


async def check_card(run: Run, server: Server) -> Result:
    assert server.client is not None
    response = await server.client.get(CARD_PATH)
    card = response.json()
    streaming = bool(card.get("capabilities", {}).get("streaming"))
    evidence = f"name={card.get('name')!r} streaming={streaming}"
    if response.status_code != 200 or card.get("name") != server.card_name:
        return failed(
            "card", f"200 with name {server.card_name!r}", f"HTTP {response.status_code}", evidence
        )
    if streaming != run.profile.streaming:
        return failed(
            "card", f"capabilities.streaming={run.profile.streaming}", evidence, CARD_PATH
        )
    return passed("card", evidence)


async def check_send(run: Run, server: Server) -> Result:
    started = time.monotonic()
    reply = await server.send(
        "Reply with the single word ok.", run.context("send"), timeout=server.timeout + 30
    )
    task = task_of(reply)
    text = reply_text(task)
    took = f"{time.monotonic() - started:.1f}s"
    if state_of(task) != "TASK_STATE_COMPLETED" or is_empty(text):
        return failed(
            "send",
            "TASK_STATE_COMPLETED with a non-empty reply",
            f"{state_of(task)} {error_of(reply)}".strip(),
            redact(text or server.log_tail()),
        )
    note = "" if "ok" in text.lower() else " (model did not say ok; adapter path is fine)"
    return passed("send", f"completed in {took}: {redact(text, 60)!r}{note}")


async def check_stream(run: Run, server: Server) -> Result:
    prompt = "Count from 1 to 5, one number per line, and nothing else."
    events, error = await server.stream(prompt, run.context("stream"), timeout=server.timeout + 30)
    if not run.profile.streaming:
        if error and error.get("code") == -32004:
            return passed("stream", "streaming=false on the card; server rejected it with -32004")
        return failed(
            "stream",
            "-32004 Streaming is not supported",
            f"{len(events)} events, error={error}",
        )
    if error:
        return failed("stream", "an SSE stream", "JSON-RPC error", redact(json.dumps(error)))
    states = [s for s in (event_state(e) for e in events) if s]
    chunks = [
        part.get("text", "")
        for e in events
        if "artifactUpdate" in e
        for part in e["artifactUpdate"].get("artifact", {}).get("parts", [])
    ]
    final = events[-1] if events else {}
    terminal_at = [i for i, e in enumerate(events) if event_state(e) in TERMINAL]
    # Exactly once: output streamed twice (a duplicated final chunk) must fail.
    in_order = re.findall(r"\d+", "".join(chunks)) == ["1", "2", "3", "4", "5"]
    evidence = f"states={states} chunks={len(chunks)}"
    if not terminal_at or terminal_at[0] != len(events) - 1:
        return failed(
            "stream", "exactly one terminal status, as the last event", evidence, str(terminal_at)
        )
    if event_state(final) != "TASK_STATE_COMPLETED":
        return failed(
            "stream",
            "final status TASK_STATE_COMPLETED",
            event_state(final) or "?",
            redact(status_text(final.get("statusUpdate"))),
        )
    if not chunks:
        return failed("stream", "at least one artifact chunk", "no artifactUpdate events", evidence)
    if not in_order:
        return failed(
            "stream",
            "1..5 once, in order, across chunks",
            redact(repr("".join(chunks)), 120),
            evidence,
        )
    return passed("stream", f"{len(chunks)} chunk(s) in order, ended {event_state(final)}")


async def check_cancel(run: Run, server: Server) -> Result:
    marker = run.marker()
    prompt = sleep_prompt(marker)
    context = run.context("cancel")
    baseline = {p.pid for p in descendants(server.pid)}
    events: list[dict] = []
    stream_task: asyncio.Task | None = None
    task_id: str | None = None
    if run.profile.streaming:
        stream_task = asyncio.create_task(
            server.stream(prompt, context, timeout=server.timeout + 60, on_event=events.append)
        )
        for _ in range(120):
            task_id = next((event_task_id(e) for e in events if event_task_id(e)), None)
            if task_id or stream_task.done():
                break
            await asyncio.sleep(0.25)
    else:
        reply = await server.send(prompt, context, timeout=60, return_immediately=True)
        task_id = (task_of(reply) or {}).get("id")
    if not task_id:
        if stream_task:
            stream_task.cancel()
        return failed("cancel", "a task id to cancel", "none", server.log_tail())

    def finished() -> bool:
        return bool(stream_task and stream_task.done()) or any(
            event_state(e) in TERMINAL for e in events
        )

    started = await wait_for_marker(marker, run.args.tool_wait, finished)
    t0 = time.monotonic()
    cancel = await server.rpc("CancelTask", {"id": task_id}, timeout=60)
    cancel_state = state_of(task_of(cancel)) if not error_of(cancel) else error_of(cancel)
    stream_end = None
    if stream_task is not None:
        try:
            await asyncio.wait_for(stream_task, 30)
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            stream_end = f"stream did not close: {type(exc).__name__}"
        else:
            # Exactly one terminal status, as the last event, and it is CANCELED.
            terminal = [i for i, e in enumerate(events) if event_state(e) in TERMINAL]
            if not terminal:
                stream_end = "without a terminal status"
            elif terminal != [len(events) - 1]:
                stream_end = f"with terminal events at {terminal} of {len(events)}"
            else:
                stream_end = event_state(events[-1])
    took = time.monotonic() - t0
    got = state_of(task_of(await server.rpc("GetTask", {"id": task_id}, timeout=30)))
    left = await settle_leftovers(run, server, marker, baseline)

    evidence = f"cancel→{cancel_state}; GetTask→{got}" + (
        f"; stream ended {stream_end}" if stream_end else ""
    )
    if got != "TASK_STATE_CANCELED" or (stream_end and stream_end != "TASK_STATE_CANCELED"):
        return failed("cancel", "TASK_STATE_CANCELED", evidence, server.log_tail())
    if left:
        return failed(
            "cancel",
            "no process from the run alive after cancel",
            f"{len(left)} left alive (killed by the driver)",
            redact("; ".join(p.describe() for p in left), 400),
        )
    if not started:
        return failed(
            "cancel",
            f"agent runs `sleep {marker}` so the leftover check means something",
            f"command not seen within {run.args.tool_wait:.0f}s; task was canceled",
            "agent refused or lacks shell permission; see tool_args in the profile",
        )
    return passed("cancel", f"canceled {took:.1f}s after CancelTask; nothing left alive")


async def check_timeout(run: Run) -> Result:
    limit = run.args.short_timeout
    async with run.server("timeout", timeout=limit) as server:
        marker = run.marker()
        baseline = {p.pid for p in descendants(server.pid)}
        started = time.monotonic()
        send = asyncio.create_task(
            server.send(sleep_prompt(marker), run.context("timeout"), timeout=limit + 90)
        )
        ran = await wait_for_marker(marker, limit + 90, send.done)
        reply = await send
        took = time.monotonic() - started
        left = await settle_leftovers(run, server, marker, baseline)
        task = task_of(reply)
        text = status_text(task)
    observed = f"{state_of(task)} after {took:.1f}s: {redact(text, 100)!r}"
    if state_of(task) != "TASK_STATE_FAILED" or "timed out" not in text.lower():
        return failed("timeout", f"TASK_STATE_FAILED 'timed out' within ~{limit:.0f}s", observed)
    if took > limit + 20:
        return failed("timeout", f"cut off near {limit:.0f}s", observed)
    if left:
        return failed(
            "timeout",
            "no process from the run alive after the timeout",
            f"{len(left)} left alive (killed by the driver)",
            redact("; ".join(p.describe() for p in left), 400),
        )
    note = "" if ran else " (sleep had not started: cut off before the tool call)"
    return passed("timeout", f"failed after {took:.1f}s with a timeout message{note}")


# A bad flag or model must fail on its own, well before the server's deadline;
# a hang that ends in "timed out" is the misbehaviour this check exists to find.
_ERROR_DEADLINE = 60.0
_ERROR_PROMPT_LIMIT = 45.0


async def check_error(run: Run, name: str, **server_kwargs) -> Result:
    async with run.server(name, timeout=_ERROR_DEADLINE, **server_kwargs) as server:
        started = time.monotonic()
        reply = await server.send(
            "Reply with the single word ok.", run.context(name), timeout=_ERROR_DEADLINE + 60
        )
        took = time.monotonic() - started
        task = task_of(reply)
        text = status_text(task)
        task_id = (task or {}).get("id", "")
        log = server.log_lines_with(task_id) if task_id else ""
    evidence = f"server log: {log}" if log else server.log_tail()
    if state_of(task) != "TASK_STATE_FAILED":
        return failed(
            name,
            "TASK_STATE_FAILED with a readable message",
            f"{state_of(task)} {error_of(reply)}".strip() + f": {redact(text, 80)!r}",
            evidence,
        )
    if not text.strip():
        return failed(name, "a status message on the failed task", "empty message", evidence)
    if took > _ERROR_PROMPT_LIMIT or "timed out" in f"{text} {log}".lower():
        return failed(
            name,
            f"the CLI rejects it within {_ERROR_PROMPT_LIMIT:.0f}s",
            (
                f"hung until the {_ERROR_DEADLINE:.0f}s deadline ({took:.1f}s)"
                if took > _ERROR_PROMPT_LIMIT
                else f"failed with a timeout message after {took:.1f}s"
            ),
            evidence,
        )
    # The A2A text is generic unless the error is user-facing; the server log
    # row holds what the CLI actually said.
    said = log.split(f"{task_id}:", 1)[-1].strip() if log else redact(text, 90)
    return passed(name, f"failed in {took:.1f}s: {redact(said, 140)}")


async def check_errors(run: Run) -> list[Result]:
    results = []
    if run.profile.native_args:
        results.append(await check_error(run, "error-flag", native=("--verify-live-no-such-flag",)))
    else:
        results.append(skipped("error-flag", "a2a-adapter dsh rejects native args after '--'"))
    if run.profile.model_flag or run.profile.model_option:
        results.append(await check_error(run, "error-model", model="verify-live-no-such-model"))
    else:
        results.append(
            skipped("error-model", f"no confirmed model flag for {run.adapter}; not guessed")
        )
    return results


SESSION_QUESTION = "What is the code word I asked you to remember? Reply with only the code word."


async def session_turn(server: Server, run: Run, text: str) -> tuple[str, str]:
    reply = await server.send(text, run.context("session"), timeout=server.timeout + 30)
    task = task_of(reply)
    return state_of(task), reply_text(task) or error_of(reply)


async def check_session_live(run: Run, server: Server, word: str) -> Result:
    first = await session_turn(server, run, f"Remember the code word {word}. Reply with only: ok")
    if first[0] != "TASK_STATE_COMPLETED" or is_empty(first[1]):
        return failed("session", "first turn completes with a reply", first[0], redact(first[1]))
    second = await session_turn(server, run, SESSION_QUESTION)
    if word not in second[1].lower():
        return failed(
            "session", f"second turn recalls {word!r}", f"{second[0]}: {redact(second[1], 80)!r}"
        )
    return passed("session", "second message continued the session")


async def check_session_restart(run: Run, word: str) -> Result:
    async with run.server("restart") as server:
        state, text = await session_turn(server, run, SESSION_QUESTION)
    if word not in text.lower():
        return failed(
            "session-restart",
            f"after restart the context still recalls {word!r}",
            f"{state}: {redact(text, 80)!r}",
        )
    return passed("session-restart", "session survived a server restart")


# ──── Main ────


def cli_version(executable: str, run_dir: Path | None = None) -> str:
    env = os.environ.copy()
    if run_dir is not None and not env.get("DSH_HOME", "").strip():
        # The dsh runtime refuses to run without an explicit DSH_HOME.
        env["DSH_HOME"] = str(run_dir / "dsh-version-home")
    try:
        out = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=20, env=env
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(unknown: {type(exc).__name__})"
    lines = [line for line in (out.stdout + out.stderr).splitlines() if line.strip()]
    return redact(lines[0], 80) if lines else "(no output)"


def table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "-"


def report(run: Run, version: str, results: list[Result]) -> int:
    model = run.args.model or "CLI default (not reported over A2A)"
    print(f"\nverify-live {run.adapter}")
    print(f"  cli      {run.executable}")
    print(f"  version  {version}")
    print(f"  model    {model}")
    print(f"  logs     {run.logs}\n")
    width = max(len(r.check) for r in results)
    for r in results:
        print(f"{r.status:<4}  {r.check:<{width}}  {r.detail}")
    failures = [r for r in results if r.status == "FAIL"]
    print("\nFindings")
    if not failures:
        print("  none: the real agent behaved as the adapter expects in every check that ran.")
    else:
        print("| flow | expected | observed | evidence |")
        print("|---|---|---|---|")
        for r in failures:
            print(
                f"| {r.check} | {table_cell(r.expected)} | {table_cell(r.observed)} "
                f"| {table_cell(r.evidence)} |"
            )
        print(
            "\nRatchet: turn each row into a fake-agent mode or unit test that fails"
            " first, then fix (see SKILL.md)."
        )
    if run.could_not_run:
        return 2
    return 1 if failures else 0


async def run_checks(run: Run, selected: set[str]) -> list[Result]:
    results = run.results  # kept on the run, so an interrupted run still reports
    word = f"lantern{secrets.randbelow(9000) + 1000}"

    async def guarded(name: str, check: Awaitable[Result | list[Result]]) -> None:
        try:
            outcome = await check
        except ServerError as exc:
            run.could_not_run = True
            results.append(failed(name, "server starts", "server did not start", str(exc)))
            return
        except ProcessListError as exc:
            run.could_not_run = True
            results.append(failed(name, "a process listing", "ps failed", str(exc)))
            return
        except Exception as exc:  # an odd reply is a finding, not a crash
            results.append(failed(name, "an A2A reply", type(exc).__name__, redact(str(exc))))
            return
        results.extend(outcome if isinstance(outcome, list) else [outcome])

    try:
        main = run.server("main")
        await main.__aenter__()
    except (ServerError, ProcessListError) as exc:
        run.could_not_run = True
        results.append(failed("card", "server starts", "server did not start", str(exc)))
        return results
    try:
        await guarded("card", check_card(run, main))
        if "send" in selected:
            await guarded("send", check_send(run, main))
        if "stream" in selected:
            await guarded("stream", check_stream(run, main))
        if "session" in selected:
            await guarded("session", check_session_live(run, main, word))
        if "cancel" in selected:
            await guarded("cancel", check_cancel(run, main))
    finally:
        await main.stop()
    if "session" in selected:
        if any(r.check == "session" and r.status == "PASS" for r in results):
            await guarded("session-restart", check_session_restart(run, word))
        else:
            results.append(skipped("session-restart", "the live session check did not pass"))
    if "timeout" in selected:
        await guarded("timeout", check_timeout(run))
    if "error" in selected:
        await guarded("error", check_errors(run))
    for name in CHECKS:
        if name not in selected and name != "card":
            results.append(skipped(name, "not selected with --only"))
    return results


def shutdown_result(run: Run) -> Result:
    """Built after the final teardown, so it also covers an interrupted run."""
    if run.teardown_leftovers:
        return failed(
            "shutdown",
            "stopping the server stops every process it started",
            f"{len(run.teardown_leftovers)} left alive or unchecked (killed by the driver)",
            redact("; ".join(run.teardown_leftovers), 400),
        )
    return passed("shutdown", "every server stopped with nothing left alive")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    native: list[str] = []
    if "--" in raw:
        index = raw.index("--")
        raw, native = raw[:index], raw[index + 1 :]
    parser = argparse.ArgumentParser(
        description="Run the core A2A flows against one real installed agent (manual only).",
        epilog="Native agent args after '--' go to every server this run starts.",
    )
    parser.add_argument("adapter", help=f"one of: {', '.join(PROFILES)}")
    parser.add_argument("port", nargs="?", type=int, help="fixed port (default: a free one)")
    parser.add_argument("--port", dest="port_option", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--executable", help="agent binary (default: found on PATH)")
    parser.add_argument("--model", help="model to request (default: the CLI's own default)")
    parser.add_argument("--only", help=f"comma-separated subset of: {', '.join(CHECKS)}")
    parser.add_argument("--agent-timeout", type=float, default=300.0, help="server --timeout")
    parser.add_argument("--short-timeout", type=float, default=30.0, help="timeout check limit")
    parser.add_argument("--tool-wait", type=float, default=90.0, help="wait for the sleep tool")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--keep", action="store_true", help="keep the workspace and agent state")
    args = parser.parse_args(raw)
    args.port = args.port_option or args.port
    args.native = native
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.adapter in UNSUPPORTED:
        print(f"verify-live: {args.adapter} is not supported: {UNSUPPORTED[args.adapter]}")
        return 2
    profile = PROFILES.get(args.adapter)
    if profile is None:
        print(f"verify-live: unknown adapter {args.adapter!r}; use one of {', '.join(PROFILES)}")
        return 2
    if args.native and not profile.native_args:
        print(f"verify-live: {args.adapter} takes no native args after '--'")
        return 2
    selected = set(CHECKS)
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = selected - set(CHECKS)
        if unknown:
            print(f"verify-live: unknown checks {sorted(unknown)}; use {', '.join(CHECKS)}")
            return 2
    executable = shutil.which(args.executable or profile.binary)
    if executable is None:
        print(f"verify-live: SKIP all: {args.executable or profile.binary} is not installed")
        return 2
    if args.adapter == "dsh":
        import importlib.util

        if importlib.util.find_spec("deepseek_harness") is None:
            print("verify-live: dsh needs the SDK: uv run --extra deepseek-harness ...")
            return 2

    # Ctrl+C, a hangup or a kill all unwind through the same teardown, even
    # when this process was started with SIGINT ignored.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    run = Run(args, profile, executable)
    version = cli_version(executable, run.dir if args.adapter == "dsh" else None)
    print(f"verify-live {args.adapter}: {executable} ({version}); logs in {run.logs}")
    try:
        results = asyncio.run(run_checks(run, selected))
    except KeyboardInterrupt:
        print("verify-live: interrupted; stopping servers")
        results = [*run.results, skipped("rest", "interrupted")]
        interrupted = True
    else:
        interrupted = False
    finally:
        # Nothing may interrupt the teardown itself.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for server in list(run.servers):
            try:
                asyncio.run(server.stop())
            except BaseException:
                server.kill_now()
        with contextlib.suppress(Exception):
            run.cleanup()
    results = [*results, shutdown_result(run)]
    status = report(run, version, results)
    return 130 if interrupted else status


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
