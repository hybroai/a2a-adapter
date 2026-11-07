"""Terminal display for the ``a2a-adapter`` CLI.

Three pieces, all plain ANSI with no extra dependency:

- ``render_card`` draws the startup card: what the agent is, where its A2A
  endpoints are, and which backend command serves it.
- ``ActivityLog`` prints one row per A2A operation and task state change
  (the "activity rail"). The executor reports task rows through the
  ``TaskObserver`` methods; ``ActivityMiddleware`` names each HTTP request
  and reports operations the executor never sees (``tasks/get``, card
  fetches, protocol errors).
- ``RailLogHandler`` routes Uvicorn and adapter log records into the rail.

Output that is not a terminal, or runs with ``NO_COLOR`` set, gets the same
information as plain lines with no box and no colour.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from importlib import metadata
from typing import Any, Callable, TextIO

from a2a.types import AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_CURRENT

__all__ = [
    "ActivityLog",
    "ActivityMiddleware",
    "BackendInfo",
    "RailLogHandler",
    "TerminalDisplay",
    "Verbosity",
    "classify_request",
    "color_enabled",
    "format_elapsed",
    "render_card",
]


class Verbosity(IntEnum):
    """How much the activity rail prints; each level adds to the one before."""

    QUIET = 0  # the card and errors only
    NORMAL = 1  # one row per operation, with one-line previews
    VERBOSE = 2  # full prompt and reply text under each row
    DEBUG = 3  # raw request, response and stream payloads, tracebacks


@dataclass(frozen=True)
class BackendInfo:
    """The local agent behind the A2A server, for the card's BACKEND section."""

    command: str
    cwd: str
    timeout: float | None = None


# ── Styling ──────────────────────────────────────────────────

_CODES = {
    "accent": "36",
    "dim": "90",
    "bold": "1",
    "green": "32",
    "red": "31",
    "yellow": "33",
    "title": "35",
    "url": "4;94",
}


def color_enabled(stream: TextIO) -> bool:
    """Colour and box drawing only for a real terminal that allows colour."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty and isatty())
    except ValueError:  # closed stream
        return False


class _Style:
    def __init__(self, color: bool) -> None:
        self.color = color

    def __call__(self, text: str, role: str | None) -> str:
        if not self.color or not role or not text:
            return text
        return f"\x1b[{_CODES[role]}m{text}\x1b[0m"


Segment = tuple[str, "str | None"]


def _join(style: _Style, segments: list[Segment]) -> str:
    return "".join(style(text, role) for text, role in segments)


def _char_cells(char: str) -> int:
    if unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _cells(text: str) -> int:
    """Terminal columns ``text`` takes: CJK and emoji take two."""
    return sum(_char_cells(char) for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _cells(text), 0)


def _width_of(segments: list[Segment]) -> int:
    return sum(_cells(text) for text, _ in segments)


def _take(text: str, width: int) -> str:
    """The longest prefix of ``text`` that fits in ``width`` columns."""
    used = 0
    for index, char in enumerate(text):
        used += _char_cells(char)
        if used > width:
            return text[:index]
    return text


def _cut_end(text: str, width: int) -> str:
    if _cells(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    return _take(text, width - 1) + "…"


def _cut_middle(text: str, width: int) -> str:
    if _cells(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    head = _take(text, width // 2)
    tail = _take(text[::-1], width - 1 - _cells(head))[::-1]
    return head + "…" + tail


_BIDI_CONTROLS = (
    set(range(0x202A, 0x202F))
    | set(range(0x2066, 0x206A))
    | {0x200E, 0x200F, 0x061C, 0x2028, 0x2029}  # direction marks, line separators
)


def _printable(text: str, *, newlines: bool = False) -> str:
    """Escape control characters so remote text can't drive the terminal.

    Prompts, replies and error messages come from network clients and
    agents; an ESC sequence in them could set the title, write the
    clipboard or clear the screen. Tabs become spaces; newlines are kept
    only where the caller wraps text itself.
    """
    out = []
    for char in text:
        code = ord(char)
        if char == "\n" and newlines:
            out.append(char)
        elif char == "\t":
            out.append(" ")
        elif unicodedata.category(char) == "Cc" or code in _BIDI_CONTROLS:
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
        else:
            out.append(char)
    return "".join(out)


def _terminal_width(fixed: int | None) -> int:
    if fixed is not None:
        return fixed
    return shutil.get_terminal_size((120, 24)).columns


def package_version() -> str:
    try:
        return metadata.version("a2a-adapter")
    except metadata.PackageNotFoundError:
        return "dev"


def format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m{rest:02d}s"


def _format_timeout(seconds: float) -> str:
    return f"{seconds:g}s"


# ── Startup card ─────────────────────────────────────────────

_LOGO = ("▄▀█ ▀█ ▄▀█", "█▀█ █▄ █▀█", "adapter")
_LOGO_WIDTH = 10
_LABEL_WIDTH = 10
_MIN_LOGO_INNER = 66
_MIN_INNER = 44
_FOOTER = " ctrl+c to stop "
_READY = "● ready"


@dataclass
class _CardInfo:
    name: str
    description: str
    version_line: str
    endpoints: list[tuple[str, str, str]]
    streaming: bool
    push: bool
    skills: str
    backend: BackendInfo | None


def _card_info(card: AgentCard, backend: BackendInfo | None, version: str) -> _CardInfo:
    interfaces = list(card.supported_interfaces)
    base = interfaces[0].url if interfaces else "http://localhost:9000/"
    root = base.rstrip("/")
    skills = ", ".join(skill.name or skill.id for skill in card.skills) or "none"
    if backend is not None:
        backend = BackendInfo(
            command=_printable(backend.command),
            cwd=_printable(backend.cwd),
            timeout=backend.timeout,
        )
    return _CardInfo(
        name=_printable(card.name),
        description=_printable(card.description),
        version_line=f"v{version}  ·  A2A protocol {PROTOCOL_VERSION_CURRENT} (0.3 compat)",
        endpoints=[
            ("card", root + AGENT_CARD_WELL_KNOWN_PATH, ""),
            ("json-rpc", base, ""),
            ("rest", base, "0.3 clients: /v1"),
        ],
        streaming=card.capabilities.streaming,
        push=card.capabilities.push_notifications,
        skills=_printable(skills),
        backend=backend,
    )


def render_card(
    card: AgentCard,
    *,
    backend: BackendInfo | None,
    version: str,
    width: int,
    color: bool,
) -> str:
    """Render the startup card for a terminal ``width`` columns wide.

    URLs are never shortened, so the card widens to fit them. When the
    terminal is too narrow for the logo it is dropped; when it is too narrow
    for the URLs the card falls back to plain lines.
    """
    info = _card_info(card, backend, version)
    if not color:
        return _plain_card(info)
    url_need = max(2 + _LABEL_WIDTH + _cells(url) + 2 for _, url, _ in info.endpoints)
    inner = max(_MIN_LOGO_INNER, url_need)
    logo = inner + 2 <= width
    if not logo:
        inner = max(_MIN_INNER, url_need)
        if inner + 2 > width:
            return _plain_card(info)
    return _box_card(info, inner, logo, _Style(color))


def _plain_card(info: _CardInfo) -> str:
    lines = [f"A2A agent ready: {info.name}"]
    if info.description:
        lines.append(f"  description: {info.description}")
    lines.append(f"  version: a2a-adapter {info.version_line}")
    for label, url, note in info.endpoints:
        lines.append(f"  {label}: {url}" + (f" ({note})" if note else ""))
    lines.append(f"  streaming: {'yes' if info.streaming else 'no'}")
    lines.append(f"  push notifications: {'yes' if info.push else 'no'}")
    lines.append(f"  skills: {info.skills}")
    if info.backend:
        lines.append(f"  command: {info.backend.command}")
        lines.append(f"  cwd: {info.backend.cwd}")
        if info.backend.timeout is not None:
            lines.append(f"  timeout: {_format_timeout(info.backend.timeout)}")
    lines.append("Press Ctrl+C to stop.")
    return "\n".join(lines)


def _box_card(info: _CardInfo, inner: int, logo: bool, style: _Style) -> str:
    border = lambda text: style(text, "accent")  # noqa: E731

    def line(segments: list[Segment]) -> str:
        pad = inner - _width_of(segments)
        return border("│") + _join(style, segments) + " " * pad + border("│")

    def field_line(label: str, value: str, role: str | None = None) -> str:
        room = inner - 2 - _LABEL_WIDTH - 2
        return line(
            [("  ", None), (label.ljust(_LABEL_WIDTH), "dim"), (_cut_middle(value, room), role)]
        )

    out = [border("╭" + "─" * inner + "╮"), line([])]

    # Header: logo on the left, identity on the right.
    text_col = 3 + _LOGO_WIDTH + 4 if logo else 3
    room = inner - text_col - 4
    left = [(" " * 3, None)]
    logo_cells = [
        [(_LOGO[0], "accent")],
        [(_LOGO[1], "accent")],
        [(_LOGO[2].ljust(_LOGO_WIDTH), "dim")],
    ]
    name = _cut_end(info.name, max(room - len(_READY) - 2, 1))
    gap = room - _cells(name) - len(_READY)
    header_rows = [
        [(name, "bold"), (" " * gap, None), (_READY, "green")],
        [(_cut_middle(info.description, room), "dim")] if info.description else [],
        [(_cut_middle(info.version_line, room), "dim")],
    ]
    if not info.description:
        header_rows = [header_rows[0], header_rows[2], []]
    for index, row in enumerate(header_rows):
        segments = list(left)
        if logo:
            segments += logo_cells[index] + [(" " * 4, None)]
        if not row and not logo:
            continue
        out.append(line(segments + row))
    out += [line([]), border("├" + "─" * inner + "┤"), line([])]

    out.append(line([("  ", None), ("ENDPOINTS", "title")]))
    for label, url, note in info.endpoints:
        segments: list[Segment] = [("  ", None), (label.ljust(_LABEL_WIDTH), "dim"), (url, "url")]
        suffix = f"  · {note}" if note else ""
        if suffix and _width_of(segments) + len(suffix) <= inner - 2:
            segments.append((suffix, "dim"))
        out.append(line(segments))
    out.append(line([]))

    out.append(line([("  ", None), ("CAPABILITIES", "title")]))
    out.append(
        line(
            [
                ("  ", None),
                ("✓" if info.streaming else "✗", "green" if info.streaming else "dim"),
                (" streaming   ", None),
                ("✓" if info.push else "✗", "green" if info.push else "dim"),
                (" push notifications", None),
            ]
        )
    )
    out.append(field_line("skills", info.skills))

    if info.backend:
        out += [line([]), line([("  ", None), ("BACKEND", "title")])]
        room = inner - 2 - _LABEL_WIDTH - 2
        command = _cut_middle(info.backend.command, room)
        executable, _, rest = command.partition(" ")
        out.append(
            line(
                [
                    ("  ", None),
                    ("command".ljust(_LABEL_WIDTH), "dim"),
                    (executable, "yellow"),
                    ((" " + rest) if rest else "", None),
                ]
            )
        )
        out.append(field_line("cwd", info.backend.cwd))
        if info.backend.timeout is not None:
            out.append(field_line("timeout", _format_timeout(info.backend.timeout)))

    out.append(line([]))
    out.append(
        border("╰" + "─" * (inner - len(_FOOTER) - 2)) + style(_FOOTER, "dim") + border("──╯")
    )
    return "\n".join(out)


# ── Request context shared by the middleware and the executor ─


@dataclass
class _RequestActivity:
    operation: str
    method: str
    path: str
    task_id: str | None
    start: float
    status: int = 0
    content_type: str = ""
    task_reported: bool = False
    error: str | None = None
    state: str | None = None
    context_id: str | None = None
    detail: str = ""
    message_id: str | None = None
    finished: bool = False
    _sse: bytes = field(default=b"", repr=False)


_current_request: contextvars.ContextVar[_RequestActivity | None] = contextvars.ContextVar(
    "a2a_adapter_current_request", default=None
)


# ── Request classification ───────────────────────────────────

_JSONRPC_OPERATIONS = {
    "message/send": "message/send",
    "SendMessage": "message/send",
    "message/stream": "message/stream",
    "SendStreamingMessage": "message/stream",
    "tasks/get": "tasks/get",
    "GetTask": "tasks/get",
    "ListTasks": "tasks/list",
    "tasks/cancel": "tasks/cancel",
    "CancelTask": "tasks/cancel",
    "tasks/resubscribe": "tasks/subscribe",
    "SubscribeToTask": "tasks/subscribe",
    "tasks/pushNotificationConfig/set": "push-config/set",
    "CreateTaskPushNotificationConfig": "push-config/set",
    "tasks/pushNotificationConfig/get": "push-config/get",
    "GetTaskPushNotificationConfig": "push-config/get",
    "tasks/pushNotificationConfig/list": "push-config/list",
    "ListTaskPushNotificationConfigs": "push-config/list",
    "tasks/pushNotificationConfig/delete": "push-config/delete",
    "DeleteTaskPushNotificationConfig": "push-config/delete",
    "GetExtendedAgentCard": "extended-card",
    "agent/getAuthenticatedExtendedCard": "extended-card",
}
_CARD_PATHS = {AGENT_CARD_WELL_KNOWN_PATH, "/.well-known/agent.json", "/card"}
_TASK_ACTION = re.compile(r"^/tasks/([^/:]+):(cancel|subscribe)$")
_TASK = re.compile(r"^/tasks/([^/:]+)$")
_PUSH_CONFIGS = re.compile(r"^/tasks/([^/:]+)/pushNotificationConfigs$")
_PUSH_CONFIG = re.compile(r"^/tasks/([^/:]+)/pushNotificationConfigs/[^/]+$")


def classify_request(method: str, path: str, body: Any) -> tuple[str, str | None]:
    """Name an A2A request the same way over JSON-RPC 1.0/0.3 and REST.

    Returns the operation (``message/send``, ``tasks/get``, ...) and the
    task id the request names, when it names one.
    """
    if method == "POST" and path == "/":
        if not isinstance(body, dict) or not isinstance(body.get("method"), str):
            return "POST /", None
        operation = _JSONRPC_OPERATIONS.get(body["method"], body["method"])
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        task_id = None
        if operation.startswith("tasks/") and operation != "tasks/list":
            task_id = params.get("id") or params.get("taskId")
        elif operation.startswith("push-config/"):
            task_id = params.get("taskId") or params.get("id")
        elif operation.startswith("message/"):
            message = params.get("message")
            if isinstance(message, dict):
                task_id = message.get("taskId")
        return operation, task_id if isinstance(task_id, str) else None

    rest = path[3:] if path == "/v1" or path.startswith("/v1/") else path
    if method == "GET" and rest in _CARD_PATHS:
        return "agent-card", None
    if method == "GET" and rest == "/extendedAgentCard":
        return "extended-card", None
    if method == "POST" and rest == "/message:send":
        return "message/send", None
    if method == "POST" and rest == "/message:stream":
        return "message/stream", None
    if match := _TASK_ACTION.match(rest):
        return f"tasks/{match.group(2)}", match.group(1)
    if method == "GET" and (match := _TASK.match(rest)):
        return "tasks/get", match.group(1)
    if method == "GET" and rest == "/tasks":
        return "tasks/list", None
    if match := _PUSH_CONFIGS.match(rest):
        if method in ("POST", "GET"):
            return ("push-config/set" if method == "POST" else "push-config/list"), match.group(1)
    if match := _PUSH_CONFIG.match(rest):
        if method in ("GET", "DELETE"):
            return ("push-config/get" if method == "GET" else "push-config/delete"), match.group(1)
    return f"{method} {path}", None


def _message_id(body: Any) -> str | None:
    """The messageId a JSON-RPC or REST send carries, if any."""
    if not isinstance(body, dict):
        return None
    params = body.get("params") if "jsonrpc" in body else body
    message = params.get("message") if isinstance(params, dict) else None
    value = message.get("messageId") if isinstance(message, dict) else None
    return value if isinstance(value, str) and value else None


def _normalize_state(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    state = value.removeprefix("TASK_STATE_").lower().replace("_", "-")
    return "canceled" if state == "cancelled" else state


def _find_task(data: Any) -> dict | None:
    """Find the task (or status update) in a JSON-RPC result or REST body."""
    if not isinstance(data, dict):
        return None
    for key in ("task", "statusUpdate"):
        if isinstance(data.get(key), dict):
            return data[key]
    if isinstance(data.get("status"), dict):
        return data
    return None


def _error_message(data: Any, status: int) -> str | None:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        return str(error.get("message") or error.get("code") or "error")
    if status >= 400:
        if isinstance(data, dict):
            for key in ("message", "detail", "error"):
                if isinstance(data.get(key), str) and data[key]:
                    return f"HTTP {status}: {data[key]}"
        return f"HTTP {status}"
    return None


# ── Activity rail ────────────────────────────────────────────

_STATE_STYLE = {
    "submitted": ("○", "dim"),
    "working": ("◐", "accent"),
    "completed": ("✓", "green"),
    "failed": ("✗", "red"),
    "rejected": ("✗", "red"),
    "canceled": ("⊘", "yellow"),
    "input-required": ("◇", "yellow"),
    "auth-required": ("◇", "yellow"),
}
_DIRECTION_STYLE = {"in": ("←", "accent"), "out": ("→", "green"), "status": ("·", "dim")}
_DIRECTION_STYLE["error"] = ("✗", "red")
_DIRECTION_STYLE["warning"] = ("!", "yellow")

_OP_WIDTH = 16
_STATE_WIDTH = 14  # fits input-required, the longest A2A state
_HEAD_WIDTH = 2 + 8 + 2 + 2 + _OP_WIDTH + 2 + 8 + 6 + 2 + 2 + _STATE_WIDTH + 2 + 7 + 3
_BODY_INDENT = 14
_PAYLOAD_LIMIT = 20_000
_CAPTURE_LIMIT = 1_000_000
_ANSWERED_LIMIT = 256


def _wrap_cells(text: str, width: int) -> list[str]:
    """Wrap one line to ``width`` terminal columns (CJK counts double).

    A line that fits is kept exactly, spacing and all, so code and lists
    survive ``-v``. A longer line wraps at spaces, and its continuation
    lines keep its indentation.
    """
    if _cells(text) <= width:
        return [text]
    indent = text[: len(text) - len(text.lstrip(" "))]
    if _cells(indent) > width // 2:
        indent = ""
    lines: list[str] = []
    current = indent
    for word in text[len(indent) :].split(" "):
        candidate = f"{current} {word}" if current.strip() else (current + word)
        if _cells(candidate) <= width:
            current = candidate
            continue
        if current.strip():
            lines.append(current)
            current = indent
        room = width - _cells(current)
        while _cells(word) > room:
            piece = _take(word, room) or word[0]
            lines.append(current + piece)
            word = word[len(piece) :]
            current = indent
            room = width - _cells(current)
        current += word
    if current.strip() or not lines:
        lines.append(current)
    return lines


def _short_id(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("-", "")[:4]


def _one_line(text: str) -> str:
    return " ".join(text.split())


class ActivityLog:
    """Prints the activity rail and receives task events from the executor.

    The ``task_*`` methods are the executor's ``TaskObserver`` interface; the
    ``request_*`` methods are fed by ``ActivityMiddleware``. All writes go
    through one lock, so rows from Uvicorn's log thread never interleave.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        width: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
        auto_header: bool = False,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.color = color_enabled(self.stream) if color is None else color
        self.verbosity = verbosity
        self._width = width
        self._clock = clock
        self._now = now or datetime.now
        self._auto_header = auto_header
        self._style = _Style(self.color)
        self._lock = threading.RLock()
        self._header_written = False
        self._closed = False
        self._task_starts: dict[str, float] = {}
        self._requests: dict[str, _RequestActivity] = {}
        # Requests answered before their message ran (a follow-up queued to
        # an active task); kept so the task rows still get their label.
        self._answered: OrderedDict[str, _RequestActivity] = OrderedDict()
        self._task_requests: dict[str, _RequestActivity | None] = {}

    # ── output ──

    def _prefix(self) -> str:
        return self._style("│", "accent") if self.color and self._header_written else ""

    def _prefix_width(self) -> int:
        return 1 if self.color and self._header_written else 0

    def _write(self, lines: list[str]) -> None:
        with self._lock:
            try:
                self.stream.write("".join(line.rstrip() + "\n" for line in lines))
                self.stream.flush()
            except (OSError, ValueError):
                pass

    def write_header(self) -> None:
        with self._lock:
            if self._header_written:
                return
            self._header_written = True
            style = self._style
            columns = (
                "  "
                + "TIME".ljust(8)
                + "    "
                + "OPERATION".ljust(_OP_WIDTH)
                + "  "
                + "CONTEXT".ljust(8)
                + "TASK".ljust(6)
                + "  "
                + "STATE".ljust(_STATE_WIDTH + 2)
                + "  "
                + "ELAPSED".rjust(7)
                + "   "
                + "DETAIL"
            )
            if self.color:
                rule = "─" * max(min(_terminal_width(self._width), 120) - 13, 10)
                self._write(
                    [
                        style("╭─ ", "accent")
                        + style("ACTIVITY", "title")
                        + style(" " + rule, "accent"),
                        self._prefix() + style(columns, "title"),
                        self._prefix(),
                    ]
                )
            else:
                self._write([columns])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._header_written:
                self._write([self._style("╰─ stopped", "accent") if self.color else "stopped"])

    def _row(
        self,
        direction: str,
        operation: str,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
        state: str | None = None,
        elapsed: float | None = None,
        detail: str = "",
        detail_role: str | None = None,
        preview: str | None = None,
        body: str | None = None,
        at: datetime | None = None,
    ) -> None:
        with self._lock:
            if self._auto_header and not self._header_written:
                self.write_header()
            style = self._style
            glyph, glyph_role = _DIRECTION_STYLE[direction]
            stamp = (at or self._now()).strftime("%H:%M:%S")
            if state:
                icon, state_role = _STATE_STYLE.get(state, ("·", "dim"))
                label = _pad(_cut_end(_printable(state), _STATE_WIDTH), _STATE_WIDTH)
                state_cell = style(icon, state_role) + " " + style(label, state_role)
            else:
                state_cell = " " * (_STATE_WIDTH + 2)
            elapsed_text = format_elapsed(elapsed) if elapsed is not None else ""
            head = (
                self._prefix()
                + "  "
                + style(stamp, "dim")
                + "  "
                + style(glyph, glyph_role)
                + " "
                + style(_pad(_cut_end(_printable(operation), _OP_WIDTH), _OP_WIDTH), "bold")
                + "  "
                + style(_pad(_short_id(_printable(context_id or "")), 8), "dim")
                + _pad(_short_id(_printable(task_id or "")), 6)
                + "  "
                + state_cell
                + "  "
                + style(elapsed_text.rjust(7), "dim")
                + "   "
            )
            room = max(_terminal_width(self._width) - self._prefix_width() - _HEAD_WIDTH, 20)
            detail, _, more = detail.strip("\n").partition("\n")
            detail = _printable(detail)
            if preview is not None and self.verbosity < Verbosity.VERBOSE:
                joiner = " · " if detail else ""
                quoted_room = max(room - _cells(detail) - len(joiner) - 2, 8)
                quoted = _cut_end(_printable(_one_line(preview)), quoted_room)
                detail = f'{detail}{joiner}"{quoted}"'
            elif preview is not None:
                body = preview
            # Errors are shown whole; everything else is cut to the line.
            shown = detail if detail_role == "red" else _cut_end(detail, room)
            lines = [head + style(shown, detail_role)]
            if more and self.verbosity >= Verbosity.VERBOSE:
                lines += self._body_lines(more, detail_role)
            elif more:
                count = len(more.splitlines())
                hint = f"(+{count} more line{'' if count == 1 else 's'}, -v to show)"
                lines[0] += " " + style(hint, "dim")
            if body is not None and self.verbosity >= Verbosity.VERBOSE:
                lines += self._body_lines(body)
            self._write(lines)

    def _body_lines(self, text: str, role: str | None = None) -> list[str]:
        indent = self._prefix() + " " * _BODY_INDENT
        wrap = max(_terminal_width(self._width) - self._prefix_width() - _BODY_INDENT, 20)
        lines = []
        for raw in _printable(text.strip("\n"), newlines=True).splitlines():
            for part in _wrap_cells(raw, wrap):
                lines.append(indent + self._style(part, role))
        return lines

    def payload(self, label: str, raw: bytes | str) -> None:
        """Print a raw request, response or stream event (``--debug``)."""
        if self.verbosity < Verbosity.DEBUG:
            return
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            pass
        text = _printable(text, newlines=True)
        if len(text) > _PAYLOAD_LIMIT:
            text = text[:_PAYLOAD_LIMIT] + f"\n… ({len(text) - _PAYLOAD_LIMIT} more characters)"
        indent = self._prefix() + " " * _BODY_INDENT
        lines = [indent + self._style(label, "title")]
        lines += [indent + self._style(line, "dim") for line in text.splitlines()]
        self._write(lines)

    # ── TaskObserver (called by the executor) ──

    def request_started(self, request: _RequestActivity) -> None:
        if request.message_id:
            with self._lock:
                self._requests[request.message_id] = request

    def _request_for(self, task_id: str, message_id: str | None = None, *, finish=False):
        """The HTTP request a task event belongs to.

        The SDK runs a task's worker in the context of the request that
        created it, and queues a follow-up message to a still-active task
        onto that same worker. So the context variable alone would credit
        the follow-up's rows to the first request; the message id finds
        the request that actually sent it.
        """
        with self._lock:
            request = None
            if message_id:
                request = self._requests.get(message_id) or self._answered.pop(message_id, None)
            if request is not None:
                self._task_requests[task_id] = request
            else:
                request = self._task_requests.get(task_id) or _current_request.get()
            if finish:
                self._task_requests.pop(task_id, None)
        return request

    def _operation(self, task_id: str, message_id: str | None = None, *, finish=False) -> str:
        request = self._request_for(task_id, message_id, finish=finish)
        if request is None:
            return "message"
        request.task_reported = True
        return request.operation if request.operation.startswith("message/") else "message"

    def _elapsed(self, task_id: str, *, finish: bool = False) -> float | None:
        start = self._task_starts.pop(task_id, None) if finish else self._task_starts.get(task_id)
        return None if start is None else self._clock() - start

    def task_submitted(
        self, task_id: str, context_id: str, prompt: str, message_id: str | None = None
    ) -> None:
        self._task_starts[task_id] = self._clock()
        operation = self._operation(task_id, message_id)
        if self.verbosity <= Verbosity.QUIET:
            return
        self._row(
            "in",
            operation,
            context_id=context_id,
            task_id=task_id,
            state="submitted",
            preview=prompt,
        )

    def task_working(self, task_id: str, context_id: str) -> None:
        if self.verbosity <= Verbosity.QUIET:
            return
        self._row(
            "status",
            "status",
            context_id=context_id,
            task_id=task_id,
            state="working",
            elapsed=self._elapsed(task_id),
        )

    def task_completed(self, task_id: str, context_id: str, text: str, chunks: int | None) -> None:
        operation = self._operation(task_id, finish=True)
        elapsed = self._elapsed(task_id, finish=True)
        if self.verbosity <= Verbosity.QUIET:
            return
        detail = "" if chunks is None else f"{chunks} chunk" + ("" if chunks == 1 else "s")
        self._row(
            "out",
            operation,
            context_id=context_id,
            task_id=task_id,
            state="completed",
            elapsed=elapsed,
            detail=detail,
            preview=text,
        )

    def task_failed(self, task_id: str, context_id: str, error: str) -> None:
        operation = self._operation(task_id, finish=True)
        self._row(
            "error",
            operation,
            context_id=context_id,
            task_id=task_id,
            state="failed",
            elapsed=self._elapsed(task_id, finish=True),
            detail=error,
            detail_role="red",
        )

    def task_canceled(self, task_id: str, context_id: str) -> None:
        self._request_for(task_id, finish=True)
        elapsed = self._elapsed(task_id, finish=True)
        if self.verbosity <= Verbosity.QUIET:
            return
        self._row(
            "status",
            "status",
            context_id=context_id,
            task_id=task_id,
            state="canceled",
            elapsed=elapsed,
        )

    # ── requests (called by the middleware) ──

    def stream_event(self, request: _RequestActivity, data: str) -> None:
        self.payload("event", data)
        try:
            event = json.loads(data)
        except ValueError:
            return
        error = _error_message(event, 0)
        if error:
            request.error = error
            return
        result = event.get("result", event) if isinstance(event, dict) else None
        task = _find_task(result)
        if task:
            request.state = (
                _normalize_state((task.get("status") or {}).get("state")) or request.state
            )
            request.task_id = request.task_id or task.get("taskId") or task.get("id")
            request.context_id = request.context_id or task.get("contextId")

    def request_rejected(self, method: str, path: str) -> None:
        """One row for a request refused by bearer auth (its body is never read)."""
        # Anyone can send these, so keep the path short.
        shown = path if len(path) <= 80 else path[:79] + "…"
        self._row("error", "unauthorized", detail=f"401 {method} {shown}", detail_role="red")

    def request_finished(self, request: _RequestActivity, body: bytes | None) -> None:
        with self._lock:
            if request.message_id and self._requests.get(request.message_id) is request:
                del self._requests[request.message_id]
                if not request.task_reported and request.operation.startswith("message/"):
                    self._answered[request.message_id] = request
                    while len(self._answered) > _ANSWERED_LIMIT:
                        self._answered.popitem(last=False)
        try:
            self._request_row(request, body)
        finally:
            if body:
                self.payload("response", body)

    def _request_row(self, request: _RequestActivity, body: bytes | None) -> None:
        data = None
        if body:
            try:
                data = json.loads(body)
            except ValueError:
                data = None
        error = request.error or _error_message(data, request.status)
        if error is None and request.status >= 400:
            error = f"HTTP {request.status}"
        result = data.get("result", data) if isinstance(data, dict) and "jsonrpc" in data else data
        task = _find_task(result)
        if task:
            request.state = (
                _normalize_state((task.get("status") or {}).get("state")) or request.state
            )
            request.task_id = request.task_id or task.get("id")
            request.context_id = request.context_id or task.get("contextId")
        elapsed = self._clock() - request.start

        if error:
            self._row(
                "error",
                request.operation,
                context_id=request.context_id,
                task_id=request.task_id,
                elapsed=elapsed,
                detail=error,
                detail_role="red",
            )
            return
        if request.task_reported or self.verbosity <= Verbosity.QUIET:
            return
        detail, role = "", None
        if request.operation in ("agent-card", "extended-card"):
            detail, role = f"{request.method} {request.path}", "dim"
        elif request.operation == "tasks/list" and isinstance(result, dict):
            tasks = result.get("tasks")
            if isinstance(tasks, list):
                detail = f"{len(tasks)} task" + ("" if len(tasks) == 1 else "s")
        self._row(
            "in",
            request.operation,
            context_id=request.context_id,
            task_id=request.task_id,
            state=request.state,
            elapsed=elapsed,
            detail=detail,
            detail_role=role,
        )

    # ── log records (called by RailLogHandler) ──

    def log_record(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR and self.verbosity <= Verbosity.QUIET:
            return
        if record.levelno >= logging.ERROR:
            direction, role = "error", "red"
        elif record.levelno >= logging.WARNING:
            direction, role = "warning", "yellow"
        else:
            direction, role = "status", "dim"
        name = record.name
        name = "uvicorn" if name.startswith("uvicorn") else name.rsplit(".", 1)[-1]
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not lose the record
            message = str(record.msg)
        first, _, extra = message.partition("\n")
        first = _printable(first)
        if record.exc_info and self.verbosity >= Verbosity.DEBUG:
            trace = logging.Formatter().formatException(record.exc_info)
            extra = f"{extra}\n{trace}" if extra else trace
        with self._lock:
            if self._auto_header and not self._header_written:
                self.write_header()
            style = self._style
            stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            glyph, glyph_role = _DIRECTION_STYLE[direction]
            line = (
                self._prefix()
                + "  "
                + style(stamp, "dim")
                + "  "
                + style(glyph, glyph_role)
                + " "
                + style(_pad(_cut_end(_printable(name), _OP_WIDTH), _OP_WIDTH), "bold")
                + "  "
                + style(first, role)
            )
            lines = [line]
            if extra:
                lines += self._body_lines(extra, "dim")
            self._write(lines)


_DISPATCHER_LOGGERS = {
    "a2a.compat.v0_3.jsonrpc_adapter",
    "a2a.compat.v0_3.rest_adapter",
    "a2a.server.routes.jsonrpc_dispatcher",
    "a2a.server.routes.jsonrpc_routes",
    "a2a.server.routes.rest_dispatcher",
    "a2a.server.routes.rest_routes",
    "a2a.utils.error_handlers",  # REST error responses
    "a2a.server.request_handlers.request_handler",  # validation failures
}


class RailLogHandler(logging.Handler):
    """Logging handler that prints records as rows on the activity rail."""

    def __init__(self, log: ActivityLog) -> None:
        super().__init__()
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        if self._log.verbosity < Verbosity.DEBUG:
            # The executor's own failure log duplicates the failed task row.
            if getattr(record, "a2a_activity_reported", False):
                return
            # The SDK's request dispatchers log protocol errors (e.g. task
            # not found) that the request's own error row reports. Only
            # while that request is in flight: the SDK's task workers keep
            # the first request's context long after it has finished.
            request = _current_request.get()
            if record.name in _DISPATCHER_LOGGERS and request is not None and not request.finished:
                return
        try:
            self._log.log_record(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ── ASGI middleware ──────────────────────────────────────────


class ActivityMiddleware:
    """Names each HTTP request as an A2A operation and reports it to the log.

    Reads the request body (to learn the JSON-RPC method) and replays it to
    the app unchanged. The operation is published in a context variable, so
    the executor's task rows, which run in a task spawned by this request,
    carry the same name.
    """

    def __init__(self, app: Any, log: ActivityLog) -> None:
        self.app = app
        self.log = log

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        body = b""
        early: list[dict] = []
        if method in ("POST", "PUT", "PATCH"):
            chunks = []
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    early.append(message)
                    break
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
        replay = [{"type": "http.request", "body": body, "more_body": False}] + early

        async def replay_receive() -> dict:
            if replay:
                return replay.pop(0)
            return await receive()

        parsed = None
        if body:
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
        operation, task_id = classify_request(method, path, parsed)
        request = _RequestActivity(
            operation=operation,
            method=method,
            path=path,
            task_id=task_id,
            start=self.log._clock(),
            message_id=_message_id(parsed),
        )
        self._safely(self.log.request_started, request)
        if body:
            self._safely(self.log.payload, "request", body)

        captured = bytearray()

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                request.status = message.get("status", 0)
                for key, value in message.get("headers", []):
                    if key.lower() == b"content-type":
                        request.content_type = value.decode("latin-1").lower()
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if "text/event-stream" in request.content_type:
                    self._safely(self._feed_sse, request, chunk)
                elif "json" in request.content_type and len(captured) < _CAPTURE_LIMIT:
                    captured.extend(chunk)
            await send(message)

        token = _current_request.set(request)
        try:
            await self.app(scope, replay_receive, send_wrapper)
        except Exception as error:
            request.error = f"{type(error).__name__}: {error}"
            request.status = request.status or 500
            raise
        finally:
            _current_request.reset(token)
            request.finished = True
            self._safely(self.log.request_finished, request, bytes(captured) or None)

    def _feed_sse(self, request: _RequestActivity, chunk: bytes) -> None:
        # Split on bytes, so a UTF-8 character cut across chunks survives.
        request._sse = (request._sse + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in request._sse:
            raw, request._sse = request._sse.split(b"\n\n", 1)
            event = raw.decode("utf-8", "replace")
            data = "\n".join(
                line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
            )
            if data:
                self.log.stream_event(request, data)

    @staticmethod
    def _safely(function: Callable, *args: Any) -> None:
        try:
            function(*args)
        except Exception:  # noqa: BLE001 - the display must never break a request
            logging.getLogger(__name__).debug("activity display failed", exc_info=True)


# ── Display (card + rail + logging) ──────────────────────────


class TerminalDisplay:
    """Everything the CLI prints: the card once Uvicorn is listening, then the rail."""

    def __init__(
        self,
        backend: BackendInfo | None = None,
        *,
        stream: TextIO | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        color: bool | None = None,
        width: int | None = None,
        version: str | None = None,
    ) -> None:
        self.backend = backend
        self.stream = stream if stream is not None else sys.stdout
        self.verbosity = verbosity
        self.color = color_enabled(self.stream) if color is None else color
        self.version = version or package_version()
        self._width = width
        self.activity = ActivityLog(
            self.stream,
            color=self.color,
            verbosity=verbosity,
            width=width,
            auto_header=verbosity <= Verbosity.QUIET,
        )
        self._handler: RailLogHandler | None = None
        self._saved: list[tuple[logging.Logger, list, bool, int]] = []

    def ready(self, card: AgentCard) -> None:
        text = render_card(
            card,
            backend=self.backend,
            version=self.version,
            width=_terminal_width(self._width),
            color=self.color,
        )
        waiting = "waiting for requests…"
        lines = [text, ("  " + _Style(self.color)(waiting, "dim")) if self.color else waiting]
        self.activity._write(lines)
        if self.verbosity > Verbosity.QUIET:
            self.activity._write([""])
            self.activity.write_header()

    def close(self) -> None:
        self.activity.close()

    def install_logging(self, uvicorn_level: str | int) -> None:
        """Send Uvicorn's and the adapters' log records to the rail."""
        if self._handler is not None:
            return
        handler = RailLogHandler(self.activity)
        self._handler = handler
        level = uvicorn_level
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
            if not isinstance(level, int):
                level = logging.WARNING

        def save(logger: logging.Logger) -> None:
            self._saved.append((logger, logger.handlers[:], logger.propagate, logger.level))

        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_error = logging.getLogger("uvicorn.error")
        package_logger = logging.getLogger("a2a_adapter")
        root = logging.getLogger()
        # uvicorn.Config(access_log=False) changes these two; put them back too.
        for logger in (
            uvicorn_logger,
            uvicorn_error,
            logging.getLogger("uvicorn.access"),
            logging.getLogger("uvicorn.asgi"),
            package_logger,
            root,
        ):
            save(logger)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
        uvicorn_error.handlers = []
        uvicorn_error.propagate = True
        uvicorn_error.setLevel(level)
        root.addHandler(handler)
        if self.verbosity >= Verbosity.DEBUG:
            package_logger.setLevel(logging.INFO)

    def uninstall_logging(self) -> None:
        for logger, handlers, propagate, level in reversed(self._saved):
            logger.handlers = handlers
            logger.propagate = propagate
            logger.setLevel(level)
        self._saved = []
        self._handler = None
