"""Tests for the terminal display: startup card, activity rail and log routing."""

import io
import logging
import re
from datetime import datetime

import pytest

from a2a_adapter import AdapterMetadata, BaseA2AAdapter
from a2a_adapter.server import build_agent_card
from a2a_adapter.terminal import (
    ActivityLog,
    BackendInfo,
    RailLogHandler,
    TerminalDisplay,
    Verbosity,
    classify_request,
    color_enabled,
    format_elapsed,
    render_card,
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BOX = set("╭╮╰╯├┤│─")


def plain(text: str) -> str:
    return ANSI.sub("", text)


class _Adapter(BaseA2AAdapter):
    def __init__(self, description="Claude Code running in /Users/you/proj", streaming=True):
        self._description = description
        self._streaming = streaming

    async def invoke(self, user_input, context_id=None, **kwargs):
        return user_input

    def supports_streaming(self):
        return self._streaming

    def get_metadata(self):
        return AdapterMetadata(
            name="Claude (proj)",
            description=self._description,
            skills=[{"id": "general", "name": "general", "description": "d", "tags": ["x"]}],
        )


def _card(url="http://127.0.0.1:9000/", **kwargs):
    return build_agent_card(_Adapter(**kwargs), url=url)


BACKEND = BackendInfo(command="claude --model opus", cwd="/Users/you/proj", timeout=600.0)


# ── Startup card ─────────────────────────────────────────────


class TestRenderCard:
    def test_logo_card_lines_share_one_width(self):
        out = render_card(_card(), backend=BACKEND, version="0.2.17", width=120, color=True)
        lines = [plain(line) for line in out.splitlines()]
        box = [line for line in lines if line[:1] in "╭│├╰"]
        assert len({len(line) for line in box}) == 1
        assert box[0].startswith("╭") and box[-1].startswith("╰")
        text = "\n".join(lines)
        assert "▄▀█ ▀█ ▄▀█" in text
        assert "● ready" in text
        assert "Claude (proj)" in text
        assert "v0.2.17" in text and "A2A protocol 1.0" in text
        assert "ctrl+c to stop" in box[-1]
        assert "\x1b[" in out

    def test_sections_and_values(self):
        text = plain(render_card(_card(), backend=BACKEND, version="0.2.17", width=120, color=True))
        for title in ("ENDPOINTS", "CAPABILITIES", "BACKEND"):
            assert title in text
        assert "http://127.0.0.1:9000/.well-known/agent-card.json" in text
        assert re.search(r"json-rpc\s+http://127\.0\.0\.1:9000/\s", text)
        assert re.search(r"rest\s+http://127\.0\.0\.1:9000/", text)
        assert "✓ streaming" in text and "✗ push notifications" in text
        assert re.search(r"skills\s+general", text)
        assert re.search(r"command\s+claude --model opus", text)
        assert re.search(r"cwd\s+/Users/you/proj", text)
        assert re.search(r"timeout\s+600s", text)

    def test_non_streaming_agent_shows_streaming_unavailable(self):
        text = plain(
            render_card(_card(streaming=False), backend=None, version="1", width=120, color=True)
        )
        assert "✗ streaming" in text

    def test_long_url_widens_card_and_is_never_cut(self):
        url = "http://my-very-long-hostname.internal.example.com:9000/"
        out = render_card(_card(url=url), backend=BACKEND, version="1", width=200, color=True)
        lines = [plain(line) for line in out.splitlines() if plain(line)[:1] in "╭│├╰"]
        assert f"{url.rstrip('/')}/.well-known/agent-card.json" in "\n".join(lines)
        assert len(lines[0]) > 68
        assert len({len(line) for line in lines}) == 1

    def test_narrow_terminal_drops_logo_but_keeps_box(self):
        # The card URL needs a 64-column body; the logo needs 66.
        out = plain(render_card(_card(), backend=BACKEND, version="1", width=67, color=True))
        assert "▄▀█" not in out
        assert out.startswith("╭")
        assert all(len(line) <= 67 for line in out.splitlines())
        assert "● ready" in out

    def test_terminal_too_narrow_for_urls_falls_back_to_plain(self):
        out = render_card(_card(), backend=BACKEND, version="1", width=40, color=True)
        assert not BOX & set(plain(out))
        assert "http://127.0.0.1:9000/.well-known/agent-card.json" in out

    def test_long_values_are_cut_in_the_middle(self):
        description = "word " * 60
        out = render_card(
            _card(description=description), backend=BACKEND, version="1", width=100, color=True
        )
        lines = [plain(line) for line in out.splitlines() if plain(line)[:1] in "╭│├╰"]
        assert len({len(line) for line in lines}) == 1
        assert "…" in "\n".join(lines)
        assert len(lines[0]) <= 100

    def test_plain_output_has_no_colour_or_box(self):
        out = render_card(_card(), backend=BACKEND, version="0.2.17", width=120, color=False)
        assert "\x1b[" not in out
        assert not BOX & set(out)
        assert "Claude (proj)" in out
        assert "card: http://127.0.0.1:9000/.well-known/agent-card.json" in out
        assert "json-rpc: http://127.0.0.1:9000/" in out
        assert "streaming: yes" in out
        assert "command: claude --model opus" in out

    def test_backend_section_is_omitted_without_backend(self):
        out = plain(render_card(_card(), backend=None, version="1", width=120, color=True))
        assert "BACKEND" not in out


class TestColorEnabled:
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    def test_tty_without_no_color(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert color_enabled(self._Tty()) is True

    def test_no_color_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert color_enabled(self._Tty()) is False

    def test_dumb_terminal_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert color_enabled(self._Tty()) is False

    def test_pipe_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert color_enabled(io.StringIO()) is False


# ── Activity rail ────────────────────────────────────────────


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _log(verbosity=Verbosity.NORMAL, color=False, width=160):
    stream = io.StringIO()
    clock = FakeClock()
    log = ActivityLog(
        stream,
        color=color,
        verbosity=verbosity,
        width=width,
        clock=clock,
        now=lambda: datetime(2026, 11, 4, 14, 2, 11),
    )
    return log, stream, clock


def _rows(stream):
    return [plain(line) for line in stream.getvalue().splitlines()]


class TestFormatElapsed:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0.0021, "2ms"), (0.2, "200ms"), (3.24, "3.2s"), (27.06, "27.1s"), (125, "2m05s")],
    )
    def test_units(self, seconds, expected):
        assert format_elapsed(seconds) == expected


class TestActivityRows:
    def test_header_columns_line_up_with_rows(self):
        log, stream, _ = _log()
        log.write_header()
        log.task_submitted("91c2aaaa-0000", "7f3abbbb-0000", "Fix the failing test")
        lines = _rows(stream)
        header = next(line for line in lines if "OPERATION" in line)
        row = next(line for line in lines if "submitted" in line)
        for column, value in (
            ("OPERATION", "message"),
            ("CONTEXT", "7f3a"),
            ("TASK", "91c2"),
            ("STATE", "○ submitted"),
        ):
            assert header.index(column) == row.index(value), column

    def test_task_lifecycle_rows(self):
        log, stream, clock = _log()
        log.task_submitted("91c2", "7f3a", "Fix the failing test in utils.py")
        clock.t += 0.2
        log.task_working("91c2", "7f3a")
        clock.t += 26.9
        log.task_completed("91c2", "7f3a", "Fixed: the off-by-one", chunks=42)
        lines = _rows(stream)
        assert len(lines) == 3
        assert "14:02:11" in lines[0] and "←" in lines[0]
        assert '"Fix the failing test in utils.py"' in lines[0]
        assert "· status" in lines[1] and "◐ working" in lines[1] and "200ms" in lines[1]
        assert "→" in lines[2] and "✓ completed" in lines[2] and "27.1s" in lines[2]
        assert '42 chunks · "Fixed: the off-by-one"' in lines[2]

    def test_failed_and_canceled_rows(self):
        log, stream, clock = _log()
        log.task_submitted("a811", "b210", "hi")
        clock.t += 3.2
        log.task_failed("a811", "b210", "TimeoutError: agent exceeded 600s")
        log.task_canceled("c999", "b210")
        lines = _rows(stream)
        assert "✗" in lines[1] and "✗ failed" in lines[1] and "3.2s" in lines[1]
        assert "TimeoutError: agent exceeded 600s" in lines[1]
        assert "⊘ canceled" in lines[2]

    def test_preview_collapses_whitespace_and_fits_width(self):
        log, stream, _ = _log(width=100)
        log.task_submitted("t1", "c1", "line one\n\n   line two " + "x" * 200)
        (line,) = _rows(stream)
        assert '"line one line two' in line
        assert line.endswith('…"')
        assert len(line) <= 100

    def test_verbose_prints_full_text_below_the_row(self):
        log, stream, _ = _log(verbosity=Verbosity.VERBOSE, width=100)
        text = "Fixed the bug. " * 12
        log.task_completed("t1", "c1", text, chunks=3)
        lines = _rows(stream)
        assert lines[0].rstrip().endswith("3 chunks")
        body = " ".join(line.strip() for line in lines[1:])
        assert body == text.strip()
        assert all(line.startswith(" " * 14) for line in lines[1:])
        assert all(len(line) <= 100 for line in lines)

    def test_verbose_keeps_line_breaks(self):
        log, stream, _ = _log(verbosity=Verbosity.VERBOSE)
        log.task_submitted("t1", "c1", "first\nsecond")
        lines = _rows(stream)
        assert [line.strip() for line in lines[1:]] == ["first", "second"]

    def test_quiet_shows_only_failures(self):
        log, stream, _ = _log(verbosity=Verbosity.QUIET)
        log.task_submitted("t1", "c1", "hi")
        log.task_working("t1", "c1")
        log.task_completed("t1", "c1", "ok", chunks=None)
        log.task_failed("t2", "c1", "boom")
        lines = _rows(stream)
        assert len(lines) == 1 and "failed" in lines[0]

    def test_colour_rows_use_the_rail(self):
        log, stream, _ = _log(color=True)
        log.write_header()
        log.task_submitted("t1", "c1", "hi")
        lines = _rows(stream)
        assert lines[0].startswith("╭─ ACTIVITY")
        assert all(line.startswith("│") for line in lines[1:])
        assert "\x1b[" in stream.getvalue()

    def test_plain_rows_have_no_rail_or_colour(self):
        log, stream, _ = _log(color=False)
        log.write_header()
        log.task_submitted("t1", "c1", "hi")
        assert "\x1b[" not in stream.getvalue()
        assert not BOX & set(stream.getvalue())

    def test_close_ends_the_rail_once(self):
        log, stream, _ = _log(color=True)
        log.write_header()
        log.close()
        log.close()
        assert plain(stream.getvalue()).count("╰─ stopped") == 1

    def test_ids_are_shortened(self):
        log, stream, _ = _log()
        log.task_submitted("91c2e4f0-1111-2222", "7f3a9b2c-3333", "hi")
        (line,) = _rows(stream)
        assert " 7f3a " in line and " 91c2 " in line
        assert "91c2e4f0" not in line


class TestLogRouting:
    def _record(self, level, msg, name="a2a_adapter.integrations.claude_code", **extra):
        record = logging.LogRecord(name, level, __file__, 1, msg, None, None)
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_warning_becomes_a_rail_row(self):
        log, stream, _ = _log()
        RailLogHandler(log).handle(self._record(logging.WARNING, "Empty user input"))
        (line,) = _rows(stream)
        assert "!" in line and "claude_code" in line and "Empty user input" in line

    def test_records_already_reported_as_task_rows_are_skipped(self):
        log, stream, _ = _log()
        handler = RailLogHandler(log)
        handler.handle(self._record(logging.ERROR, "failed", a2a_activity_reported=True))
        assert stream.getvalue() == ""

    def test_traceback_only_at_debug(self):
        try:
            raise ValueError("bad")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        for verbosity, expected in ((Verbosity.NORMAL, False), (Verbosity.DEBUG, True)):
            log, stream, _ = _log(verbosity=verbosity)
            record = logging.LogRecord(
                "uvicorn.error", logging.ERROR, __file__, 1, "x", None, exc_info
            )
            RailLogHandler(log).handle(record)
            assert ("Traceback" in stream.getvalue()) is expected

    def test_quiet_keeps_errors_and_drops_warnings(self):
        log, stream, _ = _log(verbosity=Verbosity.QUIET)
        handler = RailLogHandler(log)
        handler.handle(self._record(logging.WARNING, "warn"))
        handler.handle(self._record(logging.ERROR, "port in use", name="uvicorn.error"))
        lines = _rows(stream)
        assert len(lines) == 1 and "port in use" in lines[0] and "uvicorn" in lines[0]


# ── Request classification ───────────────────────────────────


class TestClassifyRequest:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("message/send", "message/send"),
            ("SendMessage", "message/send"),
            ("message/stream", "message/stream"),
            ("SendStreamingMessage", "message/stream"),
            ("tasks/get", "tasks/get"),
            ("GetTask", "tasks/get"),
            ("tasks/cancel", "tasks/cancel"),
            ("CancelTask", "tasks/cancel"),
            ("ListTasks", "tasks/list"),
            ("tasks/resubscribe", "tasks/subscribe"),
            ("SubscribeToTask", "tasks/subscribe"),
            ("tasks/pushNotificationConfig/set", "push-config/set"),
            ("CreateTaskPushNotificationConfig", "push-config/set"),
            ("GetExtendedAgentCard", "extended-card"),
            ("agent/getAuthenticatedExtendedCard", "extended-card"),
            ("made/up", "made/up"),
        ],
    )
    def test_jsonrpc_methods(self, method, expected):
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"id": "task-1"}}
        op, task_id = classify_request("POST", "/", body)
        assert op == expected
        if expected.startswith("tasks/") and expected != "tasks/list":
            assert task_id == "task-1"

    @pytest.mark.parametrize(
        "method,path,expected,task_id",
        [
            ("GET", "/.well-known/agent-card.json", "agent-card", None),
            ("GET", "/.well-known/agent.json", "agent-card", None),
            ("GET", "/v1/card", "agent-card", None),
            ("GET", "/extendedAgentCard", "extended-card", None),
            ("POST", "/message:send", "message/send", None),
            ("POST", "/v1/message:stream", "message/stream", None),
            ("GET", "/tasks/abc", "tasks/get", "abc"),
            ("GET", "/v1/tasks/abc", "tasks/get", "abc"),
            ("POST", "/tasks/abc:cancel", "tasks/cancel", "abc"),
            ("GET", "/tasks/abc:subscribe", "tasks/subscribe", "abc"),
            ("GET", "/tasks", "tasks/list", None),
            ("POST", "/tasks/abc/pushNotificationConfigs", "push-config/set", "abc"),
            ("GET", "/tasks/abc/pushNotificationConfigs", "push-config/list", "abc"),
            ("GET", "/tasks/abc/pushNotificationConfigs/p1", "push-config/get", "abc"),
            ("DELETE", "/tasks/abc/pushNotificationConfigs/p1", "push-config/delete", "abc"),
            ("GET", "/favicon.ico", "GET /favicon.ico", None),
        ],
    )
    def test_rest_and_card_paths(self, method, path, expected, task_id):
        assert classify_request(method, path, None) == (expected, task_id)

    def test_unparseable_jsonrpc_body(self):
        assert classify_request("POST", "/", None) == ("POST /", None)


# ── Display ──────────────────────────────────────────────────


class TestTerminalDisplay:
    def test_ready_prints_card_then_rail_header(self):
        stream = io.StringIO()
        display = TerminalDisplay(BACKEND, stream=stream, color=True, width=120, version="1")
        display.ready(_card())
        text = plain(stream.getvalue())
        assert (
            text.index("● ready") < text.index("waiting for requests…") < text.index("╭─ ACTIVITY")
        )

    def test_quiet_ready_prints_card_without_rail(self):
        stream = io.StringIO()
        display = TerminalDisplay(
            BACKEND, stream=stream, color=True, width=120, version="1", verbosity=Verbosity.QUIET
        )
        display.ready(_card())
        assert "ACTIVITY" not in plain(stream.getvalue())
        assert "● ready" in plain(stream.getvalue())

    def test_install_logging_routes_uvicorn_and_adapter_logs(self):
        stream = io.StringIO()
        display = TerminalDisplay(BACKEND, stream=stream, color=False, width=160, version="1")
        uvicorn_logger = logging.getLogger("uvicorn")
        saved = (uvicorn_logger.handlers[:], uvicorn_logger.propagate)
        root = logging.getLogger()
        root_handlers = root.handlers[:]
        try:
            display.install_logging("warning")
            logging.getLogger("uvicorn.error").info("Started server process")
            logging.getLogger("uvicorn.error").error("address already in use")
            logging.getLogger("a2a_adapter.integrations.codex").warning("codex stderr")
        finally:
            display.uninstall_logging()
            uvicorn_logger.handlers[:], uvicorn_logger.propagate = saved
            root.handlers[:] = root_handlers
        text = stream.getvalue()
        assert "Started server process" not in text
        assert "address already in use" in text
        assert "codex stderr" in text


# ── serve_agent with a display ───────────────────────────────


class TestServeAgentWithDisplay:
    @pytest.mark.parametrize("started, expected", [(True, 1), (False, 0)])
    def test_ready_callback_waits_for_the_socket(self, monkeypatch, started, expected):
        import asyncio

        import uvicorn

        from a2a_adapter.server import _ReadyServer

        async def fake_startup(self, sockets=None):
            self.started = started

        monkeypatch.setattr(uvicorn.Server, "startup", fake_startup)
        calls = []
        server = _ReadyServer(uvicorn.Config(app=None), on_ready=lambda: calls.append(1))
        asyncio.run(server.startup())
        assert len(calls) == expected

    def test_serve_agent_wires_the_display(self, monkeypatch):
        from a2a_adapter import server as server_module

        events = []

        class FakeDisplay:
            activity = ActivityLog(io.StringIO(), color=False)

            def install_logging(self, level):
                events.append(("install", level))

            def ready(self, card):
                events.append(("ready", card.name))

            def close(self):
                events.append(("close",))

            def uninstall_logging(self):
                events.append(("uninstall",))

        captured = {}

        def fake_run(self):
            captured["config"] = self.config
            self.started = True
            self._on_ready()

        monkeypatch.setattr(server_module._ReadyServer, "run", fake_run)
        server_module.serve_agent(
            _Adapter(), host="127.0.0.1", port=9100, log_level="warning", display=FakeDisplay()
        )
        assert events == [
            ("install", "warning"),
            ("ready", "Claude (proj)"),
            ("close",),
            ("uninstall",),
        ]
        config = captured["config"]
        assert config.access_log is False
        assert config.log_config is None
        assert config.port == 9100


class TestInterruptAndSdkNoise:
    def test_ctrl_c_stops_without_a_traceback(self, monkeypatch):
        from a2a_adapter import server as server_module

        closed = []

        class FakeDisplay:
            activity = ActivityLog(io.StringIO(), color=False)

            def install_logging(self, level):
                pass

            def ready(self, card):
                pass

            def close(self):
                closed.append("close")

            def uninstall_logging(self):
                closed.append("uninstall")

        def interrupted(self):
            self.started = True
            raise KeyboardInterrupt

        monkeypatch.setattr(server_module._ReadyServer, "run", interrupted)
        server_module.serve_agent(_Adapter(), display=FakeDisplay())
        assert closed == ["close", "uninstall"]

    def test_sdk_errors_inside_a_request_are_left_to_the_request_row(self):
        from a2a_adapter.terminal import _current_request, _RequestActivity

        for verbosity, shown in ((Verbosity.NORMAL, False), (Verbosity.DEBUG, True)):
            log, stream, _ = _log(verbosity=verbosity)
            handler = RailLogHandler(log)
            token = _current_request.set(_RequestActivity("tasks/get", "POST", "/", None, 0.0))
            try:
                record = logging.LogRecord(
                    "a2a.compat.v0_3.jsonrpc_adapter",
                    logging.ERROR,
                    __file__,
                    1,
                    "Unhandled exception in v0.3 JSONRPCAdapter",
                    None,
                    None,
                )
                handler.handle(record)
            finally:
                _current_request.reset(token)
            assert ("Unhandled exception" in stream.getvalue()) is shown

    def test_sdk_errors_outside_a_request_are_shown(self):
        log, stream, _ = _log()
        record = logging.LogRecord(
            "a2a.server.tasks", logging.ERROR, __file__, 1, "push failed", None, None
        )
        RailLogHandler(log).handle(record)
        assert "push failed" in stream.getvalue()


# ── Review round 1 ───────────────────────────────────────────


class _QuietDisplay:
    activity = ActivityLog(io.StringIO(), color=False)

    def install_logging(self, level):
        pass

    def ready(self, card):
        pass

    def close(self):
        pass

    def uninstall_logging(self):
        pass


def test_failed_startup_exits_with_uvicorns_code(monkeypatch):
    from a2a_adapter import server as server_module

    def never_started(self):
        self.started = False

    monkeypatch.setattr(server_module._ReadyServer, "run", never_started)
    with pytest.raises(SystemExit) as exit_info:
        server_module.serve_agent(_Adapter(), display=_QuietDisplay())
    assert exit_info.value.code == 3


@pytest.mark.parametrize("kwargs", [{"workers": 2}, {"reload": True}])
def test_display_rejects_multi_process_options(kwargs):
    from a2a_adapter import server as server_module

    with pytest.raises(ValueError, match="workers/reload"):
        server_module.serve_agent(_Adapter(), display=_QuietDisplay(), **kwargs)


def _sdk_record(name="a2a.compat.v0_3.jsonrpc_adapter", msg="Unhandled exception"):
    return logging.LogRecord(name, logging.ERROR, __file__, 1, msg, None, None)


@pytest.mark.parametrize(
    "name, finished, shown",
    [
        ("a2a.compat.v0_3.jsonrpc_adapter", False, False),
        ("a2a.compat.v0_3.jsonrpc_adapter", True, True),
        ("a2a.server.tasks.base_push_notification_sender", False, True),
        ("a2a.server.agent_execution.active_task", False, True),
    ],
)
def test_only_in_flight_dispatcher_errors_are_left_to_the_row(name, finished, shown):
    from a2a_adapter.terminal import _current_request, _RequestActivity

    log, stream, _ = _log(verbosity=Verbosity.QUIET)
    request = _RequestActivity("tasks/get", "POST", "/", None, 0.0, finished=finished)
    token = _current_request.set(request)
    try:
        RailLogHandler(log).handle(_sdk_record(name, "push failed"))
    finally:
        _current_request.reset(token)
    assert ("push failed" in stream.getvalue()) is shown


class TestControlCharacters:
    EVIL = "hi\x1b]52;c;ZXZpbA==\x07\x1b[2J\u202eend"

    def test_previews_and_errors_are_escaped(self):
        log, stream, _ = _log()
        log.task_submitted("t1", "c1", self.EVIL)
        log.task_failed("t1", "c1", self.EVIL)
        text = stream.getvalue()
        assert "\x1b" not in text and "\x07" not in text and "\u202e" not in text
        assert "\\x1b]52" in text and "\\u202e" in text

    def test_verbose_body_and_debug_payload_are_escaped(self):
        log, stream, _ = _log(verbosity=Verbosity.DEBUG)
        log.task_completed("t1", "c1", self.EVIL + "\nsecond line", chunks=None)
        log.payload("request", b"not json \x1b[2J")
        text = stream.getvalue()
        assert "\x1b" not in text
        assert "second line" in text

    def test_log_messages_are_escaped(self):
        log, stream, _ = _log()
        record = logging.LogRecord(
            "a2a_adapter.x", logging.WARNING, __file__, 1, self.EVIL, None, None
        )
        RailLogHandler(log).handle(record)
        assert "\x1b" not in stream.getvalue()

    def test_card_values_are_escaped(self):
        backend = BackendInfo(command="claude", cwd="/tmp/\x1b[2Jx", timeout=None)
        out = render_card(_card(), backend=backend, version="1", width=120, color=True)
        assert "\x1b[2J" not in out


def test_long_states_keep_columns_aligned():
    log, stream, _ = _log()
    log.write_header()
    log._row("in", "tasks/get", task_id="t1", state="input-required", elapsed=0.002)
    log._row("in", "tasks/get", task_id="t1", state="completed", elapsed=0.002)
    lines = _rows(stream)
    header = lines[0]
    for line in lines[1:]:
        assert line.index("ms") + 2 == header.index("ELAPSED") + len("ELAPSED")
    assert "input-required" in lines[1]


def _cell_width(text):
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def test_card_measures_wide_characters():
    backend = BackendInfo(command="claude", cwd="/Users/you/项目/代码", timeout=None)
    card = build_agent_card(
        _Adapter(description="在项目中运行的智能体"), url="http://127.0.0.1:9000/"
    )
    out = render_card(card, backend=backend, version="1", width=120, color=True)
    lines = [plain(line) for line in out.splitlines() if plain(line)[:1] in "╭│├╰"]
    assert len({_cell_width(line) for line in lines}) == 1
    long_cwd = BackendInfo(command="claude", cwd="/项目" * 40, timeout=None)
    out = render_card(_card(), backend=long_cwd, version="1", width=120, color=True)
    lines = [plain(line) for line in out.splitlines() if plain(line)[:1] in "╭│├╰"]
    assert len({_cell_width(line) for line in lines}) == 1


class TestMultiLineErrors:
    ERROR = "RuntimeError: codex exited 1\nstderr line one\nstderr line two"

    def test_default_shows_first_line_and_a_hint(self):
        log, stream, _ = _log(color=True)
        log.write_header()
        log.task_failed("t1", "c1", self.ERROR)
        lines = _rows(stream)[3:]
        assert len(lines) == 1
        assert "codex exited 1" in lines[0] and "(+2 more lines, -v to show)" in lines[0]

    def test_verbose_keeps_the_rest_on_the_rail(self):
        log, stream, _ = _log(color=True, verbosity=Verbosity.VERBOSE)
        log.write_header()
        log.task_failed("t1", "c1", self.ERROR)
        lines = _rows(stream)[3:]
        assert len(lines) == 3
        assert all(line.startswith("│") for line in lines)
        assert lines[2].strip("│ ") == "stderr line two"


def test_sse_events_split_inside_a_character_survive():
    from a2a_adapter.terminal import ActivityMiddleware, _RequestActivity

    log, stream, _ = _log(verbosity=Verbosity.DEBUG)
    middleware = ActivityMiddleware(app=None, log=log)
    request = _RequestActivity("message/stream", "POST", "/", None, 0.0)
    data = 'data: {"text": "café"}\r\n\r\n'.encode()
    cut = data.index("é".encode()) + 1
    middleware._feed_sse(request, data[:cut])
    middleware._feed_sse(request, data[cut:])
    assert "café" in stream.getvalue()
    assert "\ufffd" not in stream.getvalue()


def test_uninstall_restores_uvicorn_access_logger():
    import uvicorn

    access = logging.getLogger("uvicorn.access")
    sentinel = logging.NullHandler()
    access.addHandler(sentinel)
    saved_propagate = access.propagate
    root_handlers = logging.getLogger().handlers[:]
    display = TerminalDisplay(BACKEND, stream=io.StringIO(), color=False, version="1")
    try:
        display.install_logging("warning")
        uvicorn.Config(app=None, log_config=None, access_log=False, log_level="warning")
        assert sentinel not in access.handlers
        display.uninstall_logging()
        assert sentinel in access.handlers
        assert access.propagate == saved_propagate
    finally:
        access.removeHandler(sentinel)
        logging.getLogger().handlers[:] = root_handlers


# ── Review round 2 ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text, cells",
    [("👨\u200d👩\u200d👧", 6), ("❤\ufe0f", 1), ("กั", 1), ("a\u200bb", 2), ("项目", 4)],
)
def test_zero_width_characters_take_no_cells(text, cells):
    from a2a_adapter.terminal import _cells

    assert _cells(text) == cells


def test_verbose_wraps_wide_text_by_cells():
    log, stream, _ = _log(verbosity=Verbosity.VERBOSE, width=100)
    log.task_completed("t1", "c1", "智能体" * 60, chunks=None)
    lines = _rows(stream)[1:]
    assert len(lines) > 1
    assert all(_cell_width(line) <= 100 for line in lines)
    assert "".join(line.strip() for line in lines) == "智能体" * 60


@pytest.mark.parametrize("char", ["\u200e", "\u200f", "\u061c", "\u2028", "\u2029"])
def test_direction_marks_and_line_separators_are_escaped(char):
    log, stream, _ = _log()
    log.task_failed("t1", "c1", f"a{char}b")
    assert char not in stream.getvalue()


@pytest.mark.parametrize("key", ["access_log", "log_config"])
def test_display_rejects_logging_options_it_owns(key):
    from a2a_adapter import server as server_module

    with pytest.raises(ValueError, match=key):
        server_module.serve_agent(_Adapter(), display=_QuietDisplay(), **{key: None})


# ── Review round 3 ───────────────────────────────────────────


def test_verbose_keeps_indentation_and_spacing():
    log, stream, _ = _log(verbosity=Verbosity.VERBOSE, width=100)
    reply = "def f(x):\n    return x  # same\n\n- item\n  - nested " + "word " * 30
    log.task_completed("t1", "c1", reply, chunks=None)
    lines = [line[14:] for line in _rows(stream)[1:]]
    assert "    return x  # same" in lines
    nested = next(i for i, line in enumerate(lines) if line.startswith("  - nested word"))
    assert lines[nested + 1].startswith("  word")
    assert all(_cell_width(line) <= 86 for line in lines)


def test_rejected_request_row_cuts_long_paths():
    log, stream, _ = _log(width=120)
    log.request_rejected("GET", "/" + "a" * 5000)
    output = stream.getvalue()
    assert "401 GET /" in output
    assert len(output.splitlines()[-1]) < 300


async def test_failed_row_shows_full_task_id_and_error_at_default_verbosity():
    """The executor's traceback is dropped from the rail below --debug."""
    import uuid

    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.server.events.event_queue_v2 import EventQueueSource
    from a2a.types import Message, Part, Role, SendMessageRequest

    from a2a_adapter.base_adapter import BaseA2AAdapter
    from a2a_adapter.executor import AdapterAgentExecutor

    class Fails(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            raise RuntimeError("agent died")

    log, stream, _ = _log()
    handler = RailLogHandler(log)
    executor_logger = logging.getLogger("a2a_adapter.executor")
    executor_logger.addHandler(handler)
    task_id = str(uuid.uuid4())
    context = RequestContext(
        ServerCallContext(),
        SendMessageRequest(
            message=Message(role=Role.ROLE_USER, message_id="m1", parts=[Part(text="hi")])
        ),
        task_id=task_id,
        context_id="c1",
    )
    try:
        await AdapterAgentExecutor(Fails(), log).execute(context, EventQueueSource())
    finally:
        executor_logger.removeHandler(handler)
    failed = [line for line in _rows(stream) if "failed" in line]
    assert len(failed) == 1
    assert task_id in failed[0] and "RuntimeError: agent died" in failed[0]


# ── _wrap_cells ──────────────────────────────────────────────


def _reference_wrap(text: str, width: int) -> list[str]:
    """The quadratic ``_wrap_cells`` from 0.4.1, kept as the oracle for the rewrite."""
    from a2a_adapter.terminal import _cells, _take

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


@pytest.mark.parametrize(
    "text, width, expected",
    [
        # A line that fits is kept exactly, double spaces included.
        ("a  b", 10, ["a  b"]),
        # Word wrap.
        ("aaa bbb ccc ddd", 10, ["aaa bbb", "ccc ddd"]),
        # Continuation lines keep the indent.
        ("  aaa bbb ccc ddd", 10, ["  aaa bbb", "  ccc ddd"]),
        # An indent wider than half the width is dropped.
        ("      aaa bbb", 10, ["aaa bbb"]),
        # A long word after text, and at line start.
        ("ab cdefghijklmn", 5, ["ab", "cdefg", "hijkl", "mn"]),
        ("abcdefghijkl", 5, ["abcde", "fghij", "kl"]),
        # A long word after an indent.
        ("  abcdefghij", 6, ["  abcd", "  efgh", "  ij"]),
        # CJK takes two cells per character.
        ("智能体" * 7, 10, ["智能体智能", "体智能体智", "能体智能体", "智能体智能", "体"]),
        # A 2-cell character cannot fit in 1 remaining cell.
        ("中中中", 5, ["中中", "中"]),
        # room == 1 and a 2-cell character: the one-character fallback runs one cell over.
        (" 中", 2, [" 中"]),
        # Consecutive spaces mid-line in a line that wraps.
        ("aa  bb cc dd", 6, ["aa  bb", "cc dd"]),
        # A double space at the break: the extra space is swallowed.
        ("abcde  fg", 5, ["abcde", "fg"]),
        # Empty and space-only input keep one line.
        ("", 5, [""]),
        ("       ", 3, [""]),
        # Zero-width and combining characters stay with the slice before them.
        ("abca\u0301def", 4, ["abca\u0301", "def"]),
        ("abcd\u200befgh", 4, ["abcd\u200b", "efgh"]),
        # NBSP and U+3000 are whitespace to strip() but not split points.
        ("\u00a0 aaaa", 4, ["\u00a0aaa", "a"]),
        ("\u3000 ab cd", 3, ["\u3000a", "b", "cd"]),
    ],
)
def test_wrap_cells_pinned_outputs(text, width, expected):
    from a2a_adapter.terminal import _wrap_cells

    assert _wrap_cells(text, width) == expected


def test_wrap_cells_matches_reference():
    import random

    from a2a_adapter.terminal import _wrap_cells

    alphabet = [
        "a",
        "b",
        " ",
        " ",
        "\u4e2d",  # 中, two cells
        "\u3000",  # ideographic space, two cells
        "\u00a0",  # NBSP
        "\u200b",  # zero-width space
        "\u0301",  # combining acute accent
        "\U0001f44d",  # 👍, two cells
    ]
    rng = random.Random(0)
    for _ in range(20_000):
        body = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        text = " " * rng.randint(0, 15) + body
        width = rng.randint(1, 25)
        assert _wrap_cells(text, width) == _reference_wrap(text, width), (text, width)


@pytest.mark.parametrize(
    "text, width",
    [("x" * 20_000, 80), ("a " * 10_000, 80), ("中" * 10_000, 81)],
    ids=["long-word", "spaced", "cjk-odd-width"],
)
def test_wrap_cells_does_linear_work(monkeypatch, text, width):
    from a2a_adapter import terminal

    calls = 0
    original = terminal._char_cells

    def counting(char):
        nonlocal calls
        calls += 1
        return original(char)

    monkeypatch.setattr(terminal, "_char_cells", counting)
    lines = terminal._wrap_cells(text, width)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert calls <= 4 * len(text)
