"""Tests for AntigravityAdapter (Google Antigravity CLI, ``agy``).

Covers: _build_command, _parse_invoke_output, _handle_stream_line, the
error hooks, invoke and stream with a mocked subprocess, resuming a
conversation, session persistence, metadata, loader and flat import.

The event shapes follow what agy 1.5.13 printed in live runs, trimmed: the
helpers omit the "\n" delta each agent step ends with, except in the
live-shape test below.

The stream-json shapes follow the documented headless output: one ``init``
event, ``step_update`` events (agent text in ``text_delta`` of
``agent_response`` steps), and one final ``result`` envelope.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_adapter.base_adapter import default_state_dir
from a2a_adapter.integrations.antigravity import AntigravityAdapter
from a2a_adapter.loader import load_adapter
from a2a_adapter.server import build_agent_card

# ──── Helpers ────

CONV = "c3b66b04-0000-4000-8000-000000000001"


def _make_context(task_id: str = "task-1"):
    ctx = MagicMock()
    ctx.task_id = task_id
    return ctx


def _init(conversation_id: str = CONV) -> dict:
    return {
        "event": "init",
        "conversation_id": conversation_id,
        "init": {
            "cwd": "/home/user/project",
            "tools": ["ask_permission", "run_command"],
            "permission_mode": "auto",
        },
    }


def _delta(text: str, step_index: int = 1, conversation_id: str = CONV) -> dict:
    return {
        "event": "step_update",
        "step_update": {
            "conversation_id": conversation_id,
            "step_index": step_index,
            "state": "ACTIVE",
            "step_type": "agent_response",
            "text_delta": text,
        },
    }


def _tool_step(output: str) -> dict:
    return {
        "event": "step_update",
        "step_update": {
            "conversation_id": CONV,
            "step_index": 2,
            "state": "DONE",
            "step_type": "tool",
            "tool_name": "run_command",
            "tool_info": {
                "name": "run_command",
                "parameters": {"CommandLine": "ls"},
                "output": output,
            },
        },
    }


def _result(
    response: str = "",
    status: str = "SUCCESS",
    error: str | None = None,
    conversation_id: str = CONV,
) -> dict:
    result = {
        "conversation_id": conversation_id,
        "status": status,
        "response": response,
        "duration_seconds": 1.5,
        "num_turns": 1,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    if error is not None:
        result["error"] = error
    return {"event": "result", "result": result}


def _stdout(*events: object) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _json_line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


class AsyncLineIterator:
    """Simulates proc.stdout; each readline() yields to the loop, so runs interleave."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self._lines.pop(0) if self._lines else b""


class AsyncChunkReader:
    """Simulates proc.stderr: returns its data once, then EOF."""

    def __init__(self, data: bytes = b""):
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        data, self._data = self._data, b""
        return data


def _invoke_proc(stdout: str, stderr: str = "", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    proc.pid = 4242
    proc.kill = MagicMock()
    return proc


def _stream_proc(events: list[dict], stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.pid = 4343
    proc.stdout = AsyncLineIterator([_json_line(e) for e in events])
    proc.stderr = AsyncChunkReader(stderr)
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


async def _collect(adapter, text="hi", context_id="ctx-1", task_id="task-1") -> str:
    chunks = []
    async for chunk in adapter.stream(text, context_id=context_id, context=_make_context(task_id)):
        chunks.append(chunk)
    return "".join(chunks)


@pytest.fixture
def adapter(tmp_path):
    return AntigravityAdapter(
        working_dir=str(tmp_path),
        session_store_path=str(tmp_path / "sessions.json"),
    )


# ──── _build_command ────


class TestBuildCommand:
    def test_base_command_without_session(self, adapter):
        cmd = adapter._build_command("hello world", "_default")
        assert cmd.args[0] == "agy"
        assert cmd.args[cmd.args.index("--output-format") + 1] == "stream-json"
        assert "--disable-slash-commands" in cmd.args
        # One argument, last, so cli_args can't swallow it
        assert cmd.args[-1] == "--print=hello world"
        assert "-p" not in cmd.args
        assert cmd.stdin is None
        assert cmd.used_resume is False
        assert "--conversation" not in cmd.args
        assert "--dangerously-skip-permissions" not in cmd.args

    def test_command_with_session_resumes_conversation(self, adapter):
        adapter._sessions["ctx"] = CONV
        cmd = adapter._build_command("next", "ctx")
        assert cmd.args[cmd.args.index("--conversation") + 1] == CONV
        assert cmd.used_resume is True
        assert cmd.args[-1] == "--print=next"

    @pytest.mark.parametrize("prompt", ["--help", "-x", "- list items", "a=b", ""])
    def test_prompt_is_attached_to_the_flag(self, adapter, prompt):
        # agy 1.5.13 rejects `-p --help` ("flags provided but not defined")
        # but takes `--print=--help` as the prompt.
        assert adapter._build_command(prompt, "_default").args[-1] == f"--print={prompt}"

    def test_cli_args_come_before_managed_flags(self, tmp_path):
        a = AntigravityAdapter(
            working_dir=str(tmp_path),
            cli_args=["--model", "gemini-3-pro"],
            agy_path="/opt/agy",
            session_store_path=str(tmp_path / "s.json"),
        )
        cmd = a._build_command("hi", "_default")
        assert cmd.args[:3] == ["/opt/agy", "--model", "gemini-3-pro"]

    def test_cli_args_must_not_be_a_string(self, tmp_path):
        with pytest.raises(TypeError):
            AntigravityAdapter(working_dir=str(tmp_path), cli_args="--model x")

    def test_skip_permissions_flag_when_enabled(self, tmp_path):
        a = AntigravityAdapter(
            working_dir=str(tmp_path),
            skip_permissions=True,
            session_store_path=str(tmp_path / "s.json"),
        )
        assert "--dangerously-skip-permissions" in a._build_command("hi", "k").args

    @pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("no", False)])
    def test_skip_permissions_via_env_var(self, tmp_path, monkeypatch, value, expected):
        monkeypatch.setenv("A2A_ANTIGRAVITY_SKIP_PERMISSIONS", value)
        a = AntigravityAdapter(working_dir=str(tmp_path))
        assert a.skip_permissions is expected

    def test_explicit_skip_permissions_overrides_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A2A_ANTIGRAVITY_SKIP_PERMISSIONS", "1")
        a = AntigravityAdapter(working_dir=str(tmp_path), skip_permissions=False)
        assert a.skip_permissions is False

    def test_skip_permissions_logs_warning(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            AntigravityAdapter(working_dir=str(tmp_path), skip_permissions=True)
        assert "--dangerously-skip-permissions" in caplog.text

    def test_prompt_limit_counts_the_flag_prefix(self, adapter):
        # The argument is "--print=" + message; Linux caps one at 128 KiB incl. NUL.
        limit = 128 * 1024 - 1 - len("--print=")
        assert adapter._build_command("x" * limit, "_default").args[-1].startswith("--print=")
        with pytest.raises(ValueError, match="too large"):
            adapter._build_command("x" * (limit + 1), "_default")

    def test_prompt_over_the_argument_limit_is_rejected(self, adapter):
        # agy takes the prompt only as an argument; Linux caps one at 128 KiB.
        with pytest.raises(ValueError, match="too large"):
            adapter._build_command("x" * (128 * 1024), "_default")


# ──── _parse_invoke_output ────


class TestParseInvokeOutput:
    def test_joins_agent_text_deltas_and_reads_conversation_id(self, adapter):
        out = _stdout(_init(), _delta("Hello "), _delta("world"), _result("Hello world"))
        result = adapter._parse_invoke_output(out, "k")
        assert result.text == "Hello world"
        assert result.session_id == CONV

    def test_tool_output_is_not_part_of_the_reply(self, adapter):
        out = _stdout(_init(), _tool_step("secret.txt"), _delta("Done."), _result("Done."))
        assert adapter._parse_invoke_output(out, "k").text == "Done."

    def test_falls_back_to_result_response_without_deltas(self, adapter):
        out = _stdout(_init(), _result("Only in the envelope"))
        assert adapter._parse_invoke_output(out, "k").text == "Only in the envelope"

    def test_empty_success_raises(self, adapter):
        # Known agy bug: SUCCESS with an empty response.
        out = _stdout(_init(), _result(""))
        with pytest.raises(RuntimeError, match="no output"):
            adapter._parse_invoke_output(out, "k")

    def test_error_status_raises_with_its_message(self, adapter):
        out = _stdout(_init(), _result(status="ERROR", error="quota exceeded"))
        with pytest.raises(RuntimeError, match="quota exceeded"):
            adapter._parse_invoke_output(out, "k")

    def test_waiting_status_raises(self, adapter):
        out = _stdout(_init(), _delta("Which file?"), _result("", status="WAITING"))
        with pytest.raises(RuntimeError, match="waiting on input"):
            adapter._parse_invoke_output(out, "k")

    def test_empty_success_still_stores_the_conversation(self, adapter):
        # The turn ran (and may have edited files); the next message resumes it.
        with pytest.raises(RuntimeError, match="no output"):
            adapter._parse_invoke_output(_stdout(_init(), _result("")), "k")
        assert adapter._sessions["k"] == CONV

    def test_failed_turn_does_not_store_the_conversation(self, adapter):
        out = _stdout(_init(), _result(status="ERROR", error="boom"))
        with pytest.raises(RuntimeError):
            adapter._parse_invoke_output(out, "k")
        assert "k" not in adapter._sessions

    @pytest.mark.parametrize("status", ["CANCELED", "INTERRUPTED", "INVALID", "RUNNING"])
    def test_other_non_success_statuses_raise(self, adapter, status):
        out = _stdout(_init(), _delta("partial"), _result("partial", status=status))
        with pytest.raises(RuntimeError, match=status):
            adapter._parse_invoke_output(out, "k")

    def test_missing_result_event_raises(self, adapter):
        # A run cut short never prints its envelope: don't pass partial text off as done.
        out = _stdout(_init(), _delta("half an ans"))
        with pytest.raises(RuntimeError, match="result"):
            adapter._parse_invoke_output(out, "k")

    def test_ignores_non_json_and_unknown_events(self, adapter):
        out = "warning: something\n" + _stdout(
            {"event": "telemetry", "x": 1}, ["not", "a", "dict"], _delta("ok"), _result("ok")
        )
        assert adapter._parse_invoke_output(out, "k").text == "ok"

    def test_empty_conversation_id_is_not_a_session(self, adapter):
        out = _stdout(_init(""), _delta("ok"), _result("ok", conversation_id=""))
        assert adapter._parse_invoke_output(out, "k").session_id is None

    def test_conversation_id_from_result_when_init_missing(self, adapter):
        out = _stdout(_delta("ok"), _result("ok"))
        assert adapter._parse_invoke_output(out, "k").session_id == CONV


# ──── _handle_stream_line ────


class TestHandleStreamLine:
    def test_init_does_not_store_the_conversation(self, adapter):
        # Stored only once a turn succeeds, as invoke() does.
        assert adapter._handle_stream_line(json.dumps(_init())) is None

    def test_agent_delta_yields_text(self, adapter):
        assert adapter._handle_stream_line(json.dumps(_delta("Hi"))).text == "Hi"

    def test_tool_step_is_skipped(self, adapter):
        assert adapter._handle_stream_line(json.dumps(_tool_step("x"))) is None

    def test_done_step_without_delta_is_skipped(self, adapter):
        line = _delta("")
        line["step_update"]["state"] = "DONE"
        del line["step_update"]["text_delta"]
        assert adapter._handle_stream_line(json.dumps(line)) is None

    def test_success_result_carries_session_id_only(self, adapter):
        event = adapter._handle_stream_line(json.dumps(_result("full text")))
        assert event.text is None and event.error is None
        assert event.session_id == CONV

    def test_error_result_is_an_error(self, adapter):
        event = adapter._handle_stream_line(json.dumps(_result(status="ERROR", error="boom")))
        assert "boom" in event.error

    def test_waiting_result_is_an_error(self, adapter):
        event = adapter._handle_stream_line(json.dumps(_result(status="WAITING")))
        assert "WAITING" in event.error

    def test_non_json_line_is_skipped(self, adapter):
        assert adapter._handle_stream_line("not json") is None
        assert adapter._handle_stream_line("[1, 2]") is None


# ──── Error hooks ────


class TestErrorHooks:
    def test_extract_error_reads_result_error(self, adapter):
        out = _stdout(_init(), _result(status="ERROR", error="unknown model 'x'"))
        assert adapter._extract_error(out) == "unknown model 'x'"

    def test_extract_error_none_without_result(self, adapter):
        assert adapter._extract_error(_stdout(_init())) is None

    def test_missing_conversation_warning_is_not_a_stale_error(self, adapter):
        # agy 1.5.13 doesn't fail on an unknown --conversation: it warns and
        # starts a new conversation. Matching the warning would re-run a turn
        # that already ran (the base retries an empty reply when this is True).
        adapter._sessions["k"] = CONV
        stderr = f'warning: conversation "{CONV}" not found'
        assert adapter._is_stale_session_error(_stdout(_init("new"), _result("")), stderr) is False

    def test_binary_not_found_message_names_the_path(self, adapter):
        assert "'agy'" in adapter._binary_not_found_message()


# ──── invoke() with a mocked subprocess ────


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_success_saves_conversation(self, adapter):
        proc = _invoke_proc(_stdout(_init(), _delta("42"), _result("42")))
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc) as spawn:
            text = await adapter.invoke("q", context_id="ctx-1", context=_make_context())
        assert text == "42"
        assert adapter._sessions["ctx-1"] == CONV
        args = spawn.call_args.args
        assert args[-1] == "--print=q"
        assert spawn.call_args.kwargs["cwd"] == adapter.working_dir

    @pytest.mark.asyncio
    async def test_second_message_resumes_the_conversation(self, adapter):
        first = _invoke_proc(_stdout(_init(), _delta("a"), _result("a")))
        second = _invoke_proc(_stdout(_init(), _delta("b"), _result("b")))
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", side_effect=[first, second]
        ) as spawn:
            await adapter.invoke("one", context_id="ctx", context=_make_context("t1"))
            await adapter.invoke("two", context_id="ctx", context=_make_context("t2"))
        second_args = spawn.call_args_list[1].args
        assert second_args[second_args.index("--conversation") + 1] == CONV

    @pytest.mark.asyncio
    async def test_non_zero_exit_reports_envelope_error_over_stderr(self, adapter):
        proc = _invoke_proc(
            _stdout(_init(), _result(status="ERROR", error="unknown model 'nope'")),
            stderr="some progress noise",
            returncode=1,
        )
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="unknown model 'nope'"):
                await adapter.invoke("q", context_id="c", context=_make_context())

    @pytest.mark.asyncio
    async def test_auth_failure_reports_stderr(self, adapter):
        proc = _invoke_proc("", stderr="authentication required", returncode=1)
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="authentication required"):
                await adapter.invoke("q", context_id="c", context=_make_context())

    @pytest.mark.asyncio
    async def test_oversized_prompt_fails_before_spawning(self, adapter):
        with patch("a2a_adapter.base_adapter.create_subprocess_exec") as spawn:
            with pytest.raises(ValueError, match="too large"):
                await adapter.invoke("x" * (128 * 1024), context_id="c", context=_make_context())
            with pytest.raises(ValueError, match="too large"):
                await _collect(adapter, "x" * (128 * 1024))
        spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_binary_not_found(self, adapter):
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", side_effect=FileNotFoundError()
        ):
            with pytest.raises(FileNotFoundError, match="Antigravity CLI"):
                await adapter.invoke("q", context_id="c", context=_make_context())


# ──── stream() with a mocked subprocess ────


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_deltas_and_saves_conversation(self, adapter):
        proc = _stream_proc(
            [_init(), _delta("Hel"), _tool_step("x"), _delta("lo"), _result("Hello")]
        )
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            assert await _collect(adapter) == "Hello"
        assert adapter._sessions["ctx-1"] == CONV

    @pytest.mark.asyncio
    async def test_stream_error_result_raises(self, adapter):
        proc = _stream_proc(
            [_init(), _result(status="ERROR", error="quota exceeded")], returncode=1
        )
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                await _collect(adapter)

    @pytest.mark.asyncio
    async def test_stream_empty_success_raises(self, adapter):
        proc = _stream_proc([_init(), _result("")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="no output"):
                await _collect(adapter)

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_result_response_without_deltas(self, adapter):
        proc = _stream_proc([_init(), _tool_step("x"), _result("Only in the envelope")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            assert await _collect(adapter) == "Only in the envelope"
        assert adapter._sessions["ctx-1"] == CONV

    @pytest.mark.asyncio
    async def test_stream_does_not_repeat_response_after_deltas(self, adapter):
        proc = _stream_proc([_init(), _delta("Hi"), _result("Hi")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            assert await _collect(adapter) == "Hi"

    @pytest.mark.asyncio
    async def test_stream_without_result_event_raises(self, adapter):
        proc = _stream_proc([_init(), _delta("half an ans")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="result"):
                await _collect(adapter)

    @pytest.mark.asyncio
    async def test_stream_failed_turn_does_not_store_the_conversation(self, adapter):
        proc = _stream_proc([_init(), _result(status="ERROR", error="boom")], returncode=1)
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="boom"):
                await _collect(adapter)
        assert "ctx-1" not in adapter._sessions

    @pytest.mark.asyncio
    async def test_stream_empty_success_stores_the_conversation(self, adapter):
        proc = _stream_proc([_init(), _result("")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="no output"):
                await _collect(adapter)
        assert adapter._sessions["ctx-1"] == CONV

    @pytest.mark.asyncio
    async def test_concurrent_streams_keep_their_own_state(self, tmp_path):
        a = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "s"))
        procs = [
            _stream_proc(
                [
                    _init("c-1"),
                    _delta("one", conversation_id="c-1"),
                    _result("one", conversation_id="c-1"),
                ]
            ),
            _stream_proc([_init("c-2"), _result("two", conversation_id="c-2")]),
        ]
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", side_effect=procs):
            results = await asyncio.gather(
                _collect(a, context_id="x", task_id="t1"), _collect(a, context_id="y", task_id="t2")
            )
        assert results == ["one", "two"]

    @pytest.mark.asyncio
    async def test_live_shape_multi_step_reply_matches_in_both_modes(self, tmp_path):
        # Recorded shape: each agent step ends with a "\n" delta, a tool step in
        # between, and the envelope's response is all agent text joined.
        def done_newline(i):
            e = _delta("\n", step_index=i)
            e["step_update"]["state"] = "DONE"
            return e

        events = [
            _init(),
            _delta("Starting now.", step_index=1),
            done_newline(1),
            _tool_step("ls output"),
            _delta("Finished now.", step_index=3),
            done_newline(3),
            _result("Starting now.\nFinished now.\n"),
        ]
        a1 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "1"))
        a2 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "2"))
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=_invoke_proc(_stdout(*events)),
        ):
            invoked = await a1.invoke("q", context_id="c", context=_make_context())
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", return_value=_stream_proc(events)
        ):
            streamed = await _collect(a2)
        assert invoked == streamed == "Starting now.\nFinished now.\n"

    @pytest.mark.asyncio
    async def test_invoke_and_stream_give_the_same_reply(self, tmp_path):
        events = [_init(), _delta("The answer "), _delta("is 42."), _result("The answer is 42.")]
        a1 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "1"))
        a2 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "2"))
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=_invoke_proc(_stdout(*events)),
        ):
            invoked = await a1.invoke("q", context_id="c", context=_make_context())
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", return_value=_stream_proc(events)
        ):
            streamed = await _collect(a2)
        assert invoked == streamed == "The answer is 42."


# ──── Resume, retry and stream lifecycle ────

GONE = "0b5e1e7e-dead-4000-8000-000000000000"
MISSING_WARNING = f'warning: conversation "{GONE}" not found'


class TestResumeAndLifecycle:
    @pytest.mark.asyncio
    async def test_missing_conversation_starts_a_new_one_and_warns(self, adapter, caplog):
        # What agy 1.5.13 does for an unknown --conversation: a warning on
        # stderr, a new conversation, exit 0.
        adapter._sessions["ctx"] = GONE
        proc = _invoke_proc(_stdout(_init(), _delta("hi"), _result("hi")), stderr=MISSING_WARNING)
        with caplog.at_level("WARNING"):
            with patch(
                "a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc
            ) as spawn:
                text = await adapter.invoke("q", context_id="ctx", context=_make_context())
        assert text == "hi"
        assert spawn.call_count == 1
        assert adapter._sessions["ctx"] == CONV
        assert GONE in caplog.text and "new conversation" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_missing_conversation_starts_a_new_one_and_warns(self, adapter, caplog):
        adapter._sessions["ctx-1"] = GONE
        proc = _stream_proc([_init(), _delta("hi"), _result("hi")], stderr=MISSING_WARNING.encode())
        with caplog.at_level("WARNING"):
            with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
                assert await _collect(adapter) == "hi"
        assert adapter._sessions["ctx-1"] == CONV
        assert GONE in caplog.text and "new conversation" in caplog.text

    @pytest.mark.asyncio
    async def test_resumed_conversation_does_not_warn(self, adapter, caplog):
        adapter._sessions["ctx"] = CONV
        proc = _invoke_proc(_stdout(_init(), _delta("hi"), _result("hi")))
        with caplog.at_level("WARNING"):
            with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
                await adapter.invoke("q", context_id="ctx", context=_make_context())
        assert "new conversation" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["invoke", "stream"])
    async def test_empty_reply_in_a_new_conversation_is_not_retried(self, adapter, mode):
        # The turn ran (and may have changed files): running it again would repeat it.
        key = "ctx-1"
        adapter._sessions[key] = GONE
        if mode == "invoke":
            proc = _invoke_proc(_stdout(_init(), _result("")), stderr=MISSING_WARNING)
        else:
            proc = _stream_proc([_init(), _result("")], stderr=MISSING_WARNING.encode())
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc) as spawn:
            with pytest.raises(RuntimeError, match="no output"):
                if mode == "invoke":
                    await adapter.invoke("q", context_id=key, context=_make_context())
                else:
                    await _collect(adapter, context_id=key)
        assert spawn.call_count == 1
        assert adapter._sessions[key] == CONV

    @pytest.mark.asyncio
    async def test_resumed_run_that_dies_silently_retries_once_fresh(self, adapter):
        # The base adapter's rule: a resumed run with a non-zero exit and no
        # output at all retries once without --conversation.
        adapter._sessions["ctx-1"] = GONE
        dead = _stream_proc([], returncode=1)
        fresh = _stream_proc([_init(), _delta("fresh"), _result("fresh")])
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", side_effect=[dead, fresh]
        ) as spawn:
            assert await _collect(adapter) == "fresh"
        assert "--conversation" not in spawn.call_args_list[1].args
        assert adapter._sessions["ctx-1"] == CONV

    @pytest.mark.asyncio
    async def test_resumed_stream_that_died_after_running_tools_is_not_retried(self, adapter):
        # No agent text yet, but a tool ran: running the turn again would repeat it.
        adapter._sessions["ctx-1"] = CONV
        proc = _stream_proc([_init(), _tool_step("rm -rf build")], returncode=1)
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc) as spawn:
            with pytest.raises(RuntimeError, match="not retried"):
                await _collect(adapter)
        assert spawn.call_count == 1
        assert adapter._sessions["ctx-1"] == CONV

    @pytest.mark.asyncio
    async def test_retried_stream_gets_fresh_run_state(self, adapter):
        # The first attempt's state must not carry over into the retry.
        adapter._sessions["ctx-1"] = GONE
        dead = _stream_proc([], returncode=1)
        fresh = _stream_proc([_init(), _delta("cut short")])
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", side_effect=[dead, fresh]):
            with pytest.raises(RuntimeError, match="without a result"):
                await _collect(adapter)

    @pytest.mark.asyncio
    async def test_each_step_in_its_own_task(self, adapter):
        # Consumers like create_task(gen.__anext__()) run every step in a copied context.
        events = [_init(), _delta("Hel"), _delta("lo"), _result("Hello")]
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec", return_value=_stream_proc(events)
        ):
            gen = adapter.stream("q", context_id="ctx-1", context=_make_context())
            chunks = []
            while True:
                try:
                    chunks.append(await asyncio.create_task(gen.__anext__()))
                except StopAsyncIteration:
                    break
        assert chunks == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_closing_the_stream_early_kills_agy(self, adapter):
        events = [_init(), _delta("Hel"), _delta("lo"), _result("Hello")]
        with (
            patch(
                "a2a_adapter.base_adapter.create_subprocess_exec",
                return_value=_stream_proc(events),
            ),
            patch("a2a_adapter.base_adapter._kill_process_tree") as kill,
        ):
            gen = adapter.stream("q", context_id="ctx-1", context=_make_context())
            assert await gen.__anext__() == "Hel"
            await gen.aclose()
            kill.assert_called_once()
            assert adapter._active_processes == {}

    @pytest.mark.asyncio
    async def test_cancelling_a_step_kills_agy(self, adapter):
        class _Blocking(AsyncLineIterator):
            async def readline(self) -> bytes:
                if not self._lines:
                    await asyncio.Event().wait()  # agy still working
                return await super().readline()

        proc = _stream_proc([])
        proc.stdout = _Blocking([_json_line(_init()), _json_line(_delta("Hel"))])
        with (
            patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc),
            patch("a2a_adapter.base_adapter._kill_process_tree") as kill,
        ):
            gen = adapter.stream("q", context_id="ctx-1", context=_make_context())
            assert await gen.__anext__() == "Hel"
            step = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.01)
            step.cancel()
            with pytest.raises(asyncio.CancelledError):
                await step
            kill.assert_called_once()
            assert adapter._active_processes == {}

    @pytest.mark.asyncio
    async def test_invoke_between_chunks_does_not_touch_the_stream(self, tmp_path):
        streamer = AntigravityAdapter(
            working_dir=str(tmp_path), session_store_path=str(tmp_path / "a")
        )
        other = AntigravityAdapter(
            working_dir=str(tmp_path), session_store_path=str(tmp_path / "b")
        )
        stream_proc = _stream_proc([_init(), _delta("Hel"), _delta("lo"), _result("Hello")])
        invoke_proc = _invoke_proc(
            _stdout(_init("c-9"), _delta("x"), _result("x", conversation_id="c-9"))
        )
        chunks = []
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=[stream_proc, invoke_proc],
        ):
            async for chunk in streamer.stream("q", context_id="s", context=_make_context("t1")):
                chunks.append(chunk)
                if len(chunks) == 2:  # after the last delta, before the result line
                    await other.invoke("q", context_id="o", context=_make_context("t2"))
        assert chunks == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_two_streams_interleaved_in_one_task(self, tmp_path):
        a = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=str(tmp_path / "s"))
        procs = [
            _stream_proc(
                [
                    _init("c-1"),
                    _delta("a1", conversation_id="c-1"),
                    _delta("a2", conversation_id="c-1"),
                    _result("a", conversation_id="c-1"),
                ]
            ),
            _stream_proc(
                [
                    _init("c-2"),
                    _delta("b1", conversation_id="c-2"),
                    _result("b", conversation_id="c-2"),
                ]
            ),
        ]
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", side_effect=procs):
            g1 = a.stream("q", context_id="x", context=_make_context("t1"))
            g2 = a.stream("q", context_id="y", context=_make_context("t2"))
            first = [await g1.__anext__(), await g2.__anext__()]
            rest1 = [c async for c in g1]
            rest2 = [c async for c in g2]
        assert first == ["a1", "b1"] and rest1 == ["a2"] and rest2 == []

    @pytest.mark.asyncio
    async def test_unrelated_error_does_not_drop_the_conversation(self, adapter):
        adapter._sessions["ctx"] = CONV
        proc = _invoke_proc(_stdout(_result(status="ERROR", error="quota exceeded")), returncode=1)
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                await adapter.invoke("q", context_id="ctx", context=_make_context())
        assert adapter._sessions["ctx"] == CONV


# ──── Persistence, metadata, wiring ────


class TestSessionsAndMetadata:
    @pytest.mark.asyncio
    async def test_conversation_survives_restart(self, tmp_path):
        store = str(tmp_path / "sessions.json")
        a1 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=store)
        proc = _invoke_proc(_stdout(_init(), _delta("a"), _result("a")))
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            await a1.invoke("q", context_id="ctx", context=_make_context())
        a2 = AntigravityAdapter(working_dir=str(tmp_path), session_store_path=store)
        assert a2._build_command("next", "ctx").used_resume is True

    def test_default_session_store_path(self, tmp_path):
        a = AntigravityAdapter(working_dir=str(tmp_path))
        expected = os.path.join(default_state_dir("antigravity", str(tmp_path)), "sessions.json")
        assert a._session_store_path == expected

    def test_kills_tool_processes_that_left_its_group(self, adapter):
        # agy starts each terminal-tool command in a session of its own, so a
        # group kill alone leaves it running after cancel or timeout.
        assert adapter._kill_detached_descendants is True

    def test_metadata_defaults(self, adapter):
        meta = adapter.get_metadata()
        assert meta.name == "Antigravity"
        assert meta.streaming is True
        assert adapter.supports_streaming() is True

    def test_metadata_custom(self, tmp_path):
        skills = [{"id": "s", "name": "S", "description": "d", "tags": ["t"]}]
        a = AntigravityAdapter(
            working_dir=str(tmp_path), name="Reviewer", description="Reviews", skills=skills
        )
        meta = a.get_metadata()
        assert (meta.name, meta.description, meta.skills) == ("Reviewer", "Reviews", skills)

    def test_build_agent_card(self, adapter):
        card = build_agent_card(adapter)
        assert card.name == "Antigravity"
        assert card.capabilities.streaming is True

    def test_load_adapter(self, tmp_path):
        a = load_adapter(
            {"adapter": "antigravity", "working_dir": str(tmp_path), "skip_permissions": True}
        )
        assert isinstance(a, AntigravityAdapter)
        assert a.skip_permissions is True

    def test_flat_imports(self):
        from a2a_adapter import AntigravityAdapter as Flat
        from a2a_adapter.integrations import AntigravityAdapter as Pkg

        assert Flat is Pkg is AntigravityAdapter
