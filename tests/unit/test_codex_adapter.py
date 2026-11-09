"""Tests for the v0.2 CodexAdapter.

Covers: _build_command, _parse_result, invoke (subprocess mock),
supports_streaming() returns False, per-context serial execution (D1),
cancel running task (D2 _killed_tasks), cancel queued task (D2 _cancelled_tasks),
stale session retry, session persistence, loader integration, and flat import.
"""

import asyncio
import json
import os
import stat
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a_adapter.integrations.codex import CodexAdapter
from a2a_adapter.base_adapter import CommandResult, ParseResult, default_state_dir
from a2a_adapter.exceptions import CancelledByAdapterError
from a2a_adapter.loader import load_adapter
from a2a_adapter.server import build_agent_card, to_a2a


# ──── Helpers ────


def _make_context(task_id: str = "task-1"):
    """Create a mock RequestContext with a task_id attribute."""
    ctx = MagicMock()
    ctx.task_id = task_id
    return ctx


def _thread_started_event(thread_id: str = "thread-abc") -> dict:
    return {
        "type": "thread.started",
        "thread_id": thread_id,
    }


def _thread_started_nested_event(thread_id: str = "thread-nested") -> dict:
    return {
        "type": "thread.started",
        "thread": {"id": thread_id},
    }


def _item_completed_event(text: str) -> dict:
    return {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": text,
        },
    }


def _turn_failed_event(message: str = "turn failed") -> dict:
    return {
        "type": "turn.failed",
        "error": {"message": message},
    }


# ──── Fixtures ────


@pytest.fixture
def adapter(tmp_path):
    """Create a CodexAdapter with a tmp_path working directory."""
    return CodexAdapter(
        working_dir=str(tmp_path),
        session_store_path=str(tmp_path / "sessions.json"),
    )


@pytest.fixture
def mock_proc():
    """Create a mock subprocess with sane defaults."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.pid = 12345
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# ──── Test: _build_command ────


class TestBuildCommand:
    def test_first_invocation(self, adapter):
        """First invocation: codex exec --json <prompt> (prompt last)."""
        cmd = adapter._build_command("hello world", "_default")
        assert isinstance(cmd, CommandResult)
        assert cmd.args[0] == "codex"
        assert cmd.args[1] == "exec"
        # No resume on first invocation
        assert "resume" not in cmd.args
        assert "--json" in cmd.args
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args
        assert "--skip-git-repo-check" not in cmd.args
        # The prompt goes through stdin; "-" tells Codex to read it from there
        assert cmd.args[-1] == "-"
        assert cmd.stdin == "hello world"
        assert cmd.used_resume is False

    def test_resume_with_thread_id(self, adapter):
        """Resume: codex exec resume <thread_id> --json <prompt>."""
        adapter._sessions["ctx-1"] = "thread-xyz"
        cmd = adapter._build_command("continue", "ctx-1")
        assert cmd.args[0] == "codex"
        assert cmd.args[1] == "exec"
        assert cmd.args[2] == "resume"
        assert cmd.args[3] == "thread-xyz"
        assert "--json" in cmd.args
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args
        assert "--skip-git-repo-check" not in cmd.args
        # The prompt still goes through stdin
        assert cmd.args[-1] == "-"
        assert cmd.stdin == "continue"
        assert cmd.used_resume is True

    def test_custom_codex_path(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            codex_path="/usr/local/bin/codex",
            session_store_path=str(tmp_path / "s.json"),
        )
        cmd = a._build_command("msg", "_default")
        assert cmd.args[0] == "/usr/local/bin/codex"

    def test_prompt_with_special_chars(self, adapter):
        """Prompt with spaces and special chars goes through stdin unchanged."""
        cmd = adapter._build_command("fix the bug in main.py --verbose", "_default")
        assert cmd.args[-1] == "-"
        assert cmd.stdin == "fix the bug in main.py --verbose"

    def test_dangerous_flags_omitted_by_default(self, adapter):
        cmd = adapter._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args
        assert "--skip-git-repo-check" not in cmd.args
        assert "--json" in cmd.args

    def test_bypass_approvals_when_enabled(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            bypass_approvals=True,
            session_store_path=str(tmp_path / "s.json"),
        )
        cmd = a._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd.args
        assert "--skip-git-repo-check" not in cmd.args

    def test_skip_git_check_when_enabled(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            skip_git_check=True,
            session_store_path=str(tmp_path / "s.json"),
        )
        cmd = a._build_command("hello", "_default")
        assert "--skip-git-repo-check" in cmd.args
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args

    def test_both_flags_when_enabled(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            bypass_approvals=True,
            skip_git_check=True,
            session_store_path=str(tmp_path / "s.json"),
        )
        cmd = a._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd.args
        assert "--skip-git-repo-check" in cmd.args
        assert cmd.args[-1] == "-"
        assert cmd.stdin == "hello"

    def test_bypass_approvals_via_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A2A_CODEX_BYPASS_APPROVALS", "1")
        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.bypass_approvals is True
        cmd = a._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd.args

    def test_skip_git_check_via_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A2A_CODEX_SKIP_GIT_CHECK", "true")
        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.skip_git_check is True

    def test_env_vars_unset_flags_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("A2A_CODEX_BYPASS_APPROVALS", raising=False)
        monkeypatch.delenv("A2A_CODEX_SKIP_GIT_CHECK", raising=False)
        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.bypass_approvals is False
        assert a.skip_git_check is False

    def test_bypass_approvals_env_var_false_like(self, tmp_path, monkeypatch):
        for val in ("0", "false", "False", "no", ""):
            monkeypatch.setenv("A2A_CODEX_BYPASS_APPROVALS", val)
            a = CodexAdapter(
                working_dir=str(tmp_path),
                session_store_path=str(tmp_path / "s.json"),
            )
            assert a.bypass_approvals is False, f"Expected False for env var={val!r}"

    def test_skip_git_check_env_var_false_like(self, tmp_path, monkeypatch):
        for val in ("0", "false", "False", "no", ""):
            monkeypatch.setenv("A2A_CODEX_SKIP_GIT_CHECK", val)
            a = CodexAdapter(
                working_dir=str(tmp_path),
                session_store_path=str(tmp_path / "s.json"),
            )
            assert a.skip_git_check is False, f"Expected False for env var={val!r}"

    def test_bypass_approvals_explicit_overrides_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A2A_CODEX_BYPASS_APPROVALS", "1")
        a = CodexAdapter(
            working_dir=str(tmp_path),
            bypass_approvals=False,
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.bypass_approvals is False
        cmd = a._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args

    def test_skip_git_check_explicit_overrides_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A2A_CODEX_SKIP_GIT_CHECK", "1")
        a = CodexAdapter(
            working_dir=str(tmp_path),
            skip_git_check=False,
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.skip_git_check is False
        cmd = a._build_command("hello", "_default")
        assert "--skip-git-repo-check" not in cmd.args

    def test_bypass_approvals_logs_warning(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            CodexAdapter(
                working_dir=str(tmp_path),
                bypass_approvals=True,
                session_store_path=str(tmp_path / "s.json"),
            )
        assert "ENABLED" in caplog.text

    def test_skip_git_check_logs_warning(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            CodexAdapter(
                working_dir=str(tmp_path),
                skip_git_check=True,
                session_store_path=str(tmp_path / "s.json"),
            )
        assert "ENABLED" in caplog.text

    def test_prompt_still_last_with_flags(self, tmp_path):
        for bypass, skip in [(True, True), (True, False), (False, True), (False, False)]:
            a = CodexAdapter(
                working_dir=str(tmp_path),
                bypass_approvals=bypass,
                skip_git_check=skip,
                session_store_path=str(tmp_path / "s.json"),
            )
            cmd = a._build_command("the prompt", "_default")
            assert cmd.args[-1] == "-"
            assert cmd.stdin == "the prompt"

    def test_env_vars_param_does_not_enable_dangerous_flags(self, tmp_path, monkeypatch):
        """env_vars is for the subprocess environment, not adapter config."""
        monkeypatch.delenv("A2A_CODEX_BYPASS_APPROVALS", raising=False)
        monkeypatch.delenv("A2A_CODEX_SKIP_GIT_CHECK", raising=False)
        a = CodexAdapter(
            working_dir=str(tmp_path),
            env_vars={
                "A2A_CODEX_BYPASS_APPROVALS": "1",
                "A2A_CODEX_SKIP_GIT_CHECK": "1",
            },
            session_store_path=str(tmp_path / "s.json"),
        )
        assert a.bypass_approvals is False
        assert a.skip_git_check is False
        cmd = a._build_command("hello", "_default")
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd.args
        assert "--skip-git-repo-check" not in cmd.args


# ──── Test: the prompt can never be parsed as CLI arguments ────

_STUB_CODEX = """#!{python}
# Mimics how Codex's clap parser treats the final argv element.
import json, os, sys

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "argv.jsonl"), "a") as log:
    log.write(json.dumps(sys.argv) + "\\n")

last = sys.argv[-1]
if last == "-":
    prompt = sys.stdin.buffer.read().decode("utf-8")
    if not prompt:
        print("No prompt provided via stdin", file=sys.stderr)
        sys.exit(1)
elif last in ("--help", "--version", "help"):
    print("usage text")
    sys.exit(0)
elif last in ("review", "resume", "fork"):
    print("subcommand needs options", file=sys.stderr)
    sys.exit(1)
elif last.startswith("-"):
    print("error: unexpected argument", file=sys.stderr)
    sys.exit(2)
else:
    prompt = last

print(json.dumps({{"type": "thread.started", "thread_id": "thread-stub"}}))
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": prompt}}}}))
"""


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX shebang script")
class TestPromptNeverParsedAsArgs:
    """A real subprocess, against a stub that parses argv the way Codex does."""

    @pytest.fixture
    def stub_adapter(self, tmp_path):
        stub = tmp_path / "codex"
        stub.write_text(_STUB_CODEX.format(python=sys.executable))
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return CodexAdapter(
            working_dir=str(tmp_path),
            codex_path=str(stub),
            session_store_path=str(tmp_path / "s.json"),
        )

    @staticmethod
    def _recorded_argv(adapter):
        log = os.path.join(os.path.dirname(adapter.codex_path), "argv.jsonl")
        with open(log) as f:
            return [json.loads(line) for line in f]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt",
        [
            "--help",
            "--version",
            "-c foo=bar",
            "--dangerously-bypass-approvals-and-sandbox",
            "-",
            "--",
            "help",
            "review",
            "resume",
            "fork",
        ],
    )
    async def test_prompt_reaches_codex_verbatim(self, stub_adapter, prompt):
        result = await stub_adapter.invoke(prompt, context_id="ctx")
        assert result == prompt

    @pytest.mark.asyncio
    async def test_resumed_run_keeps_a_flag_like_prompt(self, stub_adapter):
        stub_adapter._sessions["ctx"] = "thread-1"
        result = await stub_adapter.invoke("--help", context_id="ctx")
        assert result == "--help"
        recorded = self._recorded_argv(stub_adapter)
        assert len(recorded) == 1
        index = recorded[0].index("resume")
        assert recorded[0][index + 1] == "thread-1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt",
        ["x" * (2 * 1024 * 1024), "line one\nline two\n\u4f60\u597d \u00e9"],
        ids=["2MiB", "newlines-and-non-ascii"],
    )
    async def test_large_and_multiline_prompts_arrive_intact(self, stub_adapter, prompt):
        assert await stub_adapter.invoke(prompt, context_id="ctx") == prompt

    @pytest.mark.asyncio
    async def test_prompt_never_appears_in_argv(self, stub_adapter):
        sentinel = "sentinel-3f9a1c prompt text"
        assert await stub_adapter.invoke(sentinel, context_id="ctx") == sentinel
        recorded = self._recorded_argv(stub_adapter)
        assert len(recorded) == 1
        assert not any("sentinel-3f9a1c" in element for element in recorded[0])

    @pytest.mark.asyncio
    async def test_stale_session_retry_sends_prompt_on_stdin_again(self, adapter):
        adapter._sessions["ctx"] = "old-thread"
        procs = []

        async def spawn(*args, **kwargs):
            proc = AsyncMock()
            proc.pid = len(procs) + 1
            if procs:
                stdout = json.dumps(_item_completed_event("retried"))
                proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
                proc.returncode = 0
            else:
                proc.communicate = AsyncMock(return_value=(b"", b""))
                proc.returncode = 1
            proc.spawn_args = args
            procs.append(proc)
            return proc

        with patch("a2a_adapter.base_adapter.create_subprocess_exec", side_effect=spawn):
            result = await adapter.invoke("--help", context_id="ctx")

        assert result == "retried"
        assert len(procs) == 2
        assert "resume" in procs[0].spawn_args
        assert "resume" not in procs[1].spawn_args
        for proc in procs:
            assert proc.spawn_args[-1] == "-"
            proc.communicate.assert_awaited_once_with(input=b"--help")


# ──── Test: _parse_result ────


class TestParseResult:
    def test_thread_started_extracts_thread_id(self, adapter):
        stdout = "\n".join(
            [
                json.dumps(_thread_started_event("thread-123")),
                json.dumps(_item_completed_event("some output")),
            ]
        )
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert isinstance(result, ParseResult)
        assert result.session_id == "thread-123"
        assert result.text == "some output"

    def test_thread_started_nested_id(self, adapter):
        stdout = "\n".join(
            [
                json.dumps(_thread_started_nested_event("thread-nested-1")),
                json.dumps(_item_completed_event("some output")),
            ]
        )
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert result.session_id == "thread-nested-1"

    def test_item_completed_extracts_text(self, adapter):
        stdout = "\n".join(
            [
                json.dumps(_item_completed_event("Hello from Codex")),
            ]
        )
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert result.text == "Hello from Codex"
        assert result.session_id is None

    def test_multiple_item_completed(self, adapter):
        stdout = "\n".join(
            [
                json.dumps(_item_completed_event("Part 1")),
                json.dumps(_item_completed_event("Part 2")),
            ]
        )
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert "Part 1" in result.text
        assert "Part 2" in result.text
        # Parts joined with double newline
        assert result.text == "Part 1\n\nPart 2"

    def test_turn_failed_raises(self, adapter):
        stdout = json.dumps(_turn_failed_event("something went wrong"))
        with pytest.raises(RuntimeError, match="Codex turn failed: something went wrong"):
            adapter._parse_invoke_output(stdout, "ctx-1")

    def test_mixed_events_correct_output(self, adapter):
        stdout = "\n".join(
            [
                json.dumps(_thread_started_event("thread-mix")),
                json.dumps(_item_completed_event("Response text")),
            ]
        )
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert result.session_id == "thread-mix"
        assert result.text == "Response text"

    def test_no_text_raises_error(self, adapter):
        """No agent_message items -> raises RuntimeError."""
        stdout = json.dumps(_thread_started_event("thread-only"))
        with pytest.raises(RuntimeError, match="no visible output"):
            adapter._parse_invoke_output(stdout, "ctx-1")

    def test_non_agent_message_items_raises_error(self, adapter):
        """item.completed with non-agent_message type only -> raises RuntimeError."""
        event = {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "text": "should be ignored",
            },
        }
        stdout = json.dumps(event)
        with pytest.raises(RuntimeError, match="no visible output"):
            adapter._parse_invoke_output(stdout, "ctx-1")

    def test_non_json_lines_skipped(self, adapter):
        stdout = "not json\n" + json.dumps(_item_completed_event("real text")) + "\n"
        result = adapter._parse_invoke_output(stdout, "ctx-1")
        assert result.text == "real text"


# ──── Test: invoke() with mock subprocess ────


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_success_with_thread_persistence(self, adapter, mock_proc):
        """Successful invoke persists thread_id."""
        stdout = "\n".join(
            [
                json.dumps(_thread_started_event("thread-inv")),
                json.dumps(_item_completed_event("Codex says hello")),
            ]
        )
        mock_proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
        mock_proc.returncode = 0

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await adapter.invoke(
                "say hello",
                context_id="ctx-1",
                context=_make_context("task-1"),
            )

        assert result == "Codex says hello"
        assert adapter._sessions["ctx-1"] == "thread-inv"

    @pytest.mark.asyncio
    async def test_invoke_error_exit_reports_turn_failure_over_stderr(self, adapter, mock_proc):
        api_error = json.dumps(
            {"type": "error", "status": 400, "error": {"message": "Model not supported"}}
        )
        stdout = "\n".join(
            json.dumps(e)
            for e in [
                {"type": "thread.started", "thread_id": "t-1"},
                {"type": "error", "message": api_error},
                {"type": "turn.failed", "error": {"message": api_error}},
            ]
        ).encode()
        mock_proc.communicate = AsyncMock(
            return_value=(stdout, b"Reading additional input from stdin...")
        )
        mock_proc.returncode = 1

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.invoke(
                    "fail",
                    context_id="ctx-1",
                    context=_make_context("task-1"),
                )

        assert str(exc_info.value) == "Command failed with exit code 1: Model not supported"

    @pytest.mark.asyncio
    async def test_invoke_error_exit(self, adapter, mock_proc):
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal error"))
        mock_proc.returncode = 1

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(RuntimeError, match="exit code 1"):
                await adapter.invoke(
                    "fail",
                    context_id="ctx-1",
                    context=_make_context("task-1"),
                )

    @pytest.mark.asyncio
    async def test_invoke_empty_output(self, adapter, mock_proc):
        """Empty stdout with zero exit code -> RuntimeError."""
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(RuntimeError, match="empty output"):
                await adapter.invoke(
                    "test",
                    context_id="ctx-1",
                    context=_make_context("task-1"),
                )

    @pytest.mark.asyncio
    async def test_invoke_binary_not_found(self, adapter):
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(FileNotFoundError, match="Codex binary not found"):
                await adapter.invoke(
                    "test",
                    context_id="ctx-1",
                    context=_make_context("task-1"),
                )

    @pytest.mark.asyncio
    async def test_invoke_bad_working_dir(self, tmp_path):
        """FileNotFoundError from nonexistent cwd is reported as bad working_dir."""
        bad_dir = str(tmp_path / "nonexistent")
        a = CodexAdapter(
            working_dir=bad_dir,
            session_store_path=str(tmp_path / "s.json"),
        )
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=FileNotFoundError("no such file"),
        ):
            with pytest.raises(FileNotFoundError, match="Working directory does not exist"):
                await a.invoke(
                    "test",
                    context_id="ctx-1",
                    context=_make_context("task-1"),
                )


# ──── Test: supports_streaming() returns False ────


class TestStreaming:
    def test_supports_streaming_returns_false(self, adapter):
        assert adapter.supports_streaming() is False

    def test_metadata_streaming_false(self, adapter):
        meta = adapter.get_metadata()
        assert meta.streaming is False


# ──── Test: Per-context serial (D1) ────


class TestPerContextSerial:
    @pytest.mark.asyncio
    async def test_same_context_serial(self, adapter):
        """Two invoke() calls with the SAME context_id execute serially."""
        execution_order = []
        event_first_started = asyncio.Event()
        event_release_first = asyncio.Event()

        async def slow_communicate(input=None):
            execution_order.append("first_start")
            event_first_started.set()
            await event_release_first.wait()
            execution_order.append("first_end")
            stdout = json.dumps(_item_completed_event("r1"))
            return (stdout.encode(), b"")

        async def fast_communicate(input=None):
            execution_order.append("second_start")
            stdout = json.dumps(_item_completed_event("r2"))
            return (stdout.encode(), b"")

        mock_proc_1 = AsyncMock()
        mock_proc_1.returncode = 0
        mock_proc_1.pid = 1
        mock_proc_1.communicate = slow_communicate

        mock_proc_2 = AsyncMock()
        mock_proc_2.returncode = 0
        mock_proc_2.pid = 2
        mock_proc_2.communicate = fast_communicate

        call_count = 0

        async def mock_create_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_proc_1
            return mock_proc_2

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            task1 = asyncio.create_task(
                adapter.invoke(
                    "first",
                    context_id="shared-ctx",
                    context=_make_context("task-A"),
                )
            )
            await event_first_started.wait()

            task2 = asyncio.create_task(
                adapter.invoke(
                    "second",
                    context_id="shared-ctx",
                    context=_make_context("task-B"),
                )
            )

            await asyncio.sleep(0.05)
            assert "second_start" not in execution_order

            event_release_first.set()

            r1 = await task1
            r2 = await task2

        assert r1 == "r1"
        assert r2 == "r2"
        assert execution_order.index("first_end") < execution_order.index("second_start")

    @pytest.mark.asyncio
    async def test_different_contexts_parallel(self, adapter):
        """Two invoke() calls with DIFFERENT context_ids run in parallel."""
        both_started = asyncio.Event()
        started_count = 0
        release = asyncio.Event()

        async def tracked_communicate(input=None):
            nonlocal started_count
            started_count += 1
            if started_count >= 2:
                both_started.set()
            await release.wait()
            stdout = json.dumps(_item_completed_event("ok"))
            return (stdout.encode(), b"")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.pid = 100
        mock_proc.communicate = tracked_communicate

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            task1 = asyncio.create_task(
                adapter.invoke(
                    "a",
                    context_id="ctx-A",
                    context=_make_context("task-A"),
                )
            )
            task2 = asyncio.create_task(
                adapter.invoke(
                    "b",
                    context_id="ctx-B",
                    context=_make_context("task-B"),
                )
            )

            await asyncio.wait_for(both_started.wait(), timeout=2.0)
            assert started_count == 2

            release.set()
            await task1
            await task2


# ──── Test: Cancel running task (D2 - _killed_tasks) ────


class TestCancelRunning:
    @pytest.mark.asyncio
    async def test_cancel_running_via_killed_tasks(self, adapter, mock_proc):
        """When returncode < 0 and task_id in _killed_tasks,
        CancelledByAdapterError is raised."""
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = -9  # Killed

        adapter._killed_tasks.add("task-kill")

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(CancelledByAdapterError, match="killed by cancel"):
                await adapter.invoke(
                    "test",
                    context_id="ctx-k",
                    context=_make_context("task-kill"),
                )

        assert "task-kill" not in adapter._killed_tasks

    @pytest.mark.asyncio
    async def test_cancel_kills_running_process(self, adapter):
        """cancel() on a running process kills it and marks _killed_tasks."""
        mock_proc = MagicMock()
        mock_proc.returncode = None  # Still running
        mock_proc.kill = MagicMock()

        adapter._active_processes["task-k"] = mock_proc

        ctx = _make_context("task-k")
        await adapter.cancel(context_id="ctx-k", context=ctx)

        mock_proc.kill.assert_called_once()
        assert "task-k" in adapter._killed_tasks

    @pytest.mark.asyncio
    async def test_cancel_finished_process_queues(self, adapter):
        """cancel() on a finished process adds to _cancelled_tasks."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Already finished

        adapter._active_processes["task-f"] = mock_proc

        ctx = _make_context("task-f")
        await adapter.cancel(context_id="ctx-f", context=ctx)

        mock_proc.kill.assert_not_called()
        assert "task-f" in adapter._cancelled_tasks


# ──── Test: Cancel queued task (D2 - _cancelled_tasks) ────


class TestCancelQueued:
    @pytest.mark.asyncio
    async def test_cancelled_before_start(self, adapter):
        """Task marked in _cancelled_tasks raises CancelledByAdapterError
        immediately without spawning a subprocess."""
        adapter._cancelled_tasks.add("task-q")

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
        ) as mock_exec:
            with pytest.raises(CancelledByAdapterError, match="cancelled before execution"):
                await adapter.invoke(
                    "test",
                    context_id="ctx-q",
                    context=_make_context("task-q"),
                )

            mock_exec.assert_not_called()

        assert "task-q" not in adapter._cancelled_tasks

    @pytest.mark.asyncio
    async def test_cancel_queued_via_api(self, adapter):
        """cancel() with no running process adds to _cancelled_tasks."""
        ctx = _make_context("task-queued")
        await adapter.cancel(context_id="ctx-1", context=ctx)
        assert "task-queued" in adapter._cancelled_tasks

    @pytest.mark.asyncio
    async def test_cancel_no_context_is_noop(self, adapter):
        """cancel() without a context object is a no-op."""
        await adapter.cancel(context_id="ctx-1")
        assert len(adapter._cancelled_tasks) == 0


# ──── Test: Stale session retry ────


class TestStaleSessionRetry:
    @pytest.mark.asyncio
    async def test_invoke_stale_session_retries(self, adapter):
        """Non-zero exit + empty stdout + empty stderr + had thread_id -> retry."""
        adapter._sessions["ctx-stale"] = "old-thread"

        call_count = 0

        async def mock_create_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.pid = call_count

            if call_count == 1:
                # First call: stale session (both stdout AND stderr empty)
                proc.communicate = AsyncMock(return_value=(b"", b""))
                proc.returncode = 1
            else:
                # Second call: success (no resume)
                stdout = "\n".join(
                    [
                        json.dumps(_thread_started_event("new-thread")),
                        json.dumps(_item_completed_event("retry success")),
                    ]
                )
                proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
                proc.returncode = 0
            return proc

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            result = await adapter.invoke(
                "retry me",
                context_id="ctx-stale",
                context=_make_context("task-retry"),
            )

        assert result == "retry success"
        assert call_count == 2
        assert adapter._sessions.get("ctx-stale") == "new-thread"

    @pytest.mark.asyncio
    async def test_no_retry_on_signal_kill(self, adapter):
        """Signal kill (returncode < 0) + empty stdout + had thread_id -> NO retry.
        Only positive exit codes trigger stale session retry."""
        adapter._sessions["ctx-sig"] = "old-thread"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = -15  # SIGTERM
        mock_proc.pid = 1

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            # Should NOT retry — signal kill is a real failure, not a stale session
            with pytest.raises(RuntimeError, match="exit code -15"):
                await adapter.invoke(
                    "signal test",
                    context_id="ctx-sig",
                    context=_make_context("task-sig"),
                )

        # Session should NOT have been cleared (no stale retry path taken)
        assert adapter._sessions.get("ctx-sig") == "old-thread"

    @pytest.mark.asyncio
    async def test_no_retry_without_thread_id(self, adapter):
        """Non-zero exit + empty stdout but NO previous thread_id -> no retry."""
        # No session set
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1
        mock_proc.pid = 1

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(RuntimeError, match="exit code 1"):
                await adapter.invoke(
                    "no retry",
                    context_id="ctx-nr",
                    context=_make_context("task-nr"),
                )

    @pytest.mark.asyncio
    async def test_invoke_no_retry_with_stderr(self, adapter):
        """Non-zero exit + empty stdout + had thread_id BUT stderr has content
        -> NO retry. Stderr indicates a real error (auth, env, etc.)."""
        adapter._sessions["ctx-stderr"] = "valid-thread"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"Authentication failed: invalid token")
        )
        mock_proc.returncode = 1
        mock_proc.pid = 1

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(RuntimeError, match="Authentication failed"):
                await adapter.invoke(
                    "test",
                    context_id="ctx-stderr",
                    context=_make_context("task-stderr"),
                )

        # Session should NOT have been cleared — it's still valid
        assert adapter._sessions.get("ctx-stderr") == "valid-thread"

    @pytest.mark.asyncio
    async def test_retry_only_once(self, adapter):
        """Retry happens at most once (allow_retry=False on second call)."""
        adapter._sessions["ctx-double"] = "old-thread"

        call_count = 0

        async def mock_create_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.pid = call_count
            # Both calls fail with empty stdout AND empty stderr
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 1
            return proc

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            with pytest.raises(RuntimeError, match="exit code 1"):
                await adapter.invoke(
                    "double fail",
                    context_id="ctx-double",
                    context=_make_context("task-df"),
                )

        # Should have tried exactly 2 times (original + 1 retry)
        assert call_count == 2

    @pytest.mark.parametrize(
        "message",
        [
            "No saved session found with ID old-thread",
            "no rollout found for thread id old-thread",
            "no rollout found for conversation id old-thread",
            "thread not found: old-thread",
        ],
    )
    @pytest.mark.parametrize("channel", ["stderr", "stdout"])
    @pytest.mark.asyncio
    async def test_invoke_retries_on_missing_thread(self, adapter, message, channel):
        adapter._sessions["ctx-miss"] = "old-thread"
        calls = []

        async def mock_create_subprocess(*args, **kwargs):
            calls.append(args)
            proc = AsyncMock()
            proc.pid = len(calls)
            if len(calls) == 1:
                if channel == "stderr":
                    out, err = b"", message.encode()
                else:
                    out, err = json.dumps({"type": "error", "message": message}).encode(), b""
                proc.communicate = AsyncMock(return_value=(out, err))
                proc.returncode = 1
            else:
                stdout = "\n".join(
                    [
                        json.dumps(_thread_started_event("new-thread")),
                        json.dumps(_item_completed_event("fresh")),
                    ]
                )
                proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
                proc.returncode = 0
            return proc

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            result = await adapter.invoke(
                "hi", context_id="ctx-miss", context=_make_context("task-miss")
            )

        assert result == "fresh"
        assert len(calls) == 2
        assert "resume" in calls[0] and "resume" not in calls[1]
        assert adapter._sessions.get("ctx-miss") == "new-thread"

    @pytest.mark.asyncio
    async def test_invoke_retries_on_missing_thread_with_exit_zero(self, adapter):
        adapter._sessions["ctx-zero"] = "old-thread"
        calls = []

        async def mock_create_subprocess(*args, **kwargs):
            calls.append(args)
            proc = AsyncMock()
            proc.pid = len(calls)
            if len(calls) == 1:
                stdout = json.dumps(_turn_failed_event("thread not found: old-thread"))
            else:
                stdout = "\n".join(
                    [
                        json.dumps(_thread_started_event("new-thread")),
                        json.dumps(_item_completed_event("fresh")),
                    ]
                )
            proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
            proc.returncode = 0
            return proc

        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            result = await adapter.invoke(
                "hi", context_id="ctx-zero", context=_make_context("task-zero")
            )

        assert result == "fresh"
        assert "resume" not in calls[1]
        assert adapter._sessions.get("ctx-zero") == "new-thread"

    def test_stale_hook_ignores_transcript_text(self, adapter):
        """Command output or agent text that quotes the message is not a stale thread."""
        transcript = "\n".join(
            [
                json.dumps(_item_completed_event("the log says thread not found: abc")),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "aggregated_output": "no rollout found for thread id abc",
                        },
                    }
                ),
                json.dumps(_turn_failed_event("rate limited")),
            ]
        )
        assert adapter._is_stale_session_error(transcript, "") is False
        assert adapter._is_stale_session_error(transcript, "something else") is False

    @pytest.mark.parametrize("returncode", [1, 0])
    @pytest.mark.asyncio
    async def test_unrelated_codex_error_still_fails(self, adapter, returncode):
        adapter._sessions["ctx-other"] = "old-thread"
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(_turn_failed_event("rate limited")).encode(), b"")
        )
        mock_proc.returncode = returncode
        mock_proc.pid = 1
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            return_value=mock_proc,
        ) as spawn:
            with pytest.raises(RuntimeError, match="rate limited"):
                await adapter.invoke(
                    "hi", context_id="ctx-other", context=_make_context("task-other")
                )
        assert spawn.call_count == 1
        assert adapter._sessions.get("ctx-other") == "old-thread"


# ──── Test: Session persistence ────


class TestSessionPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = str(tmp_path / "sessions.json")
        a1 = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=store,
        )
        a1._sessions = {"ctx-1": "thread-A", "ctx-2": "thread-B"}
        a1._save_sessions()

        a2 = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=store,
        )
        assert a2._sessions == {"ctx-1": "thread-A", "ctx-2": "thread-B"}

    def test_file_not_found_empty_dict(self, tmp_path):
        store = str(tmp_path / "nonexistent" / "sessions.json")
        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=store,
        )
        assert a._sessions == {}

    def test_corrupt_json_empty_dict(self, tmp_path):
        store = str(tmp_path / "sessions.json")
        with open(store, "w") as f:
            f.write("this is not json{{{")

        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=store,
        )
        assert a._sessions == {}

    def test_non_dict_json_empty_dict(self, tmp_path):
        store = str(tmp_path / "sessions.json")
        with open(store, "w") as f:
            json.dump(["a", "b"], f)

        a = CodexAdapter(
            working_dir=str(tmp_path),
            session_store_path=store,
        )
        assert a._sessions == {}

    def test_default_session_store_path(self, tmp_path):
        """The default store lives in the per-user state dir, not in the
        project the agent works on, and differs per working dir."""
        proj_a = tmp_path / "proj-a"
        proj_b = tmp_path / "proj-b"
        a = CodexAdapter(working_dir=str(proj_a))
        b = CodexAdapter(working_dir=str(proj_b))
        assert a._session_store_path == os.path.join(
            default_state_dir("codex", str(proj_a)), "sessions.json"
        )
        assert not a._session_store_path.startswith(str(proj_a) + os.sep)
        assert a._session_store_path != b._session_store_path

    def test_default_store_survives_restart(self, tmp_path):
        a1 = CodexAdapter(working_dir=str(tmp_path))
        a1._sessions = {"ctx-1": "sess-A"}
        a1._save_sessions()
        a2 = CodexAdapter(working_dir=str(tmp_path))
        assert a2._sessions == {"ctx-1": "sess-A"}
        assert not (tmp_path / ".a2a-adapter").exists()


# ──── Test: Metadata ────


class TestMetadata:
    def test_metadata_defaults(self, adapter):
        meta = adapter.get_metadata()
        assert meta.name == "Codex"
        assert meta.description == "OpenAI Codex coding agent"
        assert meta.streaming is False

    def test_metadata_custom(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            name="My Codex Agent",
            description="Codex does stuff",
            skills=[{"id": "fix", "name": "Fix Bugs"}],
            session_store_path=str(tmp_path / "s.json"),
        )
        meta = a.get_metadata()
        assert meta.name == "My Codex Agent"
        assert meta.description == "Codex does stuff"
        assert len(meta.skills) == 1


# ──── Test: Loader integration ────


class TestLoaderIntegration:
    def test_load_adapter_codex(self, tmp_path):
        a = load_adapter(
            {
                "adapter": "codex",
                "working_dir": str(tmp_path),
                "session_store_path": str(tmp_path / "s.json"),
            }
        )
        assert isinstance(a, CodexAdapter)
        assert a.working_dir == str(tmp_path)

    def test_load_adapter_with_options(self, tmp_path):
        a = load_adapter(
            {
                "adapter": "codex",
                "working_dir": str(tmp_path),
                "timeout": 300,
                "codex_path": "/opt/codex",
                "session_store_path": str(tmp_path / "s.json"),
            }
        )
        assert isinstance(a, CodexAdapter)
        assert a.timeout == 300
        assert a.codex_path == "/opt/codex"

    def test_load_adapter_with_dangerous_flags(self, tmp_path):
        a = load_adapter(
            {
                "adapter": "codex",
                "working_dir": str(tmp_path),
                "bypass_approvals": True,
                "skip_git_check": True,
                "session_store_path": str(tmp_path / "s.json"),
            }
        )
        assert isinstance(a, CodexAdapter)
        assert a.bypass_approvals is True
        assert a.skip_git_check is True


# ──── Test: Server integration ────


class TestServerIntegration:
    def test_build_agent_card(self, adapter):
        card = build_agent_card(adapter, url="http://localhost:9011")
        assert card.name == "Codex"
        assert card.capabilities.streaming is False

    def test_to_a2a_builds_app(self, adapter):
        app = to_a2a(adapter)
        assert app is not None


# ──── Test: Flat import ────


def test_flat_import():
    from a2a_adapter import CodexAdapter as CX

    assert CX is CodexAdapter


# ──── Test: Async context manager ────


@pytest.mark.asyncio
async def test_context_manager(tmp_path):
    async with CodexAdapter(
        working_dir=str(tmp_path),
        session_store_path=str(tmp_path / "s.json"),
    ) as adapter:
        assert isinstance(adapter, CodexAdapter)


# ──── Test: close() ────


@pytest.mark.asyncio
async def test_close_kills_active_processes(adapter):
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    adapter._active_processes["task-1"] = mock_proc
    await adapter.close()

    mock_proc.kill.assert_called_once()
    assert len(adapter._active_processes) == 0
    assert len(adapter._cancelled_tasks) == 0
    assert len(adapter._killed_tasks) == 0


# ──── Test: Environment variables ────


class TestEnvVars:
    def test_build_env_includes_custom_vars(self, tmp_path):
        a = CodexAdapter(
            working_dir=str(tmp_path),
            env_vars={"OPENAI_API_KEY": "sk-test"},
            session_store_path=str(tmp_path / "s.json"),
        )
        env = a._build_env()
        assert env["OPENAI_API_KEY"] == "sk-test"
        assert "PATH" in env

    def test_build_env_without_custom_vars(self, adapter):
        env = adapter._build_env()
        assert "PATH" in env


# ──── Test: _extract_thread_id edge cases ────


class TestExtractThreadId:
    def test_top_level_thread_id(self):
        event = {"type": "thread.started", "thread_id": "top-level"}
        assert CodexAdapter._extract_thread_id(event) == "top-level"

    def test_nested_thread_id(self):
        event = {"type": "thread.started", "thread": {"id": "nested"}}
        assert CodexAdapter._extract_thread_id(event) == "nested"

    def test_no_thread_id(self):
        event = {"type": "thread.started"}
        assert CodexAdapter._extract_thread_id(event) is None

    def test_numeric_thread_id_stringified(self):
        event = {"type": "thread.started", "thread_id": 12345}
        assert CodexAdapter._extract_thread_id(event) == "12345"

    def test_empty_string_thread_id(self):
        event = {"type": "thread.started", "thread_id": ""}
        # Empty string is falsy, should return None
        assert CodexAdapter._extract_thread_id(event) is None

    def test_thread_not_dict(self):
        event = {"type": "thread.started", "thread": "not a dict"}
        assert CodexAdapter._extract_thread_id(event) is None
