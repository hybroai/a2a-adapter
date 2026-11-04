"""Tests for the stateless DeepSeek Harness headless CLI adapter."""

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_adapter.base_adapter import CommandResult, default_state_dir
from a2a_adapter.integrations.deepseek_harness_headless import DeepSeekHarnessHeadlessAdapter
from a2a_adapter.loader import load_adapter


def _context(task_id: str = "task-1") -> MagicMock:
    context = MagicMock()
    context.task_id = task_id
    return context


def _proc(stdout: bytes = b"done\n", stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.pid = 12345
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.fixture(autouse=True)
def no_bundled_runtime(monkeypatch):
    # None in sys.modules makes the import fail, as when the extra is missing.
    monkeypatch.setitem(sys.modules, "deepseek_harness_runtime", None)


def _bundled_runtime(monkeypatch, launch_args=None, error=None):
    module = types.ModuleType("deepseek_harness_runtime")

    def resolve_bundled_launch_args():
        if error is not None:
            raise error
        return launch_args

    module.resolve_bundled_launch_args = resolve_bundled_launch_args
    monkeypatch.setitem(sys.modules, "deepseek_harness_runtime", module)


@pytest.fixture
def adapter(tmp_path):
    return DeepSeekHarnessHeadlessAdapter(
        working_dir=str(tmp_path), dsh_home=str(tmp_path / "home")
    )


class TestBuildCommand:
    def test_headless_profile_and_task_on_stdin(self, adapter):
        command = adapter._build_command("run the tests", "ctx")
        assert isinstance(command, CommandResult)
        assert command.args == ["dsh", "--profile", "headless"]
        assert command.stdin == "run the tests"
        assert command.used_resume is False

    def test_cli_args_go_between_profile_and_task(self, tmp_path):
        adapter = DeepSeekHarnessHeadlessAdapter(
            working_dir=str(tmp_path),
            dsh_path="/opt/dsh",
            cli_args=["--patch", "extra.yml"],
        )
        assert adapter._build_command("hi", "ctx").args == [
            "/opt/dsh",
            "--profile",
            "headless",
            "--patch",
            "extra.yml",
        ]

    def test_task_starting_with_dash_goes_to_stdin_unchanged(self, adapter):
        command = adapter._build_command("-x explain", "ctx")
        assert command.stdin == "-x explain"
        assert "-x explain" not in command.args

    def test_every_turn_is_a_fresh_session(self, adapter):
        first = adapter._build_command("hi", "ctx-1").args
        adapter._sessions["ctx-1"] = "ignored"
        assert adapter._build_command("hi", "ctx-1").args == first

    def test_bundled_runtime_is_launched_directly(self, monkeypatch, adapter):
        # The dsh console script only waits on the runtime on Windows, so
        # killing it would orphan the runtime.
        _bundled_runtime(monkeypatch, ("/bundled/node", "/bundled/bin.mjs"))
        assert adapter._build_command("hi", "ctx").args == [
            "/bundled/node",
            "/bundled/bin.mjs",
            "--profile",
            "headless",
        ]

    def test_unavailable_bundled_runtime_falls_back_to_dsh(self, monkeypatch, adapter):
        _bundled_runtime(monkeypatch, error=FileNotFoundError("no runtime"))
        assert adapter._build_command("hi", "ctx").args[0] == "dsh"

    def test_invalid_runtime_mode_is_not_hidden(self, monkeypatch, adapter):
        _bundled_runtime(monkeypatch, error=ValueError("unsupported runtime mode"))
        with pytest.raises(ValueError, match="unsupported runtime mode"):
            adapter._build_command("hi", "ctx")

    def test_explicit_dsh_path_wins_over_bundled_runtime(self, monkeypatch, tmp_path):
        _bundled_runtime(monkeypatch, ("/bundled/dsh",))
        adapter = DeepSeekHarnessHeadlessAdapter(working_dir=str(tmp_path), dsh_path="/opt/dsh")
        assert adapter._build_command("hi", "ctx").args[0] == "/opt/dsh"

    def test_string_cli_args_are_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            DeepSeekHarnessHeadlessAdapter(working_dir=str(tmp_path), cli_args="--patch x")


class TestEnvironment:
    def test_dsh_home_argument_sets_env(self, tmp_path, adapter):
        assert adapter.env_vars["DSH_HOME"] == str(tmp_path / "home")

    def test_dsh_home_defaults_to_state_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DSH_HOME", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        adapter = DeepSeekHarnessHeadlessAdapter(working_dir=str(tmp_path / "ws"))
        assert adapter.env_vars["DSH_HOME"] == default_state_dir(
            "deepseek-harness", str(tmp_path / "ws")
        )

    def test_dsh_home_env_is_used(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSH_HOME", str(tmp_path / "env-home"))
        adapter = DeepSeekHarnessHeadlessAdapter(working_dir=str(tmp_path))
        assert adapter.env_vars["DSH_HOME"] == str(tmp_path / "env-home")

    def test_empty_env_vars_dsh_home_is_resolved(self, tmp_path):
        adapter = DeepSeekHarnessHeadlessAdapter(
            working_dir=str(tmp_path),
            dsh_home=str(tmp_path / "arg-home"),
            env_vars={"DSH_HOME": " "},
        )
        assert adapter.env_vars["DSH_HOME"] == str(tmp_path / "arg-home")

    def test_relative_env_vars_dsh_home_resolves_against_working_dir(self, tmp_path):
        adapter = DeepSeekHarnessHeadlessAdapter(
            working_dir=str(tmp_path), env_vars={"DSH_HOME": "state"}
        )
        assert adapter.env_vars["DSH_HOME"] == str(tmp_path / "state")

    def test_env_vars_dsh_home_expands_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "user"))
        adapter = DeepSeekHarnessHeadlessAdapter(
            working_dir=str(tmp_path), env_vars={"DSH_HOME": "~/dsh"}
        )
        assert adapter.env_vars["DSH_HOME"] == str(tmp_path / "user" / "dsh")

    def test_user_env_vars_dsh_home_wins(self, tmp_path):
        adapter = DeepSeekHarnessHeadlessAdapter(
            working_dir=str(tmp_path),
            dsh_home=str(tmp_path / "arg-home"),
            env_vars={"DSH_HOME": "/custom/home", "OTHER": "1"},
        )
        assert adapter.env_vars == {"DSH_HOME": "/custom/home", "OTHER": "1"}


class TestInvoke:
    @pytest.mark.asyncio
    async def test_returns_stripped_stdout_and_runs_in_working_dir(self, tmp_path, adapter):
        proc = _proc(stdout=b"\n  All tests pass.  \n", stderr=b"dsh: reasoning:\n...")
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc) as spawn:
            result = await adapter.invoke("run the tests", context_id="ctx", context=_context())

        assert result == "All tests pass."
        proc.communicate.assert_awaited_once_with(input=b"run the tests")
        assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
        assert spawn.call_args.kwargs["env"]["DSH_HOME"] == str(tmp_path / "home")
        assert os.path.isdir(tmp_path / "home")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message", ["", "   \n"])
    async def test_empty_task_is_rejected_before_spawning(self, adapter, message):
        with patch("a2a_adapter.base_adapter.create_subprocess_exec") as spawn:
            with pytest.raises(ValueError, match="empty"):
                await adapter.invoke(message, context_id="ctx", context=_context())
        spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_without_context_get_distinct_task_ids(self, adapter, monkeypatch):
        seen = []

        async def record(key, task_id, *, user_input):
            seen.append(task_id)
            return "ok"

        monkeypatch.setattr(adapter, "_run_subprocess", record)
        await adapter.invoke("a")
        await adapter.invoke("b")
        assert len(set(seen)) == 2

    @pytest.mark.asyncio
    async def test_no_session_state_is_written(self, tmp_path, adapter):
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=_proc()):
            await adapter.invoke("hi", context_id="ctx", context=_context())
        assert adapter._sessions == {}
        assert not any(name.endswith(".json") for name in os.listdir(tmp_path))

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_with_stderr(self, adapter):
        proc = _proc(stdout=b"", stderr=b"dsh: MISSING_CREDENTIAL: no API key", returncode=1)
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="exit code 1: dsh: MISSING_CREDENTIAL"):
                await adapter.invoke("hi", context_id="ctx", context=_context())

    @pytest.mark.asyncio
    async def test_missing_binary_names_the_launched_program(self, monkeypatch, adapter):
        # Resolution works once, then fails: the message must not resolve again.
        results = [("/bundled/dsh",)]
        module = types.ModuleType("deepseek_harness_runtime")
        module.resolve_bundled_launch_args = lambda: results.pop(0)
        monkeypatch.setitem(sys.modules, "deepseek_harness_runtime", module)
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=FileNotFoundError("dsh"),
        ):
            with pytest.raises(FileNotFoundError, match="/bundled/dsh"):
                await adapter.invoke("hi", context_id="ctx", context=_context())

    @pytest.mark.asyncio
    async def test_missing_binary_points_at_install(self, adapter):
        with patch(
            "a2a_adapter.base_adapter.create_subprocess_exec",
            side_effect=FileNotFoundError("dsh"),
        ):
            with pytest.raises(FileNotFoundError, match=r"a2a-adapter\[deepseek-harness\]"):
                await adapter.invoke("hi", context_id="ctx", context=_context())


class TestMetadataAndRegistration:
    def test_metadata(self, adapter):
        metadata = adapter.get_metadata()
        assert metadata.name == "DeepSeek Harness (headless)"
        assert metadata.streaming is False
        assert adapter.supports_streaming() is False

    def test_flat_imports(self):
        from a2a_adapter import DeepSeekHarnessHeadlessAdapter as Flat
        from a2a_adapter.integrations import DeepSeekHarnessHeadlessAdapter as Integrations

        assert Flat is DeepSeekHarnessHeadlessAdapter
        assert Integrations is DeepSeekHarnessHeadlessAdapter

    def test_loader(self, tmp_path):
        loaded = load_adapter(
            {"adapter": "deepseek-harness-headless", "working_dir": str(tmp_path)}
        )
        assert isinstance(loaded, DeepSeekHarnessHeadlessAdapter)
