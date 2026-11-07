"""Tests for server.py — build_agent_card() and to_a2a()."""

import httpx
import pytest
from a2a.types import AgentCard

from a2a_adapter.base_adapter import AdapterMetadata, BaseA2AAdapter
from a2a_adapter.server import build_agent_card, to_a2a

from .conftest import StubAdapter, StreamingStubAdapter


# ── build_agent_card ─────────────────────────────────────────


class TestBuildAgentCard:
    def test_default_card(self):
        adapter = StubAdapter()
        card = build_agent_card(adapter)
        assert isinstance(card, AgentCard)
        assert card.name == "StubAdapter"
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].url == "http://localhost:9000/"
        assert card.capabilities.streaming is False

    def test_streaming_auto_detected(self):
        adapter = StreamingStubAdapter(["a", "b"])
        card = build_agent_card(adapter)
        assert card.capabilities.streaming is True

    def test_overrides(self):
        adapter = StubAdapter()
        card = build_agent_card(
            adapter,
            name="Custom",
            description="desc",
            url="http://myhost:8080",
            version="2.0.0",
        )
        assert card.name == "Custom"
        assert card.description == "desc"
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].url == "http://myhost:8080"
        assert card.version == "2.0.0"

    def test_metadata_skills(self):
        class SkillAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(
                    name="SkillAgent",
                    skills=[
                        {"id": "s1", "name": "Search", "description": "Web search"},
                    ],
                )

        card = build_agent_card(SkillAdapter())
        assert len(card.skills) == 1
        assert card.skills[0].name == "Search"

    def test_provider_from_dict(self):
        class ProviderAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(provider={"organization": "Acme", "url": "https://acme.com"})

        card = build_agent_card(ProviderAdapter())
        assert card.provider is not None
        assert card.provider.organization == "Acme"

    def test_documentation_and_icon_urls(self):
        class UrlAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(
                    documentation_url="https://docs.example.com",
                    icon_url="https://icon.example.com/logo.png",
                )

        card = build_agent_card(UrlAdapter())
        assert card.documentation_url == "https://docs.example.com"
        assert card.icon_url == "https://icon.example.com/logo.png"


# ── to_a2a ───────────────────────────────────────────────────


class TestToA2a:
    def test_returns_asgi_app(self):
        adapter = StubAdapter()
        app = to_a2a(adapter)
        assert callable(app)

    def test_with_custom_card(self):
        adapter = StubAdapter()
        card = build_agent_card(adapter, name="Custom")
        app = to_a2a(adapter, agent_card=card)
        assert callable(app)

    def test_with_card_overrides(self):
        adapter = StubAdapter()
        app = to_a2a(adapter, name="Override")
        assert callable(app)

    @pytest.mark.asyncio
    async def test_legacy_agent_card_includes_empty_skills(self):
        adapter = StubAdapter()
        app = to_a2a(adapter)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/.well-known/agent-card.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["skills"] == []
        assert data["url"] == data["supportedInterfaces"][0]["url"]


class TestPushSenderContext:
    """a2a-sdk 1.0 requires the sender's context; 1.3 deprecates it and warns."""

    def test_passes_a_context_when_the_sdk_requires_one(self, monkeypatch):
        import a2a_adapter.server as server_module
        from a2a.server.context import ServerCallContext

        seen = []

        class Sdk10Sender:
            def __init__(self, httpx_client, config_store, context):
                seen.append(context)

        monkeypatch.setattr(server_module, "BasePushNotificationSender", Sdk10Sender)
        to_a2a(StubAdapter())
        assert len(seen) == 1 and isinstance(seen[0], ServerCallContext)

    def test_omits_the_deprecated_context_when_it_is_optional(self, monkeypatch):
        import a2a_adapter.server as server_module

        seen = []

        class Sdk13Sender:
            def __init__(
                self, httpx_client, config_store, context=None, *, push_url_validator=None
            ):
                seen.append(context)

        monkeypatch.setattr(server_module, "BasePushNotificationSender", Sdk13Sender)
        to_a2a(StubAdapter())
        assert seen == [None], "a non-None context logs a deprecation warning in 1.3"

    def test_installed_sdk_gets_a_context_only_if_it_requires_one(self, caplog):
        import inspect

        from a2a.server.context import ServerCallContext
        from a2a.server.tasks import BasePushNotificationSender

        import a2a_adapter.server as server_module

        param = inspect.signature(BasePushNotificationSender).parameters.get("context")
        required = param is not None and param.default is inspect.Parameter.empty
        with caplog.at_level("WARNING"):
            sender = server_module._push_sender(httpx.AsyncClient(), None)
        assert isinstance(getattr(sender, "_call_context", None), ServerCallContext) is required
        assert "no longer uses the context" not in caplog.text

    def test_uninspectable_sender_is_built_without_a_context(self, monkeypatch):
        import a2a_adapter.server as server_module

        seen = []

        class Opaque:
            def __init__(self, httpx_client, config_store, context=None):
                seen.append(context)

        def no_signature(obj):
            raise ValueError("no signature found for builtin")

        monkeypatch.setattr(server_module, "BasePushNotificationSender", Opaque)
        monkeypatch.setattr(server_module.inspect, "signature", no_signature)
        server_module._push_sender(httpx.AsyncClient(), None)
        assert seen == [None]


# ── Shutdown kills CLI children (Fix 1) ──────────────────────
#
# Never let a real serve()/run() finish after calling handle_exit here:
# uvicorn re-raises every captured signal afterwards, which would signal the
# pytest process. handle_exit is called on a constructed, never-run server.


def _ready_server(**kwargs):
    import uvicorn

    from a2a_adapter.server import _ReadyServer

    return _ReadyServer(uvicorn.Config(app=None, log_config=None), on_ready=lambda: None, **kwargs)


class TestShutdownKillsChildren:
    @pytest.fixture(autouse=True)
    def _no_real_uvicorn_run(self, monkeypatch):
        """serve_agent must never start a real server from these tests."""

        def refuse(*args, **kwargs):
            raise AssertionError("uvicorn.run() called")

        monkeypatch.setattr("a2a_adapter.server.uvicorn.run", refuse)

    @pytest.fixture(autouse=True)
    def _restore_sse_exit_flag(self, monkeypatch):
        """sse_starlette hooks Server.handle_exit and sets a process-wide flag
        that ends every later SSE response; put it back after each test."""
        try:
            from sse_starlette.sse import AppStatus
        except ImportError:
            return
        monkeypatch.setattr(AppStatus, "should_exit", AppStatus.should_exit)

    def test_sigint_calls_on_interrupt(self):
        import signal
        from unittest.mock import MagicMock

        fake = MagicMock()
        server = _ready_server(on_interrupt=fake)
        server.handle_exit(signal.SIGINT, None)
        fake.assert_called_once_with()
        assert server.should_exit
        server.handle_exit(signal.SIGINT, None)
        fake.assert_called_once_with()

    def test_sigterm_drains_then_second_signal_kills(self):
        import signal
        from unittest.mock import MagicMock

        fake = MagicMock()
        server = _ready_server(on_interrupt=fake)
        server.handle_exit(signal.SIGTERM, None)
        fake.assert_not_called()
        assert server.should_exit
        server.handle_exit(signal.SIGTERM, None)
        fake.assert_called_once_with()

    @pytest.mark.skipif(not hasattr(__import__("signal"), "SIGHUP"), reason="no SIGHUP")
    def test_sighup_calls_on_interrupt(self):
        import signal
        from unittest.mock import MagicMock

        fake = MagicMock()
        server = _ready_server(on_interrupt=fake)
        server.handle_exit(signal.SIGHUP, None)
        fake.assert_called_once_with()
        assert server.should_exit

    @pytest.mark.skipif(not hasattr(__import__("signal"), "SIGHUP"), reason="no SIGHUP")
    def test_capture_signals_also_handles_sighup(self, monkeypatch):
        import signal
        import threading

        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers need the main thread")
        server = _ready_server()
        before = signal.getsignal(signal.SIGHUP)
        server._captured_signals = []  # nothing to re-raise afterwards
        with server.capture_signals():
            assert signal.getsignal(signal.SIGHUP) == server.handle_exit
        assert signal.getsignal(signal.SIGHUP) == before

    @pytest.mark.skipif(not hasattr(__import__("signal"), "SIGHUP"), reason="no SIGHUP")
    def test_ignored_sighup_stays_ignored(self):
        import signal
        import threading

        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers need the main thread")
        server = _ready_server()
        server._captured_signals = []
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as under nohup
        try:
            with server.capture_signals():
                assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        finally:
            signal.signal(signal.SIGHUP, previous)

    def test_on_interrupt_error_is_swallowed(self):
        import signal

        def boom():
            raise RuntimeError("boom")

        server = _ready_server(on_interrupt=boom)
        server.handle_exit(signal.SIGINT, None)
        assert server.should_exit

    @pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX process groups")
    async def test_sigint_kills_inflight_grandchild(self, tmp_path):
        import asyncio
        import signal
        from types import SimpleNamespace

        from tests.process_helpers import (
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path)
        task = asyncio.create_task(adapter.invoke("hi", context=SimpleNamespace(task_id="t")))
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            server = _ready_server(on_interrupt=adapter.kill_children_now)
            server.handle_exit(signal.SIGINT, None)
            with pytest.raises(RuntimeError, match="exit code -9"):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild)
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    def test_serve_agent_without_display_uses_ready_server(self, monkeypatch, tmp_path):
        import sys

        from a2a_adapter import server as server_module

        captured = {}
        sock = tmp_path / "a2a.sock"

        def fake_run(self):
            captured["server"] = self
            if self.config.uds:
                open(self.config.uds, "w").close()
            self.started = True

        monkeypatch.setattr(server_module._ReadyServer, "run", fake_run)
        adapter = StubAdapter()
        server_module.serve_agent(adapter, port=9123)
        server = captured["server"]
        assert isinstance(server, server_module._ReadyServer)
        assert server._on_interrupt == adapter.kill_children_now
        assert server.config.port == 9123

        server_module.serve_agent(adapter, uds=str(sock))
        assert not sock.exists()

        monkeypatch.setattr(sys, "path", list(sys.path))
        server_module.serve_agent(adapter, app_dir=str(tmp_path))
        assert sys.path[0] == str(tmp_path)

    @pytest.mark.parametrize("kwargs", [{"workers": 2}, {"reload": True}])
    def test_serve_agent_without_display_rejects_multi_process(self, kwargs):
        from a2a_adapter import server as server_module

        with pytest.raises(ValueError, match="workers/reload"):
            server_module.serve_agent(StubAdapter(), **kwargs)

    def test_serve_agent_without_display_failed_startup_exits_3(self, monkeypatch):
        from a2a_adapter import server as server_module

        monkeypatch.setattr(server_module._ReadyServer, "run", lambda self: None)
        with pytest.raises(SystemExit) as exit_info:
            server_module.serve_agent(StubAdapter())
        assert exit_info.value.code == 3

    def test_serve_agent_without_display_swallows_ctrl_c(self, monkeypatch):
        from a2a_adapter import server as server_module

        def interrupted(self):
            self.started = True
            raise KeyboardInterrupt

        monkeypatch.setattr(server_module._ReadyServer, "run", interrupted)
        server_module.serve_agent(StubAdapter())
