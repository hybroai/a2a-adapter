"""Tests for server.py — build_agent_card() and to_a2a()."""

import httpcore
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
        to_a2a(StubAdapter(), push_notifications=True)
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
        to_a2a(StubAdapter(), push_notifications=True)
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


# ── Bearer-token auth (Fix 4) ────────────────────────────────

_SEND = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
        "message": {"role": "user", "messageId": "m-1", "parts": [{"kind": "text", "text": "hi"}]}
    },
}


def _client(app, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver", headers=headers
    )


class TestAuth:
    async def test_no_token_keeps_routes_open(self):
        async with _client(to_a2a(StubAdapter())) as client:
            response = await client.post("/", json=_SEND)
        assert response.status_code == 200
        assert response.json()["result"]["status"]["state"] == "completed"

    async def test_jsonrpc_requires_token(self):
        async with _client(to_a2a(StubAdapter(), auth_token="s3cret")) as client:
            response = await client.post("/", json=_SEND)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {"error": "unauthorized"}

    @pytest.mark.parametrize(
        "header", ["Bearer wrong", "Basic s3cret", "s3cret", "Bearer", "Bearer s3cret2"]
    )
    async def test_jsonrpc_wrong_token_rejected(self, header):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with _client(app) as client:
            response = await client.post("/", json=_SEND, headers={"Authorization": header})
        assert response.status_code == 401

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    async def test_jsonrpc_right_token_accepted(self, scheme):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with _client(app) as client:
            response = await client.post(
                "/", json=_SEND, headers={"Authorization": f"{scheme} s3cret"}
            )
        assert response.status_code == 200
        assert response.json()["result"]["status"]["state"] == "completed"

    @pytest.mark.parametrize(
        "method, path",
        [
            ("POST", "/message:send"),
            ("GET", "/tasks"),
            ("GET", "/tasks/x"),
            ("GET", "/tenant-a/tasks"),
            ("POST", "/v1/message:send"),
            ("GET", "/v1/card"),
            ("GET", "/extendedAgentCard"),
            ("GET", "/no-such-route"),
        ],
    )
    async def test_rest_routes_require_token(self, method, path):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with _client(app) as client:
            response = await client.request(method, path)
        assert response.status_code == 401

    @pytest.mark.parametrize("path", ["/.well-known/agent-card.json", "/.well-known/agent.json"])
    async def test_agent_card_routes_stay_public(self, path):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with _client(app) as client:
            assert (await client.get(path)).status_code == 200
            assert (await client.head(path)).status_code == 200
            assert (await client.post(path)).status_code == 401

    async def test_card_declares_bearer_scheme_when_token_set(self):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with _client(app) as client:
            card = (await client.get("/.well-known/agent-card.json")).json()
        assert card["securitySchemes"] == {
            "bearer": {"httpAuthSecurityScheme": {"scheme": "bearer"}}
        }
        assert card["securityRequirements"] == [{"schemes": {"bearer": {}}}]

    async def test_card_has_no_security_without_token(self):
        async with _client(to_a2a(StubAdapter())) as client:
            card = (await client.get("/.well-known/agent-card.json")).json()
        assert "securitySchemes" not in card
        assert "securityRequirements" not in card
        assert not build_agent_card(StubAdapter()).security_schemes

    @pytest.mark.parametrize("token", ["", "   ", "\t\n"])
    def test_empty_token_rejected(self, token):
        with pytest.raises(ValueError, match="auth_token"):
            to_a2a(StubAdapter(), auth_token=token)

    async def test_custom_card_not_modified(self):
        card = build_agent_card(StubAdapter(), name="Custom")
        app = to_a2a(StubAdapter(), agent_card=card, auth_token="s3cret")
        assert not card.security_schemes
        async with _client(app, "s3cret") as client:
            served = (await client.get("/.well-known/agent-card.json")).json()
            assert (await client.post("/", json=_SEND)).status_code == 200
        assert served["name"] == "Custom"
        assert "securitySchemes" not in served

    @pytest.mark.parametrize("token", [" tok", "tok\n", "tok "])
    def test_token_with_surrounding_whitespace_rejected(self, token):
        with pytest.raises(ValueError, match="whitespace"):
            to_a2a(StubAdapter(), auth_token=token)

    @pytest.mark.parametrize("token", ["tök", "tok\x00", "tok\tx"])
    def test_non_printable_ascii_token_rejected(self, token):
        with pytest.raises(ValueError, match="printable ASCII"):
            to_a2a(StubAdapter(), auth_token=token)

    async def test_unauthenticated_websocket_is_closed(self):
        from a2a_adapter.server import _BearerAuthMiddleware

        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        middleware = _BearerAuthMiddleware(app, token="s3cret")
        await middleware({"type": "websocket", "path": "/ws", "headers": []}, receive, send)
        assert reached == []
        assert sent == [{"type": "websocket.close", "code": 1008}]

        authed = {
            "type": "websocket",
            "path": "/ws",
            "headers": [(b"authorization", b"Bearer s3cret")],
        }
        await middleware(authed, receive, send)
        assert reached == ["websocket"]

    async def test_lifespan_passes_through(self):
        app = to_a2a(StubAdapter(), auth_token="s3cret")
        async with app.router.lifespan_context(app):
            pass


def _fake_started_run(captured):
    def fake_run(self):
        captured.append(self)
        self.started = True

    return fake_run


class TestServeAgentBind:
    @pytest.fixture(autouse=True)
    def _no_real_server(self, monkeypatch):
        from a2a_adapter import server as server_module

        self.servers = []
        monkeypatch.setattr(server_module._ReadyServer, "run", _fake_started_run(self.servers))
        monkeypatch.setattr(
            "a2a_adapter.server.uvicorn.run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("uvicorn.run() called")),
        )

    def test_warns_on_non_loopback_without_auth(self, caplog):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), host="0.0.0.0")
        assert any("without auth" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "host, kwargs",
        [
            ("127.0.0.1", {}),
            ("::1", {}),
            ("localhost", {}),
            ("127.0.0.2", {}),
            ("0.0.0.0", {"auth_token": "t"}),
        ],
    )
    def test_no_warning_on_loopback_or_with_auth(self, caplog, host, kwargs):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), host=host, **kwargs)
        assert not any("without auth" in r.getMessage() for r in caplog.records)

    def test_no_warning_for_uds(self, caplog, tmp_path):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), host="0.0.0.0", uds=str(tmp_path / "s.sock"))
        assert not any("without auth" in r.getMessage() for r in caplog.records)

    def test_is_loopback_host(self):
        from a2a_adapter.server import _is_loopback_host

        for host in ("localhost", "LOCALHOST", "127.0.0.1", "127.1.2.3", "::1", "[::1]"):
            assert _is_loopback_host(host), host
        for host in ("0.0.0.0", "::", "192.168.1.2", "example.com", "", "localhost.evil"):
            assert not _is_loopback_host(host), host

    @pytest.mark.parametrize("with_display", [False, True])
    async def test_serve_agent_passes_token_to_app(self, with_display):
        import io

        from a2a_adapter.server import serve_agent
        from a2a_adapter.terminal import ActivityLog

        class QuietDisplay:
            activity = ActivityLog(io.StringIO(), color=False)

            def install_logging(self, level):
                pass

            def ready(self, card):
                pass

            def close(self):
                pass

            def uninstall_logging(self):
                pass

        kwargs = {"display": QuietDisplay()} if with_display else {}
        serve_agent(StubAdapter(), auth_token="s3cret", **kwargs)
        app = self.servers[-1].config.app
        async with _client(app) as client:
            assert (await client.post("/", json=_SEND)).status_code == 401
        async with _client(app, "s3cret") as client:
            assert (await client.post("/", json=_SEND)).status_code == 200

    @staticmethod
    def _quiet_display():
        import io

        from a2a_adapter.terminal import ActivityLog

        class QuietDisplay:
            activity = ActivityLog(io.StringIO(), color=False)

            def install_logging(self, level):
                pass

            def ready(self, card):
                pass

            def close(self):
                pass

            def uninstall_logging(self):
                pass

        return QuietDisplay()

    @pytest.mark.parametrize("with_display", [False, True])
    @pytest.mark.parametrize("value", [False, True])
    def test_serve_agent_passes_expose_error_details(self, monkeypatch, with_display, value):
        from a2a_adapter import server as server_module

        seen = []
        real_to_a2a = server_module.to_a2a

        def spy(adapter, agent_card=None, *args, **kwargs):
            seen.append(kwargs.get("expose_error_details"))
            return real_to_a2a(adapter, agent_card, *args, **kwargs)

        monkeypatch.setattr(server_module, "to_a2a", spy)
        kwargs = {"display": self._quiet_display()} if with_display else {}
        server_module.serve_agent(StubAdapter(), expose_error_details=value, **kwargs)
        assert seen == [value]
        assert "expose_error_details" not in vars(self.servers[-1].config)

    @pytest.mark.parametrize("with_display", [False, True])
    def test_serve_agent_warns_when_exposing_errors(self, caplog, with_display):
        from a2a_adapter.server import serve_agent

        kwargs = {"display": self._quiet_display()} if with_display else {}
        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), expose_error_details=True, **kwargs)
        assert "expose_error_details" in caplog.text

    def test_serve_agent_is_quiet_about_errors_by_default(self, caplog):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter())
        assert "expose_error_details" not in caplog.text

    def test_serve_agent_card_declares_bearer(self, monkeypatch):
        from a2a_adapter import server as server_module

        cards = []
        real_to_a2a = server_module.to_a2a

        def spy(adapter, agent_card=None, *args, **kwargs):
            cards.append(agent_card)
            return real_to_a2a(adapter, agent_card, *args, **kwargs)

        monkeypatch.setattr(server_module, "to_a2a", spy)
        server_module.serve_agent(StubAdapter(), auth_token="t")
        assert "bearer" in cards[-1].security_schemes
        server_module.serve_agent(StubAdapter())
        assert not cards[-1].security_schemes

    def test_serve_agent_default_host(self):
        from a2a_adapter.server import serve_agent

        serve_agent(StubAdapter())
        assert self.servers[-1].config.host == "127.0.0.1"


class TestChildEnvSecrets:
    def test_child_env_has_no_auth_token(self, monkeypatch, tmp_path):
        from a2a_adapter.base_adapter import BaseA2AAdapter

        class Cli(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return ""

        monkeypatch.setenv("A2A_AUTH_TOKEN", "server-secret")
        assert "A2A_AUTH_TOKEN" not in Cli(working_dir=str(tmp_path))._build_env()
        explicit = Cli(working_dir=str(tmp_path), env_vars={"A2A_AUTH_TOKEN": "mine"})
        assert explicit._build_env()["A2A_AUTH_TOKEN"] == "mine"

    async def test_pi_spawn_env_has_no_auth_token(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock

        from a2a_adapter.integrations.pi import PiRpcProcess

        monkeypatch.setenv("A2A_AUTH_TOKEN", "server-secret")
        spawn = AsyncMock(side_effect=RuntimeError("stop after spawn args"))
        monkeypatch.setattr("a2a_adapter.integrations.pi.create_subprocess_exec", spawn)
        for env_vars, expected in (({}, None), ({"A2A_AUTH_TOKEN": "mine"}, "mine")):
            rpc = PiRpcProcess(
                command=["pi"],
                working_dir=str(tmp_path),
                session_id="s",
                session_dir=str(tmp_path / "sessions"),
                env_vars=env_vars,
            )
            with pytest.raises(RuntimeError, match="stop after spawn"):
                await rpc.start()
            assert spawn.call_args.kwargs["env"].get("A2A_AUTH_TOKEN") == expected


# ── Push notifications: opt-in and URL restrictions (Fix 5) ──

_PRIVATE_PUSH_URLS = [
    "http://127.0.0.1/x",
    "http://localhost/x",
    "http://api.localhost/x",
    "http://10.0.0.5/x",
    "http://192.168.1.2/x",
    "http://100.64.0.1/x",
    "http://169.254.169.254/latest",
    "http://[::1]/x",
    "http://[fe80::1]/x",
    "http://[::ffff:127.0.0.1]/x",
    "http://[64:ff9b::7f00:1]/x",
    "http://0.0.0.0/x",
    "http://224.0.0.1/x",
    "ftp://example.com/x",
    "http:///x",
    "http://rebound.example/x",  # _resolve (patched) maps it to 127.0.0.1
    "http://[::127.0.0.1]/x",  # deprecated IPv4-compatible
    "http://[::ffff:0:7f00:1]/x",  # IPv4-translated
    "http://[fec0::1]/x",  # deprecated site-local
]

_PUBLIC_PUSH_URLS = [
    "http://8.8.8.8/x",
    "https://[2606:4700:4700::1111]/x",
    "http://[::ffff:8.8.8.8]/x",
    "http://[64:ff9b::808:808]/x",
    "https://example.com:8443/x",
]


@pytest.fixture
def fake_dns(monkeypatch):
    """Offline DNS: hostnames resolve from this dict; anything else fails."""
    table = {
        "example.com": ["93.184.216.34"],
        "hooks.example": ["93.184.216.34"],
        "rebound.example": ["127.0.0.1"],
        "mixed.example": ["93.184.216.34", "10.0.0.1"],
    }

    async def resolve(host, port):
        if host not in table:
            raise OSError(f"cannot resolve {host}")
        return list(table[host])

    monkeypatch.setattr("a2a_adapter.server._resolve", resolve)
    return table


@pytest.fixture
def push_receiver(monkeypatch):
    """Route every push POST to an in-memory receiver (never the network)."""
    import a2a_adapter.server as server_module

    received: list[httpx.Request] = []
    real = server_module._make_push_client

    def handler(request):
        received.append(request)
        return httpx.Response(200)

    def make(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)

    monkeypatch.setattr(server_module, "_make_push_client", make)
    return received


_OK = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"


class FakeStream(httpcore.AsyncNetworkStream):
    """One offline connection: records writes and TLS, replays a canned reply
    to each request."""

    def __init__(self, host, port, response):
        self.host, self.port, self.response = host, port, response
        self.writes: list[bytes] = []
        self.tls: dict | None = None
        self.closed = False
        self._awaiting_reply = False

    async def read(self, max_bytes, timeout=None):
        # One reply per request, so keep-alive reuse works (the SDK sends a
        # push per task event).
        if not self._awaiting_reply:
            return b""
        self._awaiting_reply = False
        return self.response

    async def write(self, buffer, timeout=None):
        self.writes.append(bytes(buffer))
        self._awaiting_reply = True

    async def aclose(self):
        self.closed = True

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.tls = {"ssl_context": ssl_context, "server_hostname": server_hostname}
        return self

    def get_extra_info(self, info):
        return None

    @property
    def data(self) -> bytes:
        return b"".join(self.writes)


class FakeNetwork(httpcore.AsyncNetworkBackend):
    """Offline delegate backend: records connects, never opens a socket."""

    def __init__(self):
        self.connects: list[tuple] = []
        self.streams: list[FakeStream] = []
        self.responses: list[bytes] = []  # consumed in connect order
        self.fail: set[str] = set()  # hosts that raise ConnectError
        # host -> seconds until ConnectTimeout (None: the whole timeout)
        self.hang: dict[str, float | None] = {}
        self.slept: list[float] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        import asyncio

        self.connects.append((host, port, timeout))
        if host in self.hang:
            delay = self.hang[host]
            await asyncio.sleep(timeout if delay is None else min(timeout, delay))
            raise httpcore.ConnectTimeout(f"timed out connecting to {host}")
        if host in self.fail:
            raise httpcore.ConnectError(f"cannot connect to {host}")
        stream = FakeStream(host, port, self.responses.pop(0) if self.responses else _OK)
        self.streams.append(stream)
        return stream

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise AssertionError("unexpected unix socket")

    async def sleep(self, seconds):
        self.slept.append(seconds)


@pytest.fixture
def push_network(fake_dns, monkeypatch):
    """Push sends go through the real httpx -> httpcore -> pinned backend path
    and end at a FakeNetwork. Real sockets are made to fail loudly."""
    import a2a_adapter.server as server_module

    network = FakeNetwork()
    monkeypatch.setattr(server_module, "_push_network_backend", lambda: network)

    async def no_real_connect(self, *args, **kwargs):
        raise AssertionError("a push tried to open a real connection")

    monkeypatch.setattr(httpcore.AnyIOBackend, "connect_tcp", no_real_connect)
    return network


def _capture_push_client(monkeypatch):
    """Record every push client to_a2a builds (and the kwargs it passed)."""
    import a2a_adapter.server as server_module

    made: list[tuple[dict, httpx.AsyncClient]] = []
    real = server_module._make_push_client

    def make(**kwargs):
        client = real(**kwargs)
        made.append((kwargs, client))
        return client

    monkeypatch.setattr(server_module, "_make_push_client", make)
    return made


def _redirect(location: str) -> bytes:
    return (f"HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n").encode()


@pytest.fixture
def no_real_push_post(monkeypatch):
    """Record (never send) push POSTs; works without _make_push_client.

    Patches ``AsyncClient.post``, so tests using it send with ``client.request``.
    """
    posts = []

    async def fake_post(self, url, *args, **kwargs):
        posts.append(str(url))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return posts


async def _send_task(client, **configuration):
    payload = {
        "jsonrpc": "2.0",
        "id": "s",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": f"m-{len(configuration)}-{id(configuration)}",
                "parts": [{"kind": "text", "text": "hi"}],
            }
        },
    }
    if configuration:
        payload["params"]["configuration"] = configuration
    return (await client.request("POST", "/", json=payload)).json()


async def _send_task_with_push_v1(client, url):
    body = {
        "jsonrpc": "2.0",
        "id": "s1",
        "method": "SendMessage",
        "params": {
            "message": {"role": "ROLE_USER", "messageId": f"m1-{url}", "parts": [{"text": "hi"}]},
            "configuration": {"taskPushNotificationConfig": {"url": url}},
        },
    }
    response = await client.request("POST", "/", json=body, headers={"A2A-Version": "1.0"})
    return response.json()


def _assert_push_url_rejected(reply, *, v1: bool):
    # a2a-sdk 1.0's v0.3 JSON-RPC adapter reports every handler error as
    # -32603 with the error's message; the v1 dispatcher keeps -32602.
    assert reply["error"]["message"] == _PUSH_REJECTED
    if v1:
        assert reply["error"]["code"] == -32602


# One message for every rejection: the reason (and any resolved address)
# stays in the server log, so clients can't probe internal DNS.
_PUSH_REJECTED = "Rejected push notification URL: only public http(s) URLs are allowed"


async def _create_push_config(client, task_id, url, *, v1: bool):
    if v1:
        body = {
            "jsonrpc": "2.0",
            "id": "p",
            "method": "CreateTaskPushNotificationConfig",
            "params": {"taskId": task_id, "url": url},
        }
        response = await client.request("POST", "/", json=body, headers={"A2A-Version": "1.0"})
        return response.json()
    body = {
        "jsonrpc": "2.0",
        "id": "p",
        "method": "tasks/pushNotificationConfig/set",
        "params": {"taskId": task_id, "pushNotificationConfig": {"url": url}},
    }
    return (await client.request("POST", "/", json=body)).json()


async def _settle():
    import asyncio

    for _ in range(20):
        await asyncio.sleep(0.01)


class TestPushNotifications:
    def test_card_push_default(self):
        assert build_agent_card(StubAdapter()).capabilities.push_notifications is False

    def test_card_push_opt_in(self):
        card = build_agent_card(StubAdapter(), push_notifications=True)
        assert card.capabilities.push_notifications is True

    @pytest.mark.parametrize("v1", [True, False])
    async def test_push_disabled_rejects_config_set(self, no_real_push_post, v1):
        async with _client(to_a2a(StubAdapter())) as client:
            task = (await _send_task(client))["result"]
            reply = await _create_push_config(client, task["id"], "https://example.com/h", v1=v1)
        assert "error" in reply
        assert "push" in reply["error"]["message"].lower()

    async def test_push_disabled_ignores_inline_config(self, no_real_push_post):
        async with _client(to_a2a(StubAdapter())) as client:
            reply = await _send_task(
                client, pushNotificationConfig={"url": "https://example.com/i"}
            )
            v1_reply = await _send_task_with_push_v1(client, "https://example.com/j")
            await _settle()
        assert reply["result"]["status"]["state"] == "completed"
        assert v1_reply["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert no_real_push_post == []

    async def test_push_disabled_ignores_inline_config_when_streaming(self, no_real_push_post):
        payload = {
            "jsonrpc": "2.0",
            "id": "st",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "m-stream",
                    "parts": [{"kind": "text", "text": "hi"}],
                },
                "configuration": {"pushNotificationConfig": {"url": "https://example.com/s"}},
            },
        }
        app = to_a2a(StreamingStubAdapter(["a", "b"]))
        async with _client(app) as client:
            response = await client.request("POST", "/", json=payload)
            await _settle()
        assert "completed" in response.text
        assert no_real_push_post == []

    @pytest.mark.parametrize("url", _PUBLIC_PUSH_URLS)
    async def test_validate_push_url_accepts_public(self, fake_dns, url):
        from a2a_adapter.server import _validate_push_url

        await _validate_push_url(url, allow_private=False)

    async def test_rejection_does_not_leak_resolution(self, fake_dns, push_receiver, caplog):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            with caplog.at_level("INFO", logger="a2a_adapter.server"):
                unknown = await _send_task_with_push_v1(client, "http://unknown.example/x")
                private = await _send_task_with_push_v1(client, "http://rebound.example/x")
        assert unknown["error"] == private["error"]
        assert "127.0.0.1" not in str(private)
        assert "127.0.0.1" in caplog.text, "the reason is logged on the server"

    async def test_rest_push_config_rejects_private_url(self, fake_dns, push_receiver):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            task = (await _send_task(client))["result"]
            response = await client.request(
                "POST",
                f"/tasks/{task['id']}/pushNotificationConfigs",
                json={"url": "http://127.0.0.1/x"},
                headers={"A2A-Version": "1.0"},
            )
        assert response.status_code == 400
        assert response.json()["error"]["message"] == _PUSH_REJECTED

    async def test_resolve_times_out(self, monkeypatch):
        import asyncio

        import a2a_adapter.server as server_module

        async def hanging(*args, **kwargs):
            await asyncio.sleep(30)

        monkeypatch.setattr(server_module, "_RESOLVE_TIMEOUT", 0.05)
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", hanging)
        with pytest.raises(asyncio.TimeoutError):
            await server_module._resolve("slow.example", 80)
        with pytest.raises(ValueError):
            await server_module._validate_push_url("http://slow.example/x", allow_private=False)

    @pytest.mark.parametrize("url", _PRIVATE_PUSH_URLS)
    async def test_validate_push_url_rejects_private(self, fake_dns, url):
        from a2a_adapter.server import _validate_push_url

        with pytest.raises(ValueError):
            await _validate_push_url(url, allow_private=False)

    @pytest.mark.parametrize("url", ["http://mixed.example/x", "http://unknown.example/x"])
    async def test_validate_push_url_rejects_any_private_or_unresolvable(self, fake_dns, url):
        from a2a_adapter.server import _validate_push_url

        with pytest.raises(ValueError):
            await _validate_push_url(url, allow_private=False)

    async def test_validate_push_url_resolution_timeout_rejects(self, monkeypatch):
        import asyncio

        from a2a_adapter.server import _validate_push_url

        async def slow(host, port):
            raise asyncio.TimeoutError

        monkeypatch.setattr("a2a_adapter.server._resolve", slow)
        with pytest.raises(ValueError):
            await _validate_push_url("https://slow.example/x", allow_private=False)

    @pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://anything.internal:8080/x"])
    async def test_validate_push_url_allow_private_skips_dns(self, monkeypatch, url):
        from a2a_adapter.server import _validate_push_url

        async def no_dns(host, port):
            raise AssertionError("DNS must not be used")

        monkeypatch.setattr("a2a_adapter.server._resolve", no_dns)
        await _validate_push_url(url, allow_private=True)
        with pytest.raises(ValueError):
            await _validate_push_url("ftp://x/y", allow_private=True)

    @pytest.mark.parametrize("v1", [True, False])
    @pytest.mark.parametrize("url", _PRIVATE_PUSH_URLS)
    async def test_private_push_url_rejected_inline(self, fake_dns, push_receiver, url, v1):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            if v1:
                reply = await _send_task_with_push_v1(client, url)
            else:
                reply = await _send_task(client, pushNotificationConfig={"url": url})
            await _settle()
        _assert_push_url_rejected(reply, v1=v1)
        assert push_receiver == []

    @pytest.mark.parametrize("v1", [True, False])
    @pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://rebound.example/x"])
    async def test_private_push_url_rejected_on_config_set(self, fake_dns, push_receiver, url, v1):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            task = (await _send_task(client))["result"]
            reply = await _create_push_config(client, task["id"], url, v1=v1)
        _assert_push_url_rejected(reply, v1=v1)
        assert push_receiver == []

    async def test_public_push_url_accepted(self, fake_dns, push_receiver):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            task = (await _send_task(client))["result"]
            reply = await _create_push_config(client, task["id"], "https://example.com/h", v1=True)
        assert reply["result"]["url"] == "https://example.com/h"

    async def test_allow_private_push_urls(self, push_receiver):
        app = to_a2a(StubAdapter(), push_notifications=True, allow_private_push_urls=True)
        async with _client(app) as client:
            task = (await _send_task(client))["result"]
            for url in ("http://127.0.0.1/x", "http://10.0.0.5/x", "http://receiver.test/x"):
                reply = await _create_push_config(client, task["id"], url, v1=True)
                assert reply["result"]["url"] == url

    async def test_send_time_rebinding_blocks_connect(self, monkeypatch, push_network):
        lookups = []

        async def rebinding(host, port):
            lookups.append(host)
            # Public when the config is registered, loopback when it is used.
            return ["93.184.216.34"] if len(lookups) == 1 else ["127.0.0.1"]

        monkeypatch.setattr("a2a_adapter.server._resolve", rebinding)
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            reply = await _send_task(
                client, pushNotificationConfig={"url": "https://hooks.example/h"}
            )
            await _settle()
        assert reply["result"]["status"]["state"] == "completed"
        assert len(lookups) >= 2, "the host is resolved again when connecting"
        assert push_network.connects == []

    async def test_send_connects_to_address_resolved_at_connect(self, monkeypatch, push_network):
        lookups = []

        async def moving(host, port):
            lookups.append(host)
            return ["93.184.216.34"] if len(lookups) == 1 else ["93.184.216.35"]

        monkeypatch.setattr("a2a_adapter.server._resolve", moving)
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            await _send_task(client, pushNotificationConfig={"url": "https://hooks.example/h"})
            await _settle()
        assert [c[:2] for c in push_network.connects] == [("93.184.216.35", 443)]

    @pytest.mark.parametrize("host", ["hooks.example", "hooks.example."])
    async def test_https_push_pins_ip_and_keeps_hostname(self, push_network, fake_dns, host):
        import ssl

        fake_dns["hooks.example."] = ["93.184.216.34"]
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            reply = await _send_task(client, pushNotificationConfig={"url": f"https://{host}/h"})
            await _settle()
        assert reply["result"]["status"]["state"] == "completed"
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 443)]
        stream = push_network.streams[0]
        assert stream.tls["server_hostname"] == host
        assert stream.tls["ssl_context"].check_hostname is True
        assert stream.tls["ssl_context"].verify_mode == ssl.CERT_REQUIRED
        assert stream.data.startswith(b"POST /h HTTP/1.1\r\n")
        assert f"\r\nHost: {host}\r\n".encode() in stream.data

    async def test_http_push_pins_ip_and_keeps_host_header(self, push_network):
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            await _send_task(client, pushNotificationConfig={"url": "http://hooks.example:8080/h"})
            await _settle()
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 8080)]
        stream = push_network.streams[0]
        assert stream.tls is None
        assert b"\r\nHost: hooks.example:8080\r\n" in stream.data

    async def test_push_client_does_not_follow_redirects(self, push_network):
        # A public target, so only follow_redirects (not the pinning) can
        # stop the second request.
        push_network.responses = [_redirect("https://example.com/next")]
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with app.router.lifespan_context(app), _client(app) as client:
            reply = await _send_task(
                client, pushNotificationConfig={"url": "https://hooks.example/h"}
            )
            await _settle()
        assert reply["result"]["status"]["state"] == "completed"
        assert len(push_network.connects) == 1
        assert push_network.streams[0].data.startswith(b"POST /h ")

    async def test_redirect_to_private_never_connects(self, push_network):
        # Holds even if redirects were followed: the pinned backend would
        # refuse the 169.254.169.254 hop (follow_redirects itself is covered
        # by test_push_client_does_not_follow_redirects).
        push_network.responses = [_redirect("http://169.254.169.254/latest")]
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            reply = await _send_task(
                client, pushNotificationConfig={"url": "https://hooks.example/h"}
            )
            await _settle()
        assert reply["result"]["status"]["state"] == "completed"
        assert len(push_network.connects) == 1

    async def test_followed_redirect_is_revalidated(self, push_network, monkeypatch):
        made = _capture_push_client(monkeypatch)
        to_a2a(StubAdapter(), push_notifications=True)
        ((_, push_client),) = made
        push_network.responses = [_redirect("http://rebound.example/")]
        async with push_client:
            with pytest.raises(httpx.ConnectError):
                await push_client.post("https://hooks.example/h", follow_redirects=True)
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 443)]

    async def test_pool_keys_by_hostname_not_ip(self, push_network, fake_dns, monkeypatch):
        fake_dns["a.example"] = ["93.184.216.34"]
        fake_dns["b.example"] = ["93.184.216.34"]
        made = _capture_push_client(monkeypatch)
        to_a2a(StubAdapter(), push_notifications=True)
        ((_, push_client),) = made
        async with push_client:
            await push_client.post("https://a.example/")
            await push_client.post("https://b.example/")
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 443)] * 2
        assert [s.tls["server_hostname"] for s in push_network.streams] == [
            "a.example",
            "b.example",
        ]

    async def test_allow_private_keeps_default_client(self, push_receiver, monkeypatch):
        made = _capture_push_client(monkeypatch)
        app = to_a2a(StubAdapter(), push_notifications=True, allow_private_push_urls=True)
        assert [kwargs for kwargs, _ in made] == [{"follow_redirects": False}]
        async with _client(app) as client:
            await _send_task(client, pushNotificationConfig={"url": "http://127.0.0.1/x"})
            await _settle()
        assert push_receiver and str(push_receiver[0].url) == "http://127.0.0.1/x"

    async def test_pinned_push_ignores_env_proxy(self, push_network, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:3128")
        monkeypatch.setenv("ALL_PROXY", "http://10.0.0.1:3128")
        made = _capture_push_client(monkeypatch)
        app = to_a2a(StubAdapter(), push_notifications=True)
        async with _client(app) as client:
            await _send_task(client, pushNotificationConfig={"url": "https://hooks.example/h"})
            await _settle()
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 443)]
        ((_, push_client),) = made
        assert push_client._mounts == {}

    @pytest.mark.parametrize("pool", ["object", "subclass"])
    def test_unsupported_transport_fails_loudly(self, monkeypatch, pool):
        class SubPool(httpcore.AsyncConnectionPool):
            pass

        class StubTransport:
            def __init__(self, *args, **kwargs):
                self._pool = object() if pool == "object" else SubPool()

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", StubTransport)
        with pytest.raises(RuntimeError, match="cannot pin"):
            to_a2a(StubAdapter(), push_notifications=True)

    def test_push_disabled_creates_no_client(self, monkeypatch):
        import a2a_adapter.server as server_module

        made = []
        monkeypatch.setattr(server_module, "_make_push_client", lambda **k: made.append(k))
        to_a2a(StubAdapter())
        assert made == []

    async def test_empty_url_config_still_ignored(self):
        from a2a.types import TaskPushNotificationConfig

        from a2a_adapter.server import _SafeInMemoryPushNotificationConfigStore

        from a2a.server.context import ServerCallContext

        store = _SafeInMemoryPushNotificationConfigStore()
        context = ServerCallContext()
        await store.set_info("t", TaskPushNotificationConfig(url=""), context)
        assert await store.get_info("t", context) == []

    def test_custom_card_disagreeing_with_flag_is_rejected(self):
        card = build_agent_card(StubAdapter())
        with pytest.raises(ValueError, match="push_notifications"):
            to_a2a(StubAdapter(), agent_card=card, push_notifications=True)
        to_a2a(StubAdapter(), agent_card=card, push_notifications=False)

    async def test_custom_card_capability_decides(self, push_receiver):
        card = build_agent_card(StubAdapter(), push_notifications=True)
        app = to_a2a(StubAdapter(), agent_card=card, allow_private_push_urls=True)
        async with _client(app) as client:
            task = (await _send_task(client))["result"]
            reply = await _create_push_config(client, task["id"], "http://r.test/x", v1=True)
        assert reply["result"]["url"] == "http://r.test/x"


class TestPinnedPushBackend:
    @staticmethod
    def _backend(network):
        from a2a_adapter.server import _PinnedPushBackend

        return _PinnedPushBackend(network)

    @staticmethod
    def _recording_resolver(monkeypatch, answer):
        calls = []

        async def resolve(host, port):
            calls.append(host)
            if isinstance(answer, BaseException):
                raise answer
            return list(answer(len(calls)) if callable(answer) else answer)

        monkeypatch.setattr("a2a_adapter.server._resolve", resolve)
        return calls

    async def test_connects_to_validated_ip_without_resolving_again(
        self, push_network, monkeypatch
    ):
        calls = self._recording_resolver(
            monkeypatch, lambda n: ["93.184.216.34"] if n == 1 else ["127.0.0.1"]
        )
        await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert [c[:2] for c in push_network.connects] == [("93.184.216.34", 443)]
        assert calls == ["hooks.example"]

    @pytest.mark.parametrize(
        "addresses",
        [
            ["127.0.0.1"],
            ["93.184.216.34", "10.0.0.1"],
            ["93.184.216.3" + str(i) for i in range(5)] + ["169.254.169.254"],
            ["::ffff:127.0.0.1"],
            ["64:ff9b::a9fe:a9fe"],
            [],
        ],
    )
    async def test_rejects_any_non_public_address(self, push_network, monkeypatch, addresses):
        self._recording_resolver(monkeypatch, addresses)
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert push_network.connects == []

    async def test_duplicate_addresses_tried_once(self, push_network, monkeypatch):
        self._recording_resolver(monkeypatch, ["93.184.216.34"] * 3)
        push_network.fail.add("93.184.216.34")
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert [c[0] for c in push_network.connects] == ["93.184.216.34"]

    async def test_falls_back_to_next_address(self, push_network, monkeypatch):
        self._recording_resolver(monkeypatch, ["93.184.216.34", "93.184.216.35"])
        push_network.fail.add("93.184.216.34")
        await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert [c[0] for c in push_network.connects] == ["93.184.216.34", "93.184.216.35"]
        push_network.fail.add("93.184.216.35")
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp("hooks.example", 443)

    async def test_attempts_share_one_deadline(self, push_network, monkeypatch):
        self._recording_resolver(monkeypatch, ["93.184.216.34", "93.184.216.35", "93.184.216.36"])
        # The first address times out after 0.3 s; the second waits out
        # whatever time it is given. Wide margins: CI runners can be slow.
        push_network.hang.update({"93.184.216.34": 0.3, "93.184.216.35": None})
        with pytest.raises(httpcore.ConnectTimeout):
            await self._backend(push_network).connect_tcp("hooks.example", 443, timeout=1.0)
        timeouts = [c[2] for c in push_network.connects]
        # The second attempt gets only what the first left over (~0.7 s),
        # and once that runs out the third address is not tried.
        assert len(timeouts) == 2, timeouts
        assert timeouts[0] == pytest.approx(1.0, abs=0.05)
        assert 0.3 < timeouts[1] < 0.75, timeouts

    async def test_ipv6_address_passed_as_literal(self, push_network, monkeypatch):
        self._recording_resolver(monkeypatch, ["2606:4700:4700::1111"])
        await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert [c[0] for c in push_network.connects] == ["2606:4700:4700::1111"]

    @pytest.mark.parametrize("host", ["8.8.8.8", "2606:4700:4700::1111"])
    async def test_public_ip_literal_skips_dns(self, push_network, monkeypatch, host):
        calls = self._recording_resolver(monkeypatch, AssertionError("no DNS"))
        await self._backend(push_network).connect_tcp(host, 443)
        assert [c[0] for c in push_network.connects] == [host]
        assert calls == []

    @pytest.mark.parametrize("host", ["169.254.169.254", "::1"])
    async def test_private_ip_literal_rejected(self, push_network, host):
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp(host, 80)
        assert push_network.connects == []

    @pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f.1"])
    async def test_numeric_names_validated_by_resolved_address(self, push_network, fake_dns, host):
        fake_dns[host] = ["127.0.0.1"]  # what getaddrinfo answers for these
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp(host, 80)
        assert push_network.connects == []

    @pytest.mark.parametrize("host", ["localhost", "api.localhost", "localhost.", "LocalHost"])
    async def test_localhost_names_rejected_without_dns(self, push_network, monkeypatch, host):
        calls = self._recording_resolver(monkeypatch, ["93.184.216.34"])
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp(host, 80)
        assert calls == []
        assert push_network.connects == []

    @pytest.mark.parametrize(
        "error", [OSError("no such host"), TimeoutError(), UnicodeError("bad idna")]
    )
    async def test_resolver_failure_is_connect_error(self, push_network, monkeypatch, error):
        self._recording_resolver(monkeypatch, error)
        with pytest.raises(httpcore.ConnectError):
            await self._backend(push_network).connect_tcp("hooks.example", 443)
        assert push_network.connects == []

    async def test_unix_socket_refused_and_sleep_delegates(self, push_network):
        backend = self._backend(push_network)
        with pytest.raises(httpcore.ConnectError):
            await backend.connect_unix_socket("/tmp/x.sock")
        await backend.sleep(0.5)
        assert push_network.slept == [0.5]


class TestServeAgentPush:
    @pytest.fixture(autouse=True)
    def _no_real_server(self, monkeypatch):
        from a2a_adapter import server as server_module

        self.servers = []
        self.cards = []
        monkeypatch.setattr(server_module._ReadyServer, "run", _fake_started_run(self.servers))
        real_to_a2a = server_module.to_a2a

        def spy(adapter, agent_card=None, *args, **kwargs):
            self.cards.append(agent_card)
            return real_to_a2a(adapter, agent_card, *args, **kwargs)

        monkeypatch.setattr(server_module, "to_a2a", spy)

    def test_serve_agent_push_opt_in_starts(self):
        from a2a_adapter.server import serve_agent

        serve_agent(StubAdapter(), push_notifications=True)
        assert self.cards[-1].capabilities.push_notifications is True
        serve_agent(StubAdapter())
        assert self.cards[-1].capabilities.push_notifications is False

    def test_serve_agent_custom_card_disagreeing_is_rejected(self):
        from a2a_adapter.server import serve_agent

        card = build_agent_card(StubAdapter())
        with pytest.raises(ValueError, match="push_notifications"):
            serve_agent(StubAdapter(), agent_card=card, push_notifications=True)
        assert self.servers == []

    def test_warns_when_push_enabled_without_auth(self, caplog):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), push_notifications=True)
        assert any("push" in r.getMessage().lower() for r in caplog.records)
        caplog.clear()
        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(StubAdapter(), push_notifications=True, auth_token="t")
        assert not any("push" in r.getMessage().lower() for r in caplog.records)

    def test_push_warnings_in_display_mode(self, caplog):
        import io

        from a2a_adapter.server import serve_agent
        from a2a_adapter.terminal import ActivityLog

        class QuietDisplay:
            activity = ActivityLog(io.StringIO(), color=False)

            def install_logging(self, level):
                pass

            def ready(self, card):
                pass

            def close(self):
                pass

            def uninstall_logging(self):
                pass

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(
                StubAdapter(),
                display=QuietDisplay(),
                push_notifications=True,
                allow_private_push_urls=True,
            )
        messages = " ".join(r.getMessage().lower() for r in caplog.records)
        assert "without auth" in messages and "private" in messages

    def test_warns_when_private_push_urls_allowed(self, caplog):
        from a2a_adapter.server import serve_agent

        with caplog.at_level("WARNING", logger="a2a_adapter.server"):
            serve_agent(
                StubAdapter(), push_notifications=True, allow_private_push_urls=True, auth_token="t"
            )
        assert any("private" in r.getMessage().lower() for r in caplog.records)
