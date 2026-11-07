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
