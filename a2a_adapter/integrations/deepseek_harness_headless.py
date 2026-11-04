"""
DeepSeek Harness headless CLI adapter for A2A Protocol.

Wraps ``dsh --profile headless`` as a subprocess via BaseA2AAdapter's
built-in subprocess infrastructure, with the task on stdin. This adapter
keeps no state: it does not use dsh's ``--session-id`` resume, so every turn
is a fresh dsh session and ``context_id`` carries no history. Prefer
``DeepSeekHarnessAdapter`` (the SDK runtime), which keeps one durable session
per conversation.
"""

import os
import uuid
from collections.abc import Sequence
from typing import Any

from ..base_adapter import AdapterMetadata, BaseA2AAdapter, CommandResult, ParseResult
from .deepseek_harness import _INSTALL_COMMANDS, DeepSeekHarnessAdapter


class DeepSeekHarnessHeadlessAdapter(BaseA2AAdapter):
    """Stateless adapter for the DeepSeek Harness ``dsh`` headless CLI.

    Each request runs ``dsh --profile headless`` once in ``working_dir`` and
    returns the final answer it prints. ``DSH_HOME`` is resolved like
    ``DeepSeekHarnessAdapter`` (``dsh_home``, else a non-empty ``DSH_HOME``,
    else a per-user state directory) unless ``env_vars`` already sets it; a
    relative ``env_vars`` value is taken relative to ``working_dir``.

    Without ``dsh_path`` the adapter launches the runtime bundled with
    deepseek-harness-sdk directly, else ``dsh`` on PATH. On Windows the
    ``dsh`` console script waits on the runtime instead of replacing itself,
    so killing it on cancel or timeout leaves the runtime running; pass the
    runtime executable itself as ``dsh_path`` there.

    Example::

        from a2a_adapter import DeepSeekHarnessHeadlessAdapter, serve_agent

        adapter = DeepSeekHarnessHeadlessAdapter(working_dir="/path/to/project")
        serve_agent(adapter, host="127.0.0.1", port=9014)
    """

    def __init__(
        self,
        working_dir: str,
        timeout: int = 600,
        dsh_path: str | None = None,
        cli_args: Sequence[str] | None = None,
        env_vars: dict[str, str] | None = None,
        dsh_home: str | None = None,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        agent_provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        if isinstance(cli_args, str):
            raise TypeError("cli_args must be a sequence of arguments, not a string")
        super().__init__(working_dir=working_dir, timeout=timeout, env_vars=env_vars)
        env_home = self.env_vars.get("DSH_HOME", "").strip()
        if env_home:
            # dsh runs in working_dir, so resolve a relative home there too.
            self.env_vars["DSH_HOME"] = os.path.abspath(
                os.path.join(working_dir, os.path.expanduser(env_home))
            )
        else:
            self.env_vars["DSH_HOME"] = DeepSeekHarnessAdapter._resolve_dsh_home(
                dsh_home, working_dir
            )
        self.dsh_path = dsh_path
        self._launched = dsh_path or "dsh"
        self.cli_args = list(cli_args or [])
        self._name = name
        self._description = description
        self._skills = skills or []
        self._agent_provider = agent_provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

    # ──── Public Interface ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        if not user_input.strip():
            raise ValueError("DeepSeek Harness headless needs a non-empty task")
        context = kwargs.get("context")
        key = context_id or "_default"
        # Turns share no state and take no lock, so concurrent calls without
        # a context need their own ids to stay cancellable.
        task_id = context.task_id if context else uuid.uuid4().hex
        os.makedirs(self.env_vars["DSH_HOME"], exist_ok=True)
        return await self._run_subprocess(key, task_id, user_input=user_input)

    def supports_streaming(self) -> bool:
        return False

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "DeepSeek Harness (headless)",
            description=self._description or "DeepSeek Harness coding agent, one task per request",
            streaming=False,
            skills=self._skills,
            provider=self._agent_provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Hooks ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        # The task goes through stdin: dsh reads a positional task starting
        # with "-" as an option even after "--", and argv has a size limit.
        launch = self._launch_args()
        self._launched = launch[0]
        args = [*launch, "--profile", "headless", *self.cli_args]
        return CommandResult(args=args, used_resume=False, stdin=message)

    def _launch_args(self) -> tuple[str, ...]:
        if self.dsh_path is not None:
            return (self.dsh_path,)
        try:
            from deepseek_harness_runtime import resolve_bundled_launch_args

            return tuple(resolve_bundled_launch_args())
        except (ImportError, FileNotFoundError):
            return ("dsh",)

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        return ParseResult(text=stdout_text.strip(), session_id=None)

    def _binary_not_found_message(self) -> str:
        return (
            f"dsh binary not found at '{self._launched}'. Pass dsh_path, or "
            f"install the runtime that ships with the extra.\n{_INSTALL_COMMANDS}"
        )
