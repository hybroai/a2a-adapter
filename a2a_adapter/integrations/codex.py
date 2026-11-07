"""
OpenAI Codex CLI adapter for A2A Protocol.

Wraps the Codex CLI as a subprocess via BaseA2AAdapter's built-in
subprocess infrastructure. Only implements CLI-specific hooks:
command building and output parsing.
"""

import json
import logging
import os
from collections.abc import Sequence
from typing import Any

from ..base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    CommandResult,
    ParseResult,
    default_state_dir,
)

logger = logging.getLogger(__name__)

# What Codex can print when `exec resume <id>` names an unknown thread (found
# in the Codex 0.162.1 binary; which one, and on which stream, is unconfirmed).
# Lowercase; matched against stdout and stderr.
_STALE_SESSION_PATTERNS = (
    "no saved session found with id ",
    "no rollout found for thread id ",
    "no rollout found for conversation id ",
    "thread not found: ",
)


class CodexAdapter(BaseA2AAdapter):
    """Adapter for the OpenAI Codex CLI.

    Args:
        bypass_approvals: Enable ``--dangerously-bypass-approvals-and-sandbox``.
            Disables all approval prompts and sandboxing. When ``None``
            (default), falls back to ``A2A_CODEX_BYPASS_APPROVALS`` env var
            (``1`` or ``true`` to enable). **Security risk** — only enable
            in sandboxed, trusted environments.
        skip_git_check: Enable ``--skip-git-repo-check``. Allows Codex to
            run outside git repositories. When ``None`` (default), falls
            back to ``A2A_CODEX_SKIP_GIT_CHECK`` env var.

    Example::

        from a2a_adapter import CodexAdapter, serve_agent

        adapter = CodexAdapter(working_dir="/path/to/project")
        serve_agent(adapter, port=9011)
    """

    def __init__(
        self,
        working_dir: str,
        timeout: int = 600,
        codex_path: str = "codex",
        cli_args: Sequence[str] | None = None,
        env_vars: dict[str, str] | None = None,
        session_store_path: str | None = None,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
        bypass_approvals: bool | None = None,
        skip_git_check: bool | None = None,
    ) -> None:
        if isinstance(cli_args, str):
            raise TypeError("cli_args must be a sequence of arguments, not a string")
        super().__init__(
            working_dir=working_dir,
            timeout=timeout,
            env_vars=env_vars,
            session_store_path=session_store_path
            or os.path.join(default_state_dir("codex", working_dir), "sessions.json"),
        )
        self.codex_path = codex_path
        self.cli_args = list(cli_args or [])
        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        if bypass_approvals is not None:
            self.bypass_approvals = bypass_approvals
        else:
            self.bypass_approvals = os.getenv("A2A_CODEX_BYPASS_APPROVALS", "").lower() in (
                "1",
                "true",
            )
        if self.bypass_approvals:
            logger.warning(
                "CodexAdapter: --dangerously-bypass-approvals-and-sandbox is "
                "ENABLED. Codex will bypass all approval prompts and "
                "sandboxing. Only use in trusted, sandboxed environments."
            )

        if skip_git_check is not None:
            self.skip_git_check = skip_git_check
        else:
            self.skip_git_check = os.getenv("A2A_CODEX_SKIP_GIT_CHECK", "").lower() in ("1", "true")
        if self.skip_git_check:
            logger.warning(
                "CodexAdapter: --skip-git-repo-check is ENABLED. "
                "Codex will run outside of git repositories."
            )

    # ──── Public Interface ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        context = kwargs.get("context")
        key = context_id or "_default"
        task_id = context.task_id if context else key
        lock = self._get_context_lock(key)
        async with lock:
            return await self._run_subprocess(key, task_id, user_input=user_input)

    def supports_streaming(self) -> bool:
        return False

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "Codex",
            description=self._description or "OpenAI Codex coding agent",
            streaming=False,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Hooks ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        cmd = [self.codex_path, "exec", *self.cli_args]
        thread_id = self._sessions.get(context_key)
        if thread_id:
            cmd.extend(["resume", thread_id])
        cmd.append("--json")
        if self.bypass_approvals:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if self.skip_git_check:
            cmd.append("--skip-git-repo-check")
        cmd.append(message)
        return CommandResult(args=cmd, used_resume=bool(thread_id))

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        thread_id: str | None = None
        text_parts: list[str] = []

        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "thread.started":
                thread_id = self._extract_thread_id(event)
            elif event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
            elif event_type == "turn.failed":
                error_msg = event.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Codex turn failed: {error_msg}")

        if not text_parts:
            raise RuntimeError("Codex returned no visible output")

        response_text = "\n\n".join(text_parts)
        return ParseResult(text=response_text, session_id=thread_id)

    def _extract_error(self, stdout_text: str) -> str | None:
        error_msg: str | None = None
        for line in stdout_text.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn.failed":
                error = event.get("error")
                if isinstance(error, dict) and error.get("message"):
                    error_msg = str(error["message"])
            elif event.get("type") == "error" and event.get("message") and error_msg is None:
                error_msg = str(event["message"])
        return self._unwrap_api_error(error_msg) if error_msg else None

    def _is_stale_session_error(self, stdout_text: str, stderr_text: str) -> bool:
        # Only Codex's own error events and stderr: the rest of stdout is the
        # transcript, where agent text or command output may quote a pattern.
        errors = [*self._error_records(stdout_text), stderr_text]
        output = "\n".join(errors).lower()
        return any(pattern in output for pattern in _STALE_SESSION_PATTERNS)

    @staticmethod
    def _error_records(stdout_text: str) -> list[str]:
        """Messages of ``error`` and ``turn.failed`` events in ``--json`` output."""
        records = []
        for line in stdout_text.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                records.append(str(event.get("message", "")))
            elif event.get("type") == "turn.failed":
                error = event.get("error")
                records.append(str(error.get("message", "") if isinstance(error, dict) else error))
        return records

    def _binary_not_found_message(self) -> str:
        return (
            f"Codex binary not found at '{self.codex_path}'. "
            "Ensure the OpenAI Codex CLI is installed and in PATH."
        )

    # ──── Private ────

    @staticmethod
    def _unwrap_api_error(message: str) -> str:
        """Codex embeds raw API error JSON in messages; surface its inner text."""
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return message
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
        return message

    @staticmethod
    def _extract_thread_id(event: dict[str, Any]) -> str | None:
        thread_id = event.get("thread_id")
        if thread_id:
            return str(thread_id)
        thread = event.get("thread", {})
        if isinstance(thread, dict):
            tid = thread.get("id")
            if tid:
                return str(tid)
        return None
