"""
Claude Code adapter for A2A Protocol.

Wraps the Claude Code CLI as a subprocess via BaseA2AAdapter's built-in
subprocess infrastructure. Only implements CLI-specific hooks:
command building, output parsing, and stream line handling.
"""

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Sequence

from ..base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    CommandResult,
    ParseResult,
    StreamEvent,
)

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter(BaseA2AAdapter):
    """Adapter for Claude Code via the ``claude`` CLI.

    Args:
        skip_permissions: Enable ``--dangerously-skip-permissions`` flag.
            Allows Claude Code to execute tools without confirmation.
            When ``None`` (default), falls back to the
            ``A2A_CLAUDE_SKIP_PERMISSIONS`` env var (``1`` or ``true``
            to enable). **Security risk** — only enable in sandboxed,
            trusted environments.

    Example::

        from a2a_adapter import ClaudeCodeAdapter, serve_agent

        adapter = ClaudeCodeAdapter(working_dir="/path/to/project")
        serve_agent(adapter, port=9010)
    """

    def __init__(
        self,
        working_dir: str,
        timeout: int = 600,
        claude_path: str = "claude",
        cli_args: Sequence[str] | None = None,
        env_vars: dict[str, str] | None = None,
        session_store_path: str | None = None,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
        skip_permissions: bool | None = None,
    ) -> None:
        if isinstance(cli_args, str):
            raise TypeError("cli_args must be a sequence of arguments, not a string")
        super().__init__(
            working_dir=working_dir,
            timeout=timeout,
            env_vars=env_vars,
            session_store_path=session_store_path
            or os.path.join(
                working_dir, ".a2a-adapter", "claude-code", "sessions.json"
            ),
        )
        self.session_id = f"a2a-{uuid.uuid4().hex[:12]}"
        self.claude_path = claude_path
        self.cli_args = list(cli_args or [])
        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        if skip_permissions is not None:
            self.skip_permissions = skip_permissions
        else:
            self.skip_permissions = os.getenv(
                "A2A_CLAUDE_SKIP_PERMISSIONS", ""
            ).lower() in ("1", "true")
        if self.skip_permissions:
            logger.warning(
                "ClaudeCodeAdapter: --dangerously-skip-permissions is ENABLED. "
                "Claude Code will execute tools without permission prompts. "
                "Only use in trusted, sandboxed environments."
            )

    # ──── Public Interface ────

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> str:
        context = kwargs.get("context")
        prefix = getattr(self, "session_id", "a2a")
        key = f"{prefix}-{context_id}" if context_id else prefix
        task_id = context.task_id if context else key
        lock = self._get_context_lock(key)
        async with lock:
            return await self._run_subprocess(
                key, task_id, user_input=user_input
            )

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        context = kwargs.get("context")
        prefix = getattr(self, "session_id", "a2a")
        key = f"{prefix}-{context_id}" if context_id else prefix
        task_id = context.task_id if context else key
        lock = self._get_context_lock(key)
        async with lock:
            async for chunk in self._run_subprocess_streaming(
                key, task_id, user_input=user_input
            ):
                yield chunk

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "Claude Code",
            description=self._description or "Claude Code coding agent",
            streaming=True,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Hooks ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        cmd = [
            self.claude_path,
            *self.cli_args,
            # The prompt goes through stdin (see CommandResult.stdin): `claude -p`
            # reads it from there when no prompt argument is given.
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--disallowedTools", "AskUserQuestion",
        ]
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        session_id = self._sessions.get(context_key)
        if session_id:
            cmd.extend(["--resume", session_id])
        return CommandResult(args=cmd, used_resume=bool(session_id), stdin=message)

    def _parse_invoke_output(
        self, stdout_text: str, context_key: str
    ) -> ParseResult:
        accumulated: list[str] = []
        session_id: str | None = None

        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj_type = obj.get("type")
            if obj_type == "error":
                error_msg = (
                    obj.get("error", {}).get("message", "")
                    if isinstance(obj.get("error"), dict)
                    else str(obj.get("error", "Unknown error"))
                )
                raise RuntimeError(f"Claude Code error: {error_msg}")
            if obj_type == "assistant":
                accumulated.extend(self._extract_text(obj))
            if obj_type == "result":
                session_id = obj.get("session_id") or None

        if not accumulated:
            raise RuntimeError("Claude Code returned no output")
        return ParseResult(text="".join(accumulated), session_id=session_id)

    def _handle_stream_line(self, line: str) -> StreamEvent | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None

        obj_type = obj.get("type")
        if obj_type == "error":
            error_msg = (
                obj.get("error", {}).get("message", "")
                if isinstance(obj.get("error"), dict)
                else str(obj.get("error", "Unknown error"))
            )
            return StreamEvent(error=f"Claude Code error: {error_msg}")
        if obj_type == "assistant":
            chunks = self._extract_text(obj)
            if chunks:
                return StreamEvent(text="".join(chunks))
            return None
        if obj_type == "result":
            if obj.get("is_error") and obj.get("result"):
                return StreamEvent(error=f"Claude Code error: {obj['result']}")
            return StreamEvent(session_id=obj.get("session_id") or None)
        return None

    def _extract_error(self, stdout_text: str) -> str | None:
        for line in reversed(stdout_text.splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "result":
                if obj.get("is_error") and obj.get("result"):
                    return str(obj["result"])
                return None
        return None

    def _binary_not_found_message(self) -> str:
        return (
            f"Claude Code binary not found at '{self.claude_path}'. "
            "Ensure Claude Code CLI is installed and in PATH."
        )

    # ──── Private ────

    @staticmethod
    def _extract_text(obj: dict) -> list[str]:
        """Extract text from an assistant event's message.content blocks."""
        chunks: list[str] = []
        message = obj.get("message", {})
        if not isinstance(message, dict):
            return chunks
        content = message.get("content", [])
        if not isinstance(content, list):
            return chunks
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    chunks.append(text)
        return chunks
