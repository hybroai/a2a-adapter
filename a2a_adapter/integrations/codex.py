"""
OpenAI Codex CLI adapter for A2A Protocol.

Wraps the Codex CLI as a subprocess via BaseA2AAdapter's built-in
subprocess infrastructure. Only implements CLI-specific hooks:
command building and output parsing.
"""

import json
import logging
import os
from typing import Any

from ..base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    CommandResult,
    ParseResult,
)

logger = logging.getLogger(__name__)


class CodexAdapter(BaseA2AAdapter):
    """Adapter for the OpenAI Codex CLI.

    Example::

        from a2a_adapter import CodexAdapter, serve_agent

        adapter = CodexAdapter(working_dir="/path/to/project")
        serve_agent(adapter, port=9010)
    """

    def __init__(
        self,
        working_dir: str,
        timeout: int = 600,
        codex_path: str = "codex",
        env_vars: dict[str, str] | None = None,
        session_store_path: str | None = None,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        super().__init__(
            working_dir=working_dir,
            timeout=timeout,
            env_vars=env_vars,
            session_store_path=session_store_path
            or os.path.join(
                working_dir, ".a2a-adapter", "codex", "sessions.json"
            ),
        )
        self.codex_path = codex_path
        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

    # ──── Public Interface ────

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs: Any
    ) -> str:
        context = kwargs.get("context")
        key = context_id or "_default"
        task_id = context.task_id if context else key
        lock = self._get_context_lock(key)
        async with lock:
            return await self._run_subprocess(
                key, task_id, user_input=user_input
            )

    def supports_streaming(self) -> bool:
        return False

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "CodexAdapter",
            description=self._description or "OpenAI Codex CLI agent",
            streaming=False,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Hooks ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        cmd = [self.codex_path, "exec"]
        thread_id = self._sessions.get(context_key)
        if thread_id:
            cmd.extend(["resume", thread_id])
        cmd.extend([
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ])
        cmd.append(message)
        return CommandResult(args=cmd, used_resume=bool(thread_id))

    def _parse_invoke_output(
        self, stdout_text: str, context_key: str
    ) -> ParseResult:
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
                error_msg = event.get("error", {}).get(
                    "message", "Unknown error"
                )
                raise RuntimeError(f"Codex turn failed: {error_msg}")

        if not text_parts:
            raise RuntimeError("Codex returned no visible output")

        response_text = "\n\n".join(text_parts)
        return ParseResult(text=response_text, session_id=thread_id)

    def _binary_not_found_message(self) -> str:
        return (
            f"Codex binary not found at '{self.codex_path}'. "
            "Ensure the OpenAI Codex CLI is installed and in PATH."
        )

    # ──── Private ────

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
