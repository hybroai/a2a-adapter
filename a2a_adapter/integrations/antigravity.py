"""
Google Antigravity CLI (``agy``) adapter for A2A Protocol.

Wraps ``agy`` headless mode as one subprocess per message via
BaseA2AAdapter's built-in subprocess infrastructure. Each run prints
stream-json events; the conversation id they carry is stored per A2A
context and passed back with ``--conversation`` on the next message.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from ..base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    CommandResult,
    ParseResult,
    StreamEvent,
    default_state_dir,
)

logger = logging.getLogger(__name__)


# Linux limits one argv element to MAX_ARG_STRLEN (128 KiB, including the NUL).
# agy takes the headless prompt only as an argument, "--print=<prompt>".
_PRINT_FLAG = "--print="
_MAX_PROMPT_BYTES = 128 * 1024 - 1 - len(_PRINT_FLAG)

_ENV_SKIP_PERMISSIONS = "A2A_ANTIGRAVITY_SKIP_PERMISSIONS"


@dataclass
class _StreamRun:
    """What one streamed agy run has printed so far."""

    text_seen: bool = False
    result_seen: bool = False
    resumed_id: str | None = None
    any_event: bool = False


# The stream() run whose step is executing. _handle_stream_line() only sees
# one line, and the adapter is shared by concurrent contexts, so per-run state
# lives here rather than on the instance. stream() sets it around each step.
_stream_run: ContextVar[_StreamRun | None] = ContextVar("antigravity_stream_run", default=None)


class AntigravityAdapter(BaseA2AAdapter):
    """Adapter for the Google Antigravity CLI (``agy``).

    Each message runs ``agy --output-format stream-json --print=<message>``
    in ``working_dir``. Headless runs use the credentials cached by an
    interactive ``agy`` login (or the API-key / ADC setup described in the
    Antigravity docs).

    Headless runs never stop to ask. agy's default headless mode (``auto``)
    already runs shell commands and edits files in the workspace without
    asking; an action the permission policy does not allow is skipped and
    the run still succeeds, and the agent's questions to the user are
    skipped too. Restrict or allow tools in
    ``~/.gemini/antigravity-cli/settings.json``, or pin a mode with
    ``cli_args=["--mode", "plan"]``.

    An unknown or deleted conversation is not an error for agy: it starts a
    new one, which the adapter logs as a warning and stores in its place.

    Args:
        skip_permissions: Pass ``--dangerously-skip-permissions`` so every
            tool request is approved, including ones the policy would deny. When ``None`` (default), falls back to
            the ``A2A_ANTIGRAVITY_SKIP_PERMISSIONS`` env var (``1`` or
            ``true`` to enable). **Security risk**: only enable in
            sandboxed, trusted environments.

    Example::

        from a2a_adapter import AntigravityAdapter, serve_agent

        adapter = AntigravityAdapter(working_dir="/path/to/project")
        serve_agent(adapter, port=9014)
    """

    # agy starts each terminal-tool command in a session of its own: without
    # this, cancel, timeout and shutdown kill agy but leave those running.
    _kill_detached_descendants = True

    def __init__(
        self,
        working_dir: str,
        timeout: int = 600,
        agy_path: str = "agy",
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
            or os.path.join(default_state_dir("antigravity", working_dir), "sessions.json"),
        )
        self.agy_path = agy_path
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
            self.skip_permissions = os.getenv(_ENV_SKIP_PERMISSIONS, "").lower() in ("1", "true")
        if self.skip_permissions:
            logger.warning(
                "AntigravityAdapter: --dangerously-skip-permissions is ENABLED. "
                "Antigravity will approve every tool request. "
                "Only use in trusted, sandboxed environments."
            )

    # ──── Public Interface ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        context = kwargs.get("context")
        key = context_id or "_default"
        task_id = context.task_id if context else key
        async with self._get_context_lock(key):
            return await self._run_subprocess(key, task_id, user_input=user_input)

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        context = kwargs.get("context")
        key = context_id or "_default"
        task_id = context.task_id if context else key
        async with self._get_context_lock(key):
            run = _StreamRun()
            inner = cast(
                AsyncGenerator[str, None],
                self._run_subprocess_streaming(key, task_id, user_input=user_input),
            )
            try:
                while True:
                    # Set around each step only: a consumer may run every step
                    # in its own task (a copied context), and its own code
                    # between chunks must not see this run's state.
                    token = _stream_run.set(run)
                    try:
                        chunk = await inner.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _stream_run.reset(token)
                    yield chunk
            finally:
                await inner.aclose()
            # invoke() fails the same way: a run cut short before its envelope
            # must not pass partial text off as a finished reply.
            if not run.result_seen:
                raise RuntimeError("Antigravity run ended without a result event")

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "Antigravity",
            description=self._description or "Google Antigravity coding agent",
            streaming=True,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Hooks ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        if len(message.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError(
                f"Prompt too large for the Antigravity CLI: agy takes it as one command-line "
                f"argument, limited to {_MAX_PROMPT_BYTES} bytes"
            )
        cmd = [
            self.agy_path,
            *self.cli_args,
            "--output-format",
            "stream-json",
            # A client message starting with "/" is sent as typed, not run as
            # a slash command or skill.
            "--disable-slash-commands",
        ]
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        conversation_id = self._sessions.get(context_key)
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])
        # Attached to the flag, last, so nothing in cli_args can consume it.
        # agy rejects `-p --help` as an unknown flag but reads `--print=--help`
        # as the prompt (checked with agy 1.5.13).
        cmd.append(f"{_PRINT_FLAG}{message}")
        run = _stream_run.get()
        if run is not None:  # a new attempt (the base's stale retry builds a second one)
            run.text_seen = run.result_seen = run.any_event = False
            run.resumed_id = conversation_id
        return CommandResult(args=cmd, used_resume=bool(conversation_id))

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        deltas: list[str] = []
        conversation_id: str | None = None
        result: dict[str, Any] | None = None
        resumed_id = self._sessions.get(context_key)

        for event in _events(stdout_text):
            if event.get("event") == "init":
                _warn_if_not_resumed(resumed_id, _conversation_id(event))
            conversation_id = _conversation_id(event) or conversation_id
            text = _agent_delta(event)
            if text:
                deltas.append(text)
            if event.get("event") == "result" and isinstance(event.get("result"), dict):
                result = event["result"]

        if result is None:
            raise RuntimeError("Antigravity run ended without a result event")
        error = _result_error(result)
        if error:
            raise RuntimeError(error)

        # Deltas first, then the envelope's response: what stream() yields.
        text = "".join(deltas) or str(result.get("response") or "")
        if not text.strip():
            # The turn ran (and may have changed files), so the next message
            # resumes it; stream() stores it from the result line too.
            if conversation_id:
                self._remember_session(context_key, conversation_id)
            raise RuntimeError("Antigravity returned no output")
        return ParseResult(text=text, session_id=conversation_id)

    def _handle_stream_line(self, line: str) -> StreamEvent | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None

        run = _stream_run.get()
        if run is not None:
            run.any_event = True
        if event.get("event") == "result" and isinstance(event.get("result"), dict):
            result = event["result"]
            if run is not None:
                run.result_seen = True
            # Raised at once with agy's own message. (After the loop the base
            # would only have stderr, and would treat a resumed run that failed
            # with an empty stderr as a stale session and run it again.)
            error = _result_error(result)
            if error:
                return StreamEvent(error=error)
            # Only a successful turn stores the conversation, as in invoke().
            # (invoke() checks the exit code first, so a SUCCESS envelope from
            # a run that still exits non-zero is stored only when streaming.)
            response = str(result.get("response") or "")
            fallback = run is not None and not run.text_seen and bool(response.strip())
            return StreamEvent(
                text=response if fallback else None, session_id=_conversation_id(event)
            )

        if event.get("event") == "init" and run is not None:
            _warn_if_not_resumed(run.resumed_id, _conversation_id(event))
        text = _agent_delta(event)
        if text:
            if run is not None:
                run.text_seen = True
            return StreamEvent(text=text)
        return None

    def _forget_stale_session(self, context_key: str) -> None:
        # The base retries a resumed streamed run that failed without agent
        # text or stderr. If agy printed anything, the turn started and may
        # already have run tools (in auto mode, without asking): running it
        # again would repeat them, so fail instead and keep the conversation.
        # (invoke() only retries when stdout is empty.)
        run = _stream_run.get()
        if run is not None and run.any_event:
            raise RuntimeError(
                "Antigravity run exited with an error and no stderr after it had "
                "started; not retried, since it may already have run tools"
            )
        super()._forget_stale_session(context_key)

    def _extract_error(self, stdout_text: str) -> str | None:
        for message in _result_errors(stdout_text):
            return message
        return None

    def _binary_not_found_message(self) -> str:
        return (
            f"Antigravity CLI binary not found at '{self.agy_path}'. "
            "Ensure the Antigravity CLI (agy) is installed and in PATH."
        )


# ──── stream-json helpers ────


def _warn_if_not_resumed(resumed_id: str | None, conversation_id: str | None) -> None:
    """Log when agy started a new conversation instead of resuming ``resumed_id``.

    agy (checked with 1.5.13) does not fail on an unknown --conversation: it
    prints ``warning: conversation "<id>" not found`` on stderr, starts a new
    conversation and exits 0. The reply is then answered without the earlier
    history, and the new id replaces the old one once the turn succeeds.
    """
    if resumed_id and conversation_id and conversation_id != resumed_id:
        logger.warning(
            "AntigravityAdapter: conversation %s was not found; agy started a new "
            "conversation %s without the earlier history",
            resumed_id,
            conversation_id,
        )


def _events(stdout_text: str) -> Iterator[dict[str, Any]]:
    """The JSON object lines of stream-json output; other lines are skipped."""
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _conversation_id(event: dict[str, Any]) -> str | None:
    """The conversation id on an init or result event (empty on some failures)."""
    conversation_id = event.get("conversation_id")
    if not conversation_id and isinstance(event.get("result"), dict):
        conversation_id = event["result"].get("conversation_id")
    return str(conversation_id) if conversation_id else None


def _agent_delta(event: dict[str, Any]) -> str:
    """Agent reply text in a step_update; tool output and other steps give ""."""
    if event.get("event") != "step_update":
        return ""
    step = event.get("step_update")
    if not isinstance(step, dict) or step.get("step_type") != "agent_response":
        return ""
    text = step.get("text_delta")
    return text if isinstance(text, str) else ""


def _result_error(result: dict[str, Any]) -> str | None:
    """Why a result envelope is not a success, or None for SUCCESS."""
    status = str(result.get("status") or "")
    if status == "SUCCESS":
        return None
    detail = result.get("error")
    if status == "WAITING":
        # Undocumented trigger: likely the agent asking a question, possibly an
        # approval. Either way a headless run cannot answer it.
        return (
            "Antigravity run ended waiting on input (status WAITING), which a "
            "headless run cannot answer" + (f": {detail}" if detail else "")
        )
    message = f"Antigravity run failed (status {status or 'missing'})"
    return f"{message}: {detail}" if detail else message


def _result_errors(stdout_text: str) -> list[str]:
    """``error`` texts of non-success result envelopes, last first."""
    errors = []
    for event in _events(stdout_text):
        result = event.get("result")
        if event.get("event") == "result" and isinstance(result, dict):
            if result.get("status") != "SUCCESS" and result.get("error"):
                errors.append(str(result["error"]))
    return errors[::-1]
