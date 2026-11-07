"""Thin command-line entry point for serving coding-agent adapters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from .integrations.claude_code import ClaudeCodeAdapter
from .integrations.codex import CodexAdapter
from .integrations.deepseek_harness import _INSTALL_HINT as _DSH_INSTALL_HINT
from .integrations.deepseek_harness import DeepSeekHarnessAdapter, resolve_dsh_bin
from .integrations.openclaw import OpenClawAdapter, VALID_THINKING_LEVELS
from .integrations.pi import PiAdapter
from .server import serve_agent
from .terminal import BackendInfo, TerminalDisplay, Verbosity


_PI_MANAGED_OPTIONS = {
    "--mode",
    "--print",
    "-p",
    "--continue",
    "-c",
    "--resume",
    "-r",
    "--session",
    "--session-id",
    "--session-dir",
    "--fork",
    "--no-session",
}
_CODEX_MANAGED_OPTIONS = {
    "--json",
    "--output-last-message",
    "-o",
    "--ephemeral",
    "--cd",
    "-C",
}
_CLAUDE_CODE_MANAGED_OPTIONS = {
    "--print",
    "-p",
    "--output-format",
    "--resume",
    "--continue",
    "-c",
    "--session-id",
    "--disallowedTools",
}
_OPENCLAW_MANAGED_OPTIONS = {
    "--local",
    "--message",
    "-m",
    "--message-file",
    "--json",
    "--session-id",
    "--thinking",
    "--agent",
}
_MANAGED_OPTIONS = {
    "pi": _PI_MANAGED_OPTIONS,
    "codex": _CODEX_MANAGED_OPTIONS,
    "claude": _CLAUDE_CODE_MANAGED_OPTIONS,
    "openclaw": _OPENCLAW_MANAGED_OPTIONS,
    # dsh runs through its SDK, which takes no native argv; see _validate_agent_args.
    "dsh": set(),
}
# Agents whose parsers (commander, clap) accept glued short options, so
# "-pfoo" or "-print" reaches them as "-p". Pi matches each flag exactly.
_GLUED_SHORT_OPTION_AGENTS = {"codex", "claude", "openclaw"}
_AGENT_DISPLAY_NAMES = {
    "pi": "Pi",
    "codex": "Codex",
    "claude": "Claude Code",
    "openclaw": "OpenClaw",
    "dsh": "DeepSeek Harness",
}


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


_AUTH_TOKEN_ENV = "A2A_AUTH_TOKEN"


def _token(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("token must not be empty")
    return value.strip()


def _auth_token(flag: str | None) -> str | None:
    """--auth-token, else $A2A_AUTH_TOKEN (stripped; empty means unset).

    The variable is removed from this process's environment either way, so
    it never reaches the agent or the tools it runs.
    """
    from_env = os.environ.pop(_AUTH_TOKEN_ENV, None)
    if flag is not None:
        return flag
    return from_env.strip() if from_env and from_env.strip() else None


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", "-help", action="help", help="show this help message")


def _add_server_options(
    parser: argparse.ArgumentParser,
) -> None:
    _add_help(parser)
    parser.add_argument(
        "--cwd", default=".", help="Agent working directory (default: current directory)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="A2A server bind host (use 0.0.0.0 with --auth-token for remote clients)",
    )
    parser.add_argument(
        "--auth-token",
        type=_token,
        metavar="TOKEN",
        help=(
            f"Require 'Authorization: Bearer TOKEN' on every request except the agent card "
            f"(default: ${_AUTH_TOKEN_ENV}; prefer the variable, argv is visible in ps)"
        ),
    )
    parser.add_argument("--port", type=_port, default=9000, help="A2A server port")
    parser.add_argument(
        "--timeout", type=_positive_float, default=600.0, help="Agent timeout in seconds"
    )
    parser.add_argument(
        "--name", default="", help='Agent Card display name (default: "<Agent> (<folder>)")'
    )
    parser.add_argument("--description", default="", help="Agent Card description")
    parser.add_argument(
        "--skill",
        action="append",
        type=_skill,
        metavar="JSON",
        help=(
            "Agent Card skill as a JSON object with id, name, description, tags (at least one) "
            "and optional examples; repeatable (default: one skill from the card name and description)"
        ),
    )
    parser.add_argument(
        "--log-level", default="warning", help="Uvicorn log level (default: warning)"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "-q",
        "--quiet",
        dest="verbosity",
        action="store_const",
        const=Verbosity.QUIET,
        help="Show only the startup card and errors",
    )
    output.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="store_const",
        const=Verbosity.VERBOSE,
        help="Show full prompt and reply text in the activity log",
    )
    output.add_argument(
        "--debug",
        dest="verbosity",
        action="store_const",
        const=Verbosity.DEBUG,
        help="Also show raw A2A request, response and stream payloads",
    )
    parser.set_defaults(verbosity=Verbosity.NORMAL)
    parser.add_argument("--executable", help="Path to the underlying agent executable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a2a-adapter",
        description="Expose a supported local agent as an A2A server.",
        epilog="For CLI-backed agents, pass native options after '--'.",
        add_help=False,
    )
    _add_help(parser)
    subparsers = parser.add_subparsers(dest="agent", required=True)

    pi_parser = subparsers.add_parser("pi", help="Serve Pi over A2A", add_help=False)
    _add_server_options(pi_parser)
    pi_parser.add_argument("--session-id", default="a2a-pi-agent")
    pi_parser.add_argument("--session-dir")

    codex_parser = subparsers.add_parser("codex", help="Serve Codex over A2A", add_help=False)
    _add_server_options(codex_parser)

    claude_parser = subparsers.add_parser("claude", help="Serve Claude over A2A", add_help=False)
    _add_server_options(claude_parser)

    openclaw_parser = subparsers.add_parser(
        "openclaw", help="Serve OpenClaw over A2A", add_help=False
    )
    _add_server_options(openclaw_parser)
    openclaw_parser.add_argument("--session-id")
    openclaw_parser.add_argument("--agent-id")
    openclaw_parser.add_argument("--thinking", choices=sorted(VALID_THINKING_LEVELS), default="low")

    dsh_parser = subparsers.add_parser(
        "dsh", help="Serve DeepSeek Harness over A2A", add_help=False
    )
    _add_server_options(dsh_parser)
    dsh_parser.add_argument("--model", help="Model id (default: the SDK's default)")
    dsh_parser.add_argument("--provider", help="Model provider (default: the SDK's default)")
    dsh_parser.add_argument("--profile", default="sdk", help="dsh profile (default: sdk)")
    dsh_parser.add_argument(
        "--dsh-home", help="Harness home (default: $DSH_HOME, else a per-user state directory)"
    )
    dsh_parser.add_argument("--reasoning-effort", help="Reasoning effort passed to the model")
    dsh_parser.add_argument("--max-tokens", type=_positive_int, help="Maximum output tokens")
    dsh_parser.add_argument(
        "--startup-timeout",
        type=_positive_float,
        default=30.0,
        help="Seconds to wait for the dsh runtime to start",
    )

    return parser


def _split_agent_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    try:
        separator = values.index("--")
    except ValueError:
        return values, []
    return values[:separator], values[separator + 1 :]


def _option_name(value: str) -> str:
    return value.split("=", 1)[0]


def _managed_option(value: str, managed: set[str], *, glued_short: bool) -> str | None:
    option_name = _option_name(value)
    if option_name in managed:
        return option_name
    if not glued_short:
        return None

    for option in sorted(managed):
        if len(option) == 2 and option.startswith("-") and value.startswith(option):
            return option
    return None


def _validate_agent_args(
    parser: argparse.ArgumentParser,
    agent: str,
    agent_args: Sequence[str],
) -> None:
    managed = _MANAGED_OPTIONS[agent]
    for value in agent_args:
        option = _managed_option(value, managed, glued_short=agent in _GLUED_SHORT_OPTION_AGENTS)
        if option is not None:
            parser.error(
                f"{agent} option {option!r} is managed by a2a-adapter and cannot be passed after '--'"
            )
    if agent == "codex" and any(value in {"resume", "review"} for value in agent_args):
        parser.error("Codex subcommands are managed by a2a-adapter")
    if agent == "dsh" and agent_args:
        parser.error("dsh does not accept native options after '--'; use the dsh options instead")


def _working_directory(parser: argparse.ArgumentParser, value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        parser.error(f"working directory does not exist or is not a directory: {path}")
    return str(path)


def _skill(value: str) -> dict:
    """Parse one ``--skill`` JSON object into the adapter's skill dict."""
    try:
        skill = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--skill must be a JSON object: {exc}") from None
    if not isinstance(skill, dict):
        raise argparse.ArgumentTypeError("--skill must be a JSON object")
    for key in ("id", "name", "description"):
        if not isinstance(skill.get(key), str) or not skill[key]:
            raise argparse.ArgumentTypeError(f"--skill needs a non-empty string {key!r}")
    # A2A requires at least one tag per skill; a2a-sdk warns on a missing or empty list.
    tags = skill.get("tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and t for t in tags):
        raise argparse.ArgumentTypeError(
            "--skill needs 'tags': a non-empty list of non-empty strings"
        )
    examples = skill.get("examples", [])
    if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
        raise argparse.ArgumentTypeError("--skill 'examples' must be a list of strings")
    unknown = set(skill) - {"id", "name", "description", "tags", "examples"}
    if unknown:
        raise argparse.ArgumentTypeError(f"--skill has unknown keys: {', '.join(sorted(unknown))}")
    return skill


def _card_skills(agent: str, name: str, description: str, skills: list[dict] | None) -> list[dict]:
    """A2A requires at least one skill; default to one describing the whole agent."""
    if skills:
        return skills
    return [{"id": agent, "name": name, "description": description, "tags": [agent]}]


def _card_identity(agent: str, working_dir: str, name: str, description: str) -> tuple[str, str]:
    """Default the Agent Card to "<Product> (<folder>)" so instances are distinguishable."""
    product = _AGENT_DISPLAY_NAMES[agent]
    folder = Path(working_dir).name
    if not name:
        name = f"{product} ({folder})" if folder else product
    if not description:
        description = f"{product} agent working in {folder}" if folder else f"{product} agent"
    return name, description


def _dsh_sdk_installed() -> bool:
    return importlib.util.find_spec("deepseek_harness") is not None


def _pi_command(parser: argparse.ArgumentParser, executable: str | None) -> list[str]:
    if executable:
        return [executable]
    command = shlex.split(os.getenv("A2A_PI_COMMAND", "pi"))
    if not command:
        parser.error("A2A_PI_COMMAND must not be empty")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, construct an existing adapter, and serve it."""
    parser = _build_parser()
    wrapper_args, agent_args = _split_agent_args(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(wrapper_args)
    auth_token = _auth_token(args.auth_token)
    if auth_token is not None and not all(" " <= char <= "~" for char in auth_token):
        parser.error(f"--auth-token / ${_AUTH_TOKEN_ENV} must be printable ASCII")
    _validate_agent_args(parser, args.agent, agent_args)
    working_dir = _working_directory(parser, args.cwd)
    name, description = _card_identity(args.agent, working_dir, args.name, args.description)
    skills = _card_skills(args.agent, name, description, args.skill)

    if args.agent == "pi":
        pi_command = _pi_command(parser, args.executable)
        command = shlex.join([*pi_command, *agent_args])
        adapter = PiAdapter(
            working_dir=working_dir,
            pi_command=pi_command,
            cli_args=agent_args,
            session_id=args.session_id,
            session_dir=args.session_dir,
            timeout=args.timeout,
            name=name,
            description=description,
            skills=skills,
        )
    elif args.agent == "codex":
        command = shlex.join([args.executable or "codex", *agent_args])
        adapter = CodexAdapter(
            working_dir=working_dir,
            codex_path=args.executable or "codex",
            cli_args=agent_args,
            timeout=args.timeout,
            name=name,
            description=description,
            skills=skills,
        )
    elif args.agent == "claude":
        command = shlex.join([args.executable or "claude", *agent_args])
        adapter = ClaudeCodeAdapter(
            working_dir=working_dir,
            claude_path=args.executable or "claude",
            cli_args=agent_args,
            timeout=args.timeout,
            name=name,
            description=description,
            skills=skills,
        )
    elif args.agent == "openclaw":
        command = shlex.join([args.executable or "openclaw", *agent_args])
        adapter = OpenClawAdapter(
            session_id=args.session_id,
            agent_id=args.agent_id,
            thinking=args.thinking,
            timeout=args.timeout,
            openclaw_path=args.executable or "openclaw",
            working_directory=working_dir,
            cli_args=agent_args,
            name=name,
            description=description,
            skills=skills,
        )
    elif args.agent == "dsh":
        # The SDK is imported on the first request; fail before serving instead.
        if not _dsh_sdk_installed():
            parser.error(_DSH_INSTALL_HINT)
        try:
            dsh_bin = resolve_dsh_bin(args.executable)
        except FileNotFoundError as error:
            parser.error(str(error))
        command = " · ".join(
            [dsh_bin or "dsh", f"profile {args.profile}"]
            + ([f"provider {args.provider}"] if args.provider else [])
            + ([f"model {args.model}"] if args.model else [])
        )
        adapter = DeepSeekHarnessAdapter(
            working_dir=working_dir,
            dsh_home=args.dsh_home,
            provider=args.provider,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            profile=args.profile,
            dsh_bin=args.executable,
            timeout=args.timeout,
            startup_timeout=args.startup_timeout,
            name=name,
            description=description,
            skills=skills,
        )
    else:  # pragma: no cover - argparse limits this to registered subcommands
        raise AssertionError(f"unsupported agent: {args.agent}")

    display = TerminalDisplay(
        BackendInfo(command=command, cwd=working_dir, timeout=args.timeout),
        verbosity=args.verbosity,
    )
    serve_agent(
        adapter,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        display=display,
        auth_token=auth_token,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
