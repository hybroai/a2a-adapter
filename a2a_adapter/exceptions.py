"""
Adapter-level exceptions for the a2a-adapter bridge.

These exceptions carry semantic meaning at the bridge layer
(AdapterAgentExecutor), enabling adapters to signal specific
conditions that require non-default handling in the executor.
"""


class CancelledByAdapterError(Exception):
    """Raised by adapter when it detects a task was canceled.

    Bridge-level contract: AdapterAgentExecutor.execute() catches this
    and exits silently — cancel() has already emitted the terminal
    canceled state, so execute() must NOT emit failed/completed.

    Adapters raise this in two scenarios:
        1. Task was queued-canceled (_cancelled_tasks) — detected after
           acquiring the context lock, before spawning a subprocess.
        2. Task was killed (_killed_tasks) — detected after subprocess
           exits with a negative return code matching an adapter-issued kill.
    """

    pass


class UserFacingError(RuntimeError):
    """An error whose message is safe to show to the A2A client.

    AdapterAgentExecutor hides the text of every other exception from
    clients and sends a generic message that names the task. This message
    is sent as is, so it must hold nothing internal: no paths, URLs, tokens
    or tool output.
    """
