"""Utilitiy (misc) functions."""

import signal
import types
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def timeout_context(duration: int) -> Generator[None, None, None]:
    """Context manager for adding timeout to operations."""

    def timeout_handler(_signum: int, _frame: types.FrameType | None) -> None:
        error_message = f"Operation timed out after {duration} seconds"
        raise TimeoutError(error_message)

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
