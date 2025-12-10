"""Utilitiy (misc) functions."""

import json
import signal
import types
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@contextmanager
def timeout_context(duration: int) -> Generator[None]:
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


def get_input_files(
    input_path: Path, extensions: tuple[str, ...] = (".json", ".pdf", ".png", ".jpg")
) -> list[Path]:
    """Get list of files to process from path."""
    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        logger.info(f"Processing {len(files)} files from directory: {input_path}")
        return files
    error_msg = f"Input path does not exist: {input_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def ensure_output_dir(output_path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    return output_path


def save_json(data: dict | list | str | float | bool | None, path: Path) -> None:
    """Save data to JSON file."""
    with Path.open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved results to {path}")


# TO DO: this is messy - refactor later
def to_serialisable(obj: Any) -> dict | list | str | float | bool | None:  # noqa: ANN401
    """Convert an object to a JSON-serialisable format."""
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {
            k: to_serialisable(v)
            for k, v in obj.items()
            if k not in {"original_image", "bbox_image"}
        }
    if isinstance(obj, list | tuple):
        return [to_serialisable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32 | np.float64 | np.int32 | np.int64):
        return obj.item()
    return obj
