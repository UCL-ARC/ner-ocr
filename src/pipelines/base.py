"""Abstract base class for pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import AppConfig


class BasePipeline(ABC):
    """Abstract base class for pipelines."""

    def __init__(self, config: AppConfig) -> None:
        """Initialise pipeline with config."""
        self.config = config

    # TO DO: I think this process file method will not always be typed this way
    @abstractmethod
    def process_file(self, file_path: Path) -> list[Any]:
        """Process a single file and return results."""
        ...

    @abstractmethod
    def get_output_suffix(self) -> str:
        """Return the suffix for output files (e.g., '_ocr_results')."""
        ...

    # TO DO: this assumes output is always json
    def run(self, input_path: Path, output_dir: Path) -> None:
        """Run the pipeline on input path(s)."""
        from src.utils import (
            ensure_output_dir,
            get_input_files,
            save_json,
            to_serialisable,
        )

        ensure_output_dir(output_dir)
        files = get_input_files(input_path)

        for file_path in files:
            try:
                results = self.process_file(file_path)
                if results:
                    serialised = to_serialisable(results)
                    output_path = (
                        output_dir / f"{file_path.stem}{self.get_output_suffix()}.json"
                    )
                    save_json(serialised, output_path)
                    logger.success(f"Successfully processed {file_path}")
                else:
                    logger.warning(f"No results found for {file_path}")
            except Exception as e:  # noqa: BLE001
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
