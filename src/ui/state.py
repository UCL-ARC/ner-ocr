"""Session state management for the UI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AppConfig, load_config
from src.custom_types import PageResult, SearchResult


@dataclass
class PipelineState:
    """Holds the current state of the pipeline execution."""

    # Document
    document_path: Path | None = None
    document_name: str | None = None
    preview_images: list[np.ndarray] | None = None  # Pre-OCR page previews

    # Configuration (optional - not loaded by default in workbench mode)
    config: AppConfig | None = None
    config_path: Path = field(default_factory=lambda: Path("config.yaml"))

    # User-selected configuration from each step (for compile feature)
    ocr_config: dict | None = None
    search_config: dict | None = None
    enhancement_config: dict | None = None
    entity_config: dict | None = None

    # Pipeline stage results
    ocr_results: list[PageResult] | None = None
    search_results: list[SearchResult] | None = None
    enhanced_results: list[PageResult] | None = None
    entity_results: list[dict] | None = None

    # Current page being viewed
    current_page: int = 0

    # Processing status
    ocr_complete: bool = False
    search_complete: bool = False
    enhancement_complete: bool = False
    entity_complete: bool = False

    def reset(self) -> None:
        """Reset all pipeline results."""
        self.ocr_results = None
        self.search_results = None
        self.enhanced_results = None
        self.entity_results = None
        self.preview_images = None
        self.ocr_complete = False
        self.search_complete = False
        self.enhancement_complete = False
        self.entity_complete = False
        self.current_page = 0
        # Also reset the user config selections
        self.ocr_config = None
        self.search_config = None
        self.enhancement_config = None
        self.entity_config = None

    def load_config(self, config_path: Path | None = None) -> AppConfig:
        """Load or reload configuration."""
        if config_path:
            self.config_path = config_path
        self.config = load_config(self.config_path)
        return self.config

    def get_page_image(self, page_idx: int = 0) -> np.ndarray | None:
        """Get the original image for a specific page."""
        if self.ocr_results and page_idx < len(self.ocr_results):
            return self.ocr_results[page_idx].original_image
        return None

    def get_total_pages(self) -> int:
        """Get total number of pages in current document."""
        if self.ocr_results:
            return len(self.ocr_results)
        if self.preview_images:
            return len(self.preview_images)
        return 0

    def get_total_regions(self) -> int:
        """Get total number of text regions detected."""
        if self.ocr_results:
            return sum(len(p.data) for p in self.ocr_results)
        return 0

    def to_summary(self) -> dict[str, Any]:
        """Generate a summary of current state."""
        return {
            "document": self.document_name or "No document loaded",
            "ocr_status": "✅ Complete" if self.ocr_complete else "⏳ Pending",
            "search_status": "✅ Complete" if self.search_complete else "⏳ Pending",
            "enhancement_status": (
                "✅ Complete" if self.enhancement_complete else "⏳ Pending"
            ),
            "entity_status": "✅ Complete" if self.entity_complete else "⏳ Pending",
            "total_pages": self.get_total_pages(),
            "total_regions": self.get_total_regions(),
        }
