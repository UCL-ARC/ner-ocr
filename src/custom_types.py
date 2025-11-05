"""Type definitions and data structures for OCR processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class SupportedExtensions(Enum):
    # TO DO: check what this does
    """Supported file extensions for OCR processing."""

    PDF = ".pdf"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"


@dataclass
class OCRResult:
    """Structured OCR result for a single text detection."""

    poly: list[list[float]]
    text: str
    score: float
    box: list[float]
    bbox_image: np.ndarray | None = None


@dataclass
class PageResult:
    """Results for a single page."""

    page: int
    data: list[OCRResult]
    original_image: np.ndarray | None = None


# TO do check how ABC works and what the point of this is
class BaseOCRProcessor(ABC):
    """Abstract base class for OCR processors."""

    @abstractmethod
    def extract(self, file_path: str | Path) -> list[PageResult]:
        """Extract OCR results from a file."""
