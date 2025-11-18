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
    transformer_text: str | None = None
    transformer_score: float | None = None


@dataclass
class PageResult:
    """Results for a single page."""

    page: int
    data: list[OCRResult]
    original_image: np.ndarray | None = None


@dataclass
class TransformerResult:
    """Result from Transformer-based OCR model."""

    transformer_text: str
    score: float | None = None


@dataclass
class SearchResult:
    """Structured result for a single search match."""

    page_result: PageResult
    search_type: str


@dataclass
class PositionalQuery:
    """Structured positional query for searching text in OCR results."""

    x: float
    y: float
    search_radius: float = 50.0  # Pixel radius


@dataclass
class SemanticQuery:
    """Structured semantic query for searching text in OCR results."""

    text: str
    threshold: float = 0.75  # Similarity threshold
    search_padding: float = 50.0  # Pixel radius
    search_type: str = "fuzzy"  # Type of semantic search


class BaseOCRProcessor(ABC):
    """Abstract base class for OCR processors."""

    @abstractmethod
    def extract(self, file_path: str | Path) -> list[PageResult]:
        """Extract OCR results from a file."""


class BaseTransformerOCR(ABC):
    """Abstract base class for Transformer-based OCR models."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> TransformerResult:
        """Perform OCR on the input image and return recognized text."""


class BaseRPAProcessor(ABC):
    """Abstract base class for RPA processors."""

    @abstractmethod
    def search(
        self,
        ocr_results: PageResult,
        query: PositionalQuery | SemanticQuery,
    ) -> SearchResult:
        """Search OCR results based on provided queries."""
