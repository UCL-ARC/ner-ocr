"""Data classes for structured OCR and entity extraction results."""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
    search_task: str | None = None


@dataclass
class PositionalQuery:
    """Structured positional query for searching text in OCR results using a rectangle."""

    x1: float  # Top-left X coordinate
    y1: float  # Top-left Y coordinate
    x2: float  # Bottom-right X coordinate
    y2: float  # Bottom-right Y coordinate


@dataclass
class SemanticQuery:
    """Structured semantic query for searching text in OCR results."""

    text: str
    threshold: float = 0.75  # Similarity threshold
    search_padding: float = 50.0  # Pixel radius
    search_type: str = "fuzzy"  # Type of semantic search
