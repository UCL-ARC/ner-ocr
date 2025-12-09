"""Abstract base classes for OCR processing, Transformer-based OCR models, RPA processing, and entity extraction."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .data import (
    PageResult,
    PositionalQuery,
    SearchResult,
    SemanticQuery,
    T,
    TransformerResult,
)


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


class EntityExtractor(ABC):
    """Abstract base class for LLM-based entity extraction."""

    @abstractmethod
    def extract_entities(
        self, text: str, entity_model: type[T], kwargs: dict
    ) -> dict[str, T]:
        """Extract entities from the provided text."""
