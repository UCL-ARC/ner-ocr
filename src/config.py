"""Configuration dataclasses and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class OCRConfig:
    """Configuration for OCR pipeline."""

    max_side_limit: int = 1500
    ocr_timeout: int = 400
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    return_word_box: bool = True
    device: str = "cpu"  # Options: 'cpu', 'gpu' (PaddleOCR uses 'gpu', not 'cuda')


@dataclass
class TransformerOCRConfig:
    """Configuration for Transformer OCR."""

    model: str = "LARGE_HANDWRITTEN"
    device: str = "cpu"
    use_fp16: bool = True  # Use half-precision for faster GPU loading/inference
    max_new_tokens: int = 128  # Maximum tokens to generate (increase for longer text)


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction pipeline."""

    model: str = "QWEN3_1_7B"
    device: str = "cpu"
    entities: list[str] = field(default_factory=lambda: ["AddressEntityList"])
    line_threshold: int = 10  # Y-distance threshold for grouping items on the same line
    gap_threshold: int = 40  # Y-distance threshold for inserting paragraph breaks


@dataclass
class AppConfig:
    """Main application configuration."""

    ocr: OCRConfig
    transformer_ocr: TransformerOCRConfig
    entity_extraction: EntityExtractionConfig
    queries: list[dict] = field(default_factory=list)


def load_config(path: Path) -> AppConfig:
    """Load YAML config file and return typed AppConfig."""
    with Path.open(path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        ocr=OCRConfig(**raw.get("ocr", {})),
        transformer_ocr=TransformerOCRConfig(**raw.get("transformer_ocr", {})),
        entity_extraction=EntityExtractionConfig(**raw.get("entity_extraction", {})),
        queries=raw.get("queries", []),
    )
